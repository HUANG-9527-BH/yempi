"""模块实现用于两个协程之间通信的内存流。

此模块提供了一种使用内存通道在两个协程之间通信的方式。写入者和读取者可以位于同一个事件循环中，也可以位于不同的事件循环中。当它们位于不同事件循环中时，它们也将位于不同线程中。

适用于代码中混合使用同步和异步的情况。
"""

import asyncio
import contextlib
import weakref
from asyncio import AbstractEventLoop, Queue
from collections.abc import AsyncIterator
from typing import Any, Generic, TypeVar

T = TypeVar("T")


def _close_loop_quietly(loop: AbstractEventLoop) -> None:
    """关闭内部创建的事件循环，作为最终化回调。

    Errors are suppressed rather than logged. `weakref.finalize` defaults to
    `atexit=True`, so this can run during interpreter shutdown, where `logging`
    may already be torn down. Python routes finalizer exceptions to
    `sys.unraisablehook` rather than propagating them, so suppressing here only
    avoids dumping a traceback to stderr at a nondeterministic moment, which is
    the same kind of noise this cleanup exists to remove. If `close()` does
    fail, the loop's own `__del__` still emits
    `ResourceWarning: unclosed event loop`, so the leak stays detectable.

    Args:
        loop: 要关闭的事件循环。正在运行的循环将保持不变。
    """
    if loop.is_running():
                # `close()` 会引发异常；保留 `ResourceWarning` 作为信号。
        return
    with contextlib.suppress(Exception):
        loop.close()


def _get_or_create_loop(owner: object) -> AbstractEventLoop:
    """获取当前事件循环，若此线程没有，则创建一个。

    Args:
        owner: 绑定在此处创建的循环生命周期的对象。

            由本函数创建的循环将在 `owner` 被垃圾回收时关闭。

    Returns:
        用于支持 `_MemoryStream` 的事件循环。
    """
    try:
        return asyncio.get_event_loop()
    except RuntimeError:
                # 此线程上未设置事件循环。在 Python 3.14+ 的同步上下文中始终如此，其中
        # `get_event_loop()` 不再隐式创建一个，且在任何版本的非主线程中也是如此。
                # 此事件循环由我们拥有：与 `get_event_loop()`
        # 返回的循环不同，没有调用者拥有它，因此一旦 `owner`
        # 死亡就应关闭它。否则，当垃圾回收在任意不确定的时间点运行时，循环的 `__del__`
        # 会发出 `ResourceWarning: unclosed event loop`。
                # 如果事件循环未关闭，则引发异常
        #
                # 如果事件循环未关闭，则引发异常
                # 调用方拥有它，因此在 `owner` 退出后关闭它。否则循环将
                # `__del__` 在任意位置都会发出 `ResourceWarning: 未关闭的事件循环`
                # 非确定性点垃圾回收恰好运行。
        loop = asyncio.new_event_loop()
        weakref.finalize(owner, _close_loop_quietly, loop)
        return loop


class _SendStream(Generic[T]):
    def __init__(
        self, reader_loop: AbstractEventLoop, queue: Queue[Any], done: object
    ) -> None:
        """为队列和 done 对象创建一个写入者。

        Args:
            reader_loop: 写入者使用的事件循环。

                此循环将用于调度对队列的写入。
            queue: 要写入的队列。

                这是一个 asyncio 队列。
            done: 特殊的哨兵对象，表示写入者已完成。
        """
        self._reader_loop = reader_loop
        self._queue = queue
        self._done = done

    async def send(self, item: T) -> None:
        """使用原始事件循环安排项目写入队列。

        这是一个可等待的协程。

        Args:
            item: 要写入队列的项目。
        """
        return self.send_nowait(item)

    def send_nowait(self, item: T) -> None:
        """使用原始事件循环安排项目写入队列。

        这是一个非阻塞调用。

        Args:
            item: 要写入队列的项目。

        Raises:
            RuntimeError: 如果尝试写入队列时事件循环已关闭。
        """
        try:
            self._reader_loop.call_soon_threadsafe(self._queue.put_nowait, item)
        except RuntimeError:
            if not self._reader_loop.is_closed():
                raise  # 如果循环未关闭，则抛出异常

    async def aclose(self) -> None:
        """异步调度 done 对象写入队列，使用原始事件循环。"""
        return self.close()

    def close(self) -> None:
        """使用原始事件循环安排 done 对象写入队列。

        这是一个非阻塞调用。

        Raises:
            RuntimeError: 当尝试向队列写入时，如果事件循环已关闭。
        """
        try:
            self._reader_loop.call_soon_threadsafe(self._queue.put_nowait, self._done)
        except RuntimeError:
            if not self._reader_loop.is_closed():
                raise  # 如果循环未关闭，则抛出异常


class _ReceiveStream(Generic[T]):
    def __init__(self, queue: Queue[Any], done: object) -> None:
        """创建队列和 done 对象的读取器。

        此读取器应在与传递给通道的循环相同的循环中使用。
        """
        self._queue = queue
        self._done = done
        self._is_closed = False

    async def __aiter__(self) -> AsyncIterator[T]:
        while True:
            item = await self._queue.get()
            if item is self._done:
                self._is_closed = True
                break
            yield item


class _MemoryStream(Generic[T]):
    """即使写入者和读取者位于不同线程，也能够从写入者向读取者流式传输数据。

    Uses asyncio queues to communicate between two co-routines. This implementation
    should work even if the writer and reader co-routines belong to two different event
    loops (e.g. one running from an event loop in the main thread and the other running
    in an event loop in a background thread).

    此实现旨在与单个写入者和单个读取者一起使用。

    这是 LangChain 的内部实现。请勿直接使用。
    """

    def __init__(self, loop: AbstractEventLoop) -> None:
        """为指定的循环创建通道。

        Args:
            loop: 用于通道的事件循环。

                该读取器假定在与传入此构造函数的循环相同的循环中运行。这不会在运行时进行验证。
        """
        self._loop = loop
        self._queue = asyncio.Queue[Any](maxsize=0)
        self._done = object()

    def get_send_stream(self) -> _SendStream[T]:
        """获取通道的写入器。

        Returns:
            通道的写入器。
        """
        return _SendStream[T](
            reader_loop=self._loop, queue=self._queue, done=self._done
        )

    def get_receive_stream(self) -> _ReceiveStream[T]:
        """获取通道的读取器。

        Returns:
            通道的读取器。
        """
        return _ReceiveStream[T](queue=self._queue, done=self._done)

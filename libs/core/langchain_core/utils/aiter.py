"""异步迭代器工具。

改编自
https://github.com/maxfischer2781/asyncstdlib/blob/master/asyncstdlib/itertools.py
MIT 许可证。
"""

from collections import deque
from collections.abc import (
    AsyncGenerator,
    AsyncIterable,
    AsyncIterator,
    Awaitable,
    Callable,
    Iterator,
)
from contextlib import AbstractAsyncContextManager
from types import TracebackType
from typing import (
    Any,
    Generic,
    TypeVar,
    cast,
    overload,
)

from typing_extensions import override

from langchain_core._api.deprecation import deprecated

T = TypeVar("T")

_no_default = object()


# https://github.com/python/cpython/blob/main/Lib/test/test_asyncgen.py#L54
@deprecated(since="1.1.2", removal="2.0.0")
def py_anext(
    iterator: AsyncIterator[T], default: T | Any = _no_default
) -> Awaitable[T | Any | None]:
    """用于测试目的的 `anext()` 的纯 Python 实现。

    与内置的 `anext()` C 实现非常接近。

    可用于比较内置实现的内部协程机制
    与 `__anext__()` 和 `send()` 或 `throw()` 在返回的
    生成器上的 C 实现。

    Args:
        iterator: 要推进的异步迭代器。
        default: 如果迭代器耗尽，返回的值。

            如果未提供，则会引发 `StopAsyncIteration` 异常。

    Returns:
        The next value from the iterator, or the default value if the iterator is
            exhausted.

    Raises:
        TypeError: 如果迭代器不是异步迭代器。
    """
    try:
        __anext__ = cast(
            "Callable[[AsyncIterator[T]], Awaitable[T]]", type(iterator).__anext__
        )
    except AttributeError as e:
        msg = f"{iterator!r} is not an async iterator"
        raise TypeError(msg) from e

    if default is _no_default:
        return __anext__(iterator)

    async def anext_impl() -> T | Any:
        try:
                        # C 代码比这低级得多，因为它实现了
                        # 迭代器协议的所有方法。在此实现中
                        # 我们依赖更高级别的协程概念，但
                        # 这正是我们想要的——纯 Python 高级别
                        # 实现与底层 C 语言 anext() 迭代器之间的交叉测试。
            return await __anext__(iterator)
        except StopAsyncIteration:
            return default

    return anext_impl()


class NoLock:
    """提供正确接口但无保护作用的虚拟锁。"""

    async def __aenter__(self) -> None:
        """不执行任何操作。"""

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> bool:
        """返回 False，异常未被抑制。"""
        return False


async def tee_peer(
    iterator: AsyncIterator[T],
        # 此对等体专用的缓冲区
    buffer: deque[T],
        # 所有对等体的缓冲区，包括我们自己的
    peers: list[deque[T]],
    lock: AbstractAsyncContextManager[Any],
) -> AsyncGenerator[T, None]:
    """一个 `tee` 的独立迭代器。

    This function is a generator that yields items from the shared iterator
    `iterator`. It buffers items until the least advanced iterator has yielded them as
    well.

    缓冲区与其他所有同级共享。

    Args:
        iterator: 共享的迭代器。
        buffer: 此同级的缓冲区。
        peers: 所有同级的缓冲区。
        lock: 用于同步访问共享缓冲区的锁。

    Yields:
        从共享迭代器中获取的下一个项。
    """
    try:
        while True:
            if not buffer:
                async with lock:
                                        # 在我们等待锁时，另一个对等体产生了项。
                                        # 继续下一次循环迭代以生成该项。
                    if buffer:
                        continue
                    try:
                        item = await anext(iterator)
                    except StopAsyncIteration:
                        break
                    else:
                                                # 向所有缓冲区追加，包括我们自己的。我们将从缓冲区再次获取我们的
                                                # 项，而不是直接生成它。
                                                # 这确保了如果我们的任意对等体
                                                # 并发获取项，则项的顺序正确。它们可能已经将项缓冲了。
                                                # 此对等体已完成——移除其缓冲区
                        for peer_buffer in peers:
                            peer_buffer.append(item)
            yield buffer.popleft()
    finally:
        async with lock:
                        # 如果这是我们最后一个对等体，尝试关闭迭代器
            for idx, peer_buffer in enumerate(peers):  # pragma: no branch
                if peer_buffer is buffer:
                    peers.pop(idx)
                    break
                        # 如果当前是最后一个对等节点，尝试关闭迭代器
            if not peers and hasattr(iterator, "aclose"):
                await iterator.aclose()


class Tee(Generic[T]):
    """创建 `n` 个在 `iterable` 上的独立异步迭代器。

    此操作将单个 `iterable` 拆分为多个迭代器，每个都按相同顺序提供相同的项。

    All child iterators may advance separately but share the same items from `iterable`
    -- when the most advanced iterator retrieves an item, it is buffered until the least
    advanced iterator has yielded it as well.

    `tee` 按需工作，可以处理无限的 `iterable`，前提是所有迭代器都能持续推进。

    ```python
    async def derivative(sensor_data):
        previous, current = a.tee(sensor_data, n=2)
        await a.anext(previous)  # advance one iterator
        return a.map(operator.sub, previous, current)
    ```

    Unlike `itertools.tee`, `.tee` returns a custom type instead of a `tuple`. Like a
    tuple, it can be indexed, iterated and unpacked to get the child iterators. In
    addition, its `.tee.aclose` method immediately closes all children, and it can be
    used in an `async with` context for the same effect.

    If `iterable` is an iterator and read elsewhere, `tee` will *not* provide these
    items. Also, `tee` must internally buffer each item until the last iterator has
    yielded it; if the most and least advanced iterator differ by most data, using a
    `list` is more efficient (but not lazy).

    If the underlying iterable is concurrency safe (`anext` may be awaited concurrently)
    the resulting iterators are concurrency safe as well. Otherwise, the iterators are
    safe if there is only ever one single "most advanced" iterator.

    为强制对 `anext` 的顺序使用，提供一个 `lock`

    - 例如，在 `asyncio` 应用中使用 `asyncio.Lock` 实例 - 并自动同步访问。

    """

    def __init__(
        self,
        iterable: AsyncIterator[T],
        n: int = 2,
        *,
        lock: AbstractAsyncContextManager[Any] | None = None,
    ):
        """创建一个 `tee`。

        Args:
            iterable: 要拆分的可迭代对象。
            n: 要创建的迭代器数量。
            lock: 用于同步访问共享缓冲区的锁。

        """
        self._iterator = aiter(iterable)
        self._buffers: list[deque[T]] = [deque() for _ in range(n)]
        self._children = tuple(
            tee_peer(
                iterator=self._iterator,
                buffer=buffer,
                peers=self._buffers,
                lock=lock if lock is not None else NoLock(),
            )
            for buffer in self._buffers
        )

    def __len__(self) -> int:
        """返回子迭代器的数量。"""
        return len(self._children)

    @overload
    def __getitem__(self, item: int) -> AsyncIterator[T]: ...

    @overload
    def __getitem__(self, item: slice) -> tuple[AsyncIterator[T], ...]: ...

    def __getitem__(
        self, item: int | slice
    ) -> AsyncIterator[T] | tuple[AsyncIterator[T], ...]:
        """返回指定索引或切片的子迭代器。"""
        return self._children[item]

    def __iter__(self) -> Iterator[AsyncIterator[T]]:
        """遍历所有子迭代器。

        Yields:
            子迭代器。
        """
        yield from self._children

    async def __aenter__(self) -> "Tee[T]":
        """返回 tee 实例。"""
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> bool:
        """关闭所有子迭代器。

        Returns:
            `False`，异常未被抑制。
        """
        await self.aclose()
        return False

    async def aclose(self) -> None:
        """异步关闭所有子迭代器。"""
        for child in self._children:
            await child.aclose()


atee = Tee


class aclosing(AbstractAsyncContextManager[Any]):  # noqa: N801
    """异步上下文管理器，用于包装具有 `aclose()` 方法的 `AsyncGenerator`。

    如下代码：

    ```python
    async with aclosing(<module>.fetch(<arguments>)) as agen:
        <block>
    ```

    ...is equivalent to this:

    ```python
    agen = <module>.fetch(<arguments>)
    try:
        <block>
    finally:
        await agen.aclose()

    ```
    """

    def __init__(self, thing: AsyncGenerator[Any, Any] | AsyncIterator[Any]) -> None:
        """创建上下文管理器。

        Args:
            thing: 要包装的资源。
        """
        self.thing = thing

    @override
    async def __aenter__(self) -> AsyncGenerator[Any, Any] | AsyncIterator[Any]:
        return self.thing

    @override
    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if hasattr(self.thing, "aclose"):
            await self.thing.aclose()


async def abatch_iterate(
    size: int | None, iterable: AsyncIterable[T]
) -> AsyncIterator[list[T]]:
    """用于异步可迭代对象的批量处理工具函数。

    Args:
        size: 批量大小。

            如果为 `None`，则返回单个批次。
        iterable: 要批量处理的异步可迭代对象。

    Yields:
        批次结果。

    Raises:
        ValueError: 如果 `size` 不为 `None` 且不是正整数。
    """
    if size is None:
        single_batch = [el async for el in iterable]
        if single_batch:
            yield single_batch
        return
    if size <= 0:
        msg = f"Batch size must be a positive integer, got {size}."
        raise ValueError(msg)
    batch: list[T] = []
    async for element in iterable:
        batch.append(element)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch

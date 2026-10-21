"""用于处理迭代器的工具。"""

from collections import deque
from collections.abc import Generator, Iterable, Iterator
from contextlib import AbstractContextManager
from itertools import islice
from types import TracebackType
from typing import (
    Any,
    Generic,
    Literal,
    TypeVar,
    overload,
)

T = TypeVar("T")


class NoLock:
    """空锁，提供正确的接口但不提供保护。"""

    def __enter__(self) -> None:
        """无操作。"""

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> Literal[False]:
        """返回 False（异常未被抑制）。"""
        return False


def tee_peer(
    iterator: Iterator[T],
        # 此对等方特有的缓冲区
    buffer: deque[T],
        # 所有对等方的缓冲区，包括我们自己的
    peers: list[deque[T]],
    lock: AbstractContextManager[Any],
) -> Generator[T, None, None]:
    """`.tee` 的单个迭代器。

    This function is a generator that yields items from the shared iterator `iterator`.
    It buffers items until the least advanced iterator has yielded them as well. The
    buffer is shared with all other peers.

    Args:
        iterator: 共享迭代器。
        buffer: 此对等方的缓冲区。
        peers: 所有对等方的缓冲区。
        lock: 用于同步访问共享缓冲区的锁。

    Yields:
        来自共享迭代器的下一个项目。
    """
    try:
        while True:
            if not buffer:
                with lock:
                                        # 另一个对等方在我们等待锁时生成了一个项。
                                        # 继续下一个循环迭代以传递该项。
                    if buffer:
                        continue
                    try:
                        item = next(iterator)
                    except StopIteration:
                        break
                    else:
                                                # 将项追加到所有缓冲区，包括我们自己的。我们将再次从缓冲区中获取该项，而不是直接传递它。
                                                # 这确保了如果我们的任何对等方并发获取项，则项的顺序正确。它们可能已经将项放入缓冲区。
                                                # 此对等方已完成——移除其缓冲区
                                                # 如果我們是最後一個對等方，嘗試關閉迭代器
                                                # 原因說明：https://stackoverflow.com/a/44638570
                        for peer_buffer in peers:
                            peer_buffer.append(item)
            yield buffer.popleft()
    finally:
        with lock:
                        # 此对等节点已完成 - 移除其缓冲区
            for idx, peer_buffer in enumerate(peers):  # pragma: no branch
                if peer_buffer is buffer:
                    peers.pop(idx)
                    break
                        # 如果当前是最后一个对等节点，尝试关闭迭代器
            if not peers and hasattr(iterator, "close"):
                iterator.close()


class Tee(Generic[T]):
    """在 `iterable` 上创建 `n` 个独立的异步迭代器。

    此操作将单个 `iterable` 拆分为多个迭代器，每个迭代器按相同顺序提供相同的项目。

    All child iterators may advance separately but share the same items from `iterable`
    -- when the most advanced iterator retrieves an item, it is buffered until the least
    advanced iterator has yielded it as well. A `tee` works lazily and can handle an
    infinite `iterable`, provided that all iterators advance.

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
    safe if there is only ever one single "most advanced" iterator. To enforce
    sequential use of `anext`, provide a `lock`

    - 例如，在 `asyncio` 应用中使用 `asyncio.Lock` 实例 - 并自动进行同步访问。

    """

    def __init__(
        self,
        iterable: Iterator[T],
        n: int = 2,
        *,
        lock: AbstractContextManager[Any] | None = None,
    ):
        """创建一个 `tee`。

        Args:
            iterable: 要拆分的可迭代对象。
            n: 要创建的迭代器数量。
            lock: 用于同步访问共享缓冲区的锁。

        """
        self._iterator = iter(iterable)
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
    def __getitem__(self, item: int) -> Iterator[T]: ...

    @overload
    def __getitem__(self, item: slice) -> tuple[Iterator[T], ...]: ...

    def __getitem__(self, item: int | slice) -> Iterator[T] | tuple[Iterator[T], ...]:
        """返回指定索引或切片处的子迭代器。"""
        return self._children[item]

    def __iter__(self) -> Iterator[Iterator[T]]:
        """返回子迭代器的迭代器。

        Yields:
            子迭代器。
        """
        yield from self._children

    def __enter__(self) -> "Tee[T]":
        """返回 `Tee` 实例。"""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> Literal[False]:
        """关闭所有子迭代器。

        Returns:
            `False`（异常未被抑制）。
        """
        self.close()
        return False

    def close(self) -> None:
        """关闭所有子迭代器。"""
        for child in self._children:
            child.close()


# 为何需要此操作 https://stackoverflow.com/a/44638570
safetee = Tee


def batch_iterate(size: int | None, iterable: Iterable[T]) -> Iterator[list[T]]:
    """实用的批处理函数。

    Args:
        size: 批次大小。

            如果为 `None`，返回单个批次。
        iterable: 要批处理的可迭代对象。

    Yields:
        可迭代对象的批次。

    Raises:
        ValueError: 如果 `size` 不为 `None` 且不是正整数。
    """
    if size is not None and size <= 0:
        msg = f"Batch size must be a positive integer, got {size}."
        raise ValueError(msg)
    it = iter(iterable)
    while True:
        chunk = list(islice(it, size))
        if not chunk:
            return
        yield chunk

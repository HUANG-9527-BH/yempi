"""Store 实现了键值存储和存储辅助功能。

模块提供了符合简单键值接口的各种键值存储的实现。

这些存储的主要目标是支持缓存的实现。
"""

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Iterator, Sequence
from typing import (
    Any,
    Generic,
    TypeVar,
)

from typing_extensions import override

from langchain_core.exceptions import LangChainException
from langchain_core.runnables import run_in_executor

K = TypeVar("K")
V = TypeVar("V")


class BaseStore(ABC, Generic[K, V]):
    """键值存储的抽象接口。

    This is an interface that's meant to abstract away the details of different
    key-value stores. It provides a simple interface for getting, setting, and deleting
    key-value pairs.

    The basic methods are `mget`, `mset`, and `mdelete` for getting, setting, and
    deleting multiple key-value pairs at once. The `yield_keys` method is used to
    iterate over keys that match a given prefix.

    The async versions of these methods are also provided, which are meant to be used in
    async contexts. The async methods are named with an `a` prefix, e.g., `amget`,
    `amset`, `amdelete`, and `ayield_keys`.

    默认情况下，`amget`、`amset`、`amdelete` 和 `ayield_keys`
    方法使用同步方法实现。如果存储原生支持异步操作，应重写这些方法。

    By design the methods only accept batches of keys and values, and not single keys or
    values. This is done to force user code to work with batches which will usually be
    more efficient by saving on round trips to the store.

    Examples:
        ```python
        from langchain.storage import BaseStore


        class MyInMemoryStore(BaseStore[str, int]):
            def __init__(self) -> None:
                self.store: dict[str, int] = {}

            def mget(self, keys: Sequence[str]) -> list[int | None]:
                return [self.store.get(key) for key in keys]

            def mset(self, key_value_pairs: Sequence[tuple[str, int]]) -> None:
                for key, value in key_value_pairs:
                    self.store[key] = value

            def mdelete(self, keys: Sequence[str]) -> None:
                for key in keys:
                    if key in self.store:
                        del self.store[key]

            def yield_keys(self, prefix: str | None = None) -> Iterator[str]:
                if prefix is None:
                    yield from self.store.keys()
                else:
                    for key in self.store.keys():
                        if key.startswith(prefix):
                            yield key
        ```
    """

    @abstractmethod
    def mget(self, keys: Sequence[K]) -> list[V | None]:
        """获取给定键关联的值。

        Args:
            keys: 键的序列。

        Returns:
            与键关联的可选值序列。
                如果找不到键，对应的值将为 `None`。
        """

    async def amget(self, keys: Sequence[K]) -> list[V | None]:
        """异步获取给定键关联的值。

        Args:
            keys: 键的序列。

        Returns:
            与键关联的可选值序列。
                如果找不到键，对应的值将为 `None`。
        """
        return await run_in_executor(None, self.mget, keys)

    @abstractmethod
    def mset(self, key_value_pairs: Sequence[tuple[K, V]]) -> None:
        """为给定键设置值。

        Args:
            key_value_pairs: 键值对的序列。
        """

    async def amset(self, key_value_pairs: Sequence[tuple[K, V]]) -> None:
        """异步为给定键设置值。

        Args:
            key_value_pairs: 键值对的序列。
        """
        return await run_in_executor(None, self.mset, key_value_pairs)

    @abstractmethod
    def mdelete(self, keys: Sequence[K]) -> None:
        """删除给定键及其关联的值。

        Args:
            keys: 要删除的键的序列。
        """

    async def amdelete(self, keys: Sequence[K]) -> None:
        """异步删除给定键及其关联的值。

        Args:
            keys: 要删除的键的序列。
        """
        return await run_in_executor(None, self.mdelete, keys)

    @abstractmethod
    def yield_keys(self, *, prefix: str | None = None) -> Iterator[K] | Iterator[str]:
        """获取匹配给定前缀的键的迭代器。

        Args:
            prefix: 要匹配的前缀。

        Yields:
            匹配给定前缀的键的迭代器。

                This method is allowed to return an iterator over either K or str
                depending on what makes more sense for the given store.
        """

    async def ayield_keys(
        self, *, prefix: str | None = None
    ) -> AsyncIterator[K] | AsyncIterator[str]:
        """异步获取匹配给定前缀的键的迭代器。

        Args:
            prefix: 要匹配的前缀。

        Yields:
            匹配给定前缀的键。

                This method is allowed to return an iterator over either K or str
                depending on what makes more sense for the given store.
        """
        iterator = await run_in_executor(None, self.yield_keys, prefix=prefix)
        done = object()
        while True:
            item = await run_in_executor(None, lambda it: next(it, done), iterator)
            if item is done:
                break
            yield item  # type: ignore[misc]


ByteStore = BaseStore[str, bytes]


class InMemoryBaseStore(BaseStore[str, V], Generic[V]):
    """使用字典实现的 `BaseStore` 的内存中版本。"""

    def __init__(self) -> None:
        """初始化一个空的存储。"""
        self.store: dict[str, V] = {}

    @override
    def mget(self, keys: Sequence[str]) -> list[V | None]:
        return [self.store.get(key) for key in keys]

    @override
    async def amget(self, keys: Sequence[str]) -> list[V | None]:
        return self.mget(keys)

    @override
    def mset(self, key_value_pairs: Sequence[tuple[str, V]]) -> None:
        for key, value in key_value_pairs:
            self.store[key] = value

    @override
    async def amset(self, key_value_pairs: Sequence[tuple[str, V]]) -> None:
        return self.mset(key_value_pairs)

    @override
    def mdelete(self, keys: Sequence[str]) -> None:
        for key in keys:
            if key in self.store:
                del self.store[key]

    @override
    async def amdelete(self, keys: Sequence[str]) -> None:
        self.mdelete(keys)

    def yield_keys(self, *, prefix: str | None = None) -> Iterator[str]:
        """获取与给定前缀匹配的键的迭代器。

        Args:
            prefix: 要匹配的前缀。

        Yields:
            与给定前缀匹配的键。
        """
        if prefix is None:
            yield from self.store.keys()
        else:
            for key in self.store:
                if key.startswith(prefix):
                    yield key

    async def ayield_keys(self, *, prefix: str | None = None) -> AsyncIterator[str]:
        """异步获取与给定前缀匹配的键的异步迭代器。

        Args:
            prefix: 要匹配的前缀。

        Yields:
            与给定前缀匹配的键。
        """
        if prefix is None:
            for key in self.store:
                yield key
        else:
            for key in self.store:
                if key.startswith(prefix):
                    yield key


class InMemoryStore(InMemoryBaseStore[Any]):
    """用于存储任意类型数据的内存中存储。

    Attributes:
        store: 存储键值对的底层字典。

    Examples:
        ```python
        from langchain.storage import InMemoryStore

        store = InMemoryStore()
        store.mset([("key1", "value1"), ("key2", "value2")])
        store.mget(["key1", "key2"])
        # ['value1', 'value2']
        store.mdelete(["key1"])
        list(store.yield_keys())
        # ['key2']
        list(store.yield_keys(prefix="k"))
        # ['key2']
        ```
    """


class InMemoryByteStore(InMemoryBaseStore[bytes]):
    """用于存储字节数据的内存中存储。

    Attributes:
        store: 存储键值对的底层字典。

    Examples:
        ```python
        from langchain.storage import InMemoryByteStore

        store = InMemoryByteStore()
        store.mset([("key1", b"value1"), ("key2", b"value2")])
        store.mget(["key1", "key2"])
        # [b'value1', b'value2']
        store.mdelete(["key1"])
        list(store.yield_keys())
        # ['key2']
        list(store.yield_keys(prefix="k"))
        # ['key2']
        ```
    """


class InvalidKeyException(LangChainException):
    """当键无效时抛出；例如，使用了不正确的字符。"""

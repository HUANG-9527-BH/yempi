from collections.abc import AsyncIterator, Callable, Iterator, Sequence
from typing import (
    Any,
    TypeVar,
)

from langchain_core.stores import BaseStore

K = TypeVar("K")
V = TypeVar("V")


class EncoderBackedStore(BaseStore[K, V]):
    """用键和值的编码器/解码器包装一个存储。

    使用 JSON 进行编码/解码的示例：

    ```python
    import json


    def key_encoder(key: int) -> str:
        return json.dumps(key)


    def value_serializer(value: float) -> str:
        return json.dumps(value)


    def value_deserializer(serialized_value: str) -> float:
        return json.loads(serialized_value)


    # Create an instance of the abstract store
    abstract_store = MyCustomStore()

    # Create an instance of the encoder-backed store
    store = EncoderBackedStore(
        store=abstract_store,
        key_encoder=key_encoder,
        value_serializer=value_serializer,
        value_deserializer=value_deserializer,
    )

    # Use the encoder-backed store methods
    store.mset([(1, 3.14), (2, 2.718)])
    values = store.mget([1, 2])  # Retrieves [3.14, 2.718]
    store.mdelete([1, 2])  # Deletes the keys 1 and 2
    ```
    """

    def __init__(
        self,
        store: BaseStore[str, Any],
        key_encoder: Callable[[K], str],
        value_serializer: Callable[[V], bytes],
        value_deserializer: Callable[[Any], V],
    ) -> None:
        """初始化一个 `EncodedStore`。

        Args:
            store: 要包装的底层字节存储。
            key_encoder: 将类型 `K` 的键编码为字符串的函数。
            value_serializer: 将类型 `V` 的值序列化为字节的函数。
            value_deserializer: 将字节反序列化回类型 V 的函数。
        """
        self.store = store
        self.key_encoder = key_encoder
        self.value_serializer = value_serializer
        self.value_deserializer = value_deserializer

    def mget(self, keys: Sequence[K]) -> list[V | None]:
        """获取与给定键关联的值。

        Args:
            keys: 一组键。

        Returns:
            与键关联的可选值序列。
            如果某个键未找到，对应值将为 `None`。
        """
        encoded_keys: list[str] = [self.key_encoder(key) for key in keys]
        values = self.store.mget(encoded_keys)
        return [
            self.value_deserializer(value) if value is not None else value
            for value in values
        ]

    async def amget(self, keys: Sequence[K]) -> list[V | None]:
        """异步获取与给定键关联的值。

        Args:
            keys: 一组键。

        Returns:
            与键关联的可选值序列。
            如果某个键未找到，对应值将为 `None`。
        """
        encoded_keys: list[str] = [self.key_encoder(key) for key in keys]
        values = await self.store.amget(encoded_keys)
        return [
            self.value_deserializer(value) if value is not None else value
            for value in values
        ]

    def mset(self, key_value_pairs: Sequence[tuple[K, V]]) -> None:
        """为给定键设置值。

        Args:
            key_value_pairs: 一组键值对。
        """
        encoded_pairs = [
            (self.key_encoder(key), self.value_serializer(value))
            for key, value in key_value_pairs
        ]
        self.store.mset(encoded_pairs)

    async def amset(self, key_value_pairs: Sequence[tuple[K, V]]) -> None:
        """异步为给定键设置值。

        Args:
            key_value_pairs: 一组键值对。
        """
        encoded_pairs = [
            (self.key_encoder(key), self.value_serializer(value))
            for key, value in key_value_pairs
        ]
        await self.store.amset(encoded_pairs)

    def mdelete(self, keys: Sequence[K]) -> None:
        """删除给定键及其关联的值。

        Args:
            keys: 要删除的键序列。
        """
        encoded_keys = [self.key_encoder(key) for key in keys]
        self.store.mdelete(encoded_keys)

    async def amdelete(self, keys: Sequence[K]) -> None:
        """异步删除给定键及其关联的值。

        Args:
            keys: 要删除的键序列。
        """
        encoded_keys = [self.key_encoder(key) for key in keys]
        await self.store.amdelete(encoded_keys)

    def yield_keys(
        self,
        *,
        prefix: str | None = None,
    ) -> Iterator[K] | Iterator[str]:
        """获取匹配给定前缀的键的迭代器。

        Args:
            prefix: 要匹配的前缀。

        Yields:
            匹配给定前缀的键。
        """
                # 对于当前版本，返回值不是 K，而是 str
                # 仅用于调试目的。应修复此问题。
        yield from self.store.yield_keys(prefix=prefix)

    async def ayield_keys(
        self,
        *,
        prefix: str | None = None,
    ) -> AsyncIterator[K] | AsyncIterator[str]:
        """异步获取匹配给定前缀的键的迭代器。

        Args:
            prefix: 要匹配的前缀。

        Yields:
            匹配给定前缀的键。
        """
                # 对于当前版本，返回值不是 K，而是 str
                # 仅用于调试目的。应修复此问题。
        async for key in self.store.ayield_keys(prefix=prefix):
            yield key

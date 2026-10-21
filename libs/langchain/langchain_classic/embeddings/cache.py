"""模块包含用于缓存嵌入的代码。

The cache backed embedder is a wrapper around an embedder that caches
embeddings in a key-value store. The cache is used to avoid recomputing
embeddings for the same text.

文本会被哈希，哈希值用作缓存中的键。
"""

from __future__ import annotations

import hashlib
import json
import uuid
import warnings
from collections.abc import Callable, Sequence
from typing import Literal, cast

from langchain_core.embeddings import Embeddings
from langchain_core.stores import BaseStore, ByteStore
from langchain_core.utils.iter import batch_iterate

from langchain_classic.storage.encoder_backed import EncoderBackedStore

NAMESPACE_UUID = uuid.UUID(int=1985)


def _sha1_hash_to_uuid(text: str) -> uuid.UUID:
    """返回基于 *text* 使用 SHA-1 生成的 UUID（确定性）。

    确定性且快速，**但不具备抗冲突性**。

    A malicious attacker could try to create two different texts that hash to the same
    UUID. This may not necessarily be an issue in the context of caching embeddings,
    but new applications should swap this out for a stronger hash function like
    xxHash, BLAKE2 or SHA-256, which are collision-resistant.
    """
    sha1_hex = hashlib.sha1(text.encode("utf-8"), usedforsecurity=False).hexdigest()
        # 将 `uuid5` 中的十六进制字符串嵌入以获得有效的 UUID。
    return uuid.uuid5(NAMESPACE_UUID, sha1_hex)


def _make_default_key_encoder(namespace: str, algorithm: str) -> Callable[[str], str]:
    """创建默认的键编码函数。

    Args:
        namespace: 用于隔离不同嵌入模型的键的前缀。
        algorithm:
           * `'sha1'` - 快速但不具备抗冲突性
           * `'blake2b'` - 密码学强度强，比 SHA-1 快
           * `'sha256'` - 密码学强度强，比 SHA-1 慢
           * `'sha512'` - 密码学强度强，比 SHA-1 慢

    Returns:
        使用指定算法编码键的函数。
    """
    if algorithm == "sha1":
        _warn_about_sha1_encoder()

    def _key_encoder(key: str) -> str:
        """使用指定算法编码键。"""
        if algorithm == "sha1":
            return f"{namespace}{_sha1_hash_to_uuid(key)}"
        if algorithm == "blake2b":
            return f"{namespace}{hashlib.blake2b(key.encode('utf-8')).hexdigest()}"
        if algorithm == "sha256":
            return f"{namespace}{hashlib.sha256(key.encode('utf-8')).hexdigest()}"
        if algorithm == "sha512":
            return f"{namespace}{hashlib.sha512(key.encode('utf-8')).hexdigest()}"
        msg = f"Unsupported algorithm: {algorithm}"
        raise ValueError(msg)

    return _key_encoder


def _value_serializer(value: Sequence[float]) -> bytes:
    """序列化一个值。"""
    return json.dumps(value).encode()


def _value_deserializer(serialized_value: bytes) -> list[float]:
    """反序列化一个值。"""
    return cast("list[float]", json.loads(serialized_value.decode()))


# 该警告是全局的；需跟踪其发出情况，确保仅显示一次。
_warned_about_sha1: bool = False


def _warn_about_sha1_encoder() -> None:
    """发出关于 SHA-1 冲突弱点的一次性警告。"""
    global _warned_about_sha1  # noqa: PLW0603
    if not _warned_about_sha1:
        warnings.warn(
            "Using default key encoder: SHA-1 is *not* collision-resistant. "
            "While acceptable for most cache scenarios, a motivated attacker "
            "can craft two different payloads that map to the same cache key. "
            "If that risk matters in your environment, supply a stronger "
            "encoder (e.g. SHA-256 or BLAKE2) via the `key_encoder` argument. "
            "If you change the key encoder, consider also creating a new cache, "
            "to avoid (the potential for) collisions with existing keys.",
            category=UserWarning,
            stacklevel=2,
        )
        _warned_about_sha1 = True


class CacheBackedEmbeddings(Embeddings):
    """用于缓存嵌入模型结果的接口。

    The interface allows works with any store that implements
    the abstract store interface accepting keys of type str and values of list of
    floats.

    如有需要，该接口可扩展以接受其他值序列化器和反序列化器的实现，
    以及键编码器。

    注意，默认情况下仅缓存文档嵌入。若要缓存查询嵌入，请在构造函数中传入
    query_embedding_store。

    Examples:
        ```python
        from langchain_classic.embeddings import CacheBackedEmbeddings
        from langchain_classic.storage import LocalFileStore
        from langchain_openai import OpenAIEmbeddings

        store = LocalFileStore("./my_cache")

        underlying_embedder = OpenAIEmbeddings()
        embedder = CacheBackedEmbeddings.from_bytes_store(
            underlying_embedder, store, namespace=underlying_embedder.model
        )

        # Embedding is computed and cached
        embeddings = embedder.embed_documents(["hello", "goodbye"])

        # Embeddings are retrieved from the cache, no computation is done
        embeddings = embedder.embed_documents(["hello", "goodbye"])
        ```
    """

    def __init__(
        self,
        underlying_embeddings: Embeddings,
        document_embedding_store: BaseStore[str, list[float]],
        *,
        batch_size: int | None = None,
        query_embedding_store: BaseStore[str, list[float]] | None = None,
    ) -> None:
        """初始化嵌入智能体。

        Args:
            underlying_embeddings: 用于计算嵌入的嵌入智能体。
            document_embedding_store: 用于缓存文档嵌入的存储。
            batch_size: 在存储更新之间嵌入的文档数量。
            query_embedding_store: 用于缓存查询嵌入的存储。
                如果为 `None`，则不缓存查询嵌入。
        """
        super().__init__()
        self.document_embedding_store = document_embedding_store
        self.query_embedding_store = query_embedding_store
        self.underlying_embeddings = underlying_embeddings
        self.batch_size = batch_size

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """嵌入文本列表。

        The method first checks the cache for the embeddings.
        If the embeddings are not found, the method uses the underlying embedder
        to embed the documents and stores the results in the cache.

        Args:
            texts: 要嵌入的文本列表。

        Returns:
            给定文本对应的嵌入列表。
        """
        vectors: list[list[float] | None] = self.document_embedding_store.mget(
            texts,
        )
        all_missing_indices: list[int] = [
            i for i, vector in enumerate(vectors) if vector is None
        ]

        for missing_indices in batch_iterate(self.batch_size, all_missing_indices):
            missing_texts = [texts[i] for i in missing_indices]
            missing_vectors = self.underlying_embeddings.embed_documents(missing_texts)
            self.document_embedding_store.mset(
                list(zip(missing_texts, missing_vectors, strict=False)),
            )
            for index, updated_vector in zip(
                missing_indices, missing_vectors, strict=False
            ):
                vectors[index] = updated_vector

        return cast(
            "list[list[float]]",
            vectors,
        )  # None 值此时应已解决。

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        """嵌入文本列表。

        The method first checks the cache for the embeddings.
        If the embeddings are not found, the method uses the underlying embedder
        to embed the documents and stores the results in the cache.

        Args:
            texts: 要嵌入的文本列表。

        Returns:
            给定文本对应的嵌入列表。
        """
        vectors: list[list[float] | None] = await self.document_embedding_store.amget(
            texts
        )
        all_missing_indices: list[int] = [
            i for i, vector in enumerate(vectors) if vector is None
        ]

                # batch_iterate 支持 None 的 batch_size，此时一次性返回所有元素
                # 作为单个批次。
        for missing_indices in batch_iterate(self.batch_size, all_missing_indices):
            missing_texts = [texts[i] for i in missing_indices]
            missing_vectors = await self.underlying_embeddings.aembed_documents(
                missing_texts,
            )
            await self.document_embedding_store.amset(
                list(zip(missing_texts, missing_vectors, strict=False)),
            )
            for index, updated_vector in zip(
                missing_indices, missing_vectors, strict=False
            ):
                vectors[index] = updated_vector

        return cast(
            "list[list[float]]",
            vectors,
        )  # None 值此时应已解决。

    def embed_query(self, text: str) -> list[float]:
        """嵌入查询文本。

        默认情况下，此方法不会缓存查询。要启用缓存，请在初始化嵌入器时将 `cache_query`
        参数设为 `True`。

        Args:
            text: 要嵌入的文本。

        Returns:
            给定文本的嵌入向量。
        """
        if not self.query_embedding_store:
            return self.underlying_embeddings.embed_query(text)

        (cached,) = self.query_embedding_store.mget([text])
        if cached is not None:
            return cached

        vector = self.underlying_embeddings.embed_query(text)
        self.query_embedding_store.mset([(text, vector)])
        return vector

    async def aembed_query(self, text: str) -> list[float]:
        """嵌入查询文本。

        默认情况下，此方法不会缓存查询。要启用缓存，请在初始化嵌入器时将 `cache_query`
        参数设为 `True`。

        Args:
            text: 要嵌入的文本。

        Returns:
            给定文本的嵌入向量。
        """
        if not self.query_embedding_store:
            return await self.underlying_embeddings.aembed_query(text)

        (cached,) = await self.query_embedding_store.amget([text])
        if cached is not None:
            return cached

        vector = await self.underlying_embeddings.aembed_query(text)
        await self.query_embedding_store.amset([(text, vector)])
        return vector

    @classmethod
    def from_bytes_store(
        cls,
        underlying_embeddings: Embeddings,
        document_embedding_cache: ByteStore,
        *,
        namespace: str = "",
        batch_size: int | None = None,
        query_embedding_cache: bool | ByteStore = False,
        key_encoder: Callable[[str], str]
        | Literal["sha1", "blake2b", "sha256", "sha512"] = "sha1",
    ) -> CacheBackedEmbeddings:
        """用于向存储中添加必要的序列化和编码的入口点。

        Args:
            underlying_embeddings: The embedder to use for embedding.
            document_embedding_cache: The cache to use for storing document embeddings.
            *,
            namespace: The namespace to use for document cache.
                This namespace is used to avoid collisions with other caches.
                For example, set it to the name of the embedding model used.
            batch_size: The number of documents to embed between store updates.
            query_embedding_cache: The cache to use for storing query embeddings.
                True to use the same cache as document embeddings.
                False to not cache query embeddings.
            key_encoder: Optional callable to encode keys. If not provided,
                a default encoder using SHA-1 will be used. SHA-1 is not
                collision-resistant, and a motivated attacker could craft two
                different texts that hash to the same cache key.

                新应用应使用替代编码器之一，或提供自定义且强健的键编码函数以规避此风险。

                If you change a key encoder in an existing cache, consider
                just creating a new cache, to avoid (the potential for)
                collisions with existing keys or having duplicate keys
                for the same text in the cache.

        Returns:
            使用所提供缓存的 `CacheBackedEmbeddings` 实例。
        """
        if isinstance(key_encoder, str):
            key_encoder = _make_default_key_encoder(namespace, key_encoder)
        elif callable(key_encoder):
                        # 如果提供了自定义键编码器，则不应与命名空间一起使用。
                        # 用户可在自定义键编码器中自行处理命名空间。
                        # 用户可以直接处理自定义键编码器中的命名空间。
            if namespace:
                msg = (
                    "Do not supply `namespace` when using a custom key_encoder; "
                    "add any prefixing inside the encoder itself."
                )
                raise ValueError(msg)
        else:
            msg = (  # type: ignore[unreachable]
                "key_encoder must be either 'blake2b', 'sha1', 'sha256', 'sha512' "
                "or a callable that encodes keys."
            )
            raise ValueError(msg)  # noqa: TRY004

        document_embedding_store = EncoderBackedStore[str, list[float]](
            document_embedding_cache,
            key_encoder,
            _value_serializer,
            _value_deserializer,
        )
        if query_embedding_cache is True:
            query_embedding_store = document_embedding_store
        elif query_embedding_cache is False:
            query_embedding_store = None
        else:
            query_embedding_store = EncoderBackedStore[str, list[float]](
                query_embedding_cache,
                key_encoder,
                _value_serializer,
                _value_deserializer,
            )

        return cls(
            underlying_embeddings,
            document_embedding_store,
            batch_size=batch_size,
            query_embedding_store=query_embedding_store,
        )

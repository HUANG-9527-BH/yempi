from __future__ import annotations

import functools
import os
import uuid
import warnings
from collections.abc import Callable
from itertools import islice
from operator import itemgetter
from typing import TYPE_CHECKING, Any

import numpy as np
from langchain_core._api.deprecation import deprecated
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.runnables.config import run_in_executor
from langchain_core.vectorstores import VectorStore
from qdrant_client import AsyncQdrantClient, QdrantClient
from qdrant_client.http import models
from qdrant_client.local.async_qdrant_local import AsyncQdrantLocal

from langchain_qdrant._utils import maximal_marginal_relevance

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Generator, Iterable, Sequence

    DictFilter = dict[str, str | int | bool | dict | list]
    MetadataFilter = DictFilter | models.Filter


class QdrantException(Exception):  # noqa: N818
    """与 `Qdrant` 相关的异常。"""


def sync_call_fallback(method: Callable) -> Callable:
    """如果异步方法未实现，请调用同步方法。

    This decorator should only be used for methods that are defined as async in the
    class.

    """

    @functools.wraps(method)
    async def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        try:
            return await method(self, *args, **kwargs)
        except NotImplementedError:
                        # 如果未实现异步方法，则通过移除方法名首字母来调用同步方法。例如，
                        # 如果异步方法名为 `aadd_texts`，则同步方法将被调用为 `aad_texts`。
                        # deprecated
                        # LangChain 不期望返回向量
            return await run_in_executor(
                None, getattr(self, method.__name__[1:]), *args, **kwargs
            )

    return wrapper


@deprecated(since="0.1.2", alternative="QdrantVectorStore", removal="0.5.0")
class Qdrant(VectorStore):
    """`Qdrant` 向量存储。

    ```python
    from qdrant_client import QdrantClient
    from langchain_qdrant import Qdrant

    client = QdrantClient()
    collection_name = "MyCollection"
    qdrant = Qdrant(client, collection_name, embedding_function)
    ```
    """

    CONTENT_KEY: str = "page_content"
    METADATA_KEY: str = "metadata"
    VECTOR_NAME: str | None = None

    def __init__(
        self,
        client: Any,
        collection_name: str,
        embeddings: Embeddings | None = None,
        content_payload_key: str = CONTENT_KEY,
        metadata_payload_key: str = METADATA_KEY,
        distance_strategy: str = "COSINE",
        vector_name: str | None = VECTOR_NAME,
        async_client: Any | None = None,
        embedding_function: Callable | None = None,  # LangChain 不期望返回向量
    ) -> None:
        """使用必要的组件进行初始化。"""
        if not isinstance(client, QdrantClient):
            msg = (
                f"client should be an instance of qdrant_client.QdrantClient, "
                f"got {type(client)}"
            )
            raise TypeError(msg)

        if async_client is not None and not isinstance(async_client, AsyncQdrantClient):
            msg = (
                f"async_client should be an instance of qdrant_client.AsyncQdrantClient"
                f"got {type(async_client)}"
            )
            raise ValueError(msg)

        if embeddings is None and embedding_function is None:
            msg = "`embeddings` value can't be None. Pass `embeddings` instance."
            raise ValueError(msg)

        if embeddings is not None and embedding_function is not None:
            msg = (
                "Both `embeddings` and `embedding_function` are passed. "
                "Use `embeddings` only."
            )
            raise ValueError(msg)

        self._embeddings = embeddings
        self._embeddings_function = embedding_function
        self.client: QdrantClient = client
        self.async_client: AsyncQdrantClient | None = async_client
        self.collection_name = collection_name
        self.content_payload_key = content_payload_key or self.CONTENT_KEY
        self.metadata_payload_key = metadata_payload_key or self.METADATA_KEY
        self.vector_name = vector_name or self.VECTOR_NAME

        if embedding_function is not None:
            warnings.warn(
                "Using `embedding_function` is deprecated. "
                "Pass `Embeddings` instance to `embeddings` instead.",
                stacklevel=2,
            )

        if not isinstance(embeddings, Embeddings):
            warnings.warn(
                "`embeddings` should be an instance of `Embeddings`."
                "Using `embeddings` as `embedding_function` which is deprecated",
                stacklevel=2,
            )
            self._embeddings_function = embeddings
            self._embeddings = None

        self.distance_strategy = distance_strategy.upper()

    @property
    def embeddings(self) -> Embeddings | None:
        return self._embeddings

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: list[dict] | None = None,
        ids: Sequence[str] | None = None,
        batch_size: int = 64,
        **kwargs: Any,
    ) -> list[str]:
        """通过嵌入运行更多文本并添加到 `VectorStore` 中。

        Args:
            texts: Iterable of strings to add to the `VectorStore`.
            metadatas: Optional list of metadatas associated with the texts.
            ids:
                Optional list of ids to associate with the texts. Ids have to be
                uuid-like strings.
            batch_size:
                How many vectors upload per-request.
                Default: `64`
            **kwargs: Additional keyword arguments.

        Returns:
            添加文本到 `VectorStore` 后返回的 ID 列表。

        """
        added_ids = []
        for batch_ids, points in self._generate_rest_batches(
            texts, metadatas, ids, batch_size
        ):
            self.client.upsert(
                collection_name=self.collection_name, points=points, **kwargs
            )
            added_ids.extend(batch_ids)

        return added_ids

    @sync_call_fallback
    async def aadd_texts(
        self,
        texts: Iterable[str],
        metadatas: list[dict] | None = None,
        ids: Sequence[str] | None = None,
        batch_size: int = 64,
        **kwargs: Any,
    ) -> list[str]:
        """通过嵌入运行更多文本并添加到 `VectorStore` 中。

        Args:
            texts: Iterable of strings to add to the `VectorStore`.
            metadatas: Optional list of metadatas associated with the texts.
            ids:
                Optional list of ids to associate with the texts. Ids have to be
                uuid-like strings.
            batch_size:
                How many vectors upload per-request.
                Default: `64`
            **kwargs: Additional keyword arguments.

        Returns:
            添加文本到 `VectorStore` 后返回的 ID 列表。

        """
        if self.async_client is None or isinstance(
            self.async_client._client, AsyncQdrantLocal
        ):
            msg = "QdrantLocal cannot interoperate with sync and async clients"
            raise NotImplementedError(msg)

        added_ids = []
        async for batch_ids, points in self._agenerate_rest_batches(
            texts, metadatas, ids, batch_size
        ):
            await self.async_client.upsert(
                collection_name=self.collection_name, points=points, **kwargs
            )
            added_ids.extend(batch_ids)

        return added_ids

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        filter: MetadataFilter | None = None,  # noqa: A002
        search_params: models.SearchParams | None = None,
        offset: int = 0,
        score_threshold: float | None = None,
        consistency: models.ReadConsistency | None = None,
        **kwargs: Any,
    ) -> list[Document]:
        """返回与查询最相似的文档。

        Args:
            query: 用于查找相似文档的文本。
            k: 返回的文档数量。
            filter: 按元数据过滤。
            search_params: 额外的搜索参数
            offset:
                返回结果的起始偏移量。
                可用于分页结果。
                Note: large offset values may cause performance issues.
            score_threshold:
                Define a minimal score threshold for the result.
                If defined, less similar results will not be returned.
                Score of the returned result might be higher or smaller than the
                threshold depending on the Distance function used.
                E.g. for cosine similarity only higher scores will be returned.
            consistency:
                Read consistency of the search. Defines how many replicas should be
                queried before returning the result.
                Values:
                - int - number of replicas to query, values should present in all
                        queried replicas
                - 'majority' - query all replicas, but return values present in the
                               majority of replicas
                - 'quorum' - query the majority of replicas, return values present in
                             all of them
                - 'all' - query all replicas, and return values present in all replicas
            **kwargs:
                Any other named arguments to pass through to QdrantClient.search()

        Returns:
            与查询最相似的 `Document` 对象列表。

        """
        results = self.similarity_search_with_score(
            query,
            k,
            filter=filter,
            search_params=search_params,
            offset=offset,
            score_threshold=score_threshold,
            consistency=consistency,
            **kwargs,
        )
        return list(map(itemgetter(0), results))

    @sync_call_fallback
    async def asimilarity_search(
        self,
        query: str,
        k: int = 4,
        filter: MetadataFilter | None = None,  # noqa: A002
        **kwargs: Any,
    ) -> list[Document]:
        """返回与查询最相似的文档。

        Args:
            query: 用于查找与之相似文档的文本。
            k: 返回的文档数量。
            filter: 按元数据过滤。
            **kwargs: 其他关键字参数。

        Returns:
            与查询最相似的 `Document` 对象列表。

        """
        results = await self.asimilarity_search_with_score(query, k, filter, **kwargs)
        return list(map(itemgetter(0), results))

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        filter: MetadataFilter | None = None,  # noqa: A002
        search_params: models.SearchParams | None = None,
        offset: int = 0,
        score_threshold: float | None = None,
        consistency: models.ReadConsistency | None = None,
        **kwargs: Any,
    ) -> list[tuple[Document, float]]:
        """返回与查询最相似的文档。

        Args:
            query: 用于查找与之相似文档的文本。
            k: 返回的文档数量。
            filter: 按元数据过滤。
            search_params: 额外的搜索参数
            offset:
                要返回的第一个结果的偏移量。
                可用于分页结果。
                Note: large offset values may cause performance issues.
            score_threshold:
                Define a minimal score threshold for the result.
                If defined, less similar results will not be returned.
                Score of the returned result might be higher or smaller than the
                threshold depending on the Distance function used.
                E.g. for cosine similarity only higher scores will be returned.
            consistency:
                Read consistency of the search. Defines how many replicas should be
                queried before returning the result.
                Values:
                - int - number of replicas to query, values should present in all
                        queried replicas
                - 'majority' - query all replicas, but return values present in the
                               majority of replicas
                - 'quorum' - query the majority of replicas, return values present in
                             all of them
                - 'all' - query all replicas, and return values present in all replicas
            **kwargs:
                Any other named arguments to pass through to QdrantClient.search()

        Returns:
            与查询文本最相似的文档列表及每个文档的距离。

        """
        return self.similarity_search_with_score_by_vector(
            self._embed_query(query),
            k,
            filter=filter,
            search_params=search_params,
            offset=offset,
            score_threshold=score_threshold,
            consistency=consistency,
            **kwargs,
        )

    @sync_call_fallback
    async def asimilarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        filter: MetadataFilter | None = None,  # noqa: A002
        search_params: models.SearchParams | None = None,
        offset: int = 0,
        score_threshold: float | None = None,
        consistency: models.ReadConsistency | None = None,
        **kwargs: Any,
    ) -> list[tuple[Document, float]]:
        """返回与查询最相似的文档。

        Args:
            query: 用于查找与之相似文档的文本。
            k: 返回的文档数量。
            filter: 按元数据过滤。
            search_params: 额外的搜索参数
            offset:
                要返回的第一个结果的偏移量。
                可用于分页结果。
                Note: large offset values may cause performance issues.
            score_threshold:
                Define a minimal score threshold for the result.
                If defined, less similar results will not be returned.
                Score of the returned result might be higher or smaller than the
                threshold depending on the Distance function used.
                E.g. for cosine similarity only higher scores will be returned.
            consistency:
                Read consistency of the search. Defines how many replicas should be
                queried before returning the result.
                Values:
                - int - number of replicas to query, values should present in all
                        queried replicas
                - 'majority' - query all replicas, but return values present in the
                    majority of replicas
                - 'quorum' - query the majority of replicas, return values present in
                    all of them
                - 'all' - query all replicas, and return values present in all replicas
            **kwargs:
                Any other named arguments to pass through to
                AsyncQdrantClient.Search().

        Returns:
            与查询文本最相似的文档列表及其每个结果的距离。

        """
        query_embedding = await self._aembed_query(query)
        return await self.asimilarity_search_with_score_by_vector(
            query_embedding,
            k,
            filter=filter,
            search_params=search_params,
            offset=offset,
            score_threshold=score_threshold,
            consistency=consistency,
            **kwargs,
        )

    def similarity_search_by_vector(
        self,
        embedding: list[float],
        k: int = 4,
        filter: MetadataFilter | None = None,  # noqa: A002
        search_params: models.SearchParams | None = None,
        offset: int = 0,
        score_threshold: float | None = None,
        consistency: models.ReadConsistency | None = None,
        **kwargs: Any,
    ) -> list[Document]:
        """返回与嵌入向量最相似的文档。

        Args:
            embedding: 用于查找相似文档的嵌入向量。
            k: 要返回的文档数量。
            filter: 根据元数据进行过滤。
            search_params: 额外的搜索参数
            offset:
                要返回的第一个结果的偏移量。
                可用于分页结果。
                Note: large offset values may cause performance issues.
            score_threshold:
                Define a minimal score threshold for the result.
                If defined, less similar results will not be returned.
                Score of the returned result might be higher or smaller than the
                threshold depending on the Distance function used.
                E.g. for cosine similarity only higher scores will be returned.
            consistency:
                Read consistency of the search. Defines how many replicas should be
                queried before returning the result.
                Values:
                - int - number of replicas to query, values should present in all
                        queried replicas
                - 'majority' - query all replicas, but return values present in the
                    majority of replicas
                - 'quorum' - query the majority of replicas, return values present in
                    all of them
                - 'all' - query all replicas, and return values present in all replicas
            **kwargs:
                Any other named arguments to pass through to QdrantClient.search()

        Returns:
            与查询最相似的 `Document` 对象列表。

        """
        results = self.similarity_search_with_score_by_vector(
            embedding,
            k,
            filter=filter,
            search_params=search_params,
            offset=offset,
            score_threshold=score_threshold,
            consistency=consistency,
            **kwargs,
        )
        return list(map(itemgetter(0), results))

    @sync_call_fallback
    async def asimilarity_search_by_vector(
        self,
        embedding: list[float],
        k: int = 4,
        filter: MetadataFilter | None = None,  # noqa: A002
        search_params: models.SearchParams | None = None,
        offset: int = 0,
        score_threshold: float | None = None,
        consistency: models.ReadConsistency | None = None,
        **kwargs: Any,
    ) -> list[Document]:
        """返回与嵌入向量最相似的文档。

        Args:
            embedding: 用于查找与之相似文档的嵌入向量。
            k: 返回的文档数量。
            filter: 按元数据过滤。
            search_params: 额外的搜索参数
            offset:
                返回结果的起始偏移量。
                可用于分页结果。
                Note: large offset values may cause performance issues.
            score_threshold:
                Define a minimal score threshold for the result.
                If defined, less similar results will not be returned.
                Score of the returned result might be higher or smaller than the
                threshold depending on the Distance function used.
                E.g. for cosine similarity only higher scores will be returned.
            consistency:
                Read consistency of the search. Defines how many replicas should be
                queried before returning the result.
                Values:
                - int - number of replicas to query, values should present in all
                        queried replicas
                - 'majority' - query all replicas, but return values present in the
                    majority of replicas
                - 'quorum' - query the majority of replicas, return values present in
                    all of them
                - 'all' - query all replicas, and return values present in all replicas
            **kwargs:
                Any other named arguments to pass through to
                AsyncQdrantClient.Search().

        Returns:
            与查询最相似的 `Document` 对象列表。

        """
        results = await self.asimilarity_search_with_score_by_vector(
            embedding,
            k,
            filter=filter,
            search_params=search_params,
            offset=offset,
            score_threshold=score_threshold,
            consistency=consistency,
            **kwargs,
        )
        return list(map(itemgetter(0), results))

    def similarity_search_with_score_by_vector(
        self,
        embedding: list[float],
        k: int = 4,
        filter: MetadataFilter | None = None,  # noqa: A002
        search_params: models.SearchParams | None = None,
        offset: int = 0,
        score_threshold: float | None = None,
        consistency: models.ReadConsistency | None = None,
        **kwargs: Any,
    ) -> list[tuple[Document, float]]:
        """返回与嵌入向量最相似的文档。

        Args:
            embedding: 用于查找与之相似文档的嵌入向量。
            k: 返回的文档数量。
            filter: 按元数据过滤。
            search_params: 额外的搜索参数
            offset:
                返回结果的起始偏移量。
                可用于分页结果。
                Note: large offset values may cause performance issues.
            score_threshold:
                Define a minimal score threshold for the result.
                If defined, less similar results will not be returned.
                Score of the returned result might be higher or smaller than the
                threshold depending on the Distance function used.
                E.g. for cosine similarity only higher scores will be returned.
            consistency:
                Read consistency of the search. Defines how many replicas should be
                queried before returning the result.
                Values:
                - int - number of replicas to query, values should present in all
                        queried replicas
                - 'majority' - query all replicas, but return values present in the
                    majority of replicas
                - 'quorum' - query the majority of replicas, return values present in
                    all of them
                - 'all' - query all replicas, and return values present in all replicas
            **kwargs:
                Any other named arguments to pass through to QdrantClient.search()

        Returns:
            与查询文本最相似的文档列表及每个结果的距离。

        """
        if filter is not None and isinstance(filter, dict):
            warnings.warn(
                "Using dict as a `filter` is deprecated. Please use qdrant-client "
                "filters directly: "
                "https://qdrant.tech/documentation/concepts/filtering/",
                DeprecationWarning,
                stacklevel=2,
            )
            qdrant_filter = self._qdrant_filter_from_dict(filter)
        else:
            qdrant_filter = filter

        query_vector = embedding
        if self.vector_name is not None:
            query_vector = (self.vector_name, embedding)  # type: ignore[assignment]

        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            query_filter=qdrant_filter,
            search_params=search_params,
            limit=k,
            offset=offset,
            with_payload=True,
            with_vectors=False,  # 仅进行一次快速嵌入以获取向量大小
            score_threshold=score_threshold,
            consistency=consistency,
            **kwargs,
        )
        return [
            (
                self._document_from_scored_point(
                    result,
                    self.collection_name,
                    self.content_payload_key,
                    self.metadata_payload_key,
                ),
                result.score,
            )
            for result in results
        ]

    @sync_call_fallback
    async def asimilarity_search_with_score_by_vector(
        self,
        embedding: list[float],
        k: int = 4,
        filter: MetadataFilter | None = None,  # noqa: A002
        search_params: models.SearchParams | None = None,
        offset: int = 0,
        score_threshold: float | None = None,
        consistency: models.ReadConsistency | None = None,
        **kwargs: Any,
    ) -> list[tuple[Document, float]]:
        """返回与嵌入向量最相似的文档。

        Args:
            embedding: 用于查找相似文档的嵌入向量。
            k: 要返回的文档数量。
            filter: 根据元数据进行过滤。
            search_params: 额外的搜索参数
            offset:
                要返回的第一个结果的偏移量。
                可用于分页结果。
                Note: large offset values may cause performance issues.
            score_threshold:
                Define a minimal score threshold for the result.
                If defined, less similar results will not be returned.
                Score of the returned result might be higher or smaller than the
                threshold depending on the Distance function used.
                E.g. for cosine similarity only higher scores will be returned.
            consistency:
                Read consistency of the search. Defines how many replicas should be
                queried before returning the result.
                Values:
                - int - number of replicas to query, values should present in all
                        queried replicas
                - 'majority' - query all replicas, but return values present in the
                    majority of replicas
                - 'quorum' - query the majority of replicas, return values present in
                    all of them
                - 'all' - query all replicas, and return values present in all replicas
            **kwargs:
                Any other named arguments to pass through to
                AsyncQdrantClient.Search().

        Returns:
            与查询文本最相似的文档列表及每个结果的距离。

        """
        if self.async_client is None or isinstance(
            self.async_client._client, AsyncQdrantLocal
        ):
            msg = "QdrantLocal cannot interoperate with sync and async clients"
            raise NotImplementedError(msg)
        if filter is not None and isinstance(filter, dict):
            warnings.warn(
                "Using dict as a `filter` is deprecated. Please use qdrant-client "
                "filters directly: "
                "https://qdrant.tech/documentation/concepts/filtering/",
                DeprecationWarning,
                stacklevel=2,
            )
            qdrant_filter = self._qdrant_filter_from_dict(filter)
        else:
            qdrant_filter = filter

        query_vector = embedding
        if self.vector_name is not None:
            query_vector = (self.vector_name, embedding)  # type: ignore[assignment]

        results = await self.async_client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            query_filter=qdrant_filter,
            search_params=search_params,
            limit=k,
            offset=offset,
            with_payload=True,
            with_vectors=False,  # 获取现有集合和向量的向量配置（如果已指定）。如果旧配置与当前配置不匹配，
            score_threshold=score_threshold,
            consistency=consistency,
            **kwargs,
        )
        return [
            (
                self._document_from_scored_point(
                    result,
                    self.collection_name,
                    self.content_payload_key,
                    self.metadata_payload_key,
                ),
                result.score,
            )
            for result in results
        ]

    def max_marginal_relevance_search(
        self,
        query: str,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        filter: MetadataFilter | None = None,  # noqa: A002
        search_params: models.SearchParams | None = None,
        score_threshold: float | None = None,
        consistency: models.ReadConsistency | None = None,
        **kwargs: Any,
    ) -> list[Document]:
        """返回通过最大边际相关性选择的文档。

        Maximal marginal relevance optimizes for similarity to query AND diversity
        among selected documents.

        Args:
            query: Text to look up documents similar to.
            k: Number of Documents to return.
            fetch_k: Number of Documents to fetch to pass to MMR algorithm.
            lambda_mult: Number between `0` and `1` that determines the degree
                of diversity among the results with `0` corresponding to maximum
                diversity and `1` to minimum diversity.
            filter: Filter by metadata.
            search_params: Additional search params
            score_threshold:
                Define a minimal score threshold for the result.
                If defined, less similar results will not be returned.
                Score of the returned result might be higher or smaller than the
                threshold depending on the Distance function used.
                E.g. for cosine similarity only higher scores will be returned.
            consistency:
                Read consistency of the search. Defines how many replicas should be
                queried before returning the result.
                Values:
                - int - number of replicas to query, values should present in all
                        queried replicas
                - 'majority' - query all replicas, but return values present in the
                    majority of replicas
                - 'quorum' - query the majority of replicas, return values present in
                    all of them
                - 'all' - query all replicas, and return values present in all replicas
            **kwargs:
                Any other named arguments to pass through to QdrantClient.search()

        Returns:
            通过最大边际相关性选择的 `Document` 对象列表。

        """
        query_embedding = self._embed_query(query)
        return self.max_marginal_relevance_search_by_vector(
            query_embedding,
            k=k,
            fetch_k=fetch_k,
            lambda_mult=lambda_mult,
            filter=filter,
            search_params=search_params,
            score_threshold=score_threshold,
            consistency=consistency,
            **kwargs,
        )

    @sync_call_fallback
    async def amax_marginal_relevance_search(
        self,
        query: str,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        filter: MetadataFilter | None = None,  # noqa: A002
        search_params: models.SearchParams | None = None,
        score_threshold: float | None = None,
        consistency: models.ReadConsistency | None = None,
        **kwargs: Any,
    ) -> list[Document]:
        """返回使用最大边际相关性选择的文档。

        Maximal marginal relevance optimizes for similarity to query AND diversity
        among selected documents.

        Args:
            query: Text to look up documents similar to.
            k: Number of Documents to return.
            fetch_k: Number of Documents to fetch to pass to MMR algorithm.
            lambda_mult: Number between `0` and `1` that determines the degree
                        of diversity among the results with `0` corresponding
                        to maximum diversity and `1` to minimum diversity.
            filter: Filter by metadata.
            search_params: Additional search params
            score_threshold:
                Define a minimal score threshold for the result.
                If defined, less similar results will not be returned.
                Score of the returned result might be higher or smaller than the
                threshold depending on the Distance function used.
                E.g. for cosine similarity only higher scores will be returned.
            consistency:
                Read consistency of the search. Defines how many replicas should be
                queried before returning the result.
                Values:
                - `int` - number of replicas to query, values should present in all
                        queried replicas
                - `'majority'` - query all replicas, but return values present in the
                    majority of replicas
                - `'quorum'` - query the majority of replicas, return values present in
                    all of them
                - `'all'` - query all replicas, and return values present in all
                    replicas
            **kwargs:
                Any other named arguments to pass through to
                `AsyncQdrantClient.Search()`.

        Returns:
            通过最大边际相关性选择的 `Document` 对象列表。

        """
        query_embedding = await self._aembed_query(query)
        return await self.amax_marginal_relevance_search_by_vector(
            query_embedding,
            k=k,
            fetch_k=fetch_k,
            lambda_mult=lambda_mult,
            filter=filter,
            search_params=search_params,
            score_threshold=score_threshold,
            consistency=consistency,
            **kwargs,
        )

    def max_marginal_relevance_search_by_vector(
        self,
        embedding: list[float],
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        filter: MetadataFilter | None = None,  # noqa: A002
        search_params: models.SearchParams | None = None,
        score_threshold: float | None = None,
        consistency: models.ReadConsistency | None = None,
        **kwargs: Any,
    ) -> list[Document]:
        """返回使用最大边际相关性选择的文档。

        Maximal marginal relevance optimizes for similarity to query AND diversity
        among selected documents.

        Args:
            embedding: Embedding to look up documents similar to.
            k: Number of Documents to return.
            fetch_k: Number of Documents to fetch to pass to MMR algorithm.
            lambda_mult: Number between `0` and `1` that determines the degree
                        of diversity among the results with `0` corresponding
                        to maximum diversity and `1` to minimum diversity.
            filter: Filter by metadata.
            search_params: Additional search params
            score_threshold:
                Define a minimal score threshold for the result.
                If defined, less similar results will not be returned.
                Score of the returned result might be higher or smaller than the
                threshold depending on the Distance function used.
                e.g. for cosine similarity only higher scores will be returned.
            consistency:
                Read consistency of the search. Defines how many replicas should be
                queried before returning the result.
                Values:
                - `int` - number of replicas to query, values should present in all
                        queried replicas
                - `'majority'` - query all replicas, but return values present in the
                    majority of replicas
                - `'quorum'` - query the majority of replicas, return values present in
                    all of them
                - `'all'` - query all replicas, and return values present in all
                    replicas
            **kwargs:
                Any other named arguments to pass through to `QdrantClient.search()`

        Returns:
            通过最大边际相关性选择的 `Document` 对象列表。

        """
        results = self.max_marginal_relevance_search_with_score_by_vector(
            embedding,
            k=k,
            fetch_k=fetch_k,
            lambda_mult=lambda_mult,
            filter=filter,
            search_params=search_params,
            score_threshold=score_threshold,
            consistency=consistency,
            **kwargs,
        )
        return list(map(itemgetter(0), results))

    @sync_call_fallback
    async def amax_marginal_relevance_search_by_vector(
        self,
        embedding: list[float],
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        filter: MetadataFilter | None = None,  # noqa: A002
        search_params: models.SearchParams | None = None,
        score_threshold: float | None = None,
        consistency: models.ReadConsistency | None = None,
        **kwargs: Any,
    ) -> list[Document]:
        """返回使用最大边际相关性选择的文档。

        Maximal marginal relevance optimizes for similarity to query AND diversity
        among selected documents.

        Args:
            embedding: Embedding vector to look up documents similar to.
            k: Number of `Document` objects to return.
            fetch_k: Number of `Document` to fetch to pass to MMR algorithm.
            lambda_mult: Number between `0` and `1` that determines the degree
                        of diversity among the results with `0` corresponding
                        to maximum diversity and `1` to minimum diversity.
            filter: Filter by metadata.
            search_params: Additional search params
            score_threshold:
                Define a minimal score threshold for the result.
                If defined, less similar results will not be returned.
                Score of the returned result might be higher or smaller than the
                threshold depending on the Distance function used.
                E.g. for cosine similarity only higher scores will be returned.
            consistency:
                Read consistency of the search. Defines how many replicas should be
                queried before returning the result.
                Values:
                - `int` - number of replicas to query, values should present in all
                        queried replicas
                - `'majority'` - query all replicas, but return values present in the
                    majority of replicas
                - `'quorum'` - query the majority of replicas, return values present in
                    all of them
                - `'all'` - query all replicas, and return values present in all
                    replicas
            **kwargs:
                Any other named arguments to pass through to
                `AsyncQdrantClient.Search()`.

        Returns:
            List of `Document` objects selected by maximal marginal relevance and
            distance for each.

        """
        results = await self.amax_marginal_relevance_search_with_score_by_vector(
            embedding,
            k=k,
            fetch_k=fetch_k,
            lambda_mult=lambda_mult,
            filter=filter,
            search_params=search_params,
            score_threshold=score_threshold,
            consistency=consistency,
            **kwargs,
        )
        return list(map(itemgetter(0), results))

    def max_marginal_relevance_search_with_score_by_vector(
        self,
        embedding: list[float],
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        filter: MetadataFilter | None = None,  # noqa: A002
        search_params: models.SearchParams | None = None,
        score_threshold: float | None = None,
        consistency: models.ReadConsistency | None = None,
        **kwargs: Any,
    ) -> list[tuple[Document, float]]:
        """返回使用最大边际相关性选择的文档。

        Maximal marginal relevance optimizes for similarity to query AND diversity
        among selected documents.

        Args:
            embedding: Embedding vector to look up documents similar to.
            k: Number of Documents to return.
            fetch_k: Number of Documents to fetch to pass to MMR algorithm.
            lambda_mult: Number between `0` and `1` that determines the degree of
                diversity among the results with `0` corresponding to maximum diversity
                and `1` to minimum diversity.
            filter: Filter by metadata.
            search_params: Additional search params
            score_threshold:
                Define a minimal score threshold for the result.
                If defined, less similar results will not be returned.
                Score of the returned result might be higher or smaller than the
                threshold depending on the Distance function used.
                E.g. for cosine similarity only higher scores will be returned.
            consistency:
                Read consistency of the search. Defines how many replicas should be
                queried before returning the result.
                Values:
                - int - number of replicas to query, values should present in all
                        queried replicas
                - 'majority' - query all replicas, but return values present in the
                    majority of replicas
                - 'quorum' - query the majority of replicas, return values present in
                    all of them
                - 'all' - query all replicas, and return values present in all replicas
            **kwargs:
                Any other named arguments to pass through to QdrantClient.search()

        Returns:
            List of `Document` objects selected by maximal marginal relevance and
                distance for each.
        """
        query_vector = embedding
        if self.vector_name is not None:
            query_vector = (self.vector_name, query_vector)  # type: ignore[assignment]

        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            query_filter=filter,
            search_params=search_params,
            limit=fetch_k,
            with_payload=True,
            with_vectors=True,
            score_threshold=score_threshold,
            consistency=consistency,
            **kwargs,
        )
        embeddings = [
            result.vector.get(self.vector_name)  # type: ignore[index, union-attr]
            if self.vector_name is not None
            else result.vector
            for result in results
        ]
        mmr_selected = maximal_marginal_relevance(
            np.array(embedding), embeddings, k=k, lambda_mult=lambda_mult
        )
        return [
            (
                self._document_from_scored_point(
                    results[i],
                    self.collection_name,
                    self.content_payload_key,
                    self.metadata_payload_key,
                ),
                results[i].score,
            )
            for i in mmr_selected
        ]

    @sync_call_fallback
    async def amax_marginal_relevance_search_with_score_by_vector(
        self,
        embedding: list[float],
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        filter: MetadataFilter | None = None,  # noqa: A002
        search_params: models.SearchParams | None = None,
        score_threshold: float | None = None,
        consistency: models.ReadConsistency | None = None,
        **kwargs: Any,
    ) -> list[tuple[Document, float]]:
        """返回使用最大边际相关性选择的文档。

        Maximal marginal relevance optimizes for similarity to query AND diversity
        among selected documents.

        Args:
            embedding: Embedding vector to look up documents similar to.
            k: Number of Documents to return.
            fetch_k: Number of Documents to fetch to pass to MMR algorithm.
            lambda_mult: Number between `0` and `1` that determines the degree of
                diversity among the results with `0` corresponding to maximum diversity
                and `1` to minimum diversity.
            filter: Filter by metadata.
            search_params: Additional search params.
            score_threshold: Define a minimal score threshold for the result.
            consistency: Read consistency of the search.
            **kwargs: Additional keyword arguments.

        Returns:
            List of `Document` objects selected by maximal marginal relevance and
                distance for each.
        """
        if self.async_client is None or isinstance(
            self.async_client._client, AsyncQdrantLocal
        ):
            msg = "QdrantLocal cannot interoperate with sync and async clients"
            raise NotImplementedError(msg)
        query_vector = embedding
        if self.vector_name is not None:
            query_vector = (self.vector_name, query_vector)  # type: ignore[assignment]

        results = await self.async_client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            query_filter=filter,
            search_params=search_params,
            limit=fetch_k,
            with_payload=True,
            with_vectors=True,
            score_threshold=score_threshold,
            consistency=consistency,
            **kwargs,
        )
        embeddings = [
            result.vector.get(self.vector_name)  # type: ignore[index, union-attr]
            if self.vector_name is not None
            else result.vector
            for result in results
        ]
        mmr_selected = maximal_marginal_relevance(
            np.array(embedding), embeddings, k=k, lambda_mult=lambda_mult
        )
        return [
            (
                self._document_from_scored_point(
                    results[i],
                    self.collection_name,
                    self.content_payload_key,
                    self.metadata_payload_key,
                ),
                results[i].score,
            )
            for i in mmr_selected
        ]

    def delete(self, ids: list[str] | None = None, **kwargs: Any) -> bool | None:
        """通过向量 ID 或其他条件删除。

        Args:
            ids: 要删除的 ID 列表。
            **kwargs: 子类可能使用的其他关键字参数。

        Returns:
            删除成功时为 True，否则为 False。

        """
        result = self.client.delete(
            collection_name=self.collection_name,
            points_selector=ids,
        )
        return result.status == models.UpdateStatus.COMPLETED

    @sync_call_fallback
    async def adelete(self, ids: list[str] | None = None, **kwargs: Any) -> bool | None:
        """通过向量 ID 或其他条件删除。

        Args:
            ids: 要删除的 id 列表。
            **kwargs: 子类可能使用的其他关键字参数。

        Returns:
            如果删除成功则返回 True，否则返回 False。

        """
        if self.async_client is None or isinstance(
            self.async_client._client, AsyncQdrantLocal
        ):
            msg = "QdrantLocal cannot interoperate with sync and async clients"
            raise NotImplementedError(msg)

        result = await self.async_client.delete(
            collection_name=self.collection_name,
            points_selector=ids,
        )

        return result.status == models.UpdateStatus.COMPLETED

    @classmethod
    def from_texts(
        cls: type[Qdrant],
        texts: list[str],
        embedding: Embeddings,
        metadatas: list[dict] | None = None,
        ids: Sequence[str] | None = None,
        location: str | None = None,
        url: str | None = None,
        port: int | None = 6333,
        grpc_port: int = 6334,
        prefer_grpc: bool = False,  # noqa: FBT001, FBT002
        https: bool | None = None,  # noqa: FBT001
        api_key: str | None = None,
        prefix: str | None = None,
        timeout: int | None = None,
        host: str | None = None,
        path: str | None = None,
        collection_name: str | None = None,
        distance_func: str = "Cosine",
        content_payload_key: str = CONTENT_KEY,
        metadata_payload_key: str = METADATA_KEY,
        vector_name: str | None = VECTOR_NAME,
        batch_size: int = 64,
        shard_number: int | None = None,
        replication_factor: int | None = None,
        write_consistency_factor: int | None = None,
        on_disk_payload: bool | None = None,  # noqa: FBT001
        hnsw_config: models.HnswConfigDiff | None = None,
        optimizers_config: models.OptimizersConfigDiff | None = None,
        wal_config: models.WalConfigDiff | None = None,
        quantization_config: models.QuantizationConfig | None = None,
        init_from: models.InitFrom | None = None,
        on_disk: bool | None = None,  # noqa: FBT001
        force_recreate: bool = False,  # noqa: FBT001, FBT002
        **kwargs: Any,
    ) -> Qdrant:
        """从文本列表构建 Qdrant 包装器。

        Args:
            texts: A list of texts to be indexed in Qdrant.
            embedding: A subclass of `Embeddings`, responsible for text vectorization.
            metadatas:
                An optional list of metadata. If provided it has to be of the same
                length as a list of texts.
            ids:
                Optional list of ids to associate with the texts. Ids have to be
                uuid-like strings.
            location:
                If ':memory:' - use in-memory Qdrant instance.
                If `str` - use it as a `url` parameter.
                If `None` - fallback to relying on `host` and `port` parameters.
            url: either host or str of "scheme | None, host, port | None,
                prefix | None".
            port: Port of the REST API interface. Default: 6333
            grpc_port: Port of the gRPC interface. Default: 6334
            prefer_grpc:
                If true - use gPRC interface whenever possible in custom methods.
                Default: False
            https: If true - use HTTPS(SSL) protocol. Default: None
            api_key:
                    API key for authentication in Qdrant Cloud. Default: None
                    Can also be set via environment variable `QDRANT_API_KEY`.
            prefix:
                If not None - add prefix to the REST URL path.
                Example: service/v1 will result in
                    http://localhost:6333/service/v1/{qdrant-endpoint} for REST API.
                Default: None
            timeout:
                Timeout for REST and gRPC API requests.
                Default: 5.0 seconds for REST and unlimited for gRPC
            host:
                Host name of Qdrant service. If url and host are None, set to
                'localhost'. Default: None
            path:
                Path in which the vectors will be stored while using local mode.
                Default: None
            collection_name:
                Name of the Qdrant collection to be used. If not provided,
                it will be created randomly. Default: None
            distance_func:
                Distance function. One of: "Cosine" / "Euclid" / "Dot".
                Default: "Cosine"
            content_payload_key:
                A payload key used to store the content of the document.
                Default: "page_content"
            metadata_payload_key:
                A payload key used to store the metadata of the document.
                Default: "metadata"
            vector_name:
                Name of the vector to be used internally in Qdrant.
                Default: None
            batch_size:
                How many vectors upload per-request.
                Default: 64
            shard_number: Number of shards in collection. Default is 1, minimum is 1.
            replication_factor:
                Replication factor for collection. Default is 1, minimum is 1.
                Defines how many copies of each shard will be created.
                Have effect only in distributed mode.
            write_consistency_factor:
                Write consistency factor for collection. Default is 1, minimum is 1.
                Defines how many replicas should apply the operation for us to consider
                it successful. Increasing this number will make the collection more
                resilient to inconsistencies, but will also make it fail if not enough
                replicas are available.
                Does not have any performance impact.
                Have effect only in distributed mode.
            on_disk_payload:
                If true - point`s payload will not be stored in memory.
                It will be read from the disk every time it is requested.
                This setting saves RAM by (slightly) increasing the response time.
                Note: those payload values that are involved in filtering and are
                indexed - 保留在内存中。
            hnsw_config: HNSW 索引的参数
            optimizers_config: 优化器的参数
            wal_config: Write-Ahead-Log 的参数
            quantization_config:
                量化参数，若为 None 则禁用量化
            init_from:
                使用另一个集合中存储的数据来初始化此集合
            on_disk:
                若为 true - 向量将存储在磁盘上，降低内存使用量。
            force_recreate:
                强制重新创建集合
            **kwargs:
                传入 REST 客户端初始化的额外参数

        这是一个用户友好的接口，其功能包括：

        1. Creates embeddings, one for each text
        2. Initializes the Qdrant database as an in-memory docstore by default
            (and overridable to a remote docstore)
        3. Adds the text embeddings to the Qdrant database

        此设计旨在快速上手。

        ```python
        from langchain_qdrant import Qdrant
        from langchain_openai import OpenAIEmbeddings

        embeddings = OpenAIEmbeddings()
        qdrant = Qdrant.from_texts(texts, embeddings, "localhost")
        ```
        """
        qdrant = cls.construct_instance(
            texts,
            embedding,
            location,
            url,
            port,
            grpc_port,
            prefer_grpc,
            https,
            api_key,
            prefix,
            timeout,
            host,
            path,
            collection_name,
            distance_func,
            content_payload_key,
            metadata_payload_key,
            vector_name,
            shard_number,
            replication_factor,
            write_consistency_factor,
            on_disk_payload,
            hnsw_config,
            optimizers_config,
            wal_config,
            quantization_config,
            init_from,
            on_disk,
            force_recreate,
            **kwargs,
        )
        qdrant.add_texts(texts, metadatas, ids, batch_size)
        return qdrant

    @classmethod
    def from_existing_collection(
        cls: type[Qdrant],
        embedding: Embeddings,
        path: str | None = None,
        collection_name: str | None = None,
        location: str | None = None,
        url: str | None = None,
        port: int | None = 6333,
        grpc_port: int = 6334,
        prefer_grpc: bool = False,  # noqa: FBT001, FBT002
        https: bool | None = None,  # noqa: FBT001
        api_key: str | None = None,
        prefix: str | None = None,
        timeout: int | None = None,
        host: str | None = None,
        content_payload_key: str = CONTENT_KEY,
        metadata_payload_key: str = METADATA_KEY,
        distance_strategy: str = "COSINE",
        vector_name: str | None = VECTOR_NAME,
        **kwargs: Any,
    ) -> Qdrant:
        """获取现有 Qdrant 集合的实例。

        This method will return the instance of the store without inserting any new
        embeddings.
        """
        if collection_name is None:
            msg = "Must specify collection_name. Received None."
            raise ValueError(msg)

        client, async_client = cls._generate_clients(
            location=location,
            url=url,
            port=port,
            grpc_port=grpc_port,
            prefer_grpc=prefer_grpc,
            https=https,
            api_key=api_key,
            prefix=prefix,
            timeout=timeout,
            host=host,
            path=path,
            **kwargs,
        )
        return cls(
            client=client,
            async_client=async_client,
            collection_name=collection_name,
            embeddings=embedding,
            content_payload_key=content_payload_key,
            metadata_payload_key=metadata_payload_key,
            distance_strategy=distance_strategy,
            vector_name=vector_name,
        )

    @classmethod
    @sync_call_fallback
    async def afrom_texts(
        cls: type[Qdrant],
        texts: list[str],
        embedding: Embeddings,
        metadatas: list[dict] | None = None,
        ids: Sequence[str] | None = None,
        location: str | None = None,
        url: str | None = None,
        port: int | None = 6333,
        grpc_port: int = 6334,
        prefer_grpc: bool = False,  # noqa: FBT001, FBT002
        https: bool | None = None,  # noqa: FBT001
        api_key: str | None = None,
        prefix: str | None = None,
        timeout: int | None = None,
        host: str | None = None,
        path: str | None = None,
        collection_name: str | None = None,
        distance_func: str = "Cosine",
        content_payload_key: str = CONTENT_KEY,
        metadata_payload_key: str = METADATA_KEY,
        vector_name: str | None = VECTOR_NAME,
        batch_size: int = 64,
        shard_number: int | None = None,
        replication_factor: int | None = None,
        write_consistency_factor: int | None = None,
        on_disk_payload: bool | None = None,  # noqa: FBT001
        hnsw_config: models.HnswConfigDiff | None = None,
        optimizers_config: models.OptimizersConfigDiff | None = None,
        wal_config: models.WalConfigDiff | None = None,
        quantization_config: models.QuantizationConfig | None = None,
        init_from: models.InitFrom | None = None,
        on_disk: bool | None = None,  # noqa: FBT001
        force_recreate: bool = False,  # noqa: FBT001, FBT002
        **kwargs: Any,
    ) -> Qdrant:
        """从文本列表构建 Qdrant 包装器。

        Args:
            texts: A list of texts to be indexed in Qdrant.
            embedding: A subclass of `Embeddings`, responsible for text vectorization.
            metadatas:
                An optional list of metadata. If provided it has to be of the same
                length as a list of texts.
            ids:
                Optional list of ids to associate with the texts. Ids have to be
                uuid-like strings.
            location:
                If ':memory:' - use in-memory Qdrant instance.
                If `str` - use it as a `url` parameter.
                If `None` - fallback to relying on `host` and `port` parameters.
            url: either host or str of "scheme | None, host, port | None,
                prefix | None".
            port: Port of the REST API interface. Default: 6333
            grpc_port: Port of the gRPC interface. Default: 6334
            prefer_grpc:
                If true - use gPRC interface whenever possible in custom methods.
                Default: False
            https: If true - use HTTPS(SSL) protocol. Default: None
            api_key:
                    API key for authentication in Qdrant Cloud. Default: None
                    Can also be set via environment variable `QDRANT_API_KEY`.
            prefix:
                If not None - add prefix to the REST URL path.
                Example: service/v1 will result in
                    http://localhost:6333/service/v1/{qdrant-endpoint} for REST API.
                Default: None
            timeout:
                Timeout for REST and gRPC API requests.
                Default: 5.0 seconds for REST and unlimited for gRPC
            host:
                Host name of Qdrant service. If url and host are None, set to
                'localhost'. Default: None
            path:
                Path in which the vectors will be stored while using local mode.
                Default: None
            collection_name:
                Name of the Qdrant collection to be used. If not provided,
                it will be created randomly. Default: None
            distance_func:
                Distance function. One of: "Cosine" / "Euclid" / "Dot".
                Default: "Cosine"
            content_payload_key:
                A payload key used to store the content of the document.
                Default: "page_content"
            metadata_payload_key:
                A payload key used to store the metadata of the document.
                Default: "metadata"
            vector_name:
                Name of the vector to be used internally in Qdrant.
                Default: None
            batch_size:
                How many vectors upload per-request.
                Default: 64
            shard_number: Number of shards in collection. Default is 1, minimum is 1.
            replication_factor:
                Replication factor for collection. Default is 1, minimum is 1.
                Defines how many copies of each shard will be created.
                Have effect only in distributed mode.
            write_consistency_factor:
                Write consistency factor for collection. Default is 1, minimum is 1.
                Defines how many replicas should apply the operation for us to consider
                it successful. Increasing this number will make the collection more
                resilient to inconsistencies, but will also make it fail if not enough
                replicas are available.
                Does not have any performance impact.
                Have effect only in distributed mode.
            on_disk_payload:
                If true - point`s payload will not be stored in memory.
                It will be read from the disk every time it is requested.
                This setting saves RAM by (slightly) increasing the response time.
                Note: those payload values that are involved in filtering and are
                indexed - 保留在内存中。
            hnsw_config: HNSW 索引的参数
            optimizers_config: 优化器的参数
            wal_config: 写前日志（Write-Ahead-Log）的参数
            quantization_config:
                量化参数，若为 None 则禁用量化
            init_from:
                使用另一个集合中存储的数据来初始化此集合
            on_disk:
                若为 true，则点的 payload 不会存储在内存中。
                每次请求时都会从磁盘读取。
                此设置通过略微增加响应时间来节省 RAM。
                Note: those payload values that are involved in filtering and are
                indexed - 保留在内存中。
            force_recreate:
                强制重新创建集合
            **kwargs:
                直接传递给 REST 客户端初始化的额外参数

        这是一个用户友好的接口，其功能包括：

        1. 为每段文本创建嵌入
        2. 默认以内存中的 docstore 方式初始化 Qdrant 数据库
            （可覆盖为远程 docstore）
        3. 将文本嵌入添加到 Qdrant 数据库中

        旨在快速上手使用。

        ```python
        from langchain_qdrant import Qdrant
        from langchain_openai import OpenAIEmbeddings

        embeddings = OpenAIEmbeddings()
        qdrant = await Qdrant.afrom_texts(texts, embeddings, "localhost")
        ```
        """
        qdrant = await cls.aconstruct_instance(
            texts,
            embedding,
            location,
            url,
            port,
            grpc_port,
            prefer_grpc,
            https,
            api_key,
            prefix,
            timeout,
            host,
            path,
            collection_name,
            distance_func,
            content_payload_key,
            metadata_payload_key,
            vector_name,
            shard_number,
            replication_factor,
            write_consistency_factor,
            on_disk_payload,
            hnsw_config,
            optimizers_config,
            wal_config,
            quantization_config,
            init_from,
            on_disk,
            force_recreate,
            **kwargs,
        )
        await qdrant.aadd_texts(texts, metadatas, ids, batch_size)
        return qdrant

    @classmethod
    def construct_instance(
        cls: type[Qdrant],
        texts: list[str],
        embedding: Embeddings,
        location: str | None = None,
        url: str | None = None,
        port: int | None = 6333,
        grpc_port: int = 6334,
        prefer_grpc: bool = False,  # noqa: FBT001, FBT002
        https: bool | None = None,  # noqa: FBT001
        api_key: str | None = None,
        prefix: str | None = None,
        timeout: int | None = None,
        host: str | None = None,
        path: str | None = None,
        collection_name: str | None = None,
        distance_func: str = "Cosine",
        content_payload_key: str = CONTENT_KEY,
        metadata_payload_key: str = METADATA_KEY,
        vector_name: str | None = VECTOR_NAME,
        shard_number: int | None = None,
        replication_factor: int | None = None,
        write_consistency_factor: int | None = None,
        on_disk_payload: bool | None = None,  # noqa: FBT001
        hnsw_config: models.HnswConfigDiff | None = None,
        optimizers_config: models.OptimizersConfigDiff | None = None,
        wal_config: models.WalConfigDiff | None = None,
        quantization_config: models.QuantizationConfig | None = None,
        init_from: models.InitFrom | None = None,
        on_disk: bool | None = None,  # noqa: FBT001
        force_recreate: bool = False,  # noqa: FBT001, FBT002
        **kwargs: Any,
    ) -> Qdrant:
                # 将引发异常。
        partial_embeddings = embedding.embed_documents(texts[:1])
        vector_size = len(partial_embeddings[0])
        collection_name = collection_name or uuid.uuid4().hex
        distance_func = distance_func.upper()
        client, async_client = cls._generate_clients(
            location=location,
            url=url,
            port=port,
            grpc_port=grpc_port,
            prefer_grpc=prefer_grpc,
            https=https,
            api_key=api_key,
            prefix=prefix,
            timeout=timeout,
            host=host,
            path=path,
            **kwargs,
        )
        collection_exists = client.collection_exists(collection_name)

        if collection_exists and force_recreate:
            client.delete_collection(collection_name)
            collection_exists = False

        if collection_exists:
                        # 检查向量配置是否具有相同的维度。
                        # 如果提供了向量名称，我们将使用命名向量功能
                        # 仅针对单个向量。
            collection_info = client.get_collection(collection_name=collection_name)
            current_vector_config = collection_info.config.params.vectors
            if isinstance(current_vector_config, dict) and vector_name is not None:
                if vector_name not in current_vector_config:
                    msg = (
                        f"Existing Qdrant collection {collection_name} does not "
                        f"contain vector named {vector_name}. Did you mean one of the "
                        f"existing vectors: {', '.join(current_vector_config.keys())}? "
                        f"If you want to recreate the collection, set `force_recreate` "
                        f"parameter to `True`."
                    )
                    raise QdrantException(msg)
                current_vector_config = current_vector_config.get(vector_name)  # type: ignore[assignment]
            elif isinstance(current_vector_config, dict) and vector_name is None:
                msg = (
                    f"Existing Qdrant collection {collection_name} uses named vectors. "
                    f"If you want to reuse it, please set `vector_name` to any of the "
                    f"existing named vectors: "
                    f"{', '.join(current_vector_config.keys())}."
                    f"If you want to recreate the collection, set `force_recreate` "
                    f"parameter to `True`."
                )
                raise QdrantException(msg)
            elif (
                not isinstance(current_vector_config, dict) and vector_name is not None
            ):
                msg = (
                    f"Existing Qdrant collection {collection_name} doesn't use named "
                    f"vectors. If you want to reuse it, please set `vector_name` to "
                    f"`None`. If you want to recreate the collection, set "
                    f"`force_recreate` parameter to `True`."
                )
                raise QdrantException(msg)
            if not isinstance(current_vector_config, models.VectorParams):
                msg = (
                    "Expected current_vector_config to be an instance of "
                    f"models.VectorParams, but got {type(current_vector_config)}"
                )
                raise ValueError(msg)
                        # 仅进行一次快速嵌入以获取向量大小
            if current_vector_config.size != vector_size:
                msg = (
                    f"Existing Qdrant collection is configured for vectors with "
                    f"{current_vector_config.size} "
                    f"dimensions. Selected embeddings are {vector_size}-dimensional. "
                    f"If you want to recreate the collection, set `force_recreate` "
                    f"parameter to `True`."
                )
                raise QdrantException(msg)

            current_distance_func = (
                current_vector_config.distance.name.upper()  # type: ignore[union-attr]
            )
            if current_distance_func != distance_func:
                msg = (
                    f"Existing Qdrant collection is configured for "
                    f"{current_distance_func} similarity, but requested "
                    f"{distance_func}. Please set `distance_func` parameter to "
                    f"`{current_distance_func}` if you want to reuse it. "
                    f"If you want to recreate the collection, set `force_recreate` "
                    f"parameter to `True`."
                )
                raise QdrantException(msg)
        else:
            vectors_config = models.VectorParams(
                size=vector_size,
                distance=models.Distance[distance_func],
                on_disk=on_disk,
            )

                        # 获取现有集合和向量的向量配置（如果已指定）。如果旧配置与当前配置不匹配，
                        # 将引发异常。
            if vector_name is not None:
                vectors_config = {  # type: ignore[assignment]
                    vector_name: vectors_config,
                }

            client.create_collection(
                collection_name=collection_name,
                vectors_config=vectors_config,
                shard_number=shard_number,
                replication_factor=replication_factor,
                write_consistency_factor=write_consistency_factor,
                on_disk_payload=on_disk_payload,
                hnsw_config=hnsw_config,
                optimizers_config=optimizers_config,
                wal_config=wal_config,
                quantization_config=quantization_config,
                init_from=init_from,
                timeout=timeout,  # type: ignore[arg-type]
            )
        return cls(
            client=client,
            collection_name=collection_name,
            embeddings=embedding,
            content_payload_key=content_payload_key,
            metadata_payload_key=metadata_payload_key,
            distance_strategy=distance_func,
            vector_name=vector_name,
            async_client=async_client,
        )

    @classmethod
    async def aconstruct_instance(
        cls: type[Qdrant],
        texts: list[str],
        embedding: Embeddings,
        location: str | None = None,
        url: str | None = None,
        port: int | None = 6333,
        grpc_port: int = 6334,
        prefer_grpc: bool = False,  # noqa: FBT001, FBT002
        https: bool | None = None,  # noqa: FBT001
        api_key: str | None = None,
        prefix: str | None = None,
        timeout: int | None = None,
        host: str | None = None,
        path: str | None = None,
        collection_name: str | None = None,
        distance_func: str = "Cosine",
        content_payload_key: str = CONTENT_KEY,
        metadata_payload_key: str = METADATA_KEY,
        vector_name: str | None = VECTOR_NAME,
        shard_number: int | None = None,
        replication_factor: int | None = None,
        write_consistency_factor: int | None = None,
        on_disk_payload: bool | None = None,  # noqa: FBT001
        hnsw_config: models.HnswConfigDiff | None = None,
        optimizers_config: models.OptimizersConfigDiff | None = None,
        wal_config: models.WalConfigDiff | None = None,
        quantization_config: models.QuantizationConfig | None = None,
        init_from: models.InitFrom | None = None,
        on_disk: bool | None = None,  # noqa: FBT001
        force_recreate: bool = False,  # noqa: FBT001, FBT002
        **kwargs: Any,
    ) -> Qdrant:
                # 检查向量配置是否具有相同的维度。
        partial_embeddings = await embedding.aembed_documents(texts[:1])
        vector_size = len(partial_embeddings[0])
        collection_name = collection_name or uuid.uuid4().hex
        distance_func = distance_func.upper()
        client, async_client = cls._generate_clients(
            location=location,
            url=url,
            port=port,
            grpc_port=grpc_port,
            prefer_grpc=prefer_grpc,
            https=https,
            api_key=api_key,
            prefix=prefix,
            timeout=timeout,
            host=host,
            path=path,
            **kwargs,
        )

        collection_exists = client.collection_exists(collection_name)

        if collection_exists and force_recreate:
            client.delete_collection(collection_name)
            collection_exists = False

        if collection_exists:
                        # 如果提供了向量名称，我们将使用命名向量功能
                        # 仅针对单个向量。
                        # 为批次中的每个文本获取相应的元数据和 id
            collection_info = client.get_collection(collection_name=collection_name)
            current_vector_config = collection_info.config.params.vectors
            if isinstance(current_vector_config, dict) and vector_name is not None:
                if vector_name not in current_vector_config:
                    msg = (
                        f"Existing Qdrant collection {collection_name} does not "
                        f"contain vector named {vector_name}. Did you mean one of the "
                        f"existing vectors: {', '.join(current_vector_config.keys())}? "
                        f"If you want to recreate the collection, set `force_recreate` "
                        f"parameter to `True`."
                    )
                    raise QdrantException(msg)
                current_vector_config = current_vector_config.get(vector_name)  # type: ignore[assignment]
            elif isinstance(current_vector_config, dict) and vector_name is None:
                msg = (
                    f"Existing Qdrant collection {collection_name} uses named vectors. "
                    f"If you want to reuse it, please set `vector_name` to any of the "
                    f"existing named vectors: "
                    f"{', '.join(current_vector_config.keys())}."
                    f"If you want to recreate the collection, set `force_recreate` "
                    f"parameter to `True`."
                )
                raise QdrantException(msg)
            elif (
                not isinstance(current_vector_config, dict) and vector_name is not None
            ):
                msg = (
                    f"Existing Qdrant collection {collection_name} doesn't use named "
                    f"vectors. If you want to reuse it, please set `vector_name` to "
                    f"`None`. If you want to recreate the collection, set "
                    f"`force_recreate` parameter to `True`."
                )
                raise QdrantException(msg)
            if not isinstance(current_vector_config, models.VectorParams):
                msg = (
                    "Expected current_vector_config to be an instance of "
                    f"models.VectorParams, but got {type(current_vector_config)}"
                )
                raise ValueError(msg)

                        # 为批次中的所有文本生成嵌入
            if current_vector_config.size != vector_size:
                msg = (
                    f"Existing Qdrant collection is configured for vectors with "
                    f"{current_vector_config.size} "
                    f"dimensions. Selected embeddings are {vector_size}-dimensional. "
                    f"If you want to recreate the collection, set `force_recreate` "
                    f"parameter to `True`."
                )
                raise QdrantException(msg)

            current_distance_func = (
                current_vector_config.distance.name.upper()  # type: ignore[union-attr]
            )
            if current_distance_func != distance_func:
                msg = (
                    f"Existing Qdrant collection is configured for "
                    f"{current_vector_config.distance} "  # type: ignore[union-attr]
                    f"similarity. Please set `distance_func` parameter to "
                    f"`{distance_func}` if you want to reuse it. If you want to "
                    f"recreate the collection, set `force_recreate` parameter to "
                    f"`True`."
                )
                raise QdrantException(msg)
        else:
            vectors_config = models.VectorParams(
                size=vector_size,
                distance=models.Distance[distance_func],
                on_disk=on_disk,
            )

                        # 为批次中的每个文本获取相应的元数据和 id
                        # 为批次中的所有文本生成嵌入
            if vector_name is not None:
                vectors_config = {  # type: ignore[assignment]
                    vector_name: vectors_config,
                }

            client.create_collection(
                collection_name=collection_name,
                vectors_config=vectors_config,
                shard_number=shard_number,
                replication_factor=replication_factor,
                write_consistency_factor=write_consistency_factor,
                on_disk_payload=on_disk_payload,
                hnsw_config=hnsw_config,
                optimizers_config=optimizers_config,
                wal_config=wal_config,
                quantization_config=quantization_config,
                init_from=init_from,
                timeout=timeout,  # type: ignore[arg-type]
            )
        return cls(
            client=client,
            collection_name=collection_name,
            embeddings=embedding,
            content_payload_key=content_payload_key,
            metadata_payload_key=metadata_payload_key,
            distance_strategy=distance_func,
            vector_name=vector_name,
            async_client=async_client,
        )

    @staticmethod
    def _cosine_relevance_score_fn(distance: float) -> float:
        """将距离归一化为 [0, 1] 范围内的得分。"""
        return (distance + 1.0) / 2.0

    def _select_relevance_score_fn(self) -> Callable[[float], float]:
        """您的‘正确’相关性函数可能因以下因素而有所不同。

        例如：
        - 向量存储使用的距离/相似性度量
        - 嵌入的尺度（OpenAI 的是单位范数。许多其他模型不是！）
        - 嵌入维度
        - 等等
        """
        if self.distance_strategy == "COSINE":
            return self._cosine_relevance_score_fn
        if self.distance_strategy == "DOT":
            return self._max_inner_product_relevance_score_fn
        if self.distance_strategy == "EUCLID":
            return self._euclidean_relevance_score_fn
        msg = (
            "Unknown distance strategy, must be cosine, max_inner_product, or euclidean"
        )
        raise ValueError(msg)

    def _similarity_search_with_relevance_scores(
        self,
        query: str,
        k: int = 4,
        **kwargs: Any,
    ) -> list[tuple[Document, float]]:
        """返回文档及其相关性得分，范围在 `[0, 1]` 之间。

        `0` 表示不相似，`1` 表示最相似。

        Args:
            query: 输入文本
            k: 返回文档的数量。
            **kwargs: 传递给相似性搜索的参数。

                应包含 `score_threshold`，一个可选的浮点值
                在 `0` 到 `1` 之间，用于过滤检索结果集。

        Returns:
            包含 `(doc, similarity_score)` 元组的列表

        """
        return self.similarity_search_with_score(query, k, **kwargs)

    @sync_call_fallback
    async def _asimilarity_search_with_relevance_scores(
        self,
        query: str,
        k: int = 4,
        **kwargs: Any,
    ) -> list[tuple[Document, float]]:
        """返回文档及其相关性得分，范围在 `[0, 1]` 之间。

        `0` 表示不相似，`1` 表示最相似。

        Args:
            query: 输入文本
            k: 返回文档的数量。
            **kwargs: 传递给相似性搜索的参数。

                应包含 `score_threshold`，一个可选的浮点值
                在 `0` 到 `1` 之间，用于过滤检索结果集。

        Returns:
            包含 `(doc, similarity_score)` 元组的列表

        """
        return await self.asimilarity_search_with_score(query, k, **kwargs)

    @classmethod
    def _build_payloads(
        cls,
        texts: Iterable[str],
        metadatas: list[dict] | None,
        content_payload_key: str,
        metadata_payload_key: str,
    ) -> list[dict]:
        payloads = []
        for i, text in enumerate(texts):
            if text is None:
                msg = (
                    "At least one of the texts is None. Please remove it before "
                    "calling .from_texts or .add_texts on Qdrant instance."
                )
                raise ValueError(msg)
            metadata = metadatas[i] if metadatas is not None else None
            payloads.append(
                {
                    content_payload_key: text,
                    metadata_payload_key: metadata,
                }
            )

        return payloads

    @classmethod
    def _document_from_scored_point(
        cls,
        scored_point: Any,
        collection_name: str,
        content_payload_key: str,
        metadata_payload_key: str,
    ) -> Document:
        metadata = scored_point.payload.get(metadata_payload_key) or {}
        metadata["_id"] = scored_point.id
        metadata["_collection_name"] = collection_name
        return Document(
            page_content=scored_point.payload.get(content_payload_key, ""),
            metadata=metadata,
        )

    def _build_condition(self, key: str, value: Any) -> list[models.FieldCondition]:
        out = []

        if isinstance(value, dict):
            for _key, _value in value.items():
                out.extend(self._build_condition(f"{key}.{_key}", _value))
        elif isinstance(value, list):
            for _value in value:
                if isinstance(_value, dict):
                    out.extend(self._build_condition(f"{key}[]", _value))
                else:
                    out.extend(self._build_condition(f"{key}", _value))
        else:
            out.append(
                models.FieldCondition(
                    key=f"{self.metadata_payload_key}.{key}",
                    match=models.MatchValue(value=value),
                )
            )

        return out

    def _qdrant_filter_from_dict(
        self, filter_: DictFilter | None
    ) -> models.Filter | None:
        if not filter_:
            return None

        return models.Filter(
            must=[
                condition
                for key, value in filter_.items()  # type: ignore[union-attr]
                for condition in self._build_condition(key, value)
            ]
        )

    def _embed_query(self, query: str) -> list[float]:
        """嵌入查询文本。

        用于与 `embedding_function` 参数保持向后兼容。

        Args:
            query: 查询文本。

        Returns:
            表示查询嵌入的浮点数列表。

        """
        if self.embeddings is not None:
            embedding = self.embeddings.embed_query(query)
        elif self._embeddings_function is not None:
            embedding = self._embeddings_function(query)
        else:
            msg = "Neither of embeddings or embedding_function is set"
            raise ValueError(msg)
        return embedding.tolist() if hasattr(embedding, "tolist") else embedding

    async def _aembed_query(self, query: str) -> list[float]:
        """异步嵌入查询文本。

        用于与 `embedding_function` 参数保持向后兼容。

        Args:
            query: 查询文本。

        Returns:
            表示查询嵌入的浮点数列表。

        """
        if self.embeddings is not None:
            embedding = await self.embeddings.aembed_query(query)
        elif self._embeddings_function is not None:
            embedding = self._embeddings_function(query)
        else:
            msg = "Neither of embeddings or embedding_function is set"
            raise ValueError(msg)
        return embedding.tolist() if hasattr(embedding, "tolist") else embedding

    def _embed_texts(self, texts: Iterable[str]) -> list[list[float]]:
        """嵌入搜索文本。

        用于与 `embedding_function` 参数保持向后兼容。

        Args:
            texts: 要嵌入的文本序列。

        Returns:
            表示文本嵌入的浮点数列表。

        """
        if self.embeddings is not None:
            embeddings = self.embeddings.embed_documents(list(texts))
            if hasattr(embeddings, "tolist"):
                embeddings = embeddings.tolist()
        elif self._embeddings_function is not None:
            embeddings = []
            for text in texts:
                embedding = self._embeddings_function(text)
                if hasattr(embeddings, "tolist"):
                    embedding = embedding.tolist()
                embeddings.append(embedding)
        else:
            msg = "Neither of embeddings or embedding_function is set"
            raise ValueError(msg)

        return embeddings

    async def _aembed_texts(self, texts: Iterable[str]) -> list[list[float]]:
        """嵌入搜索文本。

        用于与 `embedding_function` 参数保持向后兼容。

        Args:
            texts: 要嵌入的文本序列。

        Returns:
            表示文本嵌入的浮点数列表。

        """
        if self.embeddings is not None:
            embeddings = await self.embeddings.aembed_documents(list(texts))
            if hasattr(embeddings, "tolist"):
                embeddings = embeddings.tolist()
        elif self._embeddings_function is not None:
            embeddings = []
            for text in texts:
                embedding = self._embeddings_function(text)
                if hasattr(embeddings, "tolist"):
                    embedding = embedding.tolist()
                embeddings.append(embedding)
        else:
            msg = "Neither of embeddings or embedding_function is set"
            raise ValueError(msg)

        return embeddings

    def _generate_rest_batches(
        self,
        texts: Iterable[str],
        metadatas: list[dict] | None = None,
        ids: Sequence[str] | None = None,
        batch_size: int = 64,
    ) -> Generator[tuple[list[str], list[models.PointStruct]], None, None]:
        texts_iterator = iter(texts)
        metadatas_iterator = iter(metadatas or [])
        ids_iterator = iter(ids or [uuid.uuid4().hex for _ in iter(texts)])
        while batch_texts := list(islice(texts_iterator, batch_size)):
                        # 本地 Qdrant 无法与同步和异步客户端共存
            batch_metadatas = list(islice(metadatas_iterator, batch_size)) or None
            batch_ids = list(islice(ids_iterator, batch_size))

                        # 在此情况下，我们回退到同步操作
            batch_embeddings = self._embed_texts(batch_texts)

            points = [
                models.PointStruct(
                    id=point_id,
                    vector=vector  # type: ignore[arg-type]
                    if self.vector_name is None
                    else {self.vector_name: vector},
                    payload=payload,
                )
                for point_id, vector, payload in zip(
                    batch_ids,
                    batch_embeddings,
                    self._build_payloads(
                        batch_texts,
                        batch_metadatas,
                        self.content_payload_key,
                        self.metadata_payload_key,
                    ),
                    strict=False,
                )
            ]

            yield batch_ids, points

    async def _agenerate_rest_batches(
        self,
        texts: Iterable[str],
        metadatas: list[dict] | None = None,
        ids: Sequence[str] | None = None,
        batch_size: int = 64,
    ) -> AsyncGenerator[tuple[list[str], list[models.PointStruct]], None]:
        texts_iterator = iter(texts)
        metadatas_iterator = iter(metadatas or [])
        ids_iterator = iter(ids or [uuid.uuid4().hex for _ in iter(texts)])
        while batch_texts := list(islice(texts_iterator, batch_size)):
                        # 获取批次中每个文本对应的元数据和 id
            batch_metadatas = list(islice(metadatas_iterator, batch_size)) or None
            batch_ids = list(islice(ids_iterator, batch_size))

                        # 为批量中的所有文本生成嵌入
            batch_embeddings = await self._aembed_texts(batch_texts)

            points = [
                models.PointStruct(
                    id=point_id,
                    vector=vector  # type: ignore[arg-type]
                    if self.vector_name is None
                    else {self.vector_name: vector},
                    payload=payload,
                )
                for point_id, vector, payload in zip(
                    batch_ids,
                    batch_embeddings,
                    self._build_payloads(
                        batch_texts,
                        batch_metadatas,
                        self.content_payload_key,
                        self.metadata_payload_key,
                    ),
                    strict=False,
                )
            ]

            yield batch_ids, points

    @staticmethod
    def _generate_clients(
        location: str | None = None,
        url: str | None = None,
        port: int | None = 6333,
        grpc_port: int = 6334,
        prefer_grpc: bool = False,  # noqa: FBT001, FBT002
        https: bool | None = None,  # noqa: FBT001
        api_key: str | None = None,
        prefix: str | None = None,
        timeout: int | None = None,
        host: str | None = None,
        path: str | None = None,
        **kwargs: Any,
    ) -> tuple[QdrantClient, AsyncQdrantClient | None]:
        if api_key is None:
            api_key = os.getenv("QDRANT_API_KEY")

        sync_client = QdrantClient(
            location=location,
            url=url,
            port=port,
            grpc_port=grpc_port,
            prefer_grpc=prefer_grpc,
            https=https,
            api_key=api_key,
            prefix=prefix,
            timeout=timeout,
            host=host,
            path=path,
            **kwargs,
        )

        if location == ":memory:" or path is not None:
                        # 本地 Qdrant 无法与同步和异步客户端共存
                        # 此情况下我们回退到同步操作
            async_client = None
        else:
            async_client = AsyncQdrantClient(
                location=location,
                url=url,
                port=port,
                grpc_port=grpc_port,
                prefer_grpc=prefer_grpc,
                https=https,
                api_key=api_key,
                prefix=prefix,
                timeout=timeout,
                host=host,
                path=path,
                **kwargs,
            )

        return sync_client, async_client

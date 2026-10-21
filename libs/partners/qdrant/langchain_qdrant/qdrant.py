from __future__ import annotations

import uuid
from collections.abc import Callable
from enum import Enum
from itertools import islice
from operator import itemgetter
from typing import (
    TYPE_CHECKING,
    Any,
)

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from qdrant_client import QdrantClient, models

if TYPE_CHECKING:
    from collections.abc import Generator, Iterable, Sequence

    from langchain_qdrant.sparse_embeddings import SparseEmbeddings


class QdrantVectorStoreError(Exception):
    """与 `QdrantVectorStore` 相关的异常。"""


class RetrievalMode(str, Enum):
    """从 Qdrant 中检索向量的模式。"""

    DENSE = "dense"
    SPARSE = "sparse"
    HYBRID = "hybrid"


class QdrantVectorStore(VectorStore):
    """Qdrant 向量存储集成。

    设置：
        安装 `langchain-qdrant` 包。

        ```bash
        pip install -qU langchain-qdrant
        ```

    关键初始化参数 —— 索引参数：
        collection_name:
            集合名称。
        embedding:
            要使用的嵌入函数。
        sparse_embedding:
            可选的稀疏嵌入函数。

    关键初始化参数 —— 客户端参数：
        client:
            要使用的 Qdrant 客户端。
        retrieval_mode:
            要使用的检索模式。

    实例化：
        ```python
        from langchain_qdrant import QdrantVectorStore
        from qdrant_client import QdrantClient
        from qdrant_client.http.models import Distance, VectorParams
        from langchain_openai import OpenAIEmbeddings

        client = QdrantClient(":memory:")

        client.create_collection(
            collection_name="demo_collection",
            vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
        )

        vector_store = QdrantVectorStore(
            client=client,
            collection_name="demo_collection",
            embedding=OpenAIEmbeddings(),
        )
        ```

    添加文档：
        ```python
        from langchain_core.documents import Document
        from uuid import uuid4

        document_1 = Document(page_content="foo", metadata={"baz": "bar"})
        document_2 = Document(page_content="thud", metadata={"bar": "baz"})
        document_3 = Document(page_content="i will be deleted :(")

        documents = [document_1, document_2, document_3]
        ids = [str(uuid4()) for _ in range(len(documents))]
        vector_store.add_documents(documents=documents, ids=ids)
        ```

    删除文档：
        ```python
        vector_store.delete(ids=[ids[-1]])
        ```

    搜索：
        ```python
        results = vector_store.similarity_search(
            query="thud",
            k=1,
        )
        for doc in results:
            print(f"* {doc.page_content} [{doc.metadata}]")
        ```

        ```python
        *thud[
            {
                "bar": "baz",
                "_id": "0d706099-6dd9-412a-9df6-a71043e020de",
                "_collection_name": "demo_collection",
            }
        ]
        ```

    带过滤条件的搜索：
        ```python
        from qdrant_client.http import models

        results = vector_store.similarity_search(
            query="thud",
            k=1,
            filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="metadata.bar",
                        match=models.MatchValue(value="baz"),
                    )
                ]
            ),
        )
        for doc in results:
            print(f"* {doc.page_content} [{doc.metadata}]")
        ```

        ```python
        *thud[
            {
                "bar": "baz",
                "_id": "0d706099-6dd9-412a-9df6-a71043e020de",
                "_collection_name": "demo_collection",
            }
        ]
        ```

    带分数的搜索：
        ```python
        results = vector_store.similarity_search_with_score(query="qux", k=1)
        for doc, score in results:
            print(f"* [SIM={score:3f}] {doc.page_content} [{doc.metadata}]")
        ```

        ```python
        * [SIM=0.832268] foo [{'baz': 'bar', '_id': '44ec7094-b061-45ac-8fbf-014b0f18e8aa', '_collection_name': 'demo_collection'}]
        ```

    异步操作：
        ```python
        # add documents
        # await vector_store.aadd_documents(documents=documents, ids=ids)

        # delete documents
        # await vector_store.adelete(ids=["3"])

        # search
        # results = vector_store.asimilarity_search(query="thud",k=1)

        # search with score
        results = await vector_store.asimilarity_search_with_score(query="qux", k=1)
        for doc, score in results:
            print(f"* [SIM={score:3f}] {doc.page_content} [{doc.metadata}]")
        ```

        ```python
        * [SIM=0.832268] foo [{'baz': 'bar', '_id': '44ec7094-b061-45ac-8fbf-014b0f18e8aa', '_collection_name': 'demo_collection'}]
        ```

    用作检索器：
        ```python
        retriever = vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 1, "fetch_k": 2, "lambda_mult": 0.5},
        )
        retriever.invoke("thud")
        ```

        ```python
        [
            Document(
                metadata={
                    "bar": "baz",
                    "_id": "0d706099-6dd9-412a-9df6-a71043e020de",
                    "_collection_name": "demo_collection",
                },
                page_content="thud",
            )
        ]
        ```
    """  # noqa: E501

    CONTENT_KEY: str = "page_content"
    METADATA_KEY: str = "metadata"
    VECTOR_NAME: str = ""  # 默认/未命名向量 - https://qdrant.tech/documentation/concepts/collections/#create-a-collection
    SPARSE_VECTOR_NAME: str = "langchain-sparse"

    def __init__(
        self,
        client: QdrantClient,
        collection_name: str,
        embedding: Embeddings | None = None,
        retrieval_mode: RetrievalMode = RetrievalMode.DENSE,
        vector_name: str = VECTOR_NAME,
        content_payload_key: str = CONTENT_KEY,
        metadata_payload_key: str = METADATA_KEY,
        distance: models.Distance = models.Distance.COSINE,
        sparse_embedding: SparseEmbeddings | None = None,
        sparse_vector_name: str = SPARSE_VECTOR_NAME,
        validate_embeddings: bool = True,  # noqa: FBT001, FBT002
        validate_collection_config: bool = True,  # noqa: FBT001, FBT002
    ) -> None:
        """初始化 `QdrantVectorStore` 的新实例。

        ```python
        qdrant = Qdrant(
            client=client,
            collection_name="my-collection",
            embedding=OpenAIEmbeddings(),
            retrieval_mode=RetrievalMode.HYBRID,
            sparse_embedding=FastEmbedSparse(),
        )
        ```
        """
        if validate_embeddings:
            self._validate_embeddings(retrieval_mode, embedding, sparse_embedding)

        if validate_collection_config:
            self._validate_collection_config(
                client,
                collection_name,
                retrieval_mode,
                vector_name,
                sparse_vector_name,
                distance,
                embedding,
            )

        self._client = client
        self.collection_name = collection_name
        self._embeddings = embedding
        self.retrieval_mode = retrieval_mode
        self.vector_name = vector_name
        self.content_payload_key = content_payload_key
        self.metadata_payload_key = metadata_payload_key
        self.distance = distance
        self._sparse_embeddings = sparse_embedding
        self.sparse_vector_name = sparse_vector_name

    @property
    def client(self) -> QdrantClient:
        """获取正在使用的 Qdrant 客户端实例。

        Returns:
            QdrantClient: `QdrantClient` 的一个实例。

        """
        return self._client

    @property
    def embeddings(self) -> Embeddings | None:
        """获取正在使用的密集嵌入实例。

        Returns:
            Embeddings: `Embeddings` 的一个实例，或在 SPARSE 模式下为 None。

        """
        return self._embeddings

    def _get_retriever_tags(self) -> list[str]:
        """获取检索器的标签。

        Override the base class method to handle SPARSE mode where embeddings can be
        None. In SPARSE mode, embeddings is None, so we don't include embeddings class
        name in tags. In DENSE/HYBRID modes, embeddings is not None, so we include
        embeddings class name.
        """
        tags = [self.__class__.__name__]

                # 处理不同的检索模式
        if self.retrieval_mode == RetrievalMode.SPARSE:
                        # 稀疏模式：无密集嵌入，因此标签中不包含嵌入类名
            pass
                # 密集/混合模式：如果可用，则包含嵌入类名
        elif self.embeddings is not None:
            tags.append(self.embeddings.__class__.__name__)

        return tags

    def _require_embeddings(self, operation: str) -> Embeddings:
        """为需要嵌入的操作要求提供嵌入。

        Args:
            operation: 需要嵌入的操作描述。

        Returns:
            嵌入实例。

        Raises:
            ValueError: 如果嵌入为 None 且操作需要嵌入。
        """
        if self.embeddings is None:
            msg = f"Embeddings are required for {operation}"
            raise ValueError(msg)
        return self.embeddings

    @property
    def sparse_embeddings(self) -> SparseEmbeddings:
        """获取正在使用的稀疏嵌入实例。

        Raises:
            ValueError: 如果稀疏嵌入为 `None`。

        Returns:
            SparseEmbeddings: `SparseEmbeddings` 的一个实例。

        """
        if self._sparse_embeddings is None:
            msg = (
                "Sparse embeddings are `None`. "
                "Please set using the `sparse_embedding` parameter."
            )
            raise ValueError(msg)
        return self._sparse_embeddings

    @classmethod
    def from_texts(
        cls: type[QdrantVectorStore],
        texts: list[str],
        embedding: Embeddings | None = None,
        metadatas: list[dict] | None = None,
        ids: Sequence[str | int] | None = None,
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
        path: str | None = None,
        distance: models.Distance = models.Distance.COSINE,
        content_payload_key: str = CONTENT_KEY,
        metadata_payload_key: str = METADATA_KEY,
        vector_name: str = VECTOR_NAME,
        retrieval_mode: RetrievalMode = RetrievalMode.DENSE,
        sparse_embedding: SparseEmbeddings | None = None,
        sparse_vector_name: str = SPARSE_VECTOR_NAME,
        collection_create_options: dict[str, Any] | None = None,
        vector_params: dict[str, Any] | None = None,
        sparse_vector_params: dict[str, Any] | None = None,
        batch_size: int = 64,
        force_recreate: bool = False,  # noqa: FBT001, FBT002
        validate_embeddings: bool = True,  # noqa: FBT001, FBT002
        validate_collection_config: bool = True,  # noqa: FBT001, FBT002
        **kwargs: Any,
    ) -> QdrantVectorStore:
        """从文本列表构建 `QdrantVectorStore` 的实例。

        这是一个用户友好的接口，其功能包括：

        1. 为每个文本创建嵌入
        2. 如果集合不存在，则创建一个 Qdrant 集合。
        3. 将文本嵌入添加到 Qdrant 数据库

        此方式旨在快速上手。

        ```python
        from langchain_qdrant import Qdrant
        from langchain_openai import OpenAIEmbeddings

        embeddings = OpenAIEmbeddings()
        qdrant = Qdrant.from_texts(texts, embeddings, url="http://localhost:6333")
        ```
        """
        if sparse_vector_params is None:
            sparse_vector_params = {}
        if vector_params is None:
            vector_params = {}
        if collection_create_options is None:
            collection_create_options = {}
        client_options = {
            "location": location,
            "url": url,
            "port": port,
            "grpc_port": grpc_port,
            "prefer_grpc": prefer_grpc,
            "https": https,
            "api_key": api_key,
            "prefix": prefix,
            "timeout": timeout,
            "host": host,
            "path": path,
            **kwargs,
        }

        qdrant = cls.construct_instance(
            embedding,
            retrieval_mode,
            sparse_embedding,
            client_options,
            collection_name,
            distance,
            content_payload_key,
            metadata_payload_key,
            vector_name,
            sparse_vector_name,
            force_recreate,
            collection_create_options,
            vector_params,
            sparse_vector_params,
            validate_embeddings,
            validate_collection_config,
        )
        qdrant.add_texts(texts, metadatas, ids, batch_size)
        return qdrant

    @classmethod
    def from_existing_collection(
        cls: type[QdrantVectorStore],
        collection_name: str,
        embedding: Embeddings | None = None,
        retrieval_mode: RetrievalMode = RetrievalMode.DENSE,
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
        distance: models.Distance = models.Distance.COSINE,
        content_payload_key: str = CONTENT_KEY,
        metadata_payload_key: str = METADATA_KEY,
        vector_name: str = VECTOR_NAME,
        sparse_vector_name: str = SPARSE_VECTOR_NAME,
        sparse_embedding: SparseEmbeddings | None = None,
        validate_embeddings: bool = True,  # noqa: FBT001, FBT002
        validate_collection_config: bool = True,  # noqa: FBT001, FBT002
        **kwargs: Any,
    ) -> QdrantVectorStore:
        """从现有集合构建 `QdrantVectorStore`，不添加数据。

        Returns:
            QdrantVectorStore: `QdrantVectorStore` 的新实例。
        """
        client = QdrantClient(
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
            collection_name=collection_name,
            embedding=embedding,
            retrieval_mode=retrieval_mode,
            content_payload_key=content_payload_key,
            metadata_payload_key=metadata_payload_key,
            distance=distance,
            vector_name=vector_name,
            sparse_embedding=sparse_embedding,
            sparse_vector_name=sparse_vector_name,
            validate_embeddings=validate_embeddings,
            validate_collection_config=validate_collection_config,
        )

    def add_texts(  # type: ignore[override]
        self,
        texts: Iterable[str],
        metadatas: list[dict] | None = None,
        ids: Sequence[str | int] | None = None,
        batch_size: int = 64,
        **kwargs: Any,
    ) -> list[str | int]:
        """将带有嵌入的文本添加到 `VectorStore`。

        Returns:
            添加文本到 `VectorStore` 的 ID 列表。

        """
        added_ids = []
        for batch_ids, points in self._generate_batches(
            texts, metadatas, ids, batch_size
        ):
            self.client.upsert(
                collection_name=self.collection_name, points=points, **kwargs
            )
            added_ids.extend(batch_ids)

        return added_ids

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        filter: models.Filter | None = None,  # noqa: A002
        search_params: models.SearchParams | None = None,
        offset: int = 0,
        score_threshold: float | None = None,
        consistency: models.ReadConsistency | None = None,
        hybrid_fusion: models.FusionQuery | None = None,
        **kwargs: Any,
    ) -> list[Document]:
        """返回与查询最相似的文档。

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
            hybrid_fusion=hybrid_fusion,
            **kwargs,
        )
        return list(map(itemgetter(0), results))

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        filter: models.Filter | None = None,  # noqa: A002
        search_params: models.SearchParams | None = None,
        offset: int = 0,
        score_threshold: float | None = None,
        consistency: models.ReadConsistency | None = None,
        hybrid_fusion: models.FusionQuery | None = None,
        **kwargs: Any,
    ) -> list[tuple[Document, float]]:
        """返回与查询最相似的文档。

        Returns:
            与查询文本最相似的文档列表及每个文档的距离。

        """
        query_options = {
            "collection_name": self.collection_name,
            "query_filter": filter,
            "search_params": search_params,
            "limit": k,
            "offset": offset,
            "with_payload": True,
            "with_vectors": False,
            "score_threshold": score_threshold,
            "consistency": consistency,
            **kwargs,
        }
        if self.retrieval_mode == RetrievalMode.DENSE:
            embeddings = self._require_embeddings("DENSE mode")
            query_dense_embedding = embeddings.embed_query(query)
            results = self.client.query_points(
                query=query_dense_embedding,
                using=self.vector_name,
                **query_options,
            ).points

        elif self.retrieval_mode == RetrievalMode.SPARSE:
            query_sparse_embedding = self.sparse_embeddings.embed_query(query)
            results = self.client.query_points(
                query=models.SparseVector(
                    indices=query_sparse_embedding.indices,
                    values=query_sparse_embedding.values,
                ),
                using=self.sparse_vector_name,
                **query_options,
            ).points

        elif self.retrieval_mode == RetrievalMode.HYBRID:
            embeddings = self._require_embeddings("HYBRID mode")
            query_dense_embedding = embeddings.embed_query(query)
            query_sparse_embedding = self.sparse_embeddings.embed_query(query)
            results = self.client.query_points(
                prefetch=[
                    models.Prefetch(
                        using=self.vector_name,
                        query=query_dense_embedding,
                        filter=filter,
                        limit=k,
                        params=search_params,
                    ),
                    models.Prefetch(
                        using=self.sparse_vector_name,
                        query=models.SparseVector(
                            indices=query_sparse_embedding.indices,
                            values=query_sparse_embedding.values,
                        ),
                        filter=filter,
                        limit=k,
                        params=search_params,
                    ),
                ],
                query=hybrid_fusion or models.FusionQuery(fusion=models.Fusion.RRF),
                **query_options,
            ).points

        else:
            msg = f"Invalid retrieval mode. {self.retrieval_mode}."
            raise ValueError(msg)
        return [
            (
                self._document_from_point(
                    result,
                    self.collection_name,
                    self.content_payload_key,
                    self.metadata_payload_key,
                ),
                result.score,
            )
            for result in results
        ]

    def similarity_search_with_score_by_vector(
        self,
        embedding: list[float],
        k: int = 4,
        filter: models.Filter | None = None,  # noqa: A002
        search_params: models.SearchParams | None = None,
        offset: int = 0,
        score_threshold: float | None = None,
        consistency: models.ReadConsistency | None = None,
        **kwargs: Any,
    ) -> list[tuple[Document, float]]:
        """返回与嵌入向量最相似的文档。

        Returns:
            与查询最相似的 `Document` 对象列表及每个文档的距离。

        """
        qdrant_filter = filter

        self._validate_collection_for_dense(
            client=self.client,
            collection_name=self.collection_name,
            vector_name=self.vector_name,
            distance=self.distance,
            dense_embeddings=embedding,
        )
        results = self.client.query_points(
            collection_name=self.collection_name,
            query=embedding,
            using=self.vector_name,
            query_filter=qdrant_filter,
            search_params=search_params,
            limit=k,
            offset=offset,
            with_payload=True,
            with_vectors=False,
            score_threshold=score_threshold,
            consistency=consistency,
            **kwargs,
        ).points

        return [
            (
                self._document_from_point(
                    result,
                    self.collection_name,
                    self.content_payload_key,
                    self.metadata_payload_key,
                ),
                result.score,
            )
            for result in results
        ]

    def similarity_search_by_vector(
        self,
        embedding: list[float],
        k: int = 4,
        filter: models.Filter | None = None,  # noqa: A002
        search_params: models.SearchParams | None = None,
        offset: int = 0,
        score_threshold: float | None = None,
        consistency: models.ReadConsistency | None = None,
        **kwargs: Any,
    ) -> list[Document]:
        """返回与嵌入向量最相似的文档。

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

    def max_marginal_relevance_search(
        self,
        query: str,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        filter: models.Filter | None = None,  # noqa: A002
        search_params: models.SearchParams | None = None,
        score_threshold: float | None = None,
        consistency: models.ReadConsistency | None = None,
        **kwargs: Any,
    ) -> list[Document]:
        """返回使用密集向量进行最大边际相关性选择的文档。

        Maximal marginal relevance optimizes for similarity to query AND diversity
        among selected documents.

        Returns:
            通过最大边际相关性选择的 `Document` 对象列表。

        """
        self._validate_collection_for_dense(
            self.client,
            self.collection_name,
            self.vector_name,
            self.distance,
            self.embeddings,
        )

        embeddings = self._require_embeddings("max_marginal_relevance_search")
        query_embedding = embeddings.embed_query(query)
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

    def max_marginal_relevance_search_by_vector(
        self,
        embedding: list[float],
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        filter: models.Filter | None = None,  # noqa: A002
        search_params: models.SearchParams | None = None,
        score_threshold: float | None = None,
        consistency: models.ReadConsistency | None = None,
        **kwargs: Any,
    ) -> list[Document]:
        """使用密集向量返回通过最大边际相关性选择的文档。

        Maximal marginal relevance optimizes for similarity to query AND diversity
        among selected documents.

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

    def max_marginal_relevance_search_with_score_by_vector(
        self,
        embedding: list[float],
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        filter: models.Filter | None = None,  # noqa: A002
        search_params: models.SearchParams | None = None,
        score_threshold: float | None = None,
        consistency: models.ReadConsistency | None = None,
        **kwargs: Any,
    ) -> list[tuple[Document, float]]:
        """使用最大边际相关性返回选择的文档。

        Maximal marginal relevance optimizes for similarity to query AND diversity
        among selected documents.

        Returns:
            List of `Document` objects selected by maximal marginal relevance and
                distance for each.
        """
        results = self.client.query_points(
            collection_name=self.collection_name,
            query=models.NearestQuery(
                nearest=embedding,
                mmr=models.Mmr(diversity=lambda_mult, candidates_limit=fetch_k),
            ),
            query_filter=filter,
            search_params=search_params,
            limit=k,
            with_payload=True,
            with_vectors=True,
            score_threshold=score_threshold,
            consistency=consistency,
            using=self.vector_name,
            **kwargs,
        ).points

        return [
            (
                self._document_from_point(
                    result,
                    self.collection_name,
                    self.content_payload_key,
                    self.metadata_payload_key,
                ),
                result.score,
            )
            for result in results
        ]

    def delete(  # type: ignore[override]
        self,
        ids: list[str | int] | None = None,
        **kwargs: Any,
    ) -> bool | None:
        """根据其 ID 删除文档。

        Args:
            ids: 要删除的 ID 列表。
            **kwargs: 子类可能使用的其他关键字参数。

        Returns:
            如果删除成功则返回 `True`，否则返回 `False`。

        """
        result = self.client.delete(
            collection_name=self.collection_name,
            points_selector=ids,
        )
        return result.status == models.UpdateStatus.COMPLETED

    def get_by_ids(self, ids: Sequence[str | int], /) -> list[Document]:
        results = self.client.retrieve(self.collection_name, ids, with_payload=True)

        return [
            self._document_from_point(
                result,
                self.collection_name,
                self.content_payload_key,
                self.metadata_payload_key,
            )
            for result in results
        ]

    @classmethod
    def construct_instance(
        cls: type[QdrantVectorStore],
        embedding: Embeddings | None = None,
        retrieval_mode: RetrievalMode = RetrievalMode.DENSE,
        sparse_embedding: SparseEmbeddings | None = None,
        client_options: dict[str, Any] | None = None,
        collection_name: str | None = None,
        distance: models.Distance = models.Distance.COSINE,
        content_payload_key: str = CONTENT_KEY,
        metadata_payload_key: str = METADATA_KEY,
        vector_name: str = VECTOR_NAME,
        sparse_vector_name: str = SPARSE_VECTOR_NAME,
        force_recreate: bool = False,  # noqa: FBT001, FBT002
        collection_create_options: dict[str, Any] | None = None,
        vector_params: dict[str, Any] | None = None,
        sparse_vector_params: dict[str, Any] | None = None,
        validate_embeddings: bool = True,  # noqa: FBT001, FBT002
        validate_collection_config: bool = True,  # noqa: FBT001, FBT002
    ) -> QdrantVectorStore:
        if sparse_vector_params is None:
            sparse_vector_params = {}
        if vector_params is None:
            vector_params = {}
        if collection_create_options is None:
            collection_create_options = {}
        if client_options is None:
            client_options = {}
        if validate_embeddings:
            cls._validate_embeddings(retrieval_mode, embedding, sparse_embedding)
        collection_name = collection_name or uuid.uuid4().hex
        client = QdrantClient(**client_options)

        collection_exists = client.collection_exists(collection_name)

        if collection_exists and force_recreate:
            client.delete_collection(collection_name)
            collection_exists = False
        if collection_exists:
            if validate_collection_config:
                cls._validate_collection_config(
                    client,
                    collection_name,
                    retrieval_mode,
                    vector_name,
                    sparse_vector_name,
                    distance,
                    embedding,
                )
        else:
            vectors_config, sparse_vectors_config = {}, {}
            if retrieval_mode == RetrievalMode.DENSE:
                partial_embeddings = embedding.embed_documents(["dummy_text"])  # type: ignore[union-attr]

                vector_params["size"] = len(partial_embeddings[0])
                vector_params["distance"] = distance

                vectors_config = {
                    vector_name: models.VectorParams(
                        **vector_params,
                    )
                }

            elif retrieval_mode == RetrievalMode.SPARSE:
                sparse_vectors_config = {
                    sparse_vector_name: models.SparseVectorParams(
                        **sparse_vector_params
                    )
                }

            elif retrieval_mode == RetrievalMode.HYBRID:
                partial_embeddings = embedding.embed_documents(["dummy_text"])  # type: ignore[union-attr]

                vector_params["size"] = len(partial_embeddings[0])
                vector_params["distance"] = distance

                vectors_config = {
                    vector_name: models.VectorParams(
                        **vector_params,
                    )
                }

                sparse_vectors_config = {
                    sparse_vector_name: models.SparseVectorParams(
                        **sparse_vector_params
                    )
                }

            collection_create_options["collection_name"] = collection_name
            collection_create_options["vectors_config"] = vectors_config
            collection_create_options["sparse_vectors_config"] = sparse_vectors_config

            client.create_collection(**collection_create_options)

        return cls(
            client=client,
            collection_name=collection_name,
            embedding=embedding,
            retrieval_mode=retrieval_mode,
            content_payload_key=content_payload_key,
            metadata_payload_key=metadata_payload_key,
            distance=distance,
            vector_name=vector_name,
            sparse_embedding=sparse_embedding,
            sparse_vector_name=sparse_vector_name,
            validate_embeddings=False,
            validate_collection_config=False,
        )

    @staticmethod
    def _cosine_relevance_score_fn(distance: float) -> float:
        """将距离归一化为 `[0, 1]` 区间内的得分。"""
        return (distance + 1.0) / 2.0

    def _select_relevance_score_fn(self) -> Callable[[float], float]:
        """您的“正确”相关性函数可能因以下因素而有所不同。

        包括：
        - 向量存储使用的距离/相似性度量
        - 嵌入的尺度（OpenAI 的嵌入是单位范数。许多其他嵌入不是！）
        - 嵌入维度
        - 等等。
        """
        if self.distance == models.Distance.COSINE:
            return self._cosine_relevance_score_fn
        if self.distance == models.Distance.DOT:
            return self._max_inner_product_relevance_score_fn
        if self.distance == models.Distance.EUCLID:
            return self._euclidean_relevance_score_fn
        msg = "Unknown distance strategy, must be COSINE, DOT, or EUCLID."
        raise ValueError(msg)

    @classmethod
    def _document_from_point(
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

    def _generate_batches(
        self,
        texts: Iterable[str],
        metadatas: list[dict] | None = None,
        ids: Sequence[str | int] | None = None,
        batch_size: int = 64,
    ) -> Generator[tuple[list[str | int], list[models.PointStruct]], Any, None]:
        texts_iterator = iter(texts)
        metadatas_iterator = iter(metadatas or [])
        ids_iterator = iter(ids or [uuid.uuid4().hex for _ in iter(texts)])

        while batch_texts := list(islice(texts_iterator, batch_size)):
            batch_metadatas = list(islice(metadatas_iterator, batch_size)) or None
            batch_ids = list(islice(ids_iterator, batch_size))
            points = [
                models.PointStruct(
                    id=point_id,
                    vector=vector,
                    payload=payload,
                )
                for point_id, vector, payload in zip(
                    batch_ids,
                    self._build_vectors(batch_texts),
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
    def _build_payloads(
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
                    "calling .from_texts or .add_texts."
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

    def _build_vectors(
        self,
        texts: Iterable[str],
    ) -> list[models.VectorStruct]:
        if self.retrieval_mode == RetrievalMode.DENSE:
            embeddings = self._require_embeddings("DENSE mode")
            batch_embeddings = embeddings.embed_documents(list(texts))
            return [
                {
                    self.vector_name: vector,
                }
                for vector in batch_embeddings
            ]

        if self.retrieval_mode == RetrievalMode.SPARSE:
            batch_sparse_embeddings = self.sparse_embeddings.embed_documents(
                list(texts)
            )
            return [
                {
                    self.sparse_vector_name: models.SparseVector(
                        values=vector.values, indices=vector.indices
                    )
                }
                for vector in batch_sparse_embeddings
            ]

        if self.retrieval_mode == RetrievalMode.HYBRID:
            embeddings = self._require_embeddings("HYBRID mode")
            dense_embeddings = embeddings.embed_documents(list(texts))
            sparse_embeddings = self.sparse_embeddings.embed_documents(list(texts))

            if len(dense_embeddings) != len(sparse_embeddings):
                msg = "Mismatched length between dense and sparse embeddings."
                raise ValueError(msg)

            return [
                {
                    self.vector_name: dense_vector,
                    self.sparse_vector_name: models.SparseVector(
                        values=sparse_vector.values, indices=sparse_vector.indices
                    ),
                }
                for dense_vector, sparse_vector in zip(
                    dense_embeddings, sparse_embeddings, strict=False
                )
            ]

        msg = f"Unknown retrieval mode. {self.retrieval_mode} to build vectors."
        raise ValueError(msg)

    @classmethod
    def _validate_collection_config(
        cls: type[QdrantVectorStore],
        client: QdrantClient,
        collection_name: str,
        retrieval_mode: RetrievalMode,
        vector_name: str,
        sparse_vector_name: str,
        distance: models.Distance,
        embedding: Embeddings | None,
    ) -> None:
        if retrieval_mode == RetrievalMode.DENSE:
            cls._validate_collection_for_dense(
                client, collection_name, vector_name, distance, embedding
            )

        elif retrieval_mode == RetrievalMode.SPARSE:
            cls._validate_collection_for_sparse(
                client, collection_name, sparse_vector_name
            )

        elif retrieval_mode == RetrievalMode.HYBRID:
            cls._validate_collection_for_dense(
                client, collection_name, vector_name, distance, embedding
            )
            cls._validate_collection_for_sparse(
                client, collection_name, sparse_vector_name
            )

    @classmethod
    def _validate_collection_for_dense(
        cls: type[QdrantVectorStore],
        client: QdrantClient,
        collection_name: str,
        vector_name: str,
        distance: models.Distance,
        dense_embeddings: Embeddings | list[float] | None,
    ) -> None:
        collection_info = client.get_collection(collection_name=collection_name)
        vector_config = collection_info.config.params.vectors

        if isinstance(vector_config, dict):
                        # vector_config 是一个 Dict[str, VectorParams]
            if vector_name not in vector_config:
                msg = (
                    f"Existing Qdrant collection {collection_name} does not "
                    f"contain dense vector named {vector_name}. "
                    "Did you mean one of the "
                    f"existing vectors: {', '.join(vector_config.keys())}? "  # type: ignore[union-attr]
                    f"If you want to recreate the collection, set `force_recreate` "
                    f"parameter to `True`."
                )
                raise QdrantVectorStoreError(msg)

                        # 获取指定 vector_name 的 VectorParams 对象
            vector_config = vector_config[vector_name]  # type: ignore[assignment, index]

                # vector_config 是 VectorParams 的实例
                # 单/未命名向量集合的情况。
        elif vector_name != "":
            msg = (
                f"Existing Qdrant collection {collection_name} is built "
                "with unnamed dense vector. "
                f"If you want to reuse it, set `vector_name` to ''(empty string)."
                f"If you want to recreate the collection, "
                "set `force_recreate` to `True`."
            )
            raise QdrantVectorStoreError(msg)

        if vector_config is None:
            msg = "VectorParams is None"
            raise ValueError(msg)

        if isinstance(dense_embeddings, Embeddings):
            vector_size = len(dense_embeddings.embed_documents(["dummy_text"])[0])
        elif isinstance(dense_embeddings, list):
            vector_size = len(dense_embeddings)
        else:
            msg = "Invalid `embeddings` type."
            raise TypeError(msg)

        if vector_config.size != vector_size:
            msg = (
                f"Existing Qdrant collection is configured for dense vectors with "
                f"{vector_config.size} dimensions. "
                f"Selected embeddings are {vector_size}-dimensional. "
                f"If you want to recreate the collection, set `force_recreate` "
                f"parameter to `True`."
            )
            raise QdrantVectorStoreError(msg)

        if vector_config.distance != distance:
            msg = (
                f"Existing Qdrant collection is configured for "
                f"{vector_config.distance.name} similarity, but requested "
                f"{distance.upper()}. Please set `distance` parameter to "
                f"`{vector_config.distance.name}` if you want to reuse it. "
                f"If you want to recreate the collection, set `force_recreate` "
                f"parameter to `True`."
            )
            raise QdrantVectorStoreError(msg)

    @classmethod
    def _validate_collection_for_sparse(
        cls: type[QdrantVectorStore],
        client: QdrantClient,
        collection_name: str,
        sparse_vector_name: str,
    ) -> None:
        collection_info = client.get_collection(collection_name=collection_name)
        sparse_vector_config = collection_info.config.params.sparse_vectors

        if (
            sparse_vector_config is None
            or sparse_vector_name not in sparse_vector_config
        ):
            msg = (
                f"Existing Qdrant collection {collection_name} does not "
                f"contain sparse vectors named {sparse_vector_name}. "
                f"If you want to recreate the collection, set `force_recreate` "
                f"parameter to `True`."
            )
            raise QdrantVectorStoreError(msg)

    @classmethod
    def _validate_embeddings(
        cls: type[QdrantVectorStore],
        retrieval_mode: RetrievalMode,
        embedding: Embeddings | None,
        sparse_embedding: SparseEmbeddings | None,
    ) -> None:
        if retrieval_mode == RetrievalMode.DENSE and embedding is None:
            msg = "'embedding' cannot be None when retrieval mode is 'dense'"
            raise ValueError(msg)

        if retrieval_mode == RetrievalMode.SPARSE and sparse_embedding is None:
            msg = "'sparse_embedding' cannot be None when retrieval mode is 'sparse'"
            raise ValueError(msg)

        if retrieval_mode == RetrievalMode.HYBRID and any(
            [embedding is None, sparse_embedding is None]
        ):
            msg = (
                "Both 'embedding' and 'sparse_embedding' cannot be None "
                "when retrieval mode is 'hybrid'"
            )
            raise ValueError(msg)

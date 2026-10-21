"""这是 langchain_chroma.vectorstores 模块。

它包含 Chroma 类，该类是用于处理各种任务的向量存储。
"""

from __future__ import annotations

import base64
import logging
import uuid
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
)

import chromadb
import chromadb.config
import numpy as np
from chromadb import Search, Settings
from chromadb.api import CreateCollectionConfiguration
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.utils import xor_args
from langchain_core.vectorstores import VectorStore

if TYPE_CHECKING:
    from chromadb.api.types import Where, WhereDocument

logger = logging.getLogger()
DEFAULT_K = 4  # 要返回的文档数量。


def _results_to_docs(results: Any) -> list[Document]:
    return [doc for doc, _ in _results_to_docs_and_scores(results)]


def _results_to_docs_and_scores(results: Any) -> list[tuple[Document, float]]:
    return [
                # TODO: Chroma 支持批处理查询，
                # 我们不应硬编码只取第一个结果
        (
            Document(page_content=result[0], metadata=result[1] or {}, id=result[2]),
            result[3],
        )
        for result in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["ids"][0],
            results["distances"][0],
            strict=False,
        )
        if result[0] is not None
    ]


def _results_to_docs_and_vectors(results: Any) -> list[tuple[Document, np.ndarray]]:
    """将 ChromaDB 结果转换为文档和向量，并过滤掉内容为 None 的项。"""
    return [
        (
            Document(page_content=result[0], metadata=result[1] or {}, id=result[3]),
            result[2],
        )
        for result in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["embeddings"][0],
            results["ids"][0],
            strict=False,
        )
        if result[0] is not None
    ]


Matrix = list[list[float]], list[np.ndarray] | np.ndarray


def cosine_similarity(X: Matrix, Y: Matrix) -> np.ndarray:  # type: ignore[valid-type]
    """两个等宽矩阵之间的行级余弦相似度。

    Raises:
        ValueError: 如果 X 和 Y 的列数不相同。
    """
    if len(X) == 0 or len(Y) == 0:
        return np.array([])

    X = np.array(X)
    Y = np.array(Y)
    if X.shape[1] != Y.shape[1]:
        msg = (
            "Number of columns in X and Y must be the same. X has shape"
            f"{X.shape} "
            f"and Y has shape {Y.shape}."
        )
        raise ValueError(
            msg,
        )

    X_norm = np.linalg.norm(X, axis=1)
    Y_norm = np.linalg.norm(Y, axis=1)
        # 忽略除零错误的运行时警告，因为这些情况已在下方处理。
    with np.errstate(divide="ignore", invalid="ignore"):
        similarity = np.dot(X, Y.T) / np.outer(X_norm, Y_norm)
    similarity[np.isnan(similarity) | np.isinf(similarity)] = 0.0
    return similarity


def maximal_marginal_relevance(
    query_embedding: np.ndarray,
    embedding_list: list,
    lambda_mult: float = 0.5,
    k: int = 4,
) -> list[int]:
    """计算最大边际相关性。

    Args:
        query_embedding: Query embedding.
        embedding_list: List of embeddings to select from.
        lambda_mult: Number between `0` and `1` that determines the degree
            of diversity among the results with `0` corresponding
            to maximum diversity and `1` to minimum diversity.
        k: Number of Documents to return.

    Returns:
        通过最大边际相关性选出的嵌入索引列表。
    """
    if min(k, len(embedding_list)) <= 0:
        return []
    if query_embedding.ndim == 1:
        query_embedding = np.expand_dims(query_embedding, axis=0)
    similarity_to_query = cosine_similarity(query_embedding, embedding_list)[0]
    most_similar = int(np.argmax(similarity_to_query))
    idxs = [most_similar]
    selected = np.array([embedding_list[most_similar]])
    while len(idxs) < min(k, len(embedding_list)):
        best_score = -np.inf
        idx_to_add = -1
        similarity_to_selected = cosine_similarity(embedding_list, selected)
        for i, query_score in enumerate(similarity_to_query):
            if i in idxs:
                continue
            redundant_score = max(similarity_to_selected[i])
            equation_score = (
                lambda_mult * query_score - (1 - lambda_mult) * redundant_score
            )
            if equation_score > best_score:
                best_score = equation_score
                idx_to_add = i
        idxs.append(idx_to_add)
        selected = np.append(selected, [embedding_list[idx_to_add]], axis=0)
    return idxs


class Chroma(VectorStore):
    """Chroma 向量存储集成。

    设置：
        安装 `chromadb`、`langchain-chroma` 包：

        ```bash
        pip install -qU chromadb langchain-chroma
        ```

    关键初始化参数 —— 索引参数：
        collection_name:
            集合名称。
        embedding_function:
            要使用的嵌入函数。

    关键初始化参数 —— 客户端参数：
        client:
            要使用的 Chroma 客户端。
        client_settings:
            Chroma 客户端设置。
        persist_directory:
            用于持久化集合的目录。
        host:
            部署的 Chroma 服务器的主机名。
        port:
            与部署的 Chroma 服务器连接的端口。默认为 8000。
        ssl:
            是否与部署的 Chroma 服务器建立 SSL 连接。默认为 False。
        headers:
            发送到部署的 Chroma 服务器的 HTTP 头。
        chroma_cloud_api_key:
            Chroma Cloud API 密钥。
        tenant:
            租户 ID。连接 Chroma Cloud 时必需。本地 Chroma 服务器的默认值为 'default_tenant'。
        database:
            数据库名称。连接 Chroma Cloud 时必需。本地 Chroma 服务器的默认值为 'default_database'。

    实例化：
        ```python
        from langchain_chroma import Chroma
        from langchain_openai import OpenAIEmbeddings

        vector_store = Chroma(
            collection_name="foo",
            embedding_function=OpenAIEmbeddings(),
            # other params...
        )
        ```

    添加文档：
        ```python
        from langchain_core.documents import Document

        document_1 = Document(page_content="foo", metadata={"baz": "bar"})
        document_2 = Document(page_content="thud", metadata={"bar": "baz"})
        document_3 = Document(page_content="i will be deleted :(")

        documents = [document_1, document_2, document_3]
        ids = ["1", "2", "3"]
        vector_store.add_documents(documents=documents, ids=ids)
        ```

    更新文档：
        ```python
        updated_document = Document(
            page_content="qux",
            metadata={"bar": "baz"},
        )

        vector_store.update_documents(ids=["1"], documents=[updated_document])
        ```

    删除文档：
        ```python
        vector_store.delete(ids=["3"])
        ```

    搜索：
        ```python
        results = vector_store.similarity_search(query="thud", k=1)
        for doc in results:
            print(f"* {doc.page_content} [{doc.metadata}]")
        ```
        ```python
        *thud[{"baz": "bar"}]
        ```

    带过滤条件的搜索：
        ```python
        results = vector_store.similarity_search(
            query="thud", k=1, filter={"baz": "bar"}
        )
        for doc in results:
            print(f"* {doc.page_content} [{doc.metadata}]")
        ```
        ```python
        *foo[{"baz": "bar"}]
        ```

    带分数的搜索：
        ```python
        results = vector_store.similarity_search_with_score(query="qux", k=1)
        for doc, score in results:
            print(f"* [SIM={score:3f}] {doc.page_content} [{doc.metadata}]")
        ```
        ```python
        * [SIM=0.000000] qux [{'bar': 'baz', 'baz': 'bar'}]
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
        * [SIM=0.335463] foo [{'baz': 'bar'}]
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
        [Document(metadata={"baz": "bar"}, page_content="thud")]
        ```
    """  # noqa: E501

    _LANGCHAIN_DEFAULT_COLLECTION_NAME = "langchain"

    def __init__(
        self,
        collection_name: str = _LANGCHAIN_DEFAULT_COLLECTION_NAME,
        embedding_function: Embeddings | None = None,
        persist_directory: str | None = None,
        host: str | None = None,
        port: int | None = None,
        headers: dict[str, str] | None = None,
        chroma_cloud_api_key: str | None = None,
        tenant: str | None = None,
        database: str | None = None,
        client_settings: chromadb.config.Settings | None = None,
        collection_metadata: dict | None = None,
        collection_configuration: CreateCollectionConfiguration | None = None,
        client: chromadb.ClientAPI | None = None,
        relevance_score_fn: Callable[[float], float] | None = None,
        create_collection_if_not_exists: bool | None = True,  # noqa: FBT001, FBT002
        *,
        ssl: bool = False,
    ) -> None:
        """使用 Chroma 客户端进行初始化。

        Args:
            collection_name: Name of the collection to create.
            embedding_function: Embedding class object. Used to embed texts.
            persist_directory: Directory to persist the collection.
            host: Hostname of a deployed Chroma server.
            port: Connection port for a deployed Chroma server. Default is 8000.
            ssl: Whether to establish an SSL connection with a deployed Chroma server.
                    Default is False.
            headers: HTTP headers to send to a deployed Chroma server.
            chroma_cloud_api_key: Chroma Cloud API key.
            tenant: Tenant ID. Required for Chroma Cloud connections.
                    Default is 'default_tenant' for local Chroma servers.
            database: Database name. Required for Chroma Cloud connections.
                    Default is 'default_database'.
            client_settings: Chroma client settings
            collection_metadata: Collection configurations.
            collection_configuration: Index configuration for the collection.

            client: Chroma 客户端。文档：
                    https://docs.trychroma.com/reference/python/client
            relevance_score_fn: 从距离计算相关性分数的函数。
                    仅在 `similarity_search_with_relevance_scores` 中使用
            create_collection_if_not_exists: 如果集合不存在，是否创建集合。
                    默认为 `True`。
        """
        _tenant = tenant or chromadb.DEFAULT_TENANT
        _database = database or chromadb.DEFAULT_DATABASE
        _settings = client_settings or Settings()

        client_args = {
            "persist_directory": persist_directory,
            "host": host,
            "chroma_cloud_api_key": chroma_cloud_api_key,
        }

        if sum(arg is not None for arg in client_args.values()) > 1:
            provided = [
                name for name, value in client_args.items() if value is not None
            ]
            msg = (
                f"Only one of 'persist_directory', 'host' and 'chroma_cloud_api_key' "
                f"is allowed, but got {','.join(provided)}"
            )
            raise ValueError(msg)

        if client is not None:
            self._client = client

                # PersistentClient
        elif persist_directory is not None:
            self._client = chromadb.PersistentClient(
                path=persist_directory,
                settings=_settings,
                tenant=_tenant,
                database=_database,
            )

                # HttpClient
        elif host is not None:
            _port = port or 8000
            self._client = chromadb.HttpClient(
                host=host,
                port=_port,
                ssl=ssl,
                headers=headers,
                settings=_settings,
                tenant=_tenant,
                database=_database,
            )

                # CloudClient
        elif chroma_cloud_api_key is not None:
            if not tenant or not database:
                msg = (
                    "Must provide tenant and database values to connect to Chroma Cloud"
                )
                raise ValueError(msg)
            self._client = chromadb.CloudClient(
                tenant=tenant,
                database=database,
                api_key=chroma_cloud_api_key,
                settings=_settings,
            )

        else:
            self._client = chromadb.Client(settings=_settings)

        self._embedding_function = embedding_function
        self._chroma_collection: chromadb.Collection | None = None
        self._collection_name = collection_name
        self._collection_metadata = collection_metadata
        self._collection_configuration = collection_configuration
        if create_collection_if_not_exists:
            self.__ensure_collection()
        else:
            self._chroma_collection = self._client.get_collection(name=collection_name)
        self.override_relevance_score_fn = relevance_score_fn

    def __ensure_collection(self) -> None:
        """确保集合存在，或创建它。"""
        self._chroma_collection = self._client.get_or_create_collection(
            name=self._collection_name,
            embedding_function=None,
            metadata=self._collection_metadata,
            configuration=self._collection_configuration,
        )

    @property
    def _collection(self) -> chromadb.Collection:
        """返回底层的 Chroma 集合，或抛出异常。"""
        if self._chroma_collection is None:
            msg = (
                "Chroma collection not initialized. "
                "Use `reset_collection` to re-create and initialize the collection. "
            )
            raise ValueError(
                msg,
            )
        return self._chroma_collection

    @property
    def embeddings(self) -> Embeddings | None:
        """访问查询嵌入对象。"""
        return self._embedding_function

    @xor_args(("query_texts", "query_embeddings"))
    def __query_collection(
        self,
        query_texts: list[str] | None = None,
        query_embeddings: list[list[float]] | None = None,
        n_results: int = 4,
        where: dict[str, str] | None = None,
        where_document: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> list[Document] | chromadb.QueryResult:
        """查询 chroma 集合。

        Args:
            query_texts: 查询文本列表。
            query_embeddings: 查询嵌入列表。
            n_results: 返回的结果数量。
            where: 用于按元数据过滤结果的字典。
                    例如 {"color" : "red"}。
            where_document: 用于按文档内容过滤的字典。
                    例如 {"$contains": "hello"}。
            kwargs: 传递给 Chroma 集合查询的额外关键字参数。

        Returns:
            List of `n_results` nearest neighbor embeddings for provided
            query_embeddings or query_texts.

        See more: 详见：https://docs.trychroma.com/reference/py-collection#query
        """
        return self._collection.query(
            query_texts=query_texts,
            query_embeddings=query_embeddings,  # type: ignore[arg-type]
            n_results=n_results,
            where=where,  # type: ignore[arg-type]
            where_document=where_document,  # type: ignore[arg-type]
            **kwargs,
        )

    @staticmethod
    def encode_image(uri: str) -> str:
        """从图像 URI 获取 base64 字符串。"""
        with Path(uri).open("rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def fork(self, new_name: str) -> Chroma:
        """分叉此向量存储。

        Args:
            new_name: 分叉存储的新名称。

        Returns:
            从此向量存储分叉出的新 Chroma 存储。

        """
        forked_collection = self._collection.fork(new_name=new_name)
        return Chroma(
            client=self._client,
            embedding_function=self._embedding_function,
            collection_name=forked_collection.name,
        )

    def add_images(
        self,
        uris: list[str],
        metadatas: list[dict] | None = None,
        ids: list[str] | None = None,
    ) -> list[str]:
        """将更多图像通过嵌入并添加到 `VectorStore`。

        Args:
            uris: 图像的文件路径。
            metadatas: 可选的元数据列表。
                    查询时，可基于此元数据进行过滤。
            ids: 可选的 ID 列表。（没有 ID 的项将分配 UUID）

        Returns:
            添加的图像的 ID 列表。

        Raises:
            ValueError: 当元数据不正确时。
        """
                # 从 URI 映射到 base64 编码字符串
        b64_texts = [self.encode_image(uri=uri) for uri in uris]
                # 填充 ID
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in uris]
        else:
            ids = [id_ if id_ is not None else str(uuid.uuid4()) for id_ in ids]
        embeddings = None
                # 设置嵌入
        if self._embedding_function is not None and hasattr(
            self._embedding_function,
            "embed_image",
        ):
            embeddings = self._embedding_function.embed_image(uris=uris)
        if metadatas:
                        # 如果有人未为所有图像指定元数据，则用空字典填充元数据
                        # 如果有人未为所有图像指定元数据，则用空字典填充元数据
            length_diff = len(uris) - len(metadatas)
            if length_diff:
                metadatas = metadatas + [{}] * length_diff
            empty_ids = []
            non_empty_ids = []
            for idx, m in enumerate(metadatas):
                if m:
                    non_empty_ids.append(idx)
                else:
                    empty_ids.append(idx)
            if non_empty_ids:
                metadatas = [metadatas[idx] for idx in non_empty_ids]
                images_with_metadatas = [b64_texts[idx] for idx in non_empty_ids]
                embeddings_with_metadatas = (
                    [embeddings[idx] for idx in non_empty_ids] if embeddings else None
                )
                ids_with_metadata = [ids[idx] for idx in non_empty_ids]
                try:
                    self._collection.upsert(
                        metadatas=metadatas,  # type: ignore[arg-type]
                        embeddings=embeddings_with_metadatas,  # type: ignore[arg-type]
                        documents=images_with_metadatas,
                        ids=ids_with_metadata,
                    )
                except ValueError as e:
                    if "Expected metadata value to be" in str(e):
                        msg = (
                            "Try filtering complex metadata using "
                            "langchain_community.vectorstores.utils.filter_complex_metadata."
                        )
                        raise ValueError(e.args[0] + "\n\n" + msg) from e
                    raise e
            if empty_ids:
                images_without_metadatas = [b64_texts[j] for j in empty_ids]
                embeddings_without_metadatas = (
                    [embeddings[j] for j in empty_ids] if embeddings else None
                )
                ids_without_metadatas = [ids[j] for j in empty_ids]
                self._collection.upsert(
                    embeddings=embeddings_without_metadatas,
                    documents=images_without_metadatas,
                    ids=ids_without_metadatas,
                )
        else:
            self._collection.upsert(
                embeddings=embeddings,
                documents=b64_texts,
                ids=ids,
            )
        return ids

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: list[dict] | None = None,
        ids: list[str] | None = None,
        **kwargs: Any,
    ) -> list[str]:
        """将更多文本通过嵌入并添加到 `VectorStore`。

        Args:
            texts: 要添加到 `VectorStore` 的文本。
            metadatas: 可选的元数据列表。
                    查询时，可根据此元数据进行过滤。
            ids: 可选的 ID 列表。（没有 ID 的项将分配 UUID）
            kwargs: 其他关键字参数。

        Returns:
            添加文本的 ID 列表。

        Raises:
            ValueError: 当元数据不正确时。
        """
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in texts]
        else:
            ids = [id_ if id_ is not None else str(uuid.uuid4()) for id_ in ids]

        embeddings = None
        texts = list(texts)
        if self._embedding_function is not None:
            embeddings = self._embedding_function.embed_documents(texts)
        if metadatas:
                        # 如果有人未为所有文本指定元数据，则用空字典填充元数据
                        # 如果有人未为所有文本指定元数据，则用空字典填充元数据
            length_diff = len(texts) - len(metadatas)
            if length_diff:
                metadatas = metadatas + [{}] * length_diff
            empty_ids = []
            non_empty_ids = []
            for idx, m in enumerate(metadatas):
                if m:
                    non_empty_ids.append(idx)
                else:
                    empty_ids.append(idx)
            if non_empty_ids:
                metadatas = [metadatas[idx] for idx in non_empty_ids]
                texts_with_metadatas = [texts[idx] for idx in non_empty_ids]
                embeddings_with_metadatas = (
                    [embeddings[idx] for idx in non_empty_ids]
                    if embeddings is not None and len(embeddings) > 0
                    else None
                )
                ids_with_metadata = [ids[idx] for idx in non_empty_ids]
                try:
                    self._collection.upsert(
                        metadatas=metadatas,  # type: ignore[arg-type]
                        embeddings=embeddings_with_metadatas,  # type: ignore[arg-type]
                        documents=texts_with_metadatas,
                        ids=ids_with_metadata,
                    )
                except ValueError as e:
                    if "Expected metadata value to be" in str(e):
                        msg = (
                            "Try filtering complex metadata from the document using "
                            "langchain_community.vectorstores.utils.filter_complex_metadata."
                        )
                        raise ValueError(e.args[0] + "\n\n" + msg) from e
                    raise e
            if empty_ids:
                texts_without_metadatas = [texts[j] for j in empty_ids]
                embeddings_without_metadatas = (
                    [embeddings[j] for j in empty_ids] if embeddings else None
                )
                ids_without_metadatas = [ids[j] for j in empty_ids]
                self._collection.upsert(
                    embeddings=embeddings_without_metadatas,  # type: ignore[arg-type]
                    documents=texts_without_metadatas,
                    ids=ids_without_metadatas,
                )
        else:
            self._collection.upsert(
                embeddings=embeddings,  # type: ignore[arg-type]
                documents=texts,
                ids=ids,
            )
        return ids

    def hybrid_search(self, search: Search) -> list[Document]:
        """使用 Chroma 执行混合搜索。

        Args:
            search: 混合搜索的搜索配置。

        Returns:
            搜索操作的结果文档列表。

        Example:
            from chromadb import Search, K, Knn, Rrf

            # 使用文本查询创建 RRF 排名
            hybrid_rank = Rrf(
                ranks=[
                    Knn(query="query", return_rank=True, limit=300),
                    Knn(query="query learning applications", key="sparse_embedding")
                ],
                weights=[2.0, 1.0],  # 密集向量权重是稀疏向量的 2 倍
                k=60
            )

            # 构建完整的搜索策略
            search = (Search()
                .where(
                    (K("language") == "en") &
                    (K("year") >= 2020)
                )
                .rank(hybrid_rank)
                .limit(10)
                .select(K.DOCUMENT, K.SCORE, "title", "year")
            )

            results = vector_store.hybrid_search(search)
        """
        results = self._collection.search(search)
        return [
            Document(
                page_content=record["document"],
                metadata=record["metadata"],
                id=record["id"],
            )
            for record in results.rows()[0]
            if record["document"] is not None
        ]

    def similarity_search(
        self,
        query: str,
        k: int = DEFAULT_K,
        filter: dict[str, str] | None = None,  # noqa: A002
        **kwargs: Any,
    ) -> list[Document]:
        """使用 Chroma 执行相似度搜索。

        Args:
            query: 用于搜索的查询文本。
            k: 返回结果的数量。
            filter: 根据元数据进行过滤。
            kwargs: 传递给 Chroma 集合查询的其他关键字参数。

        Returns:
            与查询文本最相似的文档列表。
        """
        docs_and_scores = self.similarity_search_with_score(
            query,
            k,
            filter=filter,
            **kwargs,
        )
        return [doc for doc, _ in docs_and_scores]

    def similarity_search_by_vector(
        self,
        embedding: list[float],
        k: int = DEFAULT_K,
        filter: dict[str, str] | None = None,  # noqa: A002
        where_document: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> list[Document]:
        """返回与嵌入向量最相似的文档。

        Args:
            embedding: 用于查找相似文档的嵌入向量。
            k: 返回的文档数量。
            filter: 根据元数据进行过滤。
            where_document: 用于根据文档内容进行过滤的字典。
                    例如：{"$contains": "hello"}。
            kwargs: 传递给 Chroma 集合查询的其他关键字参数。

        Returns:
            与查询向量最相似的 `Document` 对象列表。
        """
        results = self.__query_collection(
            query_embeddings=[embedding],
            n_results=k,
            where=filter,
            where_document=where_document,
            **kwargs,
        )
        return _results_to_docs(results)

    def similarity_search_by_vector_with_relevance_scores(
        self,
        embedding: list[float],
        k: int = DEFAULT_K,
        filter: dict[str, str] | None = None,  # noqa: A002
        where_document: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> list[tuple[Document, float]]:
        """返回与嵌入向量最相似的文档及其相似度得分。

        Args:
            embedding (List[float]): 用于查找相似文档的嵌入向量。
            k: 返回的文档数量。
            filter: 根据元数据进行过滤。
            where_document: 用于根据文档内容进行过滤的字典。
                    例如：{"$contains": "hello"}。
            kwargs: 传递给 Chroma 集合查询的其他关键字参数。

        Returns:
            与查询文本最相似的文档列表及每个文档的相关性得分
            （浮点数），得分越低表示越相似。
        """
        results = self.__query_collection(
            query_embeddings=[embedding],
            n_results=k,
            where=filter,
            where_document=where_document,
            **kwargs,
        )
        return _results_to_docs_and_scores(results)

    def similarity_search_with_score(
        self,
        query: str,
        k: int = DEFAULT_K,
        filter: dict[str, str] | None = None,  # noqa: A002
        where_document: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> list[tuple[Document, float]]:
        """使用距离信息执行与 Chroma 的相似度搜索。

        Args:
            query: 要搜索的查询文本。
            k: 返回的结果数量。
            filter: 按元数据过滤。
            where_document: 用于按文档内容过滤的字典。
                    例如 {"$contains": "hello"}。
            kwargs: 传递给 Chroma 集合查询的额外关键字参数。

        Returns:
            与查询文本最相似的文档列表以及每个结果的距离（浮点数）。
            距离越低表示越相似。
        """
        if self._embedding_function is None:
            results = self.__query_collection(
                query_texts=[query],
                n_results=k,
                where=filter,
                where_document=where_document,
                **kwargs,
            )
        else:
            query_embedding = self._embedding_function.embed_query(query)
            results = self.__query_collection(
                query_embeddings=[query_embedding],
                n_results=k,
                where=filter,
                where_document=where_document,
                **kwargs,
            )

        return _results_to_docs_and_scores(results)

    def similarity_search_with_vectors(
        self,
        query: str,
        k: int = DEFAULT_K,
        filter: dict[str, str] | None = None,  # noqa: A002
        where_document: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> list[tuple[Document, np.ndarray]]:
        """使用向量在 Chroma 中执行相似性搜索。

        Args:
            query: 要搜索的查询文本。
            k: 返回的结果数量。
            filter: 按元数据过滤。
            where_document: 用于按文档内容过滤的字典。
                    例如 {"$contains": "hello"}。
            kwargs: 传递给 Chroma 集合查询的额外关键字参数。

        Returns:
            List of documents most similar to the query text and
            embedding vectors for each.
        """
        include = ["documents", "metadatas", "embeddings"]
        if self._embedding_function is None:
            results = self.__query_collection(
                query_texts=[query],
                n_results=k,
                where=filter,
                where_document=where_document,
                include=include,
                **kwargs,
            )
        else:
            query_embedding = self._embedding_function.embed_query(query)
            results = self.__query_collection(
                query_embeddings=[query_embedding],
                n_results=k,
                where=filter,
                where_document=where_document,
                include=include,
                **kwargs,
            )

        return _results_to_docs_and_vectors(results)

    def _select_relevance_score_fn(self) -> Callable[[float], float]:
        """根据集合的距离度量选择相关性评分函数。

        最相似的文档将具有最低的相关性评分。默认的相关性评分函数是欧几里得距离。
        距离度量必须在初始化 Chroma 对象时通过 `collection_configuration` 提供。
        Example: collection_configuration={"hnsw": {"space": "cosine"}}.
        Available distance metrics are: 支持的距离度量包括：'cosine'、'l2' 和 'ip'。

        Returns:
            相关性评分函数。

        Raises:
            ValueError: 如果距离度量不被支持。
        """
        if self.override_relevance_score_fn:
            return self.override_relevance_score_fn

        hnsw_config = self._collection.configuration.get("hnsw")
        hnsw_distance: str | None = hnsw_config.get("space") if hnsw_config else None

        spann_config = self._collection.configuration.get("spann")
        spann_distance: str | None = spann_config.get("space") if spann_config else None

        distance = hnsw_distance or spann_distance

        if distance == "cosine":
            return self._cosine_relevance_score_fn
        if distance == "l2":
            return self._euclidean_relevance_score_fn
        if distance == "ip":
            return self._max_inner_product_relevance_score_fn
        msg = (
            "No supported normalization function"
            f" for distance metric of type: {distance}."
            "Consider providing relevance_score_fn to Chroma constructor."
        )
        raise ValueError(
            msg,
        )

    def similarity_search_by_image(
        self,
        uri: str,
        k: int = DEFAULT_K,
        filter: dict[str, str] | None = None,  # noqa: A002
        **kwargs: Any,
    ) -> list[Document]:
        """基于给定图像 URI 搜索相似图像。

        Args:
            uri: 要搜索的图像的 URI。
            k: 返回的结果数量。
            filter: 按元数据过滤。
            **kwargs: 传递给函数的额外参数。


        Returns:
            与所提供图像最相似的图像列表。列表中的每个元素都是一个 LangChain 文档对象。
            页面内容为 base64 编码的图像，元数据为默认值或用户定义的值。

        Raises:
            ValueError: 如果嵌入函数不支持图像嵌入。
        """
        if self._embedding_function is not None and hasattr(
            self._embedding_function, "embed_image"
        ):
                        # 获取图像嵌入
                        # 假设 embed_image 返回单个嵌入
            image_embedding = self._embedding_function.embed_image(uris=[uri])[0]

                        # 基于获取的嵌入执行相似性搜索
            return self.similarity_search_by_vector(
                embedding=image_embedding,
                k=k,
                filter=filter,
                **kwargs,
            )
        msg = "The embedding function must support image embedding."
        raise ValueError(msg)

    def similarity_search_by_image_with_relevance_score(
        self,
        uri: str,
        k: int = DEFAULT_K,
        filter: dict[str, str] | None = None,  # noqa: A002
        **kwargs: Any,
    ) -> list[tuple[Document, float]]:
        """基于给定图像 URI 搜索相似图像。

        Args:
            uri: 要搜索的图像的 URI。
            k: 返回的结果数量。
            filter: 按元数据过滤。
            **kwargs: 传递给函数的额外参数。

        Returns:
            包含与查询图像相似的文档及其相似性评分的元组列表。每个元组的第 0 个元素是
            LangChain 文档对象。 页面内容为 base64
            编码的图像，元数据为默认值或用户定义的值。

        Raises:
            ValueError: 如果嵌入函数不支持图像嵌入。
        """
        if self._embedding_function is not None and hasattr(
            self._embedding_function, "embed_image"
        ):
                        # 获取图像嵌入
                        # 假设 embed_image 返回单个嵌入
            image_embedding = self._embedding_function.embed_image(uris=[uri])

                        # 基于获取的嵌入执行相似性搜索
            return self.similarity_search_by_vector_with_relevance_scores(
                embedding=image_embedding,
                k=k,
                filter=filter,
                **kwargs,
            )
        msg = "The embedding function must support image embedding."
        raise ValueError(msg)

    def max_marginal_relevance_search_by_vector(
        self,
        embedding: list[float],
        k: int = DEFAULT_K,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        filter: dict[str, str] | None = None,  # noqa: A002
        where_document: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> list[Document]:
        """返回使用最大边际相关性选择的文档。

        Maximal marginal relevance optimizes for similarity to query AND diversity
        among selected documents.

        Args:
            embedding: Embedding to look up documents similar to.
            k: Number of `Document` objects to return.
            fetch_k: Number of `Document` objects to fetch to pass to MMR algorithm.
            lambda_mult: Number between 0 and 1 that determines the degree
                of diversity among the results with `0` corresponding
                to maximum diversity and `1` to minimum diversity.
            filter: Filter by metadata.
            where_document: dict used to filter by the document contents.
                e.g. `{"$contains": "hello"}`.
            kwargs: Additional keyword arguments to pass to Chroma collection query.

        Returns:
            通过最大边际相关性选择的 `Document` 对象列表。
        """
        results = self.__query_collection(
            query_embeddings=[embedding],
            n_results=fetch_k,
            where=filter,
            where_document=where_document,
            include=["metadatas", "documents", "distances", "embeddings"],
            **kwargs,
        )
        mmr_selected = maximal_marginal_relevance(
            np.array(embedding, dtype=np.float32),
            results["embeddings"][0],
            k=k,
            lambda_mult=lambda_mult,
        )

        candidates = _results_to_docs(results)

        return [r for i, r in enumerate(candidates) if i in mmr_selected]

    def max_marginal_relevance_search(
        self,
        query: str,
        k: int = DEFAULT_K,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        filter: dict[str, str] | None = None,  # noqa: A002
        where_document: dict[str, str] | None = None,
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
            where_document: dict used to filter by the document contents.
                e.g. `{"$contains": "hello"}`.
            kwargs: Additional keyword arguments to pass to Chroma collection query.

        Returns:
            通过最大边际相关性选择的 `Document` 对象列表。

        Raises:
            ValueError: 如果未提供嵌入函数。
        """
        if self._embedding_function is None:
            msg = "For MMR search, you must specify an embedding function on creation."
            raise ValueError(
                msg,
            )

        embedding = self._embedding_function.embed_query(query)
        return self.max_marginal_relevance_search_by_vector(
            embedding,
            k,
            fetch_k,
            lambda_mult=lambda_mult,
            filter=filter,
            where_document=where_document,
        )

    def delete_collection(self) -> None:
        """删除集合。"""
        self._client.delete_collection(self._collection.name)
        self._chroma_collection = None

    def reset_collection(self) -> None:
        """重置集合。

        通过删除集合并重新创建一个空集合来重置集合。
        """
        self.delete_collection()
        self.__ensure_collection()

    def get(
        self,
        ids: str | list[str] | None = None,
        where: Where | None = None,
        limit: int | None = None,
        offset: int | None = None,
        where_document: WhereDocument | None = None,
        include: list[str] | None = None,
    ) -> dict[str, Any]:
        """获取集合。

        Args:
            ids: 要获取的嵌入的 ID。可选。
            where: 用于按条件筛选结果的 Where 类型字典。
                   例如：{"$and": [{"color": "red"}, {"price": 4.20}]}
            limit: 要返回的文档数量。可选。
            offset: 开始返回结果的偏移量。
                    用于结合 limit 分页结果。可选。
            where_document: 用于按文档内容过滤的 WhereDocument 类型字典。
                            例如：{"$contains": "hello"}。可选。
            include: 要包含在结果中的列表。
                     可包含 "embeddings"、"metadatas"、"documents"。
                     IDs 总是包含在内。
                     默认为 ["metadatas", "documents"]。可选。

        Returns:
            包含键 "ids"、"embeddings"、"metadatas"、"documents" 的字典。
        """
        kwargs = {
            "ids": ids,
            "where": where,
            "limit": limit,
            "offset": offset,
            "where_document": where_document,
        }

        if include is not None:
            kwargs["include"] = include

        return self._collection.get(**kwargs)  # type: ignore[arg-type, return-value]

    def get_by_ids(self, ids: Sequence[str], /) -> list[Document]:
        """通过 ID 获取文档。

        返回的文档应具有设置为向量存储中对应文档 ID 的 ID 字段。

        Fewer documents may be returned than requested if some IDs are not found or
        if there are duplicated IDs.

        用户不应假设返回的文档顺序与输入 ID 的顺序一致。相反，用户应依赖返回文档中的 ID
        字段。

        This method should **NOT** raise exceptions if no documents are found for
        some IDs.

        Args:
            ids: 要检索的 ID 列表。

        Returns:
            Document 对象列表。

        !!! version-added "0.2.1 版本新增"
        """
        results = self.get(ids=list(ids))
        return [
            Document(page_content=doc, metadata=meta, id=doc_id)
            for doc, meta, doc_id in zip(
                results["documents"],
                results["metadatas"],
                results["ids"],
                strict=False,
            )
            if doc is not None  # 过滤掉 page_content 为 None 的文档
        ]

    def update_document(self, document_id: str, document: Document) -> None:
        """更新集合中的一个文档。

        Args:
            document_id: 要更新的文档 ID。
            document: 要更新的文档。
        """
        return self.update_documents([document_id], [document])

    def update_documents(self, ids: list[str], documents: list[Document]) -> None:
        """批量更新集合中的多个文档。

        Args:
            ids: 要更新的文档 ID 列表。
            documents: 与给定 ID 对应的 Document 对象列表。

        Raises:
            ValueError: 如果未提供嵌入函数。
        """
        text = [document.page_content for document in documents]
        metadata = [document.metadata for document in documents]
        if self._embedding_function is None:
            msg = "For update, you must specify an embedding function on creation."
            raise ValueError(
                msg,
            )
        embeddings = self._embedding_function.embed_documents(text)

        if hasattr(
            self._client,
            "get_max_batch_size",
        ) or hasattr(  # 适用于 Chroma 0.5.1 及以上版本
            self._client,
            "max_batch_size",
        ):  # 适用于 Chroma 0.4.10 及以上版本
            from chromadb.utils.batch_utils import create_batches

            for batch in create_batches(
                api=self._client,
                ids=ids,
                metadatas=metadata,  # type: ignore[arg-type]
                documents=text,
                embeddings=embeddings,  # type: ignore[arg-type]
            ):
                self._collection.update(
                    ids=batch[0],
                    embeddings=batch[1],
                    documents=batch[3],
                    metadatas=batch[2],
                )
        else:
            self._collection.update(
                ids=ids,
                embeddings=embeddings,  # type: ignore[arg-type]
                documents=text,
                metadatas=metadata,  # type: ignore[arg-type]
            )

    @classmethod
    def from_texts(
        cls: type[Chroma],
        texts: list[str],
        embedding: Embeddings | None = None,
        metadatas: list[dict] | None = None,
        ids: list[str] | None = None,
        collection_name: str = _LANGCHAIN_DEFAULT_COLLECTION_NAME,
        persist_directory: str | None = None,
        host: str | None = None,
        port: int | None = None,
        headers: dict[str, str] | None = None,
        chroma_cloud_api_key: str | None = None,
        tenant: str | None = None,
        database: str | None = None,
        client_settings: chromadb.config.Settings | None = None,
        client: chromadb.ClientAPI | None = None,
        collection_metadata: dict | None = None,
        collection_configuration: CreateCollectionConfiguration | None = None,
        *,
        ssl: bool = False,
        **kwargs: Any,
    ) -> Chroma:
        """从原始文档创建 Chroma 向量存储。

        如果指定了
        persist_directory，集合将被持久化到该目录。否则，数据将作为临时内存中数据存在。

        Args:
            texts: 要添加到集合中的文本列表。
            collection_name: 要创建的集合名称。
            persist_directory: 持久化集合的目录。
            host: 部署的 Chroma 服务器的主机名。
            port: 与部署的 Chroma 服务器连接的端口。
                    默认值为 8000。
            ssl: 是否与部署的 Chroma 服务器建立 SSL 连接。
                    默认值为 False。
            headers: 发送到部署的 Chroma 服务器的 HTTP 头信息。
            chroma_cloud_api_key: Chroma Cloud API 密钥。
            tenant: 租户 ID。连接 Chroma Cloud 时必需。
                    本地 Chroma 服务器的默认值为 'default_tenant'。
            database: 数据库名称。连接 Chroma Cloud 时必需。
                    默认值为 'default_database'。
            embedding: 嵌入函数。
            metadatas: 元数据列表。
            ids: 文档 ID 列表。
            client_settings: Chroma 客户端设置。
            client: Chroma 客户端。文档：
                    https://docs.trychroma.com/reference/python/client
            collection_metadata: 集合配置。
            collection_configuration: 集合的索引配置。

            kwargs: 初始化 Chroma 客户端时的额外关键字参数。

        Returns:
            Chroma: Chroma 向量存储。
        """
        chroma_collection = cls(
            collection_name=collection_name,
            embedding_function=embedding,
            persist_directory=persist_directory,
            host=host,
            port=port,
            ssl=ssl,
            headers=headers,
            chroma_cloud_api_key=chroma_cloud_api_key,
            tenant=tenant,
            database=database,
            client_settings=client_settings,
            client=client,
            collection_metadata=collection_metadata,
            collection_configuration=collection_configuration,
            **kwargs,
        )
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in texts]
        else:
            ids = [id_ if id_ is not None else str(uuid.uuid4()) for id_ in ids]
        if hasattr(
            chroma_collection._client,
            "get_max_batch_size",
        ) or hasattr(  # 适用于 Chroma 0.5.1 及以上版本
            chroma_collection._client,
            "max_batch_size",
        ):  # 适用于 Chroma 0.4.10 及以上版本
            from chromadb.utils.batch_utils import create_batches

            for batch in create_batches(
                api=chroma_collection._client,
                ids=ids,
                metadatas=metadatas,  # type: ignore[arg-type]
                documents=texts,
            ):
                chroma_collection.add_texts(
                    texts=batch[3] if batch[3] else [],
                    metadatas=batch[2] if batch[2] else None,  # type: ignore[arg-type]
                    ids=batch[0],
                )
        else:
            chroma_collection.add_texts(texts=texts, metadatas=metadatas, ids=ids)
        return chroma_collection

    @classmethod
    def from_documents(
        cls: type[Chroma],
        documents: list[Document],
        embedding: Embeddings | None = None,
        ids: list[str] | None = None,
        collection_name: str = _LANGCHAIN_DEFAULT_COLLECTION_NAME,
        persist_directory: str | None = None,
        host: str | None = None,
        port: int | None = None,
        headers: dict[str, str] | None = None,
        chroma_cloud_api_key: str | None = None,
        tenant: str | None = None,
        database: str | None = None,
        client_settings: chromadb.config.Settings | None = None,
        client: chromadb.ClientAPI | None = None,  # 添加此行
        collection_metadata: dict | None = None,
        collection_configuration: CreateCollectionConfiguration | None = None,
        *,
        ssl: bool = False,
        **kwargs: Any,
    ) -> Chroma:
        """从文档列表创建 Chroma 向量存储。

        如果指定了
        persist_directory，集合将被持久化到该目录。否则，数据将作为临时内存中数据存在。

        Args:
            collection_name: 要创建的集合名称。
            persist_directory: 用于持久化集合的目录。
            host: 部署的 Chroma 服务器的主机名。
            port: 与部署的 Chroma 服务器连接的端口。默认为 8000。
            ssl: 是否与部署的 Chroma 服务器建立 SSL 连接。
            headers: 发送到部署的 Chroma 服务器的 HTTP 头信息。
            chroma_cloud_api_key: Chroma Cloud API 密钥。
            tenant: 租户 ID。连接 Chroma Cloud 时必需。
                    本地 Chroma 服务器的默认值为 'default_tenant'。
            database: 数据库名称。连接 Chroma Cloud 时必需。
                    默认值为 'default_database'。
            ids: 文档 ID 列表。
            documents: 要添加到 `VectorStore` 的文档列表。
            embedding: 嵌入函数。
            client_settings: Chroma 客户端设置。
            client: Chroma 客户端。文档：
                    https://docs.trychroma.com/reference/python/client
            collection_metadata: 集合配置。
            collection_configuration: 集合的索引配置。

            kwargs: 初始化 Chroma 客户端时的额外关键字参数。

        Returns:
            Chroma: Chroma 向量存储。
        """
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        if ids is None:
            ids = [doc.id if doc.id else str(uuid.uuid4()) for doc in documents]
        return cls.from_texts(
            texts=texts,
            embedding=embedding,
            metadatas=metadatas,
            ids=ids,
            collection_name=collection_name,
            persist_directory=persist_directory,
            host=host,
            port=port,
            ssl=ssl,
            headers=headers,
            chroma_cloud_api_key=chroma_cloud_api_key,
            tenant=tenant,
            database=database,
            client_settings=client_settings,
            client=client,
            collection_metadata=collection_metadata,
            collection_configuration=collection_configuration,
            **kwargs,
        )

    def delete(self, ids: list[str] | None = None, **kwargs: Any) -> None:
        """按向量 ID 删除。

        Args:
            ids: 要删除的 ID 列表。
            kwargs: 额外的关键字参数。
        """
        self._collection.delete(ids=ids, **kwargs)

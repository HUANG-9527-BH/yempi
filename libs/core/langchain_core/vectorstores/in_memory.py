"""内存中的向量存储。"""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
)

from typing_extensions import override

from langchain_core.documents import Document
from langchain_core.load import dumpd, load
from langchain_core.vectorstores import VectorStore
from langchain_core.vectorstores.utils import _cosine_similarity as cosine_similarity
from langchain_core.vectorstores.utils import maximal_marginal_relevance

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Sequence

    from langchain_core.embeddings import Embeddings

try:
    import numpy as np

    _HAS_NUMPY = True
except ImportError:
    _HAS_NUMPY = False


class InMemoryVectorStore(VectorStore):
    """内存中向量存储的实现。

    使用字典，并使用 numpy 计算搜索时的余弦相似度。

    安装：
        安装 `langchain-core`。

        ```bash
        pip install -U langchain-core
        ```

    关键初始化参数 —— 索引参数：

        * embedding_function: Embeddings
            要使用的嵌入函数。

    实例化：
        ```python
        from langchain_core.vectorstores import InMemoryVectorStore
        from langchain_openai import OpenAIEmbeddings

        vector_store = InMemoryVectorStore(OpenAIEmbeddings())
        ```

    添加文档：
        ```python
        from langchain_core.documents import Document

        document_1 = Document(id="1", page_content="foo", metadata={"baz": "bar"})
        document_2 = Document(id="2", page_content="thud", metadata={"bar": "baz"})
        document_3 = Document(id="3", page_content="i will be deleted :(")

        documents = [document_1, document_2, document_3]
        vector_store.add_documents(documents=documents)
        ```

    检查文档：
        ```python
        top_n = 10
        for index, (id, doc) in enumerate(vector_store.store.items()):
            if index < top_n:
                # docs have keys 'id', 'vector', 'text', 'metadata'
                print(f"{id}: {doc['text']}")
            else:
                break
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

        ```txt
        * thud [{'bar': 'baz'}]
        ```

    带过滤条件的搜索：
        ```python
        def _filter_function(doc: Document) -> bool:
            return doc.metadata.get("bar") == "baz"


        results = vector_store.similarity_search(
            query="thud", k=1, filter=_filter_function
        )
        for doc in results:
            print(f"* {doc.page_content} [{doc.metadata}]")
        ```

        ```txt
        * thud [{'bar': 'baz'}]
        ```

    带得分的搜索：
        ```python
        results = vector_store.similarity_search_with_score(query="qux", k=1)
        for doc, score in results:
            print(f"* [SIM={score:3f}] {doc.page_content} [{doc.metadata}]")
        ```

        ```txt
        * [SIM=0.832268] foo [{'baz': 'bar'}]
        ```

    异步操作：
        ```python
        # add documents
        # await vector_store.aadd_documents(documents=documents)

        # delete documents
        # await vector_store.adelete(ids=["3"])

        # search
        # results = vector_store.asimilarity_search(query="thud", k=1)

        # search with score
        results = await vector_store.asimilarity_search_with_score(query="qux", k=1)
        for doc, score in results:
            print(f"* [SIM={score:3f}] {doc.page_content} [{doc.metadata}]")
        ```

        ```txt
        * [SIM=0.832268] foo [{'baz': 'bar'}]
        ```

    用作检索器：
        ```python
        retriever = vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 1, "fetch_k": 2, "lambda_mult": 0.5},
        )
        retriever.invoke("thud")
        ```

        ```txt
        [Document(id='2', metadata={'bar': 'baz'}, page_content='thud')]
        ```
    """

    def __init__(self, embedding: Embeddings) -> None:
        """使用给定的嵌入函数进行初始化。

        Args:
            embedding: 要使用的嵌入函数。
        """
                # TODO: 未来希望能改为
                # dict[str, Document]（这将是一个破坏性变更）
        self.store: dict[str, dict[str, Any]] = {}
        self.embedding = embedding

    @property
    @override
    def embeddings(self) -> Embeddings:
        return self.embedding

    @override
    def delete(self, ids: Sequence[str] | None = None, **kwargs: Any) -> None:
        if ids:
            for id_ in ids:
                self.store.pop(id_, None)

    @override
    async def adelete(self, ids: Sequence[str] | None = None, **kwargs: Any) -> None:
        self.delete(ids)

    @override
    def add_documents(
        self,
        documents: list[Document],
        ids: list[str] | None = None,
        **kwargs: Any,
    ) -> list[str]:
        texts = [doc.page_content for doc in documents]
        vectors = self.embedding.embed_documents(texts)

        if ids and len(ids) != len(texts):
            msg = (
                f"ids must be the same length as texts. "
                f"Got {len(ids)} ids and {len(texts)} texts."
            )
            raise ValueError(msg)

        id_iterator: Iterator[str | None] = (
            iter(ids) if ids else iter(doc.id for doc in documents)
        )

        ids_ = []

        for doc, vector in zip(documents, vectors, strict=False):
            doc_id = next(id_iterator)
            doc_id_ = doc_id or str(uuid.uuid4())
            ids_.append(doc_id_)
            self.store[doc_id_] = {
                "id": doc_id_,
                "vector": vector,
                "text": doc.page_content,
                "metadata": doc.metadata,
            }

        return ids_

    @override
    async def aadd_documents(
        self, documents: list[Document], ids: list[str] | None = None, **kwargs: Any
    ) -> list[str]:
        texts = [doc.page_content for doc in documents]
        vectors = await self.embedding.aembed_documents(texts)

        if ids and len(ids) != len(texts):
            msg = (
                f"ids must be the same length as texts. "
                f"Got {len(ids)} ids and {len(texts)} texts."
            )
            raise ValueError(msg)

        id_iterator: Iterator[str | None] = (
            iter(ids) if ids else iter(doc.id for doc in documents)
        )
        ids_: list[str] = []

        for doc, vector in zip(documents, vectors, strict=False):
            doc_id = next(id_iterator)
            doc_id_ = doc_id or str(uuid.uuid4())
            ids_.append(doc_id_)
            self.store[doc_id_] = {
                "id": doc_id_,
                "vector": vector,
                "text": doc.page_content,
                "metadata": doc.metadata,
            }

        return ids_

    @override
    def get_by_ids(self, ids: Sequence[str], /) -> list[Document]:
        """根据其 ID 获取文档。

        Args:
            ids: 要获取的文档的 ID。

        Returns:
            一个 `Document` 对象列表。
        """
        documents = []

        for doc_id in ids:
            doc = self.store.get(doc_id)
            if doc:
                documents.append(
                    Document(
                        id=doc["id"],
                        page_content=doc["text"],
                        metadata=doc["metadata"],
                    )
                )
        return documents

    @override
    async def aget_by_ids(self, ids: Sequence[str], /) -> list[Document]:
        """异步根据其 ID 获取文档。

        Args:
            ids: 要获取的文档的 ID。

        Returns:
            一个 `Document` 对象列表。
        """
        return self.get_by_ids(ids)

    def _similarity_search_with_score_by_vector(
        self,
        embedding: list[float],
        k: int = 4,
        filter: Callable[[Document], bool] | None = None,  # noqa: A002
    ) -> list[tuple[Document, float, list[float]]]:
                # 按固定顺序获取所有文档的列表
        docs = list(self.store.values())

        if filter is not None:
            docs = [
                doc
                for doc in docs
                if filter(
                    Document(
                        id=doc["id"], page_content=doc["text"], metadata=doc["metadata"]
                    )
                )
            ]

        if not docs:
            return []

        similarity = cosine_similarity([embedding], [doc["vector"] for doc in docs])[0]

                # 按相似度得分排序的索引
        top_k_idx = similarity.argsort()[::-1][:k]

        return [
            (
                Document(
                    id=doc_dict["id"],
                    page_content=doc_dict["text"],
                    metadata=doc_dict["metadata"],
                ),
                float(similarity[idx].item()),
                doc_dict["vector"],
            )
            for idx in top_k_idx
                        # 使用海象运算符赋值以避免多次查找
            if (doc_dict := docs[idx])
        ]

    def similarity_search_with_score_by_vector(
        self,
        embedding: list[float],
        k: int = 4,
        filter: Callable[[Document], bool] | None = None,  # noqa: A002
        **_kwargs: Any,
    ) -> list[tuple[Document, float]]:
        """搜索与给定嵌入最相似的文档。

        Args:
            embedding: 要搜索的嵌入。
            k: 返回的文档数量。
            filter: 用于过滤文档的函数。

        Returns:
            一个包含 `Document` 对象及其相似度得分的元组列表。
        """
        return [
            (doc, similarity)
            for doc, similarity, _ in self._similarity_search_with_score_by_vector(
                embedding=embedding, k=k, filter=filter
            )
        ]

    @override
    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        **kwargs: Any,
    ) -> list[tuple[Document, float]]:
        embedding = self.embedding.embed_query(query)
        return self.similarity_search_with_score_by_vector(
            embedding,
            k,
            **kwargs,
        )

    @override
    async def asimilarity_search_with_score(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> list[tuple[Document, float]]:
        embedding = await self.embedding.aembed_query(query)
        return self.similarity_search_with_score_by_vector(
            embedding,
            k,
            **kwargs,
        )

    @override
    def similarity_search_by_vector(
        self,
        embedding: list[float],
        k: int = 4,
        **kwargs: Any,
    ) -> list[Document]:
        docs_and_scores = self.similarity_search_with_score_by_vector(
            embedding,
            k,
            **kwargs,
        )
        return [doc for doc, _ in docs_and_scores]

    @override
    async def asimilarity_search_by_vector(
        self, embedding: list[float], k: int = 4, **kwargs: Any
    ) -> list[Document]:
        return self.similarity_search_by_vector(embedding, k, **kwargs)

    @override
    def similarity_search(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> list[Document]:
        return [doc for doc, _ in self.similarity_search_with_score(query, k, **kwargs)]

    @override
    async def asimilarity_search(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> list[Document]:
        return [
            doc
            for doc, _ in await self.asimilarity_search_with_score(query, k, **kwargs)
        ]

    @override
    def max_marginal_relevance_search_by_vector(
        self,
        embedding: list[float],
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        *,
        filter: Callable[[Document], bool] | None = None,
        **kwargs: Any,
    ) -> list[Document]:
        prefetch_hits = self._similarity_search_with_score_by_vector(
            embedding=embedding,
            k=fetch_k,
            filter=filter,
        )

        if not _HAS_NUMPY:
            msg = (
                "numpy must be installed to use max_marginal_relevance_search "
                "pip install numpy"
            )
            raise ImportError(msg)

        mmr_chosen_indices = maximal_marginal_relevance(
            np.array(embedding, dtype=np.float32),
            [vector for _, _, vector in prefetch_hits],
            k=k,
            lambda_mult=lambda_mult,
        )
        return [prefetch_hits[idx][0] for idx in mmr_chosen_indices]

    @override
    def max_marginal_relevance_search(
        self,
        query: str,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        **kwargs: Any,
    ) -> list[Document]:
        embedding_vector = self.embedding.embed_query(query)
        return self.max_marginal_relevance_search_by_vector(
            embedding_vector,
            k,
            fetch_k,
            lambda_mult=lambda_mult,
            **kwargs,
        )

    @override
    async def amax_marginal_relevance_search(
        self,
        query: str,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        **kwargs: Any,
    ) -> list[Document]:
        embedding_vector = await self.embedding.aembed_query(query)
        return self.max_marginal_relevance_search_by_vector(
            embedding_vector,
            k,
            fetch_k,
            lambda_mult=lambda_mult,
            **kwargs,
        )

    @classmethod
    @override
    def from_texts(
        cls,
        texts: list[str],
        embedding: Embeddings,
        metadatas: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> InMemoryVectorStore:
        store = cls(
            embedding=embedding,
        )
        store.add_texts(texts=texts, metadatas=metadatas, **kwargs)
        return store

    @classmethod
    @override
    async def afrom_texts(
        cls,
        texts: list[str],
        embedding: Embeddings,
        metadatas: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> InMemoryVectorStore:
        store = cls(
            embedding=embedding,
        )
        await store.aadd_texts(texts=texts, metadatas=metadatas, **kwargs)
        return store

    @classmethod
    def load(
        cls, path: str, embedding: Embeddings, **kwargs: Any
    ) -> InMemoryVectorStore:
        """从文件加载向量存储。

        Args:
            path: 要加载向量存储的路径。
            embedding: 要使用的嵌入。
            **kwargs: 传递给构造函数的额外参数。

        Returns:
            一个 `VectorStore` 对象。
        """
        path_: Path = Path(path)
        with path_.open("r", encoding="utf-8") as f:
            store = load(json.load(f), allowed_objects=[Document])
        vectorstore = cls(embedding=embedding, **kwargs)
        vectorstore.store = store
        return vectorstore

    def dump(self, path: str) -> None:
        """将向量存储转储到文件。

        Args:
            path: 要转储向量存储的路径。
        """
        path_: Path = Path(path)
        path_.parent.mkdir(exist_ok=True, parents=True)
        with path_.open("w", encoding="utf-8") as f:
            json.dump(dumpd(self.store), f, indent=2)

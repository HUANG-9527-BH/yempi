from enum import Enum
from typing import Any

from langchain_core.callbacks import (
    AsyncCallbackManagerForRetrieverRun,
    CallbackManagerForRetrieverRun,
)
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.stores import BaseStore, ByteStore
from langchain_core.vectorstores import VectorStore
from pydantic import Field, model_validator
from typing_extensions import override

from langchain_classic.storage._lc_store import create_kv_docstore


class SearchType(str, Enum):
    """搜索类型枚举。"""

    similarity = "similarity"
    """相似度搜索。"""
    similarity_score_threshold = "similarity_score_threshold"
    """带得分阈值的相似度搜索。"""
    mmr = "mmr"
    """对相似度搜索结果进行最大边际相关性重排序。"""


class MultiVectorRetriever(BaseRetriever):
    """支持单个父文档多个嵌入的检索器。

    This retriever is designed for scenarios where documents are split into
    smaller chunks for embedding and vector search, but retrieval returns
    the original parent documents rather than individual chunks.

    其工作方式为：
    - 在嵌入的子块上执行相似度（或 MMR）搜索
    - 从块元数据中收集唯一的父文档 ID
    - 从文档存储中获取并返回对应的父文档

    This pattern is commonly used in RAG pipelines to improve answer grounding
    while preserving full document context.
    """

    vectorstore: VectorStore
    """用于存储小块及其嵌入向量的底层 `VectorStore`"""

    byte_store: ByteStore | None = None
    """父文档的底层后备存储层"""

    docstore: BaseStore[str, Document]
    """父文档的存储接口"""

    id_key: str = "doc_id"

    search_kwargs: dict = Field(default_factory=dict)
    """传递给搜索函数的关键字参数。"""

    search_type: SearchType = SearchType.similarity
    """要执行的搜索类型（相似度 / MMR）"""

    @model_validator(mode="before")
    @classmethod
    def _shim_docstore(cls, values: dict) -> Any:
        byte_store = values.get("byte_store")
        docstore = values.get("docstore")
        if byte_store is not None:
            docstore = create_kv_docstore(byte_store)
        elif docstore is None:
            msg = "You must pass a `byte_store` parameter."
            raise ValueError(msg)
        values["docstore"] = docstore
        return values

    @override
    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> list[Document]:
        """获取与查询相关的文档。

        Args:
            query: 用于查找相关文档的字符串
            run_manager: 要使用的回调处理器
        Returns:
            相关文档列表。
        """
        if self.search_type == SearchType.mmr:
            sub_docs = self.vectorstore.max_marginal_relevance_search(
                query,
                **self.search_kwargs,
            )
        elif self.search_type == SearchType.similarity_score_threshold:
            sub_docs_and_similarities = (
                self.vectorstore.similarity_search_with_relevance_scores(
                    query,
                    **self.search_kwargs,
                )
            )
            sub_docs = [sub_doc for sub_doc, _ in sub_docs_and_similarities]
        else:
            sub_docs = self.vectorstore.similarity_search(query, **self.search_kwargs)

                # 我们这样做是为了保持返回 ID 的顺序
        ids = []
        for d in sub_docs:
            if self.id_key in d.metadata and d.metadata[self.id_key] not in ids:
                ids.append(d.metadata[self.id_key])
        docs = self.docstore.mget(ids)
        return [d for d in docs if d is not None]

    @override
    async def _aget_relevant_documents(
        self,
        query: str,
        *,
        run_manager: AsyncCallbackManagerForRetrieverRun,
    ) -> list[Document]:
        """异步获取与查询相关的文档。

        Args:
            query: 用于查找相关文档的字符串
            run_manager: 要使用的回调处理器
        Returns:
            相关文档列表。
        """
        if self.search_type == SearchType.mmr:
            sub_docs = await self.vectorstore.amax_marginal_relevance_search(
                query,
                **self.search_kwargs,
            )
        elif self.search_type == SearchType.similarity_score_threshold:
            sub_docs_and_similarities = (
                await self.vectorstore.asimilarity_search_with_relevance_scores(
                    query,
                    **self.search_kwargs,
                )
            )
            sub_docs = [sub_doc for sub_doc, _ in sub_docs_and_similarities]
        else:
            sub_docs = await self.vectorstore.asimilarity_search(
                query,
                **self.search_kwargs,
            )

                # 我们这样做是为了保持返回 ID 的顺序
        ids = []
        for d in sub_docs:
            if self.id_key in d.metadata and d.metadata[self.id_key] not in ids:
                ids.append(d.metadata[self.id_key])
        docs = await self.docstore.amget(ids)
        return [d for d in docs if d is not None]

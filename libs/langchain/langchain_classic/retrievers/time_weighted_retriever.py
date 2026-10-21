import datetime
from copy import deepcopy
from typing import Any

from langchain_core.callbacks import (
    AsyncCallbackManagerForRetrieverRun,
    CallbackManagerForRetrieverRun,
)
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.vectorstores import VectorStore
from pydantic import ConfigDict, Field
from typing_extensions import override


def _get_hours_passed(time: datetime.datetime, ref_time: datetime.datetime) -> float:
    """获取两个日期时间之间经过的小时数。"""
    return (time - ref_time).total_seconds() / 3600


class TimeWeightedVectorStoreRetriever(BaseRetriever):
    """时间加权向量存储检索器。

    结合嵌入相似性与新近性进行值检索的检索器。
    """

    vectorstore: VectorStore
    """用于存储文档并确定显著性的 `VectorStore`。"""

    search_kwargs: dict = Field(default_factory=lambda: {"k": 100})
    """传递给 `VectorStore` 相似性搜索的关键词参数。"""

        # TODO: 抽象为队列
    memory_stream: list[Document] = Field(default_factory=list)
    """要搜索的文档流。"""

    decay_rate: float = Field(default=0.01)
    """用于计算 `(1.0-decay_rate)**(hrs_passed)` 的指数衰减因子。"""

    k: int = 4
    """单次调用中要检索的最大文档数量。"""

    other_score_keys: list[str] = []
    """需纳入评分计算的元数据中的其他键，例如 'importance'。"""

    default_salience: float | None = None
    """为未从向量存储中检索到的文档分配的显著性。

    None 表示不为未从向量存储中获取的文档分配显著性。
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )

    def _document_get_date(self, field: str, document: Document) -> datetime.datetime:
        """返回文档日期字段的值。"""
        if field in document.metadata:
            if isinstance(document.metadata[field], float):
                return datetime.datetime.fromtimestamp(document.metadata[field])
            return document.metadata[field]
        return datetime.datetime.now()

    def _get_combined_score(
        self,
        document: Document,
        vector_relevance: float | None,
        current_time: datetime.datetime,
    ) -> float:
        """返回文档的综合评分。"""
        hours_passed = _get_hours_passed(
            current_time,
            self._document_get_date("last_accessed_at", document),
        )
        score = (1.0 - self.decay_rate) ** hours_passed
        for key in self.other_score_keys:
            if key in document.metadata:
                score += document.metadata[key]
        if vector_relevance is not None:
            score += vector_relevance
        return score

    def get_salient_docs(self, query: str) -> dict[int, tuple[Document, float]]:
        """返回对查询具有显著性的文档。"""
        docs_and_scores: list[tuple[Document, float]]
        docs_and_scores = self.vectorstore.similarity_search_with_relevance_scores(
            query,
            **self.search_kwargs,
        )
        results = {}
        for fetched_doc, relevance in docs_and_scores:
            if "buffer_idx" in fetched_doc.metadata:
                buffer_idx = fetched_doc.metadata["buffer_idx"]
                doc = self.memory_stream[buffer_idx]
                results[buffer_idx] = (doc, relevance)
        return results

    async def aget_salient_docs(self, query: str) -> dict[int, tuple[Document, float]]:
        """返回对查询具有显著性的文档。"""
        docs_and_scores: list[tuple[Document, float]]
        docs_and_scores = (
            await self.vectorstore.asimilarity_search_with_relevance_scores(
                query,
                **self.search_kwargs,
            )
        )
        results = {}
        for fetched_doc, relevance in docs_and_scores:
            if "buffer_idx" in fetched_doc.metadata:
                buffer_idx = fetched_doc.metadata["buffer_idx"]
                doc = self.memory_stream[buffer_idx]
                results[buffer_idx] = (doc, relevance)
        return results

    def _get_rescored_docs(
        self,
        docs_and_scores: dict[Any, tuple[Document, float | None]],
    ) -> list[Document]:
        current_time = datetime.datetime.now()
        rescored_docs = [
            (doc, self._get_combined_score(doc, relevance, current_time))
            for doc, relevance in docs_and_scores.values()
        ]
        rescored_docs.sort(key=lambda x: x[1], reverse=True)
        result = []
                # 确保频繁访问的记忆不会被遗忘
        for doc, _ in rescored_docs[: self.k]:
                        # TODO: 一旦 `update` 方法暴露，更新向量存储文档说明。
            buffered_doc = self.memory_stream[doc.metadata["buffer_idx"]]
            buffered_doc.metadata["last_accessed_at"] = current_time
            result.append(buffered_doc)
        return result

    @override
    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> list[Document]:
        docs_and_scores = {
            doc.metadata["buffer_idx"]: (doc, self.default_salience)
            for doc in self.memory_stream[-self.k :]
        }
                # 如果文档被认为具有显著性，则更新其显著性评分
        docs_and_scores.update(self.get_salient_docs(query))
        return self._get_rescored_docs(docs_and_scores)

    @override
    async def _aget_relevant_documents(
        self,
        query: str,
        *,
        run_manager: AsyncCallbackManagerForRetrieverRun,
    ) -> list[Document]:
        docs_and_scores = {
            doc.metadata["buffer_idx"]: (doc, self.default_salience)
            for doc in self.memory_stream[-self.k :]
        }
                # 如果文档被认为具有显著性，则更新其显著性评分
        docs_and_scores.update(await self.aget_salient_docs(query))
        return self._get_rescored_docs(docs_and_scores)

    def add_documents(self, documents: list[Document], **kwargs: Any) -> list[str]:
        """将文档添加到向量存储。"""
        current_time = kwargs.get("current_time")
        if current_time is None:
            current_time = datetime.datetime.now()
                # 避免修改输入文档
        dup_docs = [deepcopy(d) for d in documents]
        for i, doc in enumerate(dup_docs):
            if "last_accessed_at" not in doc.metadata:
                doc.metadata["last_accessed_at"] = current_time
            if "created_at" not in doc.metadata:
                doc.metadata["created_at"] = current_time
            doc.metadata["buffer_idx"] = len(self.memory_stream) + i
        self.memory_stream.extend(dup_docs)
        return self.vectorstore.add_documents(dup_docs, **kwargs)

    async def aadd_documents(
        self,
        documents: list[Document],
        **kwargs: Any,
    ) -> list[str]:
        """将文档添加到向量存储。"""
        current_time = kwargs.get("current_time")
        if current_time is None:
            current_time = datetime.datetime.now()
                # 避免修改输入文档
        dup_docs = [deepcopy(d) for d in documents]
        for i, doc in enumerate(dup_docs):
            if "last_accessed_at" not in doc.metadata:
                doc.metadata["last_accessed_at"] = current_time
            if "created_at" not in doc.metadata:
                doc.metadata["created_at"] = current_time
            doc.metadata["buffer_idx"] = len(self.memory_stream) + i
        self.memory_stream.extend(dup_docs)
        return await self.vectorstore.aadd_documents(dup_docs, **kwargs)

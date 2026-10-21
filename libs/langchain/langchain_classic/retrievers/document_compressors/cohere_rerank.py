from __future__ import annotations

from collections.abc import Sequence
from copy import deepcopy
from typing import Any

from langchain_core._api.deprecation import deprecated
from langchain_core.callbacks import Callbacks
from langchain_core.documents import BaseDocumentCompressor, Document
from langchain_core.utils import get_from_dict_or_env
from pydantic import ConfigDict, model_validator
from typing_extensions import override


@deprecated(
    since="0.0.30",
    removal="2.0.0",
    alternative_import="langchain_cohere.CohereRerank",
)
class CohereRerank(BaseDocumentCompressor):
    """使用 `Cohere Rerank API` 的文档压缩器。"""

    client: Any = None
    """用于压缩文档的 Cohere 客户端。"""
    top_n: int | None = 3
    """要返回的文档数量。"""
    model: str = "rerank-english-v2.0"
    """用于重排序的模型。"""
    cohere_api_key: str | None = None
    """Cohere API 密钥。必须直接指定，或通过环境变量 COHERE_API_KEY 提供。"""
    user_agent: str = "langchain"
    """请求发起的应用程序标识符。"""

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="forbid",
    )

    @model_validator(mode="before")
    @classmethod
    def validate_environment(cls, values: dict) -> Any:
        """验证 API 密钥和 Python 包是否存在于环境中。"""
        if not values.get("client"):
            try:
                import cohere
            except ImportError as e:
                msg = (
                    "Could not import cohere python package. "
                    "Please install it with `pip install cohere`."
                )
                raise ImportError(msg) from e
            cohere_api_key = get_from_dict_or_env(
                values,
                "cohere_api_key",
                "COHERE_API_KEY",
            )
            client_name = values.get("user_agent", "langchain")
            values["client"] = cohere.Client(cohere_api_key, client_name=client_name)
        return values

    def rerank(
        self,
        documents: Sequence[str | Document | dict],
        query: str,
        *,
        model: str | None = None,
        top_n: int | None = -1,
        max_chunks_per_doc: int | None = None,
    ) -> list[dict[str, Any]]:
        """返回按与所提供查询的相关性排序的文档列表。

        Args:
            query: 用于重排序的查询。
            documents: 要重排序的文档序列。
            model: 用于重排序的模型。默认为 self.model。
            top_n: 要返回的结果数量。如果为 `None` 则返回所有结果。
            max_chunks_per_doc: 从一个文档中提取的最大块数。
        """  # noqa: E501
        if len(documents) == 0:  # 避免空 API 调用
            return []
        docs = [
            doc.page_content if isinstance(doc, Document) else doc for doc in documents
        ]
        model = model or self.model
        top_n = top_n if (top_n is None or top_n > 0) else self.top_n
        results = self.client.rerank(
            query=query,
            documents=docs,
            model=model,
            top_n=top_n,
            max_chunks_per_doc=max_chunks_per_doc,
        )
        if hasattr(results, "results"):
            results = results.results
        return [
            {"index": res.index, "relevance_score": res.relevance_score}
            for res in results
        ]

    @override
    def compress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Callbacks | None = None,
    ) -> Sequence[Document]:
        """使用 Cohere 的重排序 API 压缩文档。

        Args:
            documents: 要压缩的文档序列。
            query: 用于压缩文档的查询。
            callbacks: 在压缩过程中运行的回调。

        Returns:
            压缩后的文档序列。
        """
        compressed = []
        for res in self.rerank(documents, query):
            doc = documents[res["index"]]
            doc_copy = Document(doc.page_content, metadata=deepcopy(doc.metadata))
            doc_copy.metadata["relevance_score"] = res["relevance_score"]
            compressed.append(doc_copy)
        return compressed

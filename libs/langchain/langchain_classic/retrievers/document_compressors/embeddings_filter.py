from collections.abc import Callable, Sequence

from langchain_core.callbacks import Callbacks
from langchain_core.documents import BaseDocumentCompressor, Document
from langchain_core.embeddings import Embeddings
from langchain_core.utils import pre_init
from pydantic import ConfigDict, Field
from typing_extensions import override


def _get_similarity_function() -> Callable:
    try:
        from langchain_community.utils.math import cosine_similarity
    except ImportError as e:
        msg = (
            "To use please install langchain-community "
            "with `pip install langchain-community`."
        )
        raise ImportError(msg) from e
    return cosine_similarity


class EmbeddingsFilter(BaseDocumentCompressor):
    """嵌入过滤器。

    使用嵌入来剔除与查询无关文档的文档压缩器。
    """

    embeddings: Embeddings
    """用于嵌入文档内容和查询的嵌入。"""
    similarity_fn: Callable = Field(default_factory=_get_similarity_function)
    """Similarity function for comparing documents. Function expected to take as input
    two matrices (List[List[float]]) and return a matrix of scores where higher values
    indicate greater similarity."""
    k: int | None = 20
    """要返回的相关文档数量。可设置为 `None`，此时必须指定 `similarity_threshold`。"""
    similarity_threshold: float | None = None
    """Threshold for determining when two documents are similar enough
    to be considered redundant. Defaults to `None`, must be specified if `k` is set
    to None."""

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )

    @pre_init
    def validate_params(cls, values: dict) -> dict:
        """验证相似性参数。"""
        if values["k"] is None and values["similarity_threshold"] is None:
            msg = "Must specify one of `k` or `similarity_threshold`."
            raise ValueError(msg)
        return values

    @override
    def compress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Callbacks | None = None,
    ) -> Sequence[Document]:
        """基于文档嵌入与查询的相似性过滤文档。"""
        try:
            from langchain_community.document_transformers.embeddings_redundant_filter import (  # noqa: E501
                _get_embeddings_from_stateful_docs,
                get_stateful_documents,
            )
        except ImportError as e:
            msg = (
                "To use please install langchain-community "
                "with `pip install langchain-community`."
            )
            raise ImportError(msg) from e

        try:
            import numpy as np
        except ImportError as e:
            msg = "Could not import numpy, please install with `pip install numpy`."
            raise ImportError(msg) from e
        stateful_documents = get_stateful_documents(documents)
        embedded_documents = _get_embeddings_from_stateful_docs(
            self.embeddings,
            stateful_documents,
        )
        embedded_query = self.embeddings.embed_query(query)
        similarity = self.similarity_fn([embedded_query], embedded_documents)[0]
        included_idxs: np.ndarray = np.arange(len(embedded_documents))
        if self.k is not None:
            included_idxs = np.argsort(similarity)[::-1][: self.k]
        if self.similarity_threshold is not None:
            similar_enough = np.where(
                similarity[included_idxs] > self.similarity_threshold,
            )
            included_idxs = included_idxs[similar_enough]
        for i in included_idxs:
            stateful_documents[i].state["query_similarity_score"] = similarity[i]
        return [stateful_documents[i] for i in included_idxs]

    @override
    async def acompress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Callbacks | None = None,
    ) -> Sequence[Document]:
        """基于文档嵌入与查询的相似性过滤文档。"""
        try:
            from langchain_community.document_transformers.embeddings_redundant_filter import (  # noqa: E501
                _aget_embeddings_from_stateful_docs,
                get_stateful_documents,
            )
        except ImportError as e:
            msg = (
                "To use please install langchain-community "
                "with `pip install langchain-community`."
            )
            raise ImportError(msg) from e

        try:
            import numpy as np
        except ImportError as e:
            msg = "Could not import numpy, please install with `pip install numpy`."
            raise ImportError(msg) from e
        stateful_documents = get_stateful_documents(documents)
        embedded_documents = await _aget_embeddings_from_stateful_docs(
            self.embeddings,
            stateful_documents,
        )
        embedded_query = await self.embeddings.aembed_query(query)
        similarity = self.similarity_fn([embedded_query], embedded_documents)[0]
        included_idxs: np.ndarray = np.arange(len(embedded_documents))
        if self.k is not None:
            included_idxs = np.argsort(similarity)[::-1][: self.k]
        if self.similarity_threshold is not None:
            similar_enough = np.where(
                similarity[included_idxs] > self.similarity_threshold,
            )
            included_idxs = included_idxs[similar_enough]
        for i in included_idxs:
            stateful_documents[i].state["query_similarity_score"] = similarity[i]
        return [stateful_documents[i] for i in included_idxs]

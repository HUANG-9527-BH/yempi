from __future__ import annotations

import operator
from collections.abc import Sequence

from langchain_core.callbacks import Callbacks
from langchain_core.documents import BaseDocumentCompressor, Document
from pydantic import ConfigDict
from typing_extensions import override

from langchain_classic.retrievers.document_compressors.cross_encoder import (
    BaseCrossEncoder,
)


class CrossEncoderReranker(BaseDocumentCompressor):
    """使用 CrossEncoder 进行重排序的文档压缩器。"""

    model: BaseCrossEncoder
    """CrossEncoder model to use for scoring similarity
      between the query and documents."""
    top_n: int = 3
    """要返回的文档数量。"""

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="forbid",
    )

    @override
    def compress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Callbacks | None = None,
    ) -> Sequence[Document]:
        """使用 CrossEncoder 重排序文档。

        Args:
            documents: 要压缩的文档序列。
            query: 用于压缩文档的查询。
            callbacks: 在压缩过程中运行的回调。

        Returns:
            压缩后的文档序列。
        """
        scores = self.model.score([(query, doc.page_content) for doc in documents])
        docs_with_scores = list(zip(documents, scores, strict=False))
        result = sorted(docs_with_scores, key=operator.itemgetter(1), reverse=True)
        return [doc for doc, _ in result[: self.top_n]]

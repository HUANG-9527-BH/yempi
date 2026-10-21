from collections.abc import Sequence
from inspect import signature

from langchain_core.callbacks import Callbacks
from langchain_core.documents import (
    BaseDocumentCompressor,
    BaseDocumentTransformer,
    Document,
)
from pydantic import ConfigDict


class DocumentCompressorPipeline(BaseDocumentCompressor):
    """使用 Transformers 流水线的文档压缩器。"""

    transformers: list[BaseDocumentTransformer | BaseDocumentCompressor]
    """一系列串联在一起并按顺序运行的文档过滤器。"""

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )

    def compress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Callbacks | None = None,
    ) -> Sequence[Document]:
        """转换文档列表。"""
        for _transformer in self.transformers:
            if isinstance(_transformer, BaseDocumentCompressor):
                accepts_callbacks = (
                    signature(_transformer.compress_documents).parameters.get(
                        "callbacks",
                    )
                    is not None
                )
                if accepts_callbacks:
                    documents = _transformer.compress_documents(
                        documents,
                        query,
                        callbacks=callbacks,
                    )
                else:
                    documents = _transformer.compress_documents(documents, query)
            elif isinstance(_transformer, BaseDocumentTransformer):
                documents = _transformer.transform_documents(documents)
            else:
                msg = f"Got unexpected transformer type: {_transformer}"  # type: ignore[unreachable]
                raise ValueError(msg)  # noqa: TRY004
        return documents

    async def acompress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Callbacks | None = None,
    ) -> Sequence[Document]:
        """根据查询上下文压缩检索到的文档。"""
        for _transformer in self.transformers:
            if isinstance(_transformer, BaseDocumentCompressor):
                accepts_callbacks = (
                    signature(_transformer.acompress_documents).parameters.get(
                        "callbacks",
                    )
                    is not None
                )
                if accepts_callbacks:
                    documents = await _transformer.acompress_documents(
                        documents,
                        query,
                        callbacks=callbacks,
                    )
                else:
                    documents = await _transformer.acompress_documents(documents, query)
            elif isinstance(_transformer, BaseDocumentTransformer):
                documents = await _transformer.atransform_documents(documents)
            else:
                msg = f"Got unexpected transformer type: {_transformer}"  # type: ignore[unreachable]
                raise ValueError(msg)  # noqa: TRY004
        return documents

from typing import Any

from langchain_core.callbacks import (
    AsyncCallbackManagerForRetrieverRun,
    CallbackManagerForRetrieverRun,
)
from langchain_core.documents import BaseDocumentCompressor, Document
from langchain_core.retrievers import BaseRetriever, RetrieverLike
from pydantic import ConfigDict
from typing_extensions import override


class ContextualCompressionRetriever(BaseRetriever):
    """包装基础检索器并压缩结果的检索器。"""

    base_compressor: BaseDocumentCompressor
    """用于压缩检索文档的压缩器。"""

    base_retriever: RetrieverLike
    """用于获取相关文档的基础检索器。"""

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )

    @override
    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
        **kwargs: Any,
    ) -> list[Document]:
        docs = self.base_retriever.invoke(
            query,
            config={"callbacks": run_manager.get_child()},
            **kwargs,
        )
        if docs:
            compressed_docs = self.base_compressor.compress_documents(
                docs,
                query,
                callbacks=run_manager.get_child(),
            )
            return list(compressed_docs)
        return []

    @override
    async def _aget_relevant_documents(
        self,
        query: str,
        *,
        run_manager: AsyncCallbackManagerForRetrieverRun,
        **kwargs: Any,
    ) -> list[Document]:
        docs = await self.base_retriever.ainvoke(
            query,
            config={"callbacks": run_manager.get_child()},
            **kwargs,
        )
        if docs:
            compressed_docs = await self.base_compressor.acompress_documents(
                docs,
                query,
                callbacks=run_manager.get_child(),
            )
            return list(compressed_docs)
        return []

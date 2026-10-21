"""文档压缩器。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from pydantic import BaseModel

from langchain_core.runnables import run_in_executor

if TYPE_CHECKING:
    from collections.abc import Sequence

    from langchain_core.callbacks import Callbacks
    from langchain_core.documents import Document


class BaseDocumentCompressor(BaseModel, ABC):
    """文档压缩器的基类。

    此抽象主要用于检索文档的后处理。

    首先检索与给定查询匹配的 `Document` 对象。

    然后可以对文档列表进行进一步处理。

    例如，可以使用 LLM 对检索到的文档进行重新排序。

    !!! note
        Users should favor using a `RunnableLambda` instead of sub-classing from this
        interface.

    """

    @abstractmethod
    def compress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Callbacks | None = None,
    ) -> Sequence[Document]:
        """根据查询上下文压缩检索到的文档。

        Args:
            documents: 检索到的 `Document` 对象。
            query: 查询上下文。
            callbacks: 可选的 `Callbacks`，在压缩过程中运行。

        Returns:
            压缩后的文档。

        """

    async def acompress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Callbacks | None = None,
    ) -> Sequence[Document]:
        """异步根据查询上下文压缩检索到的文档。

        Args:
            documents: 检索到的 `Document` 对象。
            query: 查询上下文。
            callbacks: 可选的 `Callbacks`，在压缩过程中运行。

        Returns:
            压缩后的文档。

        """
        return await run_in_executor(
            None, self.compress_documents, documents, query, callbacks
        )

from abc import ABC, abstractmethod

from langchain_core.runnables.config import run_in_executor
from pydantic import BaseModel, Field


class SparseVector(BaseModel, extra="forbid"):
    """稀疏向量结构。"""

    indices: list[int] = Field(..., description="indices must be unique")
    values: list[float] = Field(
        ..., description="values and indices must be the same length"
    )


class SparseEmbeddings(ABC):
    """用于 Qdrant 的稀疏嵌入模型的接口。"""

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> list[SparseVector]:
        """嵌入搜索文档。"""

    @abstractmethod
    def embed_query(self, text: str) -> SparseVector:
        """嵌入查询文本。"""

    async def aembed_documents(self, texts: list[str]) -> list[SparseVector]:
        """异步嵌入搜索文档。"""
        return await run_in_executor(None, self.embed_documents, texts)

    async def aembed_query(self, text: str) -> SparseVector:
        """异步嵌入查询文本。"""
        return await run_in_executor(None, self.embed_query, text)

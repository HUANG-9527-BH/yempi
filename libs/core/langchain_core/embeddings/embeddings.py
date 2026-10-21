"""**嵌入** 接口。"""

from abc import ABC, abstractmethod

from langchain_core.runnables.config import run_in_executor


class Embeddings(ABC):
    """嵌入模型的接口。

    此接口旨在用于实现文本嵌入模型。

    文本嵌入模型用于将文本映射到向量（n 维空间中的一个点）。

    相似的文本通常会被映射为该空间中彼此接近的点。什么是“相似”以及该空间中“距离”的度量方式，取决于具体的嵌入模型。

    This abstraction contains a method for embedding a list of documents and a method
    for embedding a query text. The embedding of a query text is expected to be a single
    vector, while the embedding of a list of documents is expected to be a list of
    vectors.

    通常情况下，查询嵌入与文档嵌入是相同的，但此抽象允许将它们独立处理。

    除了同步方法外，此接口还提供异步版本的方法。

    By default, the asynchronous methods are implemented using the synchronous methods;
    however, implementations may choose to override the asynchronous methods with
    an async native implementation for performance reasons.
    """

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """嵌入搜索文档。

        Args:
            texts: 待嵌入的文本列表。

        Returns:
            嵌入向量列表。
        """

    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        """嵌入查询文本。

        Args:
            text: 待嵌入的文本。

        Returns:
            嵌入向量。
        """

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        """异步嵌入搜索文档。

        Args:
            texts: 待嵌入的文本列表。

        Returns:
            嵌入向量列表。
        """
        return await run_in_executor(None, self.embed_documents, texts)

    async def aembed_query(self, text: str) -> list[float]:
        """异步嵌入查询文本。

        Args:
            text: 待嵌入的文本。

        Returns:
            嵌入向量。
        """
        return await run_in_executor(None, self.embed_query, text)

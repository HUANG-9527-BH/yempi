"""文档转换器。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from langchain_core.runnables.config import run_in_executor

if TYPE_CHECKING:
    from collections.abc import Sequence

    from langchain_core.documents import Document


class BaseDocumentTransformer(ABC):
    """文档转换的抽象基类。

    文档转换接收一个 `Document` 对象序列，并返回一个已转换的 `Document` 对象序列。

    Example:
        ```python
        class EmbeddingsRedundantFilter(BaseDocumentTransformer, BaseModel):
            embeddings: Embeddings
            similarity_fn: Callable = cosine_similarity
            similarity_threshold: float = 0.95

            class Config:
                arbitrary_types_allowed = True

            def transform_documents(
                self, documents: Sequence[Document], **kwargs: Any
            ) -> Sequence[Document]:
                stateful_documents = get_stateful_documents(documents)
                embedded_documents = _get_embeddings_from_stateful_docs(
                    self.embeddings, stateful_documents
                )
                included_idxs = _filter_similar_embeddings(
                    embedded_documents,
                    self.similarity_fn,
                    self.similarity_threshold,
                )
                return [stateful_documents[i] for i in sorted(included_idxs)]

            async def atransform_documents(
                self, documents: Sequence[Document], **kwargs: Any
            ) -> Sequence[Document]:
                raise NotImplementedError
        ```
    """

    @abstractmethod
    def transform_documents(
        self, documents: Sequence[Document], **kwargs: Any
    ) -> Sequence[Document]:
        """转换文档列表。

        Args:
            documents: 需要转换的 `Document` 对象序列。

        Returns:
            已转换的 `Document` 对象序列。
        """

    async def atransform_documents(
        self, documents: Sequence[Document], **kwargs: Any
    ) -> Sequence[Document]:
        """异步转换文档列表。

        Args:
            documents: 需要转换的 `Document` 对象序列。

        Returns:
            已转换的 `Document` 对象序列。
        """
        return await run_in_executor(
            None, self.transform_documents, documents, **kwargs
        )

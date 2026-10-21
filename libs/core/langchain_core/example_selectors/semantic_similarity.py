"""基于语义相似度选择示例的示例选择器。"""

from __future__ import annotations

from abc import ABC
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict

from langchain_core.example_selectors.base import BaseExampleSelector
from langchain_core.vectorstores import VectorStore

if TYPE_CHECKING:
    from langchain_core.documents import Document
    from langchain_core.embeddings import Embeddings


def sorted_values(values: dict[str, str]) -> list[str]:
    """按键排序返回字典中的值列表。

    Args:
        values: A dictionary with keys as input variables
            and values as their values.

    Returns:
        按键排序的字典值列表。
    """
    return [values[val] for val in sorted(values)]


class _VectorStoreExampleSelector(BaseExampleSelector, BaseModel, ABC):
    """基于语义相似度选择示例的示例选择器。"""

    vectorstore: VectorStore
    """包含示例信息的向量存储。"""

    k: int = 4
    """要选择的示例数量。"""

    example_keys: list[str] | None = None
    """可选的用于过滤示例的键。"""

    input_keys: list[str] | None = None
    """可选的用于过滤输入的键。如果提供，则搜索基于输入变量而非所有变量。"""

    vectorstore_kwargs: dict[str, Any] | None = None
    """传递给 `VectorStore` 的 `similarity_search` 函数的额外参数。"""

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="forbid",
    )

    @staticmethod
    def _example_to_text(example: dict[str, str], input_keys: list[str] | None) -> str:
        if input_keys:
            return " ".join(sorted_values({key: example[key] for key in input_keys}))
        return " ".join(sorted_values(example))

    def _documents_to_examples(self, documents: list[Document]) -> list[dict[str, Any]]:
                # 从元数据中获取示例。
                # 此假设示例存储在元数据中。
        examples = [dict(e.metadata) for e in documents]
                # 如果提供了示例键，则仅筛选这些键对应的示例。
        if self.example_keys:
            examples = [{k: eg[k] for k in self.example_keys} for eg in examples]
        return examples

    def add_example(self, example: dict[str, str]) -> str:
        """向向量存储中添加新示例。

        Args:
            example: A dictionary with keys as input variables
                and values as their values.

        Returns:
            添加的示例的 ID。
        """
        ids = self.vectorstore.add_texts(
            [self._example_to_text(example, self.input_keys)], metadatas=[example]
        )
        return ids[0]

    async def aadd_example(self, example: dict[str, str]) -> str:
        """异步向向量存储中添加新示例。

        Args:
            example: A dictionary with keys as input variables
                and values as their values.

        Returns:
            添加的示例的 ID。
        """
        ids = await self.vectorstore.aadd_texts(
            [self._example_to_text(example, self.input_keys)], metadatas=[example]
        )
        return ids[0]


class SemanticSimilarityExampleSelector(_VectorStoreExampleSelector):
    """基于语义相似度选择示例。"""

    def select_examples(self, input_variables: dict[str, str]) -> list[dict[str, Any]]:
        """基于语义相似度选择示例。

        Args:
            input_variables: 用于搜索的输入变量。

        Returns:
            选定的示例。
        """
                # 获取相似度最高的文档。
        vectorstore_kwargs = self.vectorstore_kwargs or {}
        example_docs = self.vectorstore.similarity_search(
            self._example_to_text(input_variables, self.input_keys),
            k=self.k,
            **vectorstore_kwargs,
        )
        return self._documents_to_examples(example_docs)

    async def aselect_examples(
        self, input_variables: dict[str, str]
    ) -> list[dict[str, Any]]:
        """异步基于语义相似度选择示例。

        Args:
            input_variables: 用于搜索的输入变量。

        Returns:
            选定的示例。
        """
                # 获取相似度最高的文档。
        vectorstore_kwargs = self.vectorstore_kwargs or {}
        example_docs = await self.vectorstore.asimilarity_search(
            self._example_to_text(input_variables, self.input_keys),
            k=self.k,
            **vectorstore_kwargs,
        )
        return self._documents_to_examples(example_docs)

    @classmethod
    def from_examples(
        cls,
        examples: list[dict[str, str]],
        embeddings: Embeddings,
        vectorstore_cls: type[VectorStore],
        k: int = 4,
        input_keys: list[str] | None = None,
        *,
        example_keys: list[str] | None = None,
        vectorstore_kwargs: dict[str, Any] | None = None,
        **vectorstore_cls_kwargs: Any,
    ) -> SemanticSimilarityExampleSelector:
        """使用示例列表和嵌入创建 k-shot 示例选择器。

        根据查询相似度动态重新排列示例。

        Args:
            examples: List of examples to use in the prompt.
            embeddings: An initialized embedding API interface, e.g. OpenAIEmbeddings().
            vectorstore_cls: A vector store DB interface class, e.g. FAISS.
            k: Number of examples to select.
            input_keys: If provided, the search is based on the input variables
                instead of all variables.
            example_keys: If provided, keys to filter examples to.
            vectorstore_kwargs: Extra arguments passed to similarity_search function
                of the `VectorStore`.
            vectorstore_cls_kwargs: optional kwargs containing url for vector store

        Returns:
            由向量存储支持的实例化 ExampleSelector。
        """
        string_examples = [cls._example_to_text(eg, input_keys) for eg in examples]
        vectorstore = vectorstore_cls.from_texts(
            string_examples, embeddings, metadatas=examples, **vectorstore_cls_kwargs
        )
        return cls(
            vectorstore=vectorstore,
            k=k,
            input_keys=input_keys,
            example_keys=example_keys,
            vectorstore_kwargs=vectorstore_kwargs,
        )

    @classmethod
    async def afrom_examples(
        cls,
        examples: list[dict[str, str]],
        embeddings: Embeddings,
        vectorstore_cls: type[VectorStore],
        k: int = 4,
        input_keys: list[str] | None = None,
        *,
        example_keys: list[str] | None = None,
        vectorstore_kwargs: dict[str, Any] | None = None,
        **vectorstore_cls_kwargs: Any,
    ) -> SemanticSimilarityExampleSelector:
        """异步使用示例列表和嵌入创建 k-shot 示例选择器。

        根据查询相似度动态重新排列示例。

        Args:
            examples: List of examples to use in the prompt.
            embeddings: An initialized embedding API interface, e.g. OpenAIEmbeddings().
            vectorstore_cls: A vector store DB interface class, e.g. FAISS.
            k: Number of examples to select.
            input_keys: If provided, the search is based on the input variables
                instead of all variables.
            example_keys: If provided, keys to filter examples to.
            vectorstore_kwargs: Extra arguments passed to similarity_search function
                of the `VectorStore`.
            vectorstore_cls_kwargs: optional kwargs containing url for vector store

        Returns:
            基于向量存储实例化的 ExampleSelector。
        """
        string_examples = [cls._example_to_text(eg, input_keys) for eg in examples]
        vectorstore = await vectorstore_cls.afrom_texts(
            string_examples, embeddings, metadatas=examples, **vectorstore_cls_kwargs
        )
        return cls(
            vectorstore=vectorstore,
            k=k,
            input_keys=input_keys,
            example_keys=example_keys,
            vectorstore_kwargs=vectorstore_kwargs,
        )


class MaxMarginalRelevanceExampleSelector(_VectorStoreExampleSelector):
    """基于最大边际相关性选择示例。

    该方法在以下论文中被证明能提升性能：
    https://arxiv.org/pdf/2211.13892.pdf
    """

    fetch_k: int = 20
    """用于重新排序的要获取的示例数量。"""

    def select_examples(self, input_variables: dict[str, str]) -> list[dict[str, Any]]:
        """基于最大边际相关性（Max Marginal Relevance）选择示例。

        Args:
            input_variables: 用于搜索的输入变量。

        Returns:
            选定的示例。
        """
        example_docs = self.vectorstore.max_marginal_relevance_search(
            self._example_to_text(input_variables, self.input_keys),
            k=self.k,
            fetch_k=self.fetch_k,
        )
        return self._documents_to_examples(example_docs)

    async def aselect_examples(
        self, input_variables: dict[str, str]
    ) -> list[dict[str, Any]]:
        """异步基于最大边际相关性（Max Marginal Relevance）选择示例。

        Args:
            input_variables: 用于搜索的输入变量。

        Returns:
            选定的示例。
        """
        example_docs = await self.vectorstore.amax_marginal_relevance_search(
            self._example_to_text(input_variables, self.input_keys),
            k=self.k,
            fetch_k=self.fetch_k,
        )
        return self._documents_to_examples(example_docs)

    @classmethod
    def from_examples(
        cls,
        examples: list[dict[str, str]],
        embeddings: Embeddings,
        vectorstore_cls: type[VectorStore],
        k: int = 4,
        input_keys: list[str] | None = None,
        fetch_k: int = 20,
        example_keys: list[str] | None = None,
        vectorstore_kwargs: dict[str, Any] | None = None,
        **vectorstore_cls_kwargs: Any,
    ) -> MaxMarginalRelevanceExampleSelector:
        """使用示例列表和嵌入创建 k-shot 示例选择器。

        基于最大边际相关性动态重新排列示例。

        Args:
            examples: List of examples to use in the prompt.
            embeddings: An initialized embedding API interface, e.g. OpenAIEmbeddings().
            vectorstore_cls: A vector store DB interface class, e.g. FAISS.
            k: Number of examples to select.
            fetch_k: Number of `Document` objects to fetch to pass to MMR algorithm.
            input_keys: If provided, the search is based on the input variables
                instead of all variables.
            example_keys: If provided, keys to filter examples to.
            vectorstore_kwargs: Extra arguments passed to similarity_search function
                of the `VectorStore`.
            vectorstore_cls_kwargs: optional kwargs containing url for vector store

        Returns:
            基于向量存储实例化的 ExampleSelector。
        """
        string_examples = [cls._example_to_text(eg, input_keys) for eg in examples]
        vectorstore = vectorstore_cls.from_texts(
            string_examples, embeddings, metadatas=examples, **vectorstore_cls_kwargs
        )
        return cls(
            vectorstore=vectorstore,
            k=k,
            fetch_k=fetch_k,
            input_keys=input_keys,
            example_keys=example_keys,
            vectorstore_kwargs=vectorstore_kwargs,
        )

    @classmethod
    async def afrom_examples(
        cls,
        examples: list[dict[str, str]],
        embeddings: Embeddings,
        vectorstore_cls: type[VectorStore],
        *,
        k: int = 4,
        input_keys: list[str] | None = None,
        fetch_k: int = 20,
        example_keys: list[str] | None = None,
        vectorstore_kwargs: dict[str, Any] | None = None,
        **vectorstore_cls_kwargs: Any,
    ) -> MaxMarginalRelevanceExampleSelector:
        """使用示例列表和嵌入创建 k-shot 示例选择器。

        基于最大边际相关性动态重新排列示例。

        Args:
            examples: List of examples to use in the prompt.
            embeddings: An initialized embedding API interface, e.g. OpenAIEmbeddings().
            vectorstore_cls: A vector store DB interface class, e.g. FAISS.
            k: Number of examples to select.
            fetch_k: Number of `Document` objects to fetch to pass to MMR algorithm.
            input_keys: If provided, the search is based on the input variables
                instead of all variables.
            example_keys: If provided, keys to filter examples to.
            vectorstore_kwargs: Extra arguments passed to similarity_search function
                of the `VectorStore`.
            vectorstore_cls_kwargs: optional kwargs containing url for vector store

        Returns:
            基于向量存储实例化的 ExampleSelector。
        """
        string_examples = [cls._example_to_text(eg, input_keys) for eg in examples]
        vectorstore = await vectorstore_cls.afrom_texts(
            string_examples, embeddings, metadatas=examples, **vectorstore_cls_kwargs
        )
        return cls(
            vectorstore=vectorstore,
            k=k,
            fetch_k=fetch_k,
            input_keys=input_keys,
            example_keys=example_keys,
            vectorstore_kwargs=vectorstore_kwargs,
        )

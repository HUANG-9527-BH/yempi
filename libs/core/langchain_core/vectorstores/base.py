"""向量存储用于保存嵌入数据并执行向量搜索。

存储和搜索非结构化数据最常见的方式之一是
将其嵌入并存储生成的嵌入向量，然后查询存储
并检索与嵌入查询'最相似'的数据。
"""

from __future__ import annotations

import logging
import math
import warnings
from abc import ABC, abstractmethod
from itertools import cycle
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    TypeVar,
)

from pydantic import ConfigDict, Field, model_validator
from typing_extensions import Self, override

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever, LangSmithRetrieverParams
from langchain_core.runnables.config import run_in_executor

if TYPE_CHECKING:
    from collections.abc import Callable, Collection, Iterable, Iterator, Sequence

    from langchain_core.callbacks.manager import (
        AsyncCallbackManagerForRetrieverRun,
        CallbackManagerForRetrieverRun,
    )

logger = logging.getLogger(__name__)

VST = TypeVar("VST", bound="VectorStore")


class VectorStore(ABC):
    """向量存储的接口。"""

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: list[dict[str, Any]] | None = None,
        *,
        ids: list[str] | None = None,
        **kwargs: Any,
    ) -> list[str]:
        """将更多文本通过嵌入模型处理并添加到 `VectorStore` 中。

        Args:
            texts: 要添加到 `VectorStore` 的字符串可迭代对象。
            metadatas: 与文本关联的可选元数据列表。
            ids: 与文本关联的可选 ID 列表。
            **kwargs: `VectorStore` 特定的参数。

                One of the kwargs should be `ids` which is a list of ids
                associated with the texts.

        Returns:
            添加文本到 `VectorStore` 后返回的 ID 列表。

        Raises:
            ValueError: 如果元数据数量与文本数量不匹配。
            ValueError: 如果 ID 数量与文本数量不匹配。
        """
        if type(self).add_documents != VectorStore.add_documents:
                        # 此条件在子类提供了
                        # upsert 方法的实现时触发。
                        # 现有的 add_texts
            texts_: Sequence[str] = (
                texts if isinstance(texts, (list, tuple)) else list(texts)
            )
            if metadatas and len(metadatas) != len(texts_):
                msg = (
                    "The number of metadatas must match the number of texts."
                    f"Got {len(metadatas)} metadatas and {len(texts_)} texts."
                )
                raise ValueError(msg)
            metadatas_ = iter(metadatas) if metadatas else cycle([{}])
            ids_: Iterator[str | None] = iter(ids) if ids else cycle([None])
            docs = [
                Document(id=id_, page_content=text, metadata=metadata_)
                for text, metadata_, id_ in zip(texts, metadatas_, ids_, strict=False)
            ]
            if ids is not None:
                                # 为了向后兼容
                kwargs["ids"] = ids

            return self.add_documents(docs, **kwargs)
        msg = f"`add_texts` has not been implemented for {self.__class__.__name__} "
        raise NotImplementedError(msg)

    @property
    def embeddings(self) -> Embeddings | None:
        """如果可用，访问查询嵌入对象。"""
        logger.debug(
            "The embeddings property has not been implemented for %s",
            self.__class__.__name__,
        )
        return None

    def delete(self, ids: list[str] | None = None, **kwargs: Any) -> bool | None:
        """根据向量 ID 或其他标准删除数据。

        Args:
            ids: 要删除的 ID 列表。如果为 `None`，则删除全部。
            **kwargs: 子类可能使用的其他关键字参数。

        Returns:
            `True` if deletion is successful, `False` otherwise, `None` if not
                implemented.
        """
        msg = "delete method must be implemented by subclass."
        raise NotImplementedError(msg)

    def get_by_ids(self, ids: Sequence[str], /) -> list[Document]:
        """根据 ID 获取文档。

        返回的文档应将 ID 字段设置为向量存储中该文档的 ID。

        Fewer documents may be returned than requested if some IDs are not found or
        if there are duplicated IDs.

        用户不应假设返回文档的顺序与输入 ID 的顺序一致。相反，用户应依赖返回文档的 ID
        字段。

        This method should **NOT** raise exceptions if no documents are found for
        some IDs.

        Args:
            ids: 要检索的 ID 列表。

        Returns:
            `Document` 对象列表。
        """
        msg = f"{self.__class__.__name__} does not yet support get_by_ids."
        raise NotImplementedError(msg)

        # 实现应重写此方法以提供原生异步版本。
    async def aget_by_ids(self, ids: Sequence[str], /) -> list[Document]:
        """异步根据 ID 获取文档。

        返回的文档应将 ID 字段设置为向量存储中该文档的 ID。

        Fewer documents may be returned than requested if some IDs are not found or
        if there are duplicated IDs.

        用户不应假设返回文档的顺序与输入 ID 的顺序一致。相反，用户应依赖返回文档的 ID
        字段。

        This method should **NOT** raise exceptions if no documents are found for
        some IDs.

        Args:
            ids: 要检索的 ID 列表。

        Returns:
            `Document` 对象列表。
        """
        return await run_in_executor(None, self.get_by_ids, ids)

    async def adelete(self, ids: list[str] | None = None, **kwargs: Any) -> bool | None:
        """异步根据向量 ID 或其他标准删除数据。

        Args:
            ids: 要删除的 ID 列表。如果为 `None`，则删除全部。
            **kwargs: 子类可能使用的其他关键字参数。

        Returns:
            `True` if deletion is successful, `False` otherwise, `None` if not
                implemented.
        """
        return await run_in_executor(None, self.delete, ids, **kwargs)

    async def aadd_texts(
        self,
        texts: Iterable[str],
        metadatas: list[dict[str, Any]] | None = None,
        *,
        ids: list[str] | None = None,
        **kwargs: Any,
    ) -> list[str]:
        """异步运行更多文本通过嵌入并添加到 `VectorStore`。

        Args:
            texts: 要添加到 `VectorStore` 的字符串可迭代对象。
            metadatas: 与文本关联的可选元数据列表。
            ids: 可选列表
            **kwargs: `VectorStore` 特定参数。

        Returns:
            将文本添加到 `VectorStore` 后的 ID 列表。

        Raises:
            ValueError: 如果元数据数量与文本数量不匹配。
            ValueError: 如果 ID 数量与文本数量不匹配。
        """
        if ids is not None:
                        # 为了向后兼容
            kwargs["ids"] = ids
        if type(self).aadd_documents != VectorStore.aadd_documents:
                        # 此条件在子类提供了
                        # upsert 方法的实现时触发。
                        # 现有的 add_texts
            texts_: Sequence[str] = (
                texts if isinstance(texts, (list, tuple)) else list(texts)
            )
            if metadatas and len(metadatas) != len(texts_):
                msg = (
                    "The number of metadatas must match the number of texts."
                    f"Got {len(metadatas)} metadatas and {len(texts_)} texts."
                )
                raise ValueError(msg)
            metadatas_ = iter(metadatas) if metadatas else cycle([{}])
            ids_: Iterator[str | None] = iter(ids) if ids else cycle([None])

            docs = [
                Document(id=id_, page_content=text, metadata=metadata_)
                for text, metadata_, id_ in zip(texts, metadatas_, ids_, strict=False)
            ]
            return await self.aadd_documents(docs, **kwargs)
        return await run_in_executor(None, self.add_texts, texts, metadatas, **kwargs)

    def add_documents(self, documents: list[Document], **kwargs: Any) -> list[str]:
        """向 `VectorStore` 中添加或更新文档。

        Args:
            documents: 要添加到 `VectorStore` 的文档。
            **kwargs: 额外的关键字参数。

                如果 kwargs 包含 IDs 且文档包含 ids，则 kwargs 中的 IDs 优先级更高。

        Returns:
            添加文本的 ID 列表。
        """
        if type(self).add_texts != VectorStore.add_texts:
            if "ids" not in kwargs:
                ids = [doc.id for doc in documents]

                                # 如果至少有一个有效 ID，我们将假设应使用 ID。
                                # 如果异步方法已被重写，我们将使用该方法。
                if any(ids):
                    kwargs["ids"] = ids

            texts = [doc.page_content for doc in documents]
            metadatas = [doc.metadata for doc in documents]
            return self.add_texts(texts, metadatas, **kwargs)
        msg = (
            f"`add_documents` and `add_texts` has not been implemented "
            f"for {self.__class__.__name__} "
        )
        raise NotImplementedError(msg)

    async def aadd_documents(
        self, documents: list[Document], **kwargs: Any
    ) -> list[str]:
        """异步运行更多文档通过嵌入并添加到 `VectorStore`。

        Args:
            documents: 要添加到 `VectorStore` 的文档。
            **kwargs: 额外的关键字参数。

        Returns:
            添加文本的 ID 列表。
        """
                # 如果至少有一个有效 ID，我们将假设应使用 ID。
        if type(self).aadd_texts != VectorStore.aadd_texts:
            if "ids" not in kwargs:
                ids = [doc.id for doc in documents]

                                # 正确的相关性函数
                                # 可能因多种因素而异，包括：
                if any(ids):
                    kwargs["ids"] = ids

            texts = [doc.page_content for doc in documents]
            metadatas = [doc.metadata for doc in documents]
            return await self.aadd_texts(texts, metadatas, **kwargs)

        return await run_in_executor(None, self.add_documents, documents, **kwargs)

    def search(self, query: str, search_type: str, **kwargs: Any) -> list[Document]:
        """使用指定搜索类型返回与查询最相似的文档。

        Args:
            query: 输入文本。
            search_type: 要执行的搜索类型。

                可以是 `'similarity'`、`'mmr'` 或 `'similarity_score_threshold'`。
            **kwargs: 传递给搜索方法的参数。

        Returns:
            与查询最相似的 `Document` 对象列表。

        Raises:
            ValueError: 如果 `search_type` 不是 `'similarity'`、
                `'mmr'` 或 `'similarity_score_threshold'` 之一。
        """
        if search_type == "similarity":
            return self.similarity_search(query, **kwargs)
        if search_type == "similarity_score_threshold":
            docs_and_similarities = self.similarity_search_with_relevance_scores(
                query, **kwargs
            )
            return [doc for doc, _ in docs_and_similarities]
        if search_type == "mmr":
            return self.max_marginal_relevance_search(query, **kwargs)
        msg = (
            f"search_type of {search_type} not allowed. Expected "
            "search_type to be 'similarity', 'similarity_score_threshold'"
            " or 'mmr'."
        )
        raise ValueError(msg)

    async def asearch(
        self, query: str, search_type: str, **kwargs: Any
    ) -> list[Document]:
        """异步返回使用指定搜索类型与查询最相似的文档。

        Args:
            query: 输入文本。
            search_type: 要执行的搜索类型。

                可以是 `'similarity'`、`'mmr'` 或 `'similarity_score_threshold'`。
            **kwargs: 传递给搜索方法的参数。

        Returns:
            与查询最相似的 `Document` 对象列表。

        Raises:
            ValueError: 如果 `search_type` 不是 `'similarity'`、
                `'mmr'` 或 `'similarity_score_threshold'` 之一。
        """
        if search_type == "similarity":
            return await self.asimilarity_search(query, **kwargs)
        if search_type == "similarity_score_threshold":
            docs_and_similarities = await self.asimilarity_search_with_relevance_scores(
                query, **kwargs
            )
            return [doc for doc, _ in docs_and_similarities]
        if search_type == "mmr":
            return await self.amax_marginal_relevance_search(query, **kwargs)
        msg = (
            f"search_type of {search_type} not allowed. Expected "
            "search_type to be 'similarity', 'similarity_score_threshold' or 'mmr'."
        )
        raise ValueError(msg)

    @abstractmethod
    def similarity_search(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> list[Document]:
        """返回与查询最相似的文档。

        Args:
            query: 输入文本。
            k: 要返回的 `Document` 对象数量。
            **kwargs: 传递给搜索方法的参数。

        Returns:
            与查询最相似的 `Document` 对象列表。
        """

    @staticmethod
    def _euclidean_relevance_score_fn(distance: float) -> float:
        """返回在 [0, 1] 范围内的相似度分数。"""
                # - VectorStore 使用的距离/相似性度量
                # - 嵌入的尺度（OpenAI 的嵌入是单位范数。许多其他嵌入不是！）
                # - 嵌入维度
                # - 等等。
                #  此函数将归一化嵌入的欧几里得范数
                # (0 表示最相似，sqrt(2) 表示最不相似)
                # 转换为相似性函数（0 到 1）
                # 这是临时解决方案，用于使相似性搜索支持异步。正确的方法是在向量存储实现中使相似性搜索异步。
                # 这是临时解决方案，用于使相似性搜索支持异步。正确的方法是在向量存储实现中使相似性搜索异步。
                # 这是临时解决方案，用于使相似性搜索支持异步。正确的方法是在向量存储实现中使相似性搜索异步。
        return 1.0 - distance / math.sqrt(2)

    @staticmethod
    def _cosine_relevance_score_fn(distance: float) -> float:
        """将距离归一化为 [0, 1] 范围内的分数。"""
        return 1.0 - distance

    @staticmethod
    def _max_inner_product_relevance_score_fn(distance: float) -> float:
        """将距离归一化为 [0, 1] 范围内的分数。"""
        if distance > 0:
            return 1.0 - distance

        return -1.0 * distance

    def _select_relevance_score_fn(self) -> Callable[[float], float]:
        """正确的相关性函数。

        可能因以下因素而有所不同：

        - `VectorStore` 使用的距离/相似度度量
        - 嵌入的尺度（OpenAI 的是单位范数。许多其他模型不是！）
        - 嵌入维度
        - 等等。

        向量存储应定义其自身的基于选择的相关性方法。
        """
        raise NotImplementedError

    def similarity_search_with_score(
        self, *args: Any, **kwargs: Any
    ) -> list[tuple[Document, float]]:
        """以距离进行相似度搜索运行。

        Args:
            *args: 传递给搜索方法的参数。
            **kwargs: 传递给搜索方法的参数。

        Returns:
            包含 `(doc, similarity_score)` 元组的列表。
        """
        raise NotImplementedError

    async def asimilarity_search_with_score(
        self, *args: Any, **kwargs: Any
    ) -> list[tuple[Document, float]]:
        """异步运行相似度搜索并返回距离。

        Args:
            *args: 传递给搜索方法的参数。
            **kwargs: 传递给搜索方法的参数。

        Returns:
            包含 `(doc, similarity_score)` 元组的列表。
        """
                # 这是临时解决方案，用于使相似性搜索支持异步。正确的方法是在向量存储实现中使相似性搜索异步。
                # 这是临时解决方案，用于使相似性搜索支持异步。正确的方法是在向量存储实现中使相似性搜索异步。
                # 向量存储实现中的异步操作。
        return await run_in_executor(
            None, self.similarity_search_with_score, *args, **kwargs
        )

    def _similarity_search_with_relevance_scores(
        self,
        query: str,
        k: int = 4,
        **kwargs: Any,
    ) -> list[tuple[Document, float]]:
        """默认的相似度搜索，返回相关性得分。

        可在子类中根据需要修改。
        返回文档及其相关性得分，范围为 `[0, 1]`。

        `0` 表示不相似，`1` 表示最相似。

        Args:
            query: 输入文本。
            k: 返回的 `Document` 对象数量。
            **kwargs: 传递给相似度搜索的参数。

                应包含 `score_threshold`，一个可选的浮点值，
                范围在 `0` 到 `1` 之间，用于过滤检索到的文档集合。

        Returns:
            包含 `(doc, similarity_score)` 的元组列表
        """
        relevance_score_fn = self._select_relevance_score_fn()
        docs_and_scores = self.similarity_search_with_score(query, k, **kwargs)
        return [(doc, relevance_score_fn(score)) for doc, score in docs_and_scores]

    async def _asimilarity_search_with_relevance_scores(
        self,
        query: str,
        k: int = 4,
        **kwargs: Any,
    ) -> list[tuple[Document, float]]:
        """默认的相似度搜索，返回相关性得分。

        可在子类中根据需要修改。
        返回文档及其相关性得分，范围为 `[0, 1]`。

        `0` 表示不相似，`1` 表示最相似。

        Args:
            query: 输入文本。
            k: 返回的 `Document` 对象数量。
            **kwargs: 传递给相似度搜索的参数。

                应包含 `score_threshold`，一个可选的浮点值，
                范围在 `0` 到 `1` 之间，用于过滤检索到的文档集合。

        Returns:
            包含 `(doc, similarity_score)` 的元组列表
        """
        relevance_score_fn = self._select_relevance_score_fn()
        docs_and_scores = await self.asimilarity_search_with_score(query, k, **kwargs)
        return [(doc, relevance_score_fn(score)) for doc, score in docs_and_scores]

    def similarity_search_with_relevance_scores(
        self,
        query: str,
        k: int = 4,
        **kwargs: Any,
    ) -> list[tuple[Document, float]]:
        """返回文档及其相关性得分，范围为 `[0, 1]`。

        `0` 表示不相似，`1` 表示最相似。

        Args:
            query: 输入文本。
            k: 返回的 `Document` 对象数量。
            **kwargs: 传递给相似度搜索的参数。

                应包含 `score_threshold`，一个可选的浮点值，
                范围在 `0` 到 `1` 之间，用于过滤检索到的文档集合。

        Returns:
            包含 `(doc, similarity_score)` 的元组列表。
        """
        score_threshold = kwargs.pop("score_threshold", None)

        docs_and_similarities = self._similarity_search_with_relevance_scores(
            query, k=k, **kwargs
        )
        if any(
            similarity < 0.0 or similarity > 1.0
            for _, similarity in docs_and_similarities
        ):
            warnings.warn(
                "Relevance scores must be between"
                f" 0 and 1, got {docs_and_similarities}",
                stacklevel=2,
            )

        if score_threshold is not None:
            docs_and_similarities = [
                (doc, similarity)
                for doc, similarity in docs_and_similarities
                if similarity >= score_threshold
            ]
            if len(docs_and_similarities) == 0:
                logger.warning(
                    "No relevant docs were retrieved using the "
                    "relevance score threshold %s",
                    score_threshold,
                )
        return docs_and_similarities

    async def asimilarity_search_with_relevance_scores(
        self,
        query: str,
        k: int = 4,
        **kwargs: Any,
    ) -> list[tuple[Document, float]]:
        """异步返回文档及其相关性得分，范围为 `[0, 1]`。

        `0` 表示不相似，`1` 表示最相似。

        Args:
            query: 输入文本。
            k: 返回的 `Document` 对象数量。
            **kwargs: 传递给相似度搜索的参数。

                应包含 `score_threshold`，一个可选的浮点值，
                范围在 `0` 到 `1` 之间，用于过滤检索到的文档集合。

        Returns:
            包含 `(doc, similarity_score)` 的元组列表
        """
        score_threshold = kwargs.pop("score_threshold", None)

        docs_and_similarities = await self._asimilarity_search_with_relevance_scores(
            query, k=k, **kwargs
        )
        if any(
            similarity < 0.0 or similarity > 1.0
            for _, similarity in docs_and_similarities
        ):
            warnings.warn(
                "Relevance scores must be between"
                f" 0 and 1, got {docs_and_similarities}",
                stacklevel=2,
            )

        if score_threshold is not None:
            docs_and_similarities = [
                (doc, similarity)
                for doc, similarity in docs_and_similarities
                if similarity >= score_threshold
            ]
            if len(docs_and_similarities) == 0:
                logger.warning(
                    "No relevant docs were retrieved using the "
                    "relevance score threshold %s",
                    score_threshold,
                )
        return docs_and_similarities

    async def asimilarity_search(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> list[Document]:
        """异步返回与查询最相似的文档。

        Args:
            query: 输入文本。
            k: 返回的 `Document` 对象数量。
            **kwargs: 传递给搜索方法的参数。

        Returns:
            与查询最相似的 `Document` 对象列表。
        """
                # 这是一个临时解决方案，用于实现相似性搜索
                # 异步。正确解决方案是让相似度搜索
                # 向量存储实现中的异步操作。
        return await run_in_executor(None, self.similarity_search, query, k=k, **kwargs)

    def similarity_search_by_vector(
        self, embedding: list[float], k: int = 4, **kwargs: Any
    ) -> list[Document]:
        """返回与嵌入向量最相似的文档。

        Args:
            embedding: 用于查找相似文档的嵌入向量。
            k: 返回的 `Document` 对象数量。
            **kwargs: 传递给搜索方法的参数。

        Returns:
            与查询向量最相似的 `Document` 对象列表。
        """
        raise NotImplementedError

    async def asimilarity_search_by_vector(
        self, embedding: list[float], k: int = 4, **kwargs: Any
    ) -> list[Document]:
        """异步返回与嵌入向量最相似的文档。

        Args:
            embedding: 用于查找相似文档的嵌入。
            k: 返回的 `Document` 对象数量。
            **kwargs: 传递给搜索方法的参数。

        Returns:
            与查询向量最相似的 `Document` 对象列表。
        """
                # 这是临时的权宜之计，用于实现相似度搜索
                # 异步。正确的解决方案是让相似度搜索
                # 向量存储实现中的异步操作。
        return await run_in_executor(
            None, self.similarity_search_by_vector, embedding, k=k, **kwargs
        )

    def max_marginal_relevance_search(
        self,
        query: str,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        **kwargs: Any,
    ) -> list[Document]:
        """返回使用最大边际相关性选择的文档。

        Maximal marginal relevance optimizes for similarity to query AND diversity
        among selected documents.

        Args:
            query: Text to look up documents similar to.
            k: Number of `Document` objects to return.
            fetch_k: Number of `Document` objects to fetch to pass to MMR algorithm.
            lambda_mult: Number between `0` and `1` that determines the degree of
                diversity among the results with `0` corresponding to maximum diversity
                and `1` to minimum diversity.
            **kwargs: Arguments to pass to the search method.

        Returns:
            通过最大边际相关性选择的 `Document` 对象列表。
        """
        raise NotImplementedError

    async def amax_marginal_relevance_search(
        self,
        query: str,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        **kwargs: Any,
    ) -> list[Document]:
        """异步返回通过最大边际相关性选择的文档。

        Maximal marginal relevance optimizes for similarity to query AND diversity
        among selected documents.

        Args:
            query: Text to look up documents similar to.
            k: Number of `Document` objects to return.
            fetch_k: Number of `Document` objects to fetch to pass to MMR algorithm.
            lambda_mult: Number between `0` and `1` that determines the degree of
                diversity among the results with `0` corresponding to maximum diversity
                and `1` to minimum diversity.
            **kwargs: Arguments to pass to the search method.

        Returns:
            通过最大边际相关性选择的 `Document` 对象列表。
        """
                # 这是一个临时解决方案，用于实现相似性搜索
                # 异步。正确的解决方案是让相似度搜索
                # 向量存储实现中的异步操作。
        return await run_in_executor(
            None,
            self.max_marginal_relevance_search,
            query,
            k=k,
            fetch_k=fetch_k,
            lambda_mult=lambda_mult,
            **kwargs,
        )

    def max_marginal_relevance_search_by_vector(
        self,
        embedding: list[float],
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        **kwargs: Any,
    ) -> list[Document]:
        """返回使用最大边际相关性选择的文档。

        Maximal marginal relevance optimizes for similarity to query AND diversity
        among selected documents.

        Args:
            embedding: Embedding to look up documents similar to.
            k: Number of `Document` objects to return.
            fetch_k: Number of `Document` objects to fetch to pass to MMR algorithm.
            lambda_mult: Number between `0` and `1` that determines the degree of
                diversity among the results with `0` corresponding to maximum diversity
                and `1` to minimum diversity.
            **kwargs: Arguments to pass to the search method.

        Returns:
            通过最大边际相关性选择的 `Document` 对象列表。
        """
        raise NotImplementedError

    async def amax_marginal_relevance_search_by_vector(
        self,
        embedding: list[float],
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        **kwargs: Any,
    ) -> list[Document]:
        """异步返回通过最大边际相关性选择的文档。

        Maximal marginal relevance optimizes for similarity to query AND diversity
        among selected documents.

        Args:
            embedding: Embedding to look up documents similar to.
            k: Number of `Document` objects to return.
            fetch_k: Number of `Document` objects to fetch to pass to MMR algorithm.
            lambda_mult: Number between `0` and `1` that determines the degree of
                diversity among the results with `0` corresponding to maximum diversity
                and `1` to minimum diversity.
            **kwargs: Arguments to pass to the search method.

        Returns:
            通过最大边际相关性选择的 `Document` 对象列表。
        """
        return await run_in_executor(
            None,
            self.max_marginal_relevance_search_by_vector,
            embedding,
            k=k,
            fetch_k=fetch_k,
            lambda_mult=lambda_mult,
            **kwargs,
        )

    @classmethod
    def from_documents(
        cls,
        documents: list[Document],
        embedding: Embeddings,
        **kwargs: Any,
    ) -> Self:
        """返回从文档和嵌入初始化的 `VectorStore`。

        Args:
            documents: 要添加到 `VectorStore` 的 `Document` 对象列表。
            embedding: 使用的嵌入函数。
            **kwargs: 额外的关键字参数。

        Returns:
            从文档和嵌入初始化的 `VectorStore`。
        """
        texts = [d.page_content for d in documents]
        metadatas = [d.metadata for d in documents]

        if "ids" not in kwargs:
            ids = [doc.id for doc in documents]

                        # 如果有至少一个有效 ID，我们将假设应使用 ID。
                        # 如果有至少一个有效 ID，我们将假设应使用 ID。
            if any(ids):
                kwargs["ids"] = ids

        return cls.from_texts(texts, embedding, metadatas=metadatas, **kwargs)

    @classmethod
    async def afrom_documents(
        cls,
        documents: list[Document],
        embedding: Embeddings,
        **kwargs: Any,
    ) -> Self:
        """异步返回从文档和嵌入初始化的 `VectorStore`。

        Args:
            documents: 要添加到 `VectorStore` 的 `Document` 对象列表。
            embedding: 使用的嵌入函数。
            **kwargs: 额外的关键字参数。

        Returns:
            从文档和嵌入初始化的 `VectorStore`。
        """
        texts = [d.page_content for d in documents]
        metadatas = [d.metadata for d in documents]

        if "ids" not in kwargs:
            ids = [doc.id for doc in documents]

                        # 如果有至少一个有效 ID，我们将假设 IDs
                        # 应被使用。
            if any(ids):
                kwargs["ids"] = ids

        return await cls.afrom_texts(texts, embedding, metadatas=metadatas, **kwargs)

    @classmethod
    @abstractmethod
    def from_texts(
        cls: type[VST],
        texts: list[str],
        embedding: Embeddings,
        metadatas: list[dict[str, Any]] | None = None,
        *,
        ids: list[str] | None = None,
        **kwargs: Any,
    ) -> VST:
        """返回从文本和嵌入初始化的 `VectorStore`。

        Args:
            texts: 要添加到 `VectorStore` 的文本。
            embedding: 使用的嵌入函数。
            metadatas: 与文本关联的可选元数据列表。
            ids: 与文本关联的可选 ID 列表。
            **kwargs: 额外的关键字参数。

        Returns:
            从文本和嵌入初始化的 `VectorStore`。
        """

    @classmethod
    async def afrom_texts(
        cls,
        texts: list[str],
        embedding: Embeddings,
        metadatas: list[dict[str, Any]] | None = None,
        *,
        ids: list[str] | None = None,
        **kwargs: Any,
    ) -> Self:
        """异步返回从文本和嵌入初始化的 `VectorStore`。

        Args:
            texts: 要添加到 `VectorStore` 的文本。
            embedding: 使用的嵌入函数。
            metadatas: 与文本关联的可选元数据列表。
            ids: 与文本关联的可选 ID 列表。
            **kwargs: 额外的关键字参数。

        Returns:
            从文本和嵌入初始化的 `VectorStore`。
        """
        if ids is not None:
            kwargs["ids"] = ids
        return await run_in_executor(
            None, cls.from_texts, texts, embedding, metadatas, **kwargs
        )

    def _get_retriever_tags(self) -> list[str]:
        """获取检索器的标签。"""
        tags = [self.__class__.__name__]
        if self.embeddings:
            tags.append(self.embeddings.__class__.__name__)
        return tags

    def as_retriever(self, **kwargs: Any) -> VectorStoreRetriever:
        """返回从此 `VectorStore` 初始化的 `VectorStoreRetriever`。

        Args:
            **kwargs: 传递给搜索函数的关键字参数。

                可包含：

                * `search_type`: 定义检索器应执行的搜索类型。可以是
                `'similarity'`（默认）、`'mmr'` 或 `'similarity_score_threshold'`。 *
                `search_kwargs`: 传递给搜索函数的关键字参数。

                    可包含如下内容：

                    * `k`: Amount of documents to return (Default: `4`)
                    * `score_threshold`: Minimum relevance threshold
                        for `similarity_score_threshold`
                    * `fetch_k`: Amount of documents to pass to MMR algorithm
                        (Default: `20`)
                    * `lambda_mult`: Diversity of results returned by MMR;
                        `1` for minimum diversity and 0 for maximum. (Default: `0.5`)
                    * `filter`: Filter by document metadata

        Returns:
            用于 `VectorStore` 的检索器类。

        Examples:
        ```python
        # Retrieve more documents with higher diversity
        # Useful if your dataset has many similar documents
        docsearch.as_retriever(
            search_type="mmr", search_kwargs={"k": 6, "lambda_mult": 0.25}
        )

        # Fetch more documents for the MMR algorithm to consider
        # But only return the top 5
        docsearch.as_retriever(search_type="mmr", search_kwargs={"k": 5, "fetch_k": 50})

        # Only retrieve documents that have a relevance score
        # Above a certain threshold
        docsearch.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"score_threshold": 0.8},
        )

        # Only get the single most similar document from the dataset
        docsearch.as_retriever(search_kwargs={"k": 1})

        # Use a filter to only retrieve documents from a specific paper
        docsearch.as_retriever(
            search_kwargs={"filter": {"paper_title": "GPT-4 Technical Report"}}
        )
        ```
        """
        tags = kwargs.pop("tags", None) or [*self._get_retriever_tags()]
        return VectorStoreRetriever(vectorstore=self, tags=tags, **kwargs)


class VectorStoreRetriever(BaseRetriever):
    """向量存储的基检索器类。"""

    vectorstore: VectorStore
    """用于检索的向量存储。"""

    search_type: str = "similarity"
    """要执行的搜索类型。"""

    search_kwargs: dict[str, Any] = Field(default_factory=dict)
    """传递给搜索函数的关键字参数。"""

    allowed_search_types: ClassVar[Collection[str]] = (
        "similarity",
        "similarity_score_threshold",
        "mmr",
    )

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )

    @model_validator(mode="before")
    @classmethod
    def validate_search_type(cls, values: dict[str, Any]) -> Any:
        """验证搜索类型。

        Args:
            values: 需要验证的值。

        Returns:
            已验证的值。

        Raises:
            ValueError: 如果 `search_type` 不在允许的搜索类型列表中。
            ValueError: 如果 `score_threshold` 未指定为浮点数（`0~1` 范围内）
        """
        search_type = values.get("search_type", "similarity")
        if search_type not in cls.allowed_search_types:
            msg = (
                f"search_type of {search_type} not allowed. Valid values are: "
                f"{cls.allowed_search_types}"
            )
            raise ValueError(msg)
        if search_type == "similarity_score_threshold":
            score_threshold = values.get("search_kwargs", {}).get("score_threshold")
            if (score_threshold is None) or (not isinstance(score_threshold, float)):
                msg = (
                    "`score_threshold` is not specified with a float value(0~1) "
                    "in `search_kwargs`."
                )
                raise ValueError(msg)
        return values

    def _get_ls_params(self, **kwargs: Any) -> LangSmithRetrieverParams:
        """获取追踪的标准参数。"""
        kwargs_ = self.search_kwargs | kwargs

        ls_params = super()._get_ls_params(**kwargs_)

        ls_params["ls_vector_store_provider"] = self.vectorstore.__class__.__name__

        if self.vectorstore.embeddings:
            ls_params["ls_embedding_provider"] = (
                self.vectorstore.embeddings.__class__.__name__
            )
        elif hasattr(self.vectorstore, "embedding") and isinstance(
            self.vectorstore.embedding, Embeddings
        ):
            ls_params["ls_embedding_provider"] = (
                self.vectorstore.embedding.__class__.__name__
            )

        return ls_params

    @override
    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun, **kwargs: Any
    ) -> list[Document]:
        kwargs_ = self.search_kwargs | kwargs
        if self.search_type == "similarity":
            docs = self.vectorstore.similarity_search(query, **kwargs_)
        elif self.search_type == "similarity_score_threshold":
            docs_and_similarities = (
                self.vectorstore.similarity_search_with_relevance_scores(
                    query, **kwargs_
                )
            )
            docs = [doc for doc, _ in docs_and_similarities]
        elif self.search_type == "mmr":
            docs = self.vectorstore.max_marginal_relevance_search(query, **kwargs_)
        else:
            msg = f"search_type of {self.search_type} not allowed."
            raise ValueError(msg)
        return docs

    @override
    async def _aget_relevant_documents(
        self,
        query: str,
        *,
        run_manager: AsyncCallbackManagerForRetrieverRun,
        **kwargs: Any,
    ) -> list[Document]:
        kwargs_ = self.search_kwargs | kwargs
        if self.search_type == "similarity":
            docs = await self.vectorstore.asimilarity_search(query, **kwargs_)
        elif self.search_type == "similarity_score_threshold":
            docs_and_similarities = (
                await self.vectorstore.asimilarity_search_with_relevance_scores(
                    query, **kwargs_
                )
            )
            docs = [doc for doc, _ in docs_and_similarities]
        elif self.search_type == "mmr":
            docs = await self.vectorstore.amax_marginal_relevance_search(
                query, **kwargs_
            )
        else:
            msg = f"search_type of {self.search_type} not allowed."
            raise ValueError(msg)
        return docs

    def add_documents(self, documents: list[Document], **kwargs: Any) -> list[str]:
        """向 `VectorStore` 添加文档。

        Args:
            documents: 要添加到 `VectorStore` 的文档。
            **kwargs: 子类可能使用的其他关键字参数。

        Returns:
            已添加文本的 ID 列表。
        """
        return self.vectorstore.add_documents(documents, **kwargs)

    async def aadd_documents(
        self, documents: list[Document], **kwargs: Any
    ) -> list[str]:
        """异步向 `VectorStore` 添加文档。

        Args:
            documents: 要添加到 `VectorStore` 的文档。
            **kwargs: 子类可能使用的其他关键字参数。

        Returns:
            已添加文本的 ID 列表。
        """
        return await self.vectorstore.aadd_documents(documents, **kwargs)

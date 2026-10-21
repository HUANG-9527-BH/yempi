"""集成检索器。

通过加权倒数排名融合（Reciprocal Rank Fusion）集成多个检索器结果的集成检索器。
"""

import asyncio
from collections import defaultdict
from collections.abc import Callable, Hashable, Iterable, Iterator
from itertools import chain
from typing import (
    Any,
    TypeVar,
    cast,
)

from langchain_core.callbacks import (
    AsyncCallbackManagerForRetrieverRun,
    CallbackManagerForRetrieverRun,
)
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever, RetrieverLike
from langchain_core.runnables import RunnableConfig
from langchain_core.runnables.config import ensure_config, patch_config
from langchain_core.runnables.utils import (
    ConfigurableFieldSpec,
    get_unique_config_specs,
)
from pydantic import model_validator
from typing_extensions import override

T = TypeVar("T")
H = TypeVar("H", bound=Hashable)


def unique_by_key(iterable: Iterable[T], key: Callable[[T], H]) -> Iterator[T]:
    """基于键函数从可迭代对象中提取唯一元素。

    Args:
        iterable: 要过滤的可迭代对象。
        key: 为每个元素返回可哈希键的函数。

    Yields:
        基于键函数的可迭代对象中的唯一元素。
    """
    seen = set()
    for e in iterable:
        if (k := key(e)) not in seen:
            seen.add(k)
            yield e


class EnsembleRetriever(BaseRetriever):
    """集成多个检索器的检索器。

    它使用了排序融合机制。

    Args:
        retrievers: A list of retrievers to ensemble.
        weights: A list of weights corresponding to the retrievers. Defaults to equal
            weighting for all retrievers.
        c: A constant added to the rank, controlling the balance between the importance
            of high-ranked items and the consideration given to lower-ranked items.
        id_key: The key in the document's metadata used to determine unique documents.
            If not specified, page_content is used.
    """

    retrievers: list[RetrieverLike]
    weights: list[float]
    c: int = 60
    id_key: str | None = None

    @property
    def config_specs(self) -> list[ConfigurableFieldSpec]:
        """列出此 Runnable 的可配置字段。"""
        return get_unique_config_specs(
            spec for retriever in self.retrievers for spec in retriever.config_specs
        )

    @model_validator(mode="before")
    @classmethod
    def _set_weights(cls, values: dict[str, Any]) -> Any:
        weights = values.get("weights")

        if not weights:
            n_retrievers = len(values["retrievers"])
            values["weights"] = [1 / n_retrievers] * n_retrievers
            return values

        retrievers = values["retrievers"]
        if len(weights) != len(retrievers):
            msg = (
                "Length of weights must match number of retrievers "
                f"(got {len(weights)} weights for {len(retrievers)} retrievers)."
            )
            raise ValueError(msg)

        if not any(w > 0 for w in weights):
            msg = "At least one ensemble weight must be greater than zero."
            raise ValueError(msg)

        return values

    @override
    def invoke(
        self,
        input: str,
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> list[Document]:
        from langchain_core.callbacks import CallbackManager

        config = ensure_config(config)
        callback_manager = CallbackManager.configure(
            config.get("callbacks"),
            None,
            verbose=kwargs.get("verbose", False),
            inheritable_tags=config.get("tags", []),
            local_tags=self.tags,
            inheritable_metadata=config.get("metadata", {}),
            local_metadata=self.metadata,
        )
        run_manager = callback_manager.on_retriever_start(
            None,
            input,
            name=config.get("run_name") or self.get_name(),
            **kwargs,
        )
        try:
            result = self.rank_fusion(input, run_manager=run_manager, config=config)
        except Exception as e:
            run_manager.on_retriever_error(e)
            raise
        else:
            run_manager.on_retriever_end(
                result,
                **kwargs,
            )
            return result

    @override
    async def ainvoke(
        self,
        input: str,
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> list[Document]:
        from langchain_core.callbacks import AsyncCallbackManager

        config = ensure_config(config)
        callback_manager = AsyncCallbackManager.configure(
            config.get("callbacks"),
            None,
            verbose=kwargs.get("verbose", False),
            inheritable_tags=config.get("tags", []),
            local_tags=self.tags,
            inheritable_metadata=config.get("metadata", {}),
            local_metadata=self.metadata,
        )
        run_manager = await callback_manager.on_retriever_start(
            None,
            input,
            name=config.get("run_name") or self.get_name(),
            **kwargs,
        )
        try:
            result = await self.arank_fusion(
                input,
                run_manager=run_manager,
                config=config,
            )
        except Exception as e:
            await run_manager.on_retriever_error(e)
            raise
        else:
            await run_manager.on_retriever_end(
                result,
                **kwargs,
            )
            return result

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> list[Document]:
        """获取给定查询的相关文档。

        Args:
            query: 要搜索的查询。
            run_manager: 使用的回调处理器。

        Returns:
            重新排序后的文档列表。
        """
                # 获取检索器的融合结果。
        return self.rank_fusion(query, run_manager)

    async def _aget_relevant_documents(
        self,
        query: str,
        *,
        run_manager: AsyncCallbackManagerForRetrieverRun,
    ) -> list[Document]:
        """异步获取给定查询的相关文档。

        Args:
            query: 要搜索的查询。
            run_manager: 使用的回调处理器。

        Returns:
            重新排序后的文档列表。
        """
                # 获取检索器的融合结果。
        return await self.arank_fusion(query, run_manager)

    def rank_fusion(
        self,
        query: str,
        run_manager: CallbackManagerForRetrieverRun,
        *,
        config: RunnableConfig | None = None,
    ) -> list[Document]:
        """排序融合。

        获取检索器的结果，并使用 rank_fusion_func 得到最终结果。

        Args:
            query: 要搜索的查询。
            run_manager: 使用的回调处理器。
            config: 可选的检索器配置。

        Returns:
            重新排序后的文档列表。
        """
                # 获取所有检索器的结果。
        retriever_docs = [
            retriever.invoke(
                query,
                patch_config(
                    config,
                    callbacks=run_manager.get_child(tag=f"retriever_{i + 1}"),
                ),
            )
            for i, retriever in enumerate(self.retrievers)
        ]

                # 确保每个 retriever_docs 列表中的检索文档均为 Document 类型
        for i in range(len(retriever_docs)):
            retriever_docs[i] = [
                Document(page_content=cast("str", doc)) if isinstance(doc, str) else doc  # type: ignore[unreachable]
                for doc in retriever_docs[i]
            ]

                # 应用排序融合
        return self.weighted_reciprocal_rank(retriever_docs)

    async def arank_fusion(
        self,
        query: str,
        run_manager: AsyncCallbackManagerForRetrieverRun,
        *,
        config: RunnableConfig | None = None,
    ) -> list[Document]:
        """排序融合。

        异步获取检索器的结果
        并使用 rank_fusion_func 得到最终结果。

        Args:
            query: 要搜索的查询。
            run_manager: 使用的回调处理器。
            config: 可选的检索器配置。

        Returns:
            重新排序后的文档列表。
        """
                # 获取所有检索器的结果。
        retriever_docs = await asyncio.gather(
            *[
                retriever.ainvoke(
                    query,
                    patch_config(
                        config,
                        callbacks=run_manager.get_child(tag=f"retriever_{i + 1}"),
                    ),
                )
                for i, retriever in enumerate(self.retrievers)
            ],
        )

                # 确保每个 retriever_docs 列表中的检索文档均为 Document 类型
        for i in range(len(retriever_docs)):
            retriever_docs[i] = [
                Document(page_content=cast("str", doc)) if isinstance(doc, str) else doc  # type: ignore[unreachable]
                for doc in retriever_docs[i]
            ]

                # 应用排序融合
        return self.weighted_reciprocal_rank(retriever_docs)

    def weighted_reciprocal_rank(
        self,
        doc_lists: list[list[Document]],
    ) -> list[Document]:
        """对多个排名列表执行加权倒数排名融合。

        您可以在以下位置找到关于 RRF 的更多详细信息：
        https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf。

        Args:
            doc_lists: 包含唯一项目的排名列表的列表，每个排名列表包含唯一项。

        Returns:
            最终聚合的项目列表，按加权 RRF 分数降序排列。
        """
        if len(doc_lists) != len(self.weights):
            msg = "Number of rank lists must be equal to the number of weights."
            raise ValueError(msg)

                # 将每个文档的内容与其 RRF 分数关联，以便后续按分数排序
                # 跨检索器的重复内容被合并并累计计分
        rrf_score: dict[str, float] = defaultdict(float)
        for doc_list, weight in zip(doc_lists, self.weights, strict=False):
            for rank, doc in enumerate(doc_list, start=1):
                rrf_score[
                    (
                        doc.page_content
                        if self.id_key is None
                        else doc.metadata[self.id_key]
                    )
                ] += weight / (rank + self.c)

                # 文档根据其内容去重后，按其分数排序
        all_docs = chain.from_iterable(doc_lists)
        return sorted(
            unique_by_key(
                all_docs,
                lambda doc: (
                    doc.page_content
                    if self.id_key is None
                    else doc.metadata[self.id_key]
                ),
            ),
            reverse=True,
            key=lambda doc: rrf_score[
                doc.page_content if self.id_key is None else doc.metadata[self.id_key]
            ],
        )

import asyncio

from langchain_core.callbacks import (
    AsyncCallbackManagerForRetrieverRun,
    CallbackManagerForRetrieverRun,
)
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever


class MergerRetriever(BaseRetriever):
    """合并多个检索器结果的检索器。"""

    retrievers: list[BaseRetriever]
    """要合并的检索器列表。"""

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> list[Document]:
        """获取给定查询的相关文档。

        Args:
            query: 要搜索的查询。
            run_manager: 要使用的回调处理器。

        Returns:
            相关文档列表。
        """
                # 合并检索器的结果。
        return self.merge_documents(query, run_manager)

    async def _aget_relevant_documents(
        self,
        query: str,
        *,
        run_manager: AsyncCallbackManagerForRetrieverRun,
    ) -> list[Document]:
        """异步获取给定查询的相关文档。

        Args:
            query: 要搜索的查询。
            run_manager: 要使用的回调处理器。

        Returns:
            相关文档列表。
        """
                # 合并检索器的结果。
        return await self.amerge_documents(query, run_manager)

    def merge_documents(
        self,
        query: str,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> list[Document]:
        """合并检索器的结果。

        Args:
            query: 要搜索的查询。
            run_manager: 要使用的回调处理器。

        Returns:
            合并后的文档列表。
        """
                # 获取所有检索器的结果。
        retriever_docs = [
            retriever.invoke(
                query,
                config={"callbacks": run_manager.get_child(f"retriever_{i + 1}")},
            )
            for i, retriever in enumerate(self.retrievers)
        ]

                # 合并检索器的结果。
        merged_documents = []
        max_docs = max(map(len, retriever_docs), default=0)
        for i in range(max_docs):
            for _retriever, doc in zip(self.retrievers, retriever_docs, strict=False):
                if i < len(doc):
                    merged_documents.append(doc[i])

        return merged_documents

    async def amerge_documents(
        self,
        query: str,
        run_manager: AsyncCallbackManagerForRetrieverRun,
    ) -> list[Document]:
        """异步合并检索器的结果。

        Args:
            query: 要搜索的查询。
            run_manager: 要使用的回调处理器。

        Returns:
            合并后的文档列表。
        """
                # 获取所有检索器的结果。
        retriever_docs = await asyncio.gather(
            *(
                retriever.ainvoke(
                    query,
                    config={"callbacks": run_manager.get_child(f"retriever_{i + 1}")},
                )
                for i, retriever in enumerate(self.retrievers)
            ),
        )

                # 合并检索器的结果。
        merged_documents = []
        max_docs = max(map(len, retriever_docs), default=0)
        for i in range(max_docs):
            for _retriever, doc in zip(self.retrievers, retriever_docs, strict=False):
                if i < len(doc):
                    merged_documents.append(doc[i])

        return merged_documents

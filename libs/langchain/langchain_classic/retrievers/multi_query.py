import asyncio
import logging
from collections.abc import Sequence

from langchain_core.callbacks import (
    AsyncCallbackManagerForRetrieverRun,
    CallbackManagerForRetrieverRun,
)
from langchain_core.documents import Document
from langchain_core.language_models import BaseLanguageModel
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.prompts import BasePromptTemplate
from langchain_core.prompts.prompt import PromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import Runnable
from typing_extensions import override

from langchain_classic.chains.llm import LLMChain

logger = logging.getLogger(__name__)


class LineListOutputParser(BaseOutputParser[list[str]]):
    """用于解析行列表的输出解析器。"""

    @override
    def parse(self, text: str) -> list[str]:
        lines = text.strip().split("\n")
        return list(filter(None, lines))  # 移除空行


# 默认提示词
DEFAULT_QUERY_PROMPT = PromptTemplate(
    input_variables=["question"],
    template="""You are an AI language model assistant. Your task is
    to generate 3 different versions of the given user
    question to retrieve relevant documents from a vector  database.
    By generating multiple perspectives on the user question,
    your goal is to help the user overcome some of the limitations
    of distance-based similarity search. Provide these alternative
    questions separated by newlines. Original question: {question}""",
)


def _unique_documents(documents: Sequence[Document]) -> list[Document]:
    return [doc for i, doc in enumerate(documents) if doc not in documents[:i]]


class MultiQueryRetriever(BaseRetriever):
    """根据查询，使用 LLM 生成一组查询。

    为每个查询检索文档。返回所有检索到文档的唯一并集。
    """

    retriever: BaseRetriever
    llm_chain: Runnable
    verbose: bool = True
    parser_key: str = "lines"
    """已弃用。`parser_key` 不再使用，不应指定。"""
    include_original: bool = False
    """是否将原始查询包含在生成的查询列表中。"""

    @classmethod
    def from_llm(
        cls,
        retriever: BaseRetriever,
        llm: BaseLanguageModel,
        prompt: BasePromptTemplate = DEFAULT_QUERY_PROMPT,
        parser_key: str | None = None,  # noqa: ARG003
        include_original: bool = False,  # noqa: FBT001,FBT002
    ) -> "MultiQueryRetriever":
        """使用默认模板从 llm 初始化。

        Args:
            retriever: retriever to query documents from
            llm: llm for query generation using DEFAULT_QUERY_PROMPT
            prompt: The prompt which aims to generate several different versions
                of the given user query
            parser_key: DEPRECATED. `parser_key` is no longer used and should not be
                specified.
            include_original: Whether to include the original query in the list of
                generated queries.

        Returns:
            MultiQueryRetriever
        """
        output_parser = LineListOutputParser()
        llm_chain = prompt | llm | output_parser
        return cls(
            retriever=retriever,
            llm_chain=llm_chain,
            include_original=include_original,
        )

    async def _aget_relevant_documents(
        self,
        query: str,
        *,
        run_manager: AsyncCallbackManagerForRetrieverRun,
    ) -> list[Document]:
        """根据用户查询获取相关文档。

        Args:
            query: 用户查询
            run_manager: 要使用的回调处理器。

        Returns:
            所有生成查询的相关文档的唯一并集
        """
        queries = await self.agenerate_queries(query, run_manager)
        if self.include_original:
            queries.append(query)
        documents = await self.aretrieve_documents(queries, run_manager)
        return self.unique_union(documents)

    async def agenerate_queries(
        self,
        question: str,
        run_manager: AsyncCallbackManagerForRetrieverRun,
    ) -> list[str]:
        """根据用户输入生成查询。

        Args:
            question: 用户查询
            run_manager: 要使用的回调处理器。

        Returns:
            与用户输入相似的 LLM 生成查询列表
        """
        response = await self.llm_chain.ainvoke(
            {"question": question},
            config={"callbacks": run_manager.get_child()},
        )
        lines = response["text"] if isinstance(self.llm_chain, LLMChain) else response
        if self.verbose:
            logger.info("Generated queries: %s", lines)
        return lines

    async def aretrieve_documents(
        self,
        queries: list[str],
        run_manager: AsyncCallbackManagerForRetrieverRun,
    ) -> list[Document]:
        """运行所有 LLM 生成的查询。

        Args:
            queries: 查询列表
            run_manager: 要使用的回调处理器

        Returns:
            检索到的文档列表
        """
        document_lists = await asyncio.gather(
            *(
                self.retriever.ainvoke(
                    query,
                    config={"callbacks": run_manager.get_child()},
                )
                for query in queries
            ),
        )
        return [doc for docs in document_lists for doc in docs]

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> list[Document]:
        """根据用户查询获取相关文档。

        Args:
            query: 用户查询
            run_manager: 要使用的回调处理器。

        Returns:
            所有生成查询的相关文档的唯一并集
        """
        queries = self.generate_queries(query, run_manager)
        if self.include_original:
            queries.append(query)
        documents = self.retrieve_documents(queries, run_manager)
        return self.unique_union(documents)

    def generate_queries(
        self,
        question: str,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> list[str]:
        """根据用户输入生成查询。

        Args:
            question: 用户查询
            run_manager: 用于回调的运行管理器

        Returns:
            与用户输入相似的 LLM 生成查询列表
        """
        response = self.llm_chain.invoke(
            {"question": question},
            config={"callbacks": run_manager.get_child()},
        )
        lines = response["text"] if isinstance(self.llm_chain, LLMChain) else response
        if self.verbose:
            logger.info("Generated queries: %s", lines)
        return lines

    def retrieve_documents(
        self,
        queries: list[str],
        run_manager: CallbackManagerForRetrieverRun,
    ) -> list[Document]:
        """运行所有 LLM 生成的查询。

        Args:
            queries: 查询列表
            run_manager: 用于回调的运行管理器

        Returns:
            检索到的文档列表
        """
        documents = []
        for query in queries:
            docs = self.retriever.invoke(
                query,
                config={"callbacks": run_manager.get_child()},
            )
            documents.extend(docs)
        return documents

    def unique_union(self, documents: list[Document]) -> list[Document]:
        """获取唯一的文档。

        Args:
            documents: 检索到的文档列表

        Returns:
            唯一的检索到的文档列表
        """
        return _unique_documents(documents)

import logging

from langchain_core.callbacks import (
    AsyncCallbackManagerForRetrieverRun,
    CallbackManagerForRetrieverRun,
)
from langchain_core.documents import Document
from langchain_core.language_models import BaseLLM
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import BasePromptTemplate
from langchain_core.prompts.prompt import PromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import Runnable

logger = logging.getLogger(__name__)

# 默认模板
DEFAULT_TEMPLATE = """You are an assistant tasked with taking a natural language \
query from a user and converting it into a query for a vectorstore. \
In this process, you strip out information that is not relevant for \
the retrieval task. Here is the user query: {question}"""

# 默认提示词
DEFAULT_QUERY_PROMPT = PromptTemplate.from_template(DEFAULT_TEMPLATE)


class RePhraseQueryRetriever(BaseRetriever):
    """给定一个查询，使用 LLM 重述该查询。

    然后，为重述后的查询检索文档。
    """

    retriever: BaseRetriever
    llm_chain: Runnable

    @classmethod
    def from_llm(
        cls,
        retriever: BaseRetriever,
        llm: BaseLLM,
        prompt: BasePromptTemplate = DEFAULT_QUERY_PROMPT,
    ) -> "RePhraseQueryRetriever":
        """从 LLM 初始化，使用默认模板。

        The prompt used here expects a single input: 此处使用的提示词期望单个输入：`question`

        Args:
            retriever: 用于从文档中查询的检索器
            llm: 使用 DEFAULT_QUERY_PROMPT 进行查询生成的 LLM
            prompt: 查询生成的提示词模板

        Returns:
            RePhraseQueryRetriever
        """
        llm_chain = prompt | llm | StrOutputParser()
        return cls(
            retriever=retriever,
            llm_chain=llm_chain,
        )

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> list[Document]:
        """根据用户问题获取相关文档。

        Args:
            query: 用户问题
            run_manager: 要使用的回调处理器

        Returns:
            重述后问题的相关文档
        """
        re_phrased_question = self.llm_chain.invoke(
            query,
            {"callbacks": run_manager.get_child()},
        )
        logger.info("Re-phrased question: %s", re_phrased_question)
        return self.retriever.invoke(
            re_phrased_question,
            config={"callbacks": run_manager.get_child()},
        )

    async def _aget_relevant_documents(
        self,
        query: str,
        *,
        run_manager: AsyncCallbackManagerForRetrieverRun,
    ) -> list[Document]:
        raise NotImplementedError

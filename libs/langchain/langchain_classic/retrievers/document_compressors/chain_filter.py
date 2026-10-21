"""使用 LLM 进行过滤，丢弃与查询无关的文档。"""

from collections.abc import Callable, Sequence
from typing import Any

from langchain_core.callbacks import Callbacks
from langchain_core.documents import BaseDocumentCompressor, Document
from langchain_core.language_models import BaseLanguageModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import BasePromptTemplate, PromptTemplate
from langchain_core.runnables import Runnable
from langchain_core.runnables.config import RunnableConfig
from pydantic import ConfigDict

from langchain_classic.chains import LLMChain
from langchain_classic.output_parsers.boolean import BooleanOutputParser
from langchain_classic.retrievers.document_compressors.chain_filter_prompt import (
    prompt_template,
)


def _get_default_chain_prompt() -> PromptTemplate:
    return PromptTemplate(
        template=prompt_template,
        input_variables=["question", "context"],
        output_parser=BooleanOutputParser(),
    )


def default_get_input(query: str, doc: Document) -> dict[str, Any]:
    """返回压缩链的输入。"""
    return {"question": query, "context": doc.page_content}


class LLMChainFilter(BaseDocumentCompressor):
    """过滤掉与查询无关的文档。"""

    llm_chain: Runnable
    """用于过滤文档的 LLM 包装器。
    预期链提示词包含一个 BooleanOutputParser。"""

    get_input: Callable[[str, Document], dict] = default_get_input
    """从查询和 Document 构造链输入的可调用对象。"""

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )

    def compress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Callbacks | None = None,
    ) -> Sequence[Document]:
        """根据文档与查询的相关性进行过滤。"""
        filtered_docs = []

        config = RunnableConfig(callbacks=callbacks)
        outputs = zip(
            self.llm_chain.batch(
                [self.get_input(query, doc) for doc in documents],
                config=config,
            ),
            documents,
            strict=False,
        )

        for output_, doc in outputs:
            include_doc = None
            if isinstance(self.llm_chain, LLMChain):
                output = output_[self.llm_chain.output_key]
                if self.llm_chain.prompt.output_parser is not None:
                    include_doc = self.llm_chain.prompt.output_parser.parse(output)
            elif isinstance(output_, bool):
                include_doc = output_
            if include_doc:
                filtered_docs.append(doc)

        return filtered_docs

    async def acompress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Callbacks | None = None,
    ) -> Sequence[Document]:
        """根据文档与查询的相关性进行过滤。"""
        filtered_docs = []

        config = RunnableConfig(callbacks=callbacks)
        outputs = zip(
            await self.llm_chain.abatch(
                [self.get_input(query, doc) for doc in documents],
                config=config,
            ),
            documents,
            strict=False,
        )
        for output_, doc in outputs:
            include_doc = None
            if isinstance(self.llm_chain, LLMChain):
                output = output_[self.llm_chain.output_key]
                if self.llm_chain.prompt.output_parser is not None:
                    include_doc = self.llm_chain.prompt.output_parser.parse(output)
            elif isinstance(output_, bool):
                include_doc = output_
            if include_doc:
                filtered_docs.append(doc)

        return filtered_docs

    @classmethod
    def from_llm(
        cls,
        llm: BaseLanguageModel,
        prompt: BasePromptTemplate | None = None,
        **kwargs: Any,
    ) -> "LLMChainFilter":
        """从语言模型创建 LLMChainFilter。

        Args:
            llm: 用于过滤的语言模型。
            prompt: 用于过滤的提示词。
            kwargs: 传递给构造函数的其他参数。

        Returns:
            使用给定语言模型的 LLMChainFilter。
        """
        _prompt = prompt if prompt is not None else _get_default_chain_prompt()
        if _prompt.output_parser is not None:
            parser = _prompt.output_parser
        else:
            parser = StrOutputParser()
        llm_chain = _prompt | llm | parser
        return cls(llm_chain=llm_chain, **kwargs)

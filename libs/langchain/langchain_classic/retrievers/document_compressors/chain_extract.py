"""使用 LLM 链提取文档相关内容的文档过滤器。"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, cast

from langchain_core.callbacks import Callbacks
from langchain_core.documents import BaseDocumentCompressor, Document
from langchain_core.language_models import BaseLanguageModel
from langchain_core.output_parsers import BaseOutputParser, StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import Runnable
from pydantic import ConfigDict
from typing_extensions import override

from langchain_classic.chains.llm import LLMChain
from langchain_classic.retrievers.document_compressors.chain_extract_prompt import (
    prompt_template,
)


def default_get_input(query: str, doc: Document) -> dict[str, Any]:
    """返回压缩链的输入。"""
    return {"question": query, "context": doc.page_content}


class NoOutputParser(BaseOutputParser[str]):
    """解析可能返回空字符串的输出。"""

    no_output_str: str = "NO_OUTPUT"

    @override
    def parse(self, text: str) -> str:
        cleaned_text = text.strip()
        if cleaned_text == self.no_output_str:
            return ""
        return cleaned_text


def _get_default_chain_prompt() -> PromptTemplate:
    output_parser = NoOutputParser()
    template = prompt_template.format(no_output_str=output_parser.no_output_str)
    return PromptTemplate(
        template=template,
        input_variables=["question", "context"],
        output_parser=output_parser,
    )


class LLMChainExtractor(BaseDocumentCompressor):
    """LLM 链提取器。

    使用 LLM 链提取文档相关内容的文档压缩器。
    """

    llm_chain: Runnable
    """用于压缩文档的 LLM 包装器。"""

    get_input: Callable[[str, Document], dict] = default_get_input
    """用于从查询和 Document 构造链输入的可调用对象。"""

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )

    def compress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Callbacks | None = None,
    ) -> Sequence[Document]:
        """压缩原始文档的页面内容。"""
        compressed_docs = []
        for doc in documents:
            _input = self.get_input(query, doc)
            output_ = self.llm_chain.invoke(_input, config={"callbacks": callbacks})
            if isinstance(self.llm_chain, LLMChain):
                output = output_[self.llm_chain.output_key]
                if self.llm_chain.prompt.output_parser is not None:
                    output = self.llm_chain.prompt.output_parser.parse(output)
            else:
                output = output_
            if len(output) == 0:
                continue
            compressed_docs.append(
                Document(page_content=cast("str", output), metadata=doc.metadata),
            )
        return compressed_docs

    async def acompress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Callbacks | None = None,
    ) -> Sequence[Document]:
        """异步压缩原始文档的页面内容。"""
        inputs = [self.get_input(query, doc) for doc in documents]
        outputs = await self.llm_chain.abatch(inputs, {"callbacks": callbacks})
        compressed_docs = []
        for i, doc in enumerate(documents):
            if len(outputs[i]) == 0:
                continue
            compressed_docs.append(
                Document(page_content=outputs[i], metadata=doc.metadata),
            )
        return compressed_docs

    @classmethod
    def from_llm(
        cls,
        llm: BaseLanguageModel,
        prompt: PromptTemplate | None = None,
        get_input: Callable[[str, Document], str] | None = None,
        llm_chain_kwargs: dict | None = None,  # noqa: ARG003
    ) -> LLMChainExtractor:
        """从 LLM 初始化。"""
        _prompt = prompt if prompt is not None else _get_default_chain_prompt()
        _get_input = get_input if get_input is not None else default_get_input
        if _prompt.output_parser is not None:
            parser = _prompt.output_parser
        else:
            parser = StrOutputParser()
        llm_chain = _prompt | llm | parser
        return cls(llm_chain=llm_chain, get_input=_get_input)

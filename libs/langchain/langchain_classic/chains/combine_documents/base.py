"""组合文档的链的基础接口。"""

from abc import ABC, abstractmethod
from typing import Any

from langchain_core._api import deprecated
from langchain_core.callbacks import (
    AsyncCallbackManagerForChainRun,
    CallbackManagerForChainRun,
)
from langchain_core.documents import Document
from langchain_core.prompts import BasePromptTemplate, PromptTemplate
from langchain_core.runnables.config import RunnableConfig
from langchain_core.utils.pydantic import create_model
from langchain_text_splitters import RecursiveCharacterTextSplitter, TextSplitter
from pydantic import BaseModel, Field
from typing_extensions import override

from langchain_classic.chains.base import Chain

DEFAULT_DOCUMENT_SEPARATOR = "\n\n"
DOCUMENTS_KEY = "context"
DEFAULT_DOCUMENT_PROMPT = PromptTemplate.from_template("{page_content}")


def _validate_prompt(prompt: BasePromptTemplate, document_variable_name: str) -> None:
    if document_variable_name not in prompt.input_variables:
        msg = (
            f"Prompt must accept {document_variable_name} as an input variable. "
            f"Received prompt with input variables: {prompt.input_variables}"
        )
        raise ValueError(msg)


class BaseCombineDocumentsChain(Chain, ABC):
    """组合文档的链的基础接口。

    Subclasses of this chain deal with combining documents in a variety of
    ways. This base class exists to add some uniformity in the interface these types
    of chains should expose. Namely, they expect an input key related to the documents
    to use (default `input_documents`), and then also expose a method to calculate
    the length of a prompt from documents (useful for outside callers to use to
    determine whether it's safe to pass a list of documents into this chain or whether
    that will be longer than the context length).
    """

    input_key: str = "input_documents"
    output_key: str = "output_text"

    @override
    def get_input_schema(
        self,
        config: RunnableConfig | None = None,
    ) -> type[BaseModel]:
        return create_model(
            "CombineDocumentsInput",
            **{self.input_key: (list[Document], None)},
        )

    @override
    def get_output_schema(
        self,
        config: RunnableConfig | None = None,
    ) -> type[BaseModel]:
        return create_model(
            "CombineDocumentsOutput",
            **{self.output_key: (str, None)},
        )

    @property
    def input_keys(self) -> list[str]:
        """期望的输入键。"""
        return [self.input_key]

    @property
    def output_keys(self) -> list[str]:
        """返回的输出键。"""
        return [self.output_key]

    def prompt_length(self, docs: list[Document], **kwargs: Any) -> int | None:  # noqa: ARG002
        """根据传入的文档返回提示词长度。

        可被调用方用于判断传入文档列表是否会超出特定提示词长度。
        当尝试确保提示词大小保持在某个上下文限制以下时非常有用。

        Args:
            docs: 用于计算总提示词长度的文档列表。 **kwargs:
            计算提示词长度可能需要的其他参数。

        Returns:
            如果该方法不依赖于提示词长度，则返回 None，
            否则返回提示词的 token 长度。
        """
        return None

    @abstractmethod
    def combine_docs(self, docs: list[Document], **kwargs: Any) -> tuple[str, dict]:
        """将文档合并为单个字符串。

        Args:
            docs: List[Document]，待合并的文档 **kwargs:
            用于合并文档的其他参数，通常为提示词的其他输入。

        Returns:
            返回的第一个元素是单个字符串输出。第二个元素是需返回的其他键的字典。
        """

    @abstractmethod
    async def acombine_docs(
        self,
        docs: list[Document],
        **kwargs: Any,
    ) -> tuple[str, dict]:
        """将文档合并为单个字符串。

        Args:
            docs: List[Document]，待合并的文档 **kwargs:
            用于合并文档的其他参数，通常为提示词的其他输入。

        Returns:
            返回的第一个元素是单个字符串输出。第二个元素是需返回的其他键的字典。
        """

    def _call(
        self,
        inputs: dict[str, list[Document]],
        run_manager: CallbackManagerForChainRun | None = None,
    ) -> dict[str, str]:
        """准备输入，调用合并文档，准备输出。"""
        _run_manager = run_manager or CallbackManagerForChainRun.get_noop_manager()
        docs = inputs[self.input_key]
                # 其他键假定为 LLM 预测所需
        other_keys = {k: v for k, v in inputs.items() if k != self.input_key}
        output, extra_return_dict = self.combine_docs(
            docs,
            callbacks=_run_manager.get_child(),
            **other_keys,
        )
        extra_return_dict[self.output_key] = output
        return extra_return_dict

    async def _acall(
        self,
        inputs: dict[str, list[Document]],
        run_manager: AsyncCallbackManagerForChainRun | None = None,
    ) -> dict[str, str]:
        """准备输入，调用合并文档，准备输出。"""
        _run_manager = run_manager or AsyncCallbackManagerForChainRun.get_noop_manager()
        docs = inputs[self.input_key]
                # 其他键假定为 LLM 预测所需
        other_keys = {k: v for k, v in inputs.items() if k != self.input_key}
        output, extra_return_dict = await self.acombine_docs(
            docs,
            callbacks=_run_manager.get_child(),
            **other_keys,
        )
        extra_return_dict[self.output_key] = output
        return extra_return_dict


@deprecated(
    since="0.2.7",
    removal="2.0.0",
    alternative="langchain.agents.create_agent",
    addendum=(
        "Build new RAG flows with `create_agent`, a retrieval tool, and an "
        "appropriate text splitter. See "
        "https://docs.langchain.com/oss/python/langchain/rag"
    ),
)
class AnalyzeDocumentChain(Chain):
    """将文档拆分后，分块分析的链。

    此链由 TextSplitter 和 CombineDocumentsChain 参数化。
    此链接收单个文档作为输入，然后将其拆分为块，
    再将这些块传递给 CombineDocumentsChain。

    This class is deprecated. See below for alternative implementations which
    supports async and streaming modes of operation.

    如果底层的组合文档链仅接受一个 `input_documents` 参数（例如由
    `load_summarize_chain` 生成的链）：

        ```python
        split_text = lambda x: text_splitter.create_documents([x])

        summarize_document_chain = split_text | chain
        ```

    如果底层链需要额外参数（例如 `load_qa_chain`，它需要额外的 `question`
    参数），我们可以使用以下方式：

        ```python
        from operator import itemgetter
        from langchain_core.runnables import RunnableLambda, RunnableParallel

        split_text = RunnableLambda(lambda x: text_splitter.create_documents([x]))
        summarize_document_chain = RunnableParallel(
            question=itemgetter("question"),
            input_documents=itemgetter("input_document") | split_text,
        ) | chain.pick("output_text")
        ```

    若要额外返回输入参数，如 `AnalyzeDocumentChain` 所做，
    我们可以使用 `RunnablePassthrough` 包装此构造：

        ```python
        from operator import itemgetter
        from langchain_core.runnables import (
            RunnableLambda,
            RunnableParallel,
            RunnablePassthrough,
        )

        split_text = RunnableLambda(lambda x: text_splitter.create_documents([x]))
        summarize_document_chain = RunnablePassthrough.assign(
            output_text=RunnableParallel(
                question=itemgetter("question"),
                input_documents=itemgetter("input_document") | split_text,
            )
            | chain.pick("output_text")
        )
        ```
    """

    input_key: str = "input_document"
    text_splitter: TextSplitter = Field(default_factory=RecursiveCharacterTextSplitter)
    combine_docs_chain: BaseCombineDocumentsChain

    @property
    def input_keys(self) -> list[str]:
        """期望的输入键。"""
        return [self.input_key]

    @property
    def output_keys(self) -> list[str]:
        """返回输出键。"""
        return self.combine_docs_chain.output_keys

    @override
    def get_input_schema(
        self,
        config: RunnableConfig | None = None,
    ) -> type[BaseModel]:
        return create_model(
            "AnalyzeDocumentChain",
            **{self.input_key: (str, None)},
        )

    @override
    def get_output_schema(
        self,
        config: RunnableConfig | None = None,
    ) -> type[BaseModel]:
        return self.combine_docs_chain.get_output_schema(config)

    def _call(
        self,
        inputs: dict[str, str],
        run_manager: CallbackManagerForChainRun | None = None,
    ) -> dict[str, str]:
        """将文档分割成块，并传递给 CombineDocumentsChain。"""
        _run_manager = run_manager or CallbackManagerForChainRun.get_noop_manager()
        document = inputs[self.input_key]
        docs = self.text_splitter.create_documents([document])
                # 其他键假定为 LLM 预测所需
        other_keys: dict = {k: v for k, v in inputs.items() if k != self.input_key}
        other_keys[self.combine_docs_chain.input_key] = docs
        return self.combine_docs_chain(
            other_keys,
            return_only_outputs=True,
            callbacks=_run_manager.get_child(),
        )

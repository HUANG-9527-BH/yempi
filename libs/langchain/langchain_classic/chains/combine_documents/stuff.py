"""将文档合并到上下文中的链。"""

from typing import Any

from langchain_core._api import deprecated
from langchain_core.callbacks import Callbacks
from langchain_core.documents import Document
from langchain_core.language_models import LanguageModelLike
from langchain_core.output_parsers import BaseOutputParser, StrOutputParser
from langchain_core.prompts import BasePromptTemplate, format_document
from langchain_core.runnables import Runnable, RunnablePassthrough
from pydantic import ConfigDict, Field, model_validator
from typing_extensions import override

from langchain_classic.chains.combine_documents.base import (
    DEFAULT_DOCUMENT_PROMPT,
    DEFAULT_DOCUMENT_SEPARATOR,
    DOCUMENTS_KEY,
    BaseCombineDocumentsChain,
    _validate_prompt,
)
from langchain_classic.chains.llm import LLMChain


def create_stuff_documents_chain(
    llm: LanguageModelLike,
    prompt: BasePromptTemplate,
    *,
    output_parser: BaseOutputParser | None = None,
    document_prompt: BasePromptTemplate | None = None,
    document_separator: str = DEFAULT_DOCUMENT_SEPARATOR,
    document_variable_name: str = DOCUMENTS_KEY,
) -> Runnable[dict[str, Any], Any]:
    r"""创建一个将文档列表传递给模型的链。

    Args:
        llm: Language model.
        prompt: Prompt template. Must contain input variable `"context"` (override by
            setting document_variable), which will be used for passing in the formatted
            documents.
        output_parser: Output parser. Defaults to `StrOutputParser`.
        document_prompt: Prompt used for formatting each document into a string. Input
            variables can be "page_content" or any metadata keys that are in all
            documents. "page_content" will automatically retrieve the
            `Document.page_content`, and all other inputs variables will be
            automatically retrieved from the `Document.metadata` dictionary. Default to
            a prompt that only contains `Document.page_content`.
        document_separator: String separator to use between formatted document strings.
        document_variable_name: Variable name to use for the formatted documents in the
            prompt. Defaults to `"context"`.

    Returns:
        An LCEL Runnable. The input is a dictionary that must have a `"context"` key
        that maps to a `list[Document]`, and any other input variables expected in the
        prompt. The `Runnable` return type depends on `output_parser` used.

    Example:
        ```python
        # pip install -U langchain langchain-openai

        from langchain_openai import ChatOpenAI
        from langchain_core.documents import Document
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_classic.chains.combine_documents import (
            create_stuff_documents_chain,
        )

        prompt = ChatPromptTemplate.from_messages(
            [("system", "What are everyone's favorite colors:\n\n{context}")]
        )
        model = ChatOpenAI(model="gpt-5.5")
        chain = create_stuff_documents_chain(model, prompt)

        docs = [
            Document(page_content="Jesse loves red but not yellow"),
            Document(
                page_content="Jamal loves green but not as much as he loves orange"
            ),
        ]

        chain.invoke({"context": docs})
        ```
    """
    _validate_prompt(prompt, document_variable_name)
    _document_prompt = document_prompt or DEFAULT_DOCUMENT_PROMPT
    _output_parser = output_parser or StrOutputParser()

    def format_docs(inputs: dict) -> str:
        return document_separator.join(
            format_document(doc, _document_prompt)
            for doc in inputs[document_variable_name]
        )

    return (
        RunnablePassthrough.assign(**{document_variable_name: format_docs}).with_config(
            run_name="format_inputs",
        )
        | prompt
        | llm
        | _output_parser
    ).with_config(run_name="stuff_documents_chain")


@deprecated(
    since="0.2.13",
    removal="2.0.0",
    alternative="langchain.agents.create_agent",
    addendum=(
        "Build new RAG flows with `create_agent` and a retrieval tool. See "
        "https://docs.langchain.com/oss/python/langchain/rag"
    ),
)
class StuffDocumentsChain(BaseCombineDocumentsChain):
    """将文档合并到上下文中的链。

    This chain takes a list of documents and first combines them into a single string.
    It does this by formatting each document into a string with the `document_prompt`
    and then joining them together with `document_separator`. It then adds that new
    string to the inputs with the variable name set by `document_variable_name`.
    Those inputs are then passed to the `llm_chain`.

    Example:
        ```python
        from langchain_classic.chains import StuffDocumentsChain, LLMChain
        from langchain_core.prompts import PromptTemplate
        from langchain_openai import OpenAI

        # This controls how each document will be formatted. Specifically,
        # it will be passed to `format_document` - see that function for more
        # details.
        document_prompt = PromptTemplate(
            input_variables=["page_content"], template="{page_content}"
        )
        document_variable_name = "context"
        model = OpenAI()
        # The prompt here should take as an input variable the
        # `document_variable_name`
        prompt = PromptTemplate.from_template("Summarize this content: {context}")
        llm_chain = LLMChain(llm=model, prompt=prompt)
        chain = StuffDocumentsChain(
            llm_chain=llm_chain,
            document_prompt=document_prompt,
            document_variable_name=document_variable_name,
        )
        ```
    """

    llm_chain: LLMChain
    """LLM chain which is called with the formatted document string,
    along with any other inputs."""
    document_prompt: BasePromptTemplate = Field(
        default_factory=lambda: DEFAULT_DOCUMENT_PROMPT,
    )
    """用于格式化每个文档的提示词，会传递给 `format_document`。"""
    document_variable_name: str
    """在 llm_chain 中放置文档的变量名。
    如果 llm_chain 中只有一个变量，此参数无需提供。"""
    document_separator: str = "\n\n"
    """用于连接格式化文档的字符串"""

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="forbid",
    )

    @model_validator(mode="before")
    @classmethod
    def get_default_document_variable_name(cls, values: dict) -> Any:
        """获取默认的文档变量名，如果未提供。

        If only one variable is present in the llm_chain.prompt,
        we can infer that the formatted documents should be passed in
        with this variable name.
        """
        llm_chain_variables = values["llm_chain"].prompt.input_variables
        if "document_variable_name" not in values:
            if len(llm_chain_variables) == 1:
                values["document_variable_name"] = llm_chain_variables[0]
            else:
                msg = (
                    "document_variable_name must be provided if there are "
                    "multiple llm_chain_variables"
                )
                raise ValueError(msg)
        elif values["document_variable_name"] not in llm_chain_variables:
            msg = (
                f"document_variable_name {values['document_variable_name']} was "
                f"not found in llm_chain input_variables: {llm_chain_variables}"
            )
            raise ValueError(msg)
        return values

    @property
    @override
    def input_keys(self) -> list[str]:
        extra_keys = [
            k for k in self.llm_chain.input_keys if k != self.document_variable_name
        ]
        return super().input_keys + extra_keys

    def _get_inputs(self, docs: list[Document], **kwargs: Any) -> dict:
        """从 kwargs 和 docs 构造输入。

        Format and then join all the documents together into one input with name
        `self.document_variable_name`. Also pluck any additional variables
        from **kwargs.

        Args:
            docs: List of documents to format and then join into single input
            **kwargs: additional inputs to chain, will pluck any other required
                arguments from here.

        Returns:
            传递给 LLMChain 的输入字典
        """
                # 根据提示词格式化每个文档
        doc_strings = [format_document(doc, self.document_prompt) for doc in docs]
                # 将文档合并在一起以放入提示词中。
        inputs = {
            k: v
            for k, v in kwargs.items()
            if k in self.llm_chain.prompt.input_variables
        }
        inputs[self.document_variable_name] = self.document_separator.join(doc_strings)
        return inputs

    def prompt_length(self, docs: list[Document], **kwargs: Any) -> int | None:
        """根据传入的文档返回提示词长度。

        此方法可用于调用方判断传入文档列表是否会超过特定提示词长度。在尝试确保提示词大小始终低于某个上下文限制时非常有用。

        Args:
            docs: 用于计算总提示词长度的文档列表。
            **kwargs: 用于获取 LLMChain 输入的其他参数。

        Returns:
            Returns None if the method does not depend on the prompt length,
            otherwise the length of the prompt in tokens.
        """
        inputs = self._get_inputs(docs, **kwargs)
        prompt = self.llm_chain.prompt.format(**inputs)
        return self.llm_chain._get_num_tokens(prompt)  # noqa: SLF001

    def combine_docs(
        self,
        docs: list[Document],
        callbacks: Callbacks = None,
        **kwargs: Any,
    ) -> tuple[str, dict]:
        """将所有文档合并为一个提示词并传递给 LLM。

        Args:
            docs: 要合并成一个变量的文档列表
            callbacks: 可选的回调函数，用于传递
            **kwargs: 用于获取 LLMChain 输入的其他参数。

        Returns:
            第一个返回值是单一字符串输出。第二个返回值是其他键的字典。
        """
        inputs = self._get_inputs(docs, **kwargs)
                # 调用 LLM 的 predict 方法。
        return self.llm_chain.predict(callbacks=callbacks, **inputs), {}

    async def acombine_docs(
        self,
        docs: list[Document],
        callbacks: Callbacks = None,
        **kwargs: Any,
    ) -> tuple[str, dict]:
        """异步将所有文档合并为一个提示词并传递给 LLM。

        Args:
            docs: 要合并成一个变量的文档列表
            callbacks: 可选的回调函数，用于传递
            **kwargs: 用于获取 LLMChain 输入的其他参数。

        Returns:
            第一个返回值是单一字符串输出。第二个返回值是其他键的字典。
        """
        inputs = self._get_inputs(docs, **kwargs)
                # 调用 LLM 的 predict 方法。
        return await self.llm_chain.apredict(callbacks=callbacks, **inputs), {}

    @property
    def _chain_type(self) -> str:
        return "stuff_documents_chain"

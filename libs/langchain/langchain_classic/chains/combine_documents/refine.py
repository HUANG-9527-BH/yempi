"""通过先进行一轮处理，再对更多文档进行精炼来合并文档。"""

from __future__ import annotations

from typing import Any

from langchain_core._api import deprecated
from langchain_core.callbacks import Callbacks
from langchain_core.documents import Document
from langchain_core.prompts import BasePromptTemplate, format_document
from langchain_core.prompts.prompt import PromptTemplate
from pydantic import ConfigDict, Field, model_validator

from langchain_classic.chains.combine_documents.base import (
    BaseCombineDocumentsChain,
)
from langchain_classic.chains.llm import LLMChain


def _get_default_document_prompt() -> PromptTemplate:
    return PromptTemplate(input_variables=["page_content"], template="{page_content}")


@deprecated(
    since="0.3.1",
    removal="2.0.0",
    alternative="langchain.agents.create_agent",
    addendum=(
        "Build new RAG flows with `create_agent` and a retrieval tool. See "
        "https://docs.langchain.com/oss/python/langchain/rag"
    ),
)
class RefineDocumentsChain(BaseCombineDocumentsChain):
    """通过先进行一轮处理，再对更多文档进行精炼来合并文档。

    This algorithm first calls `initial_llm_chain` on the first document, passing
    that first document in with the variable name `document_variable_name`, and
    produces a new variable with the variable name `initial_response_name`.

    Then, it loops over every remaining document. This is called the "refine" step.
    It calls `refine_llm_chain`,
    passing in that document with the variable name `document_variable_name`
    as well as the previous response with the variable name `initial_response_name`.

    Example:
        ```python
        from langchain_classic.chains import RefineDocumentsChain, LLMChain
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
        initial_llm_chain = LLMChain(llm=model, prompt=prompt)
        initial_response_name = "prev_response"
        # The prompt here should take as an input variable the
        # `document_variable_name` as well as `initial_response_name`
        prompt_refine = PromptTemplate.from_template(
            "Here's your first summary: {prev_response}. "
            "Now add to it based on the following context: {context}"
        )
        refine_llm_chain = LLMChain(llm=model, prompt=prompt_refine)
        chain = RefineDocumentsChain(
            initial_llm_chain=initial_llm_chain,
            refine_llm_chain=refine_llm_chain,
            document_prompt=document_prompt,
            document_variable_name=document_variable_name,
            initial_response_name=initial_response_name,
        )
        ```
    """

    initial_llm_chain: LLMChain
    """用于初始文档的 LLM 链。"""
    refine_llm_chain: LLMChain
    """用于精炼时的 LLM 链。"""
    document_variable_name: str
    """在 initial_llm_chain 中放置文档的变量名。
    如果 initial_llm_chain 中只有一个变量，则无需提供此参数。"""
    initial_response_name: str
    """在精炼时格式化初始响应的变量名。"""
    document_prompt: BasePromptTemplate = Field(
        default_factory=_get_default_document_prompt,
    )
    """用于格式化每个文档的提示词，会传递给 `format_document`。"""
    return_intermediate_steps: bool = False
    """在输出中返回精炼步骤的结果。"""

    @property
    def output_keys(self) -> list[str]:
        """期望输入键。"""
        _output_keys = super().output_keys
        if self.return_intermediate_steps:
            _output_keys = [*_output_keys, "intermediate_steps"]
        return _output_keys

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="forbid",
    )

    @model_validator(mode="before")
    @classmethod
    def get_return_intermediate_steps(cls, values: dict) -> Any:
        """为向后兼容性保留。"""
        if "return_refine_steps" in values:
            values["return_intermediate_steps"] = values["return_refine_steps"]
            del values["return_refine_steps"]
        return values

    @model_validator(mode="before")
    @classmethod
    def get_default_document_variable_name(cls, values: dict) -> Any:
        """如果未提供，获取默认的文档变量名。"""
        if "initial_llm_chain" not in values:
            msg = "initial_llm_chain must be provided"
            raise ValueError(msg)

        llm_chain_variables = values["initial_llm_chain"].prompt.input_variables
        if "document_variable_name" not in values:
            if len(llm_chain_variables) == 1:
                values["document_variable_name"] = llm_chain_variables[0]
            else:
                msg = (
                    "document_variable_name must be provided if there are "
                    "multiple llm_chain input_variables"
                )
                raise ValueError(msg)
        elif values["document_variable_name"] not in llm_chain_variables:
            msg = (
                f"document_variable_name {values['document_variable_name']} was "
                f"not found in llm_chain input_variables: {llm_chain_variables}"
            )
            raise ValueError(msg)
        return values

    def combine_docs(
        self,
        docs: list[Document],
        callbacks: Callbacks = None,
        **kwargs: Any,
    ) -> tuple[str, dict]:
        """通过将第一个链映射到所有文档，然后将其填入最终链来合并。

        Args:
            docs: List of documents to combine
            callbacks: Callbacks to be passed through
            **kwargs: additional parameters to be passed to LLM calls (like other
                input variables besides the documents)

        Returns:
            返回的第一个元素是单一字符串输出。第二个元素是其他要返回的键的字典。
        """
        inputs = self._construct_initial_inputs(docs, **kwargs)
        res = self.initial_llm_chain.predict(callbacks=callbacks, **inputs)
        refine_steps = [res]
        for doc in docs[1:]:
            base_inputs = self._construct_refine_inputs(doc, res)
            inputs = {**base_inputs, **kwargs}
            res = self.refine_llm_chain.predict(callbacks=callbacks, **inputs)
            refine_steps.append(res)
        return self._construct_result(refine_steps, res)

    async def acombine_docs(
        self,
        docs: list[Document],
        callbacks: Callbacks = None,
        **kwargs: Any,
    ) -> tuple[str, dict]:
        """通过将第一个链映射到所有文档，然后将其填入最终链来合并。

        Args:
            docs: List of documents to combine
            callbacks: Callbacks to be passed through
            **kwargs: additional parameters to be passed to LLM calls (like other
                input variables besides the documents)

        Returns:
            返回的第一个元素是单一字符串输出。第二个元素是其他要返回的键的字典。
        """
        inputs = self._construct_initial_inputs(docs, **kwargs)
        res = await self.initial_llm_chain.apredict(callbacks=callbacks, **inputs)
        refine_steps = [res]
        for doc in docs[1:]:
            base_inputs = self._construct_refine_inputs(doc, res)
            inputs = {**base_inputs, **kwargs}
            res = await self.refine_llm_chain.apredict(callbacks=callbacks, **inputs)
            refine_steps.append(res)
        return self._construct_result(refine_steps, res)

    def _construct_result(self, refine_steps: list[str], res: str) -> tuple[str, dict]:
        if self.return_intermediate_steps:
            extra_return_dict = {"intermediate_steps": refine_steps}
        else:
            extra_return_dict = {}
        return res, extra_return_dict

    def _construct_refine_inputs(self, doc: Document, res: str) -> dict[str, Any]:
        return {
            self.document_variable_name: format_document(doc, self.document_prompt),
            self.initial_response_name: res,
        }

    def _construct_initial_inputs(
        self,
        docs: list[Document],
        **kwargs: Any,
    ) -> dict[str, Any]:
        base_info = {"page_content": docs[0].page_content}
        base_info.update(docs[0].metadata)
        document_info = {k: base_info[k] for k in self.document_prompt.input_variables}
        base_inputs: dict = {
            self.document_variable_name: self.document_prompt.format(**document_info),
        }
        return {**base_inputs, **kwargs}

    @property
    def _chain_type(self) -> str:
        return "refine_documents_chain"

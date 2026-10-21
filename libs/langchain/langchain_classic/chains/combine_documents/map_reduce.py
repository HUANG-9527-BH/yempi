"""通过先对每个文档应用链，再合并结果来组合文档。"""

from __future__ import annotations

from typing import Any

from langchain_core._api import deprecated
from langchain_core.callbacks import Callbacks
from langchain_core.documents import Document
from langchain_core.runnables.config import RunnableConfig
from langchain_core.utils.pydantic import create_model
from pydantic import BaseModel, ConfigDict, model_validator
from typing_extensions import override

from langchain_classic.chains.combine_documents.base import BaseCombineDocumentsChain
from langchain_classic.chains.combine_documents.reduce import ReduceDocumentsChain
from langchain_classic.chains.llm import LLMChain


@deprecated(
    since="0.3.1",
    removal="2.0.0",
    alternative="langchain.agents.create_agent",
    addendum=(
        "Build new RAG flows with `create_agent` and a retrieval tool. See "
        "https://docs.langchain.com/oss/python/langchain/rag"
    ),
)
class MapReduceDocumentsChain(BaseCombineDocumentsChain):
    """通过先对每个文档应用链，再合并结果来组合文档。

    我们首先单独对每个文档调用 `llm_chain`，传入 `page_content` 和其他任何 kwargs。这是
    `map` 步骤。

    然后在 `reduce` 步骤中处理该 `map` 步骤的结果。这通常应为 ReduceDocumentsChain。

    Example:
        ```python
        from langchain_classic.chains import (
            StuffDocumentsChain,
            LLMChain,
            ReduceDocumentsChain,
            MapReduceDocumentsChain,
        )
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
        # We now define how to combine these summaries
        reduce_prompt = PromptTemplate.from_template(
            "Combine these summaries: {context}"
        )
        reduce_llm_chain = LLMChain(llm=model, prompt=reduce_prompt)
        combine_documents_chain = StuffDocumentsChain(
            llm_chain=reduce_llm_chain,
            document_prompt=document_prompt,
            document_variable_name=document_variable_name,
        )
        reduce_documents_chain = ReduceDocumentsChain(
            combine_documents_chain=combine_documents_chain,
        )
        chain = MapReduceDocumentsChain(
            llm_chain=llm_chain,
            reduce_documents_chain=reduce_documents_chain,
        )
        # If we wanted to, we could also pass in collapse_documents_chain
        # which is specifically aimed at collapsing documents BEFORE
        # the final call.
        prompt = PromptTemplate.from_template("Collapse this content: {context}")
        llm_chain = LLMChain(llm=model, prompt=prompt)
        collapse_documents_chain = StuffDocumentsChain(
            llm_chain=llm_chain,
            document_prompt=document_prompt,
            document_variable_name=document_variable_name,
        )
        reduce_documents_chain = ReduceDocumentsChain(
            combine_documents_chain=combine_documents_chain,
            collapse_documents_chain=collapse_documents_chain,
        )
        chain = MapReduceDocumentsChain(
            llm_chain=llm_chain,
            reduce_documents_chain=reduce_documents_chain,
        )
        ```
    """

    llm_chain: LLMChain
    """要对每个文档单独应用的链。"""
    reduce_documents_chain: BaseCombineDocumentsChain
    """用于减少对每个文档应用 `llm_chain` 后结果的链。
    这通常是 ReduceDocumentChain 或 StuffDocumentChain。"""
    document_variable_name: str
    """在 llm_chain 中放置文档的变量名。
    如果 llm_chain 中只有一个变量，此参数可省略。"""
    return_intermediate_steps: bool = False
    """在输出中返回 map 步骤的结果。"""

    @override
    def get_output_schema(
        self,
        config: RunnableConfig | None = None,
    ) -> type[BaseModel]:
        if self.return_intermediate_steps:
            return create_model(
                "MapReduceDocumentsOutput",
                **{
                    self.output_key: (str, None),
                    "intermediate_steps": (list[str], None),
                },
            )

        return super().get_output_schema(config)

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
    def get_reduce_chain(cls, values: dict) -> Any:
        """为向后兼容性保留。"""
        if "combine_document_chain" in values:
            if "reduce_documents_chain" in values:
                msg = (
                    "Both `reduce_documents_chain` and `combine_document_chain` "
                    "cannot be provided at the same time. `combine_document_chain` "
                    "is deprecated, please only provide `reduce_documents_chain`"
                )
                raise ValueError(msg)
            combine_chain = values["combine_document_chain"]
            collapse_chain = values.get("collapse_document_chain")
            reduce_chain = ReduceDocumentsChain(
                combine_documents_chain=combine_chain,
                collapse_documents_chain=collapse_chain,
            )
            values["reduce_documents_chain"] = reduce_chain
            del values["combine_document_chain"]
            values.pop("collapse_document_chain", None)

        return values

    @model_validator(mode="before")
    @classmethod
    def get_return_intermediate_steps(cls, values: dict) -> Any:
        """为向后兼容性保留。"""
        if "return_map_steps" in values:
            values["return_intermediate_steps"] = values["return_map_steps"]
            del values["return_map_steps"]
        return values

    @model_validator(mode="before")
    @classmethod
    def get_default_document_variable_name(cls, values: dict) -> Any:
        """如果未提供，获取默认的文档变量名。"""
        if "llm_chain" not in values:
            msg = "llm_chain must be provided"
            raise ValueError(msg)

        llm_chain_variables = values["llm_chain"].prompt.input_variables
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

    @property
    def collapse_document_chain(self) -> BaseCombineDocumentsChain:
        """为向后兼容性保留。"""
        if isinstance(self.reduce_documents_chain, ReduceDocumentsChain):
            if self.reduce_documents_chain.collapse_documents_chain:
                return self.reduce_documents_chain.collapse_documents_chain
            return self.reduce_documents_chain.combine_documents_chain
        msg = (
            f"`reduce_documents_chain` is of type "
            f"{type(self.reduce_documents_chain)} so it does not have "
            f"this attribute."
        )
        raise ValueError(msg)

    @property
    def combine_document_chain(self) -> BaseCombineDocumentsChain:
        """为向后兼容性保留。"""
        if isinstance(self.reduce_documents_chain, ReduceDocumentsChain):
            return self.reduce_documents_chain.combine_documents_chain
        msg = (
            f"`reduce_documents_chain` is of type "
            f"{type(self.reduce_documents_chain)} so it does not have "
            f"this attribute."
        )
        raise ValueError(msg)

    def combine_docs(
        self,
        docs: list[Document],
        token_max: int | None = None,
        callbacks: Callbacks = None,
        **kwargs: Any,
    ) -> tuple[str, dict]:
        """以 map reduce 方式组合文档。

        通过先对所有文档映射第一个链，然后减少结果来组合。
        如果文档数量较多，此减少操作可以递归进行。
        """
        map_results = self.llm_chain.apply(
                        # FYI - 这是并行化的，因此速度很快。
            [{self.document_variable_name: d.page_content, **kwargs} for d in docs],
            callbacks=callbacks,
        )
        question_result_key = self.llm_chain.output_key
        result_docs = [
            Document(page_content=r[question_result_key], metadata=docs[i].metadata)
                        # 此操作使用文档中的元数据，以及 `results` 中的文本结果
            for i, r in enumerate(map_results)
        ]
        result, extra_return_dict = self.reduce_documents_chain.combine_docs(
            result_docs,
            token_max=token_max,
            callbacks=callbacks,
            **kwargs,
        )
        if self.return_intermediate_steps:
            intermediate_steps = [r[question_result_key] for r in map_results]
            extra_return_dict["intermediate_steps"] = intermediate_steps
        return result, extra_return_dict

    async def acombine_docs(
        self,
        docs: list[Document],
        token_max: int | None = None,
        callbacks: Callbacks = None,
        **kwargs: Any,
    ) -> tuple[str, dict]:
        """以 map reduce 方式组合文档。

        通过先对所有文档映射第一个链，然后减少结果来组合。
        如果文档数量较多，此减少操作可以递归进行。
        """
        map_results = await self.llm_chain.aapply(
                        # FYI - 这是并行化的，因此速度很快。
            [{self.document_variable_name: d.page_content, **kwargs} for d in docs],
            callbacks=callbacks,
        )
        question_result_key = self.llm_chain.output_key
        result_docs = [
            Document(page_content=r[question_result_key], metadata=docs[i].metadata)
                        # 此操作使用文档中的元数据，以及 `results` 中的文本结果
            for i, r in enumerate(map_results)
        ]
        result, extra_return_dict = await self.reduce_documents_chain.acombine_docs(
            result_docs,
            token_max=token_max,
            callbacks=callbacks,
            **kwargs,
        )
        if self.return_intermediate_steps:
            intermediate_steps = [r[question_result_key] for r in map_results]
            extra_return_dict["intermediate_steps"] = intermediate_steps
        return result, extra_return_dict

    @property
    def _chain_type(self) -> str:
        return "map_reduce_documents_chain"

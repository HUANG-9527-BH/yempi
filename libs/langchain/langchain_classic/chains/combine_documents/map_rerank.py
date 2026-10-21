"""通过先对每个文档应用链，然后重新排序结果来合并文档。"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, cast

from langchain_core._api import deprecated
from langchain_core.callbacks import Callbacks
from langchain_core.documents import Document
from langchain_core.runnables.config import RunnableConfig
from langchain_core.utils.pydantic import create_model
from pydantic import BaseModel, ConfigDict, model_validator
from typing_extensions import Self, override

from langchain_classic.chains.combine_documents.base import BaseCombineDocumentsChain
from langchain_classic.chains.llm import LLMChain
from langchain_classic.output_parsers.regex import RegexParser


@deprecated(
    since="0.3.1",
    removal="2.0.0",
    alternative="langchain.agents.create_agent",
    addendum=(
        "Build new RAG flows with `create_agent` and a retrieval tool. See "
        "https://docs.langchain.com/oss/python/langchain/rag"
    ),
)
class MapRerankDocumentsChain(BaseCombineDocumentsChain):
    r"""通过先对每个文档应用链，然后重新排序结果来合并文档。

    This algorithm calls an LLMChain on each input document. The LLMChain is expected
    to have an OutputParser that parses the result into both an answer (`answer_key`)
    and a score (`rank_key`). The answer with the highest score is then returned.

    Example:
        ```python
        from langchain_classic.chains import MapRerankDocumentsChain, LLMChain
        from langchain_core.prompts import PromptTemplate
        from langchain_openai import OpenAI
        from langchain_classic.output_parsers.regex import RegexParser

        document_variable_name = "context"
        model = OpenAI()
        # The prompt here should take as an input variable the
        # `document_variable_name`
        # The actual prompt will need to be a lot more complex, this is just
        # an example.
        prompt_template = (
            "Use the following context to tell me the chemical formula "
            "for water. Output both your answer and a score of how confident "
            "you are. Context: {context}"
        )
        output_parser = RegexParser(
            regex=r"(.*?)\nScore: (.*)",
            output_keys=["answer", "score"],
        )
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["context"],
            output_parser=output_parser,
        )
        llm_chain = LLMChain(llm=model, prompt=prompt)
        chain = MapRerankDocumentsChain(
            llm_chain=llm_chain,
            document_variable_name=document_variable_name,
            rank_key="score",
            answer_key="answer",
        )
        ```
    """

    llm_chain: LLMChain
    """要对每个文档单独应用的链。"""
    document_variable_name: str
    """LLMChain 中用于放置文档的变量名。
    如果 LLMChain 中只有一个变量，则无需提供此参数。"""
    rank_key: str
    """LLMChain 输出中用于排序的键。"""
    answer_key: str
    """LLMChain 输出中用于返回作为答案的键。"""
    metadata_keys: list[str] | None = None
    """返回所选文档的额外元数据。"""
    return_intermediate_steps: bool = False
    """返回中间步骤。
    中间步骤包括对每个文档调用 llm_chain 的结果。"""

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="forbid",
    )

    @override
    def get_output_schema(
        self,
        config: RunnableConfig | None = None,
    ) -> type[BaseModel]:
        schema: dict[str, Any] = {
            self.output_key: (str, None),
        }
        if self.return_intermediate_steps:
            schema["intermediate_steps"] = (list[str], None)
        if self.metadata_keys:
            schema.update(dict.fromkeys(self.metadata_keys, (Any, None)))

        return create_model("MapRerankOutput", **schema)

    @property
    def output_keys(self) -> list[str]:
        """期望输入键。"""
        _output_keys = super().output_keys
        if self.return_intermediate_steps:
            _output_keys = [*_output_keys, "intermediate_steps"]
        if self.metadata_keys is not None:
            _output_keys += self.metadata_keys
        return _output_keys

    @model_validator(mode="after")
    def validate_llm_output(self) -> Self:
        """验证组合链的输出是一个字典。"""
        output_parser = self.llm_chain.prompt.output_parser
        if not isinstance(output_parser, RegexParser):
            msg = (
                "Output parser of llm_chain should be a RegexParser,"
                f" got {output_parser}"
            )
            raise ValueError(msg)  # noqa: TRY004
        output_keys = output_parser.output_keys
        if self.rank_key not in output_keys:
            msg = (
                f"Got {self.rank_key} as key to rank on, but did not find "
                f"it in the llm_chain output keys ({output_keys})"
            )
            raise ValueError(msg)
        if self.answer_key not in output_keys:
            msg = (
                f"Got {self.answer_key} as key to return, but did not find "
                f"it in the llm_chain output keys ({output_keys})"
            )
            raise ValueError(msg)
        return self

    @model_validator(mode="before")
    @classmethod
    def get_default_document_variable_name(cls, values: dict) -> Any:
        """如果未提供，则获取默认的文档变量名。"""
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

    def combine_docs(
        self,
        docs: list[Document],
        callbacks: Callbacks = None,
        **kwargs: Any,
    ) -> tuple[str, dict]:
        """以映射重排序方式合并文档。

        先将第一个链映射到所有文档上，然后对结果进行重排序。

        Args:
            docs: List of documents to combine
            callbacks: Callbacks to be passed through
            **kwargs: additional parameters to be passed to LLM calls (like other
                input variables besides the documents)

        Returns:
            返回的第一个元素是单一字符串输出。第二个元素是其他要返回的键的字典。
        """
        results = self.llm_chain.apply_and_parse(
                        # 注意 - 此操作已并行化，因此速度很快。
            [{self.document_variable_name: d.page_content, **kwargs} for d in docs],
            callbacks=callbacks,
        )
        return self._process_results(docs, results)

    async def acombine_docs(
        self,
        docs: list[Document],
        callbacks: Callbacks = None,
        **kwargs: Any,
    ) -> tuple[str, dict]:
        """以映射重排序方式合并文档。

        先将第一个链映射到所有文档上，然后对结果进行重排序。

        Args:
            docs: List of documents to combine
            callbacks: Callbacks to be passed through
            **kwargs: additional parameters to be passed to LLM calls (like other
                input variables besides the documents)

        Returns:
            返回的第一个元素是单一字符串输出。第二个元素是其他要返回的键的字典。
        """
        results = await self.llm_chain.aapply_and_parse(
                        # 注意 - 此操作已并行化，因此速度很快。
            [{self.document_variable_name: d.page_content, **kwargs} for d in docs],
            callbacks=callbacks,
        )
        return self._process_results(docs, results)

    def _process_results(
        self,
        docs: list[Document],
        results: Sequence[str | list[str] | dict[str, str]],
    ) -> tuple[str, dict]:
        typed_results = cast("list[dict]", results)
        sorted_res = sorted(
            zip(typed_results, docs, strict=False),
            key=lambda x: -int(x[0][self.rank_key]),
        )
        output, document = sorted_res[0]
        extra_info = {}
        if self.metadata_keys is not None:
            for key in self.metadata_keys:
                extra_info[key] = document.metadata[key]
        if self.return_intermediate_steps:
            extra_info["intermediate_steps"] = results
        return output[self.answer_key], extra_info

    @property
    def _chain_type(self) -> str:
        return "map_rerank_documents_chain"

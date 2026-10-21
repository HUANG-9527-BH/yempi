"""映射-归约链。

将文档拆分，将较小的部分通过一个提示词发送给大语言模型，
然后使用另一个提示词合并结果。
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from langchain_core._api import deprecated
from langchain_core.callbacks import CallbackManagerForChainRun, Callbacks
from langchain_core.documents import Document
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import BasePromptTemplate
from langchain_text_splitters import TextSplitter
from pydantic import ConfigDict

from langchain_classic.chains import ReduceDocumentsChain
from langchain_classic.chains.base import Chain
from langchain_classic.chains.combine_documents.base import BaseCombineDocumentsChain
from langchain_classic.chains.combine_documents.map_reduce import (
    MapReduceDocumentsChain,
)
from langchain_classic.chains.combine_documents.stuff import StuffDocumentsChain
from langchain_classic.chains.llm import LLMChain


@deprecated(
    since="0.2.13",
    removal="2.0.0",
    alternative="langchain.agents.create_agent",
    addendum=(
        "For map-reduce branching, build a LangGraph using the Send API. See "
        "https://docs.langchain.com/oss/python/langgraph/use-graph-api#map-reduce-and-the-send-api"
    ),
)
class MapReduceChain(Chain):
    """映射-归约链。"""

    combine_documents_chain: BaseCombineDocumentsChain
    """用于合并文档的链。"""
    text_splitter: TextSplitter
    """使用的文本分割器。"""
    input_key: str = "input_text"
    output_key: str = "output_text"

    @classmethod
    def from_params(
        cls,
        llm: BaseLanguageModel,
        prompt: BasePromptTemplate,
        text_splitter: TextSplitter,
        callbacks: Callbacks = None,
        combine_chain_kwargs: Mapping[str, Any] | None = None,
        reduce_chain_kwargs: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> MapReduceChain:
        """构建一个使用链进行映射和归约的映射-归约链。"""
        llm_chain = LLMChain(llm=llm, prompt=prompt, callbacks=callbacks)
        stuff_chain = StuffDocumentsChain(
            llm_chain=llm_chain,
            callbacks=callbacks,
            **(reduce_chain_kwargs or {}),
        )
        reduce_documents_chain = ReduceDocumentsChain(
            combine_documents_chain=stuff_chain,
        )
        combine_documents_chain = MapReduceDocumentsChain(
            llm_chain=llm_chain,
            reduce_documents_chain=reduce_documents_chain,
            callbacks=callbacks,
            **(combine_chain_kwargs or {}),
        )
        return cls(
            combine_documents_chain=combine_documents_chain,
            text_splitter=text_splitter,
            callbacks=callbacks,
            **kwargs,
        )

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="forbid",
    )

    @property
    def input_keys(self) -> list[str]:
        """期望输入键。"""
        return [self.input_key]

    @property
    def output_keys(self) -> list[str]:
        """返回输出键。"""
        return [self.output_key]

    def _call(
        self,
        inputs: dict[str, str],
        run_manager: CallbackManagerForChainRun | None = None,
    ) -> dict[str, str]:
        _run_manager = run_manager or CallbackManagerForChainRun.get_noop_manager()
                # 将较大的文本拆分为较小的块。
        doc_text = inputs.pop(self.input_key)
        texts = self.text_splitter.split_text(doc_text)
        docs = [Document(page_content=text) for text in texts]
        _inputs: dict[str, Any] = {
            **inputs,
            self.combine_documents_chain.input_key: docs,
        }
        outputs = self.combine_documents_chain.run(
            _inputs,
            callbacks=_run_manager.get_child(),
        )
        return {self.output_key: outputs}

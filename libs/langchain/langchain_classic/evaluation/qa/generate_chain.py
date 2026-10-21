"""用于生成问答示例的 LLM 链。"""

from __future__ import annotations

from typing import Any

from langchain_core.language_models import BaseLanguageModel
from langchain_core.output_parsers import BaseLLMOutputParser
from pydantic import Field
from typing_extensions import override

from langchain_classic.chains.llm import LLMChain
from langchain_classic.evaluation.qa.generate_prompt import PROMPT
from langchain_classic.output_parsers.regex import RegexParser

_QA_OUTPUT_PARSER = RegexParser(
    regex=r"QUESTION: (.*?)\n+ANSWER: (.*)",
    output_keys=["query", "answer"],
)


class QAGenerateChain(LLMChain):
    """用于生成问答示例的 LLM 链。"""

    output_parser: BaseLLMOutputParser = Field(default=_QA_OUTPUT_PARSER)
    output_key: str = "qa_pairs"

    @classmethod
    @override
    def is_lc_serializable(cls) -> bool:
        return False

    @classmethod
    def from_llm(cls, llm: BaseLanguageModel, **kwargs: Any) -> QAGenerateChain:
        """从 LLM 加载 QA 生成链。"""
        return cls(llm=llm, prompt=PROMPT, **kwargs)

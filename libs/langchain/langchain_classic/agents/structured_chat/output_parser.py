from __future__ import annotations

import json
import logging
import re
from re import Pattern

from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.exceptions import OutputParserException
from langchain_core.language_models import BaseLanguageModel
from pydantic import Field
from typing_extensions import override

from langchain_classic.agents.agent import AgentOutputParser
from langchain_classic.agents.structured_chat.prompt import FORMAT_INSTRUCTIONS
from langchain_classic.output_parsers import OutputFixingParser

logger = logging.getLogger(__name__)


class StructuredChatOutputParser(AgentOutputParser):
    """结构化聊天智能体的输出解析器。"""

    format_instructions: str = FORMAT_INSTRUCTIONS
    """默认格式化指令"""

    pattern: Pattern = re.compile(r"```(?:json\s+)?(\W.*?)```", re.DOTALL)
    """用于解析输出的正则表达式模式。"""

    @override
    def get_format_instructions(self) -> str:
        """返回给定输出解析器的格式化指令。"""
        return self.format_instructions

    @override
    def parse(self, text: str) -> AgentAction | AgentFinish:
        try:
            action_match = self.pattern.search(text)
            if action_match is not None:
                response = json.loads(action_match.group(1).strip(), strict=False)
                if isinstance(response, list):
                                        # GPT Turbo 经常忽略要求仅输出一个动作的指令
                    logger.warning("Got multiple action responses: %s", response)
                    response = response[0]
                if response["action"] == "Final Answer":
                    return AgentFinish({"output": response["action_input"]}, text)
                return AgentAction(
                    response["action"],
                    response.get("action_input", {}),
                    text,
                )
            return AgentFinish({"output": text}, text)
        except Exception as e:
            msg = f"Could not parse LLM output: {text}"
            raise OutputParserException(msg) from e

    @property
    def _type(self) -> str:
        return "structured_chat"


class StructuredChatOutputParserWithRetries(AgentOutputParser):
    """带重试机制的结构化聊天智能体输出解析器。"""

    base_parser: AgentOutputParser = Field(default_factory=StructuredChatOutputParser)
    """要使用的基解析器。"""
    output_fixing_parser: OutputFixingParser | None = None
    """要使用的输出修复解析器。"""

    @override
    def get_format_instructions(self) -> str:
        return FORMAT_INSTRUCTIONS

    @override
    def parse(self, text: str) -> AgentAction | AgentFinish:
        try:
            if self.output_fixing_parser is not None:
                return self.output_fixing_parser.parse(text)
            return self.base_parser.parse(text)
        except Exception as e:
            msg = f"Could not parse LLM output: {text}"
            raise OutputParserException(msg) from e

    @classmethod
    def from_llm(
        cls,
        llm: BaseLanguageModel | None = None,
        base_parser: StructuredChatOutputParser | None = None,
    ) -> StructuredChatOutputParserWithRetries:
        """从语言模型创建一个 StructuredChatOutputParserWithRetries。

        Args:
            llm: 要使用的语言模型。
            base_parser: 可选的 StructuredChatOutputParser。

        Returns:
            StructuredChatOutputParserWithRetries 的实例。
        """
        if llm is not None:
            base_parser = base_parser or StructuredChatOutputParser()
            output_fixing_parser: OutputFixingParser = OutputFixingParser.from_llm(
                llm=llm,
                parser=base_parser,
            )
            return cls(output_fixing_parser=output_fixing_parser)
        if base_parser is not None:
            return cls(base_parser=base_parser)
        return cls()

    @property
    def _type(self) -> str:
        return "structured_chat_with_retries"

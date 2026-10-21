"""一个设计用于对话并使用工具的智能体。"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from langchain_core._api import deprecated
from langchain_core.callbacks import BaseCallbackManager
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import BaseTool
from pydantic import Field
from typing_extensions import override

from langchain_classic._api.deprecation import AGENT_DEPRECATION_WARNING
from langchain_classic.agents.agent import Agent, AgentOutputParser
from langchain_classic.agents.agent_types import AgentType
from langchain_classic.agents.conversational.output_parser import ConvoOutputParser
from langchain_classic.agents.conversational.prompt import (
    FORMAT_INSTRUCTIONS,
    PREFIX,
    SUFFIX,
)
from langchain_classic.agents.utils import validate_tools_single_input
from langchain_classic.chains import LLMChain


@deprecated(
    "0.1.0",
    message=AGENT_DEPRECATION_WARNING,
    removal="2.0.0",
)
class ConversationalAgent(Agent):
    """一个能够进行对话并使用工具的智能体。"""

    ai_prefix: str = "AI"
    """在 AI 输出前使用的前缀。"""
    output_parser: AgentOutputParser = Field(default_factory=ConvoOutputParser)
    """智能体的输出解析器。"""

    @classmethod
    @override
    def _get_default_output_parser(
        cls,
        ai_prefix: str = "AI",
        **kwargs: Any,
    ) -> AgentOutputParser:
        return ConvoOutputParser(ai_prefix=ai_prefix)

    @property
    def _agent_type(self) -> str:
        """返回智能体类型的标识符。"""
        return AgentType.CONVERSATIONAL_REACT_DESCRIPTION

    @property
    def observation_prefix(self) -> str:
        """在观察结果前附加的前缀。

        Returns:
            "Observation: "
        """
        return "Observation: "

    @property
    def llm_prefix(self) -> str:
        """在 LLM 调用前附加的前缀。

        Returns:
            "Thought: "
        """
        return "Thought:"

    @classmethod
    def create_prompt(
        cls,
        tools: Sequence[BaseTool],
        prefix: str = PREFIX,
        suffix: str = SUFFIX,
        format_instructions: str = FORMAT_INSTRUCTIONS,
        ai_prefix: str = "AI",
        human_prefix: str = "Human",
        input_variables: list[str] | None = None,
    ) -> PromptTemplate:
        """以零样本智能体风格创建提示词。

        Args:
            tools: 智能体可访问的工具列表，用于格式化提示词。 prefix:
            在工具列表前放置的字符串。 suffix: 在工具列表后放置的字符串。
            format_instructions: 如何使用工具的说明。 ai_prefix: 在 AI
            输出前使用的字符串。 human_prefix: 在人类输出前使用的字符串。
            input_variables: 最终提示词将期望的输入变量列表。 默认为 `["input",
            "chat_history", "agent_scratchpad"]`。

        Returns:
            一个由此处各部分组装而成的 PromptTemplate。
        """
        tool_strings = "\n".join(
            [f"> {tool.name}: {tool.description}" for tool in tools],
        )
        tool_names = ", ".join([tool.name for tool in tools])
        format_instructions = format_instructions.format(
            tool_names=tool_names,
            ai_prefix=ai_prefix,
            human_prefix=human_prefix,
        )
        template = f"{prefix}\n\n{tool_strings}\n\n{format_instructions}\n\n{suffix}"
        if input_variables is None:
            input_variables = ["input", "chat_history", "agent_scratchpad"]
        return PromptTemplate(template=template, input_variables=input_variables)

    @classmethod
    def _validate_tools(cls, tools: Sequence[BaseTool]) -> None:
        super()._validate_tools(tools)
        validate_tools_single_input(cls.__name__, tools)

    @classmethod
    def from_llm_and_tools(
        cls,
        llm: BaseLanguageModel,
        tools: Sequence[BaseTool],
        callback_manager: BaseCallbackManager | None = None,
        output_parser: AgentOutputParser | None = None,
        prefix: str = PREFIX,
        suffix: str = SUFFIX,
        format_instructions: str = FORMAT_INSTRUCTIONS,
        ai_prefix: str = "AI",
        human_prefix: str = "Human",
        input_variables: list[str] | None = None,
        **kwargs: Any,
    ) -> Agent:
        """从 LLM 和工具构建一个智能体。

        Args:
            llm: 使用的语言模型。
            tools: 要使用的工具列表。
            callback_manager: 使用的回调管理器。
            output_parser: 使用的输出解析器。
            prefix: 提示词中使用的前缀。
            suffix: 提示词中使用的后缀。
            format_instructions: 使用的格式说明。
            ai_prefix: 在 AI 输出前使用的前缀。
            human_prefix: 在人类输出前使用的前缀。
            input_variables: 使用的输入变量。
            **kwargs: 传递给智能体的任何其他关键字参数。

        Returns:
            一个智能体。
        """
        cls._validate_tools(tools)
        prompt = cls.create_prompt(
            tools,
            ai_prefix=ai_prefix,
            human_prefix=human_prefix,
            prefix=prefix,
            suffix=suffix,
            format_instructions=format_instructions,
            input_variables=input_variables,
        )
        llm_chain = LLMChain(
            llm=llm,
            prompt=prompt,
            callback_manager=callback_manager,
        )
        tool_names = [tool.name for tool in tools]
        _output_parser = output_parser or cls._get_default_output_parser(
            ai_prefix=ai_prefix,
        )
        return cls(
            llm_chain=llm_chain,
            allowed_tools=tool_names,
            ai_prefix=ai_prefix,
            output_parser=_output_parser,
            **kwargs,
        )

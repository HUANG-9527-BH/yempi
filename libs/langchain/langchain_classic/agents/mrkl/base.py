"""尝试实现 arxiv.org/pdf/2205.00445.pdf 中描述的 MRKL 系统。"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, NamedTuple

from langchain_core._api import deprecated
from langchain_core.callbacks import BaseCallbackManager
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import BaseTool, Tool
from langchain_core.tools.render import render_text_description
from pydantic import Field
from typing_extensions import override

from langchain_classic._api.deprecation import AGENT_DEPRECATION_WARNING
from langchain_classic.agents.agent import Agent, AgentExecutor, AgentOutputParser
from langchain_classic.agents.agent_types import AgentType
from langchain_classic.agents.mrkl.output_parser import MRKLOutputParser
from langchain_classic.agents.mrkl.prompt import FORMAT_INSTRUCTIONS, PREFIX, SUFFIX
from langchain_classic.agents.utils import validate_tools_single_input
from langchain_classic.chains import LLMChain


class ChainConfig(NamedTuple):
    """用于 MRKL 系统的链的配置。

    Args:
        action_name: 操作名称。
        action: 要调用的操作函数。
        action_description: 操作的描述。
    """

    action_name: str
    action: Callable
    action_description: str


@deprecated(
    "0.1.0",
    message=AGENT_DEPRECATION_WARNING,
    removal="2.0.0",
)
class ZeroShotAgent(Agent):
    """MRKL 链的智能体。

    Args:
        output_parser: 智能体的输出解析器。
    """

    output_parser: AgentOutputParser = Field(default_factory=MRKLOutputParser)

    @classmethod
    @override
    def _get_default_output_parser(cls, **kwargs: Any) -> AgentOutputParser:
        return MRKLOutputParser()

    @property
    def _agent_type(self) -> str:
        """返回智能体类型的标识符。"""
        return AgentType.ZERO_SHOT_REACT_DESCRIPTION

    @property
    def observation_prefix(self) -> str:
        """要附加到观察结果前的前缀。

        Returns:
            "Observation: "
        """
        return "Observation: "

    @property
    def llm_prefix(self) -> str:
        """要附加到 LLM 调用前的前缀。

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
        input_variables: list[str] | None = None,
    ) -> PromptTemplate:
        """以零样本智能体风格创建提示词。

        Args:
            tools: 智能体可访问的工具列表，用于格式化提示词。 prefix:
            在工具列表前放置的字符串。 suffix: 在工具列表后放置的字符串。
            format_instructions: 如何使用工具的说明。 input_variables:
            最终提示词将期望的输入变量列表。


        Returns:
            一个由这些部分组合而成的 PromptTemplate。
        """
        tool_strings = render_text_description(list(tools))
        tool_names = ", ".join([tool.name for tool in tools])
        format_instructions = format_instructions.format(tool_names=tool_names)
        template = f"{prefix}\n\n{tool_strings}\n\n{format_instructions}\n\n{suffix}"
        if input_variables:
            return PromptTemplate(template=template, input_variables=input_variables)
        return PromptTemplate.from_template(template)

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
        input_variables: list[str] | None = None,
        **kwargs: Any,
    ) -> Agent:
        """从 LLM 和工具构建智能体。

        Args:
            llm: 用作智能体 LLM 的语言模型。
            tools: 使用的工具。
            callback_manager: 使用的回调管理器。
            output_parser: 使用的输出解析器。
            prefix: 使用的前缀。
            suffix: 使用的后缀。
            format_instructions: 使用的格式说明。
            input_variables: 使用的输入变量。
            kwargs: 传递给智能体的其他参数。
        """
        cls._validate_tools(tools)
        prompt = cls.create_prompt(
            tools,
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
        _output_parser = output_parser or cls._get_default_output_parser()
        return cls(
            llm_chain=llm_chain,
            allowed_tools=tool_names,
            output_parser=_output_parser,
            **kwargs,
        )

    @classmethod
    def _validate_tools(cls, tools: Sequence[BaseTool]) -> None:
        validate_tools_single_input(cls.__name__, tools)
        if len(tools) == 0:
            msg = (
                f"Got no tools for {cls.__name__}. At least one tool must be provided."
            )
            raise ValueError(msg)
        for tool in tools:
            if tool.description is None:
                msg = (  # type: ignore[unreachable]
                    f"Got a tool {tool.name} without a description. For this agent, "
                    f"a description must always be provided."
                )
                raise ValueError(msg)
        super()._validate_tools(tools)


@deprecated(
    "0.1.0",
    message=AGENT_DEPRECATION_WARNING,
    removal="2.0.0",
)
class MRKLChain(AgentExecutor):
    """实现 MRKL 系统的链。"""

    @classmethod
    def from_chains(
        cls,
        llm: BaseLanguageModel,
        chains: list[ChainConfig],
        **kwargs: Any,
    ) -> AgentExecutor:
        """以用户友好的方式初始化 MRKL 链。

        This is intended to be an easy way to get up and running with the
        MRKL chain.

        Args:
            llm: 用作智能体 LLM 的语言模型。
            chains: MRKL 系统可访问的链。
            **kwargs: 初始化时要传递的参数。

        Returns:
            已初始化的 MRKL 链。
        """
        tools = [
            Tool(
                name=c.action_name,
                func=c.action,
                description=c.action_description,
            )
            for c in chains
        ]
        agent = ZeroShotAgent.from_llm_and_tools(llm, tools)
        return cls(agent=agent, tools=tools, **kwargs)

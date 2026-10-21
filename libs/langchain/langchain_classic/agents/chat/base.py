from collections.abc import Sequence
from typing import Any

from langchain_core._api import deprecated
from langchain_core.agents import AgentAction
from langchain_core.callbacks import BaseCallbackManager
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import BasePromptTemplate
from langchain_core.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain_core.tools import BaseTool
from pydantic import Field
from typing_extensions import override

from langchain_classic._api.deprecation import AGENT_DEPRECATION_WARNING
from langchain_classic.agents.agent import Agent, AgentOutputParser
from langchain_classic.agents.chat.output_parser import ChatOutputParser
from langchain_classic.agents.chat.prompt import (
    FORMAT_INSTRUCTIONS,
    HUMAN_MESSAGE,
    SYSTEM_MESSAGE_PREFIX,
    SYSTEM_MESSAGE_SUFFIX,
)
from langchain_classic.agents.utils import validate_tools_single_input
from langchain_classic.chains.llm import LLMChain


@deprecated(
    "0.1.0",
    message=AGENT_DEPRECATION_WARNING,
    removal="2.0.0",
)
class ChatAgent(Agent):
    """聊天智能体。"""

    output_parser: AgentOutputParser = Field(default_factory=ChatOutputParser)
    """智能体的输出解析器。"""

    @property
    def observation_prefix(self) -> str:
        """在观察结果前附加的前缀。"""
        return "Observation: "

    @property
    def llm_prefix(self) -> str:
        """在 LLM 调用前附加的前缀。"""
        return "Thought:"

    def _construct_scratchpad(
        self,
        intermediate_steps: list[tuple[AgentAction, str]],
    ) -> str:
        agent_scratchpad = super()._construct_scratchpad(intermediate_steps)
        if not isinstance(agent_scratchpad, str):
            msg = "agent_scratchpad should be of type string."
            raise ValueError(msg)  # noqa: TRY004
        if agent_scratchpad:
            return (
                f"This was your previous work "
                f"(but I haven't seen any of it! I only see what "
                f"you return as final answer):\n{agent_scratchpad}"
            )
        return agent_scratchpad

    @classmethod
    @override
    def _get_default_output_parser(cls, **kwargs: Any) -> AgentOutputParser:
        return ChatOutputParser()

    @classmethod
    def _validate_tools(cls, tools: Sequence[BaseTool]) -> None:
        super()._validate_tools(tools)
        validate_tools_single_input(class_name=cls.__name__, tools=tools)

    @property
    def _stop(self) -> list[str]:
        return ["Observation:"]

    @classmethod
    def create_prompt(
        cls,
        tools: Sequence[BaseTool],
        system_message_prefix: str = SYSTEM_MESSAGE_PREFIX,
        system_message_suffix: str = SYSTEM_MESSAGE_SUFFIX,
        human_message: str = HUMAN_MESSAGE,
        format_instructions: str = FORMAT_INSTRUCTIONS,
        input_variables: list[str] | None = None,
    ) -> BasePromptTemplate:
        """从工具列表创建提示词模板。

        Args:
            tools: 工具列表。
            system_message_prefix: 系统消息前缀。
            system_message_suffix: 系统消息后缀。
            human_message: `HumanMessage` 消息。
            format_instructions: 格式说明。
            input_variables: 输入变量。

        Returns:
            一个提示词模板。
        """
        tool_strings = "\n".join([f"{tool.name}: {tool.description}" for tool in tools])
        tool_names = ", ".join([tool.name for tool in tools])
        format_instructions = format_instructions.format(tool_names=tool_names)
        template = (
            f"{system_message_prefix}\n\n"
            f"{tool_strings}\n\n"
            f"{format_instructions}\n\n"
            f"{system_message_suffix}"
        )
        messages = [
            SystemMessagePromptTemplate.from_template(template),
            HumanMessagePromptTemplate.from_template(human_message),
        ]
        if input_variables is None:
            input_variables = ["input", "agent_scratchpad"]
        return ChatPromptTemplate(input_variables=input_variables, messages=messages)

    @classmethod
    def from_llm_and_tools(
        cls,
        llm: BaseLanguageModel,
        tools: Sequence[BaseTool],
        callback_manager: BaseCallbackManager | None = None,
        output_parser: AgentOutputParser | None = None,
        system_message_prefix: str = SYSTEM_MESSAGE_PREFIX,
        system_message_suffix: str = SYSTEM_MESSAGE_SUFFIX,
        human_message: str = HUMAN_MESSAGE,
        format_instructions: str = FORMAT_INSTRUCTIONS,
        input_variables: list[str] | None = None,
        **kwargs: Any,
    ) -> Agent:
        """从 LLM 和工具构建智能体。

        Args:
            llm: 语言模型。
            tools: 工具列表。
            callback_manager: 回调管理器。
            output_parser: 输出解析器。
            system_message_prefix: 系统消息前缀。
            system_message_suffix: 系统消息后缀。
            human_message: `HumanMessage` 消息。
            format_instructions: 格式说明。
            input_variables: 输入变量。
            kwargs: 额外的关键字参数。

        Returns:
            一个智能体。
        """
        cls._validate_tools(tools)
        prompt = cls.create_prompt(
            tools,
            system_message_prefix=system_message_prefix,
            system_message_suffix=system_message_suffix,
            human_message=human_message,
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

    @property
    def _agent_type(self) -> str:
        raise ValueError

"""一个设计用于对话并使用工具的智能体。"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from langchain_core._api import deprecated
from langchain_core.agents import AgentAction
from langchain_core.callbacks import BaseCallbackManager
from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.prompts import BasePromptTemplate
from langchain_core.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
)
from langchain_core.tools import BaseTool
from pydantic import Field
from typing_extensions import override

from langchain_classic.agents.agent import Agent, AgentOutputParser
from langchain_classic.agents.conversational_chat.output_parser import ConvoOutputParser
from langchain_classic.agents.conversational_chat.prompt import (
    PREFIX,
    SUFFIX,
    TEMPLATE_TOOL_RESPONSE,
)
from langchain_classic.agents.utils import validate_tools_single_input
from langchain_classic.chains import LLMChain


@deprecated("0.1.0", alternative="create_json_chat_agent", removal="2.0.0")
class ConversationalChatAgent(Agent):
    """一个设计用于对话并使用工具的智能体。"""

    output_parser: AgentOutputParser = Field(default_factory=ConvoOutputParser)
    """智能体的输出解析器。"""
    template_tool_response: str = TEMPLATE_TOOL_RESPONSE
    """工具响应的模板。"""

    @classmethod
    @override
    def _get_default_output_parser(cls, **kwargs: Any) -> AgentOutputParser:
        return ConvoOutputParser()

    @property
    def _agent_type(self) -> str:
        raise NotImplementedError

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
    def _validate_tools(cls, tools: Sequence[BaseTool]) -> None:
        super()._validate_tools(tools)
        validate_tools_single_input(cls.__name__, tools)

    @classmethod
    def create_prompt(
        cls,
        tools: Sequence[BaseTool],
        system_message: str = PREFIX,
        human_message: str = SUFFIX,
        input_variables: list[str] | None = None,
        output_parser: BaseOutputParser | None = None,
    ) -> BasePromptTemplate:
        """为智能体创建提示词。

        Args:
            tools: 要使用的工具。
            system_message: 要使用的 `SystemMessage`。
            human_message: 要使用的 `HumanMessage`。
            input_variables: 要使用的输入变量。
            output_parser: 要使用的输出解析器。

        Returns:
            一个 `PromptTemplate`。
        """
        tool_strings = "\n".join(
            [f"> {tool.name}: {tool.description}" for tool in tools],
        )
        tool_names = ", ".join([tool.name for tool in tools])
        _output_parser = output_parser or cls._get_default_output_parser()
        format_instructions = human_message.format(
            format_instructions=_output_parser.get_format_instructions(),
        )
        final_prompt = format_instructions.format(
            tool_names=tool_names,
            tools=tool_strings,
        )
        if input_variables is None:
            input_variables = ["input", "chat_history", "agent_scratchpad"]
        messages = [
            SystemMessagePromptTemplate.from_template(system_message),
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessagePromptTemplate.from_template(final_prompt),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
        return ChatPromptTemplate(input_variables=input_variables, messages=messages)

    def _construct_scratchpad(
        self,
        intermediate_steps: list[tuple[AgentAction, str]],
    ) -> list[BaseMessage]:
        """构建草稿板，使智能体能够继续其思考过程。"""
        thoughts: list[BaseMessage] = []
        for action, observation in intermediate_steps:
            thoughts.append(AIMessage(content=action.log))
            human_message = HumanMessage(
                content=self.template_tool_response.format(observation=observation),
            )
            thoughts.append(human_message)
        return thoughts

    @classmethod
    def from_llm_and_tools(
        cls,
        llm: BaseLanguageModel,
        tools: Sequence[BaseTool],
        callback_manager: BaseCallbackManager | None = None,
        output_parser: AgentOutputParser | None = None,
        system_message: str = PREFIX,
        human_message: str = SUFFIX,
        input_variables: list[str] | None = None,
        **kwargs: Any,
    ) -> Agent:
        """从 LLM 和工具构建智能体。

        Args:
            llm: 要使用的语言模型。
            tools: 要使用的工具列表。
            callback_manager: 要使用的回调管理器。
            output_parser: 要使用的输出解析器。
            system_message: 要使用的 `SystemMessage`。
            human_message: 要使用的 `HumanMessage`。
            input_variables: 要使用的输入变量。
            **kwargs: 任何额外参数。

        Returns:
            一个智能体。
        """
        cls._validate_tools(tools)
        _output_parser = output_parser or cls._get_default_output_parser()
        prompt = cls.create_prompt(
            tools,
            system_message=system_message,
            human_message=human_message,
            input_variables=input_variables,
            output_parser=_output_parser,
        )
        llm_chain = LLMChain(
            llm=llm,
            prompt=prompt,
            callback_manager=callback_manager,
        )
        tool_names = [tool.name for tool in tools]
        return cls(
            llm_chain=llm_chain,
            allowed_tools=tool_names,
            output_parser=_output_parser,
            **kwargs,
        )

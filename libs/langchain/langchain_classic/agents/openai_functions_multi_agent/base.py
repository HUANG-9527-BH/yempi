"""模块实现了一个使用 OpenAI API 函数功能的智能体。"""

import json
from collections.abc import Sequence
from json import JSONDecodeError
from typing import Any

from langchain_core._api import deprecated
from langchain_core.agents import AgentAction, AgentActionMessageLog, AgentFinish
from langchain_core.callbacks import BaseCallbackManager, Callbacks
from langchain_core.exceptions import OutputParserException
from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    SystemMessage,
)
from langchain_core.prompts import BasePromptTemplate
from langchain_core.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain_core.prompts.message import BaseMessagePromptTemplate
from langchain_core.tools import BaseTool
from pydantic import model_validator
from typing_extensions import Self

from langchain_classic.agents import BaseMultiActionAgent
from langchain_classic.agents.format_scratchpad.openai_functions import (
    format_to_openai_function_messages,
)

# 为了向后兼容
_FunctionsAgentAction = AgentActionMessageLog


def _parse_ai_message(message: BaseMessage) -> list[AgentAction] | AgentFinish:
    """解析 AI 消息。"""
    if not isinstance(message, AIMessage):
        msg = f"Expected an AI message got {type(message)}"
        raise TypeError(msg)

    function_call = message.additional_kwargs.get("function_call", {})

    if function_call:
        try:
            arguments = json.loads(function_call["arguments"], strict=False)
        except JSONDecodeError as e:
            msg = (
                f"Could not parse tool input: {function_call} because "
                f"the `arguments` is not valid JSON."
            )
            raise OutputParserException(msg) from e

        try:
            tools = arguments["actions"]
        except (TypeError, KeyError) as e:
            msg = (
                f"Could not parse tool input: {function_call} because "
                f"the `arguments` JSON does not contain `actions` key."
            )
            raise OutputParserException(msg) from e

        final_tools: list[AgentAction] = []
        for tool_schema in tools:
            if "action" in tool_schema:
                _tool_input = tool_schema["action"]
            else:
                                # 从模式中移除 action_name
                _tool_input = tool_schema.copy()
                del _tool_input["action_name"]
            function_name = tool_schema["action_name"]

                        # 此处为一种临时方案：
                        # 将工具输入编码为 Open AI 所用的代码中使用了一个特殊变量
                        # 名为 `__arg1`，用于处理不暴露模式且期望单个字符串参数的旧式工具。
                        # 如果存在，则在此处解包该参数。
                        # Open AI 不支持将 JSON 数组作为参数传入。
                        # OpenAI 函数返回单个工具调用
            if "__arg1" in _tool_input:
                tool_input = _tool_input["__arg1"]
            else:
                tool_input = _tool_input

            content_msg = f"responded: {message.content}\n" if message.content else "\n"
            log = f"\nInvoking: `{function_name}` with `{tool_input}`\n{content_msg}\n"
            _tool = _FunctionsAgentAction(
                tool=function_name,
                tool_input=tool_input,
                log=log,
                message_log=[message],
            )
            final_tools.append(_tool)
        return final_tools

    return AgentFinish(
        return_values={"output": message.content},
        log=str(message.content),
    )


_NOT_SET = object()


@deprecated("0.1.0", alternative="create_openai_tools_agent", removal="2.0.0")
class OpenAIMultiFunctionsAgent(BaseMultiActionAgent):
    """由 OpenAI 的函数驱动的智能体。

    Args:
        llm: This should be an instance of ChatOpenAI, specifically a model
            that supports using `functions`.
        tools: The tools this agent has access to.
        prompt: The prompt for this agent, should support agent_scratchpad as one
            of the variables. For an easy way to construct this prompt, use
            `OpenAIMultiFunctionsAgent.create_prompt(...)`
    """

    llm: BaseLanguageModel
    tools: Sequence[BaseTool]
    prompt: BasePromptTemplate

    def get_allowed_tools(self) -> list[str]:
        """获取允许的工具。"""
        return [t.name for t in self.tools]

    @model_validator(mode="after")
    def _validate_prompt(self) -> Self:
        prompt: BasePromptTemplate = self.prompt
        if "agent_scratchpad" not in prompt.input_variables:
            msg = (
                "`agent_scratchpad` should be one of the variables in the prompt, "
                f"got {prompt.input_variables}"
            )
            raise ValueError(msg)
        return self

    @property
    def input_keys(self) -> list[str]:
        """获取输入键。此处输入指用户输入。"""
        return ["input"]

    @property
    def functions(self) -> list[dict]:
        """获取智能体的函数。"""
        enum_vals = [t.name for t in self.tools]
        tool_selection = {
                        # 此处强制其返回的单个工具调用
                        # 本身成为一个工具调用列表。我们通过构建
                        # 一个具有单一参数（即要使用的工具列表）的新工具来实现这一点。
                        # 这是一个自定义项，用于捆绑 action_name
                        # 和 action。我们这样做是因为某些动作
            "name": "tool_selection",
            "description": "A list of actions to take.",
            "parameters": {
                "title": "tool_selection",
                "description": "A list of actions to take.",
                "type": "object",
                "properties": {
                    "actions": {
                        "title": "actions",
                        "type": "array",
                        "items": {
                                                        # 可能具有相同的模式，若无此机制则无法区分它们。
                                                        # 这是要执行的操作名称
                                                        # 这是要执行的操作。
                                                        # 无法区分它们。
                            "title": "tool_call",
                            "type": "object",
                            "properties": {
                                                                # 这是要执行的操作的名称
                                "action_name": {
                                    "title": "action_name",
                                    "enum": enum_vals,
                                    "type": "string",
                                    "description": (
                                        "Name of the action to take. The name "
                                        "provided here should match up with the "
                                        "parameters for the action below."
                                    ),
                                },
                                                                # 这是要采取的操作。
                                "action": {
                                    "title": "Action",
                                    "anyOf": [
                                        {
                                            "title": t.name,
                                            "type": "object",
                                            "properties": t.args,
                                        }
                                        for t in self.tools
                                    ],
                                },
                            },
                            "required": ["action_name", "action"],
                        },
                    },
                },
                "required": ["actions"],
            },
        }
        return [tool_selection]

    def plan(
        self,
        intermediate_steps: list[tuple[AgentAction, str]],
        callbacks: Callbacks = None,
        **kwargs: Any,
    ) -> list[AgentAction] | AgentFinish:
        """根据输入决定下一步操作。

        Args:
            intermediate_steps: Steps the LLM has taken to date,
                along with observations.
            callbacks: Callbacks to use.
            **kwargs: User inputs.

        Returns:
            指定要使用的工具的动作。
        """
        agent_scratchpad = format_to_openai_function_messages(intermediate_steps)
        selected_inputs = {
            k: kwargs[k] for k in self.prompt.input_variables if k != "agent_scratchpad"
        }
        full_inputs = dict(**selected_inputs, agent_scratchpad=agent_scratchpad)
        prompt = self.prompt.format_prompt(**full_inputs)
        messages = prompt.to_messages()
        predicted_message = self.llm.invoke(
            messages,
            functions=self.functions,
            callbacks=callbacks,
        )
        return _parse_ai_message(predicted_message)

    async def aplan(
        self,
        intermediate_steps: list[tuple[AgentAction, str]],
        callbacks: Callbacks = None,
        **kwargs: Any,
    ) -> list[AgentAction] | AgentFinish:
        """异步处理输入，决定下一步操作。

        Args:
            intermediate_steps: Steps the LLM has taken to date,
                along with observations.
            callbacks: Callbacks to use.
            **kwargs: User inputs.

        Returns:
            指定要使用的工具的动作。
        """
        agent_scratchpad = format_to_openai_function_messages(intermediate_steps)
        selected_inputs = {
            k: kwargs[k] for k in self.prompt.input_variables if k != "agent_scratchpad"
        }
        full_inputs = dict(**selected_inputs, agent_scratchpad=agent_scratchpad)
        prompt = self.prompt.format_prompt(**full_inputs)
        messages = prompt.to_messages()
        predicted_message = await self.llm.ainvoke(
            messages,
            functions=self.functions,
            callbacks=callbacks,
        )
        return _parse_ai_message(predicted_message)

    @classmethod
    def create_prompt(
        cls,
        system_message: SystemMessage | None = _NOT_SET,  # type: ignore[assignment]
        extra_prompt_messages: list[BaseMessagePromptTemplate] | None = None,
    ) -> BasePromptTemplate:
        """为该智能体创建提示词。

        Args:
            system_message: 用作系统消息的提示，将作为提示中的第一条消息。
            extra_prompt_messages: 位于系统消息和新的人类输入之间的额外提示消息。

        Returns:
            传递给该智能体的提示词模板。
        """
        _prompts = extra_prompt_messages or []
        system_message_ = (
            system_message
            if system_message is not _NOT_SET
            else SystemMessage(content="You are a helpful AI assistant.")
        )
        messages: list[BaseMessagePromptTemplate | BaseMessage]
        messages = [system_message_] if system_message_ else []

        messages.extend(
            [
                *_prompts,
                HumanMessagePromptTemplate.from_template("{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ],
        )
        return ChatPromptTemplate(messages=messages)

    @classmethod
    def from_llm_and_tools(
        cls,
        llm: BaseLanguageModel,
        tools: Sequence[BaseTool],
        callback_manager: BaseCallbackManager | None = None,
        extra_prompt_messages: list[BaseMessagePromptTemplate] | None = None,
        system_message: SystemMessage | None = _NOT_SET,  # type: ignore[assignment]
        **kwargs: Any,
    ) -> BaseMultiActionAgent:
        """从语言模型和工具构建一个智能体。

        Args:
            llm: 使用的语言模型。 tools: 要使用的工具列表。 callback_manager:
            使用的回调管理器。 extra_prompt_messages: 额外的提示消息。 system_message:
            使用的系统消息。默认为默认系统消息。 kwargs: 其他参数。
        """
        system_message_ = (
            system_message
            if system_message is not _NOT_SET
            else SystemMessage(content="You are a helpful AI assistant.")
        )
        prompt = cls.create_prompt(
            extra_prompt_messages=extra_prompt_messages,
            system_message=system_message_,
        )
        return cls(
            llm=llm,
            prompt=prompt,
            tools=tools,
            callback_manager=callback_manager,
            **kwargs,
        )

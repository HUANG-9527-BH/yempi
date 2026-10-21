"""用于表示智能体动作、观察结果和返回值的模式定义。

!!! warning

    这些模式定义仅用于向后兼容。

!!! warning

    新的智能体应使用
    [`langchain` 库](https://pypi.org/project/langchain/) 构建，该库提供了
    更简单且更灵活的方式来定义智能体。

    参见文档中的[构建智能体](https://docs.langchain.com/oss/python/langchain/agents)。

智能体使用语言模型来选择要执行的动作序列。

基本智能体的工作方式如下：

1. Given a prompt an agent uses an LLM to request an action to take
    (e.g., a tool to run).
2. The agent executes the action (e.g., runs the tool), and receives an observation.
3. The agent returns the observation to the LLM, which can then be used to generate
    the next action.
4. When the agent reaches a stopping condition, it returns a final return value.

智能体本身的模式定义在 `langchain.agents.agent` 中。
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any, Literal

from langchain_core.load.serializable import Serializable
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    FunctionMessage,
    HumanMessage,
)


class AgentAction(Serializable):
    """表示智能体请求执行某个动作。

    该动作包括要执行的工具名称以及传递给该工具的输入。日志用于传递关于该动作的额外信息。
    """

    tool: str
    """要执行的 `Tool` 的名称。"""

    tool_input: str | dict[Any, Any]
    """要传递给 `Tool` 的输入。"""

    log: str
    """关于该动作的日志信息。

    此日志可用于多种用途。首先，它可以用于审计 LLM 实际预测的内容，从而生成此 `(tool,
    tool_input)`。

    其次，它可在后续迭代中向 LLM 展示其先前的思考过程。这在 `(tool, tool_input)`
    无法完整包含 LLM 预测信息时非常有用（例如，任何在 tool/tool_input 之前的
    `thought`）。
    """

    type: Literal["AgentAction"] = "AgentAction"

        # 重写 init 以支持按位置实例化，保持向后兼容性。
    def __init__(
        self, tool: str, tool_input: str | dict[Any, Any], log: str, **kwargs: Any
    ):
        """创建一个 `AgentAction`。

        Args:
            tool: 要执行的工具名称。
            tool_input: 传递给 `Tool` 的输入。
            log: 关于该动作的额外日志信息。
        """
        super().__init__(tool=tool, tool_input=tool_input, log=log, **kwargs)

    @classmethod
    def is_lc_serializable(cls) -> bool:
        """`AgentAction` 是可序列化的。

        Returns:
            `True`
        """
        return True

    @classmethod
    def get_lc_namespace(cls) -> list[str]:
        """获取 LangChain 对象的命名空间。

        Returns:
            `["langchain", "schema", "agent"]`
        """
        return ["langchain", "schema", "agent"]

    @property
    def messages(self) -> Sequence[BaseMessage]:
        """返回与此动作对应的 messages。"""
        return _convert_agent_action_to_messages(self)


class AgentActionMessageLog(AgentAction):
    """表示智能体要执行的动作。

    与 `AgentAction` 类似，但包含由聊天消息组成的日志。

    This is useful when working with `ChatModels`, and is used to reconstruct
    conversation history from the agent's perspective.
    """

    message_log: Sequence[BaseMessage]
    """与 log 类似，可用于传递关于 LLM 在解析出 `(tool, tool_input)`
之前所预测的具体消息的额外信息。

    This is again useful if `(tool, tool_input)` cannot be used to fully recreate the
    LLM prediction, and you need that LLM prediction (for future agent iteration).

    与 `log` 相比，此字段在底层 LLM 为聊天模型（因此返回消息而非字符串）时尤为有用。
    """
        # 忽略类型检查，因为我们正在覆盖来自 AgentAction 的类型。
        # 在这种情况下，这是正确的做法。
        # 类型字面量用于序列化目的。
    type: Literal["AgentActionMessageLog"] = "AgentActionMessageLog"  # type: ignore[assignment]


class AgentStep(Serializable):
    """执行 `AgentAction` 后的结果。"""

    action: AgentAction
    """已执行的 `AgentAction`。"""

    observation: Any
    """该 `AgentAction` 的结果。"""

    @property
    def messages(self) -> Sequence[BaseMessage]:
        """与此观察结果对应的 messages。"""
        return _convert_agent_observation_to_messages(self.action, self.observation)


class AgentFinish(Serializable):
    """`ActionAgent` 的最终返回值。

    当智能体达到停止条件时，返回 `AgentFinish`。
    """

    return_values: dict[Any, Any]
    """返回值的字典。"""

    log: str
    """关于返回值的日志信息。

    This is used to pass along the full LLM prediction, not just the parsed out
    return value.

    For example, if the full LLM prediction was `Final Answer: 2` you may want to just
    return `2` as a return value, but pass along the full string as a `log` (for
    debugging or observability purposes).
    """
    type: Literal["AgentFinish"] = "AgentFinish"

    def __init__(self, return_values: dict[Any, Any], log: str, **kwargs: Any):
        """重写 init 以支持按位置实例化，保持向后兼容性。"""
        super().__init__(return_values=return_values, log=log, **kwargs)

    @classmethod
    def is_lc_serializable(cls) -> bool:
        """返回 `True`，因为此类是可序列化的。"""
        return True

    @classmethod
    def get_lc_namespace(cls) -> list[str]:
        """获取 LangChain 对象的命名空间。

        Returns:
            `["langchain", "schema", "agent"]`
        """
        return ["langchain", "schema", "agent"]

    @property
    def messages(self) -> Sequence[BaseMessage]:
        """与此观察结果对应的 messages。"""
        return [AIMessage(content=self.log)]


def _convert_agent_action_to_messages(
    agent_action: AgentAction,
) -> Sequence[BaseMessage]:
    """将智能体动作转换为消息。

    此代码用于从智能体动作重构原始 AI 消息。

    Args:
        agent_action: 需要转换的智能体动作。

    Returns:
        对应于原始工具调用的 `AIMessage`。
    """
    if isinstance(agent_action, AgentActionMessageLog):
        return agent_action.message_log
    return [AIMessage(content=agent_action.log)]


def _convert_agent_observation_to_messages(
    agent_action: AgentAction, observation: Any
) -> Sequence[BaseMessage]:
    """将智能体动作转换为消息。

    此代码用于从智能体动作重构原始 AI 消息。

    Args:
        agent_action: 需要转换的智能体动作。
        observation: 需要转换为消息的观察结果。

    Returns:
        对应于原始工具调用的 `AIMessage`。
    """
    if isinstance(agent_action, AgentActionMessageLog):
        return [_create_function_message(agent_action, observation)]
    content = observation
    if not isinstance(observation, str):
        try:
            content = json.dumps(observation, ensure_ascii=False)
        except Exception:
            content = str(observation)
    return [HumanMessage(content=content)]


def _create_function_message(
    agent_action: AgentAction, observation: Any
) -> FunctionMessage:
    """将智能体动作和观察结果转换为函数消息。

    Args:
        agent_action: 智能体发出的工具调用请求。
        observation: 工具调用的结果。

    Returns:
        对应于原始工具调用的 `FunctionMessage`。
    """
    if not isinstance(observation, str):
        try:
            content = json.dumps(observation, ensure_ascii=False)
        except Exception:
            content = str(observation)
    else:
        content = observation
    return FunctionMessage(
        name=agent_action.tool,
        content=content,
    )

import json
import logging
from collections.abc import Sequence
from typing import Any

from langchain_core.agents import AgentAction, AgentActionMessageLog
from langchain_core.messages import AIMessage, BaseMessage, FunctionMessage

_logger = logging.getLogger(__name__)


def _convert_agent_action_to_messages(
    agent_action: AgentAction,
    observation: str,
) -> list[BaseMessage]:
    """将智能体动作转换为消息。

    此代码用于从智能体动作重构原始 AI 消息。

    Args:
        agent_action: 要转换的智能体动作。
        observation: 工具调用的结果。

    Returns:
        AIMessage 或之前的消息加上对应于
            原始工具调用的 FunctionMessage
    """
    if isinstance(agent_action, AgentActionMessageLog):
        return [
            *list(agent_action.message_log),
            _create_function_message(agent_action, observation),
        ]
    return [AIMessage(content=agent_action.log)]


def _create_function_message(
    agent_action: AgentAction,
    observation: Any,
) -> FunctionMessage:
    """将智能体动作和观察结果转换为函数消息。

    Args:
        agent_action: 智能体发出的工具调用请求。
        observation: 工具调用的结果。

    Returns:
        对应于原始工具调用的 FunctionMessage。

    Raises:
        ValueError: 如果观察结果无法转换为字符串。
    """
    if not isinstance(observation, str):
        try:
            content = json.dumps(observation, ensure_ascii=False)
        except TypeError:
            content = str(observation)
        except Exception:
            _logger.exception("Unexpected error converting observation to string.")
            content = str(observation)
    else:
        content = observation
    return FunctionMessage(
        name=agent_action.tool,
        content=content,
    )


def format_to_openai_function_messages(
    intermediate_steps: Sequence[tuple[AgentAction, str]],
) -> list[BaseMessage]:
    """将 (AgentAction, 工具输出) 元组转换为 FunctionMessages。

    Args:
        intermediate_steps: LLM 到目前为止采取的步骤，以及观察结果

    Returns:
        发送给 LLM 以进行下一步预测的消息列表
    Raises:
        ValueError: 如果观察结果无法转换为字符串。
    """
    messages = []

    for agent_action, observation in intermediate_steps:
        messages.extend(_convert_agent_action_to_messages(agent_action, observation))

    return messages


# Backwards compatibility
format_to_openai_functions = format_to_openai_function_messages

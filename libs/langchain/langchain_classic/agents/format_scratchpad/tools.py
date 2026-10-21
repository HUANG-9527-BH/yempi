import json
import logging
from collections.abc import Sequence
from typing import Any

from langchain_core.agents import AgentAction
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    ToolMessage,
)

from langchain_classic.agents.output_parsers.tools import ToolAgentAction

_logger = logging.getLogger(__name__)


def _create_tool_message(
    agent_action: ToolAgentAction,
    observation: Any,
) -> ToolMessage:
    """将智能体动作和观察结果转换为工具消息。

    Args:
        agent_action: 智能体发出的工具调用请求。
        observation: 工具调用的结果。

    Returns:
        与原始工具调用对应的 ToolMessage。

    Raises:
        ValueError: 如果无法将观察结果转换为字符串。
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
    return ToolMessage(
        tool_call_id=agent_action.tool_call_id,
        content=content,
        additional_kwargs={"name": agent_action.tool},
    )


def format_to_tool_messages(
    intermediate_steps: Sequence[tuple[AgentAction, str]],
) -> list[BaseMessage]:
    """将 (AgentAction, 工具输出) 元组转换为 `ToolMessage` 对象。

    Args:
        intermediate_steps: LLM 到目前为止采取的步骤及其观察结果。

    Returns:
        发送给 LLM 以进行下一步预测的消息列表。

    """
    messages = []
    for agent_action, observation in intermediate_steps:
        if isinstance(agent_action, ToolAgentAction):
            new_messages = [
                *list(agent_action.message_log),
                _create_tool_message(agent_action, observation),
            ]
            messages.extend([new for new in new_messages if new not in messages])
        else:
            messages.append(AIMessage(content=agent_action.log))
    return messages

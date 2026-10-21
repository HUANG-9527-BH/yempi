from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatGeneration, Generation
from typing_extensions import override

from langchain_classic.agents.agent import MultiActionAgentOutputParser
from langchain_classic.agents.output_parsers.tools import (
    ToolAgentAction,
    parse_ai_message_to_tool_action,
)

OpenAIToolAgentAction = ToolAgentAction


def parse_ai_message_to_openai_tool_action(
    message: BaseMessage,
) -> list[AgentAction] | AgentFinish:
    """解析可能包含 tool_calls 的 AI 消息。"""
    tool_actions = parse_ai_message_to_tool_action(message)
    if isinstance(tool_actions, AgentFinish):
        return tool_actions
    final_actions: list[AgentAction] = []
    for action in tool_actions:
        if isinstance(action, ToolAgentAction):
            final_actions.append(
                OpenAIToolAgentAction(
                    tool=action.tool,
                    tool_input=action.tool_input,
                    log=action.log,
                    message_log=action.message_log,
                    tool_call_id=action.tool_call_id,
                ),
            )
        else:
            final_actions.append(action)
    return final_actions


class OpenAIToolsAgentOutputParser(MultiActionAgentOutputParser):
    """将消息解析为智能体动作或完成状态。

    专用于 OpenAI 模型，因为它依赖于 OpenAI 特定的
    tool_calls 参数来传达要使用的工具。

    如果传入了 tool_calls 参数，则使用该参数获取
    工具名称和工具输入。

    如果未传入，则假定 AIMessage 为最终输出。
    """

    @property
    def _type(self) -> str:
        return "openai-tools-agent-output-parser"

    @override
    def parse_result(
        self,
        result: list[Generation],
        *,
        partial: bool = False,
    ) -> list[AgentAction] | AgentFinish:
        if not isinstance(result[0], ChatGeneration):
            msg = "This output parser only works on ChatGeneration output"
            raise ValueError(msg)  # noqa: TRY004
        message = result[0].message
        return parse_ai_message_to_openai_tool_action(message)

    @override
    def parse(self, text: str) -> list[AgentAction] | AgentFinish:
        msg = "Can only parse messages"
        raise ValueError(msg)

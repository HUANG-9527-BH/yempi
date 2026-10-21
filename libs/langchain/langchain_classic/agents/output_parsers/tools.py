import json
from json import JSONDecodeError

from langchain_core.agents import AgentAction, AgentActionMessageLog, AgentFinish
from langchain_core.exceptions import OutputParserException
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    ToolCall,
)
from langchain_core.outputs import ChatGeneration, Generation
from typing_extensions import override

from langchain_classic.agents.agent import MultiActionAgentOutputParser


class ToolAgentAction(AgentActionMessageLog):
    """智能体工具操作。"""

    tool_call_id: str | None
    """此消息所响应的工具调用。"""


def parse_ai_message_to_tool_action(
    message: BaseMessage,
) -> list[AgentAction] | AgentFinish:
    """解析可能包含工具调用的 AI 消息。"""
    if not isinstance(message, AIMessage):
        msg = f"Expected an AI message got {type(message)}"
        raise TypeError(msg)

    actions: list = []
    if message.tool_calls:
        tool_calls = message.tool_calls
    else:
        if not message.additional_kwargs.get("tool_calls"):
            return AgentFinish(
                return_values={"output": message.content},
                log=str(message.content),
            )
                # 尽力解析
        tool_calls = []
        for tool_call in message.additional_kwargs["tool_calls"]:
            function = tool_call["function"]
            function_name = function["name"]
            try:
                args = json.loads(function["arguments"] or "{}")
                tool_calls.append(
                    ToolCall(
                        type="tool_call",
                        name=function_name,
                        args=args,
                        id=tool_call["id"],
                    ),
                )
            except JSONDecodeError as e:
                msg = (
                    f"Could not parse tool input: {function} because "
                    f"the `arguments` is not valid JSON."
                )
                raise OutputParserException(msg) from e
    for tool_call in tool_calls:
                # 此处的技巧：
                # 将工具输入编码为 Open AI 格式时使用了一个特殊变量
                # 名为 `__arg1`，用于处理不暴露模式且期望单个字符串参数的旧式工具。
                # 如果存在，则在此处展开该参数。
                # Open AI 不支持传递 JSON 数组作为参数。
                # Open AI 不支持将 JSON 数组作为参数传入。
        function_name = tool_call["name"]
        _tool_input = tool_call["args"]
        tool_input = _tool_input.get("__arg1", _tool_input)

        content_msg = f"responded: {message.content}\n" if message.content else "\n"
        log = f"\nInvoking: `{function_name}` with `{tool_input}`\n{content_msg}\n"
        actions.append(
            ToolAgentAction(
                tool=function_name,
                tool_input=tool_input,
                log=log,
                message_log=[message],
                tool_call_id=tool_call["id"],
            ),
        )
    return actions


class ToolsAgentOutputParser(MultiActionAgentOutputParser):
    """将消息解析为智能体操作/完成状态。

    如果传入了 tool_calls 参数，则使用该参数获取
    工具名称和工具输入。

    如果未传入，则假设 AIMessage 为最终输出。
    """

    @property
    def _type(self) -> str:
        return "tools-agent-output-parser"

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
        return parse_ai_message_to_tool_action(message)

    @override
    def parse(self, text: str) -> list[AgentAction] | AgentFinish:
        msg = "Can only parse messages"
        raise ValueError(msg)

import json
from json import JSONDecodeError

from langchain_core.agents import AgentAction, AgentActionMessageLog, AgentFinish
from langchain_core.exceptions import OutputParserException
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
)
from langchain_core.outputs import ChatGeneration, Generation
from typing_extensions import override

from langchain_classic.agents.agent import AgentOutputParser


class OpenAIFunctionsAgentOutputParser(AgentOutputParser):
    """将消息解析为智能体动作/完成状态。

    专用于 OpenAI 模型，因为它依赖于 OpenAI 的特定
    function_call 参数来传达使用哪些工具。

    如果传入了 function_call 参数，则使用该参数获取
    工具和工具输入。

    如果未传入，则假设 AIMessage 为最终输出。
    """

    @property
    def _type(self) -> str:
        return "openai-functions-agent"

    @staticmethod
    def parse_ai_message(message: BaseMessage) -> AgentAction | AgentFinish:
        """解析 AI 消息。"""
        if not isinstance(message, AIMessage):
            msg = f"Expected an AI message got {type(message)}"
            raise TypeError(msg)

        function_call = message.additional_kwargs.get("function_call", {})

        if function_call:
            function_name = function_call["name"]
            try:
                if len(function_call["arguments"].strip()) == 0:
                                        # OpenAI 返回空字符串表示无参数的函数
                    _tool_input = {}
                else:
                                        # 否则返回一个 JSON 对象
                    _tool_input = json.loads(function_call["arguments"], strict=False)
            except JSONDecodeError as e:
                msg = (
                    f"Could not parse tool input: {function_call} because "
                    f"the `arguments` is not valid JSON."
                )
                raise OutputParserException(msg) from e

                        # 此处为一种技巧：
                        # 编码工具输入到 Open AI 的代码使用了一个特殊变量
                        # 名为 `__arg1` 来处理不暴露 schema 且期望单个字符串参数作为输入的旧式工具。
                        # 如果存在，则在此处解包该参数。
                        # Open AI 不支持传递 JSON 数组作为参数。
                        # Open AI 不支持将 JSON 数组作为参数传入。
            if "__arg1" in _tool_input:
                tool_input = _tool_input["__arg1"]
            else:
                tool_input = _tool_input

            content_msg = f"responded: {message.content}\n" if message.content else "\n"
            log = f"\nInvoking: `{function_name}` with `{tool_input}`\n{content_msg}\n"
            return AgentActionMessageLog(
                tool=function_name,
                tool_input=tool_input,
                log=log,
                message_log=[message],
            )

        return AgentFinish(
            return_values={"output": message.content},
            log=str(message.content),
        )

    @override
    def parse_result(
        self,
        result: list[Generation],
        *,
        partial: bool = False,
    ) -> AgentAction | AgentFinish:
        if not isinstance(result[0], ChatGeneration):
            msg = "This output parser only works on ChatGeneration output"
            raise ValueError(msg)  # noqa: TRY004
        message = result[0].message
        return self.parse_ai_message(message)

    @override
    def parse(self, text: str) -> AgentAction | AgentFinish:
        msg = "Can only parse messages"
        raise ValueError(msg)

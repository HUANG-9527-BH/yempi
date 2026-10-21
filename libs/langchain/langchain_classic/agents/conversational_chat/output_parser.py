from __future__ import annotations

from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.exceptions import OutputParserException
from langchain_core.utils.json import parse_json_markdown

from langchain_classic.agents import AgentOutputParser
from langchain_classic.agents.conversational_chat.prompt import FORMAT_INSTRUCTIONS


# 定义一个用于解析对话智能体输出的类
class ConvoOutputParser(AgentOutputParser):
    """对话智能体的输出解析器。"""

    format_instructions: str = FORMAT_INSTRUCTIONS
    """默认格式化指令"""

    def get_format_instructions(self) -> str:
        """返回给定输出解析器的格式化指令。"""
        return self.format_instructions

    def parse(self, text: str) -> AgentAction | AgentFinish:
        """尝试将给定文本解析为 AgentAction 或 AgentFinish。

        Raises:
             如果解析失败，则抛出 OutputParserException。
        """
        try:
                        # 尝试将文本解析为结构化格式（假设为 JSON
                        # 以 markdown 格式存储）
            response = parse_json_markdown(text)

                        # 如果响应包含 'action' 和 'action_input'
            if "action" in response and "action_input" in response:
                action, action_input = response["action"], response["action_input"]

                                # 如果 action 表示最终答案，返回 AgentFinish
                if action == "Final Answer":
                    return AgentFinish({"output": action_input}, text)
                                # 否则，返回带有指定 action 和
                                # input 的 AgentAction
                return AgentAction(action, action_input, text)
                        # 如果响应中缺少必要键，则抛出异常
                        # 如果在解析过程中抛出任何其他异常，也抛出一个
            msg = f"Missing 'action' or 'action_input' in LLM output: {text}"
            raise OutputParserException(msg)
        except Exception as e:
                        # OutputParserException
                        # OutputParserException
            msg = f"Could not parse LLM output: {text}"
            raise OutputParserException(msg) from e

    @property
    def _type(self) -> str:
        return "conversational_chat"

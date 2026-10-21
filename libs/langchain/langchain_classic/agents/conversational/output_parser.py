import re

from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.exceptions import OutputParserException

from langchain_classic.agents.agent import AgentOutputParser
from langchain_classic.agents.conversational.prompt import FORMAT_INSTRUCTIONS


class ConvoOutputParser(AgentOutputParser):
    """对话智能体的输出解析器。"""

    ai_prefix: str = "AI"
    """在 AI 输出前使用的前缀。"""

    format_instructions: str = FORMAT_INSTRUCTIONS
    """默认格式化指令"""

    def get_format_instructions(self) -> str:
        """返回给定输出解析器的格式化指令。"""
        return self.format_instructions

    def parse(self, text: str) -> AgentAction | AgentFinish:
        """将智能体的输出解析为 AgentAction 或 AgentFinish 对象。

        Args:
            text: 需要解析的文本。

        Returns:
            一个 AgentAction 或 AgentFinish 对象。
        """
        if f"{self.ai_prefix}:" in text:
            return AgentFinish(
                {"output": text.rsplit(f"{self.ai_prefix}:", maxsplit=1)[-1].strip()},
                text,
            )
        regex = r"Action: (.*?)[\n]*Action Input: ([\s\S]*)"
        match = re.search(regex, text, re.DOTALL)
        if not match:
            msg = f"Could not parse LLM output: `{text}`"
            raise OutputParserException(msg)
        action = match.group(1)
        action_input = match.group(2)
        return AgentAction(action.strip(), action_input.strip(" ").strip('"'), text)

    @property
    def _type(self) -> str:
        return "conversational"

import json
import re
from re import Pattern

from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.exceptions import OutputParserException

from langchain_classic.agents.agent import AgentOutputParser
from langchain_classic.agents.chat.prompt import FORMAT_INSTRUCTIONS

FINAL_ANSWER_ACTION = "Final Answer:"


class ChatOutputParser(AgentOutputParser):
    """聊天智能体的输出解析器。"""

    format_instructions: str = FORMAT_INSTRUCTIONS
    """默认格式化指令"""

    pattern: Pattern = re.compile(r"^.*?`{3}(?:json)?\n(.*?)`{3}.*?$", re.DOTALL)
    """用于解析输出的正则表达式模式。"""

    def get_format_instructions(self) -> str:
        """返回给定输出解析器的格式化指令。"""
        return self.format_instructions

    def parse(self, text: str) -> AgentAction | AgentFinish:
        """将智能体的输出解析为 AgentAction 或 AgentFinish 对象。

        Args:
            text: 需要解析的文本。

        Returns:
            AgentAction 或 AgentFinish 对象。

        Raises:
            OutputParserException: 如果无法解析输出。
            ValueError: 如果无法找到动作。
        """
        includes_answer = FINAL_ANSWER_ACTION in text
        try:
            found = self.pattern.search(text)
            if not found:
                                # 快速失败解析最终答案。
                msg = "action not found"
                raise ValueError(msg)
            action = found.group(1)
            response = json.loads(action.strip())
            includes_action = "action" in response
            if includes_answer and includes_action:
                msg = (
                    "Parsing LLM output produced a final answer "
                    f"and a parse-able action: {text}"
                )
                raise OutputParserException(msg)
            return AgentAction(
                response["action"],
                response.get("action_input", {}),
                text,
            )

        except Exception as exc:
            if not includes_answer:
                msg = f"Could not parse LLM output: {text}"
                raise OutputParserException(msg) from exc
            output = text.rsplit(FINAL_ANSWER_ACTION, maxsplit=1)[-1].strip()
            return AgentFinish({"output": output}, text)

    @property
    def _type(self) -> str:
        return "chat"

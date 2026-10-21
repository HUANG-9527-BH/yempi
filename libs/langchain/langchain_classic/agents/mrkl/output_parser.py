import re

from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.exceptions import OutputParserException

from langchain_classic.agents.agent import AgentOutputParser
from langchain_classic.agents.mrkl.prompt import FORMAT_INSTRUCTIONS

FINAL_ANSWER_ACTION = "Final Answer:"
MISSING_ACTION_AFTER_THOUGHT_ERROR_MESSAGE = (
    "Invalid Format: Missing 'Action:' after 'Thought:"
)
MISSING_ACTION_INPUT_AFTER_ACTION_ERROR_MESSAGE = (
    "Invalid Format: Missing 'Action Input:' after 'Action:'"
)
FINAL_ANSWER_AND_PARSABLE_ACTION_ERROR_MESSAGE = (
    "Parsing LLM output produced both a final answer and a parse-able action:"
)


class MRKLOutputParser(AgentOutputParser):
    """聊天智能体的 MRKL 输出解析器。"""

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
            AgentAction 或 AgentFinish 对象。

        Raises:
            OutputParserException: 如果无法解析输出。
        """
        includes_answer = FINAL_ANSWER_ACTION in text
        regex = r"Action\s*\d*\s*:[\s]*(.*?)Action\s*\d*\s*Input\s*\d*\s*:[\s]*(.*)"
        action_match = re.search(regex, text, re.DOTALL)
        if action_match and includes_answer:
            if text.find(FINAL_ANSWER_ACTION) < text.find(action_match.group(0)):
                                # 如果最终答案在幻觉之前，返回最终答案
                start_index = text.find(FINAL_ANSWER_ACTION) + len(FINAL_ANSWER_ACTION)
                end_index = text.find("\n\n", start_index)
                return AgentFinish(
                    {"output": text[start_index:end_index].strip()},
                    text[:end_index],
                )
            msg = f"{FINAL_ANSWER_AND_PARSABLE_ACTION_ERROR_MESSAGE}: {text}"
            raise OutputParserException(msg)

        if action_match:
            action = action_match.group(1).strip()
            action_input = action_match.group(2)
            tool_input = action_input.strip(" ")
                        # 确保如果是格式正确的 SQL 查询，不移除任何结尾的 " 字符
            if tool_input.startswith("SELECT ") is False:
                tool_input = tool_input.strip('"')

            return AgentAction(action, tool_input, text)

        if includes_answer:
            return AgentFinish(
                {"output": text.rsplit(FINAL_ANSWER_ACTION, maxsplit=1)[-1].strip()},
                text,
            )

        if not re.search(r"Action\s*\d*\s*:[\s]*(.*?)", text, re.DOTALL):
            msg = f"Could not parse LLM output: `{text}`"
            raise OutputParserException(
                msg,
                observation=MISSING_ACTION_AFTER_THOUGHT_ERROR_MESSAGE,
                llm_output=text,
                send_to_llm=True,
            )
        if not re.search(
            r"[\s]*Action\s*\d*\s*Input\s*\d*\s*:[\s]*(.*)",
            text,
            re.DOTALL,
        ):
            msg = f"Could not parse LLM output: `{text}`"
            raise OutputParserException(
                msg,
                observation=MISSING_ACTION_INPUT_AFTER_ACTION_ERROR_MESSAGE,
                llm_output=text,
                send_to_llm=True,
            )
        msg = f"Could not parse LLM output: `{text}`"
        raise OutputParserException(msg)

    @property
    def _type(self) -> str:
        return "mrkl"

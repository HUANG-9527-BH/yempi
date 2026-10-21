import re
from typing import Literal

from langchain_core.agents import AgentAction, AgentFinish
from pydantic import Field
from typing_extensions import override

from langchain_classic.agents import AgentOutputParser


def _unescape(text: str) -> str:
    """将自定义标签分隔符转换回 XML 标签。"""
    replacements = {
        "[[tool]]": "<tool>",
        "[[/tool]]": "</tool>",
        "[[tool_input]]": "<tool_input>",
        "[[/tool_input]]": "</tool_input>",
        "[[observation]]": "<observation>",
        "[[/observation]]": "</observation>",
    }
    for repl, orig in replacements.items():
        text = text.replace(repl, orig)
    return text


class XMLAgentOutputParser(AgentOutputParser):
    """从 XML 格式的智能体输出中解析工具调用和最终答案。

    This parser extracts structured information from XML tags to determine whether
    an agent should perform a tool action or provide a final answer. It includes
    built-in escaping support to safely handle tool names and inputs
    containing XML special characters.

    Args:
        escape_format: The escaping format to use when parsing XML content.
            Supports 'minimal' which uses custom delimiters like [[tool]] to replace
            XML tags within content, preventing parsing conflicts.
            Use 'minimal' if using a corresponding encoding format that uses
            the _escape function when formatting the output (e.g., with format_xml).

    预期格式：
        工具调用（返回 AgentAction）:
            <tool>search</tool>
            <tool_input>what is 2 + 2</tool_input>

        最终答案（返回 AgentFinish）:
            <final_answer>The answer is 4</final_answer>

    !!! note 最小转义允许包含 XML 标签的工具名称被安全表示。 例如，名为
    `search<tool>nested</tool>` 的工具在 XML 中会被转义为
    `search[[tool]]nested[[/tool]]`，并在解析时自动还原。

    Raises:
        ValueError: 如果输入不匹配任一预期的 XML 格式或包含格式错误的 XML 结构。
    """

    escape_format: Literal["minimal"] | None = Field(default="minimal")
    """用于转义 XML 字符的格式。

    minimal - 使用自定义分隔符替换内容中的 XML 标签，
    防止解析冲突。这是当前唯一支持的格式。

    None - 不应用任何转义，可能导致解析冲突。
    """

    @override
    def parse(self, text: str) -> AgentAction | AgentFinish:
                # 检查是否为工具调用
        tool_matches = re.findall(r"<tool>(.*?)</tool>", text, re.DOTALL)
        if tool_matches:
            if len(tool_matches) != 1:
                msg = (
                    f"Malformed tool invocation: expected exactly one <tool> block, "
                    f"but found {len(tool_matches)}."
                )
                raise ValueError(msg)
            _tool = tool_matches[0]

                        # 匹配可选的工具输入
            input_matches = re.findall(
                r"<tool_input>(.*?)</tool_input>", text, re.DOTALL
            )
            if len(input_matches) > 1:
                msg = (
                    f"Malformed tool invocation: expected at most one <tool_input> "
                    f"block, but found {len(input_matches)}."
                )
                raise ValueError(msg)
            _tool_input = input_matches[0] if input_matches else ""

                        # 如果使用最小转义格式，则进行反转义
            if self.escape_format == "minimal":
                _tool = _unescape(_tool)
                _tool_input = _unescape(_tool_input)

            return AgentAction(tool=_tool, tool_input=_tool_input, log=text)
                # 检查是否为最终答案
        if "<final_answer>" in text and "</final_answer>" in text:
            matches = re.findall(r"<final_answer>(.*?)</final_answer>", text, re.DOTALL)
            if len(matches) != 1:
                msg = (
                    "Malformed output: expected exactly one "
                    "<final_answer>...</final_answer> block."
                )
                raise ValueError(msg)
            answer = matches[0]
                        # 在最终答案中反转义自定义分隔符
            if self.escape_format == "minimal":
                answer = _unescape(answer)
            return AgentFinish(return_values={"output": answer}, log=text)
        msg = (
            "Malformed output: expected either a tool invocation "
            "or a final answer in XML format."
        )
        raise ValueError(msg)

    @override
    def get_format_instructions(self) -> str:
        raise NotImplementedError

    @property
    def _type(self) -> str:
        return "xml-agent"

from typing import Literal

from langchain_core.agents import AgentAction


def _escape(xml: str) -> str:
    """将 XML 标签替换为自定义的安全分隔符。"""
    replacements = {
        "<tool>": "[[tool]]",
        "</tool>": "[[/tool]]",
        "<tool_input>": "[[tool_input]]",
        "</tool_input>": "[[/tool_input]]",
        "<observation>": "[[observation]]",
        "</observation>": "[[/observation]]",
    }
    for orig, repl in replacements.items():
        xml = xml.replace(orig, repl)
    return xml


def format_xml(
    intermediate_steps: list[tuple[AgentAction, str]],
    *,
    escape_format: Literal["minimal"] | None = "minimal",
) -> str:
    """将中间步骤格式化为 XML。

    Args:
        intermediate_steps: The intermediate steps.
        escape_format: The escaping format to use. Currently only 'minimal' is
            supported, which replaces XML tags with custom delimiters to prevent
            conflicts.

    Returns:
        以 XML 格式表示的中间步骤。
    """
    log = ""
    for action, observation in intermediate_steps:
        if escape_format == "minimal":
                        # 使用自定义分隔符转义工具名称和输入中的 XML 标签
            tool = _escape(action.tool)
            tool_input = _escape(str(action.tool_input))
            observation_ = _escape(str(observation))
        else:
            tool = action.tool
            tool_input = str(action.tool_input)
            observation_ = str(observation)
        log += (
            f"<tool>{tool}</tool><tool_input>{tool_input}"
            f"</tool_input><observation>{observation_}</observation>"
        )
    return log

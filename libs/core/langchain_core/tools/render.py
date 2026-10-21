"""用于渲染工具的实用工具。"""

from __future__ import annotations

from collections.abc import Callable
from inspect import signature

from langchain_core.tools.base import BaseTool

ToolsRenderer = Callable[[list[BaseTool]], str]


def render_text_description(tools: list[BaseTool]) -> str:
    """以纯文本格式渲染工具名称和描述。

    Args:
        tools: 要渲染的工具。

    Returns:
        渲染后的文本。

    输出格式如下：

    ```txt
    search: This tool is used for search
    calculator: This tool is used for math
    ```
    """
    descriptions = []
    for tool in tools:
        if hasattr(tool, "func") and tool.func:
            sig = signature(tool.func)
            description = f"{tool.name}{sig} - {tool.description}"
        else:
            description = f"{tool.name} - {tool.description}"

        descriptions.append(description)
    return "\n".join(descriptions)


def render_text_description_and_args(tools: list[BaseTool]) -> str:
    """以纯文本格式渲染工具名称、描述和参数。

    Args:
        tools: 要渲染的工具。

    Returns:
        渲染后的文本。

    输出格式如下：

    ```txt
    search: This tool is used for search, args: {"query": {"type": "string"}}
    calculator: This tool is used for math, \
    args: {"expression": {"type": "string"}}
    ```
    """
    tool_strings = []
    for tool in tools:
        args_schema = str(tool.args)
        if hasattr(tool, "func") and tool.func:
            sig = signature(tool.func)
            description = f"{tool.name}{sig} - {tool.description}"
        else:
            description = f"{tool.name} - {tool.description}"
        tool_strings.append(f"{description}, args: {args_schema}")
    return "\n".join(tool_strings)

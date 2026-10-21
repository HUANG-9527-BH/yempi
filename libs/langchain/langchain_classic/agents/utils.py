from collections.abc import Sequence

from langchain_core.tools import BaseTool


def validate_tools_single_input(class_name: str, tools: Sequence[BaseTool]) -> None:
    """验证单输入工具。

    Args:
        class_name: 类的名称。
        tools: 要验证的工具列表。

    Raises:
        ValueError: 如果在 tools 中发现多输入工具。
    """
    for tool in tools:
        if not tool.is_single_input:
            msg = f"{class_name} does not support multi-input tool {tool.name}."
            raise ValueError(msg)

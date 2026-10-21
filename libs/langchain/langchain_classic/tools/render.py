"""将工具渲染为传递给大语言模型的不同方法。

根据您使用的语言模型和提示策略，
您可能希望以不同方式渲染工具。
此模块包含各种渲染工具的方法。
"""

# # 为了向后兼容
from langchain_core.tools import (
    render_text_description,
    render_text_description_and_args,
)
from langchain_core.utils.function_calling import (
    convert_to_openai_function as format_tool_to_openai_function,
)
from langchain_core.utils.function_calling import (
    convert_to_openai_tool as format_tool_to_openai_tool,
)

__all__ = [
    "format_tool_to_openai_function",
    "format_tool_to_openai_tool",
    "render_text_description",
    "render_text_description_and_args",
]

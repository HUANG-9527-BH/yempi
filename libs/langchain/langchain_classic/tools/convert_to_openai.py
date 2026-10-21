from langchain_core.utils.function_calling import (
    convert_to_openai_function as format_tool_to_openai_function,
)

# 为向后兼容性考虑
__all__ = ["format_tool_to_openai_function"]

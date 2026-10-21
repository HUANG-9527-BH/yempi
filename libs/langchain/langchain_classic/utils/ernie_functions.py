from typing import TYPE_CHECKING, Any

from langchain_classic._api import create_importer

if TYPE_CHECKING:
    from langchain_community.utils.ernie_functions import (
        FunctionDescription,
        ToolDescription,
        convert_pydantic_to_ernie_function,
        convert_pydantic_to_ernie_tool,
    )

# 创建一种动态查找已弃用导入的方式。
# 用于整合引发弃用警告和处理可选导入的逻辑。
# 处理可选导入。
DEPRECATED_LOOKUP = {
    "FunctionDescription": "langchain_community.utils.ernie_functions",
    "ToolDescription": "langchain_community.utils.ernie_functions",
    "convert_pydantic_to_ernie_function": "langchain_community.utils.ernie_functions",
    "convert_pydantic_to_ernie_tool": "langchain_community.utils.ernie_functions",
}

_import_attribute = create_importer(__package__, deprecated_lookups=DEPRECATED_LOOKUP)


def __getattr__(name: str) -> Any:
    """动态查找属性。"""
    return _import_attribute(name)


__all__ = [
    "FunctionDescription",
    "ToolDescription",
    "convert_pydantic_to_ernie_function",
    "convert_pydantic_to_ernie_tool",
]

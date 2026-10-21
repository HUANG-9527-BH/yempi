from typing import TYPE_CHECKING, Any

from langchain_classic._api import create_importer

if TYPE_CHECKING:
    from langchain_community.chains.ernie_functions.base import (
        convert_python_function_to_ernie_function,
        convert_to_ernie_function,
        create_ernie_fn_chain,
        create_ernie_fn_runnable,
        create_structured_output_chain,
        create_structured_output_runnable,
        get_ernie_output_parser,
    )

# 创建一种动态查找已弃用导入的方式。
# 用于统一处理引发弃用警告的逻辑和
# 处理可选导入。
DEPRECATED_LOOKUP = {
    "convert_python_function_to_ernie_function": (
        "langchain_community.chains.ernie_functions.base"
    ),
    "convert_to_ernie_function": "langchain_community.chains.ernie_functions.base",
    "create_ernie_fn_chain": "langchain_community.chains.ernie_functions.base",
    "create_ernie_fn_runnable": "langchain_community.chains.ernie_functions.base",
    "create_structured_output_chain": "langchain_community.chains.ernie_functions.base",
    "create_structured_output_runnable": (
        "langchain_community.chains.ernie_functions.base"
    ),
    "get_ernie_output_parser": "langchain_community.chains.ernie_functions.base",
}

_import_attribute = create_importer(__package__, deprecated_lookups=DEPRECATED_LOOKUP)


def __getattr__(name: str) -> Any:
    """动态查找属性。"""
    return _import_attribute(name)


__all__ = [
    "convert_python_function_to_ernie_function",
    "convert_to_ernie_function",
    "create_ernie_fn_chain",
    "create_ernie_fn_runnable",
    "create_structured_output_chain",
    "create_structured_output_runnable",
    "get_ernie_output_parser",
]

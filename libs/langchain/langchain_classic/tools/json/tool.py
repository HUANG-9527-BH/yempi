"""此模块提供对 LangChain 中已弃用 JSON 工具的动态访问。

通过将引用如 `JsonGetValueTool`、`JsonListKeysTool` 和 `JsonSpec` 转发到
`langchain_community.tools` 命名空间内的更新位置，确保向后兼容性。

This setup allows legacy code to continue working while guiding developers
toward using the updated module paths.
"""

from typing import TYPE_CHECKING, Any

from langchain_classic._api import create_importer

if TYPE_CHECKING:
    from langchain_community.tools import JsonGetValueTool, JsonListKeysTool
    from langchain_community.tools.json.tool import JsonSpec

# 创建一种动态查找已弃用导入的方式。
# 用于合并逻辑以发出弃用警告并
# 处理可选导入。
DEPRECATED_LOOKUP = {
    "JsonSpec": "langchain_community.tools.json.tool",
    "JsonListKeysTool": "langchain_community.tools",
    "JsonGetValueTool": "langchain_community.tools",
}

_import_attribute = create_importer(__package__, deprecated_lookups=DEPRECATED_LOOKUP)


def __getattr__(name: str) -> Any:
    """动态从更新的模块路径中检索属性。

    此方法用于在运行时解析已弃用的属性导入，并将其转发到新位置。

    Args:
        name: 要导入的属性名称。

    Returns:
        从适当的更新模块中解析出的属性。
    """
    return _import_attribute(name)


__all__ = [
    "JsonGetValueTool",
    "JsonListKeysTool",
    "JsonSpec",
]

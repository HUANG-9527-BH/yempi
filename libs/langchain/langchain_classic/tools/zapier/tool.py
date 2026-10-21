"""此模块在 LangChain 中提供对已弃用的 Zapier 工具的动态访问。

通过将 `ZapierNLAListActions` 和 `ZapierNLARunAction` 等引用转发到
`langchain_community.tools` 包中的新位置，支持向后兼容性。

Developers using older import paths will continue to function, while LangChain
internally redirects access to the newer, supported module structure.
"""

from typing import TYPE_CHECKING, Any

from langchain_classic._api import create_importer

if TYPE_CHECKING:
    from langchain_community.tools import ZapierNLAListActions, ZapierNLARunAction

# 创建一种动态查找已弃用导入的方式。
# 用于整合引发弃用警告和
# 处理可选导入的逻辑。
DEPRECATED_LOOKUP = {
    "ZapierNLARunAction": "langchain_community.tools",
    "ZapierNLAListActions": "langchain_community.tools",
}

_import_attribute = create_importer(__package__, deprecated_lookups=DEPRECATED_LOOKUP)


def __getattr__(name: str) -> Any:
    """动态从更新后的模块路径中检索属性。

    此方法用于在运行时解析已弃用的属性导入，并将其转发到新位置。

    Args:
        name: 要导入的属性名称。

    Returns:
        从相应更新后的模块中解析出的属性。
    """
    return _import_attribute(name)


__all__ = [
    "ZapierNLAListActions",
    "ZapierNLARunAction",
]

"""此模块提供对已弃用的 Jira 工具的动态访问。

When attributes like `JiraAction` are accessed, they are redirected to their new
locations in `langchain_community.tools`. This ensures backward compatibility
while warning developers about deprecation.

Attributes:
    JiraAction (已弃用): 从 langchain_community.tools 动态加载。
"""

from typing import TYPE_CHECKING, Any

from langchain_classic._api import create_importer

if TYPE_CHECKING:
    from langchain_community.tools import JiraAction

# 创建一种动态查找已弃用导入的方式。
# 用于合并逻辑以引发弃用警告并
# 处理可选导入。
DEPRECATED_LOOKUP = {"JiraAction": "langchain_community.tools"}

_import_attribute = create_importer(__package__, deprecated_lookups=DEPRECATED_LOOKUP)


def __getattr__(name: str) -> Any:
    """从更新的模块路径中动态检索属性。

    Args:
        name: 要导入的属性名称。

    Returns:
        从更新路径解析出的属性。
    """
    return _import_attribute(name)


__all__ = [
    "JiraAction",
]

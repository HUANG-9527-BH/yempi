from typing import TYPE_CHECKING, Any

from langchain_classic._api import create_importer

if TYPE_CHECKING:
    from langchain_community.utilities.clickup import (
        ClickupAPIWrapper,
        Component,
        CUList,
        Member,
        Space,
        Task,
        Team,
    )

# 创建一种动态查找已弃用导入的方式。
# 用于统一处理引发弃用警告和
# 处理可选导入的逻辑。
DEPRECATED_LOOKUP = {
    "Component": "langchain_community.utilities.clickup",
    "Task": "langchain_community.utilities.clickup",
    "CUList": "langchain_community.utilities.clickup",
    "Member": "langchain_community.utilities.clickup",
    "Team": "langchain_community.utilities.clickup",
    "Space": "langchain_community.utilities.clickup",
    "ClickupAPIWrapper": "langchain_community.utilities.clickup",
}

_import_attribute = create_importer(__package__, deprecated_lookups=DEPRECATED_LOOKUP)


def __getattr__(name: str) -> Any:
    """动态查找属性。"""
    return _import_attribute(name)


__all__ = [
    "CUList",
    "ClickupAPIWrapper",
    "Component",
    "Member",
    "Space",
    "Task",
    "Team",
]

"""O365 工具。"""

from typing import TYPE_CHECKING, Any

from langchain_classic._api import create_importer

if TYPE_CHECKING:
    from langchain_community.tools import (
        O365CreateDraftMessage,
        O365SearchEmails,
        O365SearchEvents,
        O365SendEvent,
        O365SendMessage,
    )

# 创建一种动态查找已弃用导入的方式。
# 用于整合引发弃用警告和
# 处理可选导入的逻辑。
DEPRECATED_LOOKUP = {
    "O365SearchEmails": "langchain_community.tools",
    "O365SearchEvents": "langchain_community.tools",
    "O365CreateDraftMessage": "langchain_community.tools",
    "O365SendMessage": "langchain_community.tools",
    "O365SendEvent": "langchain_community.tools",
}

_import_attribute = create_importer(__package__, deprecated_lookups=DEPRECATED_LOOKUP)


def __getattr__(name: str) -> Any:
    """动态查找属性。"""
    return _import_attribute(name)


__all__ = [
    "O365CreateDraftMessage",
    "O365SearchEmails",
    "O365SearchEvents",
    "O365SendEvent",
    "O365SendMessage",
]

from typing import TYPE_CHECKING, Any

from langchain_classic._api import create_importer

if TYPE_CHECKING:
    from langchain_community.tools import O365SearchEmails
    from langchain_community.tools.office365.messages_search import SearchEmailsInput

# 创建一种动态查找已弃用导入的方式。
# 用于统一处理弃用警告的逻辑并
# 处理可选导入。
DEPRECATED_LOOKUP = {
    "SearchEmailsInput": "langchain_community.tools.office365.messages_search",
    "O365SearchEmails": "langchain_community.tools",
}

_import_attribute = create_importer(__package__, deprecated_lookups=DEPRECATED_LOOKUP)


def __getattr__(name: str) -> Any:
    """动态查找属性。"""
    return _import_attribute(name)


__all__ = [
    "O365SearchEmails",
    "SearchEmailsInput",
]

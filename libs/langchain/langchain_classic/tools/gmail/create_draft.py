from typing import TYPE_CHECKING, Any

from langchain_classic._api import create_importer

if TYPE_CHECKING:
    from langchain_community.tools import GmailCreateDraft
    from langchain_community.tools.gmail.create_draft import CreateDraftSchema

# 创建一种动态查找已弃用导入的方式。
# 用于合并处理引发弃用警告和
# 处理可选导入的逻辑。
DEPRECATED_LOOKUP = {
    "CreateDraftSchema": "langchain_community.tools.gmail.create_draft",
    "GmailCreateDraft": "langchain_community.tools",
}

_import_attribute = create_importer(__package__, deprecated_lookups=DEPRECATED_LOOKUP)


def __getattr__(name: str) -> Any:
    """动态查找属性。"""
    return _import_attribute(name)


__all__ = [
    "CreateDraftSchema",
    "GmailCreateDraft",
]

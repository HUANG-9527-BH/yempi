from typing import TYPE_CHECKING, Any

from langchain_classic._api import create_importer

if TYPE_CHECKING:
    from langchain_community.document_loaders import (
        TelegramChatApiLoader,
        TelegramChatFileLoader,
    )
    from langchain_community.document_loaders.telegram import (
        concatenate_rows,
        text_to_docs,
    )

# 创建一种动态查找已弃用导入的方式。
# 用于统一处理引发弃用警告和
# 处理可选导入的逻辑。
DEPRECATED_LOOKUP = {
    "concatenate_rows": "langchain_community.document_loaders.telegram",
    "TelegramChatFileLoader": "langchain_community.document_loaders",
    "text_to_docs": "langchain_community.document_loaders.telegram",
    "TelegramChatApiLoader": "langchain_community.document_loaders",
}

_import_attribute = create_importer(__package__, deprecated_lookups=DEPRECATED_LOOKUP)


def __getattr__(name: str) -> Any:
    """动态查找属性。"""
    return _import_attribute(name)


__all__ = [
    "TelegramChatApiLoader",
    "TelegramChatFileLoader",
    "concatenate_rows",
    "text_to_docs",
]

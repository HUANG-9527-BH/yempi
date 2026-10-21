from typing import Any

from langchain_core.document_loaders import Blob, BlobLoader

from langchain_classic._api import create_importer

# 创建一种动态查找已弃用导入的方式。
# 用于整合引发弃用警告的逻辑和
# 处理可选导入的逻辑。
DEPRECATED_LOOKUP = {
    "Blob": "langchain_community.document_loaders",
    "BlobLoader": "langchain_community.document_loaders",
}

_import_attribute = create_importer(__package__, deprecated_lookups=DEPRECATED_LOOKUP)


def __getattr__(name: str) -> Any:
    """动态查找属性。"""
    return _import_attribute(name)


__all__ = [
    "Blob",
    "BlobLoader",
]

from typing import TYPE_CHECKING, Any

from langchain_core.document_loaders import Blob, BlobLoader

from langchain_classic._api import create_importer

if TYPE_CHECKING:
    from langchain_community.document_loaders import (
        FileSystemBlobLoader,
        YoutubeAudioLoader,
    )

# 创建一种动态查找已弃用导入的方式。
# 用于合并处理引发弃用警告的逻辑，
# 以及处理可选导入。
DEPRECATED_LOOKUP = {
    "BlobLoader": "langchain_community.document_loaders",
    "Blob": "langchain_community.document_loaders",
    "FileSystemBlobLoader": "langchain_community.document_loaders",
    "YoutubeAudioLoader": "langchain_community.document_loaders",
}

_import_attribute = create_importer(__package__, deprecated_lookups=DEPRECATED_LOOKUP)


def __getattr__(name: str) -> Any:
    """动态查找属性。"""
    return _import_attribute(name)


__all__ = [
    "Blob",
    "BlobLoader",
    "FileSystemBlobLoader",
    "YoutubeAudioLoader",
]

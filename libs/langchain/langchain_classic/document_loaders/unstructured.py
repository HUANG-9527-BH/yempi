from typing import TYPE_CHECKING, Any

from langchain_classic._api import create_importer

if TYPE_CHECKING:
    from langchain_community.document_loaders import (
        UnstructuredAPIFileIOLoader,
        UnstructuredAPIFileLoader,
        UnstructuredFileIOLoader,
        UnstructuredFileLoader,
    )
    from langchain_community.document_loaders.unstructured import (
        UnstructuredBaseLoader,
        get_elements_from_api,
        satisfies_min_unstructured_version,
        validate_unstructured_version,
    )

# 创建一种动态查找已弃用导入的方式。
# 用于统一处理引发弃用警告的逻辑和
# 处理可选导入。
DEPRECATED_LOOKUP = {
    "satisfies_min_unstructured_version": (
        "langchain_community.document_loaders.unstructured"
    ),
    "validate_unstructured_version": (
        "langchain_community.document_loaders.unstructured"
    ),
    "UnstructuredBaseLoader": "langchain_community.document_loaders.unstructured",
    "UnstructuredFileLoader": "langchain_community.document_loaders",
    "get_elements_from_api": "langchain_community.document_loaders.unstructured",
    "UnstructuredAPIFileLoader": "langchain_community.document_loaders",
    "UnstructuredFileIOLoader": "langchain_community.document_loaders",
    "UnstructuredAPIFileIOLoader": "langchain_community.document_loaders",
}

_import_attribute = create_importer(__package__, deprecated_lookups=DEPRECATED_LOOKUP)


def __getattr__(name: str) -> Any:
    """动态查找属性。"""
    return _import_attribute(name)


__all__ = [
    "UnstructuredAPIFileIOLoader",
    "UnstructuredAPIFileLoader",
    "UnstructuredBaseLoader",
    "UnstructuredFileIOLoader",
    "UnstructuredFileLoader",
    "get_elements_from_api",
    "satisfies_min_unstructured_version",
    "validate_unstructured_version",
]

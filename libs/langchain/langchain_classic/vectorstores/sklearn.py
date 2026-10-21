from typing import TYPE_CHECKING, Any

from langchain_classic._api import create_importer

if TYPE_CHECKING:
    from langchain_community.vectorstores import SKLearnVectorStore
    from langchain_community.vectorstores.sklearn import (
        BaseSerializer,
        BsonSerializer,
        JsonSerializer,
        ParquetSerializer,
        SKLearnVectorStoreException,
    )

# 创建一种动态查找已弃用导入的方式。
# 用于统一处理引发弃用警告和
# 处理可选导入的逻辑。
DEPRECATED_LOOKUP = {
    "BaseSerializer": "langchain_community.vectorstores.sklearn",
    "JsonSerializer": "langchain_community.vectorstores.sklearn",
    "BsonSerializer": "langchain_community.vectorstores.sklearn",
    "ParquetSerializer": "langchain_community.vectorstores.sklearn",
    "SKLearnVectorStoreException": "langchain_community.vectorstores.sklearn",
    "SKLearnVectorStore": "langchain_community.vectorstores",
}

_import_attribute = create_importer(__package__, deprecated_lookups=DEPRECATED_LOOKUP)


def __getattr__(name: str) -> Any:
    """动态查找属性。"""
    return _import_attribute(name)


__all__ = [
    "BaseSerializer",
    "BsonSerializer",
    "JsonSerializer",
    "ParquetSerializer",
    "SKLearnVectorStore",
    "SKLearnVectorStoreException",
]

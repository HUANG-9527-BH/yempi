from typing import TYPE_CHECKING, Any

from langchain_classic._api import create_importer

if TYPE_CHECKING:
    from langchain_community.vectorstores import PGEmbedding
    from langchain_community.vectorstores.pgembedding import (
        CollectionStore,
        EmbeddingStore,
        QueryResult,
    )

# 创建一种动态查找已弃用导入的方式。
# 用于整合引发弃用警告和
# 处理可选导入的逻辑。
DEPRECATED_LOOKUP = {
    "CollectionStore": "langchain_community.vectorstores.pgembedding",
    "EmbeddingStore": "langchain_community.vectorstores.pgembedding",
    "QueryResult": "langchain_community.vectorstores.pgembedding",
    "PGEmbedding": "langchain_community.vectorstores",
}

_import_attribute = create_importer(__package__, deprecated_lookups=DEPRECATED_LOOKUP)


def __getattr__(name: str) -> Any:
    """动态查找属性。"""
    return _import_attribute(name)


__all__ = [
    "CollectionStore",
    "EmbeddingStore",
    "PGEmbedding",
    "QueryResult",
]

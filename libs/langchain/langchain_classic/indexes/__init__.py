"""**索引**。

**Index** is used to avoid writing duplicated content
into the vectostore and to avoid over-writing content if it's unchanged.

索引还支持：

* 从数据创建知识图谱。

* 支持从 LangChain 文档加载器到向量存储的索引工作流。

Importantly, Index keeps on working even if the content being written is derived
via a set of transformations from some source content (e.g., indexing children
documents that were derived from parent documents by chunking.)
"""

from typing import TYPE_CHECKING, Any

from langchain_core.indexing.api import IndexingResult, aindex, index

from langchain_classic._api import create_importer
from langchain_classic.indexes._sql_record_manager import SQLRecordManager
from langchain_classic.indexes.vectorstore import VectorstoreIndexCreator

if TYPE_CHECKING:
    from langchain_community.graphs.index_creator import GraphIndexCreator


# 创建一种动态查找已弃用导入的方式。
# 用于统一处理弃用警告的逻辑并
# 处理可选导入。
DEPRECATED_LOOKUP = {
    "GraphIndexCreator": "langchain_community.graphs.index_creator",
}

_import_attribute = create_importer(__package__, deprecated_lookups=DEPRECATED_LOOKUP)


def __getattr__(name: str) -> Any:
    """动态查找属性。"""
    return _import_attribute(name)


__all__ = [
    "GraphIndexCreator",
    "IndexingResult",
    "SQLRecordManager",
    "VectorstoreIndexCreator",
        # 保持有序
    "aindex",
    "index",
]

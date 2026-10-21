from typing import TYPE_CHECKING, Any

from langchain_classic._api import create_importer

if TYPE_CHECKING:
    from langchain_community.vectorstores import ElasticsearchStore
    from langchain_community.vectorstores.elasticsearch import (
        ApproxRetrievalStrategy,
        BaseRetrievalStrategy,
        ExactRetrievalStrategy,
        SparseRetrievalStrategy,
    )

# 创建一种动态查找已弃用导入的方式。
# 用于统一处理引发弃用警告和
# 处理可选导入的逻辑。
DEPRECATED_LOOKUP = {
    "BaseRetrievalStrategy": "langchain_community.vectorstores.elasticsearch",
    "ApproxRetrievalStrategy": "langchain_community.vectorstores.elasticsearch",
    "ExactRetrievalStrategy": "langchain_community.vectorstores.elasticsearch",
    "SparseRetrievalStrategy": "langchain_community.vectorstores.elasticsearch",
    "ElasticsearchStore": "langchain_community.vectorstores",
}

_import_attribute = create_importer(__package__, deprecated_lookups=DEPRECATED_LOOKUP)


def __getattr__(name: str) -> Any:
    """动态查找属性。"""
    return _import_attribute(name)


__all__ = [
    "ApproxRetrievalStrategy",
    "BaseRetrievalStrategy",
    "ElasticsearchStore",
    "ExactRetrievalStrategy",
    "SparseRetrievalStrategy",
]

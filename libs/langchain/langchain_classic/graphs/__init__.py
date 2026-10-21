"""**Graphs** 为图数据库提供了自然语言接口。"""

from typing import TYPE_CHECKING, Any

from langchain_classic._api import create_importer

if TYPE_CHECKING:
    from langchain_community.graphs import (
        ArangoGraph,
        FalkorDBGraph,
        HugeGraph,
        KuzuGraph,
        MemgraphGraph,
        NebulaGraph,
        Neo4jGraph,
        NeptuneGraph,
        NetworkxEntityGraph,
        RdfGraph,
    )


# 创建一种动态查找已弃用导入的方式。
# 用于整合引发弃用警告的逻辑并
# 处理可选导入。
DEPRECATED_LOOKUP = {
    "MemgraphGraph": "langchain_community.graphs",
    "NetworkxEntityGraph": "langchain_community.graphs",
    "Neo4jGraph": "langchain_community.graphs",
    "NebulaGraph": "langchain_community.graphs",
    "NeptuneGraph": "langchain_community.graphs",
    "KuzuGraph": "langchain_community.graphs",
    "HugeGraph": "langchain_community.graphs",
    "RdfGraph": "langchain_community.graphs",
    "ArangoGraph": "langchain_community.graphs",
    "FalkorDBGraph": "langchain_community.graphs",
}

_import_attribute = create_importer(__package__, deprecated_lookups=DEPRECATED_LOOKUP)


def __getattr__(name: str) -> Any:
    """动态查找属性。"""
    return _import_attribute(name)


__all__ = [
    "ArangoGraph",
    "FalkorDBGraph",
    "HugeGraph",
    "KuzuGraph",
    "MemgraphGraph",
    "NebulaGraph",
    "Neo4jGraph",
    "NeptuneGraph",
    "NetworkxEntityGraph",
    "RdfGraph",
]

from typing import TYPE_CHECKING, Any

from langchain_classic._api import create_importer

if TYPE_CHECKING:
    from langchain_community.chains.graph_qa.gremlin import (
        GRAPHDB_SPARQL_FIX_TEMPLATE,
        INTERMEDIATE_STEPS_KEY,
        GremlinQAChain,
        extract_gremlin,
    )

# 创建一种动态查找已弃用导入的方式。
# 用于合并引发弃用警告和处理可选导入的逻辑。
# 处理可选导入。
DEPRECATED_LOOKUP = {
    "GRAPHDB_SPARQL_FIX_TEMPLATE": "langchain_community.chains.graph_qa.gremlin",
    "GremlinQAChain": "langchain_community.chains.graph_qa.gremlin",
    "INTERMEDIATE_STEPS_KEY": "langchain_community.chains.graph_qa.gremlin",
    "extract_gremlin": "langchain_community.chains.graph_qa.gremlin",
}

_import_attribute = create_importer(__package__, deprecated_lookups=DEPRECATED_LOOKUP)


def __getattr__(name: str) -> Any:
    """动态查找属性。"""
    return _import_attribute(name)


__all__ = [
    "GRAPHDB_SPARQL_FIX_TEMPLATE",
    "INTERMEDIATE_STEPS_KEY",
    "GremlinQAChain",
    "extract_gremlin",
]

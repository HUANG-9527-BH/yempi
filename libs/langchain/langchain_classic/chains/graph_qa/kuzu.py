from typing import TYPE_CHECKING, Any

from langchain_classic._api import create_importer

if TYPE_CHECKING:
    from langchain_community.chains.graph_qa.kuzu import (
        KuzuQAChain,
        extract_cypher,
        remove_prefix,
    )

# 创建一种动态查找已弃用导入的方式。
# 用于整合引发弃用警告和
# 处理可选导入的逻辑。
DEPRECATED_LOOKUP = {
    "KuzuQAChain": "langchain_community.chains.graph_qa.kuzu",
    "extract_cypher": "langchain_community.chains.graph_qa.kuzu",
    "remove_prefix": "langchain_community.chains.graph_qa.kuzu",
}

_import_attribute = create_importer(__package__, deprecated_lookups=DEPRECATED_LOOKUP)


def __getattr__(name: str) -> Any:
    """动态查找属性。"""
    return _import_attribute(name)


__all__ = ["KuzuQAChain", "extract_cypher", "remove_prefix"]

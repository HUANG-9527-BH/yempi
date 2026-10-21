from typing import TYPE_CHECKING, Any

from langchain_classic._api import create_importer

if TYPE_CHECKING:
    from langchain_community.chains.graph_qa.cypher import (
        CYPHER_GENERATION_PROMPT,
        INTERMEDIATE_STEPS_KEY,
        GraphCypherQAChain,
        construct_schema,
        extract_cypher,
    )

# 创建一种动态查找已弃用导入的方式。
# 用于合并引发弃用警告和处理可选导入的逻辑。
# 处理可选导入。
DEPRECATED_LOOKUP = {
    "GraphCypherQAChain": "langchain_community.chains.graph_qa.cypher",
    "INTERMEDIATE_STEPS_KEY": "langchain_community.chains.graph_qa.cypher",
    "construct_schema": "langchain_community.chains.graph_qa.cypher",
    "extract_cypher": "langchain_community.chains.graph_qa.cypher",
    "CYPHER_GENERATION_PROMPT": "langchain_community.chains.graph_qa.cypher",
}

_import_attribute = create_importer(__package__, deprecated_lookups=DEPRECATED_LOOKUP)


def __getattr__(name: str) -> Any:
    """动态查找属性。"""
    return _import_attribute(name)


__all__ = [
    "CYPHER_GENERATION_PROMPT",
    "INTERMEDIATE_STEPS_KEY",
    "GraphCypherQAChain",
    "construct_schema",
    "extract_cypher",
]

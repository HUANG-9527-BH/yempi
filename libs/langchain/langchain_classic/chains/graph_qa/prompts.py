from typing import TYPE_CHECKING, Any

from langchain_classic._api import create_importer

if TYPE_CHECKING:
    from langchain_community.chains.graph_qa.prompts import (
        AQL_FIX_TEMPLATE,
        AQL_GENERATION_TEMPLATE,
        AQL_QA_TEMPLATE,
        CYPHER_GENERATION_PROMPT,
        CYPHER_GENERATION_TEMPLATE,
        CYPHER_QA_PROMPT,
        CYPHER_QA_TEMPLATE,
        GRAPHDB_QA_TEMPLATE,
        GRAPHDB_SPARQL_FIX_TEMPLATE,
        GRAPHDB_SPARQL_GENERATION_TEMPLATE,
        GREMLIN_GENERATION_TEMPLATE,
        KUZU_EXTRA_INSTRUCTIONS,
        KUZU_GENERATION_TEMPLATE,
        NEBULAGRAPH_EXTRA_INSTRUCTIONS,
        NEPTUNE_OPENCYPHER_EXTRA_INSTRUCTIONS,
        NEPTUNE_OPENCYPHER_GENERATION_SIMPLE_TEMPLATE,
        NEPTUNE_OPENCYPHER_GENERATION_TEMPLATE,
        NGQL_GENERATION_TEMPLATE,
        SPARQL_GENERATION_SELECT_TEMPLATE,
        SPARQL_GENERATION_UPDATE_TEMPLATE,
        SPARQL_INTENT_TEMPLATE,
        SPARQL_QA_TEMPLATE,
    )

# 创建一种动态查找已弃用导入的方式。
# 用于整合引发弃用警告和
# 处理可选导入的逻辑。
DEPRECATED_LOOKUP = {
    "AQL_FIX_TEMPLATE": "langchain_community.chains.graph_qa.prompts",
    "AQL_GENERATION_TEMPLATE": "langchain_community.chains.graph_qa.prompts",
    "AQL_QA_TEMPLATE": "langchain_community.chains.graph_qa.prompts",
    "CYPHER_GENERATION_TEMPLATE": "langchain_community.chains.graph_qa.prompts",
    "CYPHER_QA_TEMPLATE": "langchain_community.chains.graph_qa.prompts",
    "CYPHER_QA_PROMPT": "langchain_community.chains.graph_qa.prompts",
    "CYPHER_GENERATION_PROMPT": "langchain_community.chains.graph_qa.prompts",
    "GRAPHDB_QA_TEMPLATE": "langchain_community.chains.graph_qa.prompts",
    "GRAPHDB_SPARQL_FIX_TEMPLATE": "langchain_community.chains.graph_qa.prompts",
    "GRAPHDB_SPARQL_GENERATION_TEMPLATE": "langchain_community.chains.graph_qa.prompts",
    "GREMLIN_GENERATION_TEMPLATE": "langchain_community.chains.graph_qa.prompts",
    "KUZU_EXTRA_INSTRUCTIONS": "langchain_community.chains.graph_qa.prompts",
    "KUZU_GENERATION_TEMPLATE": "langchain_community.chains.graph_qa.prompts",
    "NEBULAGRAPH_EXTRA_INSTRUCTIONS": "langchain_community.chains.graph_qa.prompts",
    "NEPTUNE_OPENCYPHER_EXTRA_INSTRUCTIONS": (
        "langchain_community.chains.graph_qa.prompts"
    ),
    "NEPTUNE_OPENCYPHER_GENERATION_SIMPLE_TEMPLATE": (
        "langchain_community.chains.graph_qa.prompts"
    ),
    "NEPTUNE_OPENCYPHER_GENERATION_TEMPLATE": (
        "langchain_community.chains.graph_qa.prompts"
    ),
    "NGQL_GENERATION_TEMPLATE": "langchain_community.chains.graph_qa.prompts",
    "SPARQL_GENERATION_SELECT_TEMPLATE": "langchain_community.chains.graph_qa.prompts",
    "SPARQL_GENERATION_UPDATE_TEMPLATE": "langchain_community.chains.graph_qa.prompts",
    "SPARQL_INTENT_TEMPLATE": "langchain_community.chains.graph_qa.prompts",
    "SPARQL_QA_TEMPLATE": "langchain_community.chains.graph_qa.prompts",
}

_import_attribute = create_importer(__package__, deprecated_lookups=DEPRECATED_LOOKUP)


def __getattr__(name: str) -> Any:
    """动态查找属性。"""
    return _import_attribute(name)


__all__ = [
    "AQL_FIX_TEMPLATE",
    "AQL_GENERATION_TEMPLATE",
    "AQL_QA_TEMPLATE",
    "CYPHER_GENERATION_PROMPT",
    "CYPHER_GENERATION_TEMPLATE",
    "CYPHER_QA_PROMPT",
    "CYPHER_QA_TEMPLATE",
    "GRAPHDB_QA_TEMPLATE",
    "GRAPHDB_SPARQL_FIX_TEMPLATE",
    "GRAPHDB_SPARQL_GENERATION_TEMPLATE",
    "GREMLIN_GENERATION_TEMPLATE",
    "KUZU_EXTRA_INSTRUCTIONS",
    "KUZU_GENERATION_TEMPLATE",
    "NEBULAGRAPH_EXTRA_INSTRUCTIONS",
    "NEPTUNE_OPENCYPHER_EXTRA_INSTRUCTIONS",
    "NEPTUNE_OPENCYPHER_GENERATION_SIMPLE_TEMPLATE",
    "NEPTUNE_OPENCYPHER_GENERATION_TEMPLATE",
    "NGQL_GENERATION_TEMPLATE",
    "SPARQL_GENERATION_SELECT_TEMPLATE",
    "SPARQL_GENERATION_UPDATE_TEMPLATE",
    "SPARQL_INTENT_TEMPLATE",
    "SPARQL_QA_TEMPLATE",
]

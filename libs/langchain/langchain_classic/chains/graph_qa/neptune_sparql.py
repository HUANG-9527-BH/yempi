from typing import TYPE_CHECKING, Any

from langchain_classic._api import create_importer

if TYPE_CHECKING:
    from langchain_community.chains.graph_qa.neptune_sparql import (
        INTERMEDIATE_STEPS_KEY,
        SPARQL_GENERATION_TEMPLATE,
        NeptuneSparqlQAChain,
        extract_sparql,
    )

# 创建一种动态查找已弃用导入的方式。
# 用于统一处理引发弃用警告和
# 处理可选导入的逻辑。
DEPRECATED_LOOKUP = {
    "INTERMEDIATE_STEPS_KEY": "langchain_community.chains.graph_qa.neptune_sparql",
    "NeptuneSparqlQAChain": "langchain_community.chains.graph_qa.neptune_sparql",
    "SPARQL_GENERATION_TEMPLATE": "langchain_community.chains.graph_qa.neptune_sparql",
    "extract_sparql": "langchain_community.chains.graph_qa.neptune_sparql",
}

_import_attribute = create_importer(__package__, deprecated_lookups=DEPRECATED_LOOKUP)


def __getattr__(name: str) -> Any:
    """动态查找属性。"""
    return _import_attribute(name)


__all__ = [
    "INTERMEDIATE_STEPS_KEY",
    "SPARQL_GENERATION_TEMPLATE",
    "NeptuneSparqlQAChain",
    "extract_sparql",
]

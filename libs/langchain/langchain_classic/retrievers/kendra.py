from typing import TYPE_CHECKING, Any

from langchain_classic._api import create_importer

if TYPE_CHECKING:
    from langchain_community.retrievers import AmazonKendraRetriever
    from langchain_community.retrievers.kendra import (
        AdditionalResultAttribute,
        AdditionalResultAttributeValue,
        DocumentAttribute,
        DocumentAttributeValue,
        Highlight,
        QueryResult,
        QueryResultItem,
        ResultItem,
        RetrieveResult,
        RetrieveResultItem,
        TextWithHighLights,
        clean_excerpt,
        combined_text,
    )

# 创建一种动态查找已弃用导入的方式。
# 用于统一处理引发弃用警告的逻辑，
# 以及处理可选导入。
DEPRECATED_LOOKUP = {
    "clean_excerpt": "langchain_community.retrievers.kendra",
    "combined_text": "langchain_community.retrievers.kendra",
    "Highlight": "langchain_community.retrievers.kendra",
    "TextWithHighLights": "langchain_community.retrievers.kendra",
    "AdditionalResultAttributeValue": "langchain_community.retrievers.kendra",
    "AdditionalResultAttribute": "langchain_community.retrievers.kendra",
    "DocumentAttributeValue": "langchain_community.retrievers.kendra",
    "DocumentAttribute": "langchain_community.retrievers.kendra",
    "ResultItem": "langchain_community.retrievers.kendra",
    "QueryResultItem": "langchain_community.retrievers.kendra",
    "RetrieveResultItem": "langchain_community.retrievers.kendra",
    "QueryResult": "langchain_community.retrievers.kendra",
    "RetrieveResult": "langchain_community.retrievers.kendra",
    "AmazonKendraRetriever": "langchain_community.retrievers",
}

_import_attribute = create_importer(__package__, deprecated_lookups=DEPRECATED_LOOKUP)


def __getattr__(name: str) -> Any:
    """动态查找属性。"""
    return _import_attribute(name)


__all__ = [
    "AdditionalResultAttribute",
    "AdditionalResultAttributeValue",
    "AmazonKendraRetriever",
    "DocumentAttribute",
    "DocumentAttributeValue",
    "Highlight",
    "QueryResult",
    "QueryResultItem",
    "ResultItem",
    "RetrieveResult",
    "RetrieveResultItem",
    "TextWithHighLights",
    "clean_excerpt",
    "combined_text",
]

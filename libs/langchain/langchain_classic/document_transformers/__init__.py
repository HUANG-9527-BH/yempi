"""文档转换器是用于转换文档的类。

文档转换器通常用于在单次运行中转换大量文档。
"""

from typing import TYPE_CHECKING, Any

from langchain_classic._api import create_importer

if TYPE_CHECKING:
    from langchain_community.document_transformers import (
        BeautifulSoupTransformer,
        DoctranPropertyExtractor,
        DoctranQATransformer,
        DoctranTextTranslator,
        EmbeddingsClusteringFilter,
        EmbeddingsRedundantFilter,
        GoogleTranslateTransformer,
        Html2TextTransformer,
        LongContextReorder,
        NucliaTextTransformer,
        OpenAIMetadataTagger,
        get_stateful_documents,
    )

# 创建一种动态查找已弃用导入的方式。
# 用于统一处理弃用警告的逻辑并
# 处理可选导入。
DEPRECATED_LOOKUP = {
    "BeautifulSoupTransformer": "langchain_community.document_transformers",
    "DoctranQATransformer": "langchain_community.document_transformers",
    "DoctranTextTranslator": "langchain_community.document_transformers",
    "DoctranPropertyExtractor": "langchain_community.document_transformers",
    "EmbeddingsClusteringFilter": "langchain_community.document_transformers",
    "EmbeddingsRedundantFilter": "langchain_community.document_transformers",
    "GoogleTranslateTransformer": "langchain_community.document_transformers",
    "get_stateful_documents": "langchain_community.document_transformers",
    "LongContextReorder": "langchain_community.document_transformers",
    "NucliaTextTransformer": "langchain_community.document_transformers",
    "OpenAIMetadataTagger": "langchain_community.document_transformers",
    "Html2TextTransformer": "langchain_community.document_transformers",
}

_import_attribute = create_importer(__package__, deprecated_lookups=DEPRECATED_LOOKUP)


def __getattr__(name: str) -> Any:
    """动态查找属性。"""
    return _import_attribute(name)


__all__ = [
    "BeautifulSoupTransformer",
    "DoctranPropertyExtractor",
    "DoctranQATransformer",
    "DoctranTextTranslator",
    "EmbeddingsClusteringFilter",
    "EmbeddingsRedundantFilter",
    "GoogleTranslateTransformer",
    "Html2TextTransformer",
    "LongContextReorder",
    "NucliaTextTransformer",
    "OpenAIMetadataTagger",
    "get_stateful_documents",
]

from typing import TYPE_CHECKING, Any

from langchain_classic._api import create_importer

if TYPE_CHECKING:
    from langchain_community.document_transformers import OpenAIMetadataTagger
    from langchain_community.document_transformers.openai_functions import (
        create_metadata_tagger,
    )

# 创建一种动态查找已弃用导入的方式。
# 用于统一处理引发弃用警告和
# 处理可选导入的逻辑。
DEPRECATED_LOOKUP = {
    "OpenAIMetadataTagger": "langchain_community.document_transformers",
    "create_metadata_tagger": (
        "langchain_community.document_transformers.openai_functions"
    ),
}

_import_attribute = create_importer(__package__, deprecated_lookups=DEPRECATED_LOOKUP)


def __getattr__(name: str) -> Any:
    """动态查找属性。"""
    return _import_attribute(name)


__all__ = [
    "OpenAIMetadataTagger",
    "create_metadata_tagger",
]

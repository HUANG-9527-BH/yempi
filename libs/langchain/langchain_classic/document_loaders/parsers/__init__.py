from typing import TYPE_CHECKING, Any

from langchain_classic._api import create_importer

if TYPE_CHECKING:
    from langchain_community.document_loaders.parsers.audio import OpenAIWhisperParser
    from langchain_community.document_loaders.parsers.docai import DocAIParser
    from langchain_community.document_loaders.parsers.grobid import GrobidParser
    from langchain_community.document_loaders.parsers.html.bs4 import BS4HTMLParser
    from langchain_community.document_loaders.parsers.language.language_parser import (
        LanguageParser,
    )
    from langchain_community.document_loaders.parsers.pdf import (
        PDFMinerParser,
        PDFPlumberParser,
        PyMuPDFParser,
        PyPDFium2Parser,
        PyPDFParser,
    )

# 创建一种动态查找已弃用导入的方式。
# 用于统一处理引发弃用警告的逻辑，
# 以及处理可选导入。
DEPRECATED_LOOKUP = {
    "BS4HTMLParser": "langchain_community.document_loaders.parsers.html.bs4",
    "DocAIParser": "langchain_community.document_loaders.parsers.docai",
    "GrobidParser": "langchain_community.document_loaders.parsers.grobid",
    "LanguageParser": (
        "langchain_community.document_loaders.parsers.language.language_parser"
    ),
    "OpenAIWhisperParser": "langchain_community.document_loaders.parsers.audio",
    "PDFMinerParser": "langchain_community.document_loaders.parsers.pdf",
    "PDFPlumberParser": "langchain_community.document_loaders.parsers.pdf",
    "PyMuPDFParser": "langchain_community.document_loaders.parsers.pdf",
    "PyPDFium2Parser": "langchain_community.document_loaders.parsers.pdf",
    "PyPDFParser": "langchain_community.document_loaders.parsers.pdf",
}

_import_attribute = create_importer(__package__, deprecated_lookups=DEPRECATED_LOOKUP)


def __getattr__(name: str) -> Any:
    """动态查找属性。"""
    return _import_attribute(name)


__all__ = [
    "BS4HTMLParser",
    "DocAIParser",
    "GrobidParser",
    "LanguageParser",
    "OpenAIWhisperParser",
    "PDFMinerParser",
    "PDFPlumberParser",
    "PyMuPDFParser",
    "PyPDFParser",
    "PyPDFium2Parser",
]

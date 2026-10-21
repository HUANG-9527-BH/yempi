from typing import TYPE_CHECKING, Any

from langchain_classic._api import create_importer

if TYPE_CHECKING:
    from langchain_community.document_loaders.parsers.pdf import (
        AmazonTextractPDFParser,
        DocumentIntelligenceParser,
        PDFMinerParser,
        PDFPlumberParser,
        PyMuPDFParser,
        PyPDFium2Parser,
        PyPDFParser,
        extract_from_images_with_rapidocr,
    )

# 创建一种动态查找已弃用导入的方式。
# 用于统一处理引发弃用警告和
# 处理可选导入的逻辑。
DEPRECATED_LOOKUP = {
    "extract_from_images_with_rapidocr": (
        "langchain_community.document_loaders.parsers.pdf"
    ),
    "PyPDFParser": "langchain_community.document_loaders.parsers.pdf",
    "PDFMinerParser": "langchain_community.document_loaders.parsers.pdf",
    "PyMuPDFParser": "langchain_community.document_loaders.parsers.pdf",
    "PyPDFium2Parser": "langchain_community.document_loaders.parsers.pdf",
    "PDFPlumberParser": "langchain_community.document_loaders.parsers.pdf",
    "AmazonTextractPDFParser": "langchain_community.document_loaders.parsers.pdf",
    "DocumentIntelligenceParser": "langchain_community.document_loaders.parsers.pdf",
}

_import_attribute = create_importer(__package__, deprecated_lookups=DEPRECATED_LOOKUP)


def __getattr__(name: str) -> Any:
    """动态查找属性。"""
    return _import_attribute(name)


__all__ = [
    "AmazonTextractPDFParser",
    "DocumentIntelligenceParser",
    "PDFMinerParser",
    "PDFPlumberParser",
    "PyMuPDFParser",
    "PyPDFParser",
    "PyPDFium2Parser",
    "extract_from_images_with_rapidocr",
]

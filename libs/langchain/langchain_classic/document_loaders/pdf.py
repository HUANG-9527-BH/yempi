from typing import TYPE_CHECKING, Any

from langchain_classic._api import create_importer

if TYPE_CHECKING:
    from langchain_community.document_loaders import (
        AmazonTextractPDFLoader,
        MathpixPDFLoader,
        OnlinePDFLoader,
        PagedPDFSplitter,
        PDFMinerLoader,
        PDFMinerPDFasHTMLLoader,
        PDFPlumberLoader,
        PyMuPDFLoader,
        PyPDFDirectoryLoader,
        PyPDFium2Loader,
        UnstructuredPDFLoader,
    )
    from langchain_community.document_loaders.pdf import (
        BasePDFLoader,
        DocumentIntelligenceLoader,
    )

# 创建一种动态查找已弃用导入的方式。
# 用于统一处理引发弃用警告的逻辑和
# 处理可选导入。
DEPRECATED_LOOKUP = {
    "UnstructuredPDFLoader": "langchain_community.document_loaders",
    "BasePDFLoader": "langchain_community.document_loaders.pdf",
    "OnlinePDFLoader": "langchain_community.document_loaders",
    "PagedPDFSplitter": "langchain_community.document_loaders",
    "PyPDFium2Loader": "langchain_community.document_loaders",
    "PyPDFDirectoryLoader": "langchain_community.document_loaders",
    "PDFMinerLoader": "langchain_community.document_loaders",
    "PDFMinerPDFasHTMLLoader": "langchain_community.document_loaders",
    "PyMuPDFLoader": "langchain_community.document_loaders",
    "MathpixPDFLoader": "langchain_community.document_loaders",
    "PDFPlumberLoader": "langchain_community.document_loaders",
    "AmazonTextractPDFLoader": "langchain_community.document_loaders",
    "DocumentIntelligenceLoader": "langchain_community.document_loaders.pdf",
}

_import_attribute = create_importer(__package__, deprecated_lookups=DEPRECATED_LOOKUP)


def __getattr__(name: str) -> Any:
    """动态查找属性。"""
    return _import_attribute(name)


__all__ = [
    "AmazonTextractPDFLoader",
    "BasePDFLoader",
    "DocumentIntelligenceLoader",
    "MathpixPDFLoader",
    "OnlinePDFLoader",
    "PDFMinerLoader",
    "PDFMinerPDFasHTMLLoader",
    "PDFPlumberLoader",
    "PagedPDFSplitter",
    "PyMuPDFLoader",
    "PyPDFDirectoryLoader",
    "PyPDFium2Loader",
    "UnstructuredPDFLoader",
]

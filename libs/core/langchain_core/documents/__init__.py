"""用于数据检索和处理工作流的文档模块。

This module provides core abstractions for handling data in retrieval-augmented
generation (RAG) pipelines, vector stores, and document processing workflows.

!!! warning "文档与消息内容的区别"

    This module is distinct from `langchain_core.messages.content`, which provides
    multimodal content blocks for **LLM chat I/O** (text, images, audio, etc. within
    messages).

    **关键区别：**

    - **文档**（本模块）：用于 **数据检索和处理工作流**
        - 向量存储、检索器、RAG 管道
        - 文本分块、嵌入和语义搜索
        - 示例：存储在向量数据库中的 PDF 分块

    - **Content Blocks** (`messages.content`): For **LLM conversational I/O**
        - Multimodal message content sent to/from models
        - Tool calls, reasoning, citations within chat
        - Example: An image sent to a vision model in a chat message (via
            [`ImageContentBlock`][langchain.messages.ImageContentBlock])

    尽管两者均可表示相似的数据类型（文本、文件），但在 LangChain
    应用中服务于不同的架构目的。
"""

from typing import TYPE_CHECKING

from langchain_core._import_utils import import_attr

if TYPE_CHECKING:
    from langchain_core.documents.base import Document
    from langchain_core.documents.compressor import BaseDocumentCompressor
    from langchain_core.documents.transformers import BaseDocumentTransformer

__all__ = ("BaseDocumentCompressor", "BaseDocumentTransformer", "Document")

_dynamic_imports = {
    "Document": "base",
    "BaseDocumentCompressor": "compressor",
    "BaseDocumentTransformer": "transformers",
}


def __getattr__(attr_name: str) -> object:
    module_name = _dynamic_imports.get(attr_name)
    result = import_attr(attr_name, module_name, __spec__.parent)
    globals()[attr_name] = result
    return result


def __dir__() -> list[str]:
    return list(__all__)

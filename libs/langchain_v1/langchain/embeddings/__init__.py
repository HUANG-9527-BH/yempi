"""嵌入模型。

!!! warning "模块已移动"

    随着 `langchain 1.0.0` 的发布，多个嵌入模块已移至
    `langchain-classic`，例如 `CacheBackedEmbeddings` 和所有社区
    嵌入模块。请参阅 [列表](https://github.com/langchain-ai/langchain/blob/bdf1cd383ce36dc18381a3bf3fb0a579337a32b5/libs/langchain/langchain/embeddings/__init__.py)
    以了解迁移信息。
"""

from langchain_core.embeddings import Embeddings

from langchain.embeddings.base import init_embeddings

__all__ = [
    "Embeddings",
    "init_embeddings",
]

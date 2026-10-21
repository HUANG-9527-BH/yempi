"""Chroma 向量数据库的 LangChain 集成。"""

from langchain_chroma._version import __version__
from langchain_chroma.vectorstores import Chroma

__all__ = [
    "Chroma",
    "__version__",
]

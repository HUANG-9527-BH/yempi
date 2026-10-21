"""向量存储。"""

from typing import TYPE_CHECKING

from langchain_core._import_utils import import_attr

if TYPE_CHECKING:
    from langchain_core.vectorstores.base import VST, VectorStore, VectorStoreRetriever
    from langchain_core.vectorstores.in_memory import InMemoryVectorStore

__all__ = (
    "VST",
    "InMemoryVectorStore",
    "VectorStore",
    "VectorStoreRetriever",
)

_dynamic_imports = {
    "VectorStore": "base",
    "VST": "base",
    "VectorStoreRetriever": "base",
    "InMemoryVectorStore": "in_memory",
}


def __getattr__(attr_name: str) -> object:
    """动态导入并返回子模块中的属性。

    This function enables lazy loading of vectorstore classes from submodules, reducing
    initial import time and circular dependency issues.

    Args:
        attr_name: 要导入的属性名称。

    Returns:
        导入的属性对象。

    Raises:
        AttributeError: 如果属性在 `_dynamic_imports` 中未找到。
    """
    module_name = _dynamic_imports.get(attr_name)
    result = import_attr(attr_name, module_name, __spec__.parent)
    globals()[attr_name] = result
    return result


def __dir__() -> list[str]:
    """返回此模块中可用属性的列表。

    Returns:
        可从此模块导入的属性名称列表。
    """
    return list(__all__)

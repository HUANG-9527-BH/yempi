"""**Docstores** 是用于存储和加载 Document 的类。

**Docstore** 是 Document Loader 的简化版本。
"""

from typing import TYPE_CHECKING, Any

from langchain_classic._api import create_importer

if TYPE_CHECKING:
    from langchain_community.docstore.arbitrary_fn import DocstoreFn
    from langchain_community.docstore.in_memory import InMemoryDocstore
    from langchain_community.docstore.wikipedia import Wikipedia

# 创建一种动态查找已弃用导入的方式。
# 用于整合引发弃用警告的逻辑以及
# 处理可选导入。
DEPRECATED_LOOKUP = {
    "DocstoreFn": "langchain_community.docstore.arbitrary_fn",
    "InMemoryDocstore": "langchain_community.docstore.in_memory",
    "Wikipedia": "langchain_community.docstore.wikipedia",
}

_import_attribute = create_importer(__package__, deprecated_lookups=DEPRECATED_LOOKUP)


def __getattr__(name: str) -> Any:
    """动态查找属性。"""
    return _import_attribute(name)


__all__ = [
    "DocstoreFn",
    "InMemoryDocstore",
    "Wikipedia",
]

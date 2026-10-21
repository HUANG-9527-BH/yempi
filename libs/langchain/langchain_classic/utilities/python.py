"""为向后兼容性保留。"""

from typing import Any

from langchain_classic._api import create_importer

# 代码已从社区包中移除。
# 我们将代理到社区包，该包将抛出适当的异常，
# 但不会将此包含在 __all__ 中，因此它不会被列为可导入的。

_importer = create_importer(
    __package__,
    deprecated_lookups={"PythonREPL": "langchain_community.utilities.python"},
)


def __getattr__(name: str) -> Any:
    """动态查找属性。"""
    return _importer(name)

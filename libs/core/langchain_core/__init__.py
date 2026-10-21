"""``langchain-core`` 定义了 LangChain 生态系统的基础抽象。

The interfaces for core components like chat models, LLMs, vector stores, retrievers,
and more are defined here. The universal invocation protocol (Runnables) along with
a syntax for combining components are also defined here.

**此处不定义任何第三方集成。** 依赖项被有意保持非常轻量。
"""

from langchain_core._api import (
    surface_langchain_beta_warnings,
    surface_langchain_deprecation_warnings,
)
from langchain_core.version import VERSION

__version__ = VERSION

surface_langchain_deprecation_warnings()
surface_langchain_beta_warnings()

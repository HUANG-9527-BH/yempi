"""LangChain **Runnable** 和 **LangChain 表达式语言 (LCEL)**。

LangChain 表达式语言 (LCEL)
提供了一种声明式方法，用于构建利用大语言模型能力的生产级程序。

使用 LCEL 和 LangChain Runnables 创建的程序天然支持同步、异步、批处理和流式操作。

Support for **async** allows servers hosting LCEL based programs to scale better
for higher concurrent loads.

在生成过程中流式输出中间结果，可创建更响应迅速的用户体验。

此模块包含 LangChain Runnables 原语的模式和实现。
"""

from langchain_core.runnables.base import (
    Runnable,
    RunnableBinding,
    RunnableGenerator,
    RunnableLambda,
    RunnableMap,
    RunnableParallel,
    RunnableSequence,
    RunnableSerializable,
)
from langchain_core.runnables.branch import RunnableBranch
from langchain_core.runnables.config import RunnableConfig, patch_config
from langchain_core.runnables.fallbacks import RunnableWithFallbacks
from langchain_core.runnables.passthrough import RunnablePassthrough
from langchain_core.runnables.router import RouterInput, RouterRunnable
from langchain_core.runnables.utils import (
    ConfigurableField,
    ConfigurableFieldMultiOption,
    ConfigurableFieldSingleOption,
)

__all__ = [
    "ConfigurableField",
    "ConfigurableFieldMultiOption",
    "ConfigurableFieldSingleOption",
    "RouterInput",
    "RouterRunnable",
    "Runnable",
    "RunnableBinding",
    "RunnableBranch",
    "RunnableConfig",
    "RunnableGenerator",
    "RunnableLambda",
    "RunnableMap",
    "RunnableParallel",
    "RunnablePassthrough",
    "RunnableSequence",
    "RunnableSerializable",
    "RunnableWithFallbacks",
    "patch_config",
]

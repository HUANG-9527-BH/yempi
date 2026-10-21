"""LangChain **Runnable** 与 **LangChain 表达式语言 (LCEL)**。

LangChain 表达式语言 (LCEL)
提供了一种声明式方法，用于构建利用大语言模型能力的生产级程序。

使用 LCEL 和 LangChain `Runnable` 对象创建的程序，天然支持同步、异步、批处理和流式操作。

Support for **async** allows servers hosting LCEL based programs to scale bette for
higher concurrent loads.

**批处理** 操作允许并行处理多个输入。

**Streaming** of intermediate outputs, as they're being generated, allows for creating
more responsive UX.

此模块包含 LangChain `Runnable` 对象原语的模式与实现。
"""

from typing import TYPE_CHECKING

from langchain_core._import_utils import import_attr

if TYPE_CHECKING:
    from langchain_core.runnables.base import (
        Runnable,
        RunnableBinding,
        RunnableGenerator,
        RunnableLambda,
        RunnableMap,
        RunnableParallel,
        RunnableSequence,
        RunnableSerializable,
        chain,
    )
    from langchain_core.runnables.branch import RunnableBranch
    from langchain_core.runnables.config import (
        RunnableConfig,
        ensure_config,
        get_config_list,
        patch_config,
        run_in_executor,
    )
    from langchain_core.runnables.fallbacks import RunnableWithFallbacks
    from langchain_core.runnables.history import RunnableWithMessageHistory
    from langchain_core.runnables.passthrough import (
        RunnableAssign,
        RunnablePassthrough,
        RunnablePick,
    )
    from langchain_core.runnables.router import RouterInput, RouterRunnable
    from langchain_core.runnables.utils import (
        AddableDict,
        ConfigurableField,
        ConfigurableFieldMultiOption,
        ConfigurableFieldSingleOption,
        ConfigurableFieldSpec,
        aadd,
        add,
    )

__all__ = (
    "AddableDict",
    "ConfigurableField",
    "ConfigurableFieldMultiOption",
    "ConfigurableFieldSingleOption",
    "ConfigurableFieldSpec",
    "RouterInput",
    "RouterRunnable",
    "Runnable",
    "RunnableAssign",
    "RunnableBinding",
    "RunnableBranch",
    "RunnableConfig",
    "RunnableGenerator",
    "RunnableLambda",
    "RunnableMap",
    "RunnableParallel",
    "RunnablePassthrough",
    "RunnablePick",
    "RunnableSequence",
    "RunnableSerializable",
    "RunnableWithFallbacks",
    "RunnableWithMessageHistory",
    "aadd",
    "add",
    "chain",
    "ensure_config",
    "get_config_list",
    "patch_config",
    "run_in_executor",
)

_dynamic_imports = {
    "chain": "base",
    "Runnable": "base",
    "RunnableBinding": "base",
    "RunnableGenerator": "base",
    "RunnableLambda": "base",
    "RunnableMap": "base",
    "RunnableParallel": "base",
    "RunnableSequence": "base",
    "RunnableSerializable": "base",
    "RunnableBranch": "branch",
    "RunnableConfig": "config",
    "ensure_config": "config",
    "get_config_list": "config",
    "patch_config": "config",
    "run_in_executor": "config",
    "RunnableWithFallbacks": "fallbacks",
    "RunnableWithMessageHistory": "history",
    "RunnableAssign": "passthrough",
    "RunnablePassthrough": "passthrough",
    "RunnablePick": "passthrough",
    "RouterInput": "router",
    "RouterRunnable": "router",
    "AddableDict": "utils",
    "ConfigurableField": "utils",
    "ConfigurableFieldMultiOption": "utils",
    "ConfigurableFieldSingleOption": "utils",
    "ConfigurableFieldSpec": "utils",
    "aadd": "utils",
    "add": "utils",
}


def __getattr__(attr_name: str) -> object:
    module_name = _dynamic_imports.get(attr_name)
    result = import_attr(attr_name, module_name, __spec__.parent)
    globals()[attr_name] = result
    return result


def __dir__() -> list[str]:
    return list(__all__)

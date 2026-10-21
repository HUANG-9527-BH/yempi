from __future__ import annotations

from typing import TYPE_CHECKING, Any

from langchain_core.callbacks import Callbacks
from langchain_core.callbacks.manager import (
    AsyncCallbackManager,
    AsyncCallbackManagerForChainGroup,
    AsyncCallbackManagerForChainRun,
    AsyncCallbackManagerForLLMRun,
    AsyncCallbackManagerForRetrieverRun,
    AsyncCallbackManagerForToolRun,
    AsyncParentRunManager,
    AsyncRunManager,
    BaseRunManager,
    CallbackManager,
    CallbackManagerForChainGroup,
    CallbackManagerForChainRun,
    CallbackManagerForLLMRun,
    CallbackManagerForRetrieverRun,
    CallbackManagerForToolRun,
    ParentRunManager,
    RunManager,
    ahandle_event,
    atrace_as_chain_group,
    handle_event,
    trace_as_chain_group,
)
from langchain_core.tracers.context import (
    collect_runs,
    tracing_v2_enabled,
)
from langchain_core.utils.env import env_var_is_set

from langchain_classic._api import create_importer

if TYPE_CHECKING:
    from langchain_community.callbacks.manager import (
        get_openai_callback,
        wandb_tracing_enabled,
    )

# 创建一种动态查找已弃用导入的方式。
# 用于统一处理引发弃用警告的逻辑和
# 处理可选导入。
DEPRECATED_LOOKUP = {
    "get_openai_callback": "langchain_community.callbacks.manager",
    "wandb_tracing_enabled": "langchain_community.callbacks.manager",
}

_import_attribute = create_importer(__file__, deprecated_lookups=DEPRECATED_LOOKUP)


def __getattr__(name: str) -> Any:
    """动态查找属性。"""
    return _import_attribute(name)


__all__ = [
    "AsyncCallbackManager",
    "AsyncCallbackManagerForChainGroup",
    "AsyncCallbackManagerForChainRun",
    "AsyncCallbackManagerForLLMRun",
    "AsyncCallbackManagerForRetrieverRun",
    "AsyncCallbackManagerForToolRun",
    "AsyncParentRunManager",
    "AsyncRunManager",
    "BaseRunManager",
    "CallbackManager",
    "CallbackManagerForChainGroup",
    "CallbackManagerForChainRun",
    "CallbackManagerForLLMRun",
    "CallbackManagerForRetrieverRun",
    "CallbackManagerForToolRun",
    "Callbacks",
    "ParentRunManager",
    "RunManager",
    "ahandle_event",
    "atrace_as_chain_group",
    "collect_runs",
    "env_var_is_set",
    "get_openai_callback",
    "handle_event",
    "trace_as_chain_group",
    "tracing_v2_enabled",
    "wandb_tracing_enabled",
]

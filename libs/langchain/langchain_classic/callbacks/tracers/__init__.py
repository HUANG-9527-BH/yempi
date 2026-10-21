"""记录 LangChain 运行执行的追踪器。"""

from typing import TYPE_CHECKING, Any

from langchain_core.tracers.langchain import LangChainTracer
from langchain_core.tracers.stdout import (
    ConsoleCallbackHandler,
    FunctionCallbackHandler,
)

from langchain_classic._api import create_importer
from langchain_classic.callbacks.tracers.logging import LoggingCallbackHandler

if TYPE_CHECKING:
    from langchain_community.callbacks.tracers.wandb import WandbTracer

# 创建一种动态查找已弃用导入的方式。
# 用于统一处理弃用警告的逻辑和
# 处理可选导入的逻辑。
DEPRECATED_LOOKUP = {"WandbTracer": "langchain_community.callbacks.tracers.wandb"}

_import_attribute = create_importer(__file__, deprecated_lookups=DEPRECATED_LOOKUP)


def __getattr__(name: str) -> Any:
    """动态查找属性。"""
    return _import_attribute(name)


__all__ = [
    "ConsoleCallbackHandler",
    "FunctionCallbackHandler",
    "LangChainTracer",
    "LoggingCallbackHandler",
    "WandbTracer",
]

"""用于追踪器的上下文管理。"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    cast,
)
from uuid import UUID

from langsmith import run_helpers as ls_rh
from langsmith import utils as ls_utils

from langchain_core.tracers.langchain import LangChainTracer
from langchain_core.tracers.run_collector import RunCollectorCallbackHandler

if TYPE_CHECKING:
    from collections.abc import Generator

    from langsmith import Client as LangSmithClient
    from langsmith.address import AgentAddress

    from langchain_core.callbacks.base import BaseCallbackHandler, Callbacks
    from langchain_core.callbacks.manager import AsyncCallbackManager, CallbackManager

# 为了向后部分兼容，如果用户导入但未使用此模块。
tracing_callback_var: Any = None
tracing_v2_callback_var: ContextVar[LangChainTracer | None] = ContextVar(
    "tracing_callback_v2", default=None
)
run_collector_var: ContextVar[RunCollectorCallbackHandler | None] = ContextVar(
    "run_collector", default=None
)


@contextmanager
def tracing_v2_enabled(
    project_name: str | None = None,
    *,
    example_id: str | UUID | None = None,
    tags: list[str] | None = None,
    client: LangSmithClient | None = None,
    address: AgentAddress | None = None,
) -> Generator[LangChainTracer, None, None]:
    """指示 LangChain 在上下文中记录所有运行到 LangSmith。

    Args:
        project_name: 项目的名称。

            Resolved together with `address` from LangSmith config and the
            environment when no destination is supplied.
        example_id: The ID of the example.
        tags: The tags to add to the run.
        client: The client of the langsmith.
        address: The agent address to send traces to instead of a project.

            !!! warning "测试版"
                必须在 LangSmith 工作区中启用代理地址功能。

    Yields:
        LangChain 追踪器。

    Example:
        >>> with tracing_v2_enabled():
        ...     # LangChain code will automatically be traced

        可以使用此方法获取 LangSmith 运行 URL：

        >>> with tracing_v2_enabled() as cb:
        ...     chain.invoke("foo")
        ...     run_url = cb.get_run_url()
    """
    if isinstance(example_id, str):
        example_id = UUID(example_id)
    cb = LangChainTracer(
        example_id=example_id,
        project_name=project_name,
        address=address,
        tags=tags,
        client=client,
    )
    token = tracing_v2_callback_var.set(cb)
    try:
        yield cb
    finally:
        tracing_v2_callback_var.reset(token)


@contextmanager
def collect_runs() -> Generator[RunCollectorCallbackHandler, None, None]:
    """在上下文中收集所有运行追踪。

    Yields:
        运行收集回调处理器。

    Example:
        >>> with collect_runs() as runs_cb:
                chain.invoke("foo")
                run_id = runs_cb.traced_runs[0].id
    """
    cb = RunCollectorCallbackHandler()
    token = run_collector_var.set(cb)
    try:
        yield cb
    finally:
        run_collector_var.reset(token)


def _get_trace_callbacks(
    project_name: str | None = None,
    example_id: str | UUID | None = None,
    callback_manager: CallbackManager | AsyncCallbackManager | None = None,
    *,
    address: AgentAddress | None = None,
) -> Callbacks:
    if _tracing_v2_is_enabled():
        tracer = tracing_v2_callback_var.get() or LangChainTracer(
            example_id=example_id,
            **_get_tracer_kwargs(project_name, address=address),
        )
        if callback_manager is None:
            cb = cast("Callbacks", [tracer])
        else:
            if not any(
                isinstance(handler, LangChainTracer)
                for handler in callback_manager.handlers
            ):
                callback_manager.add_handler(tracer)
                                # 如果已存在 LangChainTracer，则无需再添加另一个。
                                # 这可能会破坏追踪层级结构。
            cb = callback_manager
    else:
        cb = None
    return cb


def _tracing_v2_is_enabled() -> bool | Literal["local"]:
    if tracing_v2_callback_var.get() is not None:
        return True
    return ls_utils.tracing_is_enabled()


def _get_tracer_kwargs(
    project_name: str | None = None, *, address: AgentAddress | None = None
) -> dict[str, Any]:
    """从当前 LangSmith 上下文中获取追踪默认值。

    同时传递 project 和
    address，以确保已指定地址的父级永远不会拾取环境中的项目。将环境默认值和验证留给
    RunTree。
    """
    context = ls_rh.get_tracing_context()
    parent = context["parent"]
    if project_name is not None or address is not None:
        project = project_name
    elif parent is not None:
        project, address = parent.session_name, parent.address
    else:
        project, address = context["project_name"], context["address"]
    return {
        "project_name": project,
        "address": address,
        "client": parent.client if parent is not None else context["client"],
        "tags": context["tags"],
        "metadata": context["metadata"],
    }


_configure_hooks: list[
    tuple[
        ContextVar[BaseCallbackHandler | None],
        bool,
        type[BaseCallbackHandler] | None,
        str | None,
    ]
] = []


def register_configure_hook(
    context_var: ContextVar[Any | None],
    inheritable: bool,  # noqa: FBT001
    handle_class: type[BaseCallbackHandler] | None = None,
    env_var: str | None = None,
) -> None:
    """注册一个配置钩子。

    Args:
        context_var: 上下文变量。
        inheritable: 上下文变量是否可继承。
        handle_class: 回调处理器类。
        env_var: 环境变量。

    Raises:
        ValueError: 如果设置了 `env_var`，则 `handle_class` 也必须设置为非 `None` 值。
    """
    if env_var is not None and handle_class is None:
        msg = "If env_var is set, handle_class must also be set to a non-None value."
        raise ValueError(msg)

    _configure_hooks.append(
        (
                        # ContextVar 的类型定义未将泛型参数设为协变
                        # 因此我们必须进行强制转换
            cast("ContextVar[BaseCallbackHandler | None]", context_var),
            inheritable,
            handle_class,
            env_var,
        )
    )


register_configure_hook(run_collector_var, inheritable=False)

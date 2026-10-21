"""智能体中间件的进程全局默认 `TracePolicy`。

`TracePolicy` (from langgraph) shapes what a middleware hook span records. This module
adds a process-wide default that individual middleware override via their own
`trace_policy` attribute. Set it once at startup with `configure_trace_policy`; the
default is read at span-creation time, so it applies to agents built before or after
the call.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from langgraph.types import TracePolicy

if TYPE_CHECKING:
    from collections.abc import Callable

_DEFAULT_TRACE_POLICY: TracePolicy | None = None


def configure_trace_policy(policy: TracePolicy | None) -> None:
    """设置智能体中间件钩子跨度的进程全局默认 `TracePolicy`。

    启动时调用一次。中间件自身的 `trace_policy`
    会完全覆盖此默认值（不进行字段级合并）。传入 `None` 可清除默认值。
    """
    global _DEFAULT_TRACE_POLICY  # noqa: PLW0603 - process-wide default, set once at startup
    _DEFAULT_TRACE_POLICY = policy


def _resolve(mw_policy: TracePolicy | None) -> TracePolicy | None:
    """The effective policy: 有效策略：若中间件设置了自身策略，则使用该策略；否则使用全局默认策略。"""
    return mw_policy if mw_policy is not None else _DEFAULT_TRACE_POLICY


def _resolved_transform(
    mw_policy: TracePolicy | None,
    field: Literal["process_inputs", "process_outputs"],
) -> Callable[[Any], Any]:
    """一个在调用时解析有效策略的追踪处理器。

    在调用时而非构建时读取全局策略，意味着 `configure_trace_policy` 可在 `create_agent`
    之后运行仍能生效。当没有有效策略定义 `field` 时，原样传递该值。
    """

    def process(value: Any) -> Any:
        effective = _resolve(mw_policy)
        fn = getattr(effective, field) if effective is not None else None
        return fn(value) if fn is not None else value

    return process


def _node_trace_policy(mw_policy: TracePolicy | None) -> TracePolicy:
    """为中间件节点钩子构建基于解析器的 `TracePolicy`。"""
    return TracePolicy(
        process_inputs=_resolved_transform(mw_policy, "process_inputs"),
        process_outputs=_resolved_transform(mw_policy, "process_outputs"),
    )

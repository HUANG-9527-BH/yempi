"""将嵌套的命名智能体作为类型化的 `run.subagents` 句柄暴露出来。

Detects subagents via the `lc_agent_name` transition that langgraph's base
`_TasksLifecycleBase` now computes. `create_agent(name=...)` binds
`lc_agent_name` into the run config; the base transformer records, per
namespace, the `lc_agent_name` seen on each task start (first-write-wins).

A subagent boundary is a nested run whose `lc_agent_name` is set *and* differs
from its parent namespace's `lc_agent_name`. Plain subgraphs inherit the
parent's name (so they compare equal and are excluded); unnamed agents have
`lc_agent_name == None` (also excluded). For genuine subagents the base also
recovers the originating tool call and exposes it as a `cause`
(`{"type": "toolCall", "tool_call_id": ...}`) via `self._pending_cause`, joined
from the parent task's pending tool calls.

此转换器基于该边界进行控制，并在 `run.subagents`
上暴露一个类型化的句柄，然后将子作用域事件转发到该句柄的多路复用器，使嵌套运行可独立消费。
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, ClassVar

from langgraph.stream.run_stream import (
    AsyncSubgraphRunStream,
    SubgraphRunStream,
)
from langgraph.stream.stream_channel import StreamChannel
from langgraph.stream.transformers import (
    SubgraphStatus,
    _TasksLifecycleBase,
)

if TYPE_CHECKING:
    from langchain_protocol.protocol import LifecycleCause
    from langgraph.stream._mux import StreamMux
    from langgraph.stream._types import ProtocolEvent

logger = logging.getLogger(__name__)


class SubagentRunStream(SubgraphRunStream):
    """嵌套命名智能体执行的类型化同步句柄。

    Surfaces on `run.subagents` when a nested run's `lc_agent_name` differs
    from its parent's (i.e., a `create_agent(name=...)` dispatched from a tool).
    """

    def __init__(
        self,
        mux: StreamMux,
        *,
        path: tuple[str, ...],
        graph_name: str | None = None,
        trigger_call_id: str | None = None,
        cause: LifecycleCause | None = None,
    ) -> None:
        super().__init__(
            mux,
            path=path,
            graph_name=graph_name,
            trigger_call_id=trigger_call_id,
        )
        self._cause = cause

    @property
    def name(self) -> str | None:
        """子智能体名称（嵌套运行的 `lc_agent_name`）。"""
        return self.graph_name

    @property
    def cause(self) -> LifecycleCause | None:
        """因果边 — 触发此子智能体的工具调用。

        Returns the `LifecycleCause` recovered by the base transformer (a
        `{"type": "toolCall", "tool_call_id": ...}` dict) when the originating
        tool call could be joined, else `None`.
        """
        return self._cause


class AsyncSubagentRunStream(AsyncSubgraphRunStream):
    """嵌套命名智能体执行的类型化异步句柄。"""

    def __init__(
        self,
        mux: StreamMux,
        *,
        path: tuple[str, ...],
        graph_name: str | None = None,
        trigger_call_id: str | None = None,
        cause: LifecycleCause | None = None,
    ) -> None:
        super().__init__(
            mux,
            path=path,
            graph_name=graph_name,
            trigger_call_id=trigger_call_id,
        )
        self._cause = cause

    @property
    def name(self) -> str | None:
        """子智能体名称（嵌套运行的 `lc_agent_name`）。"""
        return self.graph_name

    @property
    def cause(self) -> LifecycleCause | None:
        """因果边 — 触发此子智能体的工具调用。

        Returns the `LifecycleCause` recovered by the base transformer (a
        `{"type": "toolCall", "tool_call_id": ...}` dict) when the originating
        tool call could be joined, else `None`.
        """
        return self._cause


class SubagentTransformer(_TasksLifecycleBase):
    """将嵌套命名智能体提升为 `run.subagents` 上的类型化句柄。

    The base `_TasksLifecycleBase` records each namespace's `lc_agent_name`
    (set by `create_agent(name=...)`) and, on every task start, fires
    `_on_started` with the resolved `graph_name` and a `cause` for genuine
    subagent boundaries. This transformer gates on that boundary using the
    inherited `_lc_by_ns` map: a nested run is a subagent when it carries an
    `lc_agent_name`. Same-named nested agents (e.g. a subagent that invokes
    itself) are surfaced; unnamed agents (`None`) are excluded. Trade-off: a
    non-agent subgraph that inherited the parent's name will also surface.

    在第一次匹配的任务开始时，它会构建一个子多路复用器，并在 `run.subagents`
    上发出一个类型化句柄，然后将后续子作用域事件转发到该句柄，使嵌套运行可独立消费。
    """

    _native: ClassVar[bool] = True
        # 覆盖 `aprocess`，但也在同步通道上通过 `process`
    # 以不变方式运行，因此不能强制进入仅异步运行。
        # `process`，所以它必须不被强制进入仅异步运行。
    supports_sync: ClassVar[bool] = True

    def __init__(self, scope: tuple[str, ...] = ()) -> None:
        super().__init__(scope)
        self._log: StreamChannel[SubagentRunStream | AsyncSubagentRunStream] = StreamChannel()
        self._handles: dict[tuple[str, ...], SubagentRunStream | AsyncSubagentRunStream] = {}
        self._mux: StreamMux | None = None

    def init(self) -> dict[str, Any]:
        return {"subagents": self._log}

    def _on_register(self, mux: StreamMux) -> None:
        self._mux = mux

    def _should_track(self, ns: tuple[str, ...]) -> bool:
        depth = len(self.scope)
        return len(ns) == depth + 1 and ns[:depth] == self.scope

    def _on_started(
        self,
        ns: tuple[str, ...],
        graph_name: str | None,
        trigger_call_id: str | None,
    ) -> None:
                # langgraph >=1.2.4 通过基类的 `self._pending_cause` 实例状态传递触发的
        # `cause`，而不是通过 `_on_started` 关键字参数，因此早于 `cause`
        # 的覆盖仍能正常工作。在此处读取它，以在 handle 中呈现原始工具调用。
                # 在此处读取它，以在 handle 中呈现原始工具调用。
                # 呈现任何携带 lc_agent_name（由 create_agent 设置）的嵌套运行。
                # 同名的嵌套智能体——例如递归调用自身的子智能体——会重新声明其自身名称并被呈现。未命名的运行（None）被排除。权衡：继承父级名称的非智能体子图也会被呈现；调用此类图时传入
        # null lc_agent_name 以排除它。
        cause = self._pending_cause
        child_lc = self._lc_by_ns.get(ns)
                # 首先进行任务记账，以便在将事件转发到子 mini-mux 时，`started` handle 已存在。
                # 异步对应版本：在此重复任务记账，并通过异步通道转发到子
        # mini-mux，使子智能体自身的转换器在正确的（异步）通道上驱动，而非通过同步的
        # `process`/`push` 路径被双重驱动。
                # 镜像同步记账，使异步通道在 `_handle_task_start` -> `_on_started`
        # 中区分子智能体边界之前，能观察到父级身份 / 待处理的工具调用。
                # 排除。权衡：一个继承了父级的非智能体子图
                # name 也会被暴露；调用此类图时，lc_agent_name 为 null
                # 排除它。
        if child_lc is None:
            return
        if self._mux is None or ns in self._handles:
            return
        try:
            child_mux = self._mux._make_child(ns)  # noqa: SLF001
        except RuntimeError:
            logger.debug("SubagentTransformer: could not create child mux for %s", ns)
            return

        handle_cls = AsyncSubagentRunStream if child_mux.is_async else SubagentRunStream
        handle = handle_cls(
            mux=child_mux,
            path=ns,
            graph_name=graph_name,
            trigger_call_id=trigger_call_id,
            cause=cause,
        )
        self._handles[ns] = handle
        self._log.push(handle)

    def _on_terminal(
        self,
        ns: tuple[str, ...],
        status: SubgraphStatus,
        error: str | None,
    ) -> None:
        handle = self._handles.get(ns)
        if handle is None or not self._mark_terminal(handle, status, error):
            return
        self._close_or_fail_handle(handle, status, error)

    async def _aon_terminal(
        self,
        ns: tuple[str, ...],
        status: SubgraphStatus,
        error: str | None,
    ) -> None:
        handle = self._handles.get(ns)
        if handle is None or not self._mark_terminal(handle, status, error):
            return
        await self._aclose_or_fail_handle(handle, status, error)

    def _mark_terminal(
        self,
        handle: SubagentRunStream | AsyncSubagentRunStream,
        status: SubgraphStatus,
        error: str | None,
    ) -> bool:
        """仅标记一次句柄为终止状态。在首次转换时返回 True。"""
        if handle._seen_terminal:  # noqa: SLF001
            return False
        handle.status = status
        if error is not None and handle.error is None:
            handle.error = error
        handle._seen_terminal = True  # noqa: SLF001
        return True

    def _close_or_fail_handle(
        self,
        handle: SubagentRunStream | AsyncSubagentRunStream,
        status: SubgraphStatus,
        error: str | None,
    ) -> None:
        if handle._mux is None or handle._mux._events._closed:  # noqa: SLF001
            return
        if status == "failed":
            handle._mux.fail(RuntimeError(error or "Subagent failed"))  # noqa: SLF001
        else:
            handle._mux.close()  # noqa: SLF001

    async def _aclose_or_fail_handle(
        self,
        handle: SubagentRunStream | AsyncSubagentRunStream,
        status: SubgraphStatus,
        error: str | None,
    ) -> None:
        if handle._mux is None or handle._mux._events._closed:  # noqa: SLF001
            return
        if status == "failed":
            await handle._mux.afail(RuntimeError(error or "Subagent failed"))  # noqa: SLF001
        else:
            await handle._mux.aclose()  # noqa: SLF001

    def _handle_for_event(
        self, event: ProtocolEvent
    ) -> SubagentRunStream | AsyncSubagentRunStream | None:
        ns = tuple(event["params"]["namespace"])
        depth = len(self.scope)
        if len(ns) < depth + 1:
            return None
        handle = self._handles.get(ns[: depth + 1])
        if handle is None or handle._mux is None or handle._mux._events._closed:  # noqa: SLF001
            return None
        return handle

    def process(self, event: ProtocolEvent) -> bool:
                # 首先执行任务记账，以便在 `started` 处理程序存在时
                # 时间我们向前传递事件到子迷你多路复用器。
        keep = super().process(event)
        handle = self._handle_for_event(event)
        if handle is not None:
            handle._observe_event(event)  # noqa: SLF001
            handle._mux.push(event)  # noqa: SLF001
        return keep

    async def aprocess(self, event: ProtocolEvent) -> bool:
                # 异步对应：在此重复任务的记录工作并转发到
                # 通过其异步通道让子智能体运行
                # transformers 在正确的（异步）车道上运行，而不是被
                # 通过同步的 `process`/`push` 路径双驱动。
        if event["method"] == "tasks":
            ns = tuple(event["params"]["namespace"])
            data = event["params"]["data"]
            if "result" in data:
                for child_ns, status, error in self._pop_terminal_transitions(ns, data):
                    await self._aon_terminal(child_ns, status, error)
            else:
                                # 镜像同步的记账逻辑，以便异步通道观察到父级
                                # 身份 / 在区分之前待处理的工具调用
                                # subagent 边界在 `_handle_task_start` -> `_on_started`。
                self._record_identity(ns, data)
                self._record_pending_tool_calls(data)
                self._handle_task_start(ns, data)
            keep = False
        else:
            keep = True
        handle = self._handle_for_event(event)
        if handle is not None:
            handle._observe_event(event)  # noqa: SLF001
            await handle._mux.apush(event)  # noqa: SLF001
        return keep

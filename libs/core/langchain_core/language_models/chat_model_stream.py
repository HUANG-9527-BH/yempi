"""每条消息的流式对象，用于内容块协议事件。

`ChatModelStream` is the synchronous variant returned by
`BaseChatModel.stream_events(version="v3")`.  `AsyncChatModelStream` is the
asynchronous variant returned by `BaseChatModel.astream_events(version="v3")`.

Both expose typed projection properties (`.text`, `.reasoning`,
`.tool_calls`, `.usage`, `.output`) that accumulate protocol
events as they arrive.  Projections can be iterated for deltas or
drained for the final accumulated value.

原始协议事件也可通过直接迭代流对象获得（重放缓冲区语义——支持多个独立消费者）。
"""

from __future__ import annotations

import asyncio
import contextlib
from typing import TYPE_CHECKING, Any, cast

from langchain_core.language_models._compat_bridge import finalize_tool_call_chunk
from langchain_core.messages import AIMessage

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Generator, Iterator, Mapping

    from langchain_protocol.protocol import (
        ContentBlockDeltaData,
        ContentBlockFinishData,
        FinalizedContentBlock,
        InvalidToolCall,
        MessageFinishData,
        MessageMetadata,
        MessagesData,
        MessageStartData,
        ReasoningContentBlock,
        ServerToolCallChunk,
        TextContentBlock,
        ToolCall,
        ToolCallChunk,
        UsageInfo,
    )
    from typing_extensions import Self


# ---------------------------------------------------------------------------
# 工具调用块辅助函数（由 `tool_call_chunk` 和 `server_tool_call_chunk` 共享）
# ---------------------------------------------------------------------------


def _merge_chunk_into_store(
    store: dict[int, dict[str, Any]],
    idx: int,
    block: dict[str, Any],
) -> None:
    """合并一个工具调用块的增量：保留粘性 ID/名称，拼接参数。"""
    existing = store.get(idx, {})
    if block.get("id") and "id" not in existing:
        existing["id"] = block["id"]
    if block.get("name") and "name" not in existing:
        existing["name"] = block["name"]
    existing["args"] = existing.get("args", "") + (block.get("args") or "")
    store[idx] = existing


def _merge_block_delta_into_store(
    store: dict[int, dict[str, Any]],
    idx: int,
    fields: dict[str, Any],
) -> None:
    """浅层合并一个块增量快照到索引块存储中。"""
    existing = store.get(idx, {})
    for key, value in fields.items():
        if value is not None:
            existing[key] = value
    store[idx] = existing


def _event_content_block(data: Mapping[str, Any]) -> dict[str, Any] | None:
    """返回起始/结束内容，容忍预增量字段名。"""
    block = data.get("content") or data.get("content_block")
    return block if isinstance(block, dict) else None


def _legacy_block_to_delta(block: Mapping[str, Any]) -> dict[str, Any]:
    """将旧的内容块增量结构转换为显式的增量形式。"""
    btype = block.get("type")
    if btype == "text":
        return {"type": "text-delta", "text": block.get("text", "")}
    if btype == "reasoning":
        return {
            "type": "reasoning-delta",
            "reasoning": block.get("reasoning", ""),
        }
    if "data" in block:
        delta = {"type": "data-delta", "data": block.get("data", "")}
        if block.get("encoding") == "base64":
            delta["encoding"] = "base64"
        return delta
    return {"type": "legacy-block-delta", "fields": block}


def _event_delta(data: Mapping[str, Any]) -> dict[str, Any] | None:
    """返回一个显式增量，转换遗留的内容块增量。"""
    delta = data.get("delta")
    if isinstance(delta, dict):
        return delta
    block = data.get("content_block")
    if isinstance(block, dict):
        return _legacy_block_to_delta(block)
    return None


def _sweep_chunk_store(
    store: dict[int, dict[str, Any]],
    *,
    finalized_type: str,
    finalized_blocks: dict[int, FinalizedContentBlock],
    tool_calls_acc: list[ToolCall] | None,
    invalid_acc: list[InvalidToolCall],
) -> None:
    """解析每个未清扫块的 `args`；记录为 `finalized_type` 或无效状态。

    `tool_calls_acc` 仅在 `finalized_type == "tool_call"` 时填充
    （服务器端调用不会通过 `.tool_calls` 暴露）。

    故意不向已定型的工具调用块回填 `index`： 与 v1
    保持一致（`AIMessage.init_tool_calls` 在替换 `tool_call_chunk` → `tool_call` 时丢弃
    `index`），并防止 `merge_lists` 将后续块重新合并进已解析的参数字典中。
    """
    for idx in sorted(store):
        chunk = store[idx]
                # 继承块收集的任何未被重新初始化的字段
                # （例如 `extras`）。`_merge_chunk_into_store` 仅填充
                # `id` / `name` / `args`，因此目前实践中为空；
                # 未来的提供方特定字段将通过此处传递。
        extras = {
            k: v
            for k, v in chunk.items()
            if k not in {"type", "id", "name", "args"} and v is not None
        }
        final_block = finalize_tool_call_chunk(
            raw_args=chunk.get("args"),
            id_=chunk.get("id"),
            name=chunk.get("name"),
            extras=extras,
            finalized_type=finalized_type,
        )
        if final_block["type"] == "invalid_tool_call":
            invalid_acc.append(final_block)
        elif tool_calls_acc is not None and finalized_type == "tool_call":
            tool_calls_acc.append(cast("ToolCall", final_block))
        finalized_blocks[idx] = final_block
    store.clear()


# ---------------------------------------------------------------------------
# 投影基类 — 共享生产者 API
# ---------------------------------------------------------------------------


class _ProjectionBase:
    """同步和异步投影共享的状态与生产者 API。

    The `push` / `complete` / `fail` methods are the producer-side
    API — called by the stream as events arrive. Subclasses add the
    consumer protocol (sync iteration or async iteration + await).

    `done` and `error` are safe read-only views of the terminal state
    for iterators and other siblings that need to observe lifecycle
    without reaching into the underlying fields.
    """

    __slots__ = ("_deltas", "_done", "_error", "_final_set", "_final_value")

    def __init__(self) -> None:
        """初始化空的投影状态。"""
        self._deltas: list[Any] = []
        self._final_value: Any = None
        self._final_set: bool = False
        self._done: bool = False
        self._error: BaseException | None = None

    @property
    def done(self) -> bool:
        """投影是否已完成（成功或因错误终止）。"""
        return self._done

    @property
    def error(self) -> BaseException | None:
        """终端错误，如果有的话。"""
        return self._error

    def push(self, delta: Any) -> None:
        """追加一个 delta 值。生产者端 API。"""
        self._deltas.append(delta)

    def complete(self, final_value: Any) -> None:
        """设置最终累积值并标记为完成。生产者端 API。"""
        self._final_value = final_value
        self._final_set = True
        self._done = True

    def fail(self, error: BaseException) -> None:
        """标记为出错。生产者端 API。"""
        self._error = error
        self._done = True


# ---------------------------------------------------------------------------
# 同步投影
# ---------------------------------------------------------------------------


class SyncProjection(_ProjectionBase):
    """带有拉取式背压的同步增量可迭代对象。

    Follows the same `_request_more` convention as langgraph's
    `EventLog`: when the cursor catches up to the buffer and the
    projection is not done, it calls `_request_more()` to pull more
    events from the producer.

    每次调用 `__iter__` 都会在位置 0 创建一个新的游标。
    多个迭代器会从头开始重播所有增量。
    """

    __slots__ = ("_ensure_started", "_request_more")

    def __init__(self) -> None:
        """初始化时无拉取回调。"""
        super().__init__()
        self._ensure_started: Callable[[], None] | None = None
        self._request_more: Callable[[], bool] | None = None

    def set_start(self, cb: Callable[[], None] | None) -> None:
        """安装一个懒启动回调，在首次消费时调用。"""
        self._ensure_started = cb

    def set_request_more(self, cb: Callable[[], bool] | None) -> None:
        """安装迭代器用于耗尽源的拉取回调。"""
        self._request_more = cb

    def __iter__(self) -> Iterator[Any]:
        """生成 deltas，当追上时通过 `_request_more` 拉取。"""
        if self._ensure_started is not None:
            self._ensure_started()
        cursor = 0
        while True:
            if cursor < len(self._deltas):
                yield self._deltas[cursor]
                cursor += 1
            elif self._error is not None:
                raise self._error
            elif self._done:
                return
            elif self._request_more is not None:
                while cursor >= len(self._deltas) and not self._done:
                    if not self._request_more():
                        break
                if cursor >= len(self._deltas):
                    if self._error is not None:
                        raise self._error
                    return
            else:
                return

    def get(self) -> Any:
        """通过 `_request_more` 耗尽并返回最终值。"""
        if self._ensure_started is not None:
            self._ensure_started()
        if not self._done and self._request_more is not None:
            while not self._done:
                if not self._request_more():
                    break
        if self._error is not None:
            raise self._error
        return self._final_value


class SyncTextProjection(SyncProjection):
    """针对字符串的同步投影。

    Adds typed string producers and consumers, plus `__str__`, `__bool__`,
    `__repr__` for ergonomic use with `.text` and `.reasoning` projections.
    """

    __slots__ = ()

    def push(self, delta: str) -> None:
        """追加一个文本 delta。"""
        if not isinstance(cast("Any", delta), str):
            msg = "SyncTextProjection requires a string delta"
            raise TypeError(msg)
        super().push(delta)

    def complete(self, final_value: str) -> None:
        """设置最终累积文本并标记投影为完成。"""
        if not isinstance(cast("Any", final_value), str):
            msg = "SyncTextProjection requires a string final value"
            raise TypeError(msg)
        super().complete(final_value)

    def __iter__(self) -> Iterator[str]:
        """生成文本 deltas，如果生产者提供非字符串值则抛出异常。"""
        for delta in super().__iter__():
            if not isinstance(delta, str):
                msg = "SyncTextProjection received a non-string delta"
                raise TypeError(msg)
            yield delta

    def get(self) -> str:
        """耗尽并返回完整的累积字符串，如果未完成则返回空字符串。"""
        value = super().get()
        if value is None:
            return ""
        if not isinstance(value, str):
            msg = "SyncTextProjection received a non-string final value"
            raise TypeError(msg)
        return value

    def __str__(self) -> str:
        """耗尽并返回完整的累积字符串。"""
        return self.get()

    def __bool__(self) -> bool:
        """返回是否已推送任何 deltas。"""
        return len(self._deltas) > 0

    def __repr__(self) -> str:
        """返回目前为止累积文本的 repr。"""
        if self._final_set:
            return repr(self._final_value)
        return repr("".join(self._deltas))


# ---------------------------------------------------------------------------
# 异步投影
# ---------------------------------------------------------------------------


class AsyncProjection(_ProjectionBase):
    """既是可异步迭代的增量，也可等待最终值。

    Uses an `asyncio.Event` to notify consumers of state changes. Each
    waiter — the awaitable (`__await__`) and each async iterator cursor
    — shares the event and re-checks its own condition on wake. The event
    is cleared before a waiter awaits, so stale "something happened"
    signals don't cause spin loops.

    This is single-loop only — producers and consumers must share an
    event loop. If cross-thread wake is ever required, revert to a
    list-of-futures pattern with `call_soon_threadsafe`.
    """

    __slots__ = ("_arequest_more", "_ensure_started", "_event")

    def __init__(self) -> None:
        """初始化时事件未设置且无泵回调。"""
        super().__init__()
        self._event = asyncio.Event()
        self._arequest_more: Callable[[], Awaitable[bool]] | None = None
        self._ensure_started: Callable[[], Awaitable[None]] | None = None

    def set_start(self, cb: Callable[[], Awaitable[None]] | None) -> None:
        """安装一个懒启动回调，在首次消费时调用。"""
        self._ensure_started = cb

    def set_arequest_more(self, cb: Callable[[], Awaitable[bool]] | None) -> None:
        """连接异步拉取回调，迭代器使用它来驱动源。

        与 `SyncProjection.set_request_more` 对应。在调用方驱动的
        流式处理中，消费者在缓冲区为空时调用此回调，以使拥有图前进一步。

        Args:
            cb: 异步无参可调用对象，当产生新事件时返回 `True`， 当源耗尽时返回
            `False`。传入 `None` 以解除连接。
        """
        self._arequest_more = cb

    def push(self, delta: Any) -> None:
        """追加一个 delta 并通知等待者。"""
        super().push(delta)
        self._event.set()

    def complete(self, final_value: Any) -> None:
        """设置最终值，标记为完成，并通知等待者。"""
        super().complete(final_value)
        self._event.set()

    def fail(self, error: BaseException) -> None:
        """标记为出错并通知等待者。"""
        super().fail(error)
        self._event.set()

        # -- 异步可迭代（生成 delta） ------------------------------------

    def __aiter__(self) -> _AsyncProjectionIterator:
        """返回 deltas 的异步迭代器。"""
        return _AsyncProjectionIterator(self)

        # -- 可等待（返回最终值） -----------------------------------

    def __await__(self) -> Generator[Any, None, Any]:
        """等待最终累积值。"""
        return self._await_impl().__await__()

    async def _await_impl(self) -> Any:
        """等待最终值被设置并返回它。

        当通过 `set_arequest_more` 连接了调用方驱动的泵时，驱动该泵 而不是阻塞在
        `self._event` 上；否则回退到事件（用于手动分发的测试）。
        """
        if self._ensure_started is not None:
            await self._ensure_started()
        while not self._final_set:
            if self._error is not None:
                raise self._error
            if self._arequest_more is not None:
                if not await self._arequest_more() and not self._final_set:
                                        # 在未完成此投影的情况下耗尽泵 —
                                        # 不会再有新数据到达。返回当前状态并
                                        # 让调用方通过返回的 `None` / 未设置错误观察缺失的最终结果。
                                        # 直接访问投影的内部列表/事件是
                    break
            else:
                self._event.clear()
                await self._event.wait()
        if self._error is not None:
            raise self._error
        return self._final_value


class _AsyncProjectionIterator:
    """对 `AsyncProjection` 的增量数据的异步迭代器。"""

    __slots__ = ("_offset", "_proj")

    def __init__(self, proj: AsyncProjection) -> None:
        """将游标初始化在位置 0。"""
        self._proj = proj
        self._offset = 0

    def __aiter__(self) -> _AsyncProjectionIterator:
        """返回 self 以满足异步迭代协议。"""
        return self

    async def __anext__(self) -> Any:
        """返回下一个 delta，必要时等待。

        当投影有 `_arequest_more` 泵连接时，在内部循环中耗尽它 （对应
        `SyncProjection.__iter__`），直到此游标前进或泵报告耗尽。如果没有泵，
        则回退到等待共享事件。
        """
        proj = self._proj
        if proj._ensure_started is not None:  # noqa: SLF001
            await proj._ensure_started()  # noqa: SLF001
        while True:
                        # 直接访问投影的内部列表/事件
                        # 有意为之 — 迭代器是投影的搭档，
                        # 依赖通过游标读取共享缓冲区。
            if self._offset < len(proj._deltas):  # noqa: SLF001
                item = proj._deltas[self._offset]  # noqa: SLF001
                self._offset += 1
                return item
            if proj.error is not None:
                raise proj.error
            if proj.done:
                raise StopAsyncIteration
            if proj._arequest_more is not None:  # noqa: SLF001
                                # 由调用方驱动：驱动生产者。泵可能带来新
                                # 的增量以供兄弟投影使用 — 循环直到我们的游标
                                # 前进、投影终止或泵耗尽。
                                # 同步流
                while (
                    self._offset >= len(proj._deltas)  # noqa: SLF001
                    and not proj.done
                ):
                    if not await proj._arequest_more():  # noqa: SLF001
                        break
                if (
                    self._offset >= len(proj._deltas)  # noqa: SLF001
                    and not proj.done
                ):
                    if proj.error is not None:
                        raise proj.error
                    raise StopAsyncIteration
            else:
                proj._event.clear()  # noqa: SLF001
                await proj._event.wait()  # noqa: SLF001


class _AsyncTextProjectionIterator(_AsyncProjectionIterator):
    """对 `AsyncTextProjection` 的文本增量数据的异步迭代器。"""

    __slots__ = ()

    async def __anext__(self) -> str:
        """返回下一个文本 delta。"""
        item = await super().__anext__()
        if not isinstance(item, str):
            msg = "AsyncTextProjection received a non-string delta"
            raise TypeError(msg)
        return item


class AsyncTextProjection(AsyncProjection):
    """专用于 `.text` 和 `.reasoning` 的字符串型异步投影。"""

    __slots__ = ()

    def push(self, delta: str) -> None:
        """追加一个文本 delta 并通知等待者。"""
        if not isinstance(cast("Any", delta), str):
            msg = "AsyncTextProjection requires a string delta"
            raise TypeError(msg)
        super().push(delta)

    def complete(self, final_value: str) -> None:
        """设置最终累积文本并通知等待者。"""
        if not isinstance(cast("Any", final_value), str):
            msg = "AsyncTextProjection requires a string final value"
            raise TypeError(msg)
        super().complete(final_value)

    def __aiter__(self) -> _AsyncTextProjectionIterator:
        """返回文本 deltas 的异步迭代器。"""
        return _AsyncTextProjectionIterator(self)

    def __await__(self) -> Generator[Any, None, str]:
        """等待完整的累积文本。"""
        return self._await_text_impl().__await__()

    async def _await_text_impl(self) -> str:
        """返回累积文本，如果最终值不是字符串则抛出异常。"""
        value = await self._await_impl()
        if value is None:
            return ""
        if not isinstance(value, str):
            msg = "AsyncTextProjection received a non-string final value"
            raise TypeError(msg)
        return value


# ---------------------------------------------------------------------------
# 投影实例 — 具体子类在其自己的 __init__ 中调用 super() 后创建同步或
# ---------------------------------------------------------------------------


class _ChatModelStreamBase:
    """聊天模型流式传输的共享状态与事件分发机制。

    Holds accumulated protocol state (text, reasoning, tool calls,
    usage, metadata) and the event-dispatch machinery that drives the
    typed projections. `ChatModelStream` (sync) and
    `AsyncChatModelStream` (async) inherit from this base and add the
    projection types and consumer APIs for their flavor.
    """

        # 异步变体。
        # 累积状态
    _text_proj: _ProjectionBase
    _reasoning_proj: _ProjectionBase
    _tool_calls_proj: _ProjectionBase

    def __init__(
        self,
        *,
        namespace: list[str] | None = None,
        node: str | None = None,
        message_id: str | None = None,
    ) -> None:
        self._namespace = namespace or []
        self._node = node
        self._message_id = message_id

                # 按线程索引键控的每块文本/推理存储。用于在不交叉污染
        self._text_acc: str = ""
        self._reasoning_acc: str = ""
                # 同一消息中同类型其他块的情况下填充最终块负载。没有
                # 每块存储，消息级累加器会将先前块的文本渗入后续最终块。
                # 每个最终块的有序快照，按事件索引键控。
                # .output.content 的唯一真实来源。类型化累加器
                # (text/reasoning/tool_calls/invalid_tool_calls) 继续服务于
        self._text_per_block: dict[int, str] = {}
        self._reasoning_per_block: dict[int, str] = {}
        self._tool_call_chunks: dict[int, dict[str, Any]] = {}
        self._tool_calls_acc: list[ToolCall] = []
        self._invalid_tool_calls_acc: list[InvalidToolCall] = []
        self._server_tool_call_chunks: dict[int, dict[str, Any]] = {}
                # 公开投影。
                # 原始事件重放缓冲区
                # -- 公共属性 ------------------------------------------------
                # -- 事件摄入（公开） ------------------------------------------
        self._blocks: dict[int, FinalizedContentBlock] = {}
        self._usage_value: UsageInfo | None = None
        self._start_metadata: MessageMetadata | None = None
        self._finish_metadata: dict[str, Any] | None = None
        self._additional_kwargs: dict[str, Any] | None = None
        self._done: bool = False
        self._error: BaseException | None = None
        self._output_message: AIMessage | None = None

                # content-block-start 为信息性事件 — 无需累积
        self._events: list[MessagesData] = []

        # -- 内部推送 API（由分派调用） ----------------------------

    @property
    def namespace(self) -> list[str]:
        """此消息的图命名空间路径。"""
        return self._namespace

    @property
    def node(self) -> str | None:
        """生成此消息的图节点。"""
        return self._node

    @property
    def message_id(self) -> str | None:
        """稳定的消息标识符。"""
        return self._message_id

    def set_message_id(self, message_id: str) -> None:
        """在运行开始后分配稳定的消息标识符。

        Called by the stream driver (`stream_events(version="v3")` /
        `astream_events(version="v3")`) after `on_chat_model_start` produces a run
        id. Not intended for end-user code.
        """
        self._message_id = message_id

    @property
    def done(self) -> bool:
        """流是否已完成。"""
        return self._done

    @property
    def has_events(self) -> bool:
        """是否记录了任何协议事件。"""
        return bool(self._events)

    @property
    def output_message(self) -> AIMessage | None:
        """如果流已完成，则返回组装后的消息，否则为 `None`。

        Unlike `ChatModelStream.output` (which blocks until the stream
        finishes), this never pumps, blocks, or raises. Intended for the
        stream driver (`stream_events(version="v3")` and its async
        equivalent) to check whether the stream produced a message before
        firing `on_llm_end` callbacks.
        """
        return self._output_message

        # 二进制/模式负载增量反映在最终

    def dispatch(self, event: Mapping[str, Any]) -> None:
        """将协议事件路由到相应的内部处理器。

        Public entry point for feeding events into the stream. Called by
        the stream driver (the `stream_events(version="v3")` pump and its
        async equivalent) and by any observer or test that needs to
        inject protocol events.
        """
        self._record_event(event)
        event_type = event.get("event")
        if event_type == "message-start":
            self._push_message_start(cast("MessageStartData", event))
        elif event_type == "content-block-delta":
            self._push_content_block_delta(cast("ContentBlockDeltaData", event))
        elif event_type == "content-block-finish":
            self._push_content_block_finish(cast("ContentBlockFinishData", event))
        elif event_type == "message-finish":
            self._finish(cast("MessageFinishData", event))
        elif event_type == "error":
            self.fail(RuntimeError(event.get("message", "Unknown error")))
                # content-block 完成事件中；无专用投影。

        # 旧 `content_block` 增量的过渡遗留路径，应在 `_event_delta` 转换后不再可达，保留

    def _record_event(self, event: Mapping[str, Any]) -> None:
        """将原始事件追加到重放缓冲区。"""
        self._events.append(cast("MessagesData", event))

    def _push_message_start(self, data: MessageStartData) -> None:
        """处理 `message-start` 事件。"""
        self._start_metadata = data.get("metadata")
        message_id = data.get("id")
        if message_id:
            self._message_id = message_id

    def _push_content_block_delta(self, data: ContentBlockDeltaData) -> None:
        """处理 `content-block-delta` 事件。"""
        delta = _event_delta(data)
        if delta is None:
            return
        event_idx = data.get("index")
        dtype = delta.get("type", "")

        if dtype == "text-delta":
            delta_text = delta.get("text", "")
            if delta_text:
                self._text_acc += delta_text
                if event_idx is not None:
                    self._text_per_block[event_idx] = (
                        self._text_per_block.get(event_idx, "") + delta_text
                    )
                self._text_proj.push(delta_text)
        elif dtype == "reasoning-delta":
            delta_r = delta.get("reasoning", "")
            if delta_r:
                self._reasoning_acc += delta_r
                if event_idx is not None:
                    self._reasoning_per_block[event_idx] = (
                        self._reasoning_per_block.get(event_idx, "") + delta_r
                    )
                self._reasoning_proj.push(delta_r)
        elif dtype == "block-delta":
            fields = delta.get("fields")
            if not isinstance(fields, dict):
                return
            btype = fields.get("type", "")
            if btype == "tool_call_chunk":
                tcc = cast("ToolCallChunk", fields)
                idx = data.get("index")
                if idx is None:
                    idx = tcc.get("index", len(self._tool_call_chunks))  # type: ignore[unreachable]
                _merge_block_delta_into_store(self._tool_call_chunks, idx, dict(tcc))
                chunk_block: ToolCallChunk = {
                    "type": "tool_call_chunk",
                    "id": tcc.get("id"),
                    "name": tcc.get("name"),
                    "args": tcc.get("args"),
                }
                if "index" in tcc:
                    chunk_block["index"] = tcc["index"]
                self._tool_calls_proj.push(chunk_block)
            elif btype == "server_tool_call_chunk":
                stcc = cast("ServerToolCallChunk", fields)
                idx = data.get("index")
                if idx is None:
                    idx = len(self._server_tool_call_chunks)  # type: ignore[unreachable]
                _merge_block_delta_into_store(
                    self._server_tool_call_chunks,
                    idx,
                    dict(stcc),
                )
        elif dtype == "legacy-block-delta":
            fields = delta.get("fields")
            if not isinstance(fields, dict):
                return
            btype = fields.get("type", "")
            if btype == "tool_call_chunk":
                tcc = cast("ToolCallChunk", fields)
                idx = data.get("index")
                if idx is None:
                    idx = tcc.get("index", len(self._tool_call_chunks))  # type: ignore[unreachable]
                _merge_chunk_into_store(self._tool_call_chunks, idx, dict(tcc))
                legacy_chunk_block: ToolCallChunk = {
                    "type": "tool_call_chunk",
                    "id": tcc.get("id"),
                    "name": tcc.get("name"),
                    "args": tcc.get("args"),
                }
                if "index" in tcc:
                    legacy_chunk_block["index"] = tcc["index"]
                self._tool_calls_proj.push(legacy_chunk_block)
            elif btype == "server_tool_call_chunk":
                stcc = cast("ServerToolCallChunk", fields)
                idx = data.get("index")
                if idx is None:
                    idx = len(self._server_tool_call_chunks)  # type: ignore[unreachable]
                _merge_chunk_into_store(
                    self._server_tool_call_chunks,
                    idx,
                    dict(stcc),
                )
        elif dtype == "data-delta":
                        # 在此处以供树内自定义测试用例或第三方发射器使用。
                        # 无线程索引 — 遗留行为：使用消息级
            return
        else:
                        # 累加器。为预索引语义保留；兼容桥接或任何树内提供者均不
                        # 使用。
                        # 此块未收到增量 — 将完整
            block = data.get("content_block")
            if not isinstance(block, dict):
                return
            btype = block.get("type", "")
            if btype != "tool_call_chunk":
                return
            tcc = cast("ToolCallChunk", block)
            idx = data.get("index")
            if idx is None:
                idx = tcc.get("index", len(self._tool_call_chunks))  # type: ignore[unreachable]
            _merge_chunk_into_store(self._tool_call_chunks, idx, dict(tcc))
            fallback_chunk_block: ToolCallChunk = {
                "type": "tool_call_chunk",
                "id": tcc.get("id"),
                "name": tcc.get("name"),
                "args": tcc.get("args"),
            }
            if "index" in tcc:
                fallback_chunk_block["index"] = tcc["index"]
            self._tool_calls_proj.push(fallback_chunk_block)

    def _resolve_block_text(self, idx: int | None, full_text: str) -> str:
        """返回索引为 `idx` 的单个文本块的权威文本。

        Prefers per-block delta accumulation; reconciles with the finish
        event's `full_text` when the provider emits authoritative text
        that differs from what the deltas built up.

        Does not mutate `self._text_acc` (the delta-sum accumulator) —
        the message-wide projection value is derived from per-block
        storage at `_finish` time, so reconciliation remains correct
        regardless of finish ordering across blocks.
        """
        if idx is None:
                        # 文本作为单一增量呈现，以便流式投影
                        # 能够反映它。
                        # 权威文本扩展了部分增量 — 发出
            if full_text and full_text != self._text_acc:
                self._text_acc = full_text
            return self._text_acc
        existing = self._text_per_block.get(idx, "")
        if full_text and full_text != existing:
            if not existing:
                                # 尾部，使增量消费者能看到完成状态。
                                # 否则：权威文本完全替换部分增量。不发出修正增量（流中语义会模糊）。`_text_acc`
                # 不进行拼接 — 最终值在 `_finish` 时从每块
                                # 存储中计算得出，因此即使其他块在此期间向 `_text_acc`
                # 添加内容，仍保持正确。
                self._text_acc += full_text
                self._text_proj.push(full_text)
            elif full_text.startswith(existing):
                                # 将提供方特定字段与累积的
                                # 推理文本一同保留。Anthropic 的 `signature` 在
                tail = full_text[len(existing) :]
                self._text_acc += tail
                self._text_proj.push(tail)
                        # `extras` 下到达，且在后续轮次中必需。仅在已有累积内容时覆盖
                        # `reasoning`；OpenAI
            # 可以发出无文本增量的推理块，而在其中写入空字符串会使下游序列化器合成
                        # 一个空摘要条目。
                        # 在内容块上保留提供方特定字段（extras 等）。`_assemble_message` 分别将
                        # {id, name, args, type} 最小形态投影到
                        # `AIMessage.tool_calls`。移除 `index` 以匹配 v1
            self._text_per_block[idx] = full_text
        return self._text_per_block.get(idx, "")

    def _resolve_block_reasoning(self, idx: int | None, full_r: str) -> str:
        """返回索引为 `idx` 的单个块的权威推理文本。

        与 `_resolve_block_text` 对应，用于推理投影。
        """
        if idx is None:
            if full_r and full_r != self._reasoning_acc:
                self._reasoning_acc = full_r
            return self._reasoning_acc
        existing = self._reasoning_per_block.get(idx, "")
        if full_r and full_r != existing:
            if not existing:
                self._reasoning_acc += full_r
                self._reasoning_proj.push(full_r)
            elif full_r.startswith(existing):
                tail = full_r[len(existing) :]
                self._reasoning_acc += tail
                self._reasoning_proj.push(tail)
            self._reasoning_per_block[idx] = full_r
        return self._reasoning_per_block.get(idx, "")

    def _push_content_block_finish(self, data: ContentBlockFinishData) -> None:
        """处理 `content-block-finish` 事件。"""
        block = _event_content_block(data)
        if block is None:
            return
        btype = block.get("type", "")
        idx = data.get("index")
        finalized: FinalizedContentBlock | None = None

        if btype == "text":
            text_block = cast("TextContentBlock", block)
            full_text = text_block.get("text", "")
            block_text = self._resolve_block_text(idx, full_text)
            finalized = cast(
                "FinalizedContentBlock",
                {
                    **text_block,
                    "type": "text",
                    "text": block_text,
                },
            )
        elif btype == "reasoning":
            reasoning_block = cast("ReasoningContentBlock", block)
            full_r = reasoning_block.get("reasoning", "")
            block_reasoning = self._resolve_block_reasoning(idx, full_r)
                        # （`AIMessage.init_tool_calls` 重建块时不带
                        # `index`）；参见 `_compat_bridge.py` 中的 `_finalize_block`。
                        # 在存储块上移除 `index` 以与 `tool_call` 路径保持对称。
                        # reasoning 当我们已积累内容时；OpenAI 可
                        # 发出一个无文本增量的推理块，并写入
                        # 空字符串在那里可使下游序列化器自动生成
                        # 一个空的摘要条目。
            finalized_dict: dict[str, Any] = {**reasoning_block, "type": "reasoning"}
            if block_reasoning:
                finalized_dict["reasoning"] = block_reasoning
            finalized = cast("FinalizedContentBlock", finalized_dict)
        elif btype == "tool_call":
            tcb = cast("ToolCall", block)
                        # 保留提供方特定的字段（extras 等）
                        # 内容块。`_assemble_message` 分别投影
                        # 最小化 {id, name, args, type} 形状
                        # `AIMessage.tool_calls`。移除 `index` 以匹配 v1
                        # (`AIMessage.init_tool_calls` 重新构建块而不
                        # `index`); 参见 `_compat_bridge.py` 中的 `_finalize_block`。
            tc = cast(
                "ToolCall",
                {
                    **{k: v for k, v in tcb.items() if k != "index"},
                    "type": "tool_call",
                    "id": tcb.get("id", ""),
                    "name": tcb.get("name", ""),
                    "args": tcb.get("args", {}),
                },
            )
            self._tool_calls_acc.append(tc)
            if idx is not None and idx in self._tool_call_chunks:
                del self._tool_call_chunks[idx]
            finalized = tc
        elif btype == "invalid_tool_call":
            itc = cast("InvalidToolCall", block)
                        # 移除存储块中的 `index`，以保持对称性
                        # the `tool_call` path.
            itc = cast(
                "InvalidToolCall",
                {k: v for k, v in itc.items() if k != "index"},
            )
            self._invalid_tool_calls_acc.append(itc)
                        # 关键：丢弃过时的块，以免 _finish 的扫尾操作将其重新激活为无参数的
            # ToolCall。
                        # 将其作为无参数的 ToolCall 重新激活。
            if idx is not None and idx in self._tool_call_chunks:
                del self._tool_call_chunks[idx]
            if idx is not None and idx in self._server_tool_call_chunks:
                del self._server_tool_call_chunks[idx]
            finalized = itc
        elif btype in {
            "server_tool_call",
            "server_tool_result",
            "image",
            "audio",
            "video",
            "file",
            "non_standard",
        }:
            if btype == "server_tool_call" and idx is not None:
                self._server_tool_call_chunks.pop(idx, None)
            finalized = cast("FinalizedContentBlock", block)

        if finalized is not None and idx is not None:
                        # 在源端未提供索引时，向已定稿的块回填线程索引。`langchain_core.utils._merge`
            # 的块合并器（由 `AIMessageChunk.__add__` /
                        # `add_ai_message_chunks` 使用）依赖于 `block["index"]`
            # 来将增量分组到同一输出块中——缺少它会导致 v2 组装的 `AIMessage`
            # 在后续重新进入块聚合路径时无法正确合并。客户端的 `tool_call` /
            # `invalid_tool_call` 块被排除在外：v1 的定稿会丢弃它们的
            # `index`，以防止后续增量覆盖已解析的参数，v2 也遵循此行为。
                        # 非标准扩展，携带提供方侧的
            # `additional_kwargs`，这些字段不映射到类型化协议字段（例如 Gemini 的
            # `__gemini_function_call_thought_signatures__`）。兼容桥接在
            # `message-finish` 上发出此内容，以便组装的消息能携带 `ainvoke`
            # 本应保留的数据。
                        # 定稿所有未被扫尾处理的块——包括客户端和服务器端。
                        # 当任何带索引的文本或推理到达时，优先使用每块的求和结果——无论完成顺序如何，以及完成事件是否携带与增量不同的权威性文本，该结果始终正确。仅在旧版无索引路径中回退到增量求和累加器。
                        # 从未收到协议块。回退到累积的文本（可能为空）作为纯字符串内容。
                        # `ChatModelStream` 是 v1
            # 内容块的表面接口：只要有任何块到达，内容始终是协议块列表。
                        # 不要将单个文本块折叠为纯字符串——这会丢失块级字段（`id`、`index`、
                        # annotations、extras），下游序列化器需要这些字段在后续回合中完整往返消息。
                        # 最后固定 `output_version`：`stream_events(version="v3")` 总是将内容组装为
            # v1 协议块，无论提供方配置的输出格式如何。
            if btype not in {"tool_call", "invalid_tool_call"}:
                finalized.setdefault("index", idx)
            self._blocks[idx] = finalized

    def _finish(self, data: MessageFinishData) -> None:
        """处理 `message-finish` 事件。"""
        self._done = True
        self._usage_value = data.get("usage")
        self._finish_metadata = cast("dict[str, Any] | None", data.get("metadata"))
                # 提供方在完成元数据中提供的 `output_version`（例如
        # `ChatOpenAI(use_responses_api=True, ...)` 返回的 `"responses/v1"`）否则会导致
        # `AIMessage.content_blocks` 对已为 v1 的内容重新运行错误的转换器。
                # 同步流式
                # 投影 —— 事先创建
                # 拉取回调（由 bind_pump 或 set_request_more 设置）
                # -- 泵/拉取连接 --------------------------------------------------
        self._additional_kwargs = cast(
            "dict[str, Any] | None",
            cast("dict[str, Any]", data).get("additional_kwargs"),
        )

                # -- 公共投影 ------------------------------------------------
        _sweep_chunk_store(
            self._tool_call_chunks,
            finalized_type="tool_call",
            finalized_blocks=self._blocks,
            tool_calls_acc=self._tool_calls_acc,
            invalid_acc=self._invalid_tool_calls_acc,
        )
        _sweep_chunk_store(
            self._server_tool_call_chunks,
            finalized_type="server_tool_call",
            finalized_blocks=self._blocks,
            tool_calls_acc=None,
            invalid_acc=self._invalid_tool_calls_acc,
        )

                # -- 原始事件迭代（重放缓冲区） -------------------------------
                # -- 内部辅助函数 --------------------------------------------------
                # 异步流式
                # 由 `aclose()`
        # 调用的清理回调，仅在生产者任务在主体执行前被取消时触发（因此正常的 `_produce`
        # CancelledError 处理程序——会触发 `on_llm_error`——从未执行）。由
        # `astream_events(version="v3")` 设置。
                # -- 泵/拉取连接（异步） ------------------------------------------
        if self._text_per_block:
            text_final = "".join(
                self._text_per_block[i] for i in sorted(self._text_per_block)
            )
        else:
            text_final = self._text_acc
        if self._reasoning_per_block:
            reasoning_final = "".join(
                self._reasoning_per_block[i] for i in sorted(self._reasoning_per_block)
            )
        else:
            reasoning_final = self._reasoning_acc

        self._text_proj.complete(text_final)
        self._reasoning_proj.complete(reasoning_final)
        self._tool_calls_proj.complete(self._tool_calls_acc)
        self._output_message = self._assemble_message()

    def fail(self, error: BaseException) -> None:
        """标记流为错误状态，并向所有投影传播。

        Public API — called by the stream driver (`stream_events(version="v3")` /
        `astream_events(version="v3")`) when the underlying producer raises, by
        `dispatch` when an `error` protocol event arrives, and by
        cancellation paths.
        """
        self._done = True
        self._error = error
        self._text_proj.fail(error)
        self._reasoning_proj.fail(error)
        self._tool_calls_proj.fail(error)

    def _assemble_message(self) -> AIMessage:
        """根据累积状态构建一个 `AIMessage`。

        Content is built from `self._blocks`, an index-ordered snapshot of
        finalized protocol blocks. The bare-string fast path is used when
        the message has exactly one `text` block (the common chat case);
        otherwise content is a list of protocol-shape block dicts.
        """
        content: Any
        if not self._blocks:
                        # -- 公共投影 ------------------------------------------------
                        # -- 清理 -----------------------------------------------------------
            content = self._text_acc
        else:
                        # 通过关联的 `Future` 等待任务，而非直接等待
                        # 当任何区块到达时，始终是一个协议区块列表。
                        # 不要将单个文本块折叠为纯字符串 —
                        # 这将丢弃块级字段（`id`、`index`，
                        # 下游序列化器需要的注解、附加项
                        # 在后续轮次中往返传递消息。
            ordered_blocks = [self._blocks[idx] for idx in sorted(self._blocks)]
            content = [dict(b) for b in ordered_blocks]

        response_metadata: dict[str, Any] = {}
        if self._start_metadata:
            if "provider" in self._start_metadata:
                response_metadata["model_provider"] = self._start_metadata["provider"]
            if "model" in self._start_metadata:
                response_metadata["model_name"] = self._start_metadata["model"]
        if self._finish_metadata:
            response_metadata.update(self._finish_metadata)
                # 最后固定 `output_version`：`stream_events(version="v3")` 始终有效
                # 以 v1 协议块的形式组装内容，与之无关
                # 提供方配置的输出格式。
                # 提供方提供的 `output_version` 在 finish 元数据中（例如。
                # `"responses/v1"` 来自 `ChatOpenAI(use_responses_api=True, ...)`)
                # 否则会导致 `AIMessage.content_blocks` 重新运行错误
                # 译者注：针对已存在的 v1 内容。
        response_metadata["output_version"] = "v1"

        tool_calls = [
            {
                "id": tc.get("id", ""),
                "name": tc.get("name", ""),
                "args": tc.get("args", {}),
                "type": "tool_call",
            }
            for tc in self._tool_calls_acc
        ]

        invalid_tool_calls = [
            {
                "type": "invalid_tool_call",
                "id": itc.get("id") or None,
                "name": itc.get("name") or None,
                "args": itc.get("args") or None,
                "error": itc.get("error"),
            }
            for itc in self._invalid_tool_calls_acc
        ]

        message_kwargs: dict[str, Any] = {
            "content": content,
            "id": self._message_id,
            "tool_calls": tool_calls,
            "invalid_tool_calls": invalid_tool_calls,
            "usage_metadata": self._usage_value,
            "response_metadata": response_metadata,
        }
        if self._additional_kwargs:
            message_kwargs["additional_kwargs"] = dict(self._additional_kwargs)
        return AIMessage(**message_kwargs)


# ---------------------------------------------------------------------------
# 同步流式
# ---------------------------------------------------------------------------


class ChatModelStream(_ChatModelStreamBase):
    """单个 LLM 响应的同步消息流式对象。

    Returned by `BaseChatModel.stream_events(version="v3")`.  Content-block protocol
    events are fed into this object and accumulated into typed projections.

    投影（始终返回相同的缓存对象）：

    - `.text` —— `str` 增量的可迭代对象；通过 `str()` 获取完整文本
    - `.reasoning` —— 推理内容的处理方式与 `.text` 相同
    - `.tool_calls` —— `ToolCallChunk` 增量的可迭代对象；
      `.get()` 返回 `list[ToolCall]`
    - `.output` —— 阻塞属性，返回组装后的 `AIMessage`

    流结束之后，可通过 `.output.usage_metadata` 获取用量信息。

    !!! note "输出结构始终为 v1 内容块"

        `.output.content` 始终是 v1 协议块列表
        （文本、推理、工具调用、图像等），无论底层模型的 `output_version`
        设置如何。该属性仅控制旧版的 `stream()` / `astream()` / `invoke()`
        路径；`ChatModelStream` 基于内容块协议构建，其输出结构天然为 v1 形式。

    Raw event iteration::

        for event in stream:
            print(event)  # MessagesData 字典
    """

    _text_proj: SyncTextProjection
    _reasoning_proj: SyncTextProjection
    _tool_calls_proj: SyncProjection

    def __init__(  # noqa: D107
        self,
        *,
        namespace: list[str] | None = None,
        node: str | None = None,
        message_id: str | None = None,
    ) -> None:
        super().__init__(namespace=namespace, node=node, message_id=message_id)
                # 投影 — 会立即创建
        self._text_proj = SyncTextProjection()
        self._reasoning_proj = SyncTextProjection()
        self._tool_calls_proj = SyncProjection()
                # 拉取回调（由 bind_pump 或 set_request_more 设置）
        self._ensure_started: Callable[[], None] | None = None
        self._request_more: Callable[[], bool] | None = None

        # -- 泵/拉取布线 --------------------------------------------------

    def bind_pump(self, pump_one: Callable[[], bool]) -> None:
        """绑定一个独立流式传输的泵。

        Delegates to `set_request_more`.  Used by
        `BaseChatModel.stream_events(version="v3")`.
        """
        self.set_request_more(pump_one)

    def set_start(self, cb: Callable[[], None] | None) -> None:
        """在此流及其投影上安装一个延迟启动回调。"""
        self._ensure_started = cb
        self._text_proj.set_start(cb)
        self._reasoning_proj.set_start(cb)
        self._tool_calls_proj.set_start(cb)

    def set_request_more(self, cb: Callable[[], bool]) -> None:
        """设置此流及其所有投影的拉取回调。

        由 langgraph 的 `GraphRunStream._wire_request_more` 使用，以连接共享图泵。
        """
        self._request_more = cb
        self._text_proj.set_request_more(cb)
        self._reasoning_proj.set_request_more(cb)
        self._tool_calls_proj.set_request_more(cb)

        # -- 公共投影 ------------------------------------------------

    @property
    def text(self) -> SyncTextProjection:
        """文本内容 —— 可迭代的 `str` 增量，`str()` 返回完整内容。"""
        return self._text_proj

    @property
    def reasoning(self) -> SyncTextProjection:
        """推理内容 —— 与 :attr:`text` 具有相同接口。"""
        return self._reasoning_proj

    @property
    def tool_calls(self) -> SyncProjection:
        """工具调用 —— 可迭代的 `ToolCallChunk` 增量。

        .get() 返回最终的 `list[ToolCall]`。
        """
        return self._tool_calls_proj

    @property
    def output(self) -> AIMessage:
        """组装后的 `AIMessage` —— 阻塞直到流结束。"""
        self._drain()
        if self._error is not None:
            raise self._error
        if self._output_message is None:
            msg = "Stream finished without producing a message"
            raise RuntimeError(msg)
        return self._output_message

        # -- 原始事件迭代（重放缓冲区） -------------------------------

    def __iter__(self) -> Iterator[MessagesData]:
        """以重放缓冲区语义迭代原始协议事件。"""
        if self._ensure_started is not None:
            self._ensure_started()
        cursor = 0
        while True:
            if cursor < len(self._events):
                yield self._events[cursor]
                cursor += 1
            elif self._error is not None:
                raise self._error
            elif self._done:
                return
            elif self._request_more is not None:
                while cursor >= len(self._events) and not self._done:
                    if not self._request_more():
                        break
                if cursor >= len(self._events):
                    if self._error is not None:
                        raise self._error
                    return
            else:
                return

        # -- 内部辅助函数 --------------------------------------------------

    def _drain(self) -> None:
        """拉取所有剩余事件直至完成。"""
        if self._done:
            return
        if self._ensure_started is not None:
            self._ensure_started()
        if self._request_more is not None:
            while not self._done:
                if not self._request_more():
                    break


# ---------------------------------------------------------------------------
# 异步流式
# ---------------------------------------------------------------------------


class AsyncChatModelStream(_ChatModelStreamBase):
    """单个 LLM 响应的异步消息流式对象。

    Returned by `BaseChatModel.astream_events(version="v3")`.  Content-block events
    are fed into this object by a background producer task.

    投影：

    - `.text` —— 文本增量的异步可迭代对象；可等待获取完整文本
    - `.reasoning` —— 推理增量的异步可迭代对象；可等待
    - `.tool_calls` —— `ToolCallChunk` 增量的异步可迭代对象；
      可等待获取 `list[ToolCall]`
    - `.output` —— 可等待获取组装后的 `AIMessage`

    流结束之后，可通过 `.output.usage_metadata` 获取用量信息。

    !!! note "输出结构始终为 v1 内容块"

        The assembled message's content is always a list of v1
        protocol blocks, regardless of the model's `output_version`
        setting — see `ChatModelStream` for the full rationale.

    The stream itself is awaitable (`msg = await stream`) and
    async-iterable (`async for event in stream`).
    """

    _text_proj: AsyncTextProjection
    _reasoning_proj: AsyncTextProjection
    _tool_calls_proj: AsyncProjection

    def __init__(  # noqa: D107
        self,
        *,
        namespace: list[str] | None = None,
        node: str | None = None,
        message_id: str | None = None,
    ) -> None:
        super().__init__(namespace=namespace, node=node, message_id=message_id)
        self._text_proj = AsyncTextProjection()
        self._reasoning_proj = AsyncTextProjection()
        self._tool_calls_proj = AsyncProjection()
        self._output_proj = AsyncProjection()
        self._events_proj = AsyncProjection()
        self._ensure_started: Callable[[], Awaitable[None]] | None = None
        self._producer_task: asyncio.Task[None] | None = None
                # 由 `aclose()` 调用的清理回调，仅在生产者存在时触发
                # 任务在执行体运行前被取消（因此正常
                # _produce 取消错误处理程序 — 会触发
                # `on_llm_error` — 永远不会执行）。由 `astream_events(version="v3")` 设置。
        self._on_aclose_fail: Callable[[BaseException], Awaitable[None]] | None = None

        # -- 泵/拉取布线（异步）------------------------------------------

    def set_arequest_more(self, cb: Callable[[], Awaitable[bool]] | None) -> None:
        """将异步泵回调分发到每个投影。

        由 langgraph 的 `AsyncGraphRunStream._wire_arequest_more` 使用，以便在
        `stream.text`、`stream.reasoning` 等的缓冲区为空时，驱动共享图泵的游标。

        Args:
            cb: 异步无参可调用对象，当产生新事件时返回 `True`，当源耗尽时返回
            `False`。传入 `None` 以取消绑定。
        """
        for proj in (
            self._text_proj,
            self._reasoning_proj,
            self._tool_calls_proj,
            self._output_proj,
            self._events_proj,
        ):
            proj.set_arequest_more(cb)

    def set_start(self, cb: Callable[[], Awaitable[None]] | None) -> None:
        """在此流及其投影上安装一个延迟启动回调。"""
        self._ensure_started = cb
        for proj in (
            self._text_proj,
            self._reasoning_proj,
            self._tool_calls_proj,
            self._output_proj,
            self._events_proj,
        ):
            proj.set_start(cb)

        # -- 公共投影 ------------------------------------------------

    @property
    def text(self) -> AsyncTextProjection:
        """文本内容 — 异步可迭代的 `str` delta，可等待获取完整文本。"""
        return self._text_proj

    @property
    def reasoning(self) -> AsyncTextProjection:
        """推理内容 — 与 :attr:`text` 具有相同接口。"""
        return self._reasoning_proj

    @property
    def tool_calls(self) -> AsyncProjection:
        """工具调用 — 异步可迭代，可等待获取最终列表。"""
        return self._tool_calls_proj

    @property
    def output(self) -> AsyncProjection:
        """组装后的 `AIMessage` — 可等待。"""
        return self._output_proj

    def __await__(self) -> Generator[Any, None, AIMessage]:
        """等待组装后的 `AIMessage` 和完整的生产者生命周期。

        在输出投影解析后等待生产者任务，以确保调用方 `await`
        返回时，后续流处理工作（尤其是 `on_llm_end` 回调）已执行。
        """
        return self._await_full().__await__()

    async def _await_full(self) -> AIMessage:
        if self._ensure_started is not None:
            await self._ensure_started()
        message: AIMessage = await self._output_proj
        if self._producer_task is not None:
            await self._producer_task
        return message

    def __aiter__(self) -> _AsyncProjectionIterator:
        """异步迭代原始协议事件。"""
        return _AsyncProjectionIterator(self._events_proj)

        # -- 清理 -----------------------------------------------------------

    async def aclose(self) -> None:
        """取消后台生产者任务并释放资源。

        If a consumer cancels mid-stream or decides to stop iterating
        early, the producer task keeps pumping the provider HTTP call to
        completion because `asyncio.Task` has no implicit link to its
        awaiter. Call this method to cancel the producer explicitly; the
        stream transitions to an errored state with `CancelledError`.

        如果流已成功生成消息（例如，在 `await stream.output`
        之后），生产者可能仍在运行后续工作，如 `on_llm_end`
        回调。在这种情况下，`aclose()`
        会等待任务完成而非取消它——将一次成功运行变为取消状态会丢失结束回调并破坏追踪。

        Idempotent: safe to call multiple times, including after the
        stream has finished normally. Also invoked by the async context
        manager protocol on `__aexit__`.
        """
        if self._ensure_started is not None and self._producer_task is None:
            await self._ensure_started()

        task = self._producer_task
        if task is None:
            return
        if task.done() and self._done:
            return

        we_cancelled = not (self._output_message is not None and self._error is None)
        if we_cancelled and not task.done():
            task.cancel()

                # 通过关联的 `Future` 等待任务，而不是通过等待
                # 直接执行任务。等待任务将引发 `CancelledError`
                # 在两种无法区分的情况下：(1) 我们刚刚取消的任务
                # 已完成，(2) 我们的调用者取消了我们。`asyncio.Task.cancelling()`
                # 在 3.11+ 版本中可进行区分，但在 3.10 中不存在。
        #
                # `done_future` 在任务完成时（任何原因）解析为 `None`。它本身不是 `Task`，因此其
                # `await`
        # 仅在调用方被取消时才会引发异常——为我们提供了一个可移植且无歧义的信号以进行传播。
                # 如果任务在 `_produce` 执行前就被取消（例如
                # `astream_events(version="v3")` 紧随 `aclose()` 之后），则流
        if not task.done():
            loop = asyncio.get_running_loop()
            done_future: asyncio.Future[None] = loop.create_future()

            def _link(_: asyncio.Task[None]) -> None:
                if not done_future.done():
                    done_future.set_result(None)

            task.add_done_callback(_link)
            try:
                await done_future
            finally:
                task.remove_done_callback(_link)

                # 从未到达 `_produce` 的 CancelledError
        # 处理器——其投影仍处于待处理状态且生命周期结束回调
                # 尚未触发。在此处同时解决，以确保 `await stream.output`
        # 的调用者不会挂起，并且追踪能看到匹配的结束事件。
                # -- 内部 API（扩展基类以驱动异步投影） -------------
                # 投影仍在等待中，且没有生命周期结束回调
                # 已触发。在此处同时解决，以便调用 `await stream.output` 的调用方
                # 不要挂起，否则追踪会看到匹配的结束事件。
        if we_cancelled and not self._done:
            cancel_exc = asyncio.CancelledError()
            self.fail(cancel_exc)
            teardown = self._on_aclose_fail
            if teardown is not None:
                with contextlib.suppress(Exception):
                    await teardown(cancel_exc)

    async def __aenter__(self) -> Self:
        """进入异步上下文 — 返回自身。"""
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: object,
    ) -> None:
        """退出异步上下文 — 通过 `aclose()` 取消生产者。"""
        del exc_type, exc, tb
        await self.aclose()

        # -- 内部 API（扩展基础以驱动异步投影） -------------

    def _record_event(self, event: Mapping[str, Any]) -> None:
        """记录事件并推送到异步事件重放投影。"""
        super()._record_event(event)
        self._events_proj.push(cast("MessagesData", event))

    def _finish(self, data: MessageFinishData) -> None:
        """完成基础投影和仅异步的投影。"""
        super()._finish(data)
        self._output_proj.complete(self._output_message)
        self._events_proj.complete(self._events)

    def fail(self, error: BaseException) -> None:
        """失败基础投影和仅异步的投影。"""
        super().fail(error)
        self._output_proj.fail(error)
        self._events_proj.fail(error)


__all__ = [
    "AsyncChatModelStream",
    "AsyncProjection",
    "AsyncTextProjection",
    "ChatModelStream",
    "SyncProjection",
    "SyncTextProjection",
]

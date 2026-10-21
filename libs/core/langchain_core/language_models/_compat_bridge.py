"""Compat bridge: 兼容性桥接：将 `AIMessageChunk` 流转换为协议事件。

The bridge trusts `AIMessageChunk.content_blocks` as the single
protocol view of any chunk.  That property runs the three-tier lookup
(`output_version == "v1"` short-circuit, registered translator, or
best-effort parsing) and returns a `list[ContentBlock]` for every
well-formed message — whether the provider is a registered partner, an
unregistered community model, or not tagged at all.

每块的 `content_blocks` 输出是一个 **增量切片**，而非累积状态：此生态系统中的提供方发出
SSE 风格的块，每个块都携带其自身的增量。因此，桥接直接将每个切片转发为
`content-block-delta` 事件，并仅按索引累积状态，以便最终的 `content-block-finish`
事件能够报告已定稿的块（例如，`tool_call_chunk` 参数解析为字典）。

Lifecycle::

    message-start
      -> content-block-start   （每次索引首次出现时）
      -> content-block-delta*  （每块，携带切片）
      -> content-block-finish  （定稿块）
    -> message-finish

公共 API：

- `chunks_to_events` / `achunks_to_events` — for live streams where
  chunks arrive over time.
- `message_to_events` / `amessage_to_events` — for replaying a finalized
  `AIMessage` (cache hit, checkpoint restore, graph-node return value)
  as a synthetic event lifecycle.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, cast

from langchain_protocol.protocol import (
    ContentBlock,
    ContentBlockDeltaData,
    ContentBlockFinishData,
    ContentBlockStartData,
    FinalizedContentBlock,
    InvalidToolCall,
    MessageFinishData,
    MessageMetadata,
    MessagesData,
    MessageStartData,
    ServerToolCall,
    ServerToolCallChunk,
    ToolCall,
    ToolCallChunk,
    UsageInfo,
)

from langchain_core.messages import AIMessageChunk, BaseMessage
from langchain_core.utils._merge import merge_dicts

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Iterator

    from langchain_protocol.protocol import (
        BlockDelta,
        BlockDeltaFields,
        ContentBlockDelta,
        DataDelta,
        ReasoningDelta,
        TextDelta,
    )

    from langchain_core.messages.ai import UsageMetadata
    from langchain_core.outputs import ChatGenerationChunk


CompatBlock = dict[str, Any]
"""Internal working type for a content block.

The bridge works with plain dicts internally because two separate but
structurally similar `ContentBlock` Unions exist — one in
`langchain_core.messages.content` (returned by `msg.content_blocks`),
one in `langchain_protocol.protocol` (the wire/event shape).  They are
not mypy-compatible despite being near-isomorphic.  Passing through
`dict[str, Any]` launders between them.  See `_to_protocol_block` for
the single seam where the laundering cast lives.
"""


# ---------------------------------------------------------------------------
# 核心与协议 `ContentBlock` 联合类型之间的类型伪装
# ---------------------------------------------------------------------------


def _to_protocol_block(block: CompatBlock) -> ContentBlock:
    """将内部工作字典缩小为协议 `ContentBlock`。

    两个 `ContentBlock` 类型系统之间的唯一接口：
    `langchain_core.messages.content`（`msg.content_blocks` 返回的内容）与
    `langchain_protocol.protocol`（事件负载所需的内容）。这两个 Union
    在结构上重叠，但对 mypy 而言是名义上不同的，因此我们通过 `dict[str, Any]`
    进行转换。当两个 Union 统一时，此辅助函数及其最终版本可被删除。
    """
    return cast("ContentBlock", block)


def _to_finalized_block(block: CompatBlock) -> FinalizedContentBlock:
    """与 `_to_protocol_block` 对应的定稿块版本。"""
    return cast("FinalizedContentBlock", block)


def _to_block_delta_fields(block: CompatBlock) -> BlockDeltaFields:
    """将内部工作字典缩小为协议块增量字段。"""
    return cast("BlockDeltaFields", block)


def _to_content_delta(block: CompatBlock) -> ContentBlockDelta:
    """将内容块切片/快照转换为显式的协议增量。"""
    btype = block.get("type")
    if btype == "text":
        return cast("TextDelta", {"type": "text-delta", "text": block.get("text", "")})
    if btype == "reasoning":
        return cast(
            "ReasoningDelta",
            {
                "type": "reasoning-delta",
                "reasoning": block.get("reasoning", ""),
            },
        )
    if "data" in block:
        delta = cast("DataDelta", {"type": "data-delta", "data": block.get("data", "")})
        if block.get("encoding") == "base64":
            delta["encoding"] = "base64"
        return delta
    return cast(
        "BlockDelta",
        {
            "type": "block-delta",
            "fields": _to_block_delta_fields(block),
        },
    )


# ---------------------------------------------------------------------------
# 块迭代
# ---------------------------------------------------------------------------


def _iter_protocol_blocks(msg: BaseMessage) -> list[tuple[Any, CompatBlock]]:
    """从 `msg.content_blocks` 中读取每块的协议块。

    Returns `(key, block)` pairs.  The key is the block's stable identifier
    across the stream: the block's `index` field when present (can be an
    int or a string — some providers use string identifiers like
    `"lc_rs_305f30"`), or the positional index within the message as a
    fallback.  Callers are responsible for allocating wire-level `uint`
    indices; this helper only surfaces the source-side identity.

    For finalized `AIMessage`, also surfaces `invalid_tool_calls`
    — which `AIMessage.content_blocks` currently omits from its return
    value even though they are a defined protocol block type.

    The positional fallback is a known fragility: when a provider emits
    blocks without an `index` field (e.g. Anthropic's `_stream` with
    `coerce_content_to_string=True`, where text chunks lose their
    source-side index), every such chunk gets positional key 0 and
    successive chunks merge into one block. This works correctly for
    single-type streams (pure-text responses merge cleanly) because all
    chunks share the same key and the open-block logic collapses them.
    It would miscategorise a stream that mixed indexed structured
    blocks with non-indexed coerced-text blocks, since an indexed
    block with `index == 0` would collide with the anonymous text
    block's positional-0 key.  In the anthropic integration this
    cannot currently occur: coerce-to-string mode is only selected
    when no tools, thinking, or documents are present, and any of
    those flips the stream to structured mode where every block
    carries an integer index.  A native `_stream_chat_model_events`
    hook per provider (or a bridge-level "continue the open block when
    the source has no identity" rule) would close the gap if another
    integration ever emits mixed content.
    """
    try:
        raw = msg.content_blocks
    except Exception:
        return []

    result: list[tuple[Any, CompatBlock]] = []
    for i, block in enumerate(raw):
        if not isinstance(block, dict):
            continue  # type: ignore[unreachable]
        explicit_idx = block.get("index")
        if explicit_idx is None:
                        # 无源端身份标识。按 (哨兵, 块类型,
                        # 位置索引 `i`) 分桶，因此在不同块类型位于
                        # 相同位置的多个块（例如 Gemini 在一个分块中发出一个
                        # 推理块，在下一个分块中发出一个 `tool_call`，两者在位置
            # 0，因为每个分块只携带一个
                        # 块）会获得不同的网络块。若无此机制，第二个
                        # 类型的传入块将触发 `_accumulate` 的自包含
                        # `else`
            # 分支，覆盖第一个块。同类型分块仍共享同一分桶并可干净合并，这正是流式文本
            # / 推理所依赖的。
                        # 最终确定的 AIMessage：从专用字段中提取 invalid_tool_calls —— 当前
            # `AIMessage.content_blocks` 不包含它们。
                        # 每块辅助函数
                        # 可承载大负载的字段（内联 base64 媒体、解析后的参数、
            key: Any = ("__lc_no_index__", block.get("type"), i)
        else:
            key = explicit_idx
        result.append((key, dict(block)))

    if not isinstance(msg, AIMessageChunk):
                # 任意字典）。从 `content-block-start` 中移除，以使负载仅通过
        # `content-block-finish` 传递，避免在传输过程中被序列化两次。
                # extras 和 id
        for itc in getattr(msg, "invalid_tool_calls", None) or []:
            itc_block: CompatBlock = {"type": "invalid_tool_call"}
            for key_name in ("id", "name", "args", "error"):
                if itc.get(key_name) is not None:
                    itc_block[key_name] = itc[key_name]
            result.append((len(result), itc_block))

    return result


# ---------------------------------------------------------------------------
# extras 和 id
# ---------------------------------------------------------------------------


# 用最小占位符恢复必需但重量级的字段，以确保起始事件仍能通过该块类型的 CDDL 模式验证。
# 提供方可能在后续 delta 中发送非文本字段（如 `id` 或注解）。
# 合并（而非替换）可保留早期键值，同时接收这些延迟到达的字段。
# 提供方可能在后续 delta 中发送非文本字段。Claude 的
_HEAVY_FIELDS = frozenset({"args", "data", "output", "transcript", "value"})


def _start_skeleton(block: CompatBlock) -> ContentBlock:
    """用于 `content-block-start` 事件的空内容占位符。

    Deltaable block types (text, reasoning, the `_chunk` tool variants)
    get an empty payload so the lifecycle's "start" signal is distinct
    from the first incremental delta.  Self-contained types (image,
    audio, video, file, non_standard, finalized tool calls) drop their
    heavy payload fields; those are carried by `content-block-finish`.
    Correlation fields (id, name, toolCallId) and small metadata
    (mime_type, url, status, phase, …) are preserved on the start event.
    """
    btype = block.get("type", "text")
    if btype == "text":
                # `signature_delta` 在推理文本之后到达，作为 `extras.signature` 显现；
        text_skel = {
            k: v for k, v in block.items() if k not in {"text", "index", "annotations"}
        }
        text_skel["text"] = ""
        return _to_protocol_block(text_skel)
    if btype == "reasoning":
                # 合并（而非替换）可保留早期键值。
        r_skel = {k: v for k, v in block.items() if k not in {"reasoning", "index"}}
        r_skel["reasoning"] = ""
        return _to_protocol_block(r_skel)
    if btype == "tool_call_chunk":
        return ToolCallChunk(
            type="tool_call_chunk",
            id=block.get("id"),
            name=block.get("name"),
            args="",
        )
    if btype == "server_tool_call_chunk":
        s_skel = ServerToolCallChunk(
            type="server_tool_call_chunk",
            args="",
        )
        if block.get("id") is not None:
            s_skel["id"] = block["id"]
        if block.get("name") is not None:
            s_skel["name"] = block["name"]
        return s_skel

    stripped: CompatBlock = {k: v for k, v in block.items() if k not in _HEAVY_FIELDS}
        # 自包含或已最终确定的类型：整体替换。
        # 将累积分块中的提供方特定字段携带到最终块上。显式丢弃我们重写的仅分块存在的键。
    if btype in {"tool_call", "server_tool_call"}:
        stripped["args"] = {}
    elif btype == "non_standard":
        stripped["value"] = {}
    return _to_protocol_block(stripped)


def _should_emit_delta(block: CompatBlock) -> bool:
    """每块是否携带值得触发 delta 事件的内容。

    可增量更新类型仅在有新内容时才发出。自包含/已最终化类型则完全跳过 delta —— 其
    `finish` 事件承载它们。
    """
    btype = block.get("type")
    if btype == "text":
        return bool(block.get("text"))
    if btype == "reasoning":
        return bool(block.get("reasoning"))
    if btype in {"tool_call_chunk", "server_tool_call_chunk"}:
        return bool(
            block.get("args") or block.get("id") or block.get("name"),
        )
    if "data" in block:
        return bool(block.get("data"))
    return False


def _accumulate(state: CompatBlock | None, delta: CompatBlock) -> CompatBlock:
    """将每块的 delta 片段合并到按索引累积的状态中。

    Used only for the finalization pass — live delta events are emitted
    directly from the per-chunk block, without round-tripping through
    accumulated state.
    """
    if state is None:
        return dict(delta)
    btype = state.get("type")
    dtype = delta.get("type")
    if btype == "text" and dtype == "text":
        state["text"] = state.get("text", "") + delta.get("text", "")
                # `index` 在客户端侧被剥离
                # `tool_call` / `invalid_tool_call` 的最终化以匹配 v1
                # （`AIMessage.init_tool_calls` 重建 tool_call 块时不带 `index`），防止
        # `merge_lists` 将后续分块重新合并进已解析的参数字典。`server_tool_call`
        for key, value in delta.items():
            if key in {"type", "text"} or value is None:
                continue
            if key == "extras" and isinstance(value, dict):
                state["extras"] = {**(state.get("extras") or {}), **value}
            else:
                state[key] = value
    elif btype == "reasoning" and dtype == "reasoning":
        state["reasoning"] = state.get("reasoning", "") + delta.get("reasoning", "")
                # 保留 `index`，因为 v1 的 `init_server_tool_calls`
                # 就地最终化并保留了它。
                # 元数据、使用量、结束原因
                # 事件构建器
        for key, value in delta.items():
            if key in {"type", "reasoning"} or value is None:
                continue
            if key == "extras" and isinstance(value, dict):
                state["extras"] = {**(state.get("extras") or {}), **value}
            else:
                state[key] = value
    elif btype in {"tool_call_chunk", "server_tool_call_chunk"} and dtype == btype:
        state["args"] = (state.get("args", "") or "") + (delta.get("args") or "")
        if delta.get("id") is not None:
            state["id"] = delta["id"]
        if delta.get("name") is not None:
            state["name"] = delta["name"]
    elif btype == dtype and "data" in delta:
        state["data"] = (state.get("data", "") or "") + (delta.get("data") or "")
        for key, value in delta.items():
            if key in {"type", "data"} or value is None:
                continue
            if key == "extras" and isinstance(value, dict):
                state["extras"] = {**(state.get("extras") or {}), **value}
            else:
                state[key] = value
    else:
                # 协议 0.0.9 移除了 `MessageFinishData` 中顶层的 `reason` 字段；提供方的原始
        # `finish_reason` /
        state.clear()
        state.update(delta)
    return state


def finalize_tool_call_chunk(
    *,
    raw_args: str | None,
    id_: str | None,
    name: str | None,
    extras: dict[str, Any],
    finalized_type: str,
) -> FinalizedContentBlock:
    """将累积的工具块参数解析为最终块。

    Shared between the compat bridge's `_finalize_block` and the
    `ChatModelStream` end-of-stream sweep. Parses `raw_args` as JSON:
    on success builds the requested finalized type (`tool_call` or
    `server_tool_call`) with provider-specific fields (`extras`)
    preserved; on failure falls back to `invalid_tool_call` carrying
    the raw string so downstream consumers can still introspect the
    malformed payload.

    Args:
        raw_args: Accumulated partial-JSON string; `None` or empty
            treated as `{}`.
        id_: Tool-call id collected across chunks.
        name: Tool name collected across chunks.
        extras: Provider-specific fields to carry onto the finalized
            block. Callers are responsible for having already dropped
            keys they don't want propagated (notably `type`, `id`,
            `name`, `args`, and `index` on client-side `tool_call`).
        finalized_type: `"tool_call"` or `"server_tool_call"`.

    Returns:
        一个 `ToolCall`、`ServerToolCall` 或 `InvalidToolCall` ——
        当 `raw_args` 非空但无法解析为有效 JSON 时为 `InvalidToolCall`。
    """
    raw = raw_args or "{}"
    try:
        parsed = json.loads(raw) if raw else {}
    except (json.JSONDecodeError, TypeError):
        invalid = InvalidToolCall(
            type="invalid_tool_call",
            id=id_,
            name=name,
            args=raw,
            error="Failed to parse tool call arguments as JSON",
        )
        invalid.update(extras)  # type: ignore[typeddict-item]
        return invalid
    if finalized_type == "tool_call":
        finalized_tc = ToolCall(
            type="tool_call",
            id=id_ or "",
            name=name or "",
            args=parsed,
        )
        finalized_tc.update(extras)  # type: ignore[typeddict-item]
        return finalized_tc
    finalized_stc = ServerToolCall(
        type="server_tool_call",
        id=id_ or "",
        name=name or "",
        args=parsed,
    )
    finalized_stc.update(extras)  # type: ignore[typeddict-item]
    return finalized_stc


def _finalize_block(block: CompatBlock) -> FinalizedContentBlock:
    """将块变体提升为其最终形式。

    `tool_call_chunk` becomes `tool_call` — or `invalid_tool_call`
    if the accumulated `args` don't parse as JSON.
    `server_tool_call_chunk` becomes `server_tool_call` under the same
    rule.  Everything else passes through: text/reasoning blocks carry
    their accumulated snapshot, and self-contained types are already in
    their terminal shape.
    """
    btype = block.get("type")
    if btype in {"tool_call_chunk", "server_tool_call_chunk"}:
                # `stop_reason` 现在作为其他响应元数据的一部分，嵌入在 `metadata` 中。原样传递。
                # `additional_kwargs` 是消息结束事件上的非标准扩展（与 `metadata` 并行，而
        # `MessageFinishData` 本身也不包含 `metadata`）
                # 显式地。`index` 在客户端被移除
                # `tool_call` / `invalid_tool_call` 的最终化处理以匹配 v1
                # (`AIMessage.init_tool_calls` 重新构建 tool_call 块，不包含
                # index`），防止 `merge_lists` 进一步合并
                # 将已解析的参数字典拆分为多个块。`server_tool_call`
                # 保留 `index`，因为 v1 的 `init_server_tool_calls`
                # 就地完成并保留它。
        client_tool_call = btype == "tool_call_chunk"
        extras_drop = {"type", "id", "name", "args"}
        if client_tool_call:
            extras_drop |= {"index"}
        extras = {
            k: v for k, v in block.items() if k not in extras_drop and v is not None
        }
        return finalize_tool_call_chunk(
            raw_args=block.get("args"),
            id_=block.get("id"),
            name=block.get("name"),
            extras=extras,
            finalized_type="tool_call" if client_tool_call else "server_tool_call",
        )
    return _to_finalized_block(block)


# ---------------------------------------------------------------------------
# 元数据、使用情况、结束原因
# ---------------------------------------------------------------------------


def _extract_start_metadata(response_metadata: dict[str, Any]) -> MessageMetadata:
    """提取 `message-start` 事件的提供商/模型提示。"""
    metadata: MessageMetadata = {}
    if "model_provider" in response_metadata:
        metadata["provider"] = response_metadata["model_provider"]
    if "model_name" in response_metadata:
        metadata["model"] = response_metadata["model_name"]
    return metadata


def _accumulate_usage(current: UsageInfo | None, delta: UsageMetadata) -> UsageInfo:
    """汇总使用次数并合并各块的详细字典。

    `delta` 是块的 `usage_metadata`；`current` 是运行总计。
    双方均通过字面键读写，以确保端到端的类型结构保持不变——无需经过 `dict[str, Any]`
    的转换。
    """
    new: UsageInfo = current if current is not None else {}
    if "input_tokens" in delta:
        new["input_tokens"] = new.get("input_tokens", 0) + delta["input_tokens"]
    if "output_tokens" in delta:
        new["output_tokens"] = new.get("output_tokens", 0) + delta["output_tokens"]
    if "total_tokens" in delta:
        new["total_tokens"] = new.get("total_tokens", 0) + delta["total_tokens"]
    input_details = delta.get("input_token_details")
    if input_details:
        merged_input = new.get("input_token_details", {})
        merged_input.update(input_details)
        new["input_token_details"] = merged_input
    output_details = delta.get("output_token_details")
    if output_details:
        merged_output = new.get("output_token_details", {})
        merged_output.update(output_details)
        new["output_token_details"] = merged_output
    return new


def _isolate_usage(usage: UsageInfo | None) -> UsageInfo | None:
    """复制事件的使用信息，以便消费者无法修改源消息。

    回放路径（`message_to_events`）会使用实时 `msg.usage_metadata`，
    因此发出的事件必须不共享其字典：需复制顶层以及嵌套的 `input_token_details` /
    `output_token_details`
    以解除别名。在流式累加路径上，累加器已拥有其构建的字典，因此该复制操作是无害的空操作。
    """
    if not usage:
        return None
    result: UsageInfo = usage.copy()
    input_details = result.get("input_token_details")
    if input_details is not None:
        result["input_token_details"] = input_details.copy()
    output_details = result.get("output_token_details")
    if output_details is not None:
        result["output_token_details"] = output_details.copy()
    return result


# ---------------------------------------------------------------------------
# 事件构建器
# ---------------------------------------------------------------------------


def _build_message_start(
    msg: BaseMessage,
    message_id: str | None,
) -> MessageStartData:
    start_data = MessageStartData(event="message-start", role="ai", id="")
    resolved_id = message_id if message_id is not None else getattr(msg, "id", None)
    if resolved_id:
        start_data["id"] = resolved_id
    start_metadata = _extract_start_metadata(msg.response_metadata or {})
    if start_metadata:
        start_data["metadata"] = start_metadata
    return start_data


def _build_message_finish(
    *,
    usage: UsageInfo | None,
    response_metadata: dict[str, Any] | None,
    additional_kwargs: dict[str, Any] | None = None,
) -> MessageFinishData:
        # Protocol 0.0.9 从顶层移除了 `reason` 字段
        # `MessageFinishData`; 提供方的原始 `finish_reason` /
        # `stop_reason` 现在与其它信息一起位于 `metadata` 中
        # 响应元数据。原样传递。
    finish_data: dict[str, Any] = {"event": "message-finish"}
    usage_info = _isolate_usage(usage)
    if usage_info is not None:
        finish_data["usage"] = usage_info
    if response_metadata:
        finish_data["metadata"] = dict(response_metadata)
        # `additional_kwargs` 是消息结束部分的非标准扩展
        # 事件（与 `metadata` 并行，`MessageFinishData` 也不包含
        # 正式声明但由消费者读取）。它携带提供方侧
        # 不映射到类型化协议字段的 kwargs —— 尤其是 Gemini 的
        # `__gemini_function_call_thought_signatures__`，模型在后续轮次中需要此字段以重现之前的思考过程。若无此字段，
        # 流式组装的消息会静默丢失 `ainvoke` 保留的数据，破坏多轮流式流程。
        # 主生成器
        # v1 `stream()` 包装器在生成时将 `generation_info` 合并到
    if additional_kwargs:
        finish_data["additional_kwargs"] = dict(additional_kwargs)
    return cast("MessageFinishData", finish_data)


def _finalize_and_build_finish(
    wire_idx: int,
    block: CompatBlock,
) -> MessagesData:
    """完成一个块，并将其包装在 `content-block-finish` 事件中。"""
    return ContentBlockFinishData(
        event="content-block-finish",
        index=wire_idx,
        content=_finalize_block(block),
    )


# ---------------------------------------------------------------------------
# `response_metadata` 中后才产出（通过 `chat_models.py` 的
# ---------------------------------------------------------------------------


def chunks_to_events(
    chunks: Iterator[ChatGenerationChunk],
    *,
    message_id: str | None = None,
) -> Iterator[MessagesData]:
    """将 `ChatGenerationChunk` 的流转换为协议事件。

    块由源端标识符独立跟踪。例如 Anthropic
    等提供商可按索引交错并行的工具调用块，因此每个首次出现的块都会获得一个
    `content-block-start`，增量保持其稳定的网络索引，且所有打开的块都在消息结束时被最终化。
    源端标识符（来自块的 `index` 字段，可能是整数或字符串）会被转换为顺序的 `uint`
    网络索引。

    Args:
        chunks: 来自 `_stream()` 的 `ChatGenerationChunk` 迭代器。
        message_id: 可选的稳定消息 ID。

    Yields:
        `MessagesData` 生命周期事件。
    """
    started = False
    blocks: dict[Any, tuple[int, CompatBlock]] = {}
    next_wire_idx = 0
    usage: UsageInfo | None = None
    response_metadata: dict[str, Any] = {}
    additional_kwargs: dict[str, Any] = {}

    for chunk in chunks:
        msg = chunk.message
        if not isinstance(msg, AIMessageChunk):
            continue

                # `_gen_info_and_msg_metadata`）。我们绕过该包装器，直接读取
                # `_stream`，因此在此处以相同优先级重现合并：`generation_info` 优先，然后是
        # `message.response_metadata`
                # 覆盖。这就是当提供方通过 `generation_info` 而非消息的 `response_metadata`
        # 发出 `model_name`、
                # `system_fingerprint` 和 `finish_reason`
        # 等提供方字段时，它们如何抵达桥接层的原因。
                # 将块的 `additional_kwargs` 传递至组装后的消息。不映射到类型化
                # 协议块的提供方字段（例如 Gemini 的每个工具调用的思考签名）
                # 位于非流式 `ainvoke`
        # 结果中；在流式路径上丢弃这些字段会静默导致多轮行为差异。使用
                # `merge_dicts`，因为同一键可能在多个块中分段到达（例如累积的
        # `function_call`），与
                # `AIMessageChunk` 自身的合并方式保持一致。
        merged_rm: dict[str, Any] = {
            **(chunk.generation_info or {}),
            **(msg.response_metadata or {}),
        }
        if merged_rm:
            response_metadata.update(merged_rm)

                # 参见同步版本以了解原因：将 `generation_info` 合并到累积的
                # `response_metadata` 中，优先级与 v1 `stream()` 包装器相同。
                # 参见同步版本：将块的 `additional_kwargs` 传递下去，使提供方特定数据（例如
        # Gemini 的思考签名）能够到达
                # 组装后的消息，而非被丢弃。
                # 流式路径会静默地改变多轮对话行为。使用
                # merge_dicts 因为相同的键可能会分片到达
                # chunks（例如累积的 `function_call`），与之匹配
                # `AIMessageChunk` 会自动合并自身。
        if msg.additional_kwargs:
            additional_kwargs = merge_dicts(additional_kwargs, msg.additional_kwargs)

        if not started:
            started = True
            yield _build_message_start(msg, message_id)

        for key, block in _iter_protocol_blocks(msg):
            if key not in blocks:
                wire_idx = next_wire_idx
                next_wire_idx += 1
                blocks[key] = (wire_idx, dict(block))
                yield ContentBlockStartData(
                    event="content-block-start",
                    index=wire_idx,
                    content=_start_skeleton(block),
                )
            else:
                wire_idx, existing = blocks[key]
                blocks[key] = (wire_idx, _accumulate(existing, block))
            if _should_emit_delta(block):
                wire_idx, current = blocks[key]
                is_block_delta = block.get("type") in {
                    "tool_call_chunk",
                    "server_tool_call_chunk",
                }
                delta_source = current if is_block_delta else block
                yield ContentBlockDeltaData(
                    event="content-block-delta",
                    index=wire_idx,
                    delta=_to_content_delta(delta_source or block),
                )

        if msg.usage_metadata:
            usage = _accumulate_usage(usage, msg.usage_metadata)

    if not started:
        return

    for wire_idx, block in blocks.values():
        yield _finalize_and_build_finish(wire_idx, block)

    yield _build_message_finish(
        usage=usage,
        response_metadata=response_metadata,
        additional_kwargs=additional_kwargs,
    )


async def achunks_to_events(
    chunks: AsyncIterator[ChatGenerationChunk],
    *,
    message_id: str | None = None,
) -> AsyncIterator[MessagesData]:
    """`chunks_to_events` 的异步版本。"""
    started = False
    blocks: dict[Any, tuple[int, CompatBlock]] = {}
    next_wire_idx = 0
    usage: UsageInfo | None = None
    response_metadata: dict[str, Any] = {}
    additional_kwargs: dict[str, Any] = {}

    async for chunk in chunks:
        msg = chunk.message
        if not isinstance(msg, AIMessageChunk):
            continue

                # 参见同步孪生以了解原因：将 `generation_info` 合并到
                # 与相同优先级的 `response_metadata` 一起累积
                # v1 `stream()` 包装器。
        merged_rm: dict[str, Any] = {
            **(chunk.generation_info or {}),
            **(msg.response_metadata or {}),
        }
        if merged_rm:
            response_metadata.update(merged_rm)

                # 参见同步孪生：将 chunk 的 `additional_kwargs` 传递下去
                # 特定于提供方的数据（例如 Gemini 思考签名）会到达
                # 组装的消息而不是被丢弃。
        if msg.additional_kwargs:
            additional_kwargs = merge_dicts(additional_kwargs, msg.additional_kwargs)

        if not started:
            started = True
            yield _build_message_start(msg, message_id)

        for key, block in _iter_protocol_blocks(msg):
            if key not in blocks:
                wire_idx = next_wire_idx
                next_wire_idx += 1
                blocks[key] = (wire_idx, dict(block))
                yield ContentBlockStartData(
                    event="content-block-start",
                    index=wire_idx,
                    content=_start_skeleton(block),
                )
            else:
                wire_idx, existing = blocks[key]
                blocks[key] = (wire_idx, _accumulate(existing, block))
            if _should_emit_delta(block):
                wire_idx, current = blocks[key]
                is_block_delta = block.get("type") in {
                    "tool_call_chunk",
                    "server_tool_call_chunk",
                }
                delta_source = current if is_block_delta else block
                yield ContentBlockDeltaData(
                    event="content-block-delta",
                    index=wire_idx,
                    delta=_to_content_delta(delta_source or block),
                )

        if msg.usage_metadata:
            usage = _accumulate_usage(usage, msg.usage_metadata)

    if not started:
        return

    for wire_idx, block in blocks.values():
        yield _finalize_and_build_finish(wire_idx, block)

    yield _build_message_finish(
        usage=usage,
        response_metadata=response_metadata,
        additional_kwargs=additional_kwargs,
    )


def message_to_events(
    msg: BaseMessage,
    *,
    message_id: str | None = None,
) -> Iterator[MessagesData]:
    """回放一个已最终化的消息，作为合成事件生命周期。

    对于完整返回的消息（来自图节点、检查点或缓存），生成与实时调用相同的
    `message-start` / 每块 / `message-finish`
    事件流。下游消费者无论来源如何，都能看到统一的事件形状。

    文本和推理块会发出一个单一的
    `content-block-delta`，其中包含累积的全部内容。已最终确定的块（如
    tool_call、server_tool_call、image 等）跳过 delta，仅依赖 `content-block-finish`
    事件。

    Args:
        msg: 已最终确定的消息 —— 通常是 `AIMessage`。
        message_id: 可选的稳定消息 ID；若未提供则回退到 `msg.id`。

    Yields:
        `MessagesData` 生命周期事件。
    """
    response_metadata = msg.response_metadata or {}
    yield _build_message_start(msg, message_id)

    for wire_idx, (_key, block) in enumerate(_iter_protocol_blocks(msg)):
        yield ContentBlockStartData(
            event="content-block-start",
            index=wire_idx,
            content=_start_skeleton(block),
        )
        if _should_emit_delta(block):
            yield ContentBlockDeltaData(
                event="content-block-delta",
                index=wire_idx,
                delta=_to_content_delta(block),
            )
        yield ContentBlockFinishData(
            event="content-block-finish",
            index=wire_idx,
            content=_finalize_block(block),
        )

    yield _build_message_finish(
        usage=getattr(msg, "usage_metadata", None),
        response_metadata=response_metadata,
    )


async def amessage_to_events(
    msg: BaseMessage,
    *,
    message_id: str | None = None,
) -> AsyncIterator[MessagesData]:
    """异步版本的 `message_to_events`。"""
    for event in message_to_events(msg, message_id=message_id):
        yield event


__all__ = [
    "CompatBlock",
    "achunks_to_events",
    "amessage_to_events",
    "chunks_to_events",
    "finalize_tool_call_chunk",
    "message_to_events",
]

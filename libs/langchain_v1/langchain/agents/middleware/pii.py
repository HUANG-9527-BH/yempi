"""智能体的个人身份信息（PII）检测与处理中间件。"""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, Any, ClassVar, Literal

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langgraph.stream import StreamTransformer
from typing_extensions import override

from langchain.agents.middleware._redaction import (
    PIIDetectionError,
    PIIMatch,
    RedactionRule,
    ResolvedRedactionRule,
    apply_strategy,
    detect_credit_card,
    detect_email,
    detect_ip,
    detect_mac_address,
    detect_url,
)
from langchain.agents.middleware.types import (
    AgentMiddleware,
    AgentState,
    ContextT,
    ResponseT,
    hook_config,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from langgraph.runtime import Runtime
    from langgraph.stream._types import ProtocolEvent


_DEFAULT_STREAM_LOOKBACK = 128
"""Default trailing-buffer size for cross-delta PII detection.

The transformer always holds the last `lookback` characters in a per-content
block buffer so that PII patterns straddling delta boundaries are detected
before any text is released downstream. 128 comfortably covers the built-in
detectors (the credit-card regex tops out at 19 characters; URLs and emails
are typically well under 100) while bounding first-token latency.
"""


class _PIIStreamTransformer(StreamTransformer):
    """在进行中的 `messages` 事件中，修改 `content-block-delta` 的文本。

    在内置流式转换器之前运行，确保被脱敏的文本是下游所有消费者看到的内容——
    包括主协议事件日志和 `MessagesTransformer` 将其快照到的 `run.messages` 投影。

    Holds a sliding buffer of the most recent text per (run_id, content
    block index) so PII patterns that straddle delta boundaries are caught.
    Anything older than `lookback` characters is redacted with the resolved
    rule's strategy and emitted as the new delta text; the trailing tail
    stays in the buffer until a later delta extends it past the cap or the
    block's finish event flushes the snapshot.
    """

    before_builtins: ClassVar[bool] = True
    required_stream_modes: ClassVar[tuple[str, ...]] = ("messages", "tools", "values")

    def __init__(
        self,
        scope: tuple[str, ...] = (),
        *,
        rule: ResolvedRedactionRule,
        lookback: int = _DEFAULT_STREAM_LOOKBACK,
    ) -> None:
        super().__init__(scope)
        self._rule = rule
        self._lookback = lookback
                # 以 `(run_id, content_block_index)` 为键的文本/推理增量。
        self._buffers: dict[tuple[str, int], str] = {}
                # 以 `tool_call_id` 为键的工具输出增量缓冲区。存储在一个
                # 独立字典中，以防止消息通道上的 `_drop_run` 操作
                # 清除活跃的工具输出状态。
        self._tool_buffers: dict[str, str] = {}

    def init(self) -> dict[str, Any]:
                # 无投影——此转换器直接修改事件，而非构建派生视图。
                # 旧版 `(BaseMessage, metadata)` 结构：当模型仅实现 `_generate`
        return {}

    def process(self, event: ProtocolEvent) -> bool:
        method = event["method"]
        if method == "messages":
            return self._process_messages_event(event)
        if method == "tools":
            return self._process_tools_event(event)
        if method == "values":
            return self._process_values_event(event)
        return True

    def _process_values_event(self, event: ProtocolEvent) -> bool:
        """在 `values` 通道上脱敏状态快照。

        State snapshots emitted between nodes carry the full state dict,
        which typically includes the messages list. Walking the snapshot
        with `_redact_value` returns a fresh structure where every
        message has a redacted copy of its content — the original
        objects in graph state remain intact for the state-level
        enforcer (`apply_to_tool_results` via `before_model`) to act on
        independently when the agent loops back.
        """
        data = event["params"].get("data")
        if data is None:
            return True
        event["params"]["data"] = self._redact_value(data)
        return True

    def _process_messages_event(self, event: ProtocolEvent) -> bool:
        params = event["params"]
        data = params.get("data")
        if not isinstance(data, tuple) or len(data) != 2:  # noqa: PLR2004
            return True
        payload, metadata = data

                # （或其 `_astream` 回退）时，LangGraph→LangChain
                # 集成会发出此格式，产生单个事件
                # 携带完整消息，而非流式内容块
                # 携带完整消息，而非流式内容块
                # deltas。替换为脱敏副本，使消费者看到的是已清理的内容
                # 而原始内容在图状态中保持完整
                # 以便 `after_model` 独立处理。在 `block` 模式下
                # `_redact_base_message` 通过 `apply_strategy` 抛出 `PIIDetectionError`
                # 在我们到达此处之前。
        if isinstance(payload, BaseMessage):
            redacted = self._redact_base_message(payload)
            if redacted is not payload:
                params["data"] = (redacted, metadata)
            return True

        if not isinstance(payload, dict):
            return True
        kind = payload.get("event")
        run_id = str(metadata.get("run_id") or "") if metadata else ""

        if kind == "content-block-delta":
            self._mutate_delta(payload, run_id)
        elif kind == "content-block-finish":
            self._finalize_block(payload, run_id)
        elif kind in {"message-finish", "error"}:
            self._drop_run(run_id)
        return True

    def _process_tools_event(self, event: ProtocolEvent) -> bool:
        data = event["params"].get("data")
        if not isinstance(data, dict):
            return True
        kind = data.get("event")
        tool_call_id = data.get("tool_call_id")

        if kind == "tool-started":
                        # 工具输入可以是字典（多参数工具）、字符串
                        # （单参数工具 — `BaseTool._parse_input` 将原始字符串直接传递）
                        # 或列表（数组输入工具）。
                        # `_redact_value` 统一处理这三种情况。
            if "input" in data:
                data["input"] = self._redact_value(data["input"])
        elif kind == "tool-output-delta":
                        # 当存在 tool_call_id 时，使用其作为缓冲区键；退而求其次
                        # 对罕见的格式错误/自定义发射器情况使用 None 键槽
                        # （缓冲区变为共享，但至少脱敏仍会执行）。
            self._mutate_tool_output_delta(
                data, tool_call_id if isinstance(tool_call_id, str) else ""
            )
        elif kind == "tool-finished":
            if "output" in data:
                data["output"] = self._redact_value(data["output"])
            if isinstance(tool_call_id, str):
                self._tool_buffers.pop(tool_call_id, None)
        elif kind == "tool-error":
            msg = data.get("message")
            if isinstance(msg, str) and msg:
                matches = self._rule.detector(msg)
                if matches:
                    data["message"] = apply_strategy(msg, matches, self._rule.strategy)
            if isinstance(tool_call_id, str):
                self._tool_buffers.pop(tool_call_id, None)

        return True

    def _mutate_tool_output_delta(self, data: dict[str, Any], tool_call_id: str) -> None:
        """脱敏 `tool-output-delta` 负载。

        字符串 delta 会经过与 text-deltas 相同的 lookback 机制， 以 `tool_call_id`
        为键存储在独立的 `_tool_buffers` 字典中， 从而防止 `messages` 通道上的
        `_drop_run` 清除活跃的工具输出状态。

        结构化增量（字典/列表）会递归遍历，而无需缓冲——它们在增量之间没有位置稳定的形状，因此无法进行缓冲。
        """
        delta = data.get("delta")
        if isinstance(delta, str):
            held = self._tool_buffers.get(tool_call_id, "")
            combined = held + delta

            matches = self._rule.detector(combined)
            if matches:
                                # `apply_strategy` 在 `strategy="block"` 下抛出 `PIIDetectionError`
                                # 立即终止运行 — 比延迟到 `after_model` 再抛出更清晰
                                # 比等到 `after_model` 再统一处理 delta 更简洁
                                # `apply_strategy` 在 `block` 模式下抛出 `PIIDetectionError`
                combined = apply_strategy(combined, matches, self._rule.strategy)

            emit_end = max(0, len(combined) - self._lookback)
            self._tool_buffers[tool_call_id] = combined[emit_end:]
            data["delta"] = combined[:emit_end]
        elif isinstance(delta, (dict, list)):
            data["delta"] = self._redact_value(delta)

    def _redact_tool_call_list(self, calls: list[Any] | None) -> tuple[list[Any], bool]:
        """遍历工具调用（或无效工具调用）字典的列表。

        Returns `(new_list, changed)`. Each element's `args` is run
        through `_redact_value` regardless of its type — `tool_call.args`
        is a dict, `invalid_tool_call.args` is a raw JSON string, and
        `_redact_value` handles both shapes uniformly. If nothing
        changed, returns the input list and `changed=False`.
        """
        if not calls:
            return calls or [], False
        new_calls: list[Any] = []
        changed = False
        for tc in calls:
            if isinstance(tc, dict) and "args" in tc and tc["args"] is not None:
                redacted = self._redact_value(tc["args"])
                if redacted != tc["args"]:
                    new_tc = dict(tc)
                    new_tc["args"] = redacted
                    new_calls.append(new_tc)
                    changed = True
                    continue
            new_calls.append(tc)
        return new_calls, changed

    def _redact_value(self, value: Any) -> Any:
        """递归地在嵌套结构的字符串叶节点中清除敏感个人信息（PII）。

        返回一个新值，其中包含 PII 的每个字符串叶节点都被替换（或在 `block`
        模式下清空）。非字符串叶节点和结构本身保持不变。

        `BaseMessage` payloads (typically `ToolMessage` from
        `tool-finished.output`, or any message reached via the `values`
        channel) return a fresh copy with `.content` redacted plus
        `AIMessage.tool_calls[*].args` / `invalid_tool_calls[*].args`
        walked. The original object stays intact for state-level
        enforcers (`after_model`, `before_model` with
        `apply_to_tool_results`) to act on independently.

        作用域与预流式状态级表面一致：`.content`（字符串或内容块列表）和 `tool_calls`
        参数。其他消息属性（`additional_kwargs`、`response_metadata`、`ToolMessage.artifact`）在此处有意不被遍历——现有钩子不会在图状态中清理这些内容，因此在传输层清理它们会导致传输层与状态之间的不一致。
        """
        if isinstance(value, str):
            if not value:
                return value
            matches = self._rule.detector(value)
            if not matches:
                return value
                        # —— 运行立即失败，而不是缓冲直到状态级钩子抛出异常。
                        # 结构化内容块的形状：
                        # `[{"type": "text", "text": "..."}, {"type": "tool_call", ...}, ...]`。
            return apply_strategy(value, matches, self._rule.strategy)
        if isinstance(value, BaseMessage):
            return self._redact_base_message(value)
        if isinstance(value, dict):
            return {k: self._redact_value(v) for k, v in value.items()}
        if isinstance(value, list):
            return [self._redact_value(v) for v in value]
        if isinstance(value, tuple):
            return tuple(self._redact_value(v) for v in value)
        return value

    def _redact_base_message(self, value: BaseMessage) -> BaseMessage:
        """返回 `value` 的一个全新副本，其中携带敏感信息的表面已被清除。"""
        update: dict[str, Any] = {}

        content = value.content
        if isinstance(content, str) and content:
            matches = self._rule.detector(content)
            if matches:
                update["content"] = apply_strategy(content, matches, self._rule.strategy)
        elif isinstance(content, list) and content:
                        # `AIMessage.tool_calls` 和 `.invalid_tool_calls` 在 `args` 中携带 PII
                        # 独立于 `.content`。`tool_call.args` 是一个字典；
            redacted_content = self._redact_value(content)
            if redacted_content != content:
                update["content"] = redacted_content

                # `invalid_tool_call.args` 是原始 JSON 字符串 —
                # `_redact_value` 通过递归处理这两种结构。
                # 推理内容（来自扩展思维模型的思维链）是真正的 PII 表面 —
                # 模型会回显用户提供的数据或从上下文中合成数据。对
        if isinstance(value, AIMessage):
            new_tc_list, tc_changed = self._redact_tool_call_list(value.tool_calls)
            if tc_changed:
                update["tool_calls"] = new_tc_list
            new_inv_list, inv_changed = self._redact_tool_call_list(value.invalid_tool_calls)
            if inv_changed:
                update["invalid_tool_calls"] = new_inv_list

        if not update:
            return value
        return value.model_copy(update=update)

    def _mutate_delta(self, payload: dict[str, Any], run_id: str) -> None:
        delta = payload.get("delta")
        if not isinstance(delta, dict):
            return
        delta_type = delta.get("type")
        if delta_type == "text-delta":
            self._mutate_string_field_delta(delta, payload, run_id, "text")
            return
        if delta_type == "reasoning-delta":
                        # `reasoning` 字段运行相同的回溯机制，如同对文本 delta 所做。
                        # 块索引在消息内是唯一的，无论块类型如何，因此缓冲区键
                        # `(run_id, index)` 自然与文本 delta 键不重叠。
                        # 其他 delta 类型（`data-delta`、厂商特定块类型）直接通过。
                        # 预流式中间件仅对状态消息的 `.content` 文本进行清理
                        # ；二进制负载和提供方特定的块结构不在该表面的对齐范围内。
                        # 在拆分前对完整累积缓冲区运行检测。
            self._mutate_string_field_delta(delta, payload, run_id, "reasoning")
            return
        if delta_type == "block-delta":
            fields = delta.get("fields")
            if isinstance(fields, dict) and fields.get("type") in {
                "tool_call_chunk",
                "server_tool_call_chunk",
            }:
                self._mutate_tool_call_chunk_delta(fields)
                # 仅对即将发出的前缀进行检测会遗漏跨越回溯边界的匹配项 —
                # 检测器的正则表达式需要完整的、边界锚定的命中，因此截断的前缀
                # 无法匹配，部分 PII 将泄漏在传输线上。在 `strategy="block"`
        # 下，`apply_strategy` 在此处抛出
                # `PIIDetectionError`，一旦 PII 出现就立即失败运行，而非缓冲到 `after_model`
        # 才处理。

    def _mutate_string_field_delta(
        self,
        delta: dict[str, Any],
        payload: dict[str, Any],
        run_id: str,
        field: str,
    ) -> None:
        """对增量中的字符串字段应用回溯缓冲清除。

        Shared by `text-delta` (`field="text"`) and `reasoning-delta`
        (`field="reasoning"`). Buffer is keyed by `(run_id, block_index)`;
        block indices are unique within a message so different block
        types share the same key space without collision.
        """
        text = delta.get(field)
        if not isinstance(text, str) or not text:
            return
        index = payload.get("index")
        if not isinstance(index, int):
            return

        key = (run_id, index)
        held = self._buffers.get(key, "")
        combined = held + text

                # `apply_strategy` 在 `strategy="block"` 下抛出 `PIIDetectionError` —
                # 运行在累积 args 字符串中出现完整 PII 模式时立即失败。
                # 最终化的块携带模型原始的 delta 连接结果，而非我们在传输线上发出的内容。重新对
                # 其全文运行检测，以确保快照与脱敏流一致。
                # `tool_call` / `server_tool_call` args 是字典；
                # `invalid_tool_call.args` 是原始未解析的 JSON 字符串。
                # `_redact_value` 处理这两种结构。
                # 释放此 run_id 的任何缓冲尾部 — content-block-finish
        # 应已在正常完成时完成此操作，但 message-finish
        matches = self._rule.detector(combined)
        if matches:
            combined = apply_strategy(combined, matches, self._rule.strategy)

        emit_end = max(0, len(combined) - self._lookback)
        self._buffers[key] = combined[emit_end:]
        delta[field] = combined[:emit_end]

    def _mutate_tool_call_chunk_delta(self, fields: dict[str, Any]) -> None:
        """使用回溯保留机制清除累积的工具调用参数。

        Each `tool_call_chunk` `block-delta` event carries the full
        accumulated args string (verified against `_compat_bridge.py`
        — `delta_source = current` for these block types — and against
        the consumer-side `_merge_block_delta_into_store`, which
        replaces wholesale rather than appends).

        Detection runs on the full cumulative args so any complete PII
        anywhere in the string is redacted before emission. Lookback
        withholding then trims the trailing the lookback window characters
        from what reaches the consumer — those characters might be the
        start of a partial PII match that completes in a future
        cumulative delta. The trimmed tail surfaces at `content-block-
        finish` where `_finalize_block` redacts the parsed args dict.

        For args that fit within the lookback window (the typical case),
        this withholds the entire args string during streaming — the
        redacted args dict appears only at finalize. For args that
        exceed the lookback window, the safe prefix streams incrementally
        as the cumulative state grows. PII that appears more than
        the lookback window characters from the cumulative tail in a
        delta where it hasn't yet completed can still surface in the
        emit prefix — same residual exposure as PII longer than
        the lookback window on the text path. The `content-block-finish`
        snapshot redaction is the backstop.
        """
        args = fields.get("args")
        if not isinstance(args, str) or not args:
            return

        matches = self._rule.detector(args)
        if matches:
                        # apply_strategy 在
                        # strategy="block" — 一旦出现完整
                        # PII 模式在累积参数字符串中暴露。
            args = apply_strategy(args, matches, self._rule.strategy)

        emit_end = max(0, len(args) - self._lookback)
        fields["args"] = args[:emit_end]

    def _finalize_block(self, payload: dict[str, Any], run_id: str) -> None:
        index = payload.get("index")
        if not isinstance(index, int):
            return
        key = (run_id, index)
                # 最终的 block 携带模型的原始拼接
                # 的 delta，而非我们在网络上发出的内容。重新运行检测
                # 其完整文本，因此快照与脱敏流匹配。
        content = payload.get("content")
        if isinstance(content, dict):
            ctype = content.get("type")
            if ctype == "text":
                self._finalize_string_field(content, "text")
            elif ctype == "reasoning":
                self._finalize_string_field(content, "reasoning")
            elif (
                ctype in {"tool_call", "server_tool_call", "invalid_tool_call"}
                and "args" in content
                and content["args"] is not None
            ):
                                # `tool_call` / `server_tool_call` 参数是字典；
                                # invalid_tool_call.args 是原始未解析的 JSON
                                # 字符串。`_redact_value` 同时处理两种形状。
                content["args"] = self._redact_value(content["args"])
        self._buffers.pop(key, None)

    def _finalize_string_field(self, content: dict[str, Any], field: str) -> None:
        """在 `content-block-finish` 上重新对字符串内容块字段进行脱敏。

        Used for `text` and `reasoning` content blocks. Under
        `strategy="block"` `apply_strategy` raises `PIIDetectionError`,
        failing the run immediately.
        """
        text = content.get(field)
        if not isinstance(text, str) or not text:
            return
        matches = self._rule.detector(text)
        if not matches:
            return
        content[field] = apply_strategy(text, matches, self._rule.strategy)

    def _drop_run(self, run_id: str) -> None:
                # 释放此 run_id 的任何缓冲尾部内容 — content-block-finish
                # 应在正常完成时已完成，但消息结束
                # 错误路径需要显式扫描，以避免在长期运行的进程中积压废弃块。
                # 从类型角度而言，字面量已被 'str' 覆盖。
        stale = [key for key in self._buffers if key[0] == run_id]
        for key in stale:
            del self._buffers[key]

    def finalize(self) -> None:
        self._buffers.clear()
        self._tool_buffers.clear()

    def fail(self, err: BaseException) -> None:  # noqa: ARG002
        self._buffers.clear()
        self._tool_buffers.clear()


class PIIMiddleware(AgentMiddleware[AgentState[ResponseT], ContextT, ResponseT]):
    """检测并处理对话中的个人身份信息（PII）。

    此中间件可检测常见 PII 类型，并应用可配置的策略进行处理。
    它可在用户输入和智能体输出中检测电子邮件、信用卡号、IP 地址、MAC 地址以及 URL。

    内置 PII 类型：

    - `email`: 电子邮件地址
    - `credit_card`: 信用卡号码（通过 Luhn 算法验证）
    - `ip`: IP 地址（通过标准库验证）
    - `mac_address`: MAC 地址
    - `url`: URL（包括 `http`/`https` 和裸 URL）

    策略：

    - `block`: 检测到 PII 时抛出异常
    - `redact`: 用 `[REDACTED_TYPE]` 占位符替换 PII
    - `mask`: 部分遮蔽 PII（例如信用卡号显示为 `****-****-****-1234`）
    - `hash`: 用确定性哈希替换 PII（例如 `<email_hash:a1b2c3d4>`）

    策略选择指南：

    | 策略 | 是否保留身份？ | 适用场景                                |
    | -------- | ------------------- | --------------------------------------- |
    | `block`  | N/A                 | 完全避免 PII                    |
    | `redact` | 否                  | 通用合规性，日志净化    |
    | `mask`   | 否                  | 人类可读性，客户服务界面 |
    | `hash`   | 是（伪匿名）  | 分析、调试                    |

    Example:
        ```python
        from langchain.agents.middleware import PIIMiddleware
        from langchain.agents import create_agent

        # Redact all emails in user input
        agent = create_agent(
            "openai:gpt-5.5",
            middleware=[
                PIIMiddleware("email", strategy="redact"),
            ],
        )

        # Use different strategies for different PII types
        agent = create_agent(
            "openai:gpt-5.5",
            middleware=[
                PIIMiddleware("credit_card", strategy="mask"),
                PIIMiddleware("url", strategy="redact"),
                PIIMiddleware("ip", strategy="hash"),
            ],
        )

        # Custom PII type with regex
        agent = create_agent(
            "openai:gpt-5.5",
            middleware=[
                PIIMiddleware("api_key", detector=r"sk-[a-zA-Z0-9]{32}", strategy="block"),
            ],
        )
        ```
    """

    def __init__(
        self,
                # 尽管如此，我们仍绕过 PYI051 以保持调用方的类型提示和自动补全功能。
                # 流式转换器会清理流式输出表面中的相同消息，这些消息也会被状态级钩子在图状态中清理。
        pii_type: Literal["email", "credit_card", "ip", "mac_address", "url"] | str,  # noqa: PYI051
        *,
        strategy: Literal["block", "redact", "mask", "hash"] = "redact",
        detector: Callable[[str], list[PIIMatch]] | str | None = None,
        apply_to_input: bool = True,
        apply_to_output: bool = False,
        apply_to_tool_results: bool = False,
    ) -> None:
        """初始化 PII 检测中间件。

        Args:
            pii_type: 要检测的 PII 类型。

                可以是内置类型（`email`, `credit_card`, `ip`, `mac_address`,
                `url`）或自定义类型名称。
            strategy: 如何处理检测到的 PII。

                选项：

                * `block`: 检测到 PII 时抛出 `PIIDetectionError`
                * `redact`: 替换为 `[REDACTED_TYPE]` 占位符
                * `mask`: 部分遮蔽 PII（显示最后几个字符）
                * `hash`: 替换为确定性哈希（格式：`<type_hash:digest>`）

            detector: 自定义检测器函数或正则表达式模式。

                * If `Callable`: Function that takes content string and returns
                    list of `PIIMatch` objects
                * If `str`: Regex pattern to match PII
                * If `None`: Uses built-in detector for the `pii_type`
            apply_to_input: Whether to check user messages before model call.
            apply_to_output: Whether to check AI messages after model call.

                当为 `True` 时，还会安装一个流式转换器，
                使得智能体运行的每个数据通路都可在传输过程中被脱敏：

                * Streamed AI text deltas (`content-block-delta` of type
                  `text-delta`)
                * Streamed tool-call arguments (`content-block-delta`
                  with `tool_call_chunk` / `server_tool_call_chunk`
                  fields, plus the finalized `tool_call` content block
                  on `content-block-finish`)
                * Tool execution events on the `tools` channel
                  (`tool-started.input`, `tool-output-delta`,
                  `tool-finished.output`, `tool-error.message`)
                * State snapshots on the `values` channel — message
                  lists are walked and each message's `.content` is
                  redacted on a fresh copy (state itself stays intact
                  for `before_model` / `after_model` to act on
                  independently)

                State-level redaction via `after_model` (and
                `before_model` with `apply_to_tool_results`) remains the
                canonical enforcer; the streaming transformer ensures
                consumers reading `astream_events(version="v3")` or
                `run.messages` / `run.tool_calls` / `run.values` never
                see PII on the wire.
            apply_to_tool_results: Whether to check tool result messages after tool execution.

        Raises:
            ValueError: 如果 `pii_type` 不是内置类型且未提供检测器。
        """
        super().__init__()

        self.apply_to_input = apply_to_input
        self.apply_to_output = apply_to_output
        self.apply_to_tool_results = apply_to_tool_results

        self._resolved_rule: ResolvedRedactionRule = RedactionRule(
            pii_type=pii_type,
            strategy=strategy,
            detector=detector,
        ).resolve()
        self.pii_type = self._resolved_rule.pii_type
        self.strategy = self._resolved_rule.strategy
        self.detector = self._resolved_rule.detector

                # 只要启用任何输出端清理，就会安装该转换器 —
                # `apply_to_output` 覆盖 AI 消息（文本、工具调用参数、推理内容），
                # `apply_to_tool_results` 覆盖工具执行结果（`tools` 通道以及 `values` 和
                # `messages` 上的 `ToolMessage` 内容）。对于 `block`，该转换器会在检测到完整
        # PII 模式时，直接从其事件处理器抛出
                # `PIIDetectionError`，并通过 langgraph 的 `StreamMux.afail`
        # 路径使运行失败。状态级的
                # `after_model` / `before_model` 钩子仍作为非流式消费者的后备机制。
                # 如果启用，检查用户输入
                # 获取最后一条用户消息
                # 如果启用，检查工具结果
                # 找到最后一条 AIMessage，然后处理其后的所有 `ToolMessage` 对象
                # 获取最后一条 AI 消息之后的所有工具消息
                # 获取最后一条 AI 消息
        if self.apply_to_output or self.apply_to_tool_results:
            self.transformers = (
                partial(
                    _PIIStreamTransformer,
                    rule=self._resolved_rule,
                ),
            )

    @property
    def name(self) -> str:
        """中间件的名称。"""
        return f"{self.__class__.__name__}[{self.pii_type}]"

    def _process_content(
        self, content: str | list[str | dict[str, Any]]
    ) -> tuple[str | list[str | dict[str, Any]], bool]:
        """应用配置的脱敏规则，同时保留 `content` 的结构。

        仅对字符串叶子节点进行脱敏：直接的纯字符串 `content`，以及块列表型 `content`
        中每个块的 `text`。非文本块保持不变，因此块列表内容不会坍缩为列表的 `repr`。

        Args:
            content: 需要脱敏的消息内容。

        Returns:
            脱敏后的内容及是否进行了重写。
        """
        if isinstance(content, str):
            matches = self.detector(content)
            if not matches:
                return content, False
            return apply_strategy(content, matches, self.strategy), True

        new_blocks: list[str | dict[str, Any]] = []
        changed = False
        for block in content:
            new_block: str | dict[str, Any] = block
            text = block if isinstance(block, str) else block.get("text")
            if isinstance(text, str) and (matches := self.detector(text)):
                redacted = apply_strategy(text, matches, self.strategy)
                new_block = redacted if isinstance(block, str) else {**block, "text": redacted}
                changed = True
            new_blocks.append(new_block)
        return (new_blocks, True) if changed else (content, False)

    @hook_config(can_jump_to=["end"])
    @override
    def before_model(
        self,
        state: AgentState[Any],
        runtime: Runtime[ContextT],
    ) -> dict[str, Any] | None:
        """在模型调用前检查用户消息和工具结果中的 PII。

        Args:
            state: 当前智能体状态。
            runtime: LangGraph 运行时。

        Returns:
            Updated state with PII handled according to strategy, or `None` if no PII
                detected.

        Raises:
            PIIDetectionError: 若检测到 PII 且策略为 `'block'`。
        """
        if not self.apply_to_input and not self.apply_to_tool_results:
            return None

        messages = state["messages"]
        if not messages:
            return None

        new_messages = list(messages)
        any_modified = False

                # 检查用户输入（如果已启用）
        if self.apply_to_input:
                        # 获取最后一条用户消息
            last_user_msg = None
            last_user_idx = None
            for i in range(len(messages) - 1, -1, -1):
                if isinstance(messages[i], HumanMessage):
                    last_user_msg = messages[i]
                    last_user_idx = i
                    break

            if last_user_idx is not None and last_user_msg and last_user_msg.content:
                new_content, changed = self._process_content(last_user_msg.content)

                if changed:
                    new_messages[last_user_idx] = last_user_msg.model_copy(
                        update={"content": new_content}
                    )
                    any_modified = True

                # 检查工具结果（如果已启用）
        if self.apply_to_tool_results:
                        # 找到最后一个 AIMessage，然后处理其后的所有 `ToolMessage` 对象
            last_ai_idx = None
            for i in range(len(messages) - 1, -1, -1):
                if isinstance(messages[i], AIMessage):
                    last_ai_idx = i
                    break

            if last_ai_idx is not None:
                                # 获取最后一条 AI 消息之后的所有工具消息
                for i in range(last_ai_idx + 1, len(messages)):
                    msg = messages[i]
                    if isinstance(msg, ToolMessage):
                        tool_msg = msg
                        if not tool_msg.content:
                            continue

                        new_content, changed = self._process_content(tool_msg.content)

                        if not changed:
                            continue

                        new_messages[i] = tool_msg.model_copy(update={"content": new_content})
                        any_modified = True

        if any_modified:
            return {"messages": new_messages}

        return None

    @hook_config(can_jump_to=["end"])
    async def abefore_model(
        self,
        state: AgentState[Any],
        runtime: Runtime[ContextT],
    ) -> dict[str, Any] | None:
        """异步检查用户消息和工具结果中的 PII，发生在模型调用前。

        Args:
            state: 当前智能体状态。
            runtime: LangGraph 运行时。

        Returns:
            Updated state with PII handled according to strategy, or `None` if no PII
                detected.

        Raises:
            PIIDetectionError: 若检测到 PII 且策略为 `'block'`。
        """
        return self.before_model(state, runtime)

    @override
    def after_model(
        self,
        state: AgentState[Any],
        runtime: Runtime[ContextT],
    ) -> dict[str, Any] | None:
        """在模型调用后检查 AI 消息中的 PII。

        Args:
            state: 当前智能体状态。
            runtime: LangGraph 运行时。

        Returns:
            Updated state with PII handled according to strategy, or None if no PII
                detected.

        Raises:
            PIIDetectionError: 若检测到 PII 且策略为 `'block'`。
        """
        if not self.apply_to_output:
            return None

        messages = state["messages"]
        if not messages:
            return None

                # 获取最后一条 AI 消息
        last_ai_msg = None
        last_ai_idx = None
        for i in range(len(messages) - 1, -1, -1):
            msg = messages[i]
            if isinstance(msg, AIMessage):
                last_ai_msg = msg
                last_ai_idx = i
                break

        if last_ai_idx is None or not last_ai_msg or not last_ai_msg.content:
            return None

        new_content, changed = self._process_content(last_ai_msg.content)

        if not changed:
            return None

        new_messages = list(messages)
        new_messages[last_ai_idx] = last_ai_msg.model_copy(update={"content": new_content})

        return {"messages": new_messages}

    async def aafter_model(
        self,
        state: AgentState[Any],
        runtime: Runtime[ContextT],
    ) -> dict[str, Any] | None:
        """异步检查模型调用后 AI 消息中的 PII。

        Args:
            state: 当前智能体状态。
            runtime: LangGraph 运行时。

        Returns:
            Updated state with PII handled according to strategy, or None if no PII
                detected.

        Raises:
            PIIDetectionError: 若检测到 PII 且策略为 `'block'`。
        """
        return self.after_model(state, runtime)


__all__ = [
    "PIIDetectionError",
    "PIIMatch",
    "PIIMiddleware",
    "detect_credit_card",
    "detect_email",
    "detect_ip",
    "detect_mac_address",
    "detect_url",
]

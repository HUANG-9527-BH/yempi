"""标记并过滤中间件内部的模型调用。

Tag internal calls with `internal_call_metadata()` and declare
`transformers = (InternalCallTransformer,)` on the middleware class to keep
them out of `run.messages`. Both APIs are public for third-party middleware.
"""

from __future__ import annotations

import secrets
from typing import TYPE_CHECKING, Any, ClassVar

from langchain_core.messages import BaseMessage
from langgraph.stream import StreamTransformer

if TYPE_CHECKING:
    from langgraph.stream._types import ProtocolEvent

INTERNAL_CALL_METADATA_KEY = "lc_internal_call"
"""Metadata key marking a model call as internal to middleware.

Kept separate from `lc_source` so filtering doesn't affect its existing
semantics.
"""

_INTERNAL_CALL_TOKEN = secrets.token_hex(16)
"""Process-local marker used to prevent callers from spoofing internal calls.

A random token prevents user-supplied metadata from hiding real model calls
from `run.messages`.
"""


def internal_call_metadata() -> dict[str, Any]:
    """返回标记模型调用为内部调用的元数据。

    Returns:
        一个可合并到模型调用的 `config["metadata"]` 中的映射。
    """
    return {INTERNAL_CALL_METADATA_KEY: _INTERNAL_CALL_TOKEN}


class InternalCallTransformer(StreamTransformer):
    """将内部模型调用从 `run.messages` 和原始事件日志中排除。

    用于执行内部模型调用且在内置转换器之前运行的中间件。

    For tagged events, streamed `message-start` events are marked as tool-role and
    whole-`AIMessage` payloads are cleared so `MessagesTransformer` ignores them.
    The mutated events are then dropped from the raw log.

    仅修改此转换器作用域内的事件。
    """

    before_builtins: ClassVar[bool] = True
    required_stream_modes: ClassVar[tuple[str, ...]] = ("messages",)

    def __init__(self, scope: tuple[str, ...] = ()) -> None:
        """初始化转换器，作用域限定为单一 mux/命名空间。

        Args:
            scope: 拥有 mux 所属的命名空间元组。
        """
        super().__init__(scope)
        self._scope_list: list[str] = list(scope)

    def init(self) -> dict[str, Any]:
        """返回一个空投影——此转换器仅用于抑制事件。

        Returns:
            一个空映射；不添加任何 `run.<key>` 投影。
        """
        return {}

    def process(self, event: ProtocolEvent) -> bool:
        """在提示 `MessagesTransformer` 忽略它们后，丢弃已标记的 `messages` 事件。

        Args:
            event: 要观察的协议事件，可能进行修改。

        Returns:
            `False` to drop a tagged internal call from the raw event log and
            `run.messages`; `True` otherwise.
        """
        if event["method"] != "messages":
            return True

        params = event["params"]
        if params["namespace"] != self._scope_list:
            return True

        payload, metadata = params["data"]
        is_internal = (
            bool(metadata) and metadata.get(INTERNAL_CALL_METADATA_KEY) == _INTERNAL_CALL_TOKEN
        )
        if not is_internal:
            return True

                # 只有 `message-start` 需要修改：将其标记为 `"tool"` 可使
                # `MessagesTransformer` 忽略该运行的其余部分。所有事件仍会在下方被丢弃。
                # 从原始日志中丢弃。
        if isinstance(payload, dict) and payload.get("event") == "message-start":
            payload["role"] = "tool"
        elif isinstance(payload, BaseMessage):
            params["data"] = (None, metadata)

        return False

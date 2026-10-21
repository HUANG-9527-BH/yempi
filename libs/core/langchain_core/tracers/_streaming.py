"""用于 `stream_log` 和 `astream` 事件实现的内部追踪器。"""

import typing
from collections.abc import AsyncIterator, Iterator
from uuid import UUID

T = typing.TypeVar("T")


# THIS IS USED IN LANGGRAPH。
@typing.runtime_checkable
class _StreamingCallbackHandler(typing.Protocol[T]):
    """流式回调处理器的类型。

    这是一个通用的混合类，`astream` 事件和
    `astream log` 的回调处理器均继承自该类。

    在某些上下文中调用 `tap_output_aiter` 方法以生成中间结果的回调。
    """

    def tap_output_aiter(
        self, run_id: UUID, output: AsyncIterator[T]
    ) -> AsyncIterator[T]:
        """用于内部 `astream_log` 和 `astream` 事件实现。"""

    def tap_output_iter(self, run_id: UUID, output: Iterator[T]) -> Iterator[T]:
        """用于内部 `astream_log` 和 `astream` 事件实现。"""


# THIS IS USED IN LANGGRAPH。
class _V2StreamingCallbackHandler:
    """用于消耗 `on_stream_event`（v2）的处理器的标记基类。

    A handler inheriting from this class signals that it wants content-
    block lifecycle events from `stream_events(version="v3")` (and its
    async equivalent) rather than the v1 `on_llm_new_token` chunks.
    `BaseChatModel.invoke` uses
    `isinstance(handler, _V2StreamingCallbackHandler)` to decide whether
    to route an invoke through the v2 event generator.

    Implemented as a concrete marker class (not a `Protocol`) so opt-in
    is explicit via inheritance. An empty `runtime_checkable` Protocol
    would match every object and misroute every call. The event
    delivery contract itself lives on
    `BaseCallbackHandler.on_stream_event`.
    """


__all__ = [
    "_StreamingCallbackHandler",
    "_V2StreamingCallbackHandler",
]

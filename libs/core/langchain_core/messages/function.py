"""函数消息。"""

from typing import Any, Literal

from typing_extensions import override

from langchain_core.messages.base import (
    BaseMessage,
    BaseMessageChunk,
    merge_content,
)
from langchain_core.utils._merge import merge_dicts


class FunctionMessage(BaseMessage):
    """用于将工具执行结果传回模型的消息。

    `FunctionMessage` 是 `ToolMessage` 模式的旧版本，
    不包含 `tool_call_id` 字段。

    The `tool_call_id` field is used to associate the tool call request with the
    tool call response. Useful in situations where a chat model is able
    to request multiple tool calls in parallel.

    """

    name: str
    """已执行函数的名称。"""

    type: Literal["function"] = "function"
    """消息的类型（用于序列化）。"""


class FunctionMessageChunk(FunctionMessage, BaseMessageChunk):
    """函数消息块。"""

        # 忽略 mypy 重新赋值，因为我们正在覆盖该值
        # 以确保可以区分块变体和非块变体。
        # 以确保可以区分块变体和非块变体。
    type: Literal["FunctionMessageChunk"] = "FunctionMessageChunk"  # type: ignore[assignment]
    """消息的类型（用于序列化）。"""

    @override
    def __add__(self, other: Any) -> BaseMessageChunk:  # type: ignore[override]
        if isinstance(other, FunctionMessageChunk):
            if self.name != other.name:
                msg = "Cannot concatenate FunctionMessageChunks with different names."
                raise ValueError(msg)

            return self.__class__(
                name=self.name,
                content=merge_content(self.content, other.content),
                additional_kwargs=merge_dicts(
                    self.additional_kwargs, other.additional_kwargs
                ),
                response_metadata=merge_dicts(
                    self.response_metadata, other.response_metadata
                ),
                id=self.id,
            )

        return super().__add__(other)

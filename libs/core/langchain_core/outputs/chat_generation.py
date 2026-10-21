"""聊天生成输出类。"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from pydantic import model_validator

from langchain_core.messages import BaseMessage, BaseMessageChunk
from langchain_core.outputs.generation import Generation
from langchain_core.utils._merge import merge_dicts

if TYPE_CHECKING:
    from typing_extensions import Self


class ChatGeneration(Generation):
    """单个聊天生成输出。

    A subclass of `Generation` that represents the response from a chat model that
    generates chat messages.

    `message` 属性是聊天消息的结构化表示。大多数情况下，该消息为 `AIMessage` 类型。

    Users working with chat models will usually access information via either
    `AIMessage` (returned from runnable interfaces) or `LLMResult` (available via
    callbacks).
    """

    text: str = ""
    """输出消息的文本内容。

    !!! warning "不应直接设置！"

    """
    message: BaseMessage
    """聊天模型输出的消息。"""

        # 覆盖类型为 ChatGeneration，忽略 mypy 错误，此为有意为之
    type: Literal["ChatGeneration"] = "ChatGeneration"  # type: ignore[assignment]
    """类型仅用于序列化目的。"""

    @model_validator(mode="after")
    def set_text(self) -> Self:
        """将 text 属性设置为消息的内容。"""
                # 检查是否存在带有 "text" 键但无 "type" 字段的旧版块。
                # 否则，委托给 `message.text`。
        if isinstance(self.message.content, list):
            has_legacy_blocks = any(
                isinstance(block, dict)
                and "text" in block
                and block.get("type") is None
                for block in self.message.content
            )

            if has_legacy_blocks:
                blocks = []
                for block in self.message.content:
                    if isinstance(block, str):
                        blocks.append(block)
                    elif isinstance(block, dict):
                        block_type = block.get("type")
                        if block_type == "text" or (
                            block_type is None and "text" in block
                        ):
                            blocks.append(block.get("text", ""))
                self.text = "".join(blocks)
            else:
                self.text = self.message.text
        else:
            self.text = self.message.text

        return self


class ChatGenerationChunk(ChatGeneration):
    """`ChatGeneration` 块。

    `ChatGeneration` 块可以与其他 `ChatGeneration` 块拼接。
    """

    message: BaseMessageChunk
    """聊天模型输出的消息块。"""
        # 覆盖类型为 ChatGeneration，忽略 mypy 错误，此为有意为之

    type: Literal["ChatGenerationChunk"] = "ChatGenerationChunk"  # type: ignore[assignment]
    """类型仅用于序列化目的。"""

    def __add__(
        self, other: ChatGenerationChunk | list[ChatGenerationChunk]
    ) -> ChatGenerationChunk:
        """拼接两个 `ChatGenerationChunk`。

        Args:
            other: 要拼接的另一个 `ChatGenerationChunk` 或 `ChatGenerationChunk` 列表。

        Raises:
            TypeError: 若 other 不是 `ChatGenerationChunk` 或 `ChatGenerationChunk`
            列表。

        Returns:
            由 self 和 other 拼接而成的新 `ChatGenerationChunk`。
        """
        if isinstance(other, ChatGenerationChunk):
            generation_info = merge_dicts(
                self.generation_info or {},
                other.generation_info or {},
            )
            return ChatGenerationChunk(
                message=self.message + other.message,
                generation_info=generation_info or None,
            )
        if isinstance(other, list) and all(
            isinstance(x, ChatGenerationChunk) for x in other
        ):
            generation_info = merge_dicts(
                self.generation_info or {},
                *[chunk.generation_info for chunk in other if chunk.generation_info],
            )
            return ChatGenerationChunk(
                message=self.message + [chunk.message for chunk in other],
                generation_info=generation_info or None,
            )
        msg = f"unsupported operand type(s) for +: '{type(self)}' and '{type(other)}'"
        raise TypeError(msg)


def merge_chat_generation_chunks(
    chunks: list[ChatGenerationChunk],
) -> ChatGenerationChunk | None:
    """将一组 `ChatGenerationChunk` 合并为一个 `ChatGenerationChunk`。

    Args:
        chunks: 要合并的 `ChatGenerationChunk` 列表。

    Returns:
        合并后的 `ChatGenerationChunk`，若输入列表为空则返回 `None`。
    """
    if not chunks:
        return None

    if len(chunks) == 1:
        return chunks[0]

    return chunks[0] + chunks[1:]

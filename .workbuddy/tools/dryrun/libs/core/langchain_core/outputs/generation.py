"""生成输出的模式。"""

from __future__ import annotations

from typing import Any, Literal

from langchain_core.load import Serializable
from langchain_core.utils._merge import merge_dicts


class Generation(Serializable):
    """单个文本生成输出。

    Generation represents the response from an "old-fashioned" LLM (string-in,
    string-out) that generates regular text (not chat messages).

    此模型在内部由聊天模型使用，并最终将被映射到更通用的 `LLMResult` 对象，然后投影为
    `AIMessage` 对象。

    LangChain users working with chat models will usually access information via
    `AIMessage` (returned from runnable interfaces) or `LLMResult` (available via
    callbacks). Please refer to `AIMessage` and `LLMResult` for more information.
    """

    text: str
    """生成的文本输出。"""

    generation_info: dict[str, Any] | None = None
    """来自提供方的原始响应。

    可能包含终止原因或 token 日志概率等信息。
    """

    type: Literal["Generation"] = "Generation"
    """类型仅用于序列化目的。

    此类设置为 `'Generation'`。
    """

    @classmethod
    def is_lc_serializable(cls) -> bool:
        """返回 `True`，因为此类可序列化。"""
        return True

    @classmethod
    def get_lc_namespace(cls) -> list[str]:
        """获取 LangChain 对象的命名空间。

        Returns:
            `["langchain", "schema", "output"]`
        """
        return ["langchain", "schema", "output"]


class GenerationChunk(Generation):
    """`GenerationChunk`，可与其他 `Generation` 块拼接。"""

    def __add__(self, other: GenerationChunk) -> GenerationChunk:
        """拼接两个 `GenerationChunk` 对象。

        Args:
            other: 另一个要拼接的 `GenerationChunk`。

        Raises:
            TypeError: 如果 other 不是 `GenerationChunk`。

        Returns:
            由 self 和 other 拼接而成的新 `GenerationChunk`。
        """
        if isinstance(other, GenerationChunk):
            generation_info = merge_dicts(
                self.generation_info or {},
                other.generation_info or {},
            )
            return GenerationChunk(
                text=self.text + other.text,
                generation_info=generation_info or None,
            )
        msg = f"unsupported operand type(s) for +: '{type(self)}' and '{type(other)}'"  # type: ignore[unreachable]
        raise TypeError(msg)

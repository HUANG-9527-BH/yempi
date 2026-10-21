"""语言模型提示词的提示值。

提示值用于表示不同的提示片段。它们可用于表示文本、图像或聊天消息片段。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any, Literal, cast

from typing_extensions import TypedDict

from langchain_core.load.serializable import Serializable
from langchain_core.messages import (
    AnyMessage,
    BaseMessage,
    HumanMessage,
    get_buffer_string,
)


class PromptValue(Serializable, ABC):
    """任何语言模型输入的基类抽象类。

    `PromptValues` 可以转换为 LLM（纯文本生成）输入和聊天模型输入。
    """

    @classmethod
    def is_lc_serializable(cls) -> bool:
        """返回 `True`，因为此类是可序列化的。"""
        return True

    @classmethod
    def get_lc_namespace(cls) -> list[str]:
        """获取 LangChain 对象的命名空间。

        Returns:
            `["langchain", "schema", "prompt"]`
        """
        return ["langchain", "schema", "prompt"]

    @abstractmethod
    def to_string(self) -> str:
        """将提示值作为字符串返回。"""

    @abstractmethod
    def to_messages(self) -> list[BaseMessage]:
        """将提示作为消息列表返回。"""


class StringPromptValue(PromptValue):
    """字符串提示值。"""

    text: str
    """提示文本。"""

    type: Literal["StringPromptValue"] = "StringPromptValue"

    @classmethod
    def get_lc_namespace(cls) -> list[str]:
        """获取 LangChain 对象的命名空间。

        Returns:
            `["langchain", "prompts", "base"]`
        """
        return ["langchain", "prompts", "base"]

    def to_string(self) -> str:
        """将提示作为字符串返回。"""
        return self.text

    def to_messages(self) -> list[BaseMessage]:
        """将提示作为消息返回。"""
        return [HumanMessage(content=self.text)]


class ChatPromptValue(PromptValue):
    """聊天提示值。

    一种由消息构建的提示值类型。
    """

    messages: Sequence[BaseMessage]
    """消息列表。"""

    def to_string(self) -> str:
        """将提示作为字符串返回。"""
        return get_buffer_string(self.messages)

    def to_messages(self) -> list[BaseMessage]:
        """将提示作为消息列表返回。"""
        return list(self.messages)

    @classmethod
    def get_lc_namespace(cls) -> list[str]:
        """获取 LangChain 对象的命名空间。

        Returns:
            `["langchain", "prompts", "chat"]`
        """
        return ["langchain", "prompts", "chat"]


class ImageURL(TypedDict, total=False):
    """多模态模型输入的图像 URL（OpenAI 格式）。

    表示 OpenAI 聊天完成 API 格式中的内部 `image_url` 对象。此对象由
    `ImagePromptTemplate` 和 `ChatPromptTemplate` 使用。

    See Also:
        `ImageContentBlock`: LangChain's provider-agnostic image format used in message
        content blocks. Use `ImageContentBlock` when working with the standardized
        message format across different providers.

    Note:
        `detail` 字段的值不会在本地进行验证。无效值将被下游 API
        拒绝，允许使用新的有效值而无需 LangChain 更新。
    """

    detail: Literal["auto", "low", "high"]
    """指定图像的详细程度。

    Defaults to ``'auto'`` if not specified. Higher detail levels consume
    more tokens but provide better image understanding.
    """

    url: str
    """图像的 URL 或 base64 编码的图像数据。"""


class ImagePromptValue(PromptValue):
    """图像提示值。"""

    image_url: ImageURL
    """图像 URL。"""

    type: Literal["ImagePromptValue"] = "ImagePromptValue"

    def to_string(self) -> str:
        """将提示（图像 URL）作为字符串返回。"""
        return self.image_url.get("url", "")

    def to_messages(self) -> list[BaseMessage]:
        """将提示（图像 URL）作为消息返回。"""
        return [HumanMessage(content=[cast("dict[str, Any]", self.image_url)])]


class ChatPromptValueConcrete(ChatPromptValue):
    """显式列出其接受的消息类型的聊天提示值。

    用于外部模式中。
    """

    messages: Sequence[AnyMessage]
    """消息序列。"""

    type: Literal["ChatPromptValueConcrete"] = "ChatPromptValueConcrete"

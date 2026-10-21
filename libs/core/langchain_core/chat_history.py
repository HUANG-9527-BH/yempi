"""聊天消息历史记录存储聊天中的消息交互历史。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from langchain_core._api import deprecated
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    get_buffer_string,
)
from langchain_core.runnables.config import run_in_executor

if TYPE_CHECKING:
    from collections.abc import Sequence


@deprecated(
    since="1.6.4",
    removal="2.0.0",
    addendum=(
        "See the short-term memory documentation for recommended alternatives: "
        "https://docs.langchain.com/oss/python/langchain/short-term-memory"
    ),
)
class BaseChatMessageHistory(ABC):
    """用于存储聊天消息历史的抽象基类。

    实现指南：

    实现应重写以下方法中的全部或部分：

    * `add_messages`: 批量添加消息的同步版本
    * `aadd_messages`: 批量添加消息的异步版本
    * `messages`: 获取消息的同步版本
    * `aget_messages`: 获取消息的异步版本
    * `clear`: 清除消息的同步版本
    * `aclear`: 清除消息的异步版本

    `add_messages` contains a default implementation that calls `add_message`
    for each message in the sequence. This is provided for backwards compatibility
    with existing implementations which only had `add_message`.

    Async variants all have default implementations that call the sync variants.
    Implementers can choose to override the async implementations to provide
    truly async implementations.

    使用指南：

    When used for updating history, users should favor usage of `add_messages`
    over `add_message` or other variants like `add_user_message` and `add_ai_message`
    to avoid unnecessary round-trips to the underlying persistence layer.

    Example:
        ```python
        import json
        import os
        from langchain_core.messages import messages_from_dict, message_to_dict


        class FileChatMessageHistory(BaseChatMessageHistory):
            storage_path: str
            session_id: str

            @property
            def messages(self) -> list[BaseMessage]:
                try:
                    with open(
                        os.path.join(self.storage_path, self.session_id),
                        "r",
                        encoding="utf-8",
                    ) as f:
                        messages_data = json.load(f)
                    return messages_from_dict(messages_data)
                except FileNotFoundError:
                    return []

            def add_messages(self, messages: Sequence[BaseMessage]) -> None:
                all_messages = list(self.messages)  # Existing messages
                all_messages.extend(messages)  # Add new messages

                serialized = [message_to_dict(message) for message in all_messages]
                file_path = os.path.join(self.storage_path, self.session_id)
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                with open(file_path, "w", encoding="utf-8") as f:
                    json.dump(serialized, f)

            def clear(self) -> None:
                file_path = os.path.join(self.storage_path, self.session_id)
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                with open(file_path, "w", encoding="utf-8") as f:
                    json.dump([], f)
        ```
    """

    messages: list[BaseMessage]
    """返回消息列表的属性或属性。

    通常情况下，获取消息可能涉及对底层持久化层的 I/O
    操作，因此该操作预计会产生一定的延迟。
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """通过协同初始化以保留多重继承。"""
        super().__init__(*args, **kwargs)

    async def aget_messages(self) -> list[BaseMessage]:
        """获取消息的异步版本。

        可重写此方法以提供高效的异步实现。

        通常情况下，获取消息可能涉及对底层持久化层的 I/O 操作。

        Returns:
            返回的消息。
        """
        return await run_in_executor(None, lambda: self.messages)

    def add_user_message(self, message: HumanMessage | str) -> None:
        """用于向存储中添加人类消息字符串的便捷方法。

        !!! note

            这是一个便捷方法。代码应优先使用批量 `add_messages`
            接口，以减少对持久化层的往返调用。

        此方法在未来版本中可能会被弃用。

        Args:
            message: 要添加到存储中的 `HumanMessage`。
        """
        if isinstance(message, HumanMessage):
            self.add_message(message)
        else:
            self.add_message(HumanMessage(content=message))

    def add_ai_message(self, message: AIMessage | str) -> None:
        """用于向存储中添加 `AIMessage` 字符串的便捷方法。

        !!! note

            这是一个便捷方法。代码应优先使用批量 `add_messages`
            接口，以减少对持久化层的往返调用。

        此方法在未来版本中可能会被弃用。

        Args:
            message: 要添加的 `AIMessage`。
        """
        if isinstance(message, AIMessage):
            self.add_message(message)
        else:
            self.add_message(AIMessage(content=message))

    def add_message(self, message: BaseMessage) -> None:
        """将 Message 对象添加到存储中。

        Args:
            message: 要存储的 `BaseMessage` 对象。

        Raises:
            NotImplementedError: 如果子类未实现高效的 `add_messages` 方法。
        """
        if type(self).add_messages != BaseChatMessageHistory.add_messages:
                        # 这意味着子类已实现高效的 add_messages 方法，因此应使用它。
                        # 这表示子类已实现高效的 add_messages 方法，因此应使用它。
            self.add_messages([message])
        else:
            msg = (
                "add_message is not implemented for this class. "
                "Please implement add_message or add_messages."
            )
            raise NotImplementedError(msg)

    def add_messages(self, messages: Sequence[BaseMessage]) -> None:
        """添加一组消息。

        实现应重写此方法，以高效地处理批量消息添加，避免对底层存储产生不必要的往返调用。

        Args:
            messages: 要存储的 `BaseMessage` 对象序列。
        """
        for message in messages:
            self.add_message(message)

    async def aadd_messages(self, messages: Sequence[BaseMessage]) -> None:
        """异步添加消息列表。

        Args:
            messages: 要存储的 `BaseMessage` 对象序列。
        """
        await run_in_executor(None, self.add_messages, messages)

    @abstractmethod
    def clear(self) -> None:
        """从存储中移除所有消息。"""

    async def aclear(self) -> None:
        """异步从存储中移除所有消息。"""
        await run_in_executor(None, self.clear)

    def __str__(self) -> str:
        """返回聊天历史的字符串表示形式。"""
        return get_buffer_string(self.messages)


@deprecated(
    since="1.6.4",
    removal="2.0.0",
    addendum=(
        "See the short-term memory documentation for recommended alternatives: "
        "https://docs.langchain.com/oss/python/langchain/short-term-memory"
    ),
)
class InMemoryChatMessageHistory(BaseChatMessageHistory, BaseModel):
    """聊天消息历史的内存实现。

    将消息存储在内存列表中。
    """

    messages: list[BaseMessage] = Field(default_factory=list)
    """存储在内存中的消息列表。"""

    async def aget_messages(self) -> list[BaseMessage]:
        """获取消息的异步版本。

        可重写此方法以提供高效的异步实现。

        通常，获取消息可能涉及对底层持久化层的 I/O 操作。

        Returns:
            消息列表。
        """
        return self.messages

    def add_message(self, message: BaseMessage) -> None:
        """向存储中添加一个自创建的消息。

        Args:
            message: 要添加的消息。
        """
        self.messages.append(message)

    async def aadd_messages(self, messages: Sequence[BaseMessage]) -> None:
        """异步向存储中添加消息。

        Args:
            messages: 要添加的消息。
        """
        self.add_messages(messages)

    def clear(self) -> None:
        """清空存储中的所有消息。"""
        self.messages = []

    async def aclear(self) -> None:
        """异步清空存储中的所有消息。"""
        self.clear()

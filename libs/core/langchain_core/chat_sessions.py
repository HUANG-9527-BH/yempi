"""**聊天会话** 是消息和函数调用的集合。"""

from collections.abc import Sequence
from typing import Any, TypedDict

from langchain_core.messages import BaseMessage


class ChatSession(TypedDict, total=False):
    """聊天会话。

    聊天会话代表一次对话、频道或其他消息组。
    """

    messages: Sequence[BaseMessage]
    """从源加载的 LangChain 聊天消息序列。"""

    functions: Sequence[dict[str, Any]]
    """消息的函数调用规范序列。"""

"""聊天加载器。"""

from abc import ABC, abstractmethod
from collections.abc import Iterator

from langchain_core.chat_sessions import ChatSession


class BaseChatLoader(ABC):
    """聊天加载器的基类。"""

    @abstractmethod
    def lazy_load(self) -> Iterator[ChatSession]:
        """延迟加载聊天会话。

        Returns:
            聊天会话的迭代器。
        """

    def load(self) -> list[ChatSession]:
        """立即加载聊天会话到内存中。

        Returns:
            聊天会话的列表。
        """
        return list(self.lazy_load())

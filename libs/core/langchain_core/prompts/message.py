"""消息提示词模板。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from langchain_core.load import Serializable
from langchain_core.utils.interactive_env import is_interactive_env

if TYPE_CHECKING:
    from langchain_core.messages import BaseMessage
    from langchain_core.prompts.chat import ChatPromptTemplate


class BaseMessagePromptTemplate(Serializable, ABC):
    """消息提示词模板的基类。"""

    @classmethod
    def is_lc_serializable(cls) -> bool:
        """返回 `True`，因为此类可序列化。"""
        return True

    @classmethod
    def get_lc_namespace(cls) -> list[str]:
        """获取 LangChain 对象的命名空间。

        Returns:
            `["langchain", "prompts", "chat"]`
        """
        return ["langchain", "prompts", "chat"]

    @abstractmethod
    def format_messages(self, **kwargs: Any) -> list[BaseMessage]:
        """从关键字参数格式化消息。

        应返回 `BaseMessage` 对象列表。

        Args:
            **kwargs: 用于格式化的关键字参数。

        Returns:
            `BaseMessage` 对象列表。
        """

    async def aformat_messages(self, **kwargs: Any) -> list[BaseMessage]:
        """异步从关键字参数格式化消息。

        Args:
            **kwargs: 用于格式化的关键字参数。

        Returns:
            `BaseMessage` 对象列表。
        """
        return self.format_messages(**kwargs)

    @property
    @abstractmethod
    def input_variables(self) -> list[str]:
        """此提示词模板的输入变量。

        Returns:
            输入变量列表。
        """

    def pretty_repr(
        self,
        html: bool = False,  # noqa: FBT001,FBT002
    ) -> str:
        """人类可读的表示形式。

        Args:
            html: 是否以 HTML 格式化。

        Returns:
            人类可读的表示形式。
        """
        raise NotImplementedError

    def pretty_print(self) -> None:
        """打印人类可读的表示形式。"""
        print(self.pretty_repr(html=is_interactive_env()))  # noqa: T201

    def __add__(self, other: Any) -> ChatPromptTemplate:
        """合并两个提示词模板。

        Args:
            other: 另一个提示词模板。

        Returns:
            合并后的提示词模板。
        """
                # Import locally to avoid circular import.
        from langchain_core.prompts.chat import ChatPromptTemplate  # noqa: PLC0415

        prompt = ChatPromptTemplate(messages=[self])
        return prompt.__add__(other)

"""负责删除其他消息的消息。"""

from typing import Any, Literal

from langchain_core.messages.base import BaseMessage


class RemoveMessage(BaseMessage):
    """负责删除其他消息的消息。"""

    type: Literal["remove"] = "remove"
    """消息的类型（用于序列化）。"""

    def __init__(
        self,
        id: str,
        **kwargs: Any,
    ) -> None:
        """创建一个 RemoveMessage。

        Args:
            id: 要删除的消息的 ID。
            **kwargs: 传递给消息的额外字段。

        Raises:
            ValueError: 如果在 kwargs 中传入了 'content' 字段。

        """
        if kwargs.pop("content", None):
            msg = "RemoveMessage does not support 'content' field."
            raise ValueError(msg)

        super().__init__("", id=id, **kwargs)

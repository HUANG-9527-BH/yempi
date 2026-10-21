"""系统消息。"""

from typing import Any, Literal, cast, overload

from langchain_core.messages import content as types
from langchain_core.messages.base import BaseMessage, BaseMessageChunk


class SystemMessage(BaseMessage):
    """用于引导 AI 行为的消息。

    系统消息通常作为输入消息序列中的第一个消息传入。

    Example:
        ```python
        from langchain_core.messages import HumanMessage, SystemMessage

        messages = [
            SystemMessage(content="You are a helpful assistant! Your name is Bob."),
            HumanMessage(content="What is your name?"),
        ]

        # Define a chat model and invoke it with the messages
        print(model.invoke(messages))
        ```
    """

    type: Literal["system"] = "system"
    """消息的类型（用于序列化）。"""

    @overload
    def __init__(
        self,
        content: str | list[str | dict[Any, Any]],
        **kwargs: Any,
    ) -> None: ...

    @overload
    def __init__(
        self,
        content: str | list[str | dict[Any, Any]] | None = None,
        content_blocks: list[types.ContentBlock] | None = None,
        **kwargs: Any,
    ) -> None: ...

    def __init__(
        self,
        content: str | list[str | dict[Any, Any]] | None = None,
        content_blocks: list[types.ContentBlock] | None = None,
        **kwargs: Any,
    ) -> None:
        """指定 `content` 作为位置参数，或使用 `content_blocks` 以获得类型支持。"""
        if content_blocks is not None:
            super().__init__(
                content=cast("list[str | dict[Any, Any]]", content_blocks),
                **kwargs,
            )
        else:
            super().__init__(content=content, **kwargs)


class SystemMessageChunk(SystemMessage, BaseMessageChunk):
    """系统消息块。"""

        # 忽略 mypy 的重新赋值，因为我们正在覆盖该值
        # 以确保可以区分块变体和非块变体。
        # 非分块版本。
    type: Literal["SystemMessageChunk"] = "SystemMessageChunk"  # type: ignore[assignment]
    """消息的类型（用于序列化）。"""

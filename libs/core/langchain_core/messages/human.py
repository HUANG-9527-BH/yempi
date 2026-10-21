"""用户消息。"""

from typing import Any, Literal, cast, overload

from langchain_core.messages import content as types
from langchain_core.messages.base import BaseMessage, BaseMessageChunk


class HumanMessage(BaseMessage):
    """来自用户的消息。

    `HumanMessage` 是从用户传递给模型的消息。

    Example:
        ```python
        from langchain_core.messages import HumanMessage, SystemMessage

        messages = [
            SystemMessage(content="You are a helpful assistant! Your name is Bob."),
            HumanMessage(content="What is your name?"),
        ]

        # Instantiate a chat model and invoke it with the messages
        model = ...
        print(model.invoke(messages))
        ```
    """

    type: Literal["human"] = "human"
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
        """指定 `content` 作为位置参数，或使用 `content_blocks` 以获得更好的类型提示。"""
        if content_blocks is not None:
            super().__init__(
                content=cast("list[str | dict[Any, Any]]", content_blocks),
                **kwargs,
            )
        else:
            super().__init__(content=content, **kwargs)


class HumanMessageChunk(HumanMessage, BaseMessageChunk):
    """用户消息块。"""

        # 忽略 mypy 的重新赋值警告，因为我们正在覆盖该值
        # 以确保消息块变体可以与非块变体区分开来。
        # 非分块版本。
    type: Literal["HumanMessageChunk"] = "HumanMessageChunk"  # type: ignore[assignment]
    """消息的类型（用于序列化）。"""

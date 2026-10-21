"""基础消息。"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast, overload

from pydantic import ConfigDict, Field

from langchain_core._api.deprecation import warn_deprecated
from langchain_core.load.serializable import Serializable
from langchain_core.messages import content as types
from langchain_core.utils import get_bolded_text
from langchain_core.utils._merge import merge_dicts, merge_lists
from langchain_core.utils.interactive_env import is_interactive_env

if TYPE_CHECKING:
    from collections.abc import Sequence

    from typing_extensions import Self

    from langchain_core.prompts.chat import ChatPromptTemplate


def _extract_reasoning_from_additional_kwargs(
    message: BaseMessage,
) -> types.ReasoningContentBlock | None:
    """从 `additional_kwargs` 中提取 `reasoning_content`。

    处理以不同格式存储的推理内容：
    - `additional_kwargs["reasoning_content"]` (字符串) - Ollama、DeepSeek、XAI、Groq

    Args:
        message: 要从中提取推理内容的消息。

    Returns:
        如果找到推理内容，则返回一个 `ReasoningContentBlock`，否则返回 None。
    """
    additional_kwargs = getattr(message, "additional_kwargs", {})

    reasoning_content = additional_kwargs.get("reasoning_content")
    if reasoning_content is not None and isinstance(reasoning_content, str):
        return {"type": "reasoning", "reasoning": reasoning_content}

    return None


class TextAccessor(str):
    """支持属性和方法访问模式的字符串类对象。

    Exists to maintain backward compatibility while transitioning from method-based to
    property-based text access in message objects. In LangChain <v1.0, message text was
    accessed via `.text()` method calls. In v1.0=<, the preferred pattern is property
    access via `.text`.

    Rather than breaking existing code immediately, `TextAccessor` allows both
    patterns:
    - Modern property access: `message.text` (returns string directly)
    - Legacy method access: `message.text()` (callable, emits deprecation warning)

    """

    __slots__ = ()

    def __new__(cls, value: str) -> Self:
        """创建新的 TextAccessor 实例。"""
        return str.__new__(cls, value)

    def __call__(self) -> str:
        """启用方法式文本访问以保持向后兼容性。

        此方法仅用于支持调用 `.text()` 作为方法的旧代码。新代码应改用属性访问（`.text`）

        !!! 已弃用
            从 `langchain-core` 1.0.0 版本起，调用 `.text()` 方法已被弃用。
            请改用 `.text` 属性。该方法将在 2.0.0 版本中移除。

        Returns:
            字符串内容，与属性访问相同。

        """
        warn_deprecated(
            since="1.0.0",
            message=(
                "Calling .text() as a method is deprecated. "
                "Use .text as a property instead (e.g., message.text)."
            ),
            removal="2.0.0",
        )
        return str(self)


class BaseMessage(Serializable):
    """基础抽象消息类。

    消息是聊天模型的输入和输出。

    示例包括 [`HumanMessage`][langchain.messages.HumanMessage]、
    [`AIMessage`][langchain.messages.AIMessage] 和
    [`SystemMessage`][langchain.messages.SystemMessage]。
    """

    content: str | list[str | dict[Any, Any]]
    """消息的内容。"""

    additional_kwargs: dict[Any, Any] = Field(default_factory=dict)
    """保留用于与消息关联的附加负载数据。

    For example, for a message from an AI, this could include tool calls as
    encoded by the model provider.

    """

    response_metadata: dict[Any, Any] = Field(default_factory=dict)
    """Examples: response headers, logprobs, token counts, model name."""

    type: str
    """消息的类型。必须是唯一标识消息类型的字符串。

    The purpose of this field is to allow for easy identification of the message type
    when deserializing messages.

    """

    name: str | None = None
    """消息的可选名称。

    可用于为消息提供人类可读的名称。

    使用此字段是可选的，是否使用取决于模型实现。

    """

    id: str | None = Field(default=None, coerce_numbers_to_str=True)
    """消息的可选唯一标识符。

    此标识符应由创建该消息的提供方/模型提供。

    """

    model_config = ConfigDict(
        extra="allow",
    )

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
        """初始化一个 `BaseMessage`。

        将 `content` 作为位置参数指定，或使用 `content_blocks` 进行类型标注。

        Args:
            content: 消息的内容。
            content_blocks: 类型化的标准内容。
            **kwargs: 传递给父类的其他参数。
        """
        if content_blocks is not None:
            super().__init__(content=content_blocks, **kwargs)
        else:
            super().__init__(content=content, **kwargs)

    @classmethod
    def is_lc_serializable(cls) -> bool:
        """`BaseMessage` 是可序列化的。

        Returns:
            True
        """
        return True

    @classmethod
    def get_lc_namespace(cls) -> list[str]:
        """获取 LangChain 对象的命名空间。

        Returns:
            `["langchain", "schema", "messages"]`
        """
        return ["langchain", "schema", "messages"]

    @property
    def content_blocks(self) -> list[types.ContentBlock]:
        r"""从消息内容中加载内容块。

        !!! 版本新增 "在 `langchain-core` 1.0.0 中新增"

        """
                # 需要此处避免循环导入，因为这些类会导入 BaseMessages
        from langchain_core.messages.block_translators.anthropic import (  # noqa: PLC0415
            _convert_to_v1_from_anthropic_input,
        )
        from langchain_core.messages.block_translators.bedrock_converse import (  # noqa: PLC0415
            _convert_to_v1_from_converse_input,
        )
        from langchain_core.messages.block_translators.google_genai import (  # noqa: PLC0415
            _convert_to_v1_from_genai_input,
        )
        from langchain_core.messages.block_translators.langchain_v0 import (  # noqa: PLC0415
            _convert_v0_multimodal_input_to_v1,
        )
        from langchain_core.messages.block_translators.openai import (  # noqa: PLC0415
            _convert_to_v1_from_chat_completions_input,
        )

        blocks: list[types.ContentBlock] = []
        content = (
                        # 将字符串内容转换为列表，否则默认视为列表
            [self.content]
            if isinstance(self.content, str) and self.content
            else self.content
        )
        for item in content:
            if isinstance(item, str):
                                # 纯字符串内容被视为文本块
                blocks.append({"type": "text", "text": item})
            elif isinstance(item, dict):
                item_type = item.get("type")
                if item_type not in types.KNOWN_BLOCK_TYPES:
                                        # 处理所有特定提供方或 None 类型的块作为非标准块 -
                                        # 我们稍后会再处理这些
                    blocks.append({"type": "non_standard", "value": item})
                else:
                                        # 防止 v0 版本块共享相同的 `type` 键
                    if "source_type" in item:
                        blocks.append({"type": "non_standard", "value": item})
                        continue

                                        # 这不可能是 v0 版本块（因为它们需要 `source_type`），
                                        # 所以它是已知的 v1 版本块类型
                    blocks.append(cast("types.ContentBlock", item))

                # 后续步骤：尝试解析非标准块。
                # 这是最后一步 - 如果此处无法解析，它将保持为非标准块
        for parsing_step in [
            _convert_v0_multimodal_input_to_v1,
            _convert_to_v1_from_chat_completions_input,
            _convert_to_v1_from_anthropic_input,
            _convert_to_v1_from_genai_input,
            _convert_to_v1_from_converse_input,
        ]:
            blocks = parsing_step(blocks)
        return blocks

    @property
    def text(self) -> TextAccessor:
        """以字符串形式获取消息的文本内容。

        可以作为属性（`message.text`）或方法（`message.text()`）使用。

        支持字符串和列表内容类型（例如用于内容块）。仅提取 `type: 'text'` 的块；
        其他块类型将被忽略。

        !!! 已弃用
            从 `langchain-core` 1.0.0 版本起，调用 `.text()` 方法已被弃用。
            请改用 `.text` 属性。该方法将在 2.0.0 版本中移除。

        Returns:
            消息的文本内容。

        """
        if isinstance(self.content, str):
            text_value = self.content
        else:
                        # 必须是列表
            blocks = [
                block
                for block in self.content
                if isinstance(block, str)
                or (block.get("type") == "text" and isinstance(block.get("text"), str))
            ]
            text_value = "".join(
                block if isinstance(block, str) else block["text"] for block in blocks
            )
        return TextAccessor(text_value)

    def __add__(self, other: Any) -> ChatPromptTemplate:
        """将此消息与另一条消息拼接。

        Args:
            other: 要与此消息拼接的另一条消息。

        Returns:
            包含两条消息的 ChatPromptTemplate。
        """
                # 本地导入以防止循环导入。
        from langchain_core.prompts.chat import ChatPromptTemplate  # noqa: PLC0415

        prompt = ChatPromptTemplate(messages=[self])
        return prompt.__add__(other)

    def pretty_repr(
        self,
        html: bool = False,  # noqa: FBT001,FBT002
    ) -> str:
        """获取消息的美化表示形式。

        Args:
            html: Whether to format the message as HTML. If `True`, the message will be
                formatted with HTML tags.

        Returns:
            消息的美化表示形式。

        Example:
            ```python
            from langchain_core.messages import HumanMessage

            msg = HumanMessage(content="What is the capital of France?")
            print(msg.pretty_repr())
            ```

            结果如下：

            ```txt
            ================================ Human Message =================================

            What is the capital of France?
            ```
        """  # noqa: E501
        title = get_msg_title_repr(self.type.title() + " Message", bold=html)
                # TODO: 处理非字符串内容。
        if self.name is not None:
            title += f"\nName: {self.name}"
        return f"{title}\n\n{self.content}"

    def pretty_print(self) -> None:
        """打印消息的美化表示形式。

        Example:
            ```python
            from langchain_core.messages import AIMessage

            msg = AIMessage(content="The capital of France is Paris.")
            msg.pretty_print()
            ```

            结果如下：

            ```txt
            ================================== Ai Message ==================================

            The capital of France is Paris.
            ```
        """  # noqa: E501
        print(self.pretty_repr(html=is_interactive_env()))  # noqa: T201


def merge_content(
    first_content: str | list[str | dict[Any, Any]],
    *contents: str | list[str | dict[Any, Any]],
) -> str | list[str | dict[Any, Any]]:
    """合并多个消息内容。

    Args:
        first_content: 第一个 `content`。可以是字符串或列表。
        contents: 其他 `content`s。可以是字符串或列表。

    Returns:
        合并后的内容。

    """
    merged: str | list[str | dict[Any, Any]]
    merged = "" if first_content is None else first_content

    for content in contents:
                # 如果当前是字符串
        if isinstance(merged, str):
                        # 如果下一个块也是字符串，则直接朴素地合并它们
            if isinstance(content, str):
                merged += content
                        # 如果下一个块是列表，则将当前内容添加到列表开头
            else:
                merged = [merged, *content]
        elif isinstance(content, list):
                        # 如果两者都是列表
            merged = merge_lists(merged, content)  # type: ignore[assignment]
                # 如果第一个内容是列表，第二个内容是字符串
                # 如果第一个内容的最后一个元素是字符串
                # 将第二个内容添加到最后一个元素中
        elif merged and isinstance(merged[-1], str):
            merged[-1] += content
                # 如果第二个内容为空字符串，则视为无操作
        elif content == "":
            pass
                # 否则，将第二个内容作为列表的新元素添加
        elif merged:
            merged.append(content)
    return merged


class BaseMessageChunk(BaseMessage):
    """消息块，可与其他消息块拼接。"""

    def __add__(self, other: Any) -> BaseMessageChunk:  # type: ignore[override]
        """消息块支持与其他消息块拼接。

        此功能可用于将流式模型生成的消息块合并成完整消息。

        Args:
            other: 要与此消息块拼接的另一条消息块。

        Returns:
            一个新的消息块，是此消息块与另一消息块的拼接结果。

        Raises:
            TypeError: 如果另一个对象不是消息块。

        Example:
            ```txt
              AIMessageChunk(content="Hello", ...)
            + AIMessageChunk(content=" World", ...)
            = AIMessageChunk(content="Hello World", ...)
            ```
        """
        if isinstance(other, BaseMessageChunk):
                        # 如果两者都是 BaseMessageChunk（或其子类）
                        # 合并为单个 BaseMessageChunk

            return self.__class__(
                id=self.id,
                type=self.type,
                content=merge_content(self.content, other.content),
                additional_kwargs=merge_dicts(
                    self.additional_kwargs, other.additional_kwargs
                ),
                response_metadata=merge_dicts(
                    self.response_metadata, other.response_metadata
                ),
            )
        if isinstance(other, list) and all(
            isinstance(o, BaseMessageChunk) for o in other
        ):
            content = merge_content(self.content, *(o.content for o in other))
            additional_kwargs = merge_dicts(
                self.additional_kwargs, *(o.additional_kwargs for o in other)
            )
            response_metadata = merge_dicts(
                self.response_metadata, *(o.response_metadata for o in other)
            )
            return self.__class__(  # type: ignore[call-arg]
                id=self.id,
                content=content,
                additional_kwargs=additional_kwargs,
                response_metadata=response_metadata,
            )
        msg = (
            'unsupported operand type(s) for +: "'
            f"{self.__class__.__name__}"
            f'" and "{other.__class__.__name__}"'
        )
        raise TypeError(msg)


def message_to_dict(message: BaseMessage) -> dict[str, Any]:
    """将消息转换为字典。

    Args:
        message: 要转换的消息。

    Returns:
        Message as a dict. The dict will have a `type` key with the message type
        and a `data` key with the message data as a dict.

    """
    return {"type": message.type, "data": message.model_dump()}


def messages_to_dict(messages: Sequence[BaseMessage]) -> list[dict[str, Any]]:
    """将一系列消息转换为字典列表。

    Args:
        messages: 要转换的一系列消息（作为 `BaseMessage`）。

    Returns:
        消息的字典列表。

    """
    return [message_to_dict(m) for m in messages]


def get_msg_title_repr(title: str, *, bold: bool = False) -> str:
    """获取消息的标题表示形式。

    Args:
        title: 标题。
        bold: 是否加粗标题。

    Returns:
        标题表示形式。

    """
    padded = " " + title + " "
    sep_len = (80 - len(padded)) // 2
    sep = "=" * sep_len
    second_sep = sep + "=" if len(padded) % 2 else sep
    if bold:
        padded = get_bolded_text(padded)
    return f"{sep}{padded}{second_sep}"

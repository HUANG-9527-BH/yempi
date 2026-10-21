"""模块包含用于处理消息的实用函数。

使用这些函数可以实现以下示例功能：

* 将消息转换为字符串（序列化）
* 将字典形式的消息转换为 Message 对象（反序列化）
* 根据名称、类型或 id 等条件从消息列表中筛选消息
"""

from __future__ import annotations

import base64
import inspect
import json
import logging
import math
from collections.abc import Callable, Iterable, Sequence
from functools import partial, wraps
from typing import (
    TYPE_CHECKING,
    Annotated,
    Any,
    Concatenate,
    Literal,
    ParamSpec,
    Protocol,
    TypeVar,
    cast,
    overload,
)
from xml.sax.saxutils import escape, quoteattr

from pydantic import Discriminator, Field, Tag

from langchain_core.exceptions import ErrorCode, create_message
from langchain_core.messages.ai import AIMessage, AIMessageChunk
from langchain_core.messages.base import BaseMessage, BaseMessageChunk
from langchain_core.messages.block_translators.openai import (
    convert_to_openai_data_block,
)
from langchain_core.messages.chat import ChatMessage, ChatMessageChunk
from langchain_core.messages.content import (
    is_data_content_block,
)
from langchain_core.messages.function import FunctionMessage, FunctionMessageChunk
from langchain_core.messages.human import HumanMessage, HumanMessageChunk
from langchain_core.messages.modifier import RemoveMessage
from langchain_core.messages.system import SystemMessage, SystemMessageChunk
from langchain_core.messages.tool import ToolCall, ToolMessage, ToolMessageChunk
from langchain_core.utils.pydantic import model_json_schema as get_model_json_schema

if TYPE_CHECKING:
    from langchain_core.language_models import BaseLanguageModel
    from langchain_core.prompt_values import PromptValue
    from langchain_core.runnables.base import Runnable
    from langchain_core.tools import BaseTool

try:
    from langchain_text_splitters import TextSplitter

    _HAS_LANGCHAIN_TEXT_SPLITTERS = True
except ImportError:
    _HAS_LANGCHAIN_TEXT_SPLITTERS = False

logger = logging.getLogger(__name__)


def _get_type(v: Any) -> str:
    """获取对象关联的类型，用于序列化目的。"""
    if isinstance(v, dict) and "type" in v:
        result = v["type"]
    elif hasattr(v, "type"):
        result = v.type
    else:
        msg = (
            f"Expected either a dictionary with a 'type' key or an object "
            f"with a 'type' attribute. Instead got type {type(v)}."
        )
        raise TypeError(msg)
    if not isinstance(result, str):
        msg = f"Expected 'type' to be a str, got {type(result).__name__}"
        raise TypeError(msg)
    return result


AnyMessage = Annotated[
    Annotated[AIMessage, Tag(tag="ai")]
    | Annotated[HumanMessage, Tag(tag="human")]
    | Annotated[ChatMessage, Tag(tag="chat")]
    | Annotated[SystemMessage, Tag(tag="system")]
    | Annotated[FunctionMessage, Tag(tag="function")]
    | Annotated[ToolMessage, Tag(tag="tool")]
    | Annotated[AIMessageChunk, Tag(tag="AIMessageChunk")]
    | Annotated[HumanMessageChunk, Tag(tag="HumanMessageChunk")]
    | Annotated[ChatMessageChunk, Tag(tag="ChatMessageChunk")]
    | Annotated[SystemMessageChunk, Tag(tag="SystemMessageChunk")]
    | Annotated[FunctionMessageChunk, Tag(tag="FunctionMessageChunk")]
    | Annotated[ToolMessageChunk, Tag(tag="ToolMessageChunk")],
    Field(discriminator=Discriminator(_get_type)),
]
"""A type representing any defined `Message` or `MessageChunk` type."""


def _has_base64_data(block: dict[str, Any]) -> bool:
    """检查内容块是否包含 base64 编码数据。

    Args:
        block: 一个内容块字典。

    Returns:
        该块是否包含 base64 数据。
    """
        # 检查是否存在显式的 base64 字段（标准内容块）
    if block.get("base64"):
        return True

        # 检查 url 字段中是否存在 data: URL
    url = block.get("url", "")
    if isinstance(url, str) and url.startswith("data:"):
        return True

        # 检查 OpenAI 风格的 image_url 是否包含 data: URL
    image_url = block.get("image_url", {})
    if isinstance(image_url, dict):
        url = image_url.get("url", "")
        if isinstance(url, str) and url.startswith("data:"):
            return True

    return False


_XML_CONTENT_BLOCK_MAX_LEN = 500


def _truncate(text: str, max_len: int = _XML_CONTENT_BLOCK_MAX_LEN) -> str:
    """将文本截断至 `max_len` 个字符，若被截断则添加省略号。"""
    if len(text) <= max_len:
        return text
    return text[:max_len] + "..."


def _format_content_block_xml(block: dict[str, Any]) -> str | None:
    """将内容块格式化为 XML。

    Args:
        block: 一个 LangChain 内容块。

    Returns:
        XML string representation of the block, or `None` if the block should be
            skipped.

    Note:
        普通文本文档内容、服务器工具调用参数以及服务器工具输出结果均被截断至 500
        个字符。
    """
    block_type = block.get("type", "")

        # 跳过包含 base64 编码数据的块
    if _has_base64_data(block):
        return None

        # 文本块
    if block_type == "text":
        text = block.get("text", "")
        return escape(text) if text else None

        # 推理块
    if block_type == "reasoning":
        reasoning = block.get("reasoning", "")
        if reasoning:
            return f"<reasoning>{escape(reasoning)}</reasoning>"
        return None

        # 图像块（仅 URL，base64 已过滤）
    if block_type == "image":
        url = block.get("url")
        file_id = block.get("file_id")
        if url:
            return f"<image url={quoteattr(url)} />"
        if file_id:
            return f"<image file_id={quoteattr(file_id)} />"
        return None

        # OpenAI 风格的 image_url 块
    if block_type == "image_url":
        image_url = block.get("image_url", {})
        if isinstance(image_url, dict):
            url = image_url.get("url", "")
            if url and not url.startswith("data:"):
                return f"<image url={quoteattr(url)} />"
        return None

        # 音频块（仅 URL）
    if block_type == "audio":
        url = block.get("url")
        file_id = block.get("file_id")
        if url:
            return f"<audio url={quoteattr(url)} />"
        if file_id:
            return f"<audio file_id={quoteattr(file_id)} />"
        return None

        # 视频块（仅 URL）
    if block_type == "video":
        url = block.get("url")
        file_id = block.get("file_id")
        if url:
            return f"<video url={quoteattr(url)} />"
        if file_id:
            return f"<video file_id={quoteattr(file_id)} />"
        return None

        # 纯文本文档块
    if block_type == "text-plain":
        text = block.get("text", "")
        return escape(_truncate(text)) if text else None

        # 服务器工具调用块（来自 AI 消息）
    if block_type == "server_tool_call":
        tc_id = quoteattr(str(block.get("id") or ""))
        tc_name = quoteattr(str(block.get("name") or ""))
        tc_args_json = json.dumps(block.get("args", {}), ensure_ascii=False)
        tc_args = escape(_truncate(tc_args_json))
        return (
            f"<server_tool_call id={tc_id} name={tc_name}>{tc_args}</server_tool_call>"
        )

        # 服务器工具结果块
    if block_type == "server_tool_result":
        tool_call_id = quoteattr(str(block.get("tool_call_id") or ""))
        status = quoteattr(str(block.get("status") or ""))
        output = block.get("output")
        if output:
            output_json = json.dumps(output, ensure_ascii=False)
            output_str = escape(_truncate(output_json))
        else:
            output_str = ""
        return (
            f"<server_tool_result tool_call_id={tool_call_id} status={status}>"
            f"{output_str}</server_tool_result>"
        )

        # 未知块类型 - 静默跳过
    return None


def _get_message_type_str(
    m: BaseMessage,
    human_prefix: str,
    ai_prefix: str,
    system_prefix: str,
    function_prefix: str,
    tool_prefix: str,
) -> str:
    """获取 XML 消息元素的类型字符串。

    Args:
        m: 要获取类型字符串的消息。
        human_prefix: 用于 `HumanMessage` 的前缀。
        ai_prefix: 用于 `AIMessage` 的前缀。
        system_prefix: 用于 `SystemMessage` 的前缀。
        function_prefix: 用于 `FunctionMessage` 的前缀。
        tool_prefix: 用于 `ToolMessage` 的前缀。

    Returns:
        消息元素的类型字符串。

    Raises:
        ValueError: 如果遇到不支持的消息类型。
    """
    if isinstance(m, HumanMessage):
        return human_prefix.lower()
    if isinstance(m, AIMessage):
        return ai_prefix.lower()
    if isinstance(m, SystemMessage):
        return system_prefix.lower()
    if isinstance(m, FunctionMessage):
        return function_prefix.lower()
    if isinstance(m, ToolMessage):
        return tool_prefix.lower()
    if isinstance(m, ChatMessage):
        return m.role
    msg = f"Got unsupported message type: {m}"
    raise ValueError(msg)


_TOOL_CALL_ID_DISPLAY_LIMIT = 64


def _display_tool_call_id(tool_call_id: str) -> str:
    """在格式化消息字符串中缩写过长的工具调用 ID。"""
    if len(tool_call_id) > _TOOL_CALL_ID_DISPLAY_LIMIT:
        return f"{tool_call_id[:_TOOL_CALL_ID_DISPLAY_LIMIT]}..."
    return tool_call_id


def get_buffer_string(
    messages: Sequence[BaseMessage],
    human_prefix: str = "Human",
    ai_prefix: str = "AI",
    *,
    system_prefix: str = "System",
    function_prefix: str = "Function",
    tool_prefix: str = "Tool",
    message_separator: str = "\n",
    format: Literal["prefix", "xml"] = "prefix",  # noqa: A002
) -> str:
    r"""将一系列消息转换为字符串并连接成一个字符串。

    Args:
        messages: 需要转换为字符串的消息。 human_prefix: 用于 `HumanMessage`
        内容的前缀。 ai_prefix: 用于 `AIMessage` 内容的前缀。 system_prefix: 用于
        `SystemMessage` 内容的前缀。 function_prefix: 用于 `FunctionMessage`
        内容的前缀。 tool_prefix: 用于 `ToolMessage` 内容的前缀。 message_separator:
        用于分隔消息的分隔符。 format: 输出格式。`'prefix'` 使用 `Role: content`
        格式（默认）。 对于多模态消息，仅包含字符串内容和 `text`
        块；非文本块如图像、音频和视频会被忽略。

            `'xml'` uses XML-style `<message type='role'>` format with proper
            character escaping, which is useful when message content may
            contain role-like prefixes that could cause ambiguity.

            当需要对支持的多模态内容块进行结构化表示时，请使用 `'xml'`。

    Returns:
        所有输入消息的单一字符串拼接结果。

    Raises:
        ValueError: 如果遇到不支持的消息类型。

    !!! warning

        如果消息是 `AIMessage` 且同时包含 `tool_calls` 下的工具调用和
        `additional_kwargs["function_call"]`
        下的函数调用，则仅将工具调用附加到字符串表示中。 在 XML 格式中，长度超过 64
        个字符的工具调用 ID 会显示为前 64 个字符后跟 `...`；原始消息不会被更改。

    !!! note "XML format"

        当使用 `format='xml'` 时：

        - All messages use uniform `<message type="role">content</message>` format.
        - The `type` attribute uses `human_prefix` (lowercased) for `HumanMessage`,
            `ai_prefix` (lowercased) for `AIMessage`, `system_prefix` (lowercased)
            for `SystemMessage`, `function_prefix` (lowercased) for `FunctionMessage`,
            `tool_prefix` (lowercased) for `ToolMessage`, and the original role
            (unchanged) for `ChatMessage`.
        - Message content is escaped using `xml.sax.saxutils.escape()`.
        - Attribute values are escaped using `xml.sax.saxutils.quoteattr()`.
        - AI messages with tool calls use nested structure with `<content>` and
            `<tool_call>` elements.
        - For multi-modal content (list of content blocks), supported block types
            are: `text`, `reasoning`, `image` (URL/file_id only), `image_url`
            (OpenAI-style, URL only), `audio` (URL/file_id only), `video` (URL/file_id
            only), `text-plain`, `server_tool_call`, and `server_tool_result`.
        - Content blocks with base64-encoded data are skipped (including blocks
            with `base64` field or `data:` URLs).
        - Unknown block types are skipped.
        - Plain text document content (`text-plain`), server tool call arguments,
            and server tool result outputs are truncated to 500 characters.

    Example:
        默认前缀格式：

        ```python
        from langchain_core.messages import AIMessage, HumanMessage, get_buffer_string

        messages = [
            HumanMessage(content="Hi, how are you?"),
            AIMessage(content="Good, how are you?"),
        ]
        get_buffer_string(messages)
        # -> "Human: Hi, how are you?\nAI: Good, how are you?"
        ```

        XML 格式（当内容包含类似角色的前缀时有用）：

        ```python
        messages = [
            HumanMessage(content="Example: Human: some text"),
            AIMessage(content="I see the example."),
        ]
        get_buffer_string(messages, format="xml")
        # -> '<message type="human">Example: Human: some text</message>\\n'
        # -> '<message type="ai">I see the example.</message>'
        ```

        包含特殊字符的 XML 格式（自动转义）：

        ```python
        messages = [
            HumanMessage(content="Is 5 < 10 & 10 > 5?"),
        ]
        get_buffer_string(messages, format="xml")
        # -> '<message type="human">Is 5 &lt; 10 &amp; 10 &gt; 5?</message>'
        ```

        包含工具调用的 XML 格式：

        ```python
        messages = [
            AIMessage(
                content="I'll search for that.",
                tool_calls=[
                    {"id": "call_123", "name": "search", "args": {"query": "weather"}}
                ],
            ),
        ]
        get_buffer_string(messages, format="xml")
        # -> '<message type="ai">\\n'
        # -> '  <content>I\\'ll search for that.</content>\\n'
        # -> '  <tool_call id="call_123" name="search">'
        # -> '{"query": "weather"}</tool_call>\\n'
        # -> '</message>'
        ```
    """
    if format not in {"prefix", "xml"}:
        msg = (
            f"Unrecognized format={format!r}. Supported formats are 'prefix' and 'xml'."
        )
        raise ValueError(msg)

    string_messages = []
    for m in messages:
        if isinstance(m, HumanMessage):
            role = human_prefix
        elif isinstance(m, AIMessage):
            role = ai_prefix
        elif isinstance(m, SystemMessage):
            role = system_prefix
        elif isinstance(m, FunctionMessage):
            role = function_prefix
        elif isinstance(m, ToolMessage):
            role = tool_prefix
        elif isinstance(m, ChatMessage):
            role = m.role
        else:
            msg = f"Got unsupported message type: {m}"
            raise ValueError(msg)  # noqa: TRY004

        if format == "xml":
            msg_type = _get_message_type_str(
                m, human_prefix, ai_prefix, system_prefix, function_prefix, tool_prefix
            )

                        # 格式化内容块
            if isinstance(m.content, str):
                content_parts = [escape(m.content)] if m.content else []
            else:
                                # 内容块列表
                content_parts = []
                for block in m.content:
                    if isinstance(block, str):
                        if block:
                            content_parts.append(escape(block))
                    else:
                        formatted = _format_content_block_xml(block)
                        if formatted:
                            content_parts.append(formatted)

                        # 检查是否为带有工具调用的 AIMessage
            has_tool_calls = isinstance(m, AIMessage) and m.tool_calls
            has_function_call = (
                isinstance(m, AIMessage)
                and not m.tool_calls
                and "function_call" in m.additional_kwargs
            )

            if has_tool_calls or has_function_call:
                                # 为带有工具调用的 AI 消息使用嵌套结构
                                # 类型缩小：此时 m 为 AIMessage（已在上方验证）
                ai_msg = cast("AIMessage", m)
                parts = [f"<message type={quoteattr(msg_type)}>"]
                if content_parts:
                    parts.append(f"  <content>{' '.join(content_parts)}</content>")

                if has_tool_calls:
                    for tc in ai_msg.tool_calls:
                        tc_id = quoteattr(_display_tool_call_id(tc.get("id") or ""))
                        tc_name = quoteattr(str(tc.get("name") or ""))
                        tc_args = escape(
                            json.dumps(tc.get("args", {}), ensure_ascii=False)
                        )
                        parts.append(
                            f"  <tool_call id={tc_id} name={tc_name}>"
                            f"{tc_args}</tool_call>"
                        )
                elif has_function_call:
                    fc = ai_msg.additional_kwargs["function_call"]
                    fc_name = quoteattr(str(fc.get("name") or ""))
                    fc_args = escape(str(fc.get("arguments") or "{}"))
                    parts.append(
                        f"  <function_call name={fc_name}>{fc_args}</function_call>"
                    )

                parts.append("</message>")
                message = "\n".join(parts)
            else:
                                # 无工具调用的消息的简单结构
                joined_content = " ".join(content_parts)
                message = (
                    f"<message type={quoteattr(msg_type)}>{joined_content}</message>"
                )
        else:  # format == "prefix"
            content = m.text
            message = f"{role}: {content}"
            tool_info = ""
            if isinstance(m, AIMessage):
                if m.tool_calls:
                    tool_info = str(m.tool_calls)
                elif "function_call" in m.additional_kwargs:
                                        # 旧版行为假设每条消息仅有一个函数调用
                    tool_info = str(m.additional_kwargs["function_call"])
            if tool_info:
                message += tool_info  # 保留原始行为

        string_messages.append(message)

    return message_separator.join(string_messages)


def _message_from_dict(message: dict[str, Any]) -> BaseMessage:
    type_ = message["type"]
    if type_ == "human":
        return HumanMessage(**message["data"])
    if type_ == "ai":
        return AIMessage(**message["data"])
    if type_ == "system":
        return SystemMessage(**message["data"])
    if type_ == "chat":
        return ChatMessage(**message["data"])
    if type_ == "function":
        return FunctionMessage(**message["data"])
    if type_ == "tool":
        return ToolMessage(**message["data"])
    if type_ == "remove":
        return RemoveMessage(**message["data"])
    if type_ == "AIMessageChunk":
        return AIMessageChunk(**message["data"])
    if type_ == "HumanMessageChunk":
        return HumanMessageChunk(**message["data"])
    if type_ == "FunctionMessageChunk":
        return FunctionMessageChunk(**message["data"])
    if type_ == "ToolMessageChunk":
        return ToolMessageChunk(**message["data"])
    if type_ == "SystemMessageChunk":
        return SystemMessageChunk(**message["data"])
    if type_ == "ChatMessageChunk":
        return ChatMessageChunk(**message["data"])
    msg = f"Got unexpected message type: {type_}"
    raise ValueError(msg)


def messages_from_dict(messages: Sequence[dict[str, Any]]) -> list[BaseMessage]:
    """将消息序列从字典转换为 `Message` 对象。

    Args:
        messages: 要转换的消息序列（作为字典）。

    Returns:
        消息列表（BaseMessages）。

    """
    return [_message_from_dict(m) for m in messages]


def message_chunk_to_message(chunk: BaseMessage) -> BaseMessage:
    """将消息块转换为 `Message`。

    Args:
        chunk: 要转换的消息块。

    Returns:
        Message。
    """
    if not isinstance(chunk, BaseMessageChunk):
        return chunk
        # chunk 类始终以其对应的非 chunk 类作为第一个父类
    ignore_keys = ["type"]
    if isinstance(chunk, AIMessageChunk):
        ignore_keys.extend(["tool_call_chunks", "chunk_position"])
    return cast(
        "BaseMessage",
        chunk.__class__.__mro__[1](
            **{k: v for k, v in chunk.__dict__.items() if k not in ignore_keys}
        ),
    )


MessageLikeRepresentation = (
    BaseMessage
    | list[str]
    | tuple[str, str | list[str | dict[str, Any]]]
    | str
    | dict[str, Any]
)
"""A type representing the various ways a message can be represented."""


def _create_message_from_message_type(
    message_type: str,
    content: str | list[str | dict[str, Any]],
    name: str | None = None,
    tool_call_id: str | None = None,
    tool_calls: list[dict[str, Any]] | None = None,
    id: str | None = None,
    **additional_kwargs: Any,
) -> BaseMessage:
    """从 `Message` 类型和内容字符串创建一条消息。

    Args:
        message_type: 消息类型（例如 `'human'`、`'ai'` 等）。
        content: 内容字符串。
        name: 消息名称。
        tool_call_id: 工具调用 ID。
        tool_calls: 工具调用。
        id: 消息 ID。
        additional_kwargs: 额外关键字参数。

    Returns:
        相应类型的 message。

    Raises:
        ValueError: 如果消息类型不是 `'human'`、`'user'`、`'ai'`、
            `'assistant'`、`'function'`、`'tool'`、`'system'` 或
            `'developer'` 之一。
    """
    kwargs: dict[str, Any] = {}
    if name is not None:
        kwargs["name"] = name
    if tool_call_id is not None:
        kwargs["tool_call_id"] = tool_call_id
    if additional_kwargs:
        if response_metadata := additional_kwargs.pop("response_metadata", None):
            kwargs["response_metadata"] = response_metadata
        kwargs["additional_kwargs"] = additional_kwargs
        additional_kwargs.update(additional_kwargs.pop("additional_kwargs", {}))
    if id is not None:
        kwargs["id"] = id
    if tool_calls is not None:
        kwargs["tool_calls"] = []
        for tool_call in tool_calls:
                        # 将 OpenAI 格式的工具调用转换为 LangChain 格式。
            if "function" in tool_call:
                args = tool_call["function"]["arguments"]
                if isinstance(args, str):
                    args = json.loads(args, strict=False)
                kwargs["tool_calls"].append(
                    {
                        "name": tool_call["function"]["name"],
                        "args": args,
                        "id": tool_call["id"],
                        "type": "tool_call",
                    }
                )
            else:
                kwargs["tool_calls"].append(tool_call)
    if message_type in {"human", "user"}:
        if example := kwargs.get("additional_kwargs", {}).pop("example", False):
            kwargs["example"] = example
        message: BaseMessage = HumanMessage(content=content, **kwargs)
    elif message_type in {"ai", "assistant"}:
        if example := kwargs.get("additional_kwargs", {}).pop("example", False):
            kwargs["example"] = example
        message = AIMessage(content=content, **kwargs)
    elif message_type in {"system", "developer"}:
        if message_type == "developer":
            kwargs["additional_kwargs"] = kwargs.get("additional_kwargs") or {}
            kwargs["additional_kwargs"]["__openai_role__"] = "developer"
        message = SystemMessage(content=content, **kwargs)
    elif message_type == "function":
        message = FunctionMessage(content=content, **kwargs)
    elif message_type == "tool":
        artifact = kwargs.get("additional_kwargs", {}).pop("artifact", None)
        status = kwargs.get("additional_kwargs", {}).pop("status", None)
        if status is not None:
            kwargs["status"] = status
        message = ToolMessage(content=content, artifact=artifact, **kwargs)
    elif message_type == "remove":
        message = RemoveMessage(**kwargs)
    else:
        msg = (
            f"Unexpected message type: '{message_type}'. Use one of 'human',"
            f" 'user', 'ai', 'assistant', 'function', 'tool', 'system', or 'developer'."
        )
        msg = create_message(message=msg, error_code=ErrorCode.MESSAGE_COERCION_FAILURE)
        raise ValueError(msg)
    return message


# 在 `Serializable` 构造函数封装中发出的类名映射
# (`{"lc": 1, "type": "constructor", "id": [..., "<ClassName>"],
# "kwargs": {...}}`) 映射到消息类型字符串
# `_create_message_from_message_type` 接受。由
# `_convert_to_message` 的字典分支在展开该线形结构时读取。
# 保持为硬编码的允许列表字符串，而非类注册表
# 查找，以确保分派从不解析为由调用者选择的类。
_LC_CONSTRUCTOR_NAME_TO_TYPE: dict[str, str] = {
    "HumanMessage": "human",
    "HumanMessageChunk": "human",
    "AIMessage": "ai",
    "AIMessageChunk": "ai",
    "SystemMessage": "system",
    "SystemMessageChunk": "system",
    "FunctionMessage": "function",
    "FunctionMessageChunk": "function",
    "ToolMessage": "tool",
    "ToolMessageChunk": "tool",
    "RemoveMessage": "remove",
}


def _convert_to_message(message: MessageLikeRepresentation) -> BaseMessage:
    """从多种消息格式实例化一个 `Message`。

    消息格式可以是以下之一：

    - `BaseMessagePromptTemplate`
    - `BaseMessage`
    - 2-tuple of (role string, template); e.g., (`'human'`, `'{user_input}'`)
    - dict: a message dict with role and content keys
    - dict: the `Serializable` constructor-envelope wire shape
        `{"lc": 1, "type": "constructor", "id": [..., "<ClassName>"],
        "kwargs": {...}}` — unpacked structurally and routed through the
        standard dict-with-type dispatch.
    - string: shorthand for (`'human'`, template); e.g., `'{user_input}'`

    Args:
        message: 一种支持格式中的消息表示。

    Returns:
        消息实例或消息模板。

    Raises:
        NotImplementedError: 如果消息类型不受支持。
        ValueError: 如果消息字典不包含必需的键。

    """
    if isinstance(message, BaseMessage):
        message_ = message
    elif isinstance(message, Sequence):
        if isinstance(message, str):
            message_ = _create_message_from_message_type("human", message)
        else:
            try:
                message_type_str, template = message
            except ValueError as e:
                msg = "Message as a sequence must be (role string, template)"
                raise NotImplementedError(msg) from e
            message_ = _create_message_from_message_type(message_type_str, template)
    elif isinstance(message, dict):
                # `Serializable` 构造函数封装的线形结构。通过结构检测，通过硬编码
                # 允许列表将类名映射为已知的消息类型字符串，并递归使用标准
                # `{"type": ..., **kwargs}` 形式 — 无 `load()`，无动态
                # 类实例化。
                # 不允许消息内容为 None
        if (
            message.get("lc") == 1
            and message.get("type") == "constructor"
            and isinstance(message.get("id"), list)
            and message["id"]
            and isinstance(message.get("kwargs"), dict)
        ):
            mapped = _LC_CONSTRUCTOR_NAME_TO_TYPE.get(message["id"][-1])
            if mapped is not None:
                return _convert_to_message({"type": mapped, **message["kwargs"]})

        msg_kwargs = message.copy()
        try:
            try:
                msg_type = msg_kwargs.pop("role")
            except KeyError:
                msg_type = msg_kwargs.pop("type")
                        # 在此处导入以避免循环导入
            msg_content = msg_kwargs.pop("content") or ""
        except KeyError as e:
            msg = f"Message dict must contain 'role' and 'content' keys, got {message}"
            msg = create_message(
                message=msg, error_code=ErrorCode.MESSAGE_COERCION_FAILURE
            )
            raise ValueError(msg) from e
        message_ = _create_message_from_message_type(
            msg_type, msg_content, **msg_kwargs
        )
    else:
        msg = f"Unsupported message type: {type(message)}"  # type: ignore[unreachable]
        msg = create_message(message=msg, error_code=ErrorCode.MESSAGE_COERCION_FAILURE)
        raise NotImplementedError(msg)

    return message_


def convert_to_messages(
    messages: Iterable[MessageLikeRepresentation] | PromptValue,
) -> list[BaseMessage]:
    """将消息序列转换为消息列表。

    Args:
        messages: 要转换的消息序列。

    Returns:
        消息列表（BaseMessages）。

    """
        # 本地导入以防止循环导入。
    from langchain_core.prompt_values import PromptValue  # noqa: PLC0415

    if isinstance(messages, PromptValue):
        return messages.to_messages()
    return [_convert_to_message(m) for m in messages]


_P = ParamSpec("_P")
_R_co = TypeVar("_R_co", covariant=True)


class _RunnableSupportCallable(Protocol[_P, _R_co]):
    @overload
    def __call__(
        self,
        messages: None = None,
        *args: _P.args,
        **kwargs: _P.kwargs,
    ) -> Runnable[Sequence[MessageLikeRepresentation], _R_co]: ...

    @overload
    def __call__(
        self,
        messages: Sequence[MessageLikeRepresentation] | PromptValue,
        *args: _P.args,
        **kwargs: _P.kwargs,
    ) -> _R_co: ...

    def __call__(
        self,
        messages: Sequence[MessageLikeRepresentation] | PromptValue | None = None,
        *args: _P.args,
        **kwargs: _P.kwargs,
    ) -> _R_co | Runnable[Sequence[MessageLikeRepresentation], _R_co]: ...


def _runnable_support(
    func: Callable[
        Concatenate[Sequence[MessageLikeRepresentation] | PromptValue, _P], _R_co
    ],
) -> _RunnableSupportCallable[_P, _R_co]:
    @wraps(func)
    def wrapped(
        messages: Sequence[MessageLikeRepresentation] | PromptValue | None = None,
        *args: _P.args,
        **kwargs: _P.kwargs,
    ) -> _R_co | Runnable[Sequence[MessageLikeRepresentation], _R_co]:
                # 处理 Anthropic 内容块
        from langchain_core.runnables.base import RunnableLambda  # noqa: PLC0415

        if messages is not None:
            return func(messages, *args, **kwargs)
        return RunnableLambda(partial(func, **kwargs), name=func.__name__)

    return cast("_RunnableSupportCallable[_P, _R_co]", wrapped)


@_runnable_support
def filter_messages(
    messages: Iterable[MessageLikeRepresentation] | PromptValue,
    *,
    include_names: Sequence[str] | None = None,
    exclude_names: Sequence[str] | None = None,
    include_types: Sequence[str | type[BaseMessage]] | None = None,
    exclude_types: Sequence[str | type[BaseMessage]] | None = None,
    include_ids: Sequence[str] | None = None,
    exclude_ids: Sequence[str] | None = None,
    exclude_tool_calls: Sequence[str] | bool | None = None,
) -> list[BaseMessage]:
    """根据 `name`、`type` 或 `id` 过滤消息。

    Args:
        messages: Sequence Message-like objects to filter.
        include_names: Message names to include.
        exclude_names: Messages names to exclude.
        include_types: Message types to include. Can be specified as string names
            (e.g. `'system'`, `'human'`, `'ai'`, ...) or as `BaseMessage`
            classes (e.g. `SystemMessage`, `HumanMessage`, `AIMessage`, ...).

        exclude_types: Message types to exclude. Can be specified as string names
            (e.g. `'system'`, `'human'`, `'ai'`, ...) or as `BaseMessage`
            classes (e.g. `SystemMessage`, `HumanMessage`, `AIMessage`, ...).

        include_ids: Message IDs to include.
        exclude_ids: Message IDs to exclude.
        exclude_tool_calls: Tool call IDs to exclude.
            Can be one of the following:
            - `True`: All `AIMessage` objects with tool calls and all `ToolMessage`
                objects will be excluded.
            - a sequence of tool call IDs to exclude:
                - `ToolMessage` objects with the corresponding tool call ID will be
                    excluded.
                - The `tool_calls` in the AIMessage will be updated to exclude
                    matching tool calls. If all `tool_calls` are filtered from an
                    AIMessage, the whole message is excluded.

    Returns:
        合并相同类型的连续消息。

    Raises:
        !!! note `ToolMessage` 对象不会被合并，因为每个对象都有唯一的工具调用
        ID，无法合并。

    Example:
        ```python
        from langchain_core.messages import (
            filter_messages,
            AIMessage,
            HumanMessage,
            SystemMessage,
        )

        messages = [
            SystemMessage("you're a good assistant."),
            HumanMessage("what's your name", id="foo", name="example_user"),
            AIMessage("steve-o", id="bar", name="example_assistant"),
            HumanMessage(
                "what's your favorite color",
                id="baz",
            ),
            AIMessage(
                "silicon blue",
                id="blah",
            ),
        ]

        filter_messages(
            messages,
            include_names=("example_user", "example_assistant"),
            include_types=("system",),
            exclude_ids=("bar",),
        )
        ```

        ```python
        [
            SystemMessage("you're a good assistant."),
            HumanMessage("what's your name", id="foo", name="example_user"),
        ]
        ```
    """
    messages = convert_to_messages(messages)
    filtered: list[BaseMessage] = []
    for msg in messages:
        if (
            (exclude_names and msg.name in exclude_names)
            or (exclude_types and _is_message_type(msg, exclude_types))
            or (exclude_ids and msg.id in exclude_ids)
        ):
            continue

        if exclude_tool_calls is True and (
            (isinstance(msg, AIMessage) and msg.tool_calls)
            or isinstance(msg, ToolMessage)
        ):
            continue

        new_msg = msg
        if isinstance(exclude_tool_calls, (list, tuple, set)):
            if isinstance(msg, AIMessage) and msg.tool_calls:
                tool_calls = [
                    tool_call
                    for tool_call in msg.tool_calls
                    if tool_call["id"] not in exclude_tool_calls
                ]
                if not tool_calls:
                    continue

                content = msg.content
                                # 当未提供包含条件时，默认包含
                if isinstance(msg.content, list):
                    content = [
                        content_block
                        for content_block in msg.content
                        if (
                            not isinstance(content_block, dict)
                            or content_block.get("type") != "tool_use"
                            or content_block.get("id") not in exclude_tool_calls
                        )
                    ]

                new_msg = msg.model_copy(
                    update={"tool_calls": tool_calls, "content": content}
                )
            elif (
                isinstance(msg, ToolMessage) and msg.tool_call_id in exclude_tool_calls
            ):
                continue

                # TODO: 更新以使验证错误（例如 token_counter）在初始化阶段而非运行时抛出。
        if (
            not (include_types or include_ids or include_names)
            or (include_names and new_msg.name in include_names)
            or (include_types and _is_message_type(new_msg, include_types))
            or (include_ids and new_msg.id in include_ids)
        ):
            filtered.append(new_msg)

    return filtered


@_runnable_support
def merge_message_runs(
    messages: Iterable[MessageLikeRepresentation] | PromptValue,
    *,
    chunk_separator: str = "\n",
) -> list[BaseMessage]:
    r"""messages: 要合并的消息类对象序列。 chunk_separator: 指定插入消息块之间的字符串。

    !!! note `ToolMessage` 对象不会被合并，因为每个对象都有唯一的工具调用 ID，无法合并。

    Args:
        messages: 要合并的消息对象序列。
        chunk_separator: 指定插入消息块之间的字符串。

    Returns:
        list of BaseMessages with consecutive runs of message types merged into single
        messages. By default, if two messages being merged both have string contents,
        the merged content is a concatenation of the two strings with a new-line
        separator.
        The separator inserted between message chunks can be controlled by specifying
        any string with `chunk_separator`. If at least one of the messages has a list
        of content blocks, the merged content is a list of content blocks.

    Example:
        ```python
        from langchain_core.messages import (
            merge_message_runs,
            AIMessage,
            HumanMessage,
            SystemMessage,
            ToolCall,
        )

        messages = [
            SystemMessage("you're a good assistant."),
            HumanMessage(
                "what's your favorite color",
                id="foo",
            ),
            HumanMessage(
                "wait your favorite food",
                id="bar",
            ),
            AIMessage(
                "my favorite colo",
                tool_calls=[
                    ToolCall(
                        name="blah_tool", args={"x": 2}, id="123", type="tool_call"
                    )
                ],
                id="baz",
            ),
            AIMessage(
                [{"type": "text", "text": "my favorite dish is lasagna"}],
                tool_calls=[
                    ToolCall(
                        name="blah_tool",
                        args={"x": -10},
                        id="456",
                        type="tool_call",
                    )
                ],
                id="blur",
            ),
        ]

        merge_message_runs(messages)
        ```

        ```python
        [
            SystemMessage("you're a good assistant."),
            HumanMessage(
                "what's your favorite color\\n"
                "wait your favorite food", id="foo",
            ),
            AIMessage(
                [
                    "my favorite colo",
                    {"type": "text", "text": "my favorite dish is lasagna"}
                ],
                tool_calls=[
                    ToolCall({
                        "name": "blah_tool",
                        "args": {"x": 2},
                        "id": "123",
                        "type": "tool_call"
                    }),
                    ToolCall({
                        "name": "blah_tool",
                        "args": {"x": -10},
                        "id": "456",
                        "type": "tool_call"
                    })
                ]
                id="baz"
            ),
        ]

        ```
    """
    if not messages:
        return []
    messages = convert_to_messages(messages)
    merged: list[BaseMessage] = []
    for msg in messages:
        last = merged.pop() if merged else None
        if not last:
            merged.append(msg)
        elif isinstance(msg, ToolMessage) or not isinstance(msg, last.__class__):
            merged.extend([last, msg])
        else:
            last_chunk = _msg_to_chunk(last)
            curr_chunk = _msg_to_chunk(msg)
            if curr_chunk.response_metadata:
                curr_chunk.response_metadata.clear()
            if (
                isinstance(last_chunk.content, str)
                and isinstance(curr_chunk.content, str)
                and last_chunk.content
                and curr_chunk.content
            ):
                last_chunk.content += chunk_separator
            merged.append(_chunk_to_msg(last_chunk + curr_chunk))
    return merged


# 验证参数
# 处理 token counter 的字符串快捷方式
@_runnable_support
def trim_messages(
    messages: Iterable[MessageLikeRepresentation] | PromptValue,
    *,
    max_tokens: int,
    token_counter: Callable[[list[BaseMessage]], int]
    | Callable[[BaseMessage], int]
    | BaseLanguageModel[Any]
    | Literal["approximate"],
    strategy: Literal["first", "last"] = "last",
    allow_partial: bool = False,
    end_on: str | type[BaseMessage] | Sequence[str | type[BaseMessage]] | None = None,
    start_on: str | type[BaseMessage] | Sequence[str | type[BaseMessage]] | None = None,
    include_system: bool = False,
    text_splitter: Callable[[str], list[str]] | TextSplitter | None = None,
) -> list[BaseMessage]:
    r"""将消息修剪至低于指定 token 数量。

    `trim_messages` 可用于将聊天历史缩减到指定的 token 或消息数量。

    In either case, if passing the trimmed chat history back into a chat model
    directly, the resulting chat history should usually satisfy the following
    properties:

    1. The resulting chat history should be valid. Most chat models expect that chat
        history starts with either (1) a `HumanMessage` or (2) a `SystemMessage`
        followed by a `HumanMessage`. To achieve this, set `start_on='human'`.
        In addition, generally a `ToolMessage` can only appear after an `AIMessage`
        that involved a tool call.
    2. It includes recent messages and drops old messages in the chat history.
        To achieve this set the `strategy='last'`.
    3. Usually, the new chat history should include the `SystemMessage` if it
        was present in the original chat history since the `SystemMessage` includes
        special instructions to the chat model. The `SystemMessage` is almost always
        the first message in the history if present. To achieve this set the
        `include_system=True`.

    !!! note
        The examples below show how to configure `trim_messages` to achieve a behavior
        consistent with the above properties.

    Args:
        messages: Sequence of Message-like objects to trim.
        max_tokens: Max token count of trimmed messages.
        token_counter: Function or llm for counting tokens in a `BaseMessage` or a
            list of `BaseMessage`.

            如果传入 `BaseLanguageModel`，则使用
            `BaseLanguageModel.get_num_tokens_from_messages()`。 设置为 `len`
            以统计聊天历史中的 **消息数量**。

            您也可以使用字符串快捷方式以方便使用：

            - `'approximate'`: Uses `count_tokens_approximately` for fast, approximate
                token counts.

            !!! note

                `count_tokens_approximately` (or the shortcut `'approximate'`) is
                recommended for using `trim_messages` on the hot path, where exact token
                counting is not necessary.

        strategy: 修剪策略。

            - `'first'`: Keep the first `<= n_count` tokens of the messages.
            - `'last'`: Keep the last `<= n_count` tokens of the messages.
        allow_partial: Whether to split a message if only part of the message can be
            included.

            If `strategy='last'` then the last partial contents of a message are
            included. If `strategy='first'` then the first partial contents of a
            message are included.
        end_on: The message type to end on.

            If specified then every message after the last occurrence of this type is
            ignored. If `strategy='last'` then this is done before we attempt to get the
            last `max_tokens`. If `strategy='first'` then this is done after we get the
            first `max_tokens`. Can be specified as string names (e.g. `'system'`,
            `'human'`, `'ai'`, ...) or as `BaseMessage` classes (e.g. `SystemMessage`,
            `HumanMessage`, `AIMessage`, ...). Can be a single type or a list of types.

        start_on: 开始的消息类型。

            Should only be specified if `strategy='last'`. If specified then every
            message before the first occurrence of this type is ignored. This is done
            after we trim the initial messages to the last `max_tokens`. Does not apply
            to a `SystemMessage` at index 0 if `include_system=True`. Can be specified
            as string names (e.g. `'system'`, `'human'`, `'ai'`, ...) or as
            `BaseMessage` classes (e.g. `SystemMessage`, `HumanMessage`, `AIMessage`,
            ...). Can be a single type or a list of types.

        include_system: Whether to keep the `SystemMessage` if there is one at index
            `0`.

            Should only be specified if `strategy="last"`.
        text_splitter: Function or `langchain_text_splitters.TextSplitter` for
            splitting the string contents of a message.

            Only used if `allow_partial=True`. If `strategy='last'` then the last split
            tokens from a partial message will be included. if `strategy='first'` then
            the first split tokens from a partial message will be included. Token
            splitter assumes that separators are kept, so that split contents can be
            directly concatenated to recreate the original text. Defaults to splitting
            on newlines.

    Returns:
        修剪后的 `BaseMessage` 列表。

    Raises:
        ValueError: if two incompatible arguments are specified or an unrecognized
            `strategy` is specified.

    Example:
        Trim chat history based on token count, keeping the `SystemMessage` if
        present, and ensuring that the chat history starts with a `HumanMessage` (or a
        `SystemMessage` followed by a `HumanMessage`).

        ```python
        from langchain_core.messages import (
            AIMessage,
            HumanMessage,
            BaseMessage,
            SystemMessage,
            trim_messages,
        )

        messages = [
            SystemMessage("you're a good assistant, you always respond with a joke."),
            HumanMessage("i wonder why it's called langchain"),
            AIMessage(
                'Well, I guess they thought "WordRope" and "SentenceString" just '
                "didn't have the same ring to it!"
            ),
            HumanMessage("and who is harrison chasing anyways"),
            AIMessage(
                "Hmmm let me think.\n\nWhy, he's probably chasing after the last "
                "cup of coffee in the office!"
            ),
            HumanMessage("what do you call a speechless parrot"),
        ]


        trim_messages(
            messages,
            max_tokens=45,
            strategy="last",
            token_counter=ChatOpenAI(model="gpt-5.5"),
            # Most chat models expect that chat history starts with either:
            # (1) a HumanMessage or
            # (2) a SystemMessage followed by a HumanMessage
            start_on="human",
            # Usually, we want to keep the SystemMessage
            # if it's present in the original history.
            # The SystemMessage has special instructions for the model.
            include_system=True,
            allow_partial=False,
        )
        ```

        ```python
        [
            SystemMessage(
                content="you're a good assistant, you always respond with a joke."
            ),
            HumanMessage(content="what do you call a speechless parrot"),
        ]
        ```

        使用近似 token 计数方法 `'approximate'` 修剪聊天历史：

        ```python
        trim_messages(
            messages,
            max_tokens=45,
            strategy="last",
            # Using the "approximate" shortcut for fast token counting
            token_counter="approximate",
            start_on="human",
            include_system=True,
        )

        # This is equivalent to using `count_tokens_approximately` directly
        from langchain_core.messages.utils import count_tokens_approximately

        trim_messages(
            messages,
            max_tokens=45,
            strategy="last",
            token_counter=count_tokens_approximately,
            start_on="human",
            include_system=True,
        )
        ```

        Trim chat history based on the message count, keeping the `SystemMessage` if
        present, and ensuring that the chat history starts with a HumanMessage (
        or a `SystemMessage` followed by a `HumanMessage`).

            trim_messages(
                messages,
                # 当传递 `len` 作为 token 计数函数时，
                # max_tokens 将计算聊天历史中的消息数量。
                max_tokens=4,
                strategy="last",
                # 传递 `len` 作为 token 计数函数将
                # 计算聊天历史中的消息数量。
                token_counter=len,
                # 大多数聊天模型期望聊天历史以以下之一开始：
                # (1) HumanMessage 或
                # (2) SystemMessage 后接 HumanMessage
                start_on="human",
                # 通常，如果原始历史中存在 SystemMessage，
                # 我们希望保留它。
                # SystemMessage 包含模型的特殊指令。
                include_system=True,
                allow_partial=False,
            )

        ```python
        [
            SystemMessage(
                content="you're a good assistant, you always respond with a joke."
            ),
            HumanMessage(content="and who is harrison chasing anyways"),
            AIMessage(
                content="Hmmm let me think.\n\nWhy, he's probably chasing after "
                "the last cup of coffee in the office!"
            ),
            HumanMessage(content="what do you call a speechless parrot"),
        ]
        ```
        使用自定义 token 计数函数修剪聊天历史，该函数统计每条消息中的 token 数量。

        ```python
        messages = [
            SystemMessage("This is a 4 token text. The full message is 10 tokens."),
            HumanMessage(
                "This is a 4 token text. The full message is 10 tokens.", id="first"
            ),
            AIMessage(
                [
                    {"type": "text", "text": "This is the FIRST 4 token block."},
                    {"type": "text", "text": "This is the SECOND 4 token block."},
                ],
                id="second",
            ),
            HumanMessage(
                "This is a 4 token text. The full message is 10 tokens.", id="third"
            ),
            AIMessage(
                "This is a 4 token text. The full message is 10 tokens.",
                id="fourth",
            ),
        ]


        def dummy_token_counter(messages: list[BaseMessage]) -> int:
            # treat each message like it adds 3 default tokens at the beginning
            # of the message and at the end of the message. 3 + 4 + 3 = 10 tokens
            # per message.

            default_content_len = 4
            default_msg_prefix_len = 3
            default_msg_suffix_len = 3

            count = 0
            for msg in messages:
                if isinstance(msg.content, str):
                    count += (
                        default_msg_prefix_len
                        + default_content_len
                        + default_msg_suffix_len
                    )
                if isinstance(msg.content, list):
                    count += (
                        default_msg_prefix_len
                        + len(msg.content) * default_content_len
                        + default_msg_suffix_len
                    )
            return count
        ```

        前 30 个 token，允许部分消息：
        ```python
        trim_messages(
            messages,
            max_tokens=30,
            token_counter=dummy_token_counter,
            strategy="first",
            allow_partial=True,
        )
        ```

        ```python
        [
            SystemMessage("This is a 4 token text. The full message is 10 tokens."),
            HumanMessage(
                "This is a 4 token text. The full message is 10 tokens.",
                id="first",
            ),
            AIMessage(
                [{"type": "text", "text": "This is the FIRST 4 token block."}],
                id="second",
            ),
        ]
        ```
    """
        # 类型缩小：此时 token_counter 不是 str
    if start_on and strategy == "first":
        msg = "start_on parameter is only valid with strategy='last'"
        raise ValueError(msg)
    if include_system and strategy == "first":
        msg = "include_system parameter is only valid with strategy='last'"
        raise ValueError(msg)

    messages = convert_to_messages(messages)

        # 一系列 _SingleMessage，且不是裸 str
    if isinstance(token_counter, str):
        if token_counter in _TOKEN_COUNTER_SHORTCUTS:
            actual_token_counter = _TOKEN_COUNTER_SHORTCUTS[token_counter]
        else:
            available_shortcuts = ", ".join(
                f"'{key}'" for key in _TOKEN_COUNTER_SHORTCUTS
            )
            msg = (
                f"Invalid token_counter shortcut '{token_counter}'. "
                f"Available shortcuts: {available_shortcuts}."
            )
            raise ValueError(msg)
    else:
                # OpenAI 格式
        actual_token_counter = token_counter  # type: ignore[assignment]

    if hasattr(actual_token_counter, "get_num_tokens_from_messages"):
        list_token_counter = actual_token_counter.get_num_tokens_from_messages
    elif callable(actual_token_counter):
        if (
            next(
                iter(inspect.signature(actual_token_counter).parameters.values())
            ).annotation
            is BaseMessage
        ):

            def list_token_counter(messages: Sequence[BaseMessage]) -> int:
                return sum(actual_token_counter(msg) for msg in messages)  # type: ignore[arg-type, misc]

        else:
            list_token_counter = actual_token_counter
    else:
        msg = (  # type: ignore[unreachable]
            f"'token_counter' expected to be a model that implements "
            f"'get_num_tokens_from_messages()' or a function. Received object of type "
            f"{type(actual_token_counter)}."
        )
        raise ValueError(msg)

    text_splitter_fn: Callable[[str], list[str]]
    if _HAS_LANGCHAIN_TEXT_SPLITTERS and isinstance(text_splitter, TextSplitter):
        text_splitter_fn = text_splitter.split_text
    elif text_splitter:
        text_splitter_fn = cast("Callable[[str], list[str]]", text_splitter)
    else:
        text_splitter_fn = _default_text_splitter

    if strategy == "first":
        return _first_max_tokens(
            messages,
            max_tokens=max_tokens,
            token_counter=list_token_counter,
            text_splitter=text_splitter_fn,
            partial_strategy="first" if allow_partial else None,
            end_on=end_on,
        )
    if strategy == "last":
        return _last_max_tokens(
            messages,
            max_tokens=max_tokens,
            token_counter=list_token_counter,
            allow_partial=allow_partial,
            include_system=include_system,
            start_on=start_on,
            end_on=end_on,
            text_splitter=text_splitter_fn,
        )
    msg = f"Unrecognized {strategy=}. Supported strategies are 'last' and 'first'."  # type: ignore[unreachable]
    raise ValueError(msg)


_SingleMessage = BaseMessage | str | dict[str, Any]
_T = TypeVar("_T", bound=_SingleMessage)
# 标准多模态内容块
_MultipleMessages = Sequence[_T]


@overload
def convert_to_openai_messages(
    messages: _SingleMessage,
    *,
    text_format: Literal["string", "block"] = "string",
    include_id: bool = False,
    pass_through_unknown_blocks: bool = True,
) -> dict[str, Any]: ...


@overload
def convert_to_openai_messages(
    messages: _MultipleMessages[Any],
    *,
    text_format: Literal["string", "block"] = "string",
    include_id: bool = False,
    pass_through_unknown_blocks: bool = True,
) -> list[dict[str, Any]]: ...


def convert_to_openai_messages(
    messages: MessageLikeRepresentation | Sequence[MessageLikeRepresentation],
    *,
    text_format: Literal["string", "block"] = "string",
    include_id: bool = False,
    pass_through_unknown_blocks: bool = True,
) -> dict[str, Any] | list[dict[str, Any]]:
    """将 LangChain 消息转换为 OpenAI 消息字典。

    Args:
        messages: Message-like object or iterable of objects whose contents are
            in OpenAI, Anthropic, Bedrock Converse, or VertexAI formats.
        text_format: How to format string or text block contents:
            - `'string'`:
                If a message has a string content, this is left as a string. If
                a message has content blocks that are all of type `'text'`, these
                are joined with a newline to make a single string. If a message has
                content blocks and at least one isn't of type `'text'`, then
                all blocks are left as dicts.
            - `'block'`:
                If a message has a string content, this is turned into a list
                with a single content block of type `'text'`. If a message has
                content blocks these are left as is.
        include_id: Whether to include message IDs in the openai messages, if they
            are present in the source messages.
        pass_through_unknown_blocks: Whether to include content blocks with unknown
            formats in the output. If `False`, an error is raised if an unknown
            content block is encountered.

    Raises:
        ValueError: if an unrecognized `text_format` is specified, or if a message
            content block is missing expected keys.

    Returns:
        返回类型取决于输入类型：

        - dict: 如果传入单个 message 类型对象，则返回单个 OpenAI 消息字典。 -
        list[dict]: 如果传入多个 message 类型对象的序列，则返回 OpenAI 消息字典列表。

    Example:
        ```python
        from langchain_core.messages import (
            convert_to_openai_messages,
            AIMessage,
            SystemMessage,
            ToolMessage,
        )

        messages = [
            SystemMessage([{"type": "text", "text": "foo"}]),
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "what's in this"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64,'/9j/4AAQSk'"},
                    },
                ],
            },
            AIMessage(
                "",
                tool_calls=[
                    {
                        "name": "analyze",
                        "args": {"baz": "buz"},
                        "id": "1",
                        "type": "tool_call",
                    }
                ],
            ),
            ToolMessage("foobar", tool_call_id="1", name="bar"),
            {"role": "assistant", "content": "that's nice"},
        ]
        oai_messages = convert_to_openai_messages(messages)
        # -> [
        #   {'role': 'system', 'content': 'foo'},
        #   {'role': 'user', 'content': [{'type': 'text', 'text': 'what's in this'}, {'type': 'image_url', 'image_url': {'url': "data:image/png;base64,'/9j/4AAQSk'"}}]},
        #   {'role': 'assistant', 'tool_calls': [{'type': 'function', 'id': '1','function': {'name': 'analyze', 'arguments': '{"baz": "buz"}'}}], 'content': ''},
        #   {'role': 'tool', 'name': 'bar', 'content': 'foobar'},
        #   {'role': 'assistant', 'content': 'that's nice'}
        # ]
        ```

    !!! version-added "Added in `langchain-core` 0.3.11"

    """  # noqa: E501
    if text_format not in {"string", "block"}:
        err = f"Unrecognized {text_format=}, expected one of 'string' or 'block'."
        raise ValueError(err)

    oai_messages: list[dict[str, Any]] = []

    messages_: Sequence[MessageLikeRepresentation]
    if is_single := isinstance(messages, (BaseMessage, dict, str)):
        messages_ = [messages]
    else:
        messages_ = cast("Sequence[MessageLikeRepresentation]", messages)

    for i, message in enumerate(convert_to_messages(messages_)):
        oai_msg: dict[str, Any] = {"role": _get_message_openai_role(message)}
        tool_messages: list[dict[str, Any]] = []
        content: str | list[dict[str, Any]]

        if message.name:
            oai_msg["name"] = message.name
        if isinstance(message, AIMessage) and message.tool_calls:
            oai_msg["tool_calls"] = _convert_to_openai_tool_calls(message.tool_calls)
        if message.additional_kwargs.get("refusal"):
            oai_msg["refusal"] = message.additional_kwargs["refusal"]
        if isinstance(message, ToolMessage):
            oai_msg["tool_call_id"] = message.tool_call_id
        if include_id and message.id:
            oai_msg["id"] = message.id

        if not message.content:
            content = "" if text_format == "string" else []
        elif isinstance(message.content, str):
            if text_format == "string":
                content = message.content
            else:
                content = [{"type": "text", "text": message.content}]
        elif text_format == "string" and all(
            isinstance(block, str) or block.get("type") == "text"
            for block in message.content
        ):
            content = "\n".join(
                block if isinstance(block, str) else block["text"]
                for block in message.content
            )
        else:
            content = []
            for j, block in enumerate(message.content):
                                # Anthropic 和 Bedrock 对话格式
                if isinstance(block, str):
                    content.append({"type": "text", "text": block})
                elif block.get("type") == "text":
                    if missing := [k for k in ("text",) if k not in block]:
                        err = (
                            f"Unrecognized content block at "
                            f"messages[{i}].content[{j}] has 'type': 'text' "
                            f"but is missing expected key(s) "
                            f"{missing}. Full content block:\n\n{block}"
                        )
                        raise ValueError(err)
                    content.append({"type": block["type"], "text": block["text"]})
                elif block.get("type") == "image_url":
                    if missing := [k for k in ("image_url",) if k not in block]:
                        err = (
                            f"Unrecognized content block at "
                            f"messages[{i}].content[{j}] has 'type': 'image_url' "
                            f"but is missing expected key(s) "
                            f"{missing}. Full content block:\n\n{block}"
                        )
                        raise ValueError(err)
                    content.append(
                        {
                            "type": "image_url",
                            "image_url": block["image_url"],
                        }
                    )
                                # Anthropic
                elif is_data_content_block(block):
                    formatted_block = convert_to_openai_data_block(block)
                    if (
                        formatted_block.get("type") == "file"
                        and "file" in formatted_block
                        and "filename" not in formatted_block["file"]
                    ):
                        logger.info("Generating a fallback filename.")
                        formatted_block["file"]["filename"] = "LC_AUTOGENERATED"
                    content.append(formatted_block)
                                # Bedrock 对话
                elif (block.get("type") == "image") or "image" in block:
                                        # OpenAI 文件格式
                    if source := block.get("source"):
                        if missing := [
                            k for k in ("media_type", "type", "data") if k not in source
                        ]:
                            err = (
                                f"Unrecognized content block at "
                                f"messages[{i}].content[{j}] has 'type': 'image' "
                                f"but 'source' is missing expected key(s) "
                                f"{missing}. Full content block:\n\n{block}"
                            )
                            raise ValueError(err)
                        content.append(
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": (
                                        f"data:{source['media_type']};"
                                        f"{source['type']},{source['data']}"
                                    )
                                },
                            }
                        )
                                        # OpenAI 音频格式
                    elif image := block.get("image"):
                        if missing := [
                            k for k in ("source", "format") if k not in image
                        ]:
                            err = (
                                f"Unrecognized content block at "
                                f"messages[{i}].content[{j}] has key 'image', "
                                f"but 'image' is missing expected key(s) "
                                f"{missing}. Full content block:\n\n{block}"
                            )
                            raise ValueError(err)
                        b64_image = _bytes_to_b64_str(image["source"]["bytes"])
                        content.append(
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": (
                                        f"data:image/{image['format']};base64,{b64_image}"
                                    )
                                },
                            }
                        )
                    else:
                        err = (
                            f"Unrecognized content block at "
                            f"messages[{i}].content[{j}] has 'type': 'image' "
                            f"but does not have a 'source' or 'image' key. Full "
                            f"content block:\n\n{block}"
                        )
                        raise ValueError(err)
                                # OpenAI 响应
                elif (
                    block.get("type") == "file"
                    and isinstance(block.get("file"), dict)
                    and isinstance(block.get("file", {}).get("file_data"), str)
                ):
                    if block.get("file", {}).get("filename") is None:
                        logger.info("Generating a fallback filename.")
                        formatted_block = {
                            **block,
                            "file": {**block["file"], "filename": "LC_AUTOGENERATED"},
                        }
                        content.append(formatted_block)
                    else:
                        content.append(block)
                                # 递归以确保工具消息内容为 OpenAI 格式。
                elif (
                    block.get("type") == "input_audio"
                    and isinstance(block.get("input_audio"), dict)
                    and isinstance(block.get("input_audio", {}).get("data"), str)
                    and isinstance(block.get("input_audio", {}).get("format"), str)
                ):
                    content.append(block)
                elif block.get("type") == "tool_use":
                    if missing := [
                        k for k in ("id", "name", "input") if k not in block
                    ]:
                        err = (
                            f"Unrecognized content block at "
                            f"messages[{i}].content[{j}] has 'type': "
                            f"'tool_use', but is missing expected key(s) "
                            f"{missing}. Full content block:\n\n{block}"
                        )
                        raise ValueError(err)
                    if not any(
                        tool_call["id"] == block["id"]
                        for tool_call in cast("AIMessage", message).tool_calls
                    ):
                        oai_msg["tool_calls"] = oai_msg.get("tool_calls", [])
                        oai_msg["tool_calls"].append(
                            {
                                "type": "function",
                                "id": block["id"],
                                "function": {
                                    "name": block["name"],
                                    "arguments": json.dumps(
                                        block["input"], ensure_ascii=False
                                    ),
                                },
                            }
                        )
                elif block.get("type") == "function_call":  # VertexAI 格式
                    if not any(
                        tool_call["id"] == block.get("call_id")
                        for tool_call in cast("AIMessage", message).tool_calls
                    ):
                        if missing := [
                            k
                            for k in ("call_id", "name", "arguments")
                            if k not in block
                        ]:
                            err = (
                                f"Unrecognized content block at "
                                f"messages[{i}].content[{j}] has 'type': "
                                f"'tool_use', but is missing expected key(s) "
                                f"{missing}. Full content block:\n\n{block}"
                            )
                            raise ValueError(err)
                        oai_msg["tool_calls"] = oai_msg.get("tool_calls", [])
                        oai_msg["tool_calls"].append(
                            {
                                "type": "function",
                                "id": block.get("call_id"),
                                "function": {
                                    "name": block.get("name"),
                                    "arguments": block.get("arguments"),
                                },
                            }
                        )
                    if pass_through_unknown_blocks:
                        content.append(block)
                elif block.get("type") == "tool_result":
                    if missing := [
                        k for k in ("content", "tool_use_id") if k not in block
                    ]:
                        msg = (
                            f"Unrecognized content block at "
                            f"messages[{i}].content[{j}] has 'type': "
                            f"'tool_result', but is missing expected key(s) "
                            f"{missing}. Full content block:\n\n{block}"
                        )
                        raise ValueError(msg)
                    tool_message = ToolMessage(
                        block["content"],
                        tool_call_id=block["tool_use_id"],
                        status="error" if block.get("is_error") else "success",
                    )
                                        # 检查所有消息是否已符合 token 限制
                    tool_messages.extend(
                        convert_to_openai_messages(
                            [tool_message], text_format=text_format
                        )
                    )
                elif (block.get("type") == "json") or "json" in block:
                    if "json" not in block:
                        msg = (
                            f"Unrecognized content block at "
                            f"messages[{i}].content[{j}] has 'type': 'json' "
                            f"but does not have a 'json' key. Full "
                            f"content block:\n\n{block}"
                        )
                        raise ValueError(msg)
                    content.append(
                        {
                            "type": "text",
                            "text": json.dumps(block["json"]),
                        }
                    )
                elif (block.get("type") == "guard_content") or "guard_content" in block:
                    if (
                        "guard_content" not in block
                        or "text" not in block["guard_content"]
                    ):
                        msg = (
                            f"Unrecognized content block at "
                            f"messages[{i}].content[{j}] has 'type': "
                            f"'guard_content' but does not have a "
                            f"messages[{i}].content[{j}]['guard_content']['text'] "
                            f"key. Full content block:\n\n{block}"
                        )
                        raise ValueError(msg)
                    text = block["guard_content"]["text"]
                    if isinstance(text, dict):
                        text = text["text"]
                    content.append({"type": "text", "text": text})
                                # 当所有消息均符合时，仅在需要时应用 end_on 过滤
                elif block.get("type") == "media":
                    if missing := [k for k in ("mime_type", "data") if k not in block]:
                        err = (
                            f"Unrecognized content block at "
                            f"messages[{i}].content[{j}] has 'type': "
                            f"'media' but does not have key(s) {missing}. Full "
                            f"content block:\n\n{block}"
                        )
                        raise ValueError(err)
                    if "image" not in block["mime_type"]:
                        err = (
                            f"OpenAI messages can only support text and image data."
                            f" Received content block with media of type:"
                            f" {block['mime_type']}"
                        )
                        raise ValueError(err)
                    b64_image = _bytes_to_b64_str(block["data"])
                    content.append(
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": (f"data:{block['mime_type']};base64,{b64_image}")
                            },
                        }
                    )
                elif (
                    block.get("type") in {"thinking", "reasoning"}
                    or pass_through_unknown_blocks
                ):
                    content.append(block)
                else:
                    err = (
                        f"Unrecognized content block at "
                        f"messages[{i}].content[{j}] does not match OpenAI, "
                        f"Anthropic, Bedrock Converse, or VertexAI format. Full "
                        f"content block:\n\n{block}"
                    )
                    raise ValueError(err)
            if text_format == "string" and not any(
                block["type"] != "text" for block in content
            ):
                content = "\n".join(block["text"] for block in content)
        oai_msg["content"] = content
        if message.content and not oai_msg["content"] and tool_messages:
            oai_messages.extend(tool_messages)
        else:
            oai_messages.extend([oai_msg, *tool_messages])

    if is_single:
        return oai_messages[0]
    return oai_messages


def _first_max_tokens(
    messages: Sequence[BaseMessage],
    *,
    max_tokens: int,
    token_counter: Callable[[list[BaseMessage]], int],
    text_splitter: Callable[[str], list[str]],
    partial_strategy: Literal["first", "last"] | None = None,
    end_on: str | type[BaseMessage] | Sequence[str | type[BaseMessage]] | None = None,
) -> list[BaseMessage]:
    messages = list(messages)
    if not messages:
        return messages

        # 使用二分查找以确定可在 token 限制内包含的最大消息数量
    if token_counter(messages) <= max_tokens:
                # idx 现在包含我们可包含的最大完整消息数
        if end_on:
            for _ in range(len(messages)):
                if not _is_message_type(messages[-1], end_on):
                    messages.pop()
                else:
                    break
        return messages

        # 高效提取文本内容
    left, right = 0, len(messages)
    max_iterations = len(messages).bit_length()
    for _ in range(max_iterations):
        if left >= right:
            break
        mid = (left + right + 1) // 2
        if token_counter(messages[:mid]) <= max_tokens:
            left = mid
            idx = mid
        else:
            right = mid - 1

        # 对可包含的最大分割数进行二分查找
    idx = left

    if partial_strategy and idx < len(messages):
        included_partial = False
        copied = False
        if isinstance(messages[idx].content, list):
            excluded = messages[idx].model_copy(deep=True)
            copied = True
            num_block = len(excluded.content)
            if partial_strategy == "last":
                excluded.content = list(reversed(excluded.content))
            for _ in range(1, num_block):
                excluded.content = excluded.content[:-1]
                if token_counter([*messages[:idx], excluded]) <= max_tokens:
                    messages = [*messages[:idx], excluded]
                    idx += 1
                    included_partial = True
                    break
            if included_partial and partial_strategy == "last":
                excluded.content = list(reversed(excluded.content))
        if not included_partial:
            if not copied:
                excluded = messages[idx].model_copy(deep=True)
                copied = True

                        # 过滤掉 end_on 类型之后的消息
            text = None
            if isinstance(excluded.content, str):
                text = excluded.content
            elif isinstance(excluded.content, list) and excluded.content:
                for block in excluded.content:
                    if isinstance(block, str):
                        text = block
                        break
                    if isinstance(block, dict) and block.get("type") == "text":
                        text = block.get("text")
                        break

            if text:
                if not copied:
                    excluded = excluded.model_copy(deep=True)

                split_texts = text_splitter(text)
                base_message_count = token_counter(messages[:idx])
                if partial_strategy == "last":
                    split_texts = list(reversed(split_texts))

                                # 处理系统消息的保留
                left, right = 0, len(split_texts)
                max_iterations = len(split_texts).bit_length()
                for _ in range(max_iterations):
                    if left >= right:
                        break
                    mid = (left + right + 1) // 2
                    excluded.content = "".join(split_texts[:mid])
                    if base_message_count + token_counter([excluded]) <= max_tokens:
                        left = mid
                    else:
                        right = mid - 1

                if left > 0:
                    content_splits = split_texts[:left]
                    if partial_strategy == "last":
                        content_splits = list(reversed(content_splits))
                    excluded.content = "".join(content_splits)
                    messages = [*messages[:idx], excluded]
                    idx += 1

    if end_on:
        for _ in range(idx):
            if idx > 0 and not _is_message_type(messages[idx - 1], end_on):
                idx -= 1
            else:
                break

    return messages[:idx]


def _last_max_tokens(
    messages: Sequence[BaseMessage],
    *,
    max_tokens: int,
    token_counter: Callable[[list[BaseMessage]], int],
    text_splitter: Callable[[str], list[str]],
    allow_partial: bool = False,
    include_system: bool = False,
    start_on: str | type[BaseMessage] | Sequence[str | type[BaseMessage]] | None = None,
    end_on: str | type[BaseMessage] | Sequence[str | type[BaseMessage]] | None = None,
) -> list[BaseMessage]:
    messages = list(messages)
    if len(messages) == 0:
        return []

        # 反转消息以使用 _first_max_tokens 并反向逻辑
    if end_on:
        for _ in range(len(messages)):
            if not _is_message_type(messages[-1], end_on):
                messages.pop()
            else:
                break

        # 计算包含系统消息后的剩余 token 数量
    system_message = None
    if include_system and len(messages) > 0 and isinstance(messages[0], SystemMessage):
        system_message = messages[0]
        messages = messages[1:]

        # 重新反转消息并根据需要添加系统消息
    reversed_messages = messages[::-1]

        # 若提供则计算工具的 token 数量
    remaining_tokens = max_tokens
    if system_message:
        system_tokens = token_counter([system_message])
        remaining_tokens = max(0, max_tokens - system_tokens)

    reversed_result = _first_max_tokens(
        reversed_messages,
        max_tokens=remaining_tokens,
        token_counter=token_counter,
        text_splitter=text_splitter,
        partial_strategy="last" if allow_partial else None,
        end_on=start_on,
    )

        # tool_call_schema 在实例级别缓存
    result = reversed_result[::-1]
    if system_message:
        result = [system_message, *result]

    return result


_MSG_CHUNK_MAP: dict[type[BaseMessage], type[BaseMessageChunk]] = {
    HumanMessage: HumanMessageChunk,
    AIMessage: AIMessageChunk,
    SystemMessage: SystemMessageChunk,
    ToolMessage: ToolMessageChunk,
    FunctionMessage: FunctionMessageChunk,
    ChatMessage: ChatMessageChunk,
}
_CHUNK_MSG_MAP = {v: k for k, v in _MSG_CHUNK_MAP.items()}


def _msg_to_chunk(message: BaseMessage) -> BaseMessageChunk:
    if message.__class__ in _MSG_CHUNK_MAP:
        return _MSG_CHUNK_MAP[message.__class__](**message.model_dump(exclude={"type"}))

    for msg_cls, chunk_cls in _MSG_CHUNK_MAP.items():
        if isinstance(message, msg_cls):
            return chunk_cls(**message.model_dump(exclude={"type"}))

    msg = (
        f"Unrecognized message class {message.__class__}. Supported classes are "
        f"{list(_MSG_CHUNK_MAP.keys())}"
    )
    msg = create_message(message=msg, error_code=ErrorCode.MESSAGE_COERCION_FAILURE)
    raise ValueError(msg)


def _chunk_to_msg(chunk: BaseMessageChunk) -> BaseMessage:
    if chunk.__class__ in _CHUNK_MSG_MAP:
        return _CHUNK_MSG_MAP[chunk.__class__](
            **chunk.model_dump(exclude={"type", "tool_call_chunks", "chunk_position"})
        )
    for chunk_cls, msg_cls in _CHUNK_MSG_MAP.items():
        if isinstance(chunk, chunk_cls):
            return msg_cls(
                **chunk.model_dump(
                    exclude={"type", "tool_call_chunks", "chunk_position"}
                )
            )

    msg = (
        f"Unrecognized message chunk class {chunk.__class__}. Supported classes are "
        f"{list(_CHUNK_MSG_MAP.keys())}"
    )
    msg = create_message(message=msg, error_code=ErrorCode.MESSAGE_COERCION_FAILURE)
    raise ValueError(msg)


def _default_text_splitter(text: str) -> list[str]:
    splits = text.split("\n")
    return [s + "\n" for s in splits[:-1]] + splits[-1:]


def _is_message_type(
    message: BaseMessage,
    type_: str | type[BaseMessage] | Sequence[str | type[BaseMessage]],
) -> bool:
    types = [type_] if isinstance(type_, (str, type)) else type_
    types_str = [t for t in types if isinstance(t, str)]
    types_types = tuple(t for t in types if isinstance(t, type))

    return message.type in types_str or isinstance(message, types_types)


def _bytes_to_b64_str(bytes_: bytes) -> str:
    return base64.b64encode(bytes_).decode("utf-8")


def _get_message_openai_role(message: BaseMessage) -> str:
    if isinstance(message, AIMessage):
        return "assistant"
    if isinstance(message, HumanMessage):
        return "user"
    if isinstance(message, ToolMessage):
        return "tool"
    if isinstance(message, SystemMessage):
        role = message.additional_kwargs.get("__openai_role__", "system")
        if not isinstance(role, str):
            msg = f"Expected '__openai_role__' to be a str, got {type(role).__name__}"
            raise TypeError(msg)
        return role
    if isinstance(message, FunctionMessage):
        return "function"
    if isinstance(message, ChatMessage):
        return message.role
    msg = f"Unknown BaseMessage type {message.__class__}."
    raise ValueError(msg)


def _convert_to_openai_tool_calls(tool_calls: list[ToolCall]) -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "id": tool_call["id"],
            "function": {
                "name": tool_call["name"],
                "arguments": json.dumps(tool_call["args"], ensure_ascii=False),
            },
        }
        for tool_call in tool_calls
    ]


def count_tokens_approximately(
    messages: Iterable[MessageLikeRepresentation],
    *,
    chars_per_token: float = 4.0,
    extra_tokens_per_message: float = 3.0,
    count_name: bool = True,
    tokens_per_image: int = 85,
    use_usage_metadata_scaling: bool = False,
    tools: list[BaseTool | dict[str, Any]] | None = None,
) -> int:
    """近似计算消息中的总 token 数量。

    token 数量包括序列化后的消息内容、角色，以及（可选的）名称。

    - For AI messages, the token count also includes stringified tool calls.
    - For tool messages, the token count also includes the tool call ID.
    - For multimodal messages with images, applies a fixed token penalty per image
        instead of counting base64-encoded characters.
    - If tools are provided, the token count also includes stringified tool schemas.

    Args:
        messages: List of messages to count tokens for.
        chars_per_token: Number of characters per token to use for the approximation.
            One token corresponds to ~4 chars for common English text.
            You can also specify `float` values for more fine-grained control.
            [See more here](https://platform.openai.com/tokenizer).
        extra_tokens_per_message: Number of extra tokens to add per message, e.g.
            special tokens, including beginning/end of message.
            You can also specify `float` values for more fine-grained control.
            [See more here](https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb).
        count_name: Whether to include message names in the count.
        tokens_per_image: Fixed token cost per image (default: 85, aligned with
            OpenAI's low-resolution image token cost).
        use_usage_metadata_scaling: If True, and all AI messages have consistent
            `response_metadata['model_provider']`, scale the approximate token count
            using the **most recent** AI message that has
            `usage_metadata['total_tokens']`. The scaling factor is:
            `AI_total_tokens / approx_tokens_up_to_that_AI_message`
        tools: List of tools to include in the token count. Each tool can be a
            `BaseTool` instance, a dict representing a tool schema, or a plain
            callable. `BaseTool` instances and callables are converted to
            OpenAI tool format before counting.

    Returns:
        消息（以及提供的工具）的近似 token 数量。

    Note:
        这是一个简单的近似方法，可能不匹配特定模型使用的精确 token
        数量。如需准确统计，请使用模型特定的 tokenizer。

        对于包含图像的多模态消息，按每张图像应用固定的 token 惩罚值，而不是计算 base64
        编码的字符数，这能提供更真实的近似结果。

    !!! version-added "Added in `langchain-core` 0.3.46"
    """
    converted_messages = convert_to_messages(messages)

    token_count = 0.0

    ai_model_provider: str | None = None
    invalid_model_provider = False
    last_ai_total_tokens: int | None = None
    approx_at_last_ai: float | None = None

        # 删除 schema 自身的 `title`/`description` 以避免重复计数：
    if tools:
        tools_chars = 0
        for tool in tools:
            if isinstance(tool, dict):
                tool_dict = tool
            else:
                                # 它们已在顶层重复。
                schema = tool.tool_call_schema
                if isinstance(schema, dict):
                    parameters = dict(schema)
                else:
                    parameters = dict(get_model_json_schema(schema))
                                # 处理多模态内容（内容块列表）
                                # 字符串块
                parameters.pop("title", None)
                parameters.pop("description", None)
                tool_dict = {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": parameters,
                }
            tools_chars += len(json.dumps(tool_dict))
        token_count += math.ceil(tools_chars / chars_per_token)

    for message in converted_messages:
        message_chars = 0

        if isinstance(message.content, str):
            message_chars += len(message.content)
                # 对图像块应用固定惩罚
        elif isinstance(message.content, list):
            for block in message.content:
                if isinstance(block, str):
                                        # 正常计数文本块
                    message_chars += len(block)
                elif isinstance(block, dict):
                    block_type = block.get("type", "")

                                        # 对未知块类型进行保守估计
                    if block_type in {"image", "image_url"}:
                        token_count += tokens_per_image
                                        # 对意外块类型的回退处理
                    elif block_type == "text":
                        text = block.get("text", "")
                        message_chars += len(text)
                                        # 对其他内容类型的回退处理
                    else:
                        message_chars += len(repr(block))
                else:
                                        # 意外块类型的备用处理
                    message_chars += len(repr(block))  # type: ignore[unreachable]
        else:
                        # 其他内容类型的备用方案
            content = repr(message.content)  # type: ignore[unreachable]
            message_chars += len(content)

        if (
            isinstance(message, AIMessage)
                        # 排除 Anthropic 格式，因为工具调用已包含在内容中
            and not isinstance(message.content, list)
            and message.tool_calls
        ):
            tool_calls_content = repr(message.tool_calls)
            message_chars += len(tool_calls_content)

        if isinstance(message, ToolMessage):
            message_chars += len(message.tool_call_id)

        role = _get_message_openai_role(message)
        message_chars += len(role)

        if message.name and count_name:
            message_chars += len(message.name)

                # NOTE: 我们按每条消息向上取整以确保
                # 每条消息的 token 数量加起来等于总数量
                # 针对消息列表
        token_count += math.ceil(message_chars / chars_per_token)

                # 每条消息额外增加 token 数
        token_count += extra_tokens_per_message

        if use_usage_metadata_scaling and isinstance(message, AIMessage):
            model_provider = message.response_metadata.get("model_provider")
            if ai_model_provider is None:
                ai_model_provider = model_provider
            elif model_provider != ai_model_provider:
                invalid_model_provider = True

            if message.usage_metadata and isinstance(
                (total_tokens := message.usage_metadata.get("total_tokens")), int
            ):
                last_ai_total_tokens = total_tokens
                approx_at_last_ai = token_count

    if (
        use_usage_metadata_scaling
        and len(converted_messages) > 1
        and not invalid_model_provider
        and ai_model_provider is not None
        and last_ai_total_tokens is not None
        and approx_at_last_ai
        and approx_at_last_ai > 0
    ):
        scale_factor = last_ai_total_tokens / approx_at_last_ai
        token_count *= min(1.25, max(1.0, scale_factor))

        # 如果 extra_tokens_per_message 是浮点数，再向上取整一次
    return math.ceil(token_count)


# 从字符串快捷方式到 token 计数函数的映射
def _approximate_token_counter(messages: Sequence[BaseMessage]) -> int:
    """包装 `count_tokens_approximately` 的函数，以匹配预期的签名。"""
    return count_tokens_approximately(messages)


_TOKEN_COUNTER_SHORTCUTS = {
    "approximate": _approximate_token_counter,
}

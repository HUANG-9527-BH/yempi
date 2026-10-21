"""Anthropic 聊天模型。"""

from __future__ import annotations

import copy
import datetime
import hashlib
import json
import re
import warnings
from collections.abc import AsyncIterator, Callable, Iterator, Mapping, Sequence
from functools import cached_property
from operator import itemgetter
from typing import Any, Final, Literal, TypeGuard, cast

import anthropic
from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.exceptions import (
    ContextOverflowError,
    ModelAPIError,
    ModelAuthenticationError,
    ModelConnectionError,
    ModelInvalidRequestError,
    ModelNotFoundError,
    ModelPermissionDeniedError,
    ModelRateLimitError,
    ModelTimeoutError,
    OutputParserException,
)
from langchain_core.language_models import (
    LanguageModelInput,
    ModelProfile,
    ModelProfileRegistry,
)
from langchain_core.language_models.chat_models import BaseChatModel, LangSmithParams
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolCall,
    ToolMessage,
    is_data_content_block,
)
from langchain_core.messages import content as types
from langchain_core.messages.ai import (
    InputTokenDetails,
    OutputTokenDetails,
    UsageMetadata,
)
from langchain_core.messages.tool import tool_call_chunk as create_tool_call_chunk
from langchain_core.output_parsers import (
    JsonOutputKeyToolsParser,
    JsonOutputParser,
    PydanticOutputParser,
    PydanticToolsParser,
)
from langchain_core.output_parsers.base import OutputParserLike
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.runnables import Runnable, RunnableMap, RunnablePassthrough
from langchain_core.tools import BaseTool
from langchain_core.utils import from_env, get_pydantic_field_names
from langchain_core.utils._gateway import (
    GATEWAY_METADATA_RESPONSE_KEY,
    _apply_gateway_config,
    _parse_gateway_metadata,
)
from langchain_core.utils.function_calling import (
    convert_to_json_schema,
    convert_to_openai_tool,
)
from langchain_core.utils.pydantic import is_basemodel_subclass
from langchain_core.utils.utils import _build_model_kwargs
from pydantic import BaseModel, ConfigDict, Field, SecretStr, model_validator
from typing_extensions import NotRequired, Self, TypedDict

from langchain_anthropic import __version__
from langchain_anthropic._client_utils import (
    _get_default_async_httpx_client,
    _get_default_httpx_client,
)
from langchain_anthropic._compat import (
    _convert_from_v1_to_anthropic,
    _unwrap_non_standard,
)
from langchain_anthropic._sdk_compat import (
    _aparse,
    _route_unsupported_sampling_params,
)
from langchain_anthropic.data._profiles import _PROFILES
from langchain_anthropic.output_parsers import extract_tool_calls

_message_type_lookups = {
    "human": "user",
    "ai": "assistant",
    "AIMessageChunk": "assistant",
    "HumanMessageChunk": "user",
}

_MODEL_PROFILES = cast(ModelProfileRegistry, _PROFILES)

_USER_AGENT: Final[str] = f"langchain-anthropic/{__version__}"


def _add_gateway_metadata(generation_info: dict[str, Any], raw_response: Any) -> None:
    """如果存在，将解析后的 LangSmith 网关元数据添加到 `generation_info` 中。

    Args:
        generation_info: 要就地修改的生成信息。
        raw_response: 原始提供方响应，或 None。
    """
    headers = getattr(raw_response, "headers", None)
    if headers is None:
        return
    gateway_metadata = _parse_gateway_metadata(headers)
    if gateway_metadata is not None:
        generation_info[GATEWAY_METADATA_RESPONSE_KEY] = gateway_metadata


def _get_default_model_profile(model_name: str) -> ModelProfile:
    """获取模型的默认配置文件。

    Args:
        model_name: 模型标识符。

    Returns:
        模型配置字典，如果未找到则返回空字典。
    """
    default = _MODEL_PROFILES.get(model_name)
    if default:
        return default.copy()
    return {}


_FALLBACK_MAX_OUTPUT_TOKENS: Final[int] = 4096


class AnthropicTool(TypedDict):
    """Anthropic 工具定义，用于自定义（用户定义）工具。

    自定义工具使用 `name` 和 `input_schema` 字段来定义工具的接口。 这些字段通过
    `convert_to_anthropic_tool` 从 LangChain 工具格式（函数、Pydantic 模型、`BaseTool`
    对象）转换而来。
    """

    name: str

    input_schema: dict[str, Any]

    description: NotRequired[str]

    strict: NotRequired[bool]

    cache_control: NotRequired[dict[str, str]]

    defer_loading: NotRequired[bool]

    input_examples: NotRequired[list[dict[str, Any]]]

    allowed_callers: NotRequired[list[str]]


# ---------------------------------------------------------------------------
# 当 Anthropic 发布新的内置工具时，可能需要更新以下两个位置：
# ---------------------------------------------------------------------------
# 1. _TOOL_TYPE_TO_BETA (如下) - 如果工具需要 beta 标头，则添加映射。
#
# 并非所有工具都需要此设置；仅当 API 要求 beta 标头时才添加。
#     2. _is_builtin_tool() - 将工具类型前缀添加到 _BUILTIN_TOOL_PREFIXES。
#
# 这确保工具字典原样传递给 API（而不是通过 convert_to_anthropic_tool
# 转换，后者可能失败）。
#     Base64 编码的图像
#     从内容块中提取 cache_control 并提升至 tool_result 层级。Anthropic 的 API 不支持在
#     tool_result 内容子块上设置 cache_control。
# ---------------------------------------------------------------------------

_TOOL_TYPE_TO_BETA: dict[str, str] = {
    "web_fetch_20250910": "web-fetch-2025-09-10",
    "code_execution_20250522": "code-execution-2025-05-22",
    "mcp_toolset": "mcp-client-2025-11-20",
    "memory_20250818": "context-management-2025-06-27",
    "computer_20250124": "computer-use-2025-01-24",
    "computer_20251124": "computer-use-2025-11-24",
    "tool_search_tool_regex_20251119": "advanced-tool-use-2025-11-20",
    "tool_search_tool_bm25_20251119": "advanced-tool-use-2025-11-20",
    "advisor_20260301": "advisor-tool-2026-03-01",
}
"""Mapping of tool type to required beta header.

Some tool types require specific beta headers to be enabled.
"""

_BUILTIN_TOOL_PREFIXES = [
    "text_editor_",
    "computer_",
    "bash_",
    "web_search_",
    "web_fetch_",
    "code_execution_",
    "mcp_toolset",
    "memory_",
    "tool_search_",
    "advisor_",
]

_ANTHROPIC_EXTRA_FIELDS: set[str] = {
    "allowed_callers",
    "cache_control",
    "defer_loading",
    "eager_input_streaming",
    "input_examples",
}
"""Valid Anthropic-specific extra fields"""


def _is_builtin_tool(tool: Any) -> TypeGuard[dict[str, Any]]:
    """检查工具是否为内置（服务器端）Anthropic 工具。

    `tool` 必须是 `dict` 类型，并且具有以已知内置工具前缀开头的 `type` 键。

    [Claude 文档](https://platform.claude.com/docs/en/agents-and-tools/tool-use/overview)
    """
    if not isinstance(tool, dict):
        return False

    tool_type = tool.get("type")
    if not tool_type or not isinstance(tool_type, str):
        return False

    return any(tool_type.startswith(prefix) for prefix in _BUILTIN_TOOL_PREFIXES)


def _format_image(url: str) -> dict:
    """将 part["image_url"]["url"] 字符串（OpenAI 格式）转换为 Anthropic 格式。

    {
        "type": "base64",
        "media_type": "image/jpeg",
        "data": "/9j/4AAQSkZJRg...",
    }

    Or

    {
        "type": "url",
        "url": "https://example.com/image.jpg",
    }
    """
        # 数据 URI
    base64_regex = r"^data:(?P<media_type>image/.+);base64,(?P<data>.+)$"
    base64_match = re.match(base64_regex, url)

    if base64_match:
        return {
            "type": "base64",
            "media_type": base64_match.group("media_type"),
            "data": base64_match.group("data"),
        }

    # Url
    url_regex = r"^https?://.*$"
    url_match = re.match(url_regex, url)

    if url_match:
        return {
            "type": "url",
            "url": url,
        }

    msg = (
        "Malformed url parameter."
        " Must be either an image URL (https://example.com/image.jpg)"
        " or base64 encoded string (data:image/png;base64,'/9j/4AAQSk'...)"
    )
    raise ValueError(
        msg,
    )


_TOOL_CALL_ID_PATTERN = re.compile(r"^[a-zA-Z0-9_-]+$")
"""Anthropic requires `tool_use`/`tool_result` IDs to match this pattern."""


def _normalize_tool_call_id(tool_call_id: str | None) -> str | None:
    """在需要时，将工具调用 ID 映射为 Anthropic 兼容形式。

    Anthropic 会拒绝 `tool_use`/`tool_result` ID 中不匹配 `^[a-zA-Z0-9_-]+$` 的情况。
    其他提供方生成的 ID 在跨提供方重放线程时可能违反此规则（例如 Fireworks/Kimi 生成
    `functions.write_todos:0`，其中的 `.` 和 `:` 是非法字符）。 有效 ID 保持不变；无效
    ID 会被确定性哈希处理，使得重写的 `tool_use.id` 及其对应的 `tool_use_id`
    在请求内和跨轮次中仍能解析为相同值。

    空字符串和 `None` ID 保持不变，以便真正错误的请求能清晰地由 Anthropic
    抛出错误，而不是被合成的 ID 掩盖。

    Args:
        tool_call_id: 要规范化的工具调用 ID。

    Returns:
        如果 ID 为空、`None` 或已有效，则返回原始 ID；否则返回一个确定性的 Anthropic
        兼容替代品。
    """
    if not tool_call_id or _TOOL_CALL_ID_PATTERN.match(tool_call_id):
        return tool_call_id
    digest = hashlib.sha256(tool_call_id.encode()).hexdigest()
    return f"toolu_{digest[:24]}"


def _normalize_block_tool_use_id(block: dict) -> dict:
    """返回 `block`，并规范化其 `tool_use_id`（如果存在）。

    与 `_normalize_tool_call_id` 对应，用于 `tool_result` 风格的内容块，
    以确保 `tool_use_id` 在预结构化时（例如 `ToolMessage` 的内容已经是 `tool_result` 块列表）
    与其配对的、已规范化的 `tool_use.id` 保持一致。
    对于已有效的 ID 无操作。
    """
    if "tool_use_id" in block:
        return {**block, "tool_use_id": _normalize_tool_call_id(block["tool_use_id"])}
    return block


def _merge_messages(
    messages: Sequence[BaseMessage],
) -> list[SystemMessage | AIMessage | HumanMessage]:
    """将人类消息/工具消息的连续运行合并为单个包含内容块的人类消息。"""  # noqa: E501
    merged: list = []
    for curr in messages:
        if isinstance(curr, ToolMessage):
            if (
                isinstance(curr.content, list)
                and curr.content
                and all(
                    isinstance(block, dict) and block.get("type") == "tool_result"
                    for block in curr.content
                )
            ):
                curr = HumanMessage(curr.content)  # type: ignore[misc]
            else:
                tool_content = curr.content
                cache_ctrl = None
                                # 清理引用以移除 null 的 file_id 字段
                # 向后兼容
                                # 此辅助函数位于 `_format_messages` 和警告站点之间。
                if isinstance(tool_content, list):
                    cleaned = []
                    for block in tool_content:
                        if isinstance(block, dict) and "cache_control" in block:
                            cache_ctrl = block["cache_control"]
                            block = {
                                k: v for k, v in block.items() if k != "cache_control"
                            }
                        cleaned.append(block)
                    tool_content = cleaned
                tool_result: dict = {
                    "type": "tool_result",
                    "content": tool_content,
                    "tool_use_id": _normalize_tool_call_id(curr.tool_call_id),
                    "is_error": curr.status == "error",
                }
                if cache_ctrl:
                    tool_result["cache_control"] = cache_ctrl
                curr = HumanMessage(  # type: ignore[misc]
                    [tool_result],
                )
        last = merged[-1] if merged else None
        if any(
            all(isinstance(m, c) for m in (curr, last))
            for c in (SystemMessage, HumanMessage)
        ):
            if isinstance(cast("BaseMessage", last).content, str):
                new_content: list = [
                    {"type": "text", "text": cast("BaseMessage", last).content},
                ]
            else:
                new_content = copy.copy(cast("list", cast("BaseMessage", last).content))
            if isinstance(curr.content, str):
                new_content.append({"type": "text", "text": curr.content})
            else:
                new_content.extend(curr.content)
            merged[-1] = curr.model_copy(update={"content": new_content})
        else:
            merged.append(curr)
    return merged


def _format_data_content_block(block: dict) -> dict:
    """将标准数据内容块格式化为 Anthropic 所期望的格式。"""
    if block["type"] == "image":
        if "url" in block:
            if block["url"].startswith("data:"):
                                # 填充 content
                formatted_block = {
                    "type": "image",
                    "source": _format_image(block["url"]),
                }
            else:
                formatted_block = {
                    "type": "image",
                    "source": {"type": "url", "url": block["url"]},
                }
        elif "base64" in block or block.get("source_type") == "base64":
            formatted_block = {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": block["mime_type"],
                    "data": block.get("base64") or block.get("data", ""),
                },
            }
        elif "file_id" in block:
            formatted_block = {
                "type": "image",
                "source": {
                    "type": "file",
                    "file_id": block["file_id"],
                },
            }
        elif block.get("source_type") == "id":
            formatted_block = {
                "type": "image",
                "source": {
                    "type": "file",
                    "file_id": block["id"],
                },
            }
        else:
            msg = (
                "Anthropic only supports 'url', 'base64', or 'id' keys for image "
                "content blocks."
            )
            raise ValueError(
                msg,
            )

    elif block["type"] == "file":
        if "url" in block:
            formatted_block = {
                "type": "document",
                "source": {
                    "type": "url",
                    "url": block["url"],
                },
            }
        elif "base64" in block or block.get("source_type") == "base64":
            formatted_block = {
                "type": "document",
                "source": {
                    "type": "base64",
                    "media_type": block.get("mime_type") or "application/pdf",
                    "data": block.get("base64") or block.get("data", ""),
                },
            }
        elif block.get("source_type") == "text":
            formatted_block = {
                "type": "document",
                "source": {
                    "type": "text",
                    "media_type": block.get("mime_type") or "text/plain",
                    "data": block["text"],
                },
            }
        elif "file_id" in block:
            formatted_block = {
                "type": "document",
                "source": {
                    "type": "file",
                    "file_id": block["file_id"],
                },
            }
        elif block.get("source_type") == "id":
            formatted_block = {
                "type": "document",
                "source": {
                    "type": "file",
                    "file_id": block["id"],
                },
            }
        else:
            msg = (
                "Anthropic only supports 'url', 'base64', or 'id' keys for file "
                "content blocks."
            )
            raise ValueError(msg)

    elif block["type"] == "text-plain":
        formatted_block = {
            "type": "document",
            "source": {
                "type": "text",
                "media_type": block.get("mime_type") or "text/plain",
                "data": block["text"],
            },
        }

    else:
        msg = f"Block of type {block['type']} is not supported."
        raise ValueError(msg)

    if formatted_block:
        for key in ["cache_control", "citations", "title", "context"]:
            if key in block:
                formatted_block[key] = block[key]
            elif (metadata := block.get("extras")) and key in metadata:
                formatted_block[key] = metadata[key]
            elif (metadata := block.get("metadata")) and key in metadata:
                # 解析为字典
                formatted_block[key] = metadata[key]

    return formatted_block


def _format_text_block(block: dict) -> dict:
    """将文本内容块缩小为 Anthropic API 支持的字段。

    删除 Anthropic 拒绝的 LangChain 内部字段（例如 ``create_text_block`` 生成的
    ``id``），这些字段被视为额外输入。
    """
    formatted_block = {
        k: v
        for k, v in block.items()
        if k in ("type", "text", "cache_control", "citations")
    }
        # 如果存在与 tool_use 内容块相同 id 的 tool_call，优先使用 tool_call。
    if formatted_block.get("citations"):
        cleaned_citations = []
        for citation in formatted_block["citations"]:
            cleaned_citation = {
                k: v for k, v in citation.items() if not (k == "file_id" and v is None)
            }
            cleaned_citations.append(cleaned_citation)
        formatted_block["citations"] = cleaned_citations
    return formatted_block


_TOOL_CHANGE_BLOCK_TYPES = ("tool_addition", "tool_removal")
"""Anthropic-native system content blocks that change the tool set mid-conversation."""

_MID_CONVERSATION_TOOL_CHANGES_BETA = "mid-conversation-tool-changes-2026-07-01"
"""Beta header required to send `tool_addition` / `tool_removal` blocks."""

_INLINE_TOOLS_BETA = "inline-tools-2026-09-15"
"""Beta header required to define a tool in a `tool_addition` block."""


def _is_tool_change_block(block: object) -> bool:
    """返回内容块是否改变了工具集（包括拼写）。"""
    return (
        isinstance(block, dict)
        and _unwrap_non_standard(block).get("type") in _TOOL_CHANGE_BLOCK_TYPES
    )


def _has_tool_change_block(content: object) -> bool:
    """返回 `content` 中任意块是否改变了工具集。"""
    return isinstance(content, list) and any(_is_tool_change_block(b) for b in content)


def _has_inline_tool_definition(content: object) -> bool:
    """判断 `content` 是否在 `tool_addition` 块中定义了一个工具。"""
    if not isinstance(content, list):
        return False
    return any(
        isinstance(block, dict)
        and block.get("type") == "tool_addition"
        and isinstance(tool := block.get("tool"), dict)
        and tool.get("type") == "tool_definition"
        for block in content
    )


def _format_system_content(
    content: str | list[Any],
    *,
    model: str | None = None,
    preserve_tool_changes: bool = False,
    stacklevel: int = 3,
) -> str | list[dict]:
    """将系统消息内容缩小到 Anthropic 接受的范围。

    字符串内容保持不变；将其提升为单元素块数组会破坏现有调用者的提示词缓存。

    Anthropic 文档规定了系统内容块的封闭集合：`text` 可出现在任意位置，以及在对话中途的
    `system` 转换中使用 `tool_addition` / `tool_removal`。其他任何内容都会被 API
    拒绝，因此会被丢弃并发出警告，而不是转发。通过核心的 `non_standard`
    逃逸机制包装的块会先被解包，因此两种拼写方式行为一致。

    Args:
        content: 系统消息的内容。 model: 请求目标的模型，仅用于警告文本。
        preserve_tool_changes: 是否应转发工具变更块而非丢弃。 stacklevel:
        在归因警告时跳过的帧数，以便警告指向 `_format_messages`
        的调用者而非本模块。默认值适用于直接调用；通过辅助函数调用的调用者需额外增加一帧。

    Returns:
        缩小至 Anthropic 在此位置接受的块。
    """
    if not isinstance(content, list):
        return content

    formatted: list[dict] = []
    for raw_block in content:
        if not isinstance(raw_block, dict):
            formatted.append({"type": "text", "text": raw_block})
            continue

        block = _unwrap_non_standard(raw_block)
        block_type = block.get("type")
        if block_type == "text":
            formatted.append(_format_text_block(block))
        elif block_type in _TOOL_CHANGE_BLOCK_TYPES:
            if preserve_tool_changes:
                formatted.append(block)
            else:
                warnings.warn(
                    f"Tool-change block {block_type!r} was dropped: it is only "
                    "valid on a `SystemMessage` sent in place, and this one was "
                    "hoisted into the top-level `system` field. Use a model whose "
                    "profile sets `mid_conversation_system_messages` and place "
                    "the message after a human or tool message, either last or "
                    f"before an AI message (model: {model!r}).",
                    UserWarning,
                    stacklevel=stacklevel,
                )
        else:
            warnings.warn(
                f"Unrecognized system content block {block_type!r} was dropped. "
                "Anthropic accepts `text` in any system message, plus "
                "`tool_addition` and `tool_removal` on a mid-conversation one.",
                UserWarning,
                stacklevel=stacklevel,
            )
    return formatted


def _warn_system_message_hoisted(model: str | None) -> None:
    """警告非首条系统消息被提升。"""
    warnings.warn(
        "A non-leading `SystemMessage` was moved to the top-level `system` field "
        "and now applies to the entire conversation. To send it in place, use a "
        "model whose profile sets `mid_conversation_system_messages` and place "
        "it after a human or tool message, either last or before an AI message "
        f"(model: {model!r}).",
        UserWarning,
        stacklevel=3,
    )


def _is_server_tool_result_block(block: object) -> bool:
    """返回内容块是否为服务器端工具结果。"""
    return (
        isinstance(block, dict)
        and isinstance(block.get("type"), str)
        and block["type"].endswith("_tool_result")
    )


def _previous_turn_allows_system(previous_turn: dict | None) -> bool:
    """返回系统消息是否可在格式化回合后出现。"""
    if previous_turn is None:
        return False

    previous_role = previous_turn.get("role")
    previous_content = previous_turn.get("content")
    return previous_role == "user" or (
        previous_role == "assistant"
        and isinstance(previous_content, list)
        and bool(previous_content)
        and _is_server_tool_result_block(previous_content[-1])
    )


def _format_in_place_system_messages(
    pending_system: Sequence[BaseMessage],
    *,
    model: str | None,
) -> list[dict]:
    """格式化保持其在消息数组中位置的系统消息。

    内容完全被缩小的消息块将被省略：Anthropic 拒绝空内容的 `system`
    转换，且已被丢弃的块已发出警告。

    Args:
        pending_system: 待发出的系统消息，按顺序排列。
        model: 请求目标的模型，仅用于警告文本。

    Returns:
        格式化后的 `system` 角色回合。
    """
    turns: list[dict] = []
    for pending in pending_system:
        content = _format_system_content(
            pending.content,
            model=model,
            preserve_tool_changes=True,
            # 转换格式
            stacklevel=4,
        )
        if content == []:
            continue
        turns.append({"role": "system", "content": content})
    return turns


def _format_messages(
    messages: Sequence[BaseMessage],
    *,
    model: str | None,
    mid_conversation_system_messages: bool = False,
) -> tuple[str | list[dict] | None, list[dict]]:
    """为 Anthropic API 格式化消息。

    Args:
        messages: 需要格式化的消息。 model: 请求目标的模型，仅用于警告文本。
        mid_conversation_system_messages: 是否允许在对话中间发送非首条系统消息。当为
        `False` 时，该消息将被提升至 `system`。
    """
    system: str | list[dict] | None = None
    formatted_messages: list[dict] = []
    toolsets: dict[str, str] = {}
    merged_messages = _merge_messages(messages)
    last_non_system_index = max(
        (i for i, m in enumerate(merged_messages) if m.type != "system"),
        default=-1,
    )
    pending_system: list[BaseMessage] = []
    for _i, message in enumerate(merged_messages):
        if message.type == "system":
            if _i == 0:
                system = _format_system_content(
                    message.content,
                    model=model,
                    preserve_tool_changes=True,
                )
                continue
            if mid_conversation_system_messages and (
                pending_system
                or _previous_turn_allows_system(
                    formatted_messages[-1] if formatted_messages else None
                )
            ):
                pending_system.append(message)
                continue
            if system is not None:
                msg = "Received multiple non-consecutive system messages."
                raise ValueError(msg)
            system = _format_system_content(message.content, model=model)
            _warn_system_message_hoisted(model)
            continue

        role = _message_type_lookups[message.type]
        content: str | list

        if not isinstance(message.content, str):
                        # for mcp_tool_use
            if not isinstance(message.content, list):
                msg = "Anthropic message content must be str or list of dicts"
                raise ValueError(
                    msg,
                )

                        # https://github.com/anthropics/anthropic-sdk-python/issues/461
            content = []
            for block in message.content:
                if isinstance(block, str):
                    content.append({"type": "text", "text": block})
                elif isinstance(block, dict):
                    if "type" not in block:
                        msg = "Dict content block must have a type key"
                        raise ValueError(msg)
                    if block["type"] in ("reasoning", "function_call") and (
                        not isinstance(message, AIMessage)
                        or message.response_metadata.get("model_provider")
                        != "anthropic"
                    ):
                        continue
                    if block["type"] == "image_url":
                        # 尝试解析流式输出
                        source = _format_image(block["image_url"]["url"])
                        content.append({"type": "image", "source": source})
                    elif is_data_content_block(block):
                        content.append(_format_data_content_block(block))
                    elif block["type"] == "tool_use":
                        # 目前仅添加非空字符串，因为空字符串不被接受。
                                                # 包含 tool_reference 块的工具搜索结果
                        if (
                            isinstance(message, AIMessage)
                            and (block["id"] in [tc["id"] for tc in message.tool_calls])
                            and not block.get("caller")
                        ):
                            overlapping = [
                                tc
                                for tc in message.tool_calls
                                if tc["id"] == block["id"]
                            ]
                            tool_blocks = _lc_tool_calls_to_anthropic_tool_use_blocks(
                                overlapping,
                            )
                            if toolset_name := block.get("toolset_name"):
                                for tool_block in tool_blocks:
                                    tool_block["toolset_name"] = toolset_name
                            content.extend(tool_blocks)
                        else:
                            if tool_input := block.get("input"):
                                args = tool_input
                            elif "partial_json" in block:
                                try:
                                    args = json.loads(block["partial_json"] or "{}")
                                except json.JSONDecodeError:
                                    args = {}
                            else:
                                args = {}
                            tool_use_block = _AnthropicToolUse(
                                type="tool_use",
                                name=block["name"],
                                input=args,
                                id=cast("str", _normalize_tool_call_id(block["id"])),
                            )
                            if caller := block.get("caller"):
                                tool_use_block["caller"] = caller
                            if toolset_name := block.get("toolset_name"):
                                tool_use_block["toolset_name"] = toolset_name
                            content.append(tool_use_block)
                    elif block["type"] in ("server_tool_use", "mcp_tool_use"):
                        formatted_block = {
                            k: v
                            for k, v in block.items()
                            if k
                            in (
                                "type",
                                "id",
                                "input",
                                "name",
                                "server_name",  # 省略仅支持流式传输的字段，例如 `index`。
                                "cache_control",
                            )
                        }
                                                # for mcp_tool_result
                        if block.get("input") == {} and "partial_json" in block:
                            try:
                                input_ = json.loads(block["partial_json"])
                                if input_:
                                    formatted_block["input"] = input_
                            except json.JSONDecodeError:
                                pass
                        content.append(formatted_block)
                    elif block["type"] == "text":
                        text = block.get("text", "")
                        # 需要内容格式化的常规工具结果
                                                # for web_fetch_tool_result
                                                # `tool_use_ids`
                        # 已通过上述分支进行归一化，因此比较归一化后的工具调用
                        # ID，以避免因原始 ID 重写而发出重复的 `tool_use` 块。
                        if text.strip():
                            content.append(_format_text_block(block))
                    elif block["type"] == "thinking":
                        formatted_thinking = {
                            k: v
                            for k, v in block.items()
                            if k in ("type", "thinking", "cache_control", "signature")
                        }
                        if "signature" in formatted_thinking:
                            formatted_thinking.setdefault("thinking", "")
                        content.append(formatted_thinking)
                    elif block["type"] == "redacted_thinking":
                        content.append(
                            {
                                k: v
                                for k, v in block.items()
                                if k in ("type", "cache_control", "data")
                            },
                        )
                    elif (
                        block["type"] == "tool_result"
                        and isinstance(block.get("content"), list)
                        and any(
                            isinstance(item, dict)
                            and item.get("type") == "tool_reference"
                            for item in block["content"]
                        )
                    ):
                                                # anthropic.BadRequestError:
                        # 错误代码：400：所有消息必须具有非空
                        # content，除非是可选的最后一个 assistant 消息
                        content.append(
                            _normalize_block_tool_use_id(
                                {
                                    k: v
                                    for k, v in block.items()
                                    if k
                                    in (
                                        "type",
                                        "content",
                                        "tool_use_id",
                                        "cache_control",
                                    )
                                },
                            ),
                        )
                    elif block["type"] == "tool_search_tool_result":
                                                # `APITimeoutError` 继承自
                        # `APIConnectionError`，因此应先检查它。
                        content.append(
                            _normalize_block_tool_use_id(
                                {
                                    k: v
                                    for k, v in block.items()
                                    if k
                                    in (
                                        "type",
                                        "content",
                                        "tool_use_id",
                                        "cache_control",
                                    )
                                },
                            ),
                        )
                    elif block["type"] == "tool_result":
                                                # sdk 默认值 =
                        # 2：https://github.com/anthropics/anthropic-sdk-python?tab=readme-ov-file#retries
                        tool_content = _format_messages(
                            [HumanMessage(block["content"])],
                            model=None,
                        )[1][0]["content"]
                        content.append(
                            _normalize_block_tool_use_id(
                                {**block, "content": tool_content},
                            ),
                        )
                    elif block["type"] in (
                        "code_execution_tool_result",
                        "bash_code_execution_tool_result",
                        "text_editor_code_execution_tool_result",
                        "mcp_tool_result",
                        "web_search_tool_result",
                        "web_fetch_tool_result",
                    ):
                        content.append(
                            _normalize_block_tool_use_id(
                                {
                                    k: v
                                    for k, v in block.items()
                                    if k
                                    in (
                                        "type",
                                        "content",
                                        "tool_use_id",
                                        "is_error",  # 也可通过 model_kwargs 传入，但将其作为参数会提供更好的开发体验
                                        "cache_control",
                                        "retrieved_at",  # 1. 调用时的 kwargs（例如，llm.invoke(..., betas=[...]))
                                    )
                                },
                            ),
                        )
                    elif block["type"] == "advisor_tool_result":
                        content.append({k: v for k, v in block.items() if k != "index"})
                    else:
                        content.append(block)
                else:
                    msg = (
                        f"Content blocks must be str or dict, instead was: "
                        f"{type(block)}"
                    )
                    raise ValueError(
                        msg,
                    )
        else:
            content = message.content

        if isinstance(message, AIMessage) and (
            message.tool_calls or message.invalid_tool_calls
        ):
            content = content or []
            content = (
                [{"type": "text", "text": message.content}]
                if isinstance(content, str) and content
                else content
            )
            tool_use_ids = [
                cast("dict", block)["id"]
                for block in content
                if cast("dict", block)["type"] == "tool_use"
            ]
            # 优先级顺序：
                        # 2. model_kwargs（例如，ChatAnthropic(model_kwargs={"betas": [...]}))
                        # 3. 直接参数（例如，ChatAnthropic(betas=[...]))
            missing_tool_calls = [
                tc
                for tc in message.tool_calls
                if _normalize_tool_call_id(tc["id"]) not in tool_use_ids
            ]
            cast("list", content).extend(
                _lc_tool_calls_to_anthropic_tool_use_blocks(missing_tool_calls),
            )
            tool_use_ids.extend(
                cast("str", _normalize_tool_call_id(tc["id"]))
                for tc in missing_tool_calls
            )
            for invalid_call in message.invalid_tool_calls:
                tool_call_id = invalid_call.get("id")
                tool_name = invalid_call.get("name")
                if not tool_call_id or not tool_name:
                    continue
                normalized_id = _normalize_tool_call_id(tool_call_id)
                if normalized_id in tool_use_ids:
                    continue
                try:
                    args = json.loads(invalid_call.get("args") or "{}")
                except json.JSONDecodeError:
                    args = {}
                cast("list", content).append(
                    _AnthropicToolUse(
                        type="tool_use",
                        name=tool_name,
                        input=args if isinstance(args, dict) else {},
                        id=cast("str", normalized_id),
                    )
                )
                tool_use_ids.append(normalized_id)

        if role == "assistant" and _i == last_non_system_index:
            if isinstance(content, str):
                content = content.rstrip()
            elif (
                isinstance(content, list)
                and content
                and isinstance(content[-1], dict)
                and content[-1].get("type") == "text"
            ):
                content[-1]["text"] = content[-1]["text"].rstrip()

        if not content and role == "assistant" and _i < last_non_system_index:
                        # 将 User-Agent 与用户提供的头部合并（用户头部优先）
                        # 值 <= 0 表示该参数应被忽略。None 是一个有意义的值
            continue
        if pending_system:
            if role == "assistant":
                formatted_messages.extend(
                    _format_in_place_system_messages(pending_system, model=model)
                )
            else:
                for pending in pending_system:
                    if system is not None:
                        msg = "Received multiple non-consecutive system messages."
                        raise ValueError(msg)
                    system = _format_system_content(pending.content, model=model)
                    _warn_system_message_hoisted(model)
            pending_system = []
        if isinstance(content, list):
            for block in content:
                if not isinstance(block, dict):
                    continue
                if block.get("type") == "tool_use" and block.get("toolset_name"):
                    toolsets[block["id"]] = block["toolset_name"]
                elif block.get("type") == "tool_result" and (
                    toolset_name := toolsets.get(block.get("tool_use_id", ""))
                ):
                    block.setdefault("toolset_name", toolset_name)
        formatted_messages.append({"role": role, "content": content})

    formatted_messages.extend(
        _format_in_place_system_messages(pending_system, model=model)
    )
    return system, formatted_messages


def _container_id(container: Any) -> str | None:
    """从任一接受的 `container` 形式中返回容器 ID。"""
    if isinstance(container, str):
        return container
    if isinstance(container, dict):
        return container.get("id")
    return None


def _collect_code_execution_tool_ids(formatted_messages: list[dict]) -> set[str]:
    """收集由 `code_execution` 调用的 `tool_use` ID。

    根据 Anthropic API 要求，这些块不能应用 `cache_control`。
    """
    code_execution_tool_ids: set[str] = set()

    for message in formatted_messages:
        if message.get("role") != "assistant":
            continue
        content = message.get("content", [])
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") != "tool_use":
                continue
            caller = block.get("caller")
            if isinstance(caller, dict):
                caller_type = caller.get("type", "")
                if caller_type.startswith("code_execution"):
                    tool_id = block.get("id")
                    if tool_id:
                        code_execution_tool_ids.add(tool_id)

    return code_execution_tool_ids


def _is_code_execution_related_block(
    block: dict,
    code_execution_tool_ids: set[str],
) -> bool:
    """返回内容块是否与 `code_execution` 相关。

    对于不应应用 `cache_control` 的块返回 `True`。
    """
    if not isinstance(block, dict):
        return False

    block_type = block.get("type")

    if block_type == "tool_use":
        caller = block.get("caller")
        if isinstance(caller, dict):
            caller_type = caller.get("type", "")
            if caller_type.startswith("code_execution"):
                return True

    if block_type == "tool_result":
        tool_use_id = block.get("tool_use_id")
        if tool_use_id and tool_use_id in code_execution_tool_ids:
            return True

    return False


def _reasoning_effort_levels(profile: object) -> tuple[str, ...]:
    """返回模型配置文件中声明的推理努力等级（如有）。

    防御性处理缺失或格式错误的配置文件：缺少 `profile`、非映射值，或
    `reasoning_effort_levels` 缺失/非列表值，均视为“未声明等级”而非抛出异常。
    """
    if not isinstance(profile, Mapping):
        return ()
    levels = profile.get("reasoning_effort_levels")
    if not isinstance(levels, (list, tuple)):
        return ()
    return tuple(levels)


def _supports_forced_tool_choice(model: str) -> bool:
    """返回模型是否接受 `tool_choice` 类型的 `any` 和 `tool`。"""
    return not model.startswith(
        ("claude-fable-5-1", "claude-opus-5-5", "claude-sonnet-5-5")
    )


def _is_direct_anthropic_llm_type(llm_type: object) -> bool:
    """返回 `_llm_type` 是否通过直接 Anthropic API 连接到 Claude。

    只有直接 API 接受顶级的 `cache_control` 请求参数。 子类若通过其他传输方式（如
    Bedrock、未来后端）路由，需重写 `_llm_type`，并将 `cache_control`
    关键字参数展开为块级断点。

    非字符串类型的 `_llm_type` 值返回 `False`
    而非抛出异常，因此行为异常的子类会降级到更安全的非直接分支。
    """
    return llm_type == "anthropic-chat"


def _apply_cache_control_to_last_eligible_block(
    formatted_messages: list[dict],
    cache_control: Any,
    code_execution_tool_ids: set[str],
) -> bool:
    """将 `cache_control` 放在最后一个可设置断点的块上。

    按最新到最旧遍历消息，并在每条消息内也按最新到最旧遍历块，跳过 `code_execution`
    相关的块（Anthropic 会拒绝在此类块上设置断点）。
    字符串类型的消息内容会被提升为单个文本块，以便附加断点。

    Returns:
        `True` 表示已应用断点，`False` 表示所有候选块均为 `code_execution`
        相关（调用方应发出警告并丢弃该关键字参数）。
    """
    for formatted_message in reversed(formatted_messages):
        content = formatted_message.get("content")
        if isinstance(content, list) and content:
            for block in reversed(content):
                if not isinstance(block, dict):
                    continue
                if _is_code_execution_related_block(block, code_execution_tool_ids):
                    continue
                block["cache_control"] = cache_control
                return True
        elif isinstance(content, str):
            formatted_message["content"] = [
                {
                    "type": "text",
                    "text": content,
                    "cache_control": cache_control,
                }
            ]
            return True
    return False


class AnthropicContextOverflowError(anthropic.BadRequestError, ContextOverflowError):
    """输入超出 Anthropic 上下文限制时抛出 BadRequestError。"""


class AnthropicAuthenticationError(
    anthropic.AuthenticationError, ModelAuthenticationError
):
    """Anthropic 认证错误被归类为 LangChain 模型错误。"""


class AnthropicPermissionDeniedError(
    anthropic.PermissionDeniedError, ModelPermissionDeniedError
):
    """Anthropic 权限错误被归类为 LangChain 模型错误。"""


class AnthropicInvalidRequestError(anthropic.BadRequestError, ModelInvalidRequestError):
    """Anthropic 无效请求错误被归类为 LangChain 模型错误。"""


class AnthropicModelNotFoundError(anthropic.NotFoundError, ModelNotFoundError):
    """Anthropic 找不到资源错误被归类为 LangChain 模型错误。"""


class AnthropicRateLimitError(anthropic.RateLimitError, ModelRateLimitError):
    """Anthropic 速率限制错误被归类为 LangChain 模型错误。"""


class AnthropicAPIError(anthropic.InternalServerError, ModelAPIError):
    """Anthropic 服务器错误被归类为 LangChain 模型错误。"""


class AnthropicOverloadedError(anthropic.OverloadedError, ModelAPIError):
    """Anthropic 负载过重错误（HTTP 529）被归类为 LangChain 模型错误。"""


class AnthropicConnectionError(anthropic.APIConnectionError, ModelConnectionError):
    """Anthropic 连接错误被归类为 LangChain 模型错误。"""


class AnthropicTimeoutError(anthropic.APITimeoutError, ModelTimeoutError):
    """Anthropic 超时错误被归类为 LangChain 模型错误。"""


def _raise_if_authentication_error(e: TypeError) -> None:
    """重新抛出 anthropic SDK 的缺失凭证 `TypeError`，并附带指引信息。"""
    if "Could not resolve authentication method" in str(e):
        msg = (
            "Anthropic authentication failed: no API key or authorization "
            "credentials were provided. Set the ANTHROPIC_API_KEY environment "
            "variable, pass api_key=... to ChatAnthropic, or provide "
            'credentials via default_headers={"Authorization": ...}. If you '
            "are routing through the LangSmith gateway, set LANGSMITH_GATEWAY "
            "and LANGSMITH_GATEWAY_API_KEY."
        )
        raise TypeError(msg) from e


def _handle_anthropic_bad_request(e: anthropic.BadRequestError) -> None:
    """处理 Anthropic BadRequestError。"""
    if "prompt is too long" in e.message:
        raise AnthropicContextOverflowError(
            message=e.message, response=e.response, body=e.body
        ) from e
    if ("messages: at least one message is required") in e.message:
        message = "Received only system message(s). "
        warnings.warn(message, stacklevel=2)
    raise AnthropicInvalidRequestError(
        message=e.message, response=e.response, body=e.body
    ) from e


def _handle_anthropic_api_error(e: anthropic.APIError) -> None:
    """将 Anthropic SDK 错误重新抛出为对应的 LangChain 归类版本。"""
    if isinstance(e, anthropic.AuthenticationError):
        raise AnthropicAuthenticationError(
            message=e.message, response=e.response, body=e.body
        ) from e
    if isinstance(e, anthropic.PermissionDeniedError):
        raise AnthropicPermissionDeniedError(
            message=e.message, response=e.response, body=e.body
        ) from e
    if isinstance(e, anthropic.NotFoundError):
        raise AnthropicModelNotFoundError(
            message=e.message, response=e.response, body=e.body
        ) from e
    if isinstance(e, anthropic.RateLimitError):
        raise AnthropicRateLimitError(
            message=e.message, response=e.response, body=e.body
        ) from e
    if isinstance(e, anthropic.OverloadedError):
        raise AnthropicOverloadedError(
            message=e.message, response=e.response, body=e.body
        ) from e
    if isinstance(e, anthropic.InternalServerError):
        raise AnthropicAPIError(
            message=e.message, response=e.response, body=e.body
        ) from e
        # 对于 Anthropic 客户端，其处理方式不同于未指定该参数的情况
    if isinstance(e, anthropic.APITimeoutError):
        raise AnthropicTimeoutError(e.request) from e
    if isinstance(e, anthropic.APIConnectionError):
        raise AnthropicConnectionError(message=e.message, request=e.request) from e
    raise


class ChatAnthropic(BaseChatModel):
    """Anthropic（Claude）聊天模型。

    参见 [LangChain 文档中的 `ChatAnthropic`](https://docs.langchain.com/oss/python/integrations/chat/anthropic)
    获取教程、功能演示和示例。

    参见 [Claude 平台文档](https://platform.claude.com/docs/en/about-claude/models/overview)
    查看最新模型列表、其功能及定价信息。

    Example:
        ```python
        # pip install -U langchain-anthropic
        # export ANTHROPIC_API_KEY="your-api-key"

        from langchain_anthropic import ChatAnthropic

        model = ChatAnthropic( model="claude-sonnet-4-5-20250929", # temperature=, #
        max_tokens=, # timeout=, # max_retries=, # base_url="...", # 参考 API
        文档获取完整参数列表 )

    Example: Add a tool mid-conversation
        ```python
        from langchain_core.messages import HumanMessage, SystemMessage
        from langchain_anthropic import ChatAnthropic

        model = ChatAnthropic(model="claude-opus-5-5") model.invoke( [
        HumanMessage("What time is it?"), SystemMessage( [ { "type": "tool_addition",
        "tool": { "type": "tool_definition", "definition": { "name": "get_time",
        "description": "获取当前时间。", "input_schema": { "type": "object",
        "properties": {}, }, }, }, } ] ), ] )

    Note:
        任何未显式支持的参数都将直接传递给
        [`Anthropic.messages.create(...)`](https://platform.claude.com/docs/en/api/python/messages/create)
        每次调用模型时。
    """

    model_config = ConfigDict(
        populate_by_name=True,
    )

    model: str = Field(alias="model_name")
    """Model name to use."""

    max_tokens: int | None = Field(default=None, alias="max_tokens_to_sample")
    """Denotes the number of tokens to predict per generation.

    If not specified, this is set dynamically using the model's `max_output_tokens`
    from its model profile.

    See docs on [model profiles](https://docs.langchain.com/oss/python/langchain/models#model-profiles)
    for more information.
    """

    temperature: float | None = None
    """A non-negative float that tunes the degree of randomness in generation."""

    top_k: int | None = None
    """Number of most likely tokens to consider at each step."""

    top_p: float | None = None
    """Total probability mass of tokens to consider at each step."""

    default_request_timeout: float | None = Field(None, alias="timeout")
    """Timeout for requests to Claude API."""

        # 每个块都被缩小了（或消息从一开始就是空的）。空块数组不携带任何指令，因此应删除该字段而非发送。`_format_messages`
    # 已经占用了该位置，因此第二个提升的系统消息仍会报错。
    max_retries: int = 2
    """Number of retries allowed for requests sent to the Claude API."""

    stop_sequences: list[str] | None = Field(None, alias="stop")
    """Default stop sequences."""

    anthropic_api_url: str | None = Field(default=None, alias="base_url")
    """Base URL for API requests. Only specify if using a proxy or service emulator.

    If a value isn't passed in, will attempt to read the value first from
    `ANTHROPIC_API_URL` and if that is not set, `ANTHROPIC_BASE_URL`.

    If `LANGSMITH_GATEWAY` is set, it is used as a fallback after those env vars.
    """

    anthropic_api_key: SecretStr = Field(default=SecretStr(""), alias="api_key")
    """Automatically read from env var `ANTHROPIC_API_KEY` if not provided.

    If `LANGSMITH_GATEWAY` is enabled and the base URL points at the gateway,
    `LANGSMITH_GATEWAY_API_KEY` is used instead.
    """

    anthropic_proxy: str | None = Field(
        default_factory=from_env("ANTHROPIC_PROXY", default=None)
    )
    """Proxy to use for the Anthropic clients, will be used for every API call.

    If not provided, will attempt to read from the `ANTHROPIC_PROXY` environment
    variable.
    """

    default_headers: Mapping[str, str] | None = None
    """Headers to pass to the Anthropic clients, will be used for every API call."""

    betas: list[str] | None = None
    """List of beta features to enable. If specified, invocations will be routed
    through `client.beta.messages.create`.

    Example: `#!python betas=["token-efficient-tools-2025-02-19"]`
    """
    # 翻译 v1 内容
    #
        # 只有直接 Anthropic API 接受顶级 `cache_control`。
        # 通过其他传输方式（如 Bedrock）路由的子类会将 `cache_control` kwargs
    # 扩展为块级断点，这是这些传输方式唯一接受的形式。
        # 空的 `formatted_messages`
    # 没有任何内容可供附加断点；静默跳过。下方的警告保留用于处理消息存在但所有候选块均不适用的意外情况。
        # 在应用 `self.thinking` 之前捕获，因此调用时的 `thinking` kwarg
    # 也被视为“显式设置”。

    model_kwargs: dict[str, Any] = Field(default_factory=dict)

    streaming: bool = False
    """Whether to use streaming or not."""

    stream_usage: bool = True
    """Whether to include usage metadata in streaming output.

    If `True`, additional message chunks will be generated during the stream including
    usage metadata.
    """

    thinking: dict[str, Any] | None = Field(default=None)
    """Parameters for Claude reasoning.

    Examples:

    - `#!python {"type": "enabled", "budget_tokens": 10_000}` (pre-4.7 models)
    - `#!python {"type": "adaptive"}` (Opus 4.6+, Opus 5, Opus 5.5, Sonnet 5)
    - `#!python {"type": "adaptive", "display": "summarized"}` (Opus 4.7+,
      Opus 5, Opus 5.5, Sonnet 5)
    - `#!python {"type": "disabled"}` (Opus 5 and Sonnet 5, where adaptive
      thinking is on by default)

    !!! note "Claude Opus 4.7+, Opus 5, Opus 5.5, and Sonnet 5"

        `budget_tokens` is removed on these models — use `{"type": "adaptive"}`
        with `output_config.effort` to control reasoning effort. The default
        `display` is `"omitted"`; set it to `"summarized"` to receive
        summarized reasoning in the response. On Opus 5, disabled thinking is
        supported only at `"high"` effort or below. On Opus 5.5, thinking
        can't be disabled; omit `thinking` and use `output_config.effort`.
    """

    output_config: dict[str, Any] | None = None
    """Configuration options for the model's output.

    Supports the following keys:

    - `effort`: Controls how many tokens Claude uses when responding.
      One of `"max"`, `"xhigh"`, `"high"`, `"medium"`, or `"low"`.
    - `format`: Structured output format configuration (typically set via
      `with_structured_output`).
    - `task_budget`: Advisory token budget for an agentic loop (beta).
      E.g., `#!python {"type": "tokens", "total": 128_000}`.

    Example:

    .. code-block:: python

        ChatAnthropic(
            model="claude-opus-4-7",
            output_config={
                "effort": "xhigh",
                "task_budget": {"type": "tokens", "total": 128_000},
            },
        )

    See Anthropic docs on
    [extended output](https://platform.claude.com/docs/en/api/go/beta/messages/create).
    """

    reasoning_effort: Literal["max", "xhigh", "high", "medium", "low"] | None = Field(
        default=None,
        alias="effort",
    )
    """Reasoning effort.

    Configures `output_config.effort`. If `thinking` isn't set explicitly,
    defaults it to `{"type": "adaptive", "display": "summarized"}`. Can also
    be passed at call time (for example,
    `model.invoke(..., reasoning_effort="high")`).

    !!! note "`effort` alias"

        `effort` is also accepted as an alias for this field, at both
        construction and call time. If both `effort` and `reasoning_effort` are
        set, `effort` wins (Pydantic's alias-resolution precedence).

    !!! note

        On most models, setting `reasoning_effort` to `'high'` produces exactly
        the same behavior as omitting the parameter altogether. On Opus 5.5 the
        default is `'medium'`.

    Example: `reasoning_effort="medium"`
    """

    mcp_servers: list[dict[str, Any]] | None = None
    """List of MCP servers to use for the request.

    Example: `#!python mcp_servers=[{"type": "url", "url": "https://mcp.example.com/mcp",
    "name": "example-mcp"}]`
    """

    context_management: dict[str, Any] | None = None
    """Configuration for
    [context management](https://platform.claude.com/docs/en/build-with-claude/context-editing).
    """

    container: dict[str, Any] | str | None = None
    """Code execution container for the request.

    Either a container ID from a previous response, or a dict of container
    parameters — notably
    [skills](https://platform.claude.com/docs/en/build-with-claude/skills-guide)
    to load into the container. Skills require a
    [code execution](https://docs.langchain.com/oss/python/integrations/chat/anthropic#code-execution)
    tool to be bound.

    ```python
    model = ChatAnthropic(
        model="claude-opus-5",
        container={
            "skills": [{"type": "anthropic", "skill_id": "pptx", "version": "latest"}]
        },
    ).bind_tools([{"type": "code_execution_20260521", "name": "code_execution"}])
    ```

    Can also be passed at call time, which overrides the value set here.
    """

    reuse_last_container: bool | None = None
    """Automatically reuse container from most recent response (code execution).

    When using the built-in
    [code execution tool](https://docs.langchain.com/oss/python/integrations/chat/anthropic#code-execution),
    model responses will include container metadata. Set `reuse_last_container=True`
    to automatically reuse the container from the most recent response for subsequent
    invocations.
    """

    inference_geo: str | None = None
    """Controls where model inference runs. See Anthropic's
    [data residency](https://platform.claude.com/docs/en/build-with-claude/data-residency)
    docs for more information.
    """

    user_profile_id: str | None = None
    """User profile ID to attribute the request to.

    Use when acting on behalf of a party other than your organization. Setting this
    automatically enables the required `user-profiles` beta, routing the request
    through `client.beta.messages.create`.

    Can also be passed at call time, which overrides the value set here (for example,
    `model.invoke(..., user_profile_id="uprof_...")`).
    """

    @property
    def effort(self) -> Literal["max", "xhigh", "high", "medium", "low"] | None:
        """对 `reasoning_effort` 的别名。"""
        return self.reasoning_effort

    @property
    def _llm_type(self) -> str:
        """聊天模型的返回类型。"""
        return "anthropic-chat"

    @property
    def _uses_gateway(self) -> bool:
        """请求是否通过 LangSmith 网关路由。"""
        return self.anthropic_api_key.get_secret_value().startswith("lsv2_")

    @property
    def lc_secrets(self) -> dict[str, str]:
        """返回一个密钥到环境变量的映射。"""
        return {
            "anthropic_api_key": "ANTHROPIC_API_KEY",
            "mcp_servers": "ANTHROPIC_MCP_SERVERS",
        }

    @classmethod
    def is_lc_serializable(cls) -> bool:
        """该类在 langchain 中是否可序列化。"""
        return True

    @classmethod
    def get_lc_namespace(cls) -> list[str]:
        """获取 LangChain 对象的命名空间。

        Returns:
            ["langchain", "chat_models", "anthropic"]
        """
        return ["langchain", "chat_models", "anthropic"]

    @property
    def _identifying_params(self) -> dict[str, Any]:
        """获取标识参数。"""
        return {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_k": self.top_k,
            "top_p": self.top_p,
            "model_kwargs": self.model_kwargs,
            "streaming": self.streaming,
            "max_retries": self.max_retries,
            "default_request_timeout": self.default_request_timeout,
            "thinking": self.thinking,
            "output_config": self.output_config,
        }

    def _get_ls_params(
        self,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> LangSmithParams:
        """获取用于追踪的标准参数。"""
        params = self._get_invocation_params(stop=stop, **kwargs)
        ls_params = LangSmithParams(
            ls_provider="anthropic",
            ls_model_name=params.get("model", self.model),
            ls_model_type="chat",
            ls_temperature=params.get("temperature", self.temperature),
        )
        if ls_max_tokens := params.get("max_tokens", self.max_tokens):
            ls_params["ls_max_tokens"] = ls_max_tokens
        if ls_stop := stop or params.get("stop", None):
            ls_params["ls_stop"] = ls_stop
        return ls_params

    @model_validator(mode="before")
    @classmethod
    def set_default_max_tokens(cls, values: dict[str, Any]) -> Any:
        """从模型配置中设置默认 `max_tokens`，并提供回退值。"""
        if values.get("max_tokens") is None:
            model = values.get("model") or values.get("model_name")
            profile = _get_default_model_profile(model) if model else {}
            values["max_tokens"] = profile.get(
                "max_output_tokens", _FALLBACK_MAX_OUTPUT_TOKENS
            )
        return values

    @model_validator(mode="before")
    @classmethod
    def build_extra(cls, values: dict) -> Any:
        """构建模型参数字典。"""
        all_required_field_names = get_pydantic_field_names(cls)
        return _build_model_kwargs(values, all_required_field_names)

    @model_validator(mode="after")
    def _set_anthropic_version(self) -> Self:
        """在元数据中设置包版本。"""
        self._add_version("langchain-anthropic", __version__)
        return self

    @model_validator(mode="before")
    @classmethod
    def _resolve_gateway(cls, values: Any) -> Any:
        """解析基础 URL 和 API 密钥，并应用 LangSmith 网关设置。

        明确指定的 ``base_url``/``api_key`` 总是优先。否则基础 URL 会回退到
        ``ANTHROPIC_API_URL``/``ANTHROPIC_BASE_URL``，然后是 LangSmith 网关，最后是
        Anthropic 默认值。只有当基础 URL
        来自网关时，网关密钥才被优先使用；对于其他端点，提供方密钥优先，而网关密钥仅在网关启用时才作为候选。
        """
        if isinstance(values, dict):
            _apply_gateway_config(
                values,
                cls,
                base_url_field="anthropic_api_url",
                api_key_field="anthropic_api_key",
                provider_path="anthropic",
                base_url_env=["ANTHROPIC_API_URL", "ANTHROPIC_BASE_URL"],
                api_key_env="ANTHROPIC_API_KEY",
                default_base_url="https://api.anthropic.com",
            )
        return values

    def _resolve_model_profile(self) -> ModelProfile | None:
        profile = _get_default_model_profile(self.model) or None
        if profile is not None and self.betas and "context-1m-2025-08-07" in self.betas:
            profile["max_input_tokens"] = 1_000_000
        return profile

    @cached_property
    def _client_params(self) -> dict[str, Any]:
                # value <= 0 表示该参数应被忽略。None 是一个有意义的值
        default_headers = {"User-Agent": _USER_AGENT}
        if self.default_headers:
            default_headers.update(self.default_headers)

        client_params: dict[str, Any] = {
            "api_key": self.anthropic_api_key.get_secret_value(),
            "base_url": self.anthropic_api_url,
            "max_retries": self.max_retries,
            "default_headers": default_headers,
        }
                # 对于 Anthropic 客户端，此参数的处理方式不同于未指定该参数的情况
                # with). 一个空的块数组不携带任何指令，因此直接忽略
        # all.
        if self.default_request_timeout is None or self.default_request_timeout > 0:
            client_params["timeout"] = self.default_request_timeout

        return client_params

    @cached_property
    def _client(self) -> anthropic.Client:
        client_params = self._client_params
        http_client_params = {"base_url": client_params["base_url"]}
        if "timeout" in client_params:
            http_client_params["timeout"] = client_params["timeout"]
        if self.anthropic_proxy:
            http_client_params["anthropic_proxy"] = self.anthropic_proxy
        http_client = _get_default_httpx_client(**http_client_params)
        params = {
            **client_params,
            "http_client": http_client,
        }
        return anthropic.Client(**params)

    @cached_property
    def _async_client(self) -> anthropic.AsyncClient:
        client_params = self._client_params
        http_client_params = {"base_url": client_params["base_url"]}
        if "timeout" in client_params:
            http_client_params["timeout"] = client_params["timeout"]
        if self.anthropic_proxy:
            http_client_params["anthropic_proxy"] = self.anthropic_proxy
        http_client = _get_default_async_httpx_client(**http_client_params)
        params = {
            **client_params,
            "http_client": http_client,
        }
        return anthropic.AsyncClient(**params)

    def _assert_valid_model_configuration(self, kwargs: Mapping[str, Any]) -> None:
        """根据模型特定的不变性验证已解析的请求配置。"""
        request_config = {**self.model_kwargs, **kwargs}
        thinking = request_config.get("thinking")
        if self.thinking is not None:
            thinking = self.thinking

        output_config = dict(self.output_config or {})
        if self.reasoning_effort:
            output_config["effort"] = self.reasoning_effort
        request_output_config = request_config.get("output_config")
        if isinstance(request_output_config, dict):
            output_config.update(request_output_config)
        effort = request_config.get("effort")
        if effort is None:
            effort = request_config.get("reasoning_effort")
        if effort:
            output_config["effort"] = effort

        is_fable_model = self.model.startswith("claude-fable-5")
        is_sonnet_55 = self.model.startswith("claude-sonnet-5-5")
        if is_fable_model or is_sonnet_55:
            top_k = request_config.get("top_k", self.top_k)
            top_p = request_config.get("top_p", self.top_p)
            temperature = request_config.get("temperature", self.temperature)
            if top_k is not None:
                msg = f"`top_k` is not supported for {self.model}."
                raise ValueError(msg)
            if top_p is not None and top_p != 1:
                msg = (
                    f"`top_p` is not supported for {self.model} at non-default values."
                )
                raise ValueError(msg)
            if temperature is not None and temperature != 1:
                msg = (
                    f"`temperature` is not supported for {self.model} at "
                    "non-default values."
                )
                raise ValueError(msg)
            if isinstance(thinking, Mapping) and thinking.get("type") == "disabled":
                msg = (
                    '`thinking={"type": "disabled"}` is not supported for '
                    f"{self.model}; omit `thinking` to use adaptive thinking."
                )
                raise ValueError(msg)

        if (
            (self.model.startswith("claude-opus-5") or is_fable_model or is_sonnet_55)
            and isinstance(thinking, Mapping)
            and thinking.get("type") == "enabled"
        ):
            msg = (
                '`thinking={"type": "enabled", "budget_tokens": ...}` is not '
                f"supported for {self.model}; use adaptive thinking and "
                "`output_config.effort` instead."
            )
            raise ValueError(msg)

        if (
            self.model.startswith("claude-opus-5")
            and isinstance(thinking, Mapping)
            and thinking.get("type") == "disabled"
            and output_config.get("effort") in {"xhigh", "max"}
        ):
            msg = (
                '`thinking={"type": "disabled"}` is not supported for '
                f"{self.model} with "
                f"`output_config.effort={output_config['effort']!r}`; use adaptive "
                "thinking, omit `thinking`, or set effort to `high` or below."
            )
            raise ValueError(msg)

    def _get_request_payload(
        self,
        input_: LanguageModelInput,
        *,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> dict:
        """获取 Anthropic API 的请求负载。"""
        self._assert_valid_model_configuration(kwargs)
        messages = self._convert_input(input_).to_messages()

        for idx, message in enumerate(messages):
            # 翻译 v1 内容
            if (
                isinstance(message, AIMessage)
                and message.response_metadata.get("output_version") == "v1"
            ):
                tcs: list[types.ToolCall] = [
                    {
                        "type": "tool_call",
                        "name": tool_call["name"],
                        "args": tool_call["args"],
                        "id": tool_call.get("id"),
                    }
                    for tool_call in message.tool_calls
                ]
                messages[idx] = message.model_copy(
                    update={
                        "content": _convert_from_v1_to_anthropic(
                            cast(list[types.ContentBlock], message.content),
                            tcs,
                            message.response_metadata.get("model_provider"),
                        )
                    }
                )

        system, formatted_messages = _format_messages(
            messages,
            model=self.model,
            mid_conversation_system_messages=bool(
                (self.profile or {}).get("mid_conversation_system_messages")
            ),
        )
        if isinstance(system, list) and not system:
            # 每个块都被缩小了（或消息从一开始就是空的）
                        # 字段而非发送它。`_format_messages` 已经
                        # 只有直接的 Anthropic API 接受顶级 `cache_control`。
            # 已占用该槽位，因此第二个提升的系统消息仍会报错。
            system = None

                # 通过其他传输方式（例如 Bedrock）进行路由的子类扩展
                # `cache_control` 参数转换为块级断点，这是唯一的形式
                # 空的 `formatted_messages` 没有任何可设置断点的内容；
        # 这些传输方式接受。
        if not _is_direct_anthropic_llm_type(getattr(self, "_llm_type", None)):
            cache_control = kwargs.pop("cache_control", None)
                        # 在下方应用 `self.thinking` 之前捕获，因此是调用时
            # 静默跳过。以下警告保留用于意外情况
            # 消息存在但每个候选块均不适用的情况。
            if cache_control and formatted_messages:
                code_execution_tool_ids = _collect_code_execution_tool_ids(
                    formatted_messages
                )
                applied = _apply_cache_control_to_last_eligible_block(
                    formatted_messages, cache_control, code_execution_tool_ids
                )
                if not applied:
                    warnings.warn(
                        "`cache_control` kwarg was dropped: no eligible "
                        "content block found (all candidates are "
                        "`code_execution`-related, which Anthropic forbids "
                        "breakpoints on).",
                        UserWarning,
                        stacklevel=2,
                    )

        payload = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": formatted_messages,
            "temperature": self.temperature,
            "top_k": self.top_k,
            "top_p": self.top_p,
            "stop_sequences": stop or self.stop_sequences,
            "betas": self.betas,
            "context_management": self.context_management,
            "mcp_servers": self.mcp_servers,
            "container": self.container,
            "user_profile_id": self.user_profile_id,
            "system": system,
            **self.model_kwargs,
            **kwargs,
        }
                # `thinking` 参数关键字也被视为"显式设置"。
                # 处理 output_config 和 effort 参数
        thinking_explicitly_set = "thinking" in payload or self.thinking is not None
        if self.thinking is not None:
            payload["thinking"] = self.thinking
        if self.inference_geo is not None:
            payload["inference_geo"] = self.inference_geo
        if self.model.startswith("claude-fable-5"):
            payload.pop("temperature", None)
            payload.pop("top_k", None)
            payload.pop("top_p", None)

                # 优先级：关键字参数 `effort`/`reasoning_effort` > 关键字参数 `output_config`
                # > self.reasoning_effort > self.output_config
                # `reasoning_effort` 和其别名 `effort` 均非 Anthropic API 字段
        output_config: dict[str, Any] = {}
        if self.output_config:
            output_config.update(self.output_config)
        reasoning_effort_applied = False
        if self.reasoning_effort:
            output_config["effort"] = self.reasoning_effort
            reasoning_effort_applied = True
        payload_oc = payload.get("output_config")
        if isinstance(payload_oc, dict):
            output_config.update(payload_oc)

                # 若在调用时两者均设置，则 `effort`
        # 优先，与构造时的别名解析优先级一致（`Field(alias="effort")`）
        # 将其移除，以确保它们不会作为顶层键泄漏
        effort_kwarg = payload.pop("effort", None)
        reasoning_effort_kwarg = payload.pop("reasoning_effort", None)
                # 当 `reasoning_effort` 设置时，默认启用自适应思考，除非调用方显式提供了
        # `thinking`。该功能受 `xhigh` 支持限制：仅 Opus 4.7+/Sonnet 5
        # 支持自适应+摘要形式的 `thinking` 结构——向较旧模型（如 Opus
        # 4.5、4.6）发送此结构会被 API 拒绝，提示“此模型不支持自适应思考”
                # 使用 agents.create_agent 的 ProviderStrategy 时，response_format 存在
        reasoning_effort_override = (
            effort_kwarg if effort_kwarg is not None else reasoning_effort_kwarg
        )
        if reasoning_effort_override:
            output_config["effort"] = reasoning_effort_override
            reasoning_effort_applied = True

        if output_config:
            payload["output_config"] = output_config

                # 使用 agents.create_agent 的 ProviderStrategy 时，response_format 存在
                # ProviderStrategy 将其转换为 OpenAI 风格格式，该格式将参数传递给
        # ChatAnthropic，最终进入我们的 payload
                # 将 OpenAI 风格的 response_format 转换为 Anthropic 的 output_config.format
                # 处理已弃用的 output_format 参数以保证向后兼容性
        # 重用最近响应中的容器（代码执行）
        if (
            reasoning_effort_applied
            and not thinking_explicitly_set
            and "xhigh" in _reasoning_effort_levels(self.profile)
        ):
            payload["thinking"] = {"type": "adaptive", "display": "summarized"}

        if "response_format" in payload:
            # 注意：结构化输出不再需要测试版头信息
            # ---
                        # (output_config.format 或严格工具使用) 因为这些功能现已普遍可用
                        # 根据特定工具类型和 input_examples 自动追加所需的 beta 标志
            response_format = payload.pop("response_format")
            if (
                isinstance(response_format, dict)
                and response_format.get("type") == "json_schema"
                and "schema" in response_format.get("json_schema", {})
            ):
                response_format = cast(dict, response_format["json_schema"]["schema"])
                        # 检查 input_examples
            output_config = payload.setdefault("output_config", {})
            output_config["format"] = _convert_to_anthropic_output_config_format(
                response_format
            )

                # 自动追加 input_examples 所需的头信息
        if "output_format" in payload:
            warnings.warn(
                "The 'output_format' parameter is deprecated and will be removed in "
                "langchain-anthropic 2.0.0. Use 'output_config={\"format\": ...}' "
                "instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            output_config = payload.setdefault("output_config", {})
            output_config["format"] = payload.pop("output_format")

        container = payload.get("container")

        if self.reuse_last_container and not _container_id(container):
                        # 自动追加 mcp_servers 所需的 beta 标志
            for message in reversed(messages):
                if (
                    isinstance(message, AIMessage)
                    and isinstance(
                        last_container := message.response_metadata.get("container"),
                        dict,
                    )
                    and (container_id := last_container.get("id"))
                ):
                    payload["container"] = (
                        {**container, "id": container_id}
                        if isinstance(container, dict)
                        else container_id
                    )
                    break

        if (
            isinstance(container, dict)
            and container.get("skills")
            and not any(
                isinstance(tool, dict)
                and str(tool.get("type", "")).startswith("code_execution")
                for tool in (payload.get("tools") or [])
            )
        ):
            warnings.warn(
                "Skills require a code execution tool to be bound, e.g. "
                '`bind_tools([{"type": "code_execution_20260521", '
                '"name": "code_execution"}])`.',
                UserWarning,
                stacklevel=2,
            )

                # 若不存在则追加到现有 beta 标志中
                # 自动追加 task_budget 所需的 beta 标志
        if "tools" in payload and isinstance(payload["tools"], list):
                        # 自动追加 `updates` 思考显示模式所需的 beta 标志
            has_input_examples = False
            for tool in payload["tools"]:
                if isinstance(tool, dict):
                    tool_type = tool.get("type")
                    if tool_type and tool_type in _TOOL_TYPE_TO_BETA:
                        required_beta = _TOOL_TYPE_TO_BETA[tool_type]
                        if payload["betas"]:
                            if required_beta not in payload["betas"]:
                                payload["betas"] = [
                                    *payload["betas"],
                                    required_beta,
                                ]
                        else:
                            payload["betas"] = [required_beta]
                                        # 自动追加 user_profile_id 所需的 beta 标志
                    if tool.get("input_examples"):
                        has_input_examples = True

                        # 参考：Anthropic SDK 流式实现
            if has_input_examples:
                required_beta = "advanced-tool-use-2025-11-20"
                if payload["betas"]:
                    if required_beta not in payload["betas"]:
                        payload["betas"] = [*payload["betas"], required_beta]
                else:
                    payload["betas"] = [required_beta]

                # https://github.com/anthropics/anthropic-sdk-python/blob/main/src/anthropic/lib/streaming/_messages.py
        # # noqa: E501
        if payload.get("mcp_servers"):
            required_beta = "mcp-client-2025-11-20"
            if payload["betas"]:
                                # 捕获模型名称，但暂不包含 usage_metadata
                if required_beta not in payload["betas"]:
                    payload["betas"] = [*payload["betas"], required_beta]
            else:
                payload["betas"] = [required_beta]

                # 因为它将在 message_delta 中以完整信息正确报告
        resolved_oc = payload.get("output_config")
        if isinstance(resolved_oc, dict) and resolved_oc.get("task_budget"):
            required_beta = "task-budgets-2026-03-13"
            if payload.get("betas"):
                if required_beta not in payload["betas"]:
                    payload["betas"] = [*payload["betas"], required_beta]
            else:
                payload["betas"] = [required_beta]

                # 在某些情况下，解析后的参数以 start 事件形式呈现，且无后续 input_json_delta
        # 事件
        thinking = payload.get("thinking")
        if isinstance(thinking, dict) and thinking.get("display") == "updates":
            required_beta = "thinking-display-updates-2026-08-18"
            if payload.get("betas"):
                if required_beta not in payload["betas"]:
                    payload["betas"] = [*payload["betas"], required_beta]
            else:
                payload["betas"] = [required_beta]

        system_contents = [
            message.get("content")
            for message in (payload.get("messages") or [])
            if message.get("role") == "system"
        ]
        has_tool_change = any(_has_tool_change_block(c) for c in system_contents)
        has_inline_definition = any(
            _has_inline_tool_definition(c) for c in system_contents
        )
        explicit_betas = payload.get("betas") or []
        tool_change_beta = (
            _INLINE_TOOLS_BETA
            if has_inline_definition
            else (
                _MID_CONVERSATION_TOOL_CHANGES_BETA
                if has_tool_change and _INLINE_TOOLS_BETA not in explicit_betas
                else None
            )
        )
        if tool_change_beta and tool_change_beta not in explicit_betas:
            payload["betas"] = [*explicit_betas, tool_change_beta]

                # Anthropic 可能将文本或思考块的起始内容直接放在 `content_block_start`
        # 事件中，而非后续的 delta
        # 中。这在工具结果后的助手回合中很常见。在此处发出初始内容，以免其在聚合消息中丢失。后续的
        # deltas 作为共享该块 `index` 的独立块发出；稍后通过 chunk addition
        # (`AIMessageChunk.__add__`) 将其合并为一个块
        if payload.get("user_profile_id"):
            required_beta = "user-profiles-2026-03-24"
            if payload.get("betas"):
                if required_beta not in payload["betas"]:
                    payload["betas"] = [*payload["betas"], required_beta]
            else:
                payload["betas"] = [required_beta]

        return _route_unsupported_sampling_params(
            {k: v for k, v in payload.items() if v is not None}
        )

    def _create(self, payload: dict) -> Any:
        try:
            if "betas" in payload:
                return self._client.beta.messages.with_raw_response.create(**payload)
            return self._client.messages.with_raw_response.create(**payload)
        except TypeError as e:
            _raise_if_authentication_error(e)
            raise

    async def _acreate(self, payload: dict) -> Any:
        try:
            if "betas" in payload:
                return await self._async_client.beta.messages.with_raw_response.create(
                    **payload
                )
            return await self._async_client.messages.with_raw_response.create(**payload)
        except TypeError as e:
            _raise_if_authentication_error(e)
            raise

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        *,
        stream_usage: bool | None = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        if stream_usage is None:
            stream_usage = self.stream_usage
        kwargs["stream"] = True
        payload = self._get_request_payload(messages, stop=stop, **kwargs)
        try:
            raw_response = self._create(payload)
            base_generation_info: dict[str, Any] = {}
            if self._uses_gateway:
                _add_gateway_metadata(base_generation_info, raw_response)
            stream = raw_response.parse()
            coerce_content_to_string = (
                not _tools_in_params(payload)
                and not _documents_in_params(payload)
                and not _thinking_in_params(payload)
                and not _compact_in_params(payload)
            )
            block_start_event = None
            is_first_chunk = True
            for event in stream:
                msg, block_start_event = self._make_message_chunk_from_anthropic_event(
                    event,
                    stream_usage=stream_usage,
                    coerce_content_to_string=coerce_content_to_string,
                    block_start_event=block_start_event,
                )
                if msg is not None:
                    chunk = ChatGenerationChunk(
                        message=msg,
                        generation_info=base_generation_info
                        if is_first_chunk
                        else None,
                    )
                    is_first_chunk = False
                    if run_manager and isinstance(msg.content, str):
                        run_manager.on_llm_new_token(msg.content, chunk=chunk)
                    yield chunk
        except anthropic.BadRequestError as e:
            _handle_anthropic_bad_request(e)
        except anthropic.APIError as e:
            _handle_anthropic_api_error(e)

    async def _astream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        *,
        stream_usage: bool | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        if stream_usage is None:
            stream_usage = self.stream_usage
        kwargs["stream"] = True
        payload = self._get_request_payload(messages, stop=stop, **kwargs)
        try:
            raw_response = await self._acreate(payload)
            base_generation_info: dict[str, Any] = {}
            if self._uses_gateway:
                _add_gateway_metadata(base_generation_info, raw_response)
            stream = await _aparse(raw_response)
            coerce_content_to_string = (
                not _tools_in_params(payload)
                and not _documents_in_params(payload)
                and not _thinking_in_params(payload)
                and not _compact_in_params(payload)
            )
            block_start_event = None
            is_first_chunk = True
            async for event in stream:
                msg, block_start_event = self._make_message_chunk_from_anthropic_event(
                    event,
                    stream_usage=stream_usage,
                    coerce_content_to_string=coerce_content_to_string,
                    block_start_event=block_start_event,
                )
                if msg is not None:
                    chunk = ChatGenerationChunk(
                        message=msg,
                        generation_info=base_generation_info
                        if is_first_chunk
                        else None,
                    )
                    is_first_chunk = False
                    if run_manager and isinstance(msg.content, str):
                        await run_manager.on_llm_new_token(msg.content, chunk=chunk)
                    yield chunk
        except anthropic.BadRequestError as e:
            _handle_anthropic_bad_request(e)
        except anthropic.APIError as e:
            _handle_anthropic_api_error(e)

    def _make_message_chunk_from_anthropic_event(
        self,
        event: anthropic.types.RawMessageStreamEvent,
        *,
        stream_usage: bool = True,
        coerce_content_to_string: bool,
        block_start_event: anthropic.types.RawMessageStreamEvent | None = None,
    ) -> tuple[AIMessageChunk | None, anthropic.types.RawMessageStreamEvent | None]:
        """将 Anthropic 流式事件转换为 `AIMessageChunk`。

        Args:
            event: 来自 Anthropic SDK 的原始流式事件 stream_usage:
            是否在输出块中包含使用量元数据。 coerce_content_to_string:
            是否将结构化内容转换为纯文本字符串。

                当为 `True` 时，仅保留文本内容；当为 `False`
                时，保留工具调用和引用等结构化内容。 block_start_event:
                上一个内容块开始事件，用于跟踪工具使用块并跨相关事件保持上下文。

        Returns:
            元组包含 - `AIMessageChunk`:
            转换后的消息块，带有适当的内容和元数据，如果事件不产生块则为 `None` -
            `RawMessageStreamEvent`: 用于跨连续事件跟踪内容块的更新
            `block_start_event`，如果不适用则为 `None`

        Note:
            并非所有 Anthropic 事件都会生成消息块。例如内部状态变更事件会返回 `None`
            作为消息块，但可能更新 `block_start_event` 以用于上下文追踪。
        """
        message_chunk: AIMessageChunk | None = None
                # thinking
        # 处理增量内容更新
        if event.type == "message_start" and stream_usage:
                        # 文本和引用 delta（增量文本内容）
                        # 所有引用 delta 均属于文本块
            if hasattr(event.message, "model"):
                response_metadata: dict[str, Any] = {"model_name": event.message.model}
            else:
                response_metadata = {}

            message_chunk = AIMessageChunk(
                content="" if coerce_content_to_string else [],
                response_metadata=response_metadata,
            )

        elif (
            event.type == "content_block_start"
            and event.content_block is not None
            and (
                "tool_result" in event.content_block.type
                or "tool_use" in event.content_block.type
                or "document" in event.content_block.type
                or "redacted_thinking" in event.content_block.type
            )
        ):
            if coerce_content_to_string:
                warnings.warn("Received unexpected tool content block.", stacklevel=2)

            content_block = event.content_block.model_dump()
            if event.content_block.type == "advisor_tool_result":
                content_block = {
                    key: content_block[key]
                    for key in ("type", "tool_use_id", "content", "cache_control")
                    if key in content_block
                }
            if "caller" in content_block and content_block["caller"] is None:
                content_block.pop("caller")
            content_block["index"] = event.index
            if event.content_block.type == "tool_use":
                if (
                    parsed_args := getattr(event.content_block, "input", None)
                ) and isinstance(parsed_args, dict):
                    # 将引用分配至列表（若存在）
                    # 推理
                    args = json.dumps(parsed_args)
                else:
                    args = ""
                tool_call_chunk = create_tool_call_chunk(
                    index=event.index,
                    id=event.content_block.id,
                    name=event.content_block.name,
                    args=args,
                )
                tool_call_chunks = [tool_call_chunk]
            else:
                tool_call_chunks = []
            message_chunk = AIMessageChunk(
                content=[content_block],
                tool_call_chunks=tool_call_chunks,
            )
            block_start_event = event

        elif (
            event.type == "content_block_start"
            and event.content_block is not None
            and event.content_block.type in ("text", "thinking")
        ):
                        # 工具输入 JSON（流式工具参数）
            # 压缩块
            # 处理最终的使用元数据和完成信息
                        # 标记最终的 Anthropic 流式块
                        # 未处理的事件类型（例如 `content_block_stop`、`ping` 事件）
                        # https://platform.claude.com/docs/en/build-with-claude/streaming#other-events
                        # (`AIMessageChunk.__add__`) 后续会将其合并为一个块。
            block_start_event = event
            if event.content_block.type == "text":
                text = getattr(event.content_block, "text", "") or ""
                if text:
                    if coerce_content_to_string:
                        message_chunk = AIMessageChunk(content=text)
                    else:
                        content_block = event.content_block.model_dump()
                        content_block["index"] = event.index
                        if content_block.get("citations") is None:
                            content_block.pop("citations", None)
                        message_chunk = AIMessageChunk(content=[content_block])
            else:  # 思考
                thinking = getattr(event.content_block, "thinking", "") or ""
                signature = getattr(event.content_block, "signature", "") or ""
                if thinking or signature:
                    content_block = event.content_block.model_dump()
                    content_block["index"] = event.index
                    content_block["type"] = "thinking"
                    message_chunk = AIMessageChunk(content=[content_block])

        # 处理增量内容更新
        elif event.type == "content_block_delta":
            # 文本和引用增量（增量式文本内容）
            if event.delta.type in ("text_delta", "citations_delta"):
                if coerce_content_to_string and hasattr(event.delta, "text"):
                    text = getattr(event.delta, "text", "")
                    message_chunk = AIMessageChunk(content=text)
                else:
                    content_block = event.delta.model_dump()
                    content_block["index"] = event.index

                    # 所有引用差异都属于一个文本块
                    content_block["type"] = "text"
                    if "citation" in content_block:
                        # 将引用分配给列表（如果存在）
                        content_block["citations"] = [content_block.pop("citation")]
                    message_chunk = AIMessageChunk(content=[content_block])

            # 推理
            elif event.delta.type in {"thinking_delta", "signature_delta"}:
                content_block = event.delta.model_dump()
                content_block["index"] = event.index
                content_block["type"] = "thinking"
                if event.delta.type == "signature_delta":
                    content_block.setdefault("thinking", "")
                message_chunk = AIMessageChunk(content=[content_block])

                        # 工具输入 JSON（流式工具参数）
            elif event.delta.type == "input_json_delta":
                content_block = event.delta.model_dump()
                content_block["index"] = event.index
                start_event_block = (
                    getattr(block_start_event, "content_block", None)
                    if block_start_event
                    else None
                )
                if (
                    start_event_block is not None
                    and getattr(start_event_block, "type", None) == "tool_use"
                ):
                    tool_call_chunk = create_tool_call_chunk(
                        index=event.index,
                        id=None,
                        name=None,
                        args=event.delta.partial_json,
                    )
                    tool_call_chunks = [tool_call_chunk]
                else:
                    tool_call_chunks = []
                message_chunk = AIMessageChunk(
                    content=[content_block],
                    tool_call_chunks=tool_call_chunks,
                )

            # 压缩块
            elif event.delta.type == "compaction_delta":
                content_block = event.delta.model_dump()
                content_block["index"] = event.index
                content_block["type"] = "compaction"
                if (
                    "encrypted_content" in content_block
                    and content_block["encrypted_content"] is None
                ):
                    content_block.pop("encrypted_content")
                message_chunk = AIMessageChunk(content=[content_block])

        # 处理最终的使用元数据和完成信息
        elif event.type == "message_delta" and stream_usage:
            usage_metadata = _create_usage_metadata(event.usage)
            response_metadata = {
                "stop_reason": event.delta.stop_reason,
                "stop_sequence": event.delta.stop_sequence,
            }
            if stop_details := event.delta.model_dump().get("stop_details"):
                response_metadata["stop_details"] = stop_details
            if context_management := getattr(event, "context_management", None):
                response_metadata["context_management"] = (
                    context_management.model_dump()
                )
            message_delta = getattr(event, "delta", None)
            if message_delta and (
                container := getattr(message_delta, "container", None)
            ):
                response_metadata["container"] = container.model_dump(mode="json")
            message_chunk = AIMessageChunk(
                content="" if coerce_content_to_string else [],
                usage_metadata=usage_metadata,
                response_metadata=response_metadata,
            )
            if message_chunk.response_metadata.get("stop_reason"):
                                # 标记最终的 Anthropic 流式数据块
                message_chunk.chunk_position = "last"
                # 未处理的事件类型（例如 `content_block_stop`、`ping` 事件）
                # https://platform.claude.com/docs/en/build-with-claude/streaming#other-events
        else:
            pass

        if message_chunk:
            message_chunk.response_metadata["model_provider"] = "anthropic"
        return message_chunk, block_start_event

    def _format_output(
        self,
        data: Any,
        *,
        generation_info: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        """将 Anthropic API 的输出格式化为 LC 格式。"""
        data_dict = data.model_dump()
        content = data_dict["content"]

                # 如果引用为 None 则移除 - 由 anthropic sdk 0.45 引入
        for block in content:
            if isinstance(block, dict):
                if "citations" in block and block["citations"] is None:
                    block.pop("citations")
                if "caller" in block and block["caller"] is None:
                    block.pop("caller")
                if "encrypted_content" in block and block["encrypted_content"] is None:
                    block.pop("encrypted_content")
                if (
                    block.get("type") == "thinking"
                    and "text" in block
                    and block["text"] is None
                ):
                    block.pop("text")

        llm_output = {
            k: v for k, v in data_dict.items() if k not in ("content", "role", "type")
        }
        if (
            (container := llm_output.get("container"))
            and isinstance(container, dict)
            and (expires_at := container.get("expires_at"))
            and isinstance(expires_at, datetime.datetime)
        ):
                        # TODO: 将所有 `data` 以 `mode="json"` 格式导出
            llm_output["container"]["expires_at"] = expires_at.isoformat()
        response_metadata = {"model_provider": "anthropic"}
        if "model" in llm_output and "model_name" not in llm_output:
            llm_output["model_name"] = llm_output["model"]
        if (
            len(content) == 1
            and content[0]["type"] == "text"
            and not content[0].get("citations")
        ):
            msg = AIMessage(
                content=content[0]["text"], response_metadata=response_metadata
            )
        elif any(block["type"] == "tool_use" for block in content):
            tool_calls = extract_tool_calls(content)
            msg = AIMessage(
                content=content,
                tool_calls=tool_calls,
                response_metadata=response_metadata,
            )
        else:
            msg = AIMessage(content=content, response_metadata=response_metadata)
        msg.usage_metadata = _create_usage_metadata(data.usage)
        return ChatResult(
            generations=[
                ChatGeneration(message=msg, generation_info=generation_info or None)
            ],
            llm_output=llm_output,
        )

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        payload = self._get_request_payload(messages, stop=stop, **kwargs)
        try:
            raw_response = self._create(payload)
        except anthropic.BadRequestError as e:
            _handle_anthropic_bad_request(e)
        except anthropic.APIError as e:
            _handle_anthropic_api_error(e)
        generation_info: dict[str, Any] = {}
        if self._uses_gateway:
            _add_gateway_metadata(generation_info, raw_response)
        return self._format_output(
            raw_response.parse(), generation_info=generation_info, **kwargs
        )

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        payload = self._get_request_payload(messages, stop=stop, **kwargs)
        try:
            raw_response = await self._acreate(payload)
        except anthropic.BadRequestError as e:
            _handle_anthropic_bad_request(e)
        except anthropic.APIError as e:
            _handle_anthropic_api_error(e)
        generation_info: dict[str, Any] = {}
        if self._uses_gateway:
            _add_gateway_metadata(generation_info, raw_response)
        return self._format_output(
            await _aparse(raw_response), generation_info=generation_info, **kwargs
        )

    def _get_llm_for_structured_output_when_thinking_is_enabled(
        self,
        schema: dict | type,
        formatted_tool: AnthropicTool,
    ) -> Runnable[LanguageModelInput, BaseMessage]:
        thinking_admonition = (
            "You are attempting to use structured output via forced tool calling, "
            "which is not supported when `thinking` is enabled or on "
            f"{self.model}. This method will raise an OutputParserException if tool "
            "calls are not generated. Consider `method='json_schema'`, disabling "
            "`thinking` where supported, or adjusting your prompt to ensure the "
            "tool is called."
        )
        warnings.warn(thinking_admonition, stacklevel=2)
        llm = self.bind_tools(
            [schema],
                        # 此处未指定 tool_choice，因为 API 会拒绝在 thinking=true 时强制工具调用
                        # 当 thinking=true 时，API 会拒绝强制工具调用
            ls_structured_output_format={
                "kwargs": {"method": "function_calling"},
                "schema": formatted_tool,
            },
        )

        def _raise_if_no_tool_calls(message: AIMessage) -> AIMessage:
            if not message.tool_calls:
                raise OutputParserException(thinking_admonition)
            return message

        return llm | _raise_if_no_tool_calls

    def bind_tools(
        self,
        tools: Sequence[Mapping[str, Any] | type | Callable | BaseTool],
        *,
        tool_choice: dict[str, str] | str | None = None,
        parallel_tool_calls: bool | None = None,
        strict: bool | None = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, AIMessage]:
        r"""将工具类对象绑定到 `ChatAnthropic`。

        Args:
            tools: 要绑定到此聊天模型的工具定义列表。

                支持 Anthropic 格式的工具模式，以及由
                [`convert_to_openai_tool`][langchain_core.utils.function_calling.convert_to_openai_tool]
                处理的任何工具定义。 tool_choice: 要求模型调用的工具。选项包括：

                - 工具名称作为字符串或字典 `{"type": "tool", "name":
                "<<tool_name>>"}`：调用对应工具 - `'auto'`, `{"type: "auto"}`, 或
                `None`：自动选择工具（包括不调用工具） - `'any'` 或 `{"type:
                "any"}`：强制至少调用一个工具 parallel_tool_calls: 设置为 `False`
                以禁用并行工具调用。

                默认为 `None`（无指定，允许并行使用工具）。

                !!! version-added "在 `langchain-anthropic` 0.3.2 中新增"

                strict: 若为 `True`，则对工具调用应用 Claude 的模式遵循性。 参见
                [文档](https://docs.langchain.com/oss/python/integrations/chat/anthropic#strict-tool-use)
                获取更多信息。 kwargs: 任何其他参数将直接传递给 `bind`。

        Raises:
            ValueError: 如果在使用顶层模式组合时，`tools`
            中的所有工具均被丢弃，导致模型无法调用任何工具。 ValueError: 如果
            `tool_choice` 强制使用工具（特定工具或
            `'any'`），而任意工具被丢弃，则强制的工具可能不可达。当启用 `thinking`
            时此情况不适用，因为强制选择会在请求发送前被丢弃。 ValueError: 如果
            `tool_choice` 既不是 `dict`，也不是 `str` 或 `None`。

        Example:
            ```python
            from langchain_anthropic import ChatAnthropic
            from pydantic import BaseModel, Field


            class GetWeather(BaseModel):
                '''获取指定位置的当前天气'''

                location: str = Field(..., description="城市和州，例如旧金山，加利福尼亚州")


            class GetPrice(BaseModel):
                '''获取特定产品的价格。'''

                product: str = Field(..., description="要查询的产品。")


            model = ChatAnthropic(model="claude-sonnet-4-5-20250929", temperature=0)
            model_with_tools = model.bind_tools([GetWeather, GetPrice])
            model_with_tools.invoke(
                "What is the weather like in San Francisco",
            )
            # -> AIMessage(
            #     content=[
            #         {'text': '<thinking>\n根据用户的问题，需要调用的相关函数是 GetWeather，该函数需要 "location" 参数。\n\n用户已明确指定位置为 "San Francisco"。由于旧金山是知名城市，我可以合理推断其意为旧金山，加利福尼亚州，无需额外提供州名。\n\n所有必需参数均已提供，因此可以继续执行 API 调用。\n</thinking>', 'type': 'text'},
            #         {'text': None, 'type': 'tool_use', 'id': 'toolu_01SCgExKzQ7eqSkMHfygvYuu', 'name': 'GetWeather', 'input': {'location': 'San Francisco, CA'}}
            #     ],
            #     response_metadata={'id': 'msg_01GM3zQtoFv8jGQMW7abLnhi', 'model': 'claude-sonnet-4-5-20250929', 'stop_reason': 'tool_use', 'stop_sequence': None, 'usage': {'input_tokens': 487, 'output_tokens': 145}},
            #     id='run-87b1331e-9251-4a68-acef-f0a018b639cc-0'
            # )
            ```
        """  # noqa: E501
        # 允许通过以下方式使用内置工具：
                # - 原始 `dict` 格式
                # - 若提供了 BaseTool 的 extras["provider_tool_definition"]，则提取该值
        formatted_tools: list[Mapping[str, Any]] = [
            tool
            if _is_builtin_tool(tool)
            else convert_to_anthropic_tool(tool, strict=strict)
            for tool in tools
        ]
        formatted_tools, dropped_tool_names = _drop_unsupported_root_composition_tools(
            formatted_tools
        )

        # 丢弃工具时若仍有可用工具可恢复请求。若所有工具均被丢弃，则无法恢复：调用者请求了带有工具的模型，却得到一个无法调用任何工具的模型，因此应立即报错而非让后续出现永远无法执行的工具调用
        # 丢弃工具后仍存在可用工具时可恢复请求。若所有工具均被丢弃，则无法恢复：调用者请求了带有工具的模型，却得到一个无法调用任何工具的模型，因此应立即报错而非让后续出现永远无法执行的工具调用
        # 丢弃工具后仍存在可用工具时可恢复请求。若所有工具均被丢弃，则无法恢复：调用者请求了带有工具的模型，却得到一个无法调用任何工具的模型，因此应立即报错而非让后续出现永远无法执行的工具调用
                # 将 tool_choice
        # 与过滤后的列表进行协调：强制调用已被丢弃的工具，或在调用者依赖的工具集无声缩小的情况下强制使用工具，仍会导致
        # 400 错误或智能体循环卡死
        if tools and not formatted_tools:
            msg = (
                f"All {len(tools)} bound tool(s) use a top-level "
                f"{'/'.join(_TOP_LEVEL_SCHEMA_COMPOSITION_KEYS)} in their "
                f"input_schema, which the Anthropic API rejects: "
                f"{sorted(dropped_tool_names)}. No tool is left for the model to "
                "call. If you control the schema, move the combinator under "
                "`properties`; for structured output, `with_structured_output(..., "
                "method='json_schema')` accepts these schemas directly. Otherwise "
                "these tools cannot be used with Anthropic -- bind a subset that "
                "excludes them, or raise it with the tool's author."
            )
            raise ValueError(msg)

        # 思考模式会在请求发送前丢弃强制选择，因此无需指向过滤后仍存在的工具
        # 强制使用工具意味着调用者依赖于特定可达的工具集，因此丢失其中任意成员都是致命的——不仅仅是全部丢失
        # 强制使用工具意味着调用者依赖于特定可达的工具集，因此丢失其中任意成员都是致命的——不仅仅是全部丢失
        if tool_choice and dropped_tool_names:
            choice_type: str | None = None
            choice_name: str | None = None
            if isinstance(tool_choice, dict):
                choice_type = tool_choice.get("type")
                choice_name = tool_choice.get("name")
            elif isinstance(tool_choice, str):
                if tool_choice in ("any", "auto"):
                    choice_type = tool_choice
                else:
                    choice_type, choice_name = "tool", tool_choice
                        # Anthropic API 在启用思考模式时拒绝强制工具使用：
                        # "Thinking may not be enabled when tool_choice forces tool
            # use."（启用思考时不能强制工具使用）
            thinking_discards_forced_choice = (
                self.thinking is not None
                and self.thinking.get("type") in ("enabled", "adaptive")
                and choice_type in ("any", "tool")
            )
            if not thinking_discards_forced_choice:
                if choice_type == "tool" and choice_name in dropped_tool_names:
                    msg = (
                        f"tool_choice forces {choice_name!r}, but that tool was "
                        "dropped because its input_schema uses a top-level "
                        f"{'/'.join(_TOP_LEVEL_SCHEMA_COMPOSITION_KEYS)}, which "
                        "the Anthropic API rejects. Stop forcing it, or -- if you "
                        "control the schema -- move the combinator under "
                        "`properties`. For structured output, "
                        "`with_structured_output(..., method='json_schema')` "
                        "accepts these schemas directly."
                    )
                    raise ValueError(msg)
                if choice_type == "any":
                                        # 删除强制的 tool_choice 并发出警告，与
                    # _get_llm_for_structured_output_when_thinking_is_enabled
                    # 中的行为保持一致
                                        # TODO: 在未来版本中将 `method="json_schema"` 设为默认值
                                        # convert_to_anthropic_tool 对 'method=function_calling'
                    # 的结果将始终是 AnthropicTool
                    msg = (
                        "tool_choice='any' forces the model to call a tool, but "
                        f"{sorted(dropped_tool_names)} were dropped because their "
                        "input_schema uses a top-level "
                        f"{'/'.join(_TOP_LEVEL_SCHEMA_COMPOSITION_KEYS)}, which "
                        "the Anthropic API rejects, so the model can no longer "
                        "call them. Use tool_choice='auto' to proceed with the "
                        "remaining tools, or -- if you control the schema -- move "
                        "the combinator under `properties`."
                    )
                    raise ValueError(msg)

        if not tool_choice:
            pass
        elif isinstance(tool_choice, dict):
            kwargs["tool_choice"] = tool_choice.copy()
        elif isinstance(tool_choice, str) and tool_choice in ("any", "auto"):
            kwargs["tool_choice"] = {"type": tool_choice}
        elif isinstance(tool_choice, str):
            kwargs["tool_choice"] = {"type": "tool", "name": tool_choice}
        else:
            msg = (
                f"Unrecognized 'tool_choice' type {tool_choice=}. Expected dict, "
                f"str, or None."
            )
            raise ValueError(
                msg,
            )

        # 强制工具调用
                # 过滤与 `bind_tools` 丢弃的相同 schema，确保统计 token 和发送请求时对 API
        # 接受的工具达成一致
        # 传递内置工具定义
                # Anthropic 工具格式
        if (
            self.thinking is not None
            and self.thinking.get("type") in ("enabled", "adaptive")
            and "tool_choice" in kwargs
            and kwargs["tool_choice"].get("type") in ("any", "tool")
        ):
            warnings.warn(
                "tool_choice is forced but thinking is enabled. The Anthropic "
                "API does not support forced tool use with thinking. "
                "Dropping tool_choice to avoid an API error. Tool calls are "
                "not guaranteed. Consider disabling thinking or adjusting "
                "your prompt to ensure the tool is called.",
                stacklevel=2,
            )
            del kwargs["tool_choice"]

        if parallel_tool_calls is not None:
            disable_parallel_tool_use = not parallel_tool_calls
            if "tool_choice" in kwargs:
                kwargs["tool_choice"]["disable_parallel_tool_use"] = (
                    disable_parallel_tool_use
                )
            else:
                kwargs["tool_choice"] = {
                    "type": "auto",
                    "disable_parallel_tool_use": disable_parallel_tool_use,
                }

        return self.bind(tools=formatted_tools, **kwargs)

    def with_structured_output(
        self,
        schema: dict | type,
        *,
        include_raw: bool = False,
        method: Literal["function_calling", "json_schema"] = "function_calling",
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, dict | BaseModel]:
        """返回格式与给定模式匹配的模型包装器。

        参见 [LangChain
        文档](https://docs.langchain.com/oss/python/integrations/chat/anthropic#structured-output)
        获取更多详细信息和示例。

        Args:
            schema: 输出模式。可传入以下形式：

                - Anthropic 工具模式，
                - OpenAI 函数/工具模式，
                - JSON Schema，
                - `TypedDict` 类，
                - 或 Pydantic 类。

                若 `schema` 为 Pydantic 类，则模型输出将是一个该类的 Pydantic
                实例，且模型生成的字段将由 Pydantic
                类进行验证。否则模型输出将是一个字典，不会进行验证。

                参见 `langchain_core.utils.function_calling.convert_to_openai_tool`
                了解如何正确指定 Pydantic 或 `TypedDict` 类中模式字段的类型和描述。

                include_raw: 若为 `False`，则仅返回解析后的结构化输出。

                若在模型输出解析过程中发生错误，则会抛出异常。

                若为 `True`，则同时返回原始模型响应（一个
                `BaseMessage`）和解析后的模型响应。

                若在输出解析过程中发生错误，该错误将被捕获并一并返回。

                最终输出始终是一个包含键 `'raw'`、`'parsed'` 和 `'parsing_error'` 的
                `dict`。

            kwargs: method: 使用的结构化输出方法。选项包括：

        Returns:
            - `function_calling`（默认）：使用强制工具调用以获取结构化输出。当启用
            `thinking`，或在不支持强制工具调用的模型（Claude Opus 5.5、Claude Fable
            5.1、Claude Sonnet 5.5）上，工具调用不会被强制执行，缺少工具调用将引发
            `OutputParserException`。 - `json_schema`: 使用 Claude 的专用
            [结构化输出](https://platform.claude.com/docs/en/build-with-claude/structured-outputs)
            功能。

                kwargs: 其他关键字参数将被忽略。

                一个与 `langchain_core.language_models.chat.BaseChatModel` 输入相同的 `Runnable`。

                若 `include_raw` 为 `False` 且 `schema` 为 Pydantic 类，则 `Runnable`
                输出 `schema` 的实例（即 Pydantic 对象）。否则，若 `include_raw` 为
                `False`，则 `Runnable` 输出一个 `dict`。

        Example:
            ```python hl_lines="13"
            from langchain_anthropic import ChatAnthropic
            from pydantic import BaseModel, Field

            model = ChatAnthropic(model="claude-sonnet-4-5")

            class Movie(BaseModel):
                \"\"\"A movie with details.\"\"\"
                title: str = Field(..., description="The title of the movie")
                year: int = Field(..., description="The year the movie was released")
                director: str = Field(..., description="The director of the movie")
                rating: float = Field(..., description="The movie's rating out of 10")

            model_with_structure = model.with_structured_output(Movie, method="json_schema")
            response = model_with_structure.invoke("Provide details about the movie Inception")
            print(response)
            # -> Movie(title="Inception", year=2010, director="Christopher Nolan", rating=8.8)
            ```
        """  # noqa: E501
        if method == "json_mode":
            warning_message = (
                "Unrecognized structured output method 'json_mode'. Defaulting to "
                "'json_schema' method."
            )
            warnings.warn(warning_message, stacklevel=2)
            method = "json_schema"

                # 从 tool.extras 中选择参数
        if method == "function_calling":
            formatted_tool = cast(AnthropicTool, convert_to_anthropic_tool(schema))
            # 所有字段均为顶层填充
                        # TypedDict
            tool_name = formatted_tool["name"]
            if (
                self.thinking is not None
                and self.thinking.get("type") in ("enabled", "adaptive")
            ) or not _supports_forced_tool_choice(self.model):
                llm = self._get_llm_for_structured_output_when_thinking_is_enabled(
                    schema,
                    formatted_tool,
                )
            else:
                llm = self.bind_tools(
                    [schema],
                    tool_choice=tool_name,  # 若提供则添加缓存 TTL 信息（5分钟和1小时临时缓存）
                    ls_structured_output_format={
                        "kwargs": {"method": "function_calling"},
                        "schema": formatted_tool,
                    },
                )

            if isinstance(schema, type) and is_basemodel_subclass(schema):
                output_parser: OutputParserLike = PydanticToolsParser(
                    tools=[schema],
                    first_tool_only=True,
                )
            else:
                output_parser = JsonOutputKeyToolsParser(
                    key_name=tool_name,
                    first_tool_only=True,
                )
        elif method == "json_schema":
            llm = self.bind(
                output_config={
                    "format": _convert_to_anthropic_output_config_format(schema)
                },
                ls_structured_output_format={
                    "kwargs": {"method": "json_schema"},
                    "schema": convert_to_openai_tool(schema),
                },
            )
            if isinstance(schema, type) and is_basemodel_subclass(schema):
                output_parser = PydanticOutputParser(pydantic_object=schema)
            else:
                output_parser = JsonOutputParser()
        else:
            error_message = (
                f"Unrecognized structured output method '{method}'. "
                f"Expected 'function_calling' or 'json_schema'."
            )
            raise ValueError(error_message)

        if include_raw:
            parser_assign = RunnablePassthrough.assign(
                parsed=itemgetter("raw") | output_parser,
                parsing_error=lambda _: None,
            )
            parser_none = RunnablePassthrough.assign(parsed=lambda _: None)
            parser_with_fallback = parser_assign.with_fallbacks(
                [parser_none],
                exception_key="parsing_error",
            )
            return RunnableMap(raw=llm) | parser_with_fallback
        return llm | output_parser

    def get_num_tokens_from_messages(
        self,
        messages: list[BaseMessage],
        tools: Sequence[dict[str, Any] | type | Callable | BaseTool] | None = None,
        **kwargs: Any,
    ) -> int:
        """Count tokens in a sequence of input messages.

        This uses Anthropic's official [token counting API](https://platform.claude.com/docs/en/build-with-claude/token-counting).

        Args:
            messages: The message inputs to tokenize.
            tools: If provided, sequence of `dict`, `BaseModel`, function, or `BaseTool`
                objects to be converted to tool schemas.
            kwargs: Additional keyword arguments are passed to the Anthropic
                `messages.count_tokens` method.

        ???+ example "Basic usage"

            ```python
            from langchain_anthropic import ChatAnthropic
            from langchain_core.messages import HumanMessage, SystemMessage

            model = ChatAnthropic(model="claude-sonnet-4-5-20250929")

            messages = [ SystemMessage(content="You are a scientist"),
            HumanMessage(content="Hello, Claude"), ]
            model.get_num_tokens_from_messages(messages)

            ```txt
            14
            ```

        ??? example "Pass tool schemas"

            ```python
            from langchain_anthropic import ChatAnthropic
            from langchain_core.messages import HumanMessage
            from langchain_core.tools import tool

            model = ChatAnthropic(model="claude-sonnet-4-5-20250929")

            @tool(parse_docstring=True) def get_weather(location: str) -> str:
            """获取指定位置的当前天气 location: 城市和州，例如：旧金山，加利福尼亚 """
            return "晴天"

                Args:
                    messages = [ HumanMessage(content="旧金山的天气怎么样？"), ]
                    model.get_num_tokens_from_messages(messages, tools=[get_weather])

            ```txt
            403
            ```

            Built-in Tool Support
        """  # noqa: D214
        formatted_system, formatted_messages = _format_messages(
            messages,
            model=self.model,
            mid_conversation_system_messages=bool(
                (self.profile or {}).get("mid_conversation_system_messages")
            ),
        )
        if formatted_system is not None:
            kwargs["system"] = formatted_system
        if tools:
            # 目前仅复制 5m 和 1h 键，但未来若有更多缓存类型需扩展此元组
                        # 移除通用键以避免缓存创建 token 的重复计算
            kwargs["tools"], _ = _drop_unsupported_root_composition_tools(
                [convert_to_anthropic_tool(tool) for tool in tools]
            )
        if self.context_management is not None:
            kwargs["context_management"] = self.context_management

        if self.betas is not None:
            beta_response = self._client.beta.messages.count_tokens(
                betas=self.betas,
                model=self.model,
                messages=formatted_messages,  # type: ignore[arg-type]
                **kwargs,
            )
            return beta_response.input_tokens
        response = self._client.messages.count_tokens(
            model=self.model,
            messages=formatted_messages,  # type: ignore[arg-type]
            **kwargs,
        )
        return response.input_tokens


_TOP_LEVEL_SCHEMA_COMPOSITION_KEYS = ("oneOf", "anyOf")


def _drop_unsupported_root_composition_tools(
    tools: Sequence[Mapping[str, Any]],
) -> tuple[list[Mapping[str, Any]], set[str]]:
    """移除根 `input_schema` 使用 `oneOf`/`anyOf` 的工具。

    Anthropic API 在请求验证阶段会拒绝这些内容，导致整个请求失败。 只有当
    `input_schema` 是包含根组合器的映射时，才会丢弃工具； 内置（服务端）工具——没有
    `input_schema`——以及组合器嵌套在 `properties` 下的工具则保持不变，原样通过。

    每个被丢弃的工具都会发出一个 `UserWarning`。

    Args:
        tools: 已格式化的工具定义，由 `bind_tools` 构建而成。

    Returns:
        保留的工具，以及被丢弃工具的名称。没有字符串 `name`
        的被丢弃工具不会在名称集合中产生条目。
    """
    kept: list[Mapping[str, Any]] = []
    dropped_tool_names: set[str] = set()
    for tool in tools:
        input_schema = tool.get("input_schema")
        offending_keys = (
            [k for k in _TOP_LEVEL_SCHEMA_COMPOSITION_KEYS if k in input_schema]
            if isinstance(input_schema, Mapping)
            else []
        )
        if not offending_keys:
            kept.append(tool)
            continue
        tool_name = tool.get("name")
        if isinstance(tool_name, str):
            dropped_tool_names.add(tool_name)
            described = repr(tool_name)
        else:
            described = "with no name"
        warnings.warn(
            f"Dropping tool {described}: its input_schema has a "
            f"top-level {'/'.join(offending_keys)}, which the Anthropic API does "
            "not support. The tool will not be available to the model.",
            stacklevel=3,
        )
    return kept, dropped_tool_names


def convert_to_anthropic_tool(
    tool: Mapping[str, Any] | type | Callable | BaseTool,
    *,
    strict: bool | None = None,
) -> AnthropicTool:
    """将工具类对象转换为 Anthropic 工具定义。

    Args:
        tool: 要转换的工具类对象。可以是 Anthropic 工具字典、Pydantic 模型、函数或
        `BaseTool`。 strict: 如果为 `True`，则启用工具的严格模式。

            !!! note

                需要 Claude Sonnet 4.5 或 Opus 4.1。

    Returns:
        `AnthropicTool` 用于自定义/用户定义的工具
    """
    if (
        isinstance(tool, BaseTool)
        and hasattr(tool, "extras")
        and isinstance(tool.extras, dict)
        and "provider_tool_definition" in tool.extras
    ):
                # 计算总输入 token 数：Anthropic 的 `input_tokens` 不包含缓存
        # token，因此需要将其加回以获得真实的总输入 token 数量
        return tool.extras["provider_tool_definition"]  # type: ignore[return-value]

    if isinstance(tool, dict) and all(
        k in tool for k in ("name", "description", "input_schema")
    ):
                # 基础输入 token 数
        anthropic_formatted = AnthropicTool(tool)  # type: ignore[misc]
    else:
        oai_formatted = convert_to_openai_tool(tool, strict=strict)["function"]
        anthropic_formatted = AnthropicTool(
            name=oai_formatted["name"],
            input_schema=oai_formatted["parameters"],
        )
        if "description" in oai_formatted:
            anthropic_formatted["description"] = oai_formatted["description"]
        if "strict" in oai_formatted and isinstance(strict, bool):
            anthropic_formatted["strict"] = oai_formatted["strict"]
                # 从缓存读取的 token 数
        if (
            isinstance(tool, BaseTool)
            and hasattr(tool, "extras")
            and isinstance(tool.extras, dict)
        ):
            for key, value in tool.extras.items():
                if key in _ANTHROPIC_EXTRA_FIELDS:
                                        # 用于创建缓存的 token 数
                    anthropic_formatted[key] = value  # type: ignore[literal-required]
    return anthropic_formatted


def _tools_in_params(params: dict) -> bool:
    return (
        "tools" in params
        or ("extra_body" in params and params["extra_body"].get("tools"))
        or "mcp_servers" in params
    )


def _thinking_in_params(params: dict) -> bool:
    return params.get("thinking", {}).get("type") in ("enabled", "adaptive")


def _documents_in_params(params: dict) -> bool:
    for message in params.get("messages", []):
        if isinstance(message.get("content"), list):
            for block in message["content"]:
                if (
                    isinstance(block, dict)
                    and block.get("type") == "document"
                    and block.get("citations", {}).get("enabled")
                ):
                    return True
    return False


def _compact_in_params(params: dict) -> bool:
    edits = params.get("context_management", {}).get("edits") or []

    return any("compact" in (edit.get("type") or "") for edit in edits)


class _AnthropicToolUse(TypedDict):
    type: Literal["tool_use"]
    name: str
    input: dict
    id: str
    caller: NotRequired[dict[str, Any]]
    toolset_name: NotRequired[str]


def _lc_tool_calls_to_anthropic_tool_use_blocks(
    tool_calls: list[ToolCall],
) -> list[_AnthropicToolUse]:
    return [
        _AnthropicToolUse(
            type="tool_use",
            name=tool_call["name"],
            input=tool_call["args"],
            id=cast("str", _normalize_tool_call_id(tool_call["id"])),
        )
        for tool_call in tool_calls
    ]


def _convert_to_anthropic_output_config_format(schema: dict | type) -> dict[str, Any]:
    """将 JSON schema、Pydantic 模型或 `TypedDict` 转换为 `output_config.format`。

    参见 Claude 文档中的 [结构化输出](https://platform.claude.com/docs/en/build-with-claude/structured-outputs)。

    Args:
        schema: 一个 JSON schema 字典、Pydantic 模型类或 TypedDict。

    Returns:
        一个包含 `type` 和 `schema` 键的字典，适用于 `output_config.format`。
    """
    from anthropic import transform_schema

    is_pydantic_class = isinstance(schema, type) and is_basemodel_subclass(schema)
    if is_pydantic_class or isinstance(schema, dict):
        json_schema = transform_schema(schema)
    else:
                # 推理（thinking）token 是 `output_tokens` 的分解（非累加），由 Anthropic 通过
        # `output_tokens_details.thinking_tokens` 报告
        json_schema = transform_schema(convert_to_json_schema(schema))
    return {"type": "json_schema", "schema": json_schema}


def _create_usage_metadata(anthropic_usage: BaseModel) -> UsageMetadata:
    """从 Anthropic 的 `Usage` 数据创建 LangChain `UsageMetadata`。

    Note:
        Anthropic 的 `input_tokens` 不包含缓存 token，因此我们手动添加 `cache_read` 和
        `cache_creation` token 以获得真实总数。
    """
    input_token_details: dict = {
        "cache_read": getattr(anthropic_usage, "cache_read_input_tokens", None),
        "cache_creation": getattr(anthropic_usage, "cache_creation_input_tokens", None),
    }

        # 旧版模型完全省略 `output_tokens_details`，因此需使用 `getattr` 进行保护
    cache_creation = getattr(anthropic_usage, "cache_creation", None)

    # 目前只是复制了 5m 和 1h 的键，但如果后续添加了更多
    # 未来我们需要扩展这个元组
    cache_creation_keys = ("ephemeral_5m_input_tokens", "ephemeral_1h_input_tokens")
    specific_cache_creation_tokens = 0
    if cache_creation:
        if isinstance(cache_creation, BaseModel):
            cache_creation = cache_creation.model_dump()
        for k in cache_creation_keys:
            specific_cache_creation_tokens += cache_creation.get(k, 0)
            input_token_details[k] = cache_creation.get(k)
        if not isinstance(specific_cache_creation_tokens, int):
            specific_cache_creation_tokens = 0
        if specific_cache_creation_tokens > 0:
                        # 移除通用键以避免重复计算缓存创建的 token
            input_token_details["cache_creation"] = 0

        # 计算总输入 token 数：Anthropic 的 `input_tokens` 不包含缓存的 token，
        # 所以我们需要将它们加回去，以获得真实的总输入 token 数量
    input_tokens = (
        (getattr(anthropic_usage, "input_tokens", 0) or 0)  # 基础输入 token
        + (input_token_details["cache_read"] or 0)  # 从缓存读取的 token
        + (
            specific_cache_creation_tokens or input_token_details["cache_creation"] or 0
        )  # 用于创建缓存的 token
    )
    output_tokens = getattr(anthropic_usage, "output_tokens", 0) or 0

        # 推理（思考）token 是 `output_tokens` 的分解（非
        # additive)，由 Anthropic 通过 `output_tokens_details.thinking_tokens` 报告。
        # 旧版模型会完全省略 `output_tokens_details`，因此需使用 `getattr` 进行防护。
    output_token_details: dict = {
        "reasoning": getattr(
            getattr(anthropic_usage, "output_tokens_details", None),
            "thinking_tokens",
            None,
        ),
    }

    usage_metadata = UsageMetadata(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=input_tokens + output_tokens,
        input_token_details=InputTokenDetails(
            **{k: v for k, v in input_token_details.items() if v is not None},
        ),
    )

    filtered_output_token_details = {
        k: v for k, v in output_token_details.items() if v is not None
    }
    if filtered_output_token_details:
        usage_metadata["output_token_details"] = OutputTokenDetails(
            **filtered_output_token_details,
        )

    return usage_metadata

"""Fireworks 聊天封装。"""

from __future__ import annotations

import contextlib
import json
import logging
from collections.abc import AsyncIterator, Callable, Iterator, Mapping, Sequence
from contextvars import ContextVar
from operator import itemgetter
from typing import (
    Any,
    Literal,
    NoReturn,
    TypeAlias,
    cast,
)

import httpx
from fireworks import (
    APIConnectionError,
    APIError,
    APITimeoutError,
    AsyncFireworks,
    AuthenticationError,
    BadRequestError,
    Fireworks,
    FireworksError,
    InternalServerError,
    NotFoundError,
    PermissionDeniedError,
    RateLimitError,
)
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
)
from langchain_core.language_models import (
    LanguageModelInput,
    ModelProfile,
    ModelProfileRegistry,
)
from langchain_core.language_models.chat_models import (
    BaseChatModel,
    LangSmithParams,
    agenerate_from_stream,
    generate_from_stream,
)
from langchain_core.language_models.llms import create_base_retry_decorator
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    BaseMessageChunk,
    ChatMessage,
    ChatMessageChunk,
    FunctionMessage,
    FunctionMessageChunk,
    HumanMessage,
    HumanMessageChunk,
    InvalidToolCall,
    SystemMessage,
    SystemMessageChunk,
    ToolCall,
    ToolMessage,
    ToolMessageChunk,
    UsageMetadata,
    is_data_content_block,
)
from langchain_core.messages.block_translators.openai import (
    convert_to_openai_data_block,
)
from langchain_core.messages.tool import (
    ToolCallChunk,
)
from langchain_core.messages.tool import (
    tool_call_chunk as create_tool_call_chunk,
)
from langchain_core.output_parsers import JsonOutputParser, PydanticOutputParser
from langchain_core.output_parsers.base import OutputParserLike
from langchain_core.output_parsers.openai_tools import (
    JsonOutputKeyToolsParser,
    PydanticToolsParser,
    make_invalid_tool_call,
    parse_tool_call,
)
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.runnables import Runnable, RunnableMap, RunnablePassthrough
from langchain_core.tools import BaseTool
from langchain_core.utils import (
    get_pydantic_field_names,
)
from langchain_core.utils._gateway import _apply_gateway_config
from langchain_core.utils.function_calling import (
    convert_to_json_schema,
    convert_to_openai_tool,
)
from langchain_core.utils.pydantic import is_basemodel_subclass
from langchain_core.utils.utils import _build_model_kwargs
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    SecretStr,
    model_validator,
)
from typing_extensions import Self

from langchain_fireworks._compat import _convert_from_v1_to_chat_completions
from langchain_fireworks._version import __version__
from langchain_fireworks.data._profiles import _PROFILES

logger = logging.getLogger(__name__)

_PROMPT_CACHE_AFFINITY: ContextVar[str | None] = ContextVar(
    "fireworks_prompt_cache_affinity", default=None
)


_MODEL_PROFILES = cast("ModelProfileRegistry", _PROFILES)


def _get_default_model_profile(model_name: str) -> ModelProfile:
    default = _MODEL_PROFILES.get(model_name) or {}
    return default.copy()


def _convert_dict_to_message(_dict: Mapping[str, Any]) -> BaseMessage:
    """将字典转换为 LangChain 消息。

    Args:
        _dict: 字典对象。

    Returns:
        LangChain 消息。

    """
    role = _dict.get("role")
    if role == "user":
        return HumanMessage(content=_dict.get("content", ""))
    if role == "assistant":
                # Fix for azure
                # Also Fireworks returns None for tool invocations
        content = _dict.get("content", "") or ""
        additional_kwargs: dict = {}
        if (
            isinstance(reasoning_content := _dict.get("reasoning_content"), str)
            and reasoning_content
        ):
            additional_kwargs["reasoning_content"] = reasoning_content

        if function_call := _dict.get("function_call"):
            additional_kwargs["function_call"] = dict(function_call)

        tool_calls = []
        invalid_tool_calls = []
        if raw_tool_calls := _dict.get("tool_calls"):
            additional_kwargs["tool_calls"] = raw_tool_calls
            for raw_tool_call in raw_tool_calls:
                try:
                    tool_calls.append(parse_tool_call(raw_tool_call, return_id=True))
                except Exception as e:
                    invalid_tool_calls.append(
                        dict(make_invalid_tool_call(raw_tool_call, str(e)))
                    )
        return AIMessage(
            content=content,
            additional_kwargs=additional_kwargs,
            tool_calls=tool_calls,
            invalid_tool_calls=invalid_tool_calls,
        )
    if role == "system":
        return SystemMessage(content=_dict.get("content", ""))
    if role == "function":
        return FunctionMessage(
            content=_dict.get("content", ""), name=_dict.get("name", "")
        )
    if role == "tool":
        additional_kwargs = {}
        if "name" in _dict:
            additional_kwargs["name"] = _dict["name"]
        return ToolMessage(
            content=_dict.get("content", ""),
            tool_call_id=_dict.get("tool_call_id", ""),
            additional_kwargs=additional_kwargs,
        )
    return ChatMessage(content=_dict.get("content", ""), role=role or "")


def _allowed_content_part_keys() -> frozenset[str]:
    """Fireworks 内容部分允许的 wire 格式键列表。

    Derived at import time from the stainless-generated TypedDict so the
    allowlist tracks the upstream OpenAPI spec as `fireworks-ai` is bumped:
    new fields widen the allowlist for free, removed/renamed fields shrink it
    in lockstep. If the SDK reshuffles its module layout the import falls back
    to a conservative hand-coded set and emits a warning, and the layout test
    (`test_fireworks_sdk_request_layout_stable`) fails to surface the drift.
    """
    try:
        from typing import get_type_hints

        from fireworks.types.shared_params.chat_message import (
            ContentUnionMember1,
        )

        return frozenset(get_type_hints(ContentUnionMember1))
    except ImportError:
        logger.warning(
            "Could not import `fireworks.types.shared_params.chat_message."
            "ContentUnionMember1`; falling back to a conservative content-part "
            "key allowlist. Bump `fireworks-ai` or update "
            "`_allowed_content_part_keys` if the SDK has moved this type.",
        )
        return frozenset({"type", "text", "image_url", "video_url"})


_ALLOWED_CONTENT_PART_KEYS: frozenset[str] = _allowed_content_part_keys()


_DROPPED_CONTENT_BLOCK_TYPES: frozenset[str] = frozenset(
    {
        "tool_use",
        "thinking",
        "reasoning",
        "reasoning_content",
        "function_call",
        "code_interpreter_call",
    }
)
"""Content block types the chat completions wire format does not carry.

These arise from provider-specific or canonical v1 content (e.g. Anthropic
`tool_use`/`thinking` blocks, or a `reasoning` block on an AIMessage) that
reaches `_convert_message_to_dict` as conversation history. Fireworks rejects
them, so they are dropped rather than forwarded.
"""


def _sanitize_chat_completions_content(content: Any) -> Any:
    """在序列化为 Fireworks 之前，从内容块中移除非 wire 键。

    Fireworks's chat completions endpoint rejects unknown fields on message
    content parts with `Extra inputs are not permitted, field: 'messages[N]
    .content.list[ChatMessageContent][i].<key>'`. This surfaces when a
    conversation accumulates AIMessages from a different provider (e.g.
    Anthropic's v1 streaming-reassembly `index` marker on text blocks, or the
    LangChain-internal `caller` key on `tool_use` blocks) and that history is
    later forwarded to a Fireworks-hosted model.

    For list content:
        - each block dict is filtered down to keys in
            `_ALLOWED_CONTENT_PART_KEYS` (sourced from the SDK TypedDict, so it
            stays in sync with the upstream spec).
        - if the result is a list of exactly one block that, post-strip, is
            `{"type": "text", "text": <str>}` and nothing else, it is coerced to
            a plain string. Fireworks's `content` union lists `str` first
            (`Input should be a valid string, field: 'messages[N].content.str'`),
            and the stricter shape avoids the union-validation noise on the
            server side.
    Non-list content (strings, None) passes through unchanged.
    """
    if not isinstance(content, list):
        return content
    sanitized: list[Any] = []
    for block in content:
        if isinstance(block, dict):
            sanitized.append(
                {k: v for k, v in block.items() if k in _ALLOWED_CONTENT_PART_KEYS}
            )
        else:
            sanitized.append(block)
    if (
        len(sanitized) == 1
        and isinstance(sanitized[0], dict)
        and set(sanitized[0]) == {"type", "text"}
        and sanitized[0]["type"] == "text"
        and isinstance(sanitized[0]["text"], str)
    ):
        return sanitized[0]["text"]
    return sanitized


def _format_message_content(content: Any) -> Any:
    """为 Fireworks 聊天补全 wire 格式格式化消息内容。

    Adapted from `langchain_openai.chat_models.base._format_message_content`,
    scoped to the chat completions API: drops content block types the wire
    format does not carry, translates canonical v0/v1 multimodal data blocks
    via `convert_to_openai_data_block(block, api="chat/completions")`, and
    converts legacy Anthropic-shape image blocks (`{"type": "image",
    "source": {...}}`) to OpenAI `image_url` blocks. String and non-list
    content are returned unchanged.

    Args:
        content: 消息内容。字符串和非列表值直接返回；列表按块逐个遍历处理。

    Returns:
        The formatted content, ready to be placed on the chat completions
        wire. List inputs return a new list with translations applied; other
        inputs are returned unchanged.
    """
    if not isinstance(content, list):
        return content
    formatted: list[Any] = []
    for block in content:
        if isinstance(block, dict) and "type" in block:
            btype = block["type"]
            if btype in _DROPPED_CONTENT_BLOCK_TYPES:
                continue
            if is_data_content_block(block):
                formatted.append(
                    convert_to_openai_data_block(block, api="chat/completions")
                )
                continue
            if (
                btype == "image"
                and (source := block.get("source"))
                and isinstance(source, dict)
            ):
                if (
                    source.get("type") == "base64"
                    and (media_type := source.get("media_type"))
                    and (data := source.get("data"))
                ):
                    formatted.append(
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{media_type};base64,{data}"},
                        }
                    )
                    continue
                if source.get("type") == "url" and (url := source.get("url")):
                    formatted.append({"type": "image_url", "image_url": {"url": url}})
                    continue
                continue
        formatted.append(block)
    return formatted


def _format_tool_call_arguments(arguments: str | dict | None) -> str | dict:
    """保留无效的历史参数在 JSON 对象中以供重放。"""
    if isinstance(arguments, dict):
        return arguments
    try:
        parsed = json.loads(arguments) if arguments is not None else None
        if isinstance(parsed, dict) and arguments is not None:
            json.dumps(parsed, allow_nan=False)
            return arguments
    except ValueError:
        logger.debug("Invalid JSON in historical Fireworks tool call arguments")
    logger.warning("Wrapping invalid historical Fireworks tool call arguments")
    return json.dumps({"__invalid_tool_call_arguments": arguments}, ensure_ascii=False)


def _convert_message_to_dict(message: BaseMessage) -> dict:
    """将 LangChain 消息转换为字典。

    Args:
        message: LangChain 消息。

    Returns:
        字典。

    """
    message_dict: dict[str, Any]
    if isinstance(message, ChatMessage):
        message_dict = {
            "role": message.role,
            "content": _sanitize_chat_completions_content(
                _format_message_content(message.content)
            ),
        }
    elif isinstance(message, HumanMessage):
        message_dict = {
            "role": "user",
            "content": _sanitize_chat_completions_content(
                _format_message_content(message.content)
            ),
        }
    elif isinstance(message, AIMessage):
                # Translate v1 content
        if message.response_metadata.get("output_version") == "v1":
            message = _convert_from_v1_to_chat_completions(message)
        message_dict = {
            "role": "assistant",
            "content": _sanitize_chat_completions_content(
                _format_message_content(message.content)
            ),
        }
                # Replay reasoning only when the message is tagged `fireworks` or has no
                # `model_provider` (hand-built messages, or integrations that don't tag
                # theirs); reasoning from other tagged providers may not be valid input.
        if isinstance(
            reasoning_content := message.additional_kwargs.get("reasoning_content"), str
        ) and message.response_metadata.get("model_provider") in (None, "fireworks"):
            message_dict["reasoning_content"] = reasoning_content
        if "function_call" in message.additional_kwargs:
            message_dict["function_call"] = message.additional_kwargs["function_call"]
                        # If function call only, content is None not empty string
            if message_dict["content"] == "":
                message_dict["content"] = None
        if message.tool_calls or message.invalid_tool_calls:
            message_dict["tool_calls"] = [
                _lc_tool_call_to_fireworks_tool_call(tc) for tc in message.tool_calls
            ] + [
                _lc_invalid_tool_call_to_fireworks_tool_call(tc)
                for tc in message.invalid_tool_calls
            ]
        elif "tool_calls" in message.additional_kwargs:
            message_dict["tool_calls"] = message.additional_kwargs["tool_calls"]
        if "tool_calls" in message_dict:
            message_dict["tool_calls"] = [
                {
                    **tool_call,
                    "function": {
                        **tool_call["function"],
                        "arguments": _format_tool_call_arguments(
                            tool_call["function"]["arguments"]
                        ),
                    },
                }
                for tool_call in message_dict["tool_calls"]
            ]
                # If tool calls only, content is None not empty string
        if "tool_calls" in message_dict and message_dict["content"] == "":
            message_dict["content"] = None
        else:
            pass
    elif isinstance(message, SystemMessage):
        message_dict = {
            "role": "system",
            "content": _sanitize_chat_completions_content(
                _format_message_content(message.content)
            ),
        }
    elif isinstance(message, FunctionMessage):
        message_dict = {
            "role": "function",
            "content": message.content,
            "name": message.name,
        }
    elif isinstance(message, ToolMessage):
        message_dict = {
            "role": "tool",
            "content": _sanitize_chat_completions_content(
                _format_message_content(message.content)
            ),
            "tool_call_id": message.tool_call_id,
        }
    else:
        msg = f"Got unknown type {message}"
        raise TypeError(msg)
    if "name" in message.additional_kwargs:
        message_dict["name"] = message.additional_kwargs["name"]
    return message_dict


def _usage_to_metadata(usage: Mapping[str, Any]) -> UsageMetadata:
    input_tokens = usage.get("prompt_tokens") or 0
    output_tokens = usage.get("completion_tokens") or 0
    usage_metadata: UsageMetadata = {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": usage.get("total_tokens") or input_tokens + output_tokens,
    }
    cached_tokens = (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
    if cached_tokens is not None:
        usage_metadata["input_token_details"] = {"cache_read": cached_tokens}
    reasoning_tokens = (usage.get("completion_tokens_details") or {}).get(
        "reasoning_tokens"
    )
    if reasoning_tokens is not None:
        usage_metadata["output_token_details"] = {"reasoning": reasoning_tokens}
    return usage_metadata


TokenUsageTree: TypeAlias = "int | dict[str, TokenUsageTree]"
"""Raw provider token usage: a tree of `int` leaves and nested `dict` nodes
(e.g. `prompt_tokens_details`).

Modeled as a recursive alias so the merge helper's signature carries the shape
rather than leaving it to `Any`.
"""


def _update_token_usage(
    overall_token_usage: TokenUsageTree, new_usage: TokenUsageTree
) -> TokenUsageTree:
    """递归合并生成过程中的原始提供方 token 使用量。

    token 使用量是一个 `int` 叶子节点（求和）和 `dict` 节点（如
    `prompt_tokens_details`，按键合并，跳过 `None` 值）的树结构。

    累加器与传入值之间的类型不匹配（例如一边是 `int`，另一边是
    `dict`）表示提供方数据格式错误，会抛出异常而非静默转换。完全意外的叶子类型（既非
    `int` 也非 `dict`）会被记录并传递通过，因此遥测异常会优雅降级而非导致响应失败。
    """
    if isinstance(new_usage, int):
        if not isinstance(overall_token_usage, int):
            msg = (
                "Got different types for token usage: "
                f"{new_usage!r} ({type(new_usage).__name__}) and "
                f"{overall_token_usage!r} ({type(overall_token_usage).__name__})"
            )
            raise ValueError(msg)
        return overall_token_usage + new_usage
    if isinstance(new_usage, dict):
        if not isinstance(overall_token_usage, dict):
            msg = (
                "Got different types for token usage: "
                f"{new_usage!r} ({type(new_usage).__name__}) and "
                f"{overall_token_usage!r} ({type(overall_token_usage).__name__})"
            )
            raise ValueError(msg)
        updated_token_usage = dict(overall_token_usage)
        for key, value in new_usage.items():
            if value is not None:
                                # Seed a first-seen key with an empty node of the same kind so a
                                # nested `dict` value merges rather than colliding with an `int`.
                default: TokenUsageTree = {} if isinstance(value, dict) else 0
                updated_token_usage[key] = _update_token_usage(
                    overall_token_usage.get(key, default), value
                )
        return updated_token_usage
    logger.warning("Unexpected type for token usage: %s", type(new_usage).__name__)
    return new_usage


def _convert_chunk_to_message_chunk(
    chunk: Mapping[str, Any], default_class: type[BaseMessageChunk]
) -> BaseMessageChunk:
    choices = chunk.get("choices") or []
    response_metadata: dict[str, Any] = {"model_provider": "fireworks"}
    if not choices:
                # Final chunk emitted when `stream_options.include_usage=True`:
                # `choices` is empty and the chunk carries only `usage`.
        usage = chunk.get("usage")
        if not usage:
            logger.debug(
                "Received stream chunk with no choices and no usage: %s", chunk
            )
        usage_metadata = _usage_to_metadata(usage) if usage else None
        return AIMessageChunk(
            content="",
            usage_metadata=usage_metadata,
            response_metadata=response_metadata,
        )
    choice = choices[0]
    _dict = choice["delta"]
    role = cast(str, _dict.get("role"))
    content = cast(str, _dict.get("content") or "")
    additional_kwargs: dict = {}
    tool_call_chunks: list[ToolCallChunk] = []
    if (
        isinstance(reasoning_content := _dict.get("reasoning_content"), str)
        and reasoning_content
    ):
        additional_kwargs["reasoning_content"] = reasoning_content
    if _dict.get("function_call"):
        function_call = dict(_dict["function_call"])
        if "name" in function_call and function_call["name"] is None:
            function_call["name"] = ""
        additional_kwargs["function_call"] = function_call
    if raw_tool_calls := _dict.get("tool_calls"):
        additional_kwargs["tool_calls"] = raw_tool_calls
        for rtc in raw_tool_calls:
            with contextlib.suppress(KeyError):
                tool_call_chunks.append(
                    create_tool_call_chunk(
                        name=rtc["function"].get("name"),
                        args=rtc["function"].get("arguments"),
                        id=rtc.get("id"),
                        index=rtc.get("index"),
                    )
                )
    if role == "user" or default_class == HumanMessageChunk:
        return HumanMessageChunk(content=content)
    if role == "assistant" or default_class == AIMessageChunk:
        usage = chunk.get("usage")
        usage_metadata = _usage_to_metadata(usage) if usage else None
        return AIMessageChunk(
            content=content,
            additional_kwargs=additional_kwargs,
            tool_call_chunks=tool_call_chunks,
            usage_metadata=usage_metadata,
            response_metadata=response_metadata,
        )
    if role == "system" or default_class == SystemMessageChunk:
        return SystemMessageChunk(content=content)
    if role == "function" or default_class == FunctionMessageChunk:
        return FunctionMessageChunk(content=content, name=_dict["name"])
    if role == "tool" or default_class == ToolMessageChunk:
        return ToolMessageChunk(content=content, tool_call_id=_dict["tool_call_id"])
    if role or default_class == ChatMessageChunk:
        return ChatMessageChunk(content=content, role=role)
    return default_class(content=content)  # type: ignore[call-arg]


def _format_chunk_for_v1(message_chunk: AIMessageChunk) -> AIMessageChunk:
    """将流式分块转换为具有独立索引命名空间的 v1 内容块。

    Fireworks streams a single text field and a single `reasoning_content` field,
    but numbers tool calls from 0. Left to core, streaming (`stream`, `astream`,
    and streaming `invoke`) gives unindexed blocks an integer that increments each
    time the block type changes, so tool call 0 would be merged into the reasoning
    block by `merge_lists`, and reasoning interrupted by text would split in two.

    `merge_lists` merges blocks with matching indices only when the index is an int
    or a string starting with `lc_`; other strings are appended unmerged. The raw
    `tool_call_chunks` keep the provider's integer indices, which
    `AIMessageChunk.__add__` relies on to merge tool-call arguments.
    """
    blocks: list[dict[str, Any]] = []
    if reasoning := message_chunk.additional_kwargs.get("reasoning_content"):
        blocks.append(
            {"type": "reasoning", "reasoning": reasoning, "index": "lc_reasoning"}
        )
    if message_chunk.content:
        blocks.append(
            {"type": "text", "text": message_chunk.content, "index": "lc_text"}
        )
    for tool_call_chunk in message_chunk.tool_call_chunks:
        block: dict[str, Any] = {
            "type": "tool_call_chunk",
            "id": tool_call_chunk["id"],
            "name": tool_call_chunk["name"],
            "args": tool_call_chunk["args"],
        }
        if (index := tool_call_chunk["index"]) is not None:
            block["index"] = f"lc_tc_{index}"
        blocks.append(block)
    return message_chunk.model_copy(
        update={
            "content": blocks,
            "response_metadata": {
                **message_chunk.response_metadata,
                "output_version": "v1",
            },
        }
    )


def _finalize_stream(
    stream: Iterator[ChatGenerationChunk],
) -> Iterator[ChatGenerationChunk]:
    """在内部聚合 v1 分块时，提供核心的最终化标记。"""
    yield from stream
    yield ChatGenerationChunk(message=AIMessageChunk(content="", chunk_position="last"))


async def _afinalize_stream(
    stream: AsyncIterator[ChatGenerationChunk],
) -> AsyncIterator[ChatGenerationChunk]:
    """在内部聚合 v1 分块时，提供核心的最终化标记。"""
    async for chunk in stream:
        yield chunk
    yield ChatGenerationChunk(message=AIMessageChunk(content="", chunk_position="last"))


class _RetryableHTTPStatusError(FireworksError):
    """内部标记，用于 5xx `httpx.HTTPStatusError` 响应。

    1.x SDK 将每个状态响应包装为一个类型的 `APIStatusError`
    子类，因此此路径属于纵深防御：仅当原始 `httpx.HTTPStatusError` 溜出 SDK
    时触发（例如自定义 `http_client`
    或被猴子补丁的传输直接引发）。在此处提升它，可使可重试集合表达为
    `create_base_retry_decorator` 所需的类列表。
    """


_RETRYABLE_ERRORS: tuple[type[BaseException], ...] = (
    APIConnectionError,
    InternalServerError,
    RateLimitError,
    httpx.TimeoutException,
    httpx.TransportError,
    _RetryableHTTPStatusError,
)


def _promote_http_status_error(exc: httpx.HTTPStatusError) -> NoReturn:
    """将 5xx `httpx.HTTPStatusError` 重新抛出为可重试标记。"""
    if exc.response.status_code >= 500:
        msg = f"Retryable {exc.response.status_code} from Fireworks: {exc}"
        raise _RetryableHTTPStatusError(msg) from exc
    raise exc


class FireworksContextOverflowError(BadRequestError, ContextOverflowError):
    """`BadRequestError` 在输入超出 Fireworks 上下文限制时抛出。"""


class FireworksAuthenticationError(AuthenticationError, ModelAuthenticationError):
    """Fireworks 认证错误，归类为 LangChain 模型错误。"""


class FireworksPermissionDeniedError(PermissionDeniedError, ModelPermissionDeniedError):
    """Fireworks 权限错误，归类为 LangChain 模型错误。"""


class FireworksInvalidRequestError(BadRequestError, ModelInvalidRequestError):
    """Fireworks 无效请求错误，归类为 LangChain 模型错误。"""


class FireworksModelNotFoundError(NotFoundError, ModelNotFoundError):
    """Fireworks 找不到资源错误，归类为 LangChain 模型错误。"""


class FireworksRateLimitError(RateLimitError, ModelRateLimitError):
    """Fireworks 速率限制错误，归类为 LangChain 模型错误。"""


class FireworksAPIError(InternalServerError, ModelAPIError):
    """Fireworks 服务器错误，归类为 LangChain 模型错误。"""


class FireworksConnectionError(APIConnectionError, ModelConnectionError):
    """Fireworks 连接错误，归类为 LangChain 模型错误。"""


class FireworksTimeoutError(APITimeoutError, ModelTimeoutError):
    """Fireworks 超时错误，归类为 LangChain 模型错误。"""


class FireworksReadTimeoutError(httpx.ReadTimeout, ModelTimeoutError):
    """Fireworks 流读取超时，归类为可重试模型错误。"""


def _handle_stream_read_timeout(error: httpx.ReadTimeout) -> NoReturn:
    try:
        request = error.request
    except RuntimeError:
        request = None
    raise FireworksReadTimeoutError(str(error), request=request) from error


def _handle_fireworks_invalid_request(e: BadRequestError) -> NoReturn:
    """将提示词过长错误提升为 `FireworksContextOverflowError`。"""
    if "prompt is too long" in str(e):
        raise FireworksContextOverflowError(
            str(e), response=e.response, body=e.body
        ) from e
    raise FireworksInvalidRequestError(str(e), response=e.response, body=e.body) from e


def _handle_fireworks_api_error(e: APIError) -> NoReturn:
    """将 Fireworks SDK 错误重新抛出为其 LangChain 归类的等效错误。"""
    if isinstance(e, AuthenticationError):
        raise FireworksAuthenticationError(
            str(e), response=e.response, body=e.body
        ) from e
    if isinstance(e, PermissionDeniedError):
        raise FireworksPermissionDeniedError(
            str(e), response=e.response, body=e.body
        ) from e
    if isinstance(e, NotFoundError):
        raise FireworksModelNotFoundError(
            str(e), response=e.response, body=e.body
        ) from e
    if isinstance(e, RateLimitError):
        raise FireworksRateLimitError(str(e), response=e.response, body=e.body) from e
    if isinstance(e, InternalServerError):
        raise FireworksAPIError(str(e), response=e.response, body=e.body) from e
        # `APITimeoutError` subclasses `APIConnectionError`, so check it first.
    if isinstance(e, APITimeoutError):
        raise FireworksTimeoutError(e.request) from e
    if isinstance(e, APIConnectionError):
        raise FireworksConnectionError(message=str(e), request=e.request) from e
    raise e


def _raise_empty_stream() -> NoReturn:
    """当 SDK 返回零分块流时，抛出描述性错误。"""
    msg = "Received empty stream from Fireworks"
    raise FireworksError(msg)


def _create_retry_decorator(
    llm: ChatFireworks,
    run_manager: AsyncCallbackManagerForLLMRun | CallbackManagerForLLMRun | None = None,
) -> Callable[[Any], Any]:
    """返回一个用于 Fireworks SDK 调用的 tenacity 重试装饰器。

    Retries live here rather than in the SDK so each attempt is visible to the
    LangChain `run_manager.on_retry` callback. The SDK's own retry layer is
    suppressed via `max_retries=0` on the client; see `validate_environment`.
    """
        # `max_retries` counts retries *after* the initial attempt (default lives on
        # the `ChatFireworks.max_retries` field). `create_base_retry_decorator
        # forwards its `max_retries` to `stop_after_attempt`, which counts total
        # 尝试次数 — 因此偏移 1。`None` 和 `0` 都表示“单次尝试，无重试”。
        # HTTP 头名称不区分大小写；保留请求中的拼写方式。
    attempts = (llm.max_retries + 1) if llm.max_retries else 1
    return create_base_retry_decorator(
        error_types=list(_RETRYABLE_ERRORS),
        max_retries=attempts,
        run_manager=run_manager,
    )


def _prepare_sdk_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    """将 1.x SDK 未建模的字段移至 `extra_body`。

    The Stainless-generated `chat.completions.create` signature has a fixed set
    of typed parameters. Fireworks accepts additional fields on the wire (notably
    `stream_options.include_usage`) that the SDK schema does not declare. The
    SDK exposes `extra_body` precisely for this — merge anything that looks
    extra-body-shaped into it so it lands in the JSON request body.

    If a caller supplies both `extra_body={"stream_options": ...}` and a
    top-level `stream_options=...`, the value already in `extra_body` wins
    (callers using `extra_body` are presumed to want explicit control); the
    discarded top-level value is logged.
    """
    extra_body = dict(kwargs.pop("extra_body", None) or {})
    top_level_stream_options = kwargs.pop("stream_options", None)
    if top_level_stream_options is not None:
        if "stream_options" in extra_body:
            logger.warning(
                "Both `extra_body['stream_options']` and a top-level "
                "`stream_options` were supplied; using `extra_body`'s value "
                "and discarding the top-level value.",
            )
        else:
            extra_body["stream_options"] = top_level_stream_options
    if extra_body:
        kwargs["extra_body"] = extra_body
    return kwargs


def _merge_model_headers(llm: ChatFireworks, kwargs: dict[str, Any]) -> dict[str, Any]:
    """仅在调用其拥有模型时合并模型本地的请求头。"""
    model_headers = llm.model_kwargs.get("extra_headers")
    request_headers = kwargs.get("extra_headers")
    if isinstance(model_headers, Mapping) and isinstance(request_headers, Mapping):
                # SDK 在发送请求前会将 extra_body 合并到顶层字段之上。
        headers = {
            key.lower(): (key, value)
            for source in (model_headers, request_headers)
            for key, value in source.items()
        }
        return {
            **kwargs,
            "extra_headers": dict(headers.values()),
        }
    return kwargs


def _apply_prompt_cache_affinity(kwargs: dict[str, Any]) -> dict[str, Any]:
    """在合并选定模型的设置后应用中间件默认值。"""
    affinity = _PROMPT_CACHE_AFFINITY.get()
    if affinity is None:
        return kwargs

        # 流式生成器是惰性的 —— 先推进一次，使 HTTP 连接和任何传输错误发生在重试边界内。
    extra_body = kwargs.get("extra_body")
    body = {**kwargs, **extra_body} if isinstance(extra_body, Mapping) else kwargs
    if any(body.get(key) for key in ("user", "prompt_cache_key")):
        return kwargs

    headers = kwargs.get("extra_headers")
    if isinstance(headers, Mapping):
        if any(
            isinstance(key, str) and key.lower() == "x-session-affinity"
            for key in headers
        ):
            return kwargs
    elif headers is not None:
        logger.warning(
            "Cannot set Fireworks session affinity because extra_headers is %s",
            type(headers).__name__,
        )
        return kwargs

    return {
        **kwargs,
        "prompt_cache_key": affinity,
        "extra_headers": {**(headers or {}), "x-session-affinity": affinity},
    }


def _completion_with_retry(
    llm: ChatFireworks,
    run_manager: CallbackManagerForLLMRun | None = None,
    **kwargs: Any,
) -> Any:
    """重试同步完成调用，包括流式设置。"""
    retry_decorator = _create_retry_decorator(llm, run_manager=run_manager)
    kwargs = _prepare_sdk_kwargs(
        _apply_prompt_cache_affinity(_merge_model_headers(llm, kwargs))
    )

    @retry_decorator
    def _call() -> Any:
        try:
            result = llm.client.create(**kwargs)
        except httpx.HTTPStatusError as e:
            _promote_http_status_error(e)
        if kwargs.get("stream"):
                        # `_prepend_chunk`
            # 随后重新生成已消耗的块，使其位于其余内容之前，以便调用方仍能看见每个事件。
                        # 1.x 异步 `create()` 是一个协程，当 `stream=True` 时解析为
            # `AsyncStream`。需等待它，然后在重试边界内推进一次异步迭代器，使传输错误在此处暴露，而非在首个下游消费者处出现。
                        # 0.x 接受 `(connect, read)` 元组。1.x 的 SDK 仅接受浮点数、`httpx.Timeout`
            # 或 `None` —— 进行标准化以确保现有用户代码继续正常工作。
                        # `langchain-fireworks` 通过 `_create_retry_decorator`
            # 自行管理重试/退避机制，因此 LangChain `run_manager` 能看到每次尝试。抑制
            # SDK 内置的重试层，避免重复重试。
            try:
                iterator = iter(result)
                first = next(iterator)
            except StopIteration:
                _raise_empty_stream()
            except httpx.HTTPStatusError as e:
                _promote_http_status_error(e)
            return _prepend_chunk(first, iterator)
        return result

    return _call()


async def _acompletion_with_retry(
    llm: ChatFireworks,
    run_manager: AsyncCallbackManagerForLLMRun | None = None,
    **kwargs: Any,
) -> Any:
    """重试异步完成调用，包括流式设置。"""
    retry_decorator = _create_retry_decorator(llm, run_manager=run_manager)
    kwargs = _prepare_sdk_kwargs(
        _apply_prompt_cache_affinity(_merge_model_headers(llm, kwargs))
    )

    @retry_decorator
    async def _call() -> Any:
        if kwargs.get("stream"):
            try:
                                # 发生在流式传输中
                                # 当 `stream=True` 时为 `AsyncStream`。等待它，然后推进
                                # 异步迭代器一旦进入重试边界，传输
                                # 错误在此处暴露，而不是在第一个下游消费者处。
                result = await llm.async_client.create(**kwargs)
                agen = result.__aiter__()
                first = await agen.__anext__()
            except StopAsyncIteration:
                _raise_empty_stream()
            except httpx.HTTPStatusError as e:
                _promote_http_status_error(e)
            return _aprepend_chunk(first, agen)
        try:
            return await llm.async_client.create(**kwargs)
        except httpx.HTTPStatusError as e:
            _promote_http_status_error(e)

    return await _call()


def _prepend_chunk(first: Any, rest: Iterator[Any]) -> Iterator[Any]:
    yield first
    try:
        yield from rest
    except httpx.ReadTimeout as e:
        _handle_stream_read_timeout(e)


async def _aprepend_chunk(first: Any, rest: AsyncIterator[Any]) -> AsyncIterator[Any]:
    yield first
    try:
        async for item in rest:
            yield item
    except httpx.ReadTimeout as e:
        _handle_stream_read_timeout(e)


class ChatFireworks(BaseChatModel):
    """`Fireworks` 聊天大语言模型 API。

    To use, you should have the
    environment variable `FIREWORKS_API_KEY` set with your API key.

    Any parameters that are valid to be passed to the fireworks.create call
    can be passed in, even if not explicitly saved on this class.

    Example:
        ```python
        from langchain_fireworks.chat_models import ChatFireworks

        model = ChatFireworks(model_name="accounts/fireworks/models/gpt-oss-120b")
        ```

    Fireworks request headers can be passed with `extra_headers`. For prompt
    caching, `x-session-affinity` pins requests to a replica so related calls can
    reuse the same prompt-cache session:

    ```python
    model.invoke(
        "Hello",
        extra_headers={"x-session-affinity": "user-42"},
    )
    ```

    Fireworks SDK 还接受一个类型化的 `prompt_cache_key`
    字段（作为常规关键字参数传入），它将其视为对原始 `x-session-affinity`
    头的首选替代方案：

    ```python
    model.invoke("Hello", prompt_cache_key="user-42")
    ```
    """

    @property
    def lc_secrets(self) -> dict[str, str]:
        return {"fireworks_api_key": "FIREWORKS_API_KEY"}

    @classmethod
    def get_lc_namespace(cls) -> list[str]:
        """获取 LangChain 对象的命名空间。

        Returns:
            ["langchain", "chat_models", "fireworks"]
        """
        return ["langchain", "chat_models", "fireworks"]

    @property
    def lc_attributes(self) -> dict[str, Any]:
        attributes: dict[str, Any] = {}
        if self.fireworks_api_base:
            attributes["fireworks_api_base"] = self.fireworks_api_base

        return attributes

    @classmethod
    def is_lc_serializable(cls) -> bool:
        """返回此模型是否可被 LangChain 序列化。"""
        return True

    client: Any = Field(default=None, exclude=True)
    """内部 `fireworks.Fireworks().chat.completions` 资源。

    Constructed with `max_retries=0` so retries are owned by
    `_create_retry_decorator` (which surfaces each attempt to the LangChain
    `run_manager`). Callers reaching for this directly should set their own
    retry layer.
    """

    async_client: Any = Field(default=None, exclude=True)
    """内部 `fireworks.AsyncFireworks().chat.completions` 资源。

    通过 `max_retries=0` 构造；参见 `client`。
    """

    _sdk_client: Any = PrivateAttr(default=None)
    """拥有 `fireworks.Fireworks` 实例，保留以供 `close()` 调用底层 HTTPX 客户端。1.x SDK
本身不对 `chat.completions` 资源暴露生命周期方法。
    """

    _async_sdk_client: Any = PrivateAttr(default=None)
    """拥有 `fireworks.AsyncFireworks` 实例；参见 `_sdk_client`。"""

    model_name: str = Field(alias="model")
    """要使用的模型名称。"""

    @property
    def model(self) -> str:
        """与 model_name 相同。"""
        return self.model_name

    temperature: float | None = None
    """使用的采样温度。"""

    stop: str | list[str] | None = Field(default=None, alias="stop_sequences")
    """默认停止序列。"""

    model_kwargs: dict[str, Any] = Field(default_factory=dict)
    """持有适用于 `create` 调用但未显式指定的任何模型参数。"""

    fireworks_api_key: SecretStr = Field(default=SecretStr(""), alias="api_key")
    """Fireworks API 密钥。

    如果不提供，则自动从环境变量 `FIREWORKS_API_KEY` 读取。

    如果启用了 `LANGSMITH_GATEWAY` 且基础 URL 指向网关，则使用
    `LANGSMITH_GATEWAY_API_KEY`。
    """

    fireworks_api_base: str | None = Field(default=None, alias="base_url")
    """Base URL path for API requests, leave blank if not using a proxy or service
    emulator.

    如果设置了 `LANGSMITH_GATEWAY`，则在 `FIREWORKS_API_BASE` 之后作为备用使用。
    """

    request_timeout: float | tuple[float, float] | Any | None = Field(
        default=None, alias="timeout"
    )
    """Timeout for requests to Fireworks completion API. Can be `float`,
    `httpx.Timeout` or `None`.
    """

    streaming: bool = False
    """是否启用流式输出。"""

    stream_usage: bool = True
    """是否在流式输出中包含使用量元数据。

    If `True`, a final empty-content chunk carrying `usage_metadata` is emitted
    during the stream. Set to `False` if the upstream model/proxy rejects
    `stream_options`, or pass `stream_options` explicitly via `model_kwargs` or
    a runtime kwarg to override.

    !!! version-added "在 `langchain-fireworks` 1.2.0 中新增"

    !!! warning "在 `langchain-fireworks` 1.2.0 中行为已更改"

        流式现在默认启用 `stream_options.include_usage`，并且最终的空 `choices`
        块以携带 `usage_metadata` 的 `AIMessageChunk` 形式呈现，而非被静默丢弃。
    """

    n: int = 1
    """为每个提示词生成的聊天补全数量。"""

    max_tokens: int | None = None
    """生成的最大 token 数量。"""

    max_retries: int | None = 2
    """首次尝试后生成时的最大重试次数。

    重试使用指数退避策略，并在瞬态错误时触发：`RateLimitError`、`APIConnectionError`（包括其
    `APITimeoutError` 子类）、5xx 响应（包括那些以 `httpx.HTTPStatusError` 而非类型化
    SDK
    错误形式出现的情况），以及底层传输错误（`httpx.TimeoutException`、`httpx.TransportError`）。值为
    `None` 或 `0` 时禁用重试。
    """

    service_tier: str | None = None
    """请求的服务层级。

    当设置时，作为 Fireworks 聊天补全请求中的 `service_tier` 字段转发。传入
    `'priority'` 可选择 Fireworks 的优先级层级；保持 `None` 以使用默认层级。

    To use Fireworks' fast mode instead, select a fast-routed `model`; fast mode
    is not controlled by this field. See Fireworks'
    [serverless product docs](https://docs.fireworks.ai/guides/serverless-products)
    for the current list of fast routers and tiers.

    !!! version-added "在 `langchain-fireworks` 1.3.0 中新增"
    """
    reasoning_effort: str | None = None
    """推理努力程度。

    作为 `reasoning_effort` 请求字段转发。支持的值因模型而异；请参见模型的
    `profile.reasoning_effort_levels`。

    Can also be passed at call time, e.g.
    `model.invoke(..., reasoning_effort="high")`.
    """

    model_config = ConfigDict(
        populate_by_name=True,
    )

    @model_validator(mode="before")
    @classmethod
    def build_extra(cls, values: dict[str, Any]) -> Any:
        """从传入的额外参数构建额外的 kwargs。"""
        all_required_field_names = get_pydantic_field_names(cls)
        return _build_model_kwargs(values, all_required_field_names)

    @model_validator(mode="before")
    @classmethod
    def _resolve_gateway(cls, values: Any) -> Any:
        """解析基础 URL 和 API 密钥，应用 LangSmith 网关设置。

        An explicit ``base_url``/``api_key`` always wins. Otherwise the base URL
        falls back to ``FIREWORKS_API_BASE``, then the LangSmith gateway. The
        gateway key is preferred only when the base URL came from the gateway;
        for any other endpoint the provider key wins, and the gateway key is a
        candidate only when the gateway is enabled.
        """
        if isinstance(values, dict):
            config = _apply_gateway_config(
                values,
                cls,
                base_url_field="fireworks_api_base",
                api_key_field="fireworks_api_key",
                provider_path="fireworks",
                base_url_env="FIREWORKS_API_BASE",
                api_key_env="FIREWORKS_API_KEY",
            )
            if config.api_key is None:
                msg = (
                    "You must specify an api key. "
                    "You can pass it an argument as `api_key=...` or "
                    "set the environment variable `FIREWORKS_API_KEY`."
                )
                raise ValueError(msg)
        return values

    @model_validator(mode="after")
    def _set_fireworks_chat_version(self) -> Self:
        """在元数据中设置包版本。"""
        self._add_version("langchain-fireworks", __version__)
        return self

    @model_validator(mode="after")
    def validate_environment(self) -> Self:
        """验证环境中的 API 密钥和 Python 包是否存在。"""
        if self.n < 1:
            msg = "n must be at least 1."
            raise ValueError(msg)
        if self.n > 1 and self.streaming:
            msg = "n must be 1 when streaming."
            raise ValueError(msg)

        api_key = self.fireworks_api_key.get_secret_value()
        base_url = self.fireworks_api_base
                # 0.x 接受 `(connect, read)` 元组。1.x 的 SDK 仅接受
                # float，`httpx.Timeout`，或 `None` — 统一为现有用户代码的格式
                # 持续工作。
        if isinstance(self.request_timeout, tuple):
            connect, read = self.request_timeout
            timeout: Any = httpx.Timeout(read, connect=connect)
        else:
            timeout = self.request_timeout
                # `langchain-fireworks` 通过 `_create_retry_decorator` 拥有重试/退避机制
                # 因此 LangChain `run_manager` 能够看到每次尝试。抑制
                # SDK 内置的重试层，以避免重复重试。
        if not self.client:
            self._sdk_client = Fireworks(
                api_key=api_key,
                base_url=base_url,
                timeout=timeout,
                max_retries=0,
            )
            self.client = self._sdk_client.chat.completions
        if not self.async_client:
            self._async_sdk_client = AsyncFireworks(
                api_key=api_key,
                base_url=base_url,
                timeout=timeout,
                max_retries=0,
            )
            self.async_client = self._async_sdk_client.chat.completions
        return self

    def close(self) -> None:
        """关闭底层的同步 HTTP 客户端。

        调用后，对该模型的同步调用将引发异常。异步调用仍可用，直到也调用
        `aclose()`。可多次安全调用。
        """
        if self._sdk_client is not None:
            self._sdk_client.close()

    async def aclose(self) -> None:
        """关闭底层的异步 HTTP 客户端。

        Releases the aiohttp-backed connector that the 1.x SDK uses by
        default. Without this, transient `ChatFireworks` instances can leak
        an `Unclosed connector` warning at GC if the event loop has already
        stopped. Safe to call multiple times.
        """
        if self._async_sdk_client is not None:
            await self._async_sdk_client.close()

    def _resolve_model_profile(self) -> ModelProfile | None:
        return _get_default_model_profile(self.model_name) or None

    @property
    def _default_params(self) -> dict[str, Any]:
        """获取调用 Fireworks API 的默认参数。"""
        params = {
            "model": self.model_name,
            "stream": self.streaming,
            "n": self.n,
            "stop": self.stop,
            **self.model_kwargs,
        }
        if self.temperature is not None:
            params["temperature"] = self.temperature
        if self.max_tokens is not None:
            params["max_tokens"] = self.max_tokens
        if self.service_tier is not None:
            params["service_tier"] = self.service_tier
        if self.reasoning_effort is not None:
            params["reasoning_effort"] = self.reasoning_effort
        return params

    def _get_ls_params(
        self, stop: list[str] | None = None, **kwargs: Any
    ) -> LangSmithParams:
        """获取用于追踪的标准参数。"""
        params = self._get_invocation_params(stop=stop, **kwargs)
        ls_params = LangSmithParams(
            ls_provider="fireworks",
            ls_model_name=params.get("model", self.model_name),
            ls_model_type="chat",
            ls_temperature=params.get("temperature", self.temperature),
        )
        if ls_max_tokens := params.get("max_tokens", self.max_tokens):
            ls_params["ls_max_tokens"] = ls_max_tokens
        if ls_stop := stop or params.get("stop", None):
            ls_params["ls_stop"] = ls_stop
        return ls_params

    def _combine_llm_outputs(self, llm_outputs: list[dict | None]) -> dict:
        overall_token_usage: dict = {}
        system_fingerprint = None
        for output in llm_outputs:
            if output is None:
                                # 发生在流式传输时
                continue
            token_usage = output.get("token_usage")
            if token_usage is not None:
                for k, v in token_usage.items():
                    if v is None:
                        continue
                    if k in overall_token_usage:
                        overall_token_usage[k] = _update_token_usage(
                            overall_token_usage[k], v
                        )
                    else:
                        overall_token_usage[k] = v
            if system_fingerprint is None:
                system_fingerprint = output.get("system_fingerprint")
        combined = {"token_usage": overall_token_usage, "model_name": self.model_name}
        if system_fingerprint:
            combined["system_fingerprint"] = system_fingerprint
        return combined

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        message_dicts, params = self._create_message_dicts(messages, stop)
        params = {**params, **kwargs, "stream": True}
        if self.stream_usage and "stream_options" not in params:
            params["stream_options"] = {"include_usage": True}

        default_chunk_class: type[BaseMessageChunk] = AIMessageChunk
        try:
            stream = _completion_with_retry(
                self, run_manager=run_manager, messages=message_dicts, **params
            )
        except BadRequestError as e:
            _handle_fireworks_invalid_request(e)
        except APIError as e:
            _handle_fireworks_api_error(e)
        for chunk in stream:
            if not isinstance(chunk, dict):
                chunk = chunk.model_dump()
            message_chunk = _convert_chunk_to_message_chunk(chunk, default_chunk_class)
            if self.output_version == "v1" and isinstance(
                message_chunk, AIMessageChunk
            ):
                message_chunk = _format_chunk_for_v1(message_chunk)
            generation_info: dict[str, Any] = {}
            logprobs = None
            if choices := chunk.get("choices"):
                choice = choices[0]
                if finish_reason := choice.get("finish_reason"):
                    generation_info["finish_reason"] = finish_reason
                    generation_info["model_name"] = self.model_name
                    if service_tier := chunk.get("service_tier"):
                        generation_info["service_tier"] = service_tier
                logprobs = choice.get("logprobs")
                if logprobs:
                    generation_info["logprobs"] = logprobs
            default_chunk_class = message_chunk.__class__
            generation_chunk = ChatGenerationChunk(
                message=message_chunk, generation_info=generation_info or None
            )
            if run_manager:
                run_manager.on_llm_new_token(
                    generation_chunk.text, chunk=generation_chunk, logprobs=logprobs
                )
            yield generation_chunk

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        stream: bool | None = None,  # noqa: FBT001
        **kwargs: Any,
    ) -> ChatResult:
        should_stream = stream if stream is not None else self.streaming
        if should_stream:
            stream_iter = self._stream(
                messages, stop=stop, run_manager=run_manager, **kwargs
            )
            if self.output_version == "v1":
                stream_iter = _finalize_stream(stream_iter)
            return generate_from_stream(stream_iter)
        message_dicts, params = self._create_message_dicts(messages, stop)
        params = {
            **params,
            **({"stream": stream} if stream is not None else {}),
            **kwargs,
        }
        try:
            response = _completion_with_retry(
                self, run_manager=run_manager, messages=message_dicts, **params
            )
        except BadRequestError as e:
            _handle_fireworks_invalid_request(e)
        except APIError as e:
            _handle_fireworks_api_error(e)
        return self._create_chat_result(response)

    def _create_message_dicts(
        self, messages: list[BaseMessage], stop: list[str] | None
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        params = self._default_params
        if stop is not None:
            params["stop"] = stop
        message_dicts = [_convert_message_to_dict(m) for m in messages]
        return message_dicts, params

    def _create_chat_result(self, response: dict | BaseModel) -> ChatResult:
        generations = []
        if not isinstance(response, dict):
            response = response.model_dump()
        token_usage = response.get("usage", {})
        service_tier = response.get("service_tier")
        for res in response["choices"]:
            message = _convert_dict_to_message(res["message"])
            if isinstance(message, AIMessage):
                if token_usage:
                    message.usage_metadata = _usage_to_metadata(token_usage)
                    message.response_metadata["model_provider"] = "fireworks"
                    message.response_metadata["model_name"] = self.model_name
                if service_tier:
                    message.response_metadata["service_tier"] = service_tier
            generation_info = {"finish_reason": res.get("finish_reason")}
            if "logprobs" in res:
                generation_info["logprobs"] = res["logprobs"]
            gen = ChatGeneration(
                message=message,
                generation_info=generation_info,
            )
            generations.append(gen)
        llm_output = {
            "token_usage": token_usage,
            "system_fingerprint": response.get("system_fingerprint", ""),
        }
        if service_tier:
            llm_output["service_tier"] = service_tier
        return ChatResult(generations=generations, llm_output=llm_output)

    async def _astream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        message_dicts, params = self._create_message_dicts(messages, stop)
        params = {**params, **kwargs, "stream": True}
        if self.stream_usage and "stream_options" not in params:
            params["stream_options"] = {"include_usage": True}

        default_chunk_class: type[BaseMessageChunk] = AIMessageChunk
        try:
            stream = await _acompletion_with_retry(
                self, run_manager=run_manager, messages=message_dicts, **params
            )
        except BadRequestError as e:
            _handle_fireworks_invalid_request(e)
        except APIError as e:
            _handle_fireworks_api_error(e)
        async for chunk in stream:
            if not isinstance(chunk, dict):
                chunk = chunk.model_dump()
            message_chunk = _convert_chunk_to_message_chunk(chunk, default_chunk_class)
            if self.output_version == "v1" and isinstance(
                message_chunk, AIMessageChunk
            ):
                message_chunk = _format_chunk_for_v1(message_chunk)
            generation_info: dict[str, Any] = {}
            logprobs = None
            if choices := chunk.get("choices"):
                choice = choices[0]
                if finish_reason := choice.get("finish_reason"):
                    generation_info["finish_reason"] = finish_reason
                    generation_info["model_name"] = self.model_name
                    if service_tier := chunk.get("service_tier"):
                        generation_info["service_tier"] = service_tier
                logprobs = choice.get("logprobs")
                if logprobs:
                    generation_info["logprobs"] = logprobs
            default_chunk_class = message_chunk.__class__
            generation_chunk = ChatGenerationChunk(
                message=message_chunk, generation_info=generation_info or None
            )
            if run_manager:
                await run_manager.on_llm_new_token(
                    token=generation_chunk.text,
                    chunk=generation_chunk,
                    logprobs=logprobs,
                )
            yield generation_chunk

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        stream: bool | None = None,  # noqa: FBT001
        **kwargs: Any,
    ) -> ChatResult:
        should_stream = stream if stream is not None else self.streaming
        if should_stream:
            stream_iter = self._astream(
                messages, stop=stop, run_manager=run_manager, **kwargs
            )
            if self.output_version == "v1":
                stream_iter = _afinalize_stream(stream_iter)
            return await agenerate_from_stream(stream_iter)

        message_dicts, params = self._create_message_dicts(messages, stop)
        params = {
            **params,
            **({"stream": stream} if stream is not None else {}),
            **kwargs,
        }
        try:
            response = await _acompletion_with_retry(
                self, run_manager=run_manager, messages=message_dicts, **params
            )
        except BadRequestError as e:
            _handle_fireworks_invalid_request(e)
        except APIError as e:
            _handle_fireworks_api_error(e)
        return self._create_chat_result(response)

    @property
    def _identifying_params(self) -> dict[str, Any]:
        """获取标识性参数。"""
        return {"model_name": self.model_name, **self._default_params}

    def _get_invocation_params(
        self, stop: list[str] | None = None, **kwargs: Any
    ) -> dict[str, Any]:
        """获取用于调用模型的参数。"""
        return {
            "model": self.model_name,
            **super()._get_invocation_params(stop=stop),
            **self._default_params,
            **kwargs,
        }

    @property
    def _llm_type(self) -> str:
        """聊天模型的返回类型。"""
        return "fireworks-chat"

    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type[BaseModel] | Callable | BaseTool],
        *,
        tool_choice: dict | str | bool | None = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, AIMessage]:
        """将工具类对象绑定到此聊天模型。

        假设模型与 Fireworks 工具调用 API 兼容。

        Args:
            tools: 要绑定到此聊天模型的工具定义列表。

                支持由 [`convert_to_openai_tool`][langchain_core.utils.function_calling.convert_to_openai_tool] 处理的任何工具定义。
            tool_choice: 要求模型调用的工具。
                必须是单个提供的函数名称，
                `'auto'` 表示自动决定调用哪个函数（可选择不调用任何函数），
                `'any'` 表示强制调用某个函数，
                或者是一个字典，格式为：
                `{"type": "function", "function": {"name": <<tool_name>>}}`。
            **kwargs: 传递给
                `langchain_fireworks.chat_models.ChatFireworks.bind` 的任何其他参数
        """  # noqa: E501
        strict = kwargs.pop("strict", None)
        formatted_tools = [
            convert_to_openai_tool(tool, strict=strict) for tool in tools
        ]
        if tool_choice is not None and tool_choice:
            if isinstance(tool_choice, str) and (
                tool_choice not in ("auto", "any", "none")
            ):
                tool_choice = {"type": "function", "function": {"name": tool_choice}}
            if isinstance(tool_choice, bool):
                if len(tools) > 1:
                    msg = (
                        "tool_choice can only be True when there is one tool. Received "
                        f"{len(tools)} tools."
                    )
                    raise ValueError(msg)
                tool_name = formatted_tools[0]["function"]["name"]
                tool_choice = {
                    "type": "function",
                    "function": {"name": tool_name},
                }

            kwargs["tool_choice"] = tool_choice
        return super().bind(tools=formatted_tools, **kwargs)

    def with_structured_output(
        self,
        schema: dict | type[BaseModel] | None = None,
        *,
        method: Literal[
            "function_calling", "json_mode", "json_schema"
        ] = "function_calling",
        include_raw: bool = False,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, dict | BaseModel]:
        """返回与给定模式匹配格式输出的模型包装器。

        Args:
            schema: 输出模式。可作为以下形式传入：

                - OpenAI 函数/工具模式，
                - JSON 模式，
                - `TypedDict` 类，
                - 或 Pydantic 类。

                If `schema` is a Pydantic class then the model output will be a
                Pydantic instance of that class, and the model-generated fields will be
                validated by the Pydantic class. Otherwise the model output will be a
                dict and will not be validated.

                有关如何正确指定模式字段的类型和描述，请参见
                `langchain_core.utils.function_calling.convert_to_openai_tool`。

            method: 控制模型生成的方法，取值之一：

                - `'function_calling'`:
                    使用 Fireworks 的 [工具调用功能](https://docs.fireworks.ai/guides/function-calling)。
                - `'json_schema'`:
                    使用 Fireworks 的 [结构化输出功能](https://docs.fireworks.ai/structured-responses/structured-response-formatting)。
                - `'json_mode'`:
                    使用 Fireworks 的 [JSON 模式功能](https://docs.fireworks.ai/structured-responses/structured-response-formatting)。

                !!! warning "在 `langchain-fireworks` 0.2.8 中行为已更改"

                    添加对 `'json_schema'` 的支持。

            include_raw:
                如果为 `False`，则仅返回解析后的结构化输出。

                如果模型输出解析过程中发生错误，将抛出异常。

                如果为 `True`，则同时返回原始模型响应（一个
                `BaseMessage`）和解析后的模型响应。

                如果解析过程中发生错误，该错误将被捕获并一同返回。

                The final output is always a `dict` with keys `'raw'`, `'parsed'`, and
                `'parsing_error'`.

            kwargs: 传递给 `langchain.runnable.Runnable` 构造函数的任何其他参数。

        Returns:
            A `Runnable` that takes same inputs as a
                `langchain_core.language_models.chat.BaseChatModel`. If `include_raw` is
                `False` and `schema` is a Pydantic class, `Runnable` outputs an instance
                of `schema` (i.e., a Pydantic object). Otherwise, if `include_raw` is
                `False` then `Runnable` outputs a `dict`.

                如果 `include_raw` 为 `True`，则 `Runnable` 输出一个包含以下键的 `dict`：

                - `'raw'`: `BaseMessage`
                - `'parsed'`: `None` if there was a parsing error, otherwise the type
                    depends on the `schema` as described above.
                - `'parsing_error'`: `BaseException | None`

        Example: schema=Pydantic class, method="function_calling", include_raw=False:

        ```python
        from typing import Optional

        from langchain_fireworks import ChatFireworks
        from pydantic import BaseModel, Field


        class AnswerWithJustification(BaseModel):
            '''An answer to the user question along with justification for the answer.'''

            answer: str
            # If we provide default values and/or descriptions for fields, these will be passed
            # to the model. This is an important part of improving a model's ability to
            # correctly return structured outputs.
            justification: str | None = Field(
                default=None, description="A justification for the answer."
            )


        model = ChatFireworks(
            model="accounts/fireworks/models/gpt-oss-120b",
            temperature=0,
        )
        structured_model = model.with_structured_output(AnswerWithJustification)

        structured_model.invoke(
            "What weighs more a pound of bricks or a pound of feathers"
        )

        # -> AnswerWithJustification(
        #     answer='They weigh the same',
        #     justification='Both a pound of bricks and a pound of feathers weigh one pound. The weight is the same, but the volume or density of the objects may differ.'
        # )
        ```

        Example: schema=Pydantic class, method="function_calling", include_raw=True:

        ```python
        from langchain_fireworks import ChatFireworks
        from pydantic import BaseModel


        class AnswerWithJustification(BaseModel):
            '''An answer to the user question along with justification for the answer.'''

            answer: str
            justification: str


        model = ChatFireworks(
            model="accounts/fireworks/models/gpt-oss-120b",
            temperature=0,
        )
        structured_model = model.with_structured_output(
            AnswerWithJustification, include_raw=True
        )

        structured_model.invoke(
            "What weighs more a pound of bricks or a pound of feathers"
        )
        # -> {
        #     'raw': AIMessage(content='', additional_kwargs={'tool_calls': [{'id': 'call_Ao02pnFYXD6GN1yzc0uXPsvF', 'function': {'arguments': '{"answer":"They weigh the same.","justification":"Both a pound of bricks and a pound of feathers weigh one pound. The weight is the same, but the volume or density of the objects may differ."}', 'name': 'AnswerWithJustification'}, 'type': 'function'}]}),
        #     'parsed': AnswerWithJustification(answer='They weigh the same.', justification='Both a pound of bricks and a pound of feathers weigh one pound. The weight is the same, but the volume or density of the objects may differ.'),
        #     'parsing_error': None
        # }
        ```

        Example: schema=TypedDict class, method="function_calling", include_raw=False:

        ```python
        from typing_extensions import Annotated, TypedDict

        from langchain_fireworks import ChatFireworks


        class AnswerWithJustification(TypedDict):
            '''An answer to the user question along with justification for the answer.'''

            answer: str
            justification: Annotated[
                str | None, None, "A justification for the answer."
            ]


        model = ChatFireworks(
            model="accounts/fireworks/models/gpt-oss-120b",
            temperature=0,
        )
        structured_model = model.with_structured_output(AnswerWithJustification)

        structured_model.invoke(
            "What weighs more a pound of bricks or a pound of feathers"
        )
        # -> {
        #     'answer': 'They weigh the same',
        #     'justification': 'Both a pound of bricks and a pound of feathers weigh one pound. The weight is the same, but the volume and density of the two substances differ.'
        # }
        ```

        Example: schema=OpenAI function schema, method="function_calling", include_raw=False:

        ```python
        from langchain_fireworks import ChatFireworks

        oai_schema = {
            "name": "AnswerWithJustification",
            "description": "An answer to the user question along with justification for the answer.",
            "parameters": {
                "type": "object",
                "properties": {
                    "answer": {"type": "string"},
                    "justification": {
                        "description": "A justification for the answer.",
                        "type": "string",
                    },
                },
                "required": ["answer"],
            },
        }

        model = ChatFireworks(
            model="accounts/fireworks/models/gpt-oss-120b",
            temperature=0,
        )
        structured_model = model.with_structured_output(oai_schema)

        structured_model.invoke(
            "What weighs more a pound of bricks or a pound of feathers"
        )
        # -> {
        #     'answer': 'They weigh the same',
        #     'justification': 'Both a pound of bricks and a pound of feathers weigh one pound. The weight is the same, but the volume and density of the two substances differ.'
        # }
        ```

        Example: schema=Pydantic class, method="json_mode", include_raw=True:

        ```python
        from langchain_fireworks import ChatFireworks
        from pydantic import BaseModel


        class AnswerWithJustification(BaseModel):
            answer: str
            justification: str


        model = ChatFireworks(
            model="accounts/fireworks/models/gpt-oss-120b", temperature=0
        )
        structured_model = model.with_structured_output(
            AnswerWithJustification, method="json_mode", include_raw=True
        )

        structured_model.invoke(
            "Answer the following question. "
            "Make sure to return a JSON blob with keys 'answer' and 'justification'. "
            "What's heavier a pound of bricks or a pound of feathers?"
        )
        # -> {
        #     'raw': AIMessage(content='{"answer": "They are both the same weight.", "justification": "Both a pound of bricks and a pound of feathers weigh one pound. The difference lies in the volume and density of the materials, not the weight."}'),
        #     'parsed': AnswerWithJustification(answer='They are both the same weight.', justification='Both a pound of bricks and a pound of feathers weigh one pound. The difference lies in the volume and density of the materials, not the weight.'),
        #     'parsing_error': None
        # }
        ```

        Example: schema=None, method="json_mode", include_raw=True:

        ```python
        structured_model = model.with_structured_output(
            method="json_mode", include_raw=True
        )

        structured_model.invoke(
            "Answer the following question. "
            "Make sure to return a JSON blob with keys 'answer' and 'justification'. "
            "What's heavier a pound of bricks or a pound of feathers?"
        )
        # -> {
        #     'raw': AIMessage(content='{"answer": "They are both the same weight.", "justification": "Both a pound of bricks and a pound of feathers weigh one pound. The difference lies in the volume and density of the materials, not the weight."}'),
        #     'parsed': {
        #         'answer': 'They are both the same weight.',
        #         'justification': 'Both a pound of bricks and a pound of feathers weigh one pound. The difference lies in the volume and density of the materials, not the weight.'
        #     },
        #     'parsing_error': None
        # }
        ```

        """  # noqa: E501
        _ = kwargs.pop("strict", None)
        if kwargs:
            msg = f"Received unsupported arguments {kwargs}"
            raise ValueError(msg)
        is_pydantic_schema = _is_pydantic_class(schema)
        if method == "function_calling":
            if schema is None:
                msg = (
                    "schema must be specified when method is 'function_calling'. "
                    "Received None."
                )
                raise ValueError(msg)
            formatted_tool = convert_to_openai_tool(schema)
            tool_name = formatted_tool["function"]["name"]
            llm = self.bind_tools(
                [schema],
                tool_choice=tool_name,
                ls_structured_output_format={
                    "kwargs": {"method": "function_calling"},
                    "schema": formatted_tool,
                },
            )
            if is_pydantic_schema:
                output_parser: OutputParserLike = PydanticToolsParser(
                    tools=[schema],  # type: ignore[list-item]
                    first_tool_only=True,  # type: ignore[list-item]
                )
            else:
                output_parser = JsonOutputKeyToolsParser(
                    key_name=tool_name, first_tool_only=True
                )
        elif method == "json_schema":
            if schema is None:
                msg = (
                    "schema must be specified when method is 'json_schema'. "
                    "Received None."
                )
                raise ValueError(msg)
            formatted_schema = convert_to_json_schema(schema)
            llm = self.bind(
                response_format={"type": "json_object", "schema": formatted_schema},
                ls_structured_output_format={
                    "kwargs": {"method": "json_schema"},
                    "schema": schema,
                },
            )
            output_parser = (
                PydanticOutputParser(pydantic_object=schema)  # type: ignore[arg-type]
                if is_pydantic_schema
                else JsonOutputParser()
            )
        elif method == "json_mode":
            llm = self.bind(
                response_format={"type": "json_object"},
                ls_structured_output_format={
                    "kwargs": {"method": "json_mode"},
                    "schema": schema,
                },
            )
            output_parser = (
                PydanticOutputParser(pydantic_object=schema)  # type: ignore[type-var, arg-type]
                if is_pydantic_schema
                else JsonOutputParser()
            )
        else:
            msg = (
                f"Unrecognized method argument. Expected one of 'function_calling' or "
                f"'json_mode'. Received: '{method}'"
            )
            raise ValueError(msg)

        if include_raw:
            parser_assign = RunnablePassthrough.assign(
                parsed=itemgetter("raw") | output_parser, parsing_error=lambda _: None
            )
            parser_none = RunnablePassthrough.assign(parsed=lambda _: None)
            parser_with_fallback = parser_assign.with_fallbacks(
                [parser_none], exception_key="parsing_error"
            )
            return RunnableMap(raw=llm) | parser_with_fallback
        return llm | output_parser


def _is_pydantic_class(obj: Any) -> bool:
    return isinstance(obj, type) and is_basemodel_subclass(obj)


def _lc_tool_call_to_fireworks_tool_call(tool_call: ToolCall) -> dict:
    return {
        "type": "function",
        "id": tool_call["id"],
        "function": {
            "name": tool_call["name"],
            "arguments": json.dumps(tool_call["args"], ensure_ascii=False),
        },
    }


def _lc_invalid_tool_call_to_fireworks_tool_call(
    invalid_tool_call: InvalidToolCall,
) -> dict:
    return {
        "type": "function",
        "id": invalid_tool_call["id"],
        "function": {
            "name": invalid_tool_call["name"],
            "arguments": invalid_tool_call["args"],
        },
    }

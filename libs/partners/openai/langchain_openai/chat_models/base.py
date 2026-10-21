"""OpenAI 聊天封装。

!!! warning "API 范围"

        `ChatOpenAI` 仅针对 [官方 OpenAI API
        规范](https://github.com/openai/openai-openapi) 。第三方服务商（例如，
        `reasoning_content`, `reasoning_details`) 添加的非标准响应字段将 **不会**
        被提取或保留。如果您将 `base_url` 指向 OpenRouter、vLLM 或 DeepSeek
        等服务商，请改用对应的提供商专用 LangChain 包（例如
        `ChatDeepSeek`、`ChatOpenRouter`）
"""

from __future__ import annotations

import base64
import json
import logging
import os
import re
import ssl
import sys
import warnings
from collections.abc import (
    AsyncIterator,
    Awaitable,
    Callable,
    Iterator,
    Mapping,
    Sequence,
)
from functools import partial
from io import BytesIO
from json import JSONDecodeError
from math import ceil
from operator import itemgetter
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    TypeAlias,
    TypeVar,
    cast,
)
from urllib.parse import urlparse

import certifi
import openai
import tiktoken
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
    ModelProfileRegistry,
)
from langchain_core.language_models.chat_models import (
    BaseChatModel,
    LangSmithParams,
)
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
    is_data_content_block,
)
from langchain_core.messages import content as types
from langchain_core.messages.ai import (
    InputTokenDetails,
    OutputTokenDetails,
    UsageMetadata,
)
from langchain_core.messages.block_translators.openai import (
    _convert_from_v03_ai_message,
    convert_to_openai_data_block,
)
from langchain_core.messages.tool import tool_call_chunk
from langchain_core.output_parsers import JsonOutputParser, PydanticOutputParser
from langchain_core.output_parsers.openai_tools import (
    JsonOutputKeyToolsParser,
    PydanticToolsParser,
    make_invalid_tool_call,
    parse_tool_call,
)
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.runnables import (
    Runnable,
    RunnableLambda,
    RunnableMap,
    RunnablePassthrough,
)
from langchain_core.runnables.config import run_in_executor
from langchain_core.tools import BaseTool
from langchain_core.tools.base import _stringify
from langchain_core.utils import get_pydantic_field_names
from langchain_core.utils._gateway import (
    GATEWAY_METADATA_RESPONSE_KEY,
    _parse_gateway_metadata,
    _resolve_gateway_config,
)
from langchain_core.utils.function_calling import (
    convert_to_openai_function,
    convert_to_openai_tool,
)
from langchain_core.utils.pydantic import (
    PydanticBaseModel,
    TypeBaseModel,
    is_basemodel_subclass,
)
from langchain_core.utils.utils import LC_AUTO_PREFIX, _build_model_kwargs, from_env
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SecretStr,
    ValidationError,
    field_validator,
    model_validator,
)
from pydantic.v1 import BaseModel as BaseModelV1
from typing_extensions import Self

from langchain_openai._version import __version__
from langchain_openai.chat_models._client_utils import (
    _astream_with_chunk_timeout,
    _build_proxied_async_httpx_client,
    _build_proxied_sync_httpx_client,
    _float_env,
    _get_default_async_httpx_client,
    _get_default_httpx_client,
    _log_proxy_env_bypass_once,
    _resolve_socket_options,
    _resolve_sync_and_async_api_keys,
    _should_bypass_socket_options_for_proxy_env,
    _warn_if_proxy_env_shadowed,
)
from langchain_openai.chat_models._compat import (
    _convert_from_v1_to_chat_completions,
    _convert_from_v1_to_responses,
    _convert_to_v03_ai_message,
    _unwrap_non_standard,
)
from langchain_openai.data._profiles import _PROFILES

if TYPE_CHECKING:
    import httpx
    from langchain_core.language_models import ModelProfile
    from openai.types.responses import Response

logger = logging.getLogger(__name__)

# 此 SSL 上下文等同于默认的 `verify=True`。
# https://www.python-httpx.org/advanced/ssl/#configuring-client-instances
global_ssl_context = ssl.create_default_context(cafile=certifi.where())

_ssrf_client: httpx.Client | None = None


def _get_ssrf_safe_client() -> httpx.Client:
    global _ssrf_client
    if _ssrf_client is None:
        from langchain_core._security._transport import ssrf_safe_client

        _ssrf_client = ssrf_safe_client(
            verify=global_ssl_context, follow_redirects=False
        )
    return _ssrf_client


_MODEL_PROFILES = cast(ModelProfileRegistry, _PROFILES)


def _get_default_model_profile(model_name: str) -> ModelProfile:
    default = _MODEL_PROFILES.get(model_name) or {}
    return default.copy()


WellKnownTools = (
    "file_search",
    "web_search_preview",
    "web_search",
    "computer_use_preview",
    "code_interpreter",
    "mcp",
    "image_generation",
    "tool_search",
    "apply_patch",
)

_TOOL_EXTRAS_PASSTHROUGH = ("defer_loading", "async")


def _convert_dict_to_message(_dict: Mapping[str, Any]) -> BaseMessage:
    """将字典转换为 LangChain 消息。

    Args:
        _dict: 字典本身。

    Returns:
        LangChain 消息。
    """
    role = _dict.get("role")
    name = _dict.get("name")
    id_ = _dict.get("id")
    if role == "user":
        return HumanMessage(content=_dict.get("content", ""), id=id_, name=name)
    if role == "assistant":
                # Azure 的修复
                # OpenAI 也对工具调用返回 None
        content = _dict.get("content", "") or ""
        additional_kwargs: dict = {}
        if function_call := _dict.get("function_call"):
            additional_kwargs["function_call"] = dict(function_call)
        tool_calls = []
        invalid_tool_calls = []
        if raw_tool_calls := _dict.get("tool_calls"):
            for raw_tool_call in raw_tool_calls:
                try:
                    tool_calls.append(parse_tool_call(raw_tool_call, return_id=True))
                except Exception as e:
                    invalid_tool_calls.append(
                        make_invalid_tool_call(raw_tool_call, str(e))
                    )
        if audio := _dict.get("audio"):
            additional_kwargs["audio"] = audio
        return AIMessage(
            content=content,
            additional_kwargs=additional_kwargs,
            name=name,
            id=id_,
            tool_calls=tool_calls,
            invalid_tool_calls=invalid_tool_calls,
        )
    if role in ("system", "developer"):
        additional_kwargs = {"__openai_role__": role} if role == "developer" else {}
        return SystemMessage(
            content=_dict.get("content", ""),
            name=name,
            id=id_,
            additional_kwargs=additional_kwargs,
        )
    if role == "function":
        return FunctionMessage(
            content=_dict.get("content", ""), name=cast(str, _dict.get("name")), id=id_
        )
    if role == "tool":
        additional_kwargs = {}
        if "name" in _dict:
            additional_kwargs["name"] = _dict["name"]
        return ToolMessage(
            content=_dict.get("content", ""),
            tool_call_id=cast(str, _dict.get("tool_call_id")),
            additional_kwargs=additional_kwargs,
            name=name,
            id=id_,
        )
    return ChatMessage(content=_dict.get("content", ""), role=role, id=id_)  # type: ignore[arg-type]


def _apply_prompt_cache_breakpoint(
    source_block: dict[str, Any], formatted_block: dict[str, Any]
) -> dict[str, Any]:
    """对格式化内容块应用 OpenAI 提示词缓存断点。

    直接设置在块上的断点优先于嵌套在 `extras` 中的断点。
    是否复制由成员关系决定（而非真值），因此即使值存在但为假（例如 `None`）也会被保留。
    """
    if "prompt_cache_breakpoint" in source_block:
        formatted_block["prompt_cache_breakpoint"] = source_block[
            "prompt_cache_breakpoint"
        ]
    elif isinstance(extras := source_block.get("extras"), dict) and (
        "prompt_cache_breakpoint" in extras
    ):
        formatted_block["prompt_cache_breakpoint"] = extras["prompt_cache_breakpoint"]
    return formatted_block


def _sanitize_chat_completions_content(content: str | list[dict]) -> str | list[dict]:
    """清理用于聊天/补全 API 的内容。

    对于列表类型内容，仅保留支持的键。
    """
    if isinstance(content, list):
        sanitized = []
        for block in content:
            if (
                isinstance(block, dict)
                and block.get("type") == "text"
                and "text" in block
            ):
                sanitized_block = {"type": "text", "text": block["text"]}
                if "prompt_cache_breakpoint" in block:
                    sanitized_block["prompt_cache_breakpoint"] = block[
                        "prompt_cache_breakpoint"
                    ]
                sanitized.append(sanitized_block)
            else:
                sanitized.append(block)
        return sanitized
    return content


_ADDITIONAL_TOOLS_BLOCK_TYPE = "additional_tools"
"""Responses API input item that adds tools partway through a conversation."""


def _is_ai_role(role: str | None) -> bool:
    """判断消息的角色是否为助手角色。

    Assistant content is replayed model output, not a caller's instruction, so it is
    exempt from the placement checks a caller's own blocks are held to.
    """
    return str(role).lower().startswith("ai")


def _is_system_role(role: str | None) -> bool:
    """判断消息的角色是否携带提供方指令。

    `SystemMessage` 无论是否以 OpenAI 的 `developer` 角色发出，均报告
    `"system"`，因此一次检查覆盖两种拼写形式。
    """
    return role in ("system", "developer")


def _raise_if_additional_tools(content: Any, reason: str) -> None:
    """拒绝无法在当前位置生效的 `additional_tools` 块。

    该块仅作为系统消息携带的 Responses
    顶层输入项到达网络。在其他任何位置，它都是此提供方自身的块类型，且处于该提供方禁止的位置，错误分类体系会明确提示而非静默处理：由于消息内容不会路由请求至
    Responses API，静默丢弃会导致错误情况成为默认结果。

    Args:
        content: 消息的内容。
        reason: 解释为何该位置无法生效，以及如何修复。
            附加到错误信息中。

    Raises:
        ValueError: 如果存在 `additional_tools` 块，无论拼写如何。
    """
    if not isinstance(content, list):
        return
    for raw_block in content:
        if (
            isinstance(raw_block, dict)
            and _unwrap_non_standard(raw_block).get("type")
            == _ADDITIONAL_TOOLS_BLOCK_TYPE
        ):
            msg = f"`additional_tools` {reason}"
            raise ValueError(msg)


def _format_message_content(
    content: Any,
    api: Literal["chat/completions", "responses"] = "chat/completions",
    role: str | None = None,
) -> Any:
    """格式化消息内容。"""
    if _is_ai_role(role):
                # 重放的助手输出；`additional_tools` 也是一个输出项，因此
                # 一个回显项必须能通过一轮往返，而不是中止请求。
        pass
    elif not _is_system_role(role):
        _raise_if_additional_tools(
            content,
            "must be carried on a `SystemMessage`. OpenAI restricts the input item "
            'to `role: "developer"`, so it cannot be sent on any other message.',
        )
    elif api == "chat/completions":
        _raise_if_additional_tools(
            content,
            "requires the Responses API and cannot be sent via Chat Completions. "
            "Set `use_responses_api=True`.",
        )
    if content and isinstance(content, list):
        formatted_content = []
        for block in content:
                        # 移除意外的块类型
            if (
                isinstance(block, dict)
                and "type" in block
                and (
                    block["type"] in ("tool_use", "thinking", "reasoning_content")
                    or (
                        block["type"] in ("function_call", "code_interpreter_call")
                        and api == "chat/completions"
                    )
                )
            ):
                continue
            if (
                isinstance(block, dict)
                and is_data_content_block(block)
                                # 响应 API 消息在 _compat 中单独处理（解析为
                                # 图像生成调用）
                and not (api == "responses" and str(role).lower().startswith("ai"))
            ):
                formatted_block = convert_to_openai_data_block(block, api=api)
                formatted_content.append(
                    _apply_prompt_cache_breakpoint(block, formatted_block)
                )
            elif (
                isinstance(block, dict)
                and block.get("type") == "text"
                and "text" in block
                and isinstance(extras := block.get("extras"), dict)
                and "prompt_cache_breakpoint" in extras
            ):
                formatted_block = {"type": "text", "text": block["text"]}
                formatted_content.append(
                    _apply_prompt_cache_breakpoint(block, formatted_block)
                )
                        # Anthropic 图像块
            elif (
                isinstance(block, dict)
                and block.get("type") == "image"
                and (source := block.get("source"))
                and isinstance(source, dict)
            ):
                if source.get("type") == "base64" and (
                    (media_type := source.get("media_type"))
                    and (data := source.get("data"))
                ):
                    formatted_content.append(
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{media_type};base64,{data}"},
                        }
                    )
                elif source.get("type") == "url" and (url := source.get("url")):
                    formatted_content.append(
                        {"type": "image_url", "image_url": {"url": url}}
                    )
                else:
                    continue
            else:
                formatted_content.append(block)
    else:
        formatted_content = content

    return formatted_content


def _convert_message_to_dict(
    message: BaseMessage,
    api: Literal["chat/completions", "responses"] = "chat/completions",
) -> dict:
    """将 LangChain 消息转换为 OpenAI 所期望的字典格式。"""
    message_dict: dict[str, Any] = {
        "content": _format_message_content(message.content, api=api, role=message.type)
    }
    if (name := message.name or message.additional_kwargs.get("name")) is not None:
        message_dict["name"] = name

        # 填充角色和其他消息数据
    if isinstance(message, ChatMessage):
        message_dict["role"] = message.role
    elif isinstance(message, HumanMessage):
        message_dict["role"] = "user"
    elif isinstance(message, AIMessage):
        message_dict["role"] = "assistant"
        if message.tool_calls or message.invalid_tool_calls:
            message_dict["tool_calls"] = [
                _lc_tool_call_to_openai_tool_call(tc) for tc in message.tool_calls
            ] + [
                _lc_invalid_tool_call_to_openai_tool_call(tc)
                for tc in message.invalid_tool_calls
            ]
        elif "tool_calls" in message.additional_kwargs:
            message_dict["tool_calls"] = message.additional_kwargs["tool_calls"]
            tool_call_supported_props = {"id", "type", "function"}
            message_dict["tool_calls"] = [
                {k: v for k, v in tool_call.items() if k in tool_call_supported_props}
                for tool_call in message_dict["tool_calls"]
            ]
        elif "function_call" in message.additional_kwargs:
                        # OpenAI 在同一消息中同时存在 function_call 和 tool_calls 时会抛出 400
            # 错误。
                        # 如果存在工具调用，content 的 null 值应为 None 而非空字符串。
            message_dict["function_call"] = message.additional_kwargs["function_call"]
        else:
            pass
                # openai 不支持返回数据 - 仅支持返回 id
        if "function_call" in message_dict or "tool_calls" in message_dict:
            message_dict["content"] = message_dict["content"] or None

        audio: dict[str, Any] | None = None
        for block in message.content:
            if (
                isinstance(block, dict)
                and block.get("type") == "audio"
                and (id_ := block.get("id"))
                and api != "responses"
            ):
                                # https://platform.openai.com/docs/guides/audio/multi-turn-conversations
                                # token 使用量可以是整数或字典
                audio = {"id": id_}
        if not audio and "audio" in message.additional_kwargs:
            raw_audio = message.additional_kwargs["audio"]
            audio = (
                {"id": message.additional_kwargs["audio"]["id"]}
                if "id" in raw_audio
                else raw_audio
            )
        if audio:
            message_dict["audio"] = audio
    elif isinstance(message, SystemMessage):
        message_dict["role"] = message.additional_kwargs.get(
            "__openai_role__", "system"
        )
    elif isinstance(message, FunctionMessage):
        message_dict["role"] = "function"
    elif isinstance(message, ToolMessage):
        message_dict["role"] = "tool"
        message_dict["tool_call_id"] = message.tool_call_id
        message_dict["content"] = _sanitize_chat_completions_content(
            message_dict["content"]
        )
        supported_props = {"content", "role", "tool_call_id"}
        message_dict = {k: v for k, v in message_dict.items() if k in supported_props}
    else:
        msg = f"Got unknown type {message}"
        raise TypeError(msg)
    return message_dict


def _convert_delta_to_message_chunk(
    _dict: Mapping[str, Any], default_class: type[BaseMessageChunk]
) -> BaseMessageChunk:
    """转换为 LangChain 消息块。"""
    id_ = _dict.get("id")
    role = cast(str, _dict.get("role"))
    content = cast(str, _dict.get("content") or "")
    additional_kwargs: dict = {}
    if _dict.get("function_call"):
        function_call = dict(_dict["function_call"])
        if "name" in function_call and function_call["name"] is None:
            function_call["name"] = ""
        additional_kwargs["function_call"] = function_call
    tool_call_chunks = []
    if raw_tool_calls := _dict.get("tool_calls"):
        try:
            tool_call_chunks = [
                tool_call_chunk(
                    name=rtc["function"].get("name"),
                    args=rtc["function"].get("arguments"),
                    id=rtc.get("id"),
                    index=rtc["index"],
                )
                for rtc in raw_tool_calls
            ]
        except KeyError:
            pass

    if role == "user" or default_class == HumanMessageChunk:
        return HumanMessageChunk(content=content, id=id_)
    if role == "assistant" or default_class == AIMessageChunk:
        return AIMessageChunk(
            content=content,
            additional_kwargs=additional_kwargs,
            id=id_,
            tool_call_chunks=tool_call_chunks,  # type: ignore[arg-type]
        )
    if role in ("system", "developer") or default_class == SystemMessageChunk:
        if role == "developer":
            additional_kwargs = {"__openai_role__": "developer"}
        else:
            additional_kwargs = {}
        return SystemMessageChunk(
            content=content, id=id_, additional_kwargs=additional_kwargs
        )
    if role == "function" or default_class == FunctionMessageChunk:
        return FunctionMessageChunk(content=content, name=_dict["name"], id=id_)
    if role == "tool" or default_class == ToolMessageChunk:
        return ToolMessageChunk(
            content=content, tool_call_id=_dict["tool_call_id"], id=id_
        )
    if role or default_class == ChatMessageChunk:
        return ChatMessageChunk(content=content, role=role, id=id_)
    return default_class(content=content, id=id_)  # type: ignore[call-arg]


def _update_token_usage(
    overall_token_usage: int | dict, new_usage: int | dict
) -> int | dict:
        # `reasoning_tokens` 嵌套在 `completion_tokens_details` 内部
        # 支持为 OpenAI 显式配置代理
    if isinstance(new_usage, int):
        if not isinstance(overall_token_usage, int):
            msg = (
                f"Got different types for token usage: "
                f"{type(new_usage)} and {type(overall_token_usage)}"
            )
            raise ValueError(msg)
        return new_usage + overall_token_usage
    if isinstance(new_usage, dict):
        if not isinstance(overall_token_usage, dict):
            msg = (
                f"Got different types for token usage: "
                f"{type(new_usage)} and {type(overall_token_usage)}"
            )
            raise ValueError(msg)
        return {
            k: _update_token_usage(overall_token_usage.get(k, 0), v)
            for k, v in new_usage.items()
        }
    warnings.warn(f"Unexpected type for token usage: {type(new_usage)}")
    return new_usage


class OpenAIContextOverflowError(openai.BadRequestError, ContextOverflowError):
    """当输入超出 OpenAI 上下文限制时抛出 BadRequestError。"""


class OpenAIAPIContextOverflowError(openai.APIError, ContextOverflowError):
    """当输入超出 OpenAI 上下文限制时抛出 APIError。"""


class OpenAIAuthenticationError(openai.AuthenticationError, ModelAuthenticationError):
    """OpenAI 认证错误归类为 LangChain 模型错误。"""


class OpenAIPermissionDeniedError(
    openai.PermissionDeniedError, ModelPermissionDeniedError
):
    """OpenAI 权限错误归类为 LangChain 模型错误。"""


class OpenAIInvalidRequestError(openai.BadRequestError, ModelInvalidRequestError):
    """OpenAI 错误请求错误归类为 LangChain 模型错误。"""


class OpenAIModelNotFoundError(openai.NotFoundError, ModelNotFoundError):
    """OpenAI 未找到错误归类为 LangChain 模型错误。"""


class OpenAIRateLimitError(openai.RateLimitError, ModelRateLimitError):
    """OpenAI 速率限制错误归类为 LangChain 模型错误。"""


class OpenAIAPIError(openai.InternalServerError, ModelAPIError):
    """OpenAI 服务器错误归类为 LangChain 模型错误。"""


class OpenAIConnectionError(openai.APIConnectionError, ModelConnectionError):
    """OpenAI 连接错误归类为 LangChain 模型错误。"""


class OpenAITimeoutError(openai.APITimeoutError, ModelTimeoutError):
    """OpenAI 超时错误归类为 LangChain 模型错误。"""


def _handle_openai_bad_request(e: openai.BadRequestError) -> None:
    if (
        "context_length_exceeded" in str(e)
        or "Input tokens exceed the configured limit" in e.message
        or "prompt is too long" in e.message
        or "ContextWindowExceededError" in e.message
    ):
        raise OpenAIContextOverflowError(
            message=e.message, response=e.response, body=e.body
        ) from e
    if (
        "'response_format' of type 'json_schema' is not supported with this model"
    ) in e.message:
        message = (
            "This model does not support OpenAI's structured output feature, which "
            "is the default method for `with_structured_output` as of "
            "langchain-openai==0.3. To use `with_structured_output` with this model, "
            'specify `method="function_calling"`.'
        )
        warnings.warn(message)
        raise OpenAIInvalidRequestError(
            message=e.message, response=e.response, body=e.body
        ) from e
    if "Invalid schema for response_format" in e.message:
        message = (
            "Invalid schema for OpenAI's structured output feature, which is the "
            "default method for `with_structured_output` as of langchain-openai==0.3. "
            'Specify `method="function_calling"` instead or update your schema. '
            "See supported schemas: "
            "https://platform.openai.com/docs/guides/structured-outputs#supported-schemas"
        )
        warnings.warn(message)
    raise OpenAIInvalidRequestError(
        message=e.message, response=e.response, body=e.body
    ) from e


def _handle_openai_api_error(e: openai.APIError) -> None:
    error_message = str(e)
    if "exceeds the context window" in error_message:
        raise OpenAIAPIContextOverflowError(
            message=e.message, request=e.request, body=e.body
        ) from e
    if isinstance(e, openai.AuthenticationError):
        raise OpenAIAuthenticationError(
            message=e.message, response=e.response, body=e.body
        ) from e
    if isinstance(e, openai.PermissionDeniedError):
        raise OpenAIPermissionDeniedError(
            message=e.message, response=e.response, body=e.body
        ) from e
    if isinstance(e, openai.NotFoundError):
        raise OpenAIModelNotFoundError(
            message=e.message, response=e.response, body=e.body
        ) from e
    if isinstance(e, openai.RateLimitError):
        raise OpenAIRateLimitError(
            message=e.message, response=e.response, body=e.body
        ) from e
    if isinstance(e, openai.InternalServerError):
        raise OpenAIAPIError(message=e.message, response=e.response, body=e.body) from e
    if isinstance(e, openai.APITimeoutError):
        raise OpenAITimeoutError(e.request) from e
    if isinstance(e, openai.APIConnectionError):
        raise OpenAIConnectionError(message=e.message, request=e.request) from e
    raise


def _add_gateway_metadata(generation_info: dict[str, Any], raw_response: Any) -> None:
    """若存在，将解析后的 LangSmith 网关元数据添加至 `generation_info`

    Args:
        generation_info: 用于就地修改的生成信息。
        raw_response: 原始提供方响应，或 None。
    """
    headers = getattr(raw_response, "headers", None)
    if headers is None:
        return
    gateway_metadata = _parse_gateway_metadata(headers)
    if gateway_metadata is not None:
        generation_info[GATEWAY_METADATA_RESPONSE_KEY] = gateway_metadata


_RESPONSES_API_ONLY_PREFIXES = (
    "gpt-5-pro",
    "gpt-5.2-pro",
    "gpt-5.4-pro",
    "gpt-5.5-pro",
    "gpt-5.6-sol",
)


def _model_prefers_responses_api(model_name: str | None) -> bool:
    if not model_name:
        return False
    return model_name.startswith(_RESPONSES_API_ONLY_PREFIXES) or "codex" in model_name


_BM = TypeVar("_BM", bound=BaseModel)
_DictOrPydanticClass: TypeAlias = dict[str, Any] | type[_BM] | type
_DictOrPydantic: TypeAlias = dict | _BM


class BaseChatOpenAI(BaseChatModel):
    """OpenAI 大型语言模型聊天功能的基础封装。

    This base class targets
    [official OpenAI API specifications](https://github.com/openai/openai-openapi)
    only. Non-standard response fields added by third-party providers (e.g.,
    `reasoning_content`) are not extracted. Use a provider-specific subclass for
    full provider support.
    """

    client: Any = Field(default=None, exclude=True)

    async_client: Any = Field(default=None, exclude=True)

    root_client: Any = Field(default=None, exclude=True)

    root_async_client: Any = Field(default=None, exclude=True)

    model_name: str = Field(default="gpt-3.5-turbo", alias="model")
    """要使用的模型名称。"""

    temperature: float | None = None
    """采样温度。"""

    model_kwargs: dict[str, Any] = Field(default_factory=dict)
    """保存任何对 `create` 调用有效的模型参数，但未明确指定的参数。"""

    openai_api_key: (
        SecretStr | None | Callable[[], str] | Callable[[], Awaitable[str]]
    ) = Field(alias="api_key", default=None)
    """API 密钥。

    Can be inferred from the `OPENAI_API_KEY` environment variable, or specified
    as a string, or sync or async callable that returns a string.

    ??? example "通过环境变量指定"

        ```bash
        export OPENAI_API_KEY=...
        ```
        ```python
        from langchain_openai import ChatOpenAI

        model = ChatOpenAI(model="gpt-5-nano")
        ```

    ??? example "通过字符串指定"

        ```python
        from langchain_openai import ChatOpenAI

        model = ChatOpenAI(model="gpt-5-nano", api_key="...")
        ```

    ??? example "通过同步可调用对象指定"

        ```python
        from langchain_openai import ChatOpenAI

        def get_api_key() -> str:
            # Custom logic to retrieve API key
            return "..."

        model = ChatOpenAI(model="gpt-5-nano", api_key=get_api_key)
        ```

    ??? example "通过异步可调用对象指定"

        ```python
        from langchain_openai import ChatOpenAI

        async def get_api_key() -> str:
            # Custom async logic to retrieve API key
            return "..."

        model = ChatOpenAI(model="gpt-5-nano", api_key=get_api_key)
        ```
    """

    openai_api_base: str | None = Field(default=None, alias="base_url")
    """API 请求的基 URL 路径，若不使用代理或服务模拟器则留空。

    解析顺序（首个匹配项生效）：

    1. 显式 `base_url`（或 `openai_api_base`）关键字参数。
    2. 环境变量 `OPENAI_API_BASE`（LangChain 初始化时读取）。
    3. 环境变量 `OPENAI_BASE_URL`（底层 `openai` SDK 客户端读取）。

    `OPENAI_BASE_URL` 也会被 LangChain 检查，以决定是否默认启用 `stream_usage` ——
    当设置时，默认关闭，因为许多非 OpenAI 接口不支持流式 token 使用量。
    """  # noqa: E501

    openai_organization: str | None = Field(default=None, alias="organization")
    """如果未提供，则自动从环境变量 `OPENAI_ORG_ID` 推断。"""

        # 配置自定义 httpx 客户端。详见
    openai_proxy: str | None = Field(
        default_factory=from_env("OPENAI_PROXY", default=None)
    )

    request_timeout: float | tuple[float, float] | Any | None = Field(
        default=None, alias="timeout"
    )
    """对 OpenAI 补全 API 请求的超时时间。

    可为 float、`httpx.Timeout` 或 `None`。
    """

    stream_usage: bool | None = None
    """是否在流式输出中包含使用元数据。

    若启用，流式过程中将额外生成一条包含使用元数据的消息块。

    This parameter is enabled unless `openai_api_base` is set or the model is
    initialized with a custom client, as many chat completions APIs do not
    support streaming token usage.

    !!! version-added "在 `langchain-openai` 0.3.9 中新增"

    !!! warning "在 `langchain-openai` 0.3.35 中行为已更改"

        默认基 URL 和客户端下启用。
    """

    max_retries: int | None = None
    """生成时的最大重试次数。"""

    presence_penalty: float | None = None
    """惩罚重复 token。"""

    frequency_penalty: float | None = None
    """根据频率惩罚重复 token。"""

    seed: int | None = None
    """生成的随机种子。"""

    logprobs: bool | None = None
    """是否返回 logprobs。"""

    top_logprobs: int | None = None
    """Number of most likely tokens to return at each token position, each with an
    associated log probability.

    `logprobs` 必须设为 true 才能使用此参数。
    """

    logit_bias: dict[int, int] | None = None
    """修改指定 token 出现在补全结果中的可能性。"""

    streaming: bool = False
    """是否流式返回结果。"""

    n: int | None = None
    """为每个提示生成的聊天补全数量。"""

    top_p: float | None = None
    """在每一步考虑的 token 的总概率质量。"""

    max_tokens: int | None = Field(default=None)
    """最大生成 token 数量。"""

    reasoning_effort: str | None = None
    """限制推理模型的推理努力程度。

    仅用于 Chat Completions API。仅适用于推理模型。

    当前支持的值有 `'minimal'`、`'low'`、`'medium'` 和
    `'high'`。降低推理努力程度可带来更快的响应速度，并减少响应中的推理 token 数量。

    !!! note "在对话过程中更改推理努力程度"

        在对话中途更改此值会修改请求级别的参数，从而导致缓存的提示词前缀失效。

        支持该功能的模型（目前为 GPT-6）可以将新的努力程度作为附加到消息中的
        `configuration_update` 项来传递：

        ```python
        HumanMessage(
            [
                {"type": "configuration_update", "reasoning": {"effort": "high"}},
                {"type": "text", "text": "Analyze the failure modes."},
            ]
        )
        ```

        The new effort applies from that message onward, until another update
        overrides it.
    """

    reasoning: dict[str, Any] | None = None
    """推理模型的推理参数。设为 None 表示禁用推理。

    仅用于 Responses API。

    ```python
    reasoning={
        "effort": None,  # Default None; can be "low", "medium", or "high"
        "summary": "auto",  # Can be "auto", "concise", or "detailed"
    }
    ```

    !!! version-added "在 `langchain-openai` 0.3.24 中新增"
    """

    verbosity: str | None = None
    """控制推理模型响应的详细程度。

    仅用于 Responses API。

    当前支持的值有 `'low'`、`'medium'` 和 `'high'`。

    !!! version-added "在 `langchain-openai` 0.3.28 中新增"
    """

    tiktoken_model_name: str | None = None
    """当使用此类时，要传递给 tiktoken 的模型名称。

    tiktoken 用于计算文档中的 token 数量，以确保其不超过某个限制。

    By default, when set to `None`, this will be the same as the embedding model name.
    However, there are some cases where you may want to use this `Embedding` class with
    a model name not supported by tiktoken. This can include when using Azure embeddings
    or when using one of the many model providers that expose an OpenAI-like
    API but with different models. In those cases, in order to avoid erroring
    when tiktoken is called, you can specify a model name to use here.
    """

    default_headers: Mapping[str, str] | None = None

    default_query: Mapping[str, object] | None = None

        # [httpx 文档](https://www.python-httpx.org/api/#client) 获取更多详情。
        # 对于 o1 模型，若未提供，则设置 temperature=1
    http_client: Any | None = Field(default=None, exclude=True)
    """可选的 `httpx.Client`。

    Only used for sync invocations. Must specify `http_async_client` as well if
    you'd like a custom client for async invocations.
    """

    http_async_client: Any | None = Field(default=None, exclude=True)
    """可选的 `httpx.AsyncClient`。

    Only used for async invocations. Must specify `http_client` as well if you'd
    like a custom client for sync invocations.
    """

    http_socket_options: Sequence[tuple[int, int, int]] | None = Field(
        default=None, exclude=True
    )
    """应用于由本实例构建的 httpx 传输的 TCP 套接字选项。

    Defaults to a conservative TCP-keepalive + `TCP_USER_TIMEOUT` profile that
    targets a ~2-minute bound on silent connection hangs (silent mid-stream peer
    loss, gVisor/NAT idle timeouts, silent TCP black holes) on platforms that
    support the full option set. On platforms that only support a subset
    (macOS without `TCP_USER_TIMEOUT`, Windows with only `SO_KEEPALIVE`,
    minimal kernels), unsupported options are silently dropped and the bound
    degrades to whatever the remaining options + OS defaults provide — still
    better than indefinite hang.

    接受的值：

    - `None` (default): use env-driven defaults. Matches the "unset" convention
        used by `http_client` elsewhere on this class.
    - `()` (empty): disable socket-option injection entirely. Inherits the OS
        defaults and restores httpx's native env-proxy auto-detection.
    - A non-empty sequence of `(level, option, value)` tuples: explicit
        override; passed verbatim to the transport (not filtered). Unsupported
        options raise `OSError` at connect time rather than being silently
        dropped — the user chose them explicitly.

    环境变量（仅在本字段为 `None` 时查阅）: `LANGCHAIN_OPENAI_TCP_KEEPALIVE`（设为 `0`
    以完全禁用——即关闭开关），`LANGCHAIN_OPENAI_TCP_KEEPIDLE`，
    `LANGCHAIN_OPENAI_TCP_KEEPINTVL`，`LANGCHAIN_OPENAI_TCP_KEEPCNT`，
    `LANGCHAIN_OPENAI_TCP_USER_TIMEOUT_MS`。

    Applied per side: if `http_client` is supplied, the sync path uses
    that user-owned client's socket options as-is; the async path still
    gets `http_socket_options` applied to its default builder (and
    vice-versa for `http_async_client`). Supply both to take full control.

    !!! note "与环境代理自动检测的交互"

        当启用自定义 `httpx` 传输时，`httpx`
        会禁用其原生的环境代理自动检测（`HTTP_PROXY` / `HTTPS_PROXY` / `ALL_PROXY` /
        `NO_PROXY` 以及 macOS/Windows 系统代理设置）。

        为保持默认行为的安全性，`ChatOpenAI` 会检测 "代理环境阴影" 模式，并在
        **以下所有条件均满足** 时 **完全跳过自定义传输**:

        - `http_socket_options` 保持默认值（`None`）
        - 未提供 `http_client` 或 `http_async_client`
        - 未提供 `openai_proxy`
        - httpx 可见代理环境变量或系统代理

        On that specific shape, the instance falls back to pre-PR behavior
        and httpx's env-proxy auto-detection applies (a one-time `INFO` log
        records the bypass for observability).

        If you explicitly set `http_socket_options=[...]` while a proxy
        env var is also set, no bypass — you opted into the transport, and
        a one-time `WARNING` records the shadowing. Set
        `http_socket_options=()` or `LANGCHAIN_OPENAI_TCP_KEEPALIVE=0` to
        disable transport injection explicitly, or pass a fully-configured
        `http_async_client` / `http_client` to take full control. The
        `openai_proxy` constructor kwarg is unaffected — socket options
        are applied cleanly through the proxied transport on that path.
    """

    stream_chunk_timeout: float | None = Field(
        default_factory=lambda: _float_env(
            "LANGCHAIN_OPENAI_STREAM_CHUNK_TIMEOUT_S", 120.0
        ),
        exclude=True,
    )
    """每个数据块的墙钟超时时间（秒），适用于异步流式响应。

    Applies to async invocations only (`astream`, `ainvoke` with streaming,
    etc.). Sync streaming (`stream`) is not affected.

    在 openai SDK 流式迭代器生成的内容块之间触发（即每次对响应调用 `__anext__`
    时）。关键的是，这与 httpx 的 `timeout.read` **并不相同**:

    - httpx 的读取超时是字节间间隔的，每当 *任何* 字节到达套接字时就会重置——包括 OpenAI
    的 SSE 心跳注释（`: keepalive`），这些注释在长模型生成过程中会持续传来。一个在
    *内容* 上静默但仍在产生心跳的流对 httpx 来说看起来永远是活跃的。 -
    `stream_chunk_timeout` 衡量的是 *已解析块* 之间的间隔。openai SDK 的 SSE
    解析器会内部处理心跳注释，并不会将其作为块发出，因此心跳 *不会*
    重置此计时器。它会在真正的内容静默时触发。

    When it fires, a `StreamChunkTimeoutError`
    (subclass of `asyncio.TimeoutError`) is raised with a self-describing
    message naming this knob, the env-var override, the model, and the
    number of chunks received before the stall. A WARNING log with
    `extra={"source": "stream_chunk_timeout", "timeout_s": <value>,
    "model_name": <value>, "chunks_received": <value>}` also fires so
    aggregate logging can distinguish app-layer timeouts from
    transport-layer failures.

    Defaults to 120s. Set to `None` or `0` to disable. Overridable via the
    `LANGCHAIN_OPENAI_STREAM_CHUNK_TIMEOUT_S` env var. Negative values
    (from either the env var or the constructor kwarg — e.g., hydrated
    from YAML/JSON configs) fall back to the default with a `WARNING` log
    rather than silently disabling the wrapper, so a misconfigured value
    still boots safely and the fallback is visible.
    """

    stop: list[str] | str | None = Field(default=None, alias="stop_sequences")
    """默认停止序列。"""

    extra_body: Mapping[str, Any] | None = None
    """可选的额外 JSON 属性，用于在向 OpenAI 兼容 API（如 vLLM、LM Studio
或其他提供方）发送请求时包含在请求参数中。

    这是传递特定于您 OpenAI 兼容 API 提供方的自定义参数的推荐方式，但这些参数不属于标准
    OpenAI API。

    Examples:
    - [LM Studio](https://lmstudio.ai/) TTL 参数：`extra_body={"ttl": 300}`
    - [vLLM](https://github.com/vllm-project/vllm) 自定义参数：
        `extra_body={"use_beam_search": True}`
    - 任何其他提供方特定的参数

    !!! warning

        Do not use `model_kwargs` for custom parameters that are not part of the
        standard OpenAI API, as this will cause errors when making API calls. Use
        `extra_body` instead.
    """

    include_response_headers: bool = False
    """是否将响应头包含在输出消息的 `response_metadata` 中。

    Note: some inference providers return additional metadata (such as served model
    响应头中的名称）。启用以捕获这些元数据。
    """

    disabled_params: dict[str, Any] | None = Field(default=None)
    """Parameters of the OpenAI client or `chat.completions` endpoint that should be
    disabled for the given model.

    Should be specified as `{"param": None | ['val1', 'val2']}` where the key is the
    parameter and the value is either None, meaning that parameter should never be
    used, or it's a list of disabled values for the parameter.

    For example, older models may not support the `'parallel_tool_calls'` parameter at
    all, in which case `disabled_params={"parallel_tool_calls": None}` can be passed
    in.

    If a parameter is disabled then it will not be used by default in any methods, e.g.
    in `with_structured_output`. However this does not prevent a user from directly
    passed in the parameter during invocation.
    """

    context_management: list[dict[str, Any]] | None = None
    """配置项用于
    [上下文管理](https://developers.openai.com/api/docs/guides/compaction)。
    """

    include: list[str] | None = None
    """额外字段，包含在 Responses API 的生成结果中。

    支持的值：

    - `'file_search_call.results'`
    - `'message.input_image.image_url'`
    - `'computer_call_output.output.image_url'`
    - `'reasoning.encrypted_content'`
    - `'code_interpreter_call.outputs'`

    !!! version-added "在 `langchain-openai` 0.3.24 版本中新增"
    """

    prompt_cache_options: dict[str, Any] | None = None
    """控制 OpenAI 提示词缓存行为的选项。

    !!! version-added "在 `langchain-openai` 1.3.5 版本中新增"
    """

    service_tier: str | None = None
    """请求的延迟层级。

    可选值为 `'auto'`、`'default'` 或 `'flex'`。

    适用于 OpenAI 规模层级服务的用户。
    """

    store: bool | None = None
    """如果为 `True`，OpenAI 可能会存储响应数据以供未来使用。

    Responses API 默认为 `True`，Chat Completions API 默认为 `False`。

    !!! version-added "在 `langchain-openai` 0.3.24 版本中新增"
    """

    truncation: str | None = None
    """截断策略（Responses API）。

    可以是 `'auto'` 或 `'disabled'`（默认值）。

    If `'auto'`, model may drop input items from the middle of the message sequence to
    fit the context window.

    !!! version-added "在 `langchain-openai` 0.3.24 版本中新增"
    """

    use_previous_response_id: bool = False
    """如果为 `True`，则始终通过最近一次响应的 ID 传递 `previous_response_id`。仅适用于
Responses API。

    Input messages up to the most recent response will be dropped from request
    payloads.

    例如，以下两种方式等价：

    ```python
    model = ChatOpenAI(
        model="...",
        use_previous_response_id=True,
    )
    model.invoke(
        [
            HumanMessage("Hello"),
            AIMessage("Hi there!", response_metadata={"id": "resp_123"}),
            HumanMessage("How are you?"),
        ]
    )
    ```

    ```python
    model = ChatOpenAI(model="...", use_responses_api=True)
    model.invoke([HumanMessage("How are you?")], previous_response_id="resp_123")
    ```

    !!! version-added "在 `langchain-openai` 0.3.26 版本中新增"
    """

    use_responses_api: bool | None = None
    """是否使用 Responses API 而非 Chat API。

    如果未指定，则当实例设置要求使用 Responses API 时设为 `True`，

    !!! version-added "在 `langchain-openai` 0.3.9 版本中新增"
    """

    output_version: str | None = Field(
        default_factory=from_env("LC_OUTPUT_VERSION", default=None)
    )
    """要使用的 `AIMessage` 输出格式版本。

    This field is used to roll-out new output formats for chat model `AIMessage`
    responses in a backwards-compatible way.

    支持的值：

    - `'v0'`: `langchain-openai 0.3.x` 版本中的 `AIMessage` 格式。
    - `'responses/v1'`: 将 Responses API 输出项格式化为 AIMessage 内容块
        （仅适用于 Responses API）
    - `'v1'`: LangChain 跨提供商标准的 v1 版本。

    !!! warning "行为变更于 `langchain-openai` 1.0.0 版本中"

        默认值已更新为 `"responses/v1"`。
    """

    model_config = ConfigDict(populate_by_name=True)

    @property
    def _uses_gateway(self) -> bool:
        """请求是否通过 LangSmith 网关路由。

        Detected from the resolved API key: LangSmith keys (used to authenticate
        to the gateway) carry the `lsv2_` prefix. Callable keys cannot be
        inspected without invoking them, so they are treated as non-gateway.
        """
        api_key = self.openai_api_key
        if isinstance(api_key, SecretStr):
            return api_key.get_secret_value().startswith("lsv2_")
        return False

    @property
    def model(self) -> str:
        """与 model_name 相同。"""
        return self.model_name

    @model_validator(mode="before")
    @classmethod
    def build_extra(cls, values: dict[str, Any]) -> Any:
        """从传入的额外参数构建额外关键字参数。"""
        all_required_field_names = get_pydantic_field_names(cls)
        return _build_model_kwargs(values, all_required_field_names)

    @field_validator("stream_chunk_timeout", mode="after")
    @classmethod
    def _validate_stream_chunk_timeout(cls, value: float | None) -> float | None:
        """拒绝负数构造器值；回退到环境驱动的默认值。

        Matches the env-var path in `_float_env`: a negative value is a typo,
        not an opt-out (`None`/`0` are the documented off switches). Configs
        hydrated from YAML/JSON would otherwise silently disable the wrapper
        and reintroduce the indefinite-stream hang the feature prevents.
        """
        if value is not None and value < 0:
            fallback = _float_env("LANGCHAIN_OPENAI_STREAM_CHUNK_TIMEOUT_S", 120.0)
            logger.warning(
                "Invalid `stream_chunk_timeout=%r` (negative); "
                "falling back to %s. Pass `None` or `0` to disable.",
                value,
                fallback,
            )
            return fallback
        return value

    @model_validator(mode="before")
    @classmethod
    def validate_temperature(cls, values: dict[str, Any]) -> Any:
        """验证不同模型的 temperature 参数。

        - gpt-5 模型（不包括 gpt-5-chat）仅允许 `temperature=1` 或未设置
            （默认值为 1）
        """
        model = values.get("model_name") or values.get("model") or ""
        model_lower = model.lower()

                # 对于 gpt-5 模型，处理 temperature 限制。temperature 由 gpt-5-chat 和 gpt-5
        # 模型支持，且 reasoning_effort='none' 或
        if model_lower.startswith("o1") and "temperature" not in values:
            values["temperature"] = 1

                # reasoning={'effort': 'none'}。
                # 对于 gpt-5（非 chat），仅支持 temperature=1
                # 因此我们移除所有非默认值
        if (
            model_lower.startswith("gpt-5")
            and ("chat" not in model_lower)
            and values.get("reasoning_effort") != "none"
            and (values.get("reasoning") or {}).get("effort") != "none"
        ):
            temperature = values.get("temperature")
            if temperature is not None and temperature != 1:
                                # 检查 OPENAI_ORGANIZATION 以保持向后兼容性。
                                # 解析基础 URL 和 API 密钥，并应用 LangSmith 网关设置。
                values.pop("temperature", None)

        return values

    @model_validator(mode="after")
    def _set_openai_chat_version(self) -> Self:
        """在元数据中设置包版本。

        Note: Subclasses that inherit from `BaseChatOpenAI` (e.g.
        `ChatDeepSeek`、`ChatXAI` 等必须使用 **唯一** 的验证器名称 （例如
        `_set_deepseek_version`），而不是覆盖此名称。Pydantic 会替换同名的
        `model_validator` 方法，而非链式调用，因此复用 `_set_openai_chat_version`
        会无声丢弃父类的 `langchain-openai` 版本条目。
        """
        self._add_version("langchain-openai", __version__)
        return self

    @model_validator(mode="after")
    def _infer_use_responses_api(self) -> Self:
        """暴露无条件的实例级 Responses API 路由。"""
        if self.use_responses_api is None and self._use_responses_api({}):
            self.use_responses_api = True
        return self

    @model_validator(mode="after")
    def validate_environment(self) -> Self:
        """验证 API 密钥和 Python 包在环境中是否存在。"""
        if self.n is not None and self.n < 1:
            msg = "n must be at least 1."
            raise ValueError(msg)
        if self.n is not None and self.n > 1 and self.streaming:
            msg = "n must be 1 when streaming."
            raise ValueError(msg)

                # 如果使用默认基础 URL 和客户端，或基础 URL 由 LangSmith
        # 网关设置（其作为代理）时，默认启用 stream_usage
        self.openai_organization = (
            self.openai_organization
            or os.getenv("OPENAI_ORG_ID")
            or os.getenv("OPENAI_ORGANIZATION")
        )
                # 或当基础 URL 由 LangSmith 网关设置（其作为代理
        _gateway_config = _resolve_gateway_config(
            base_url=self.openai_api_base,
            api_key=self.openai_api_key,
            provider_path="openai/v1",
            base_url_env="OPENAI_API_BASE",
            api_key_env="OPENAI_API_KEY",
        )
        self.openai_api_base = _gateway_config.base_url
        self.openai_api_key = _gateway_config.api_key
        _base_url_from_gateway = _gateway_config.base_url_from_gateway

                # 如果使用默认的 base URL 和 client， 默认启用 stream_usage
                # 或当基础 URL 由 LangSmith 网关设置时（该网关充当代理
                # 支持 OpenAI 且支持流式 token 使用。
        if all(
            getattr(self, key, None) is None
            for key in (
                "stream_usage",
                "openai_proxy",
                "client",
                "root_client",
                "async_client",
                "root_async_client",
                "http_client",
                "http_async_client",
            )
        ) and (
            _base_url_from_gateway
            or (self.openai_api_base is None and "OPENAI_BASE_URL" not in os.environ)
        ):
            self.stream_usage = True

                # 从 SecretStr 或 Callable 解析 API 密钥
        sync_api_key_value: str | Callable[[], str] | None = None
        async_api_key_value: str | Callable[[], Awaitable[str]] | None = None

        if self.openai_api_key is not None:
                        # 因为 OpenAI 和 AsyncOpenAI 客户端支持同步或异步
                        # API 密钥的可调用对象，所以我们需要在此处分别解析值。
            sync_api_key_value, async_api_key_value = _resolve_sync_and_async_api_keys(
                self.openai_api_key
            )

        client_params: dict = {
            "organization": self.openai_organization,
            "base_url": self.openai_api_base,
            "timeout": self.request_timeout,
            "default_headers": self.default_headers,
            "default_query": self.default_query,
        }
        if self.max_retries is not None:
            client_params["max_retries"] = self.max_retries

        if self.openai_proxy and (self.http_client or self.http_async_client):
            openai_proxy = self.openai_proxy
            http_client = self.http_client
            http_async_client = self.http_async_client
            msg = (
                "Cannot specify 'openai_proxy' if one of "
                "'http_client'/'http_async_client' is already specified. Received:\n"
                f"{openai_proxy=}\n{http_client=}\n{http_async_client=}"
            )
            raise ValueError(msg)
        if _should_bypass_socket_options_for_proxy_env(
            http_socket_options=self.http_socket_options,
            http_client=self.http_client,
            http_async_client=self.http_async_client,
            openai_proxy=self.openai_proxy,
        ):
                        # 默认形状构造 + 对 httpx 可见的代理环境变量：
                        # 跳过自定义传输，以便 httpx 的环境代理自动检测
                        # 仍然生效。希望在使用环境代理的同时进行内核级 TCP 调优的用户
                        # 可以通过 `http_socket_options` 显式选择启用。
            resolved_socket_options: tuple[tuple[int, int, int], ...] = ()
            _log_proxy_env_bypass_once()
        else:
            resolved_socket_options = _resolve_socket_options(self.http_socket_options)
            _warn_if_proxy_env_shadowed(
                resolved_socket_options, openai_proxy=self.openai_proxy
            )
        if not self.client:
            if sync_api_key_value is None:
                                # 没有有效的同步 API 密钥，将客户端留空为 None 并在调用时抛出信息性
                                # 错误。
                self.client = None
                self.root_client = None
            else:
                if self.openai_proxy and not self.http_client:
                    self.http_client = _build_proxied_sync_httpx_client(
                        proxy=self.openai_proxy,
                        verify=global_ssl_context,
                        socket_options=resolved_socket_options,
                    )
                sync_specific = {
                    "http_client": self.http_client
                    or _get_default_httpx_client(
                        self.openai_api_base,
                        self.request_timeout,
                        resolved_socket_options,
                    ),
                    "api_key": sync_api_key_value,
                }
                self.root_client = openai.OpenAI(**client_params, **sync_specific)  # type: ignore[arg-type]
                self.client = self.root_client.chat.completions
        if not self.async_client:
            if self.openai_proxy and not self.http_async_client:
                self.http_async_client = _build_proxied_async_httpx_client(
                    proxy=self.openai_proxy,
                    verify=global_ssl_context,
                    socket_options=resolved_socket_options,
                )
            async_specific = {
                "http_client": self.http_async_client
                or _get_default_async_httpx_client(
                    self.openai_api_base,
                    self.request_timeout,
                    resolved_socket_options,
                ),
                "api_key": async_api_key_value,
            }
            self.root_async_client = openai.AsyncOpenAI(
                **client_params,
                **async_specific,  # type: ignore[arg-type]
            )
            self.async_client = self.root_async_client.chat.completions
        return self

    def _resolve_model_profile(self) -> ModelProfile | None:
        profile = _get_default_model_profile(self.model_name) or None
        if (
            profile is not None
            and profile.get("mid_conversation_tools")
            and not self._use_responses_api({})
        ):
                        # 只有 Responses API 携带 `additional_tools` 输入项。
                        # 依赖请求的路由无法提前声明。
            profile["mid_conversation_tools"] = False
        return profile

    @property
    def _default_params(self) -> dict[str, Any]:
        """获取调用 OpenAI API 的默认参数。"""
        exclude_if_none = {
            "presence_penalty": self.presence_penalty,
            "frequency_penalty": self.frequency_penalty,
            "seed": self.seed,
            "top_p": self.top_p,
            "logprobs": self.logprobs,
            "top_logprobs": self.top_logprobs,
            "logit_bias": self.logit_bias,
            "stop": self.stop or None,  # 同时排除空列表
            "max_tokens": self.max_tokens,
            "extra_body": self.extra_body,
            "n": self.n,
            "temperature": self.temperature,
            "reasoning_effort": self.reasoning_effort,
            "reasoning": self.reasoning,
            "verbosity": self.verbosity,
            "context_management": self.context_management,
            "include": self.include,
            "prompt_cache_options": self.prompt_cache_options,
            "service_tier": self.service_tier,
            "truncation": self.truncation,
            "store": self.store,
        }

        return {
            "model": self.model_name,
            "stream": self.streaming,
            **{k: v for k, v in exclude_if_none.items() if v is not None},
            **self.model_kwargs,
        }

    def _combine_llm_outputs(self, llm_outputs: list[dict | None]) -> dict:
        overall_token_usage: dict = {}
        system_fingerprint = None
        for output in llm_outputs:
            if output is None:
                                # 发生在流式场景中
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

    def _convert_chunk_to_generation_chunk(
        self,
        chunk: dict,
        default_chunk_class: type,
        base_generation_info: dict | None,
    ) -> ChatGenerationChunk | None:
        if chunk.get("type") == "content.delta":  # 来自 beta.chat.completions.stream
            return None
        token_usage = chunk.get("usage")
        choices = (
            chunk.get("choices", [])
                        # 来自 beta.chat.completions.stream
            or chunk.get("chunk", {}).get("choices", [])
        )

        usage_metadata: UsageMetadata | None = (
            _create_usage_metadata(token_usage, chunk.get("service_tier"))
            if token_usage
            else None
        )
        if len(choices) == 0:
                        # logprobs 默认为 None
            generation_chunk = ChatGenerationChunk(
                message=default_chunk_class(content="", usage_metadata=usage_metadata),
                generation_info=base_generation_info,
            )
                        # 保持 content 为空字符串（默认值），而不是转换为 []。
                        # 聊天补全内容的增量已规范化为字符串，在
                        # _convert_delta_to_message_chunk 中。从 [] 开始会导致
                        # merge_content
            # 静默丢弃字符串内容（空列表是假值，因此不会进入合并分支）。空列表还会触发
                        # content_blocks isinstance(list) 的短路逻辑，这将返回 [] 并遗漏
            # tool_call_chunks。
                        # 传播 output_version 以便 content_blocks 能检测 v1 模式。
                        # 优先级顺序
            if self.output_version == "v1":
                generation_chunk.message.response_metadata["output_version"] = "v1"

            return generation_chunk

        choice = choices[0]
        if choice["delta"] is None:
            return None

        message_chunk = _convert_delta_to_message_chunk(
            choice["delta"], default_chunk_class
        )
        generation_info = {**base_generation_info} if base_generation_info else {}

        if finish_reason := choice.get("finish_reason"):
            generation_info["finish_reason"] = finish_reason
            if model_name := chunk.get("model"):
                generation_info["model_name"] = model_name
            if system_fingerprint := chunk.get("system_fingerprint"):
                generation_info["system_fingerprint"] = system_fingerprint
            if service_tier := chunk.get("service_tier"):
                generation_info["service_tier"] = service_tier

        logprobs = choice.get("logprobs")
        if logprobs:
            generation_info["logprobs"] = logprobs

        if usage_metadata and isinstance(message_chunk, AIMessageChunk):
            message_chunk.usage_metadata = usage_metadata

        message_chunk.response_metadata["model_provider"] = "openai"
                # 网关元数据应位于 `generation_info`，而非由 `metadata` 填充的 message
        if self.output_version == "v1":
            message_chunk.response_metadata["output_version"] = "v1"
        return ChatGenerationChunk(
            message=message_chunk, generation_info=generation_info or None
        )

    def _ensure_sync_client_available(self) -> None:
        """检查同步客户端是否可用，若不可用则抛出错误。"""
        if self.client is None:
            msg = (
                "Sync client is not available. This happens when an async callable "
                "was provided for the API key. Use async methods (ainvoke, astream) "
                "instead, or provide a string or sync callable for the API key."
            )
            raise ValueError(msg)

    def _stream_responses(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        self._ensure_sync_client_available()
        kwargs["stream"] = True
        payload = self._get_request_payload(messages, stop=stop, **kwargs)
        headers: dict = {}
        base_generation_info: dict = {}
        try:
            if self.include_response_headers or self._uses_gateway:
                raw_context_manager = (
                    self.root_client.with_raw_response.responses.create(**payload)
                )
                context_manager = raw_context_manager.parse()
                if self.include_response_headers:
                    headers = {"headers": dict(raw_context_manager.headers)}
                _add_gateway_metadata(base_generation_info, raw_context_manager)
            else:
                context_manager = self.root_client.responses.create(**payload)
            original_schema_obj = kwargs.get("response_format")

            with context_manager as response:
                is_first_chunk = True
                current_index = -1
                current_output_index = -1
                current_sub_index = -1
                has_reasoning = False
                item_output_indices: dict[str, int] = {}
                for chunk in response:
                    metadata = headers if is_first_chunk else {}
                    (
                        current_index,
                        current_output_index,
                        current_sub_index,
                        generation_chunk,
                    ) = _convert_responses_chunk_to_generation_chunk(
                        chunk,
                        current_index,
                        current_output_index,
                        current_sub_index,
                        schema=original_schema_obj,
                        metadata=metadata,
                        has_reasoning=has_reasoning,
                        output_version=self.output_version,
                        item_output_indices=item_output_indices,
                    )
                    if generation_chunk:
                        if is_first_chunk and base_generation_info:
                            generation_chunk.generation_info = {
                                **base_generation_info,
                                **(generation_chunk.generation_info or {}),
                            }
                        if run_manager:
                            run_manager.on_llm_new_token(
                                generation_chunk.text, chunk=generation_chunk
                            )
                        is_first_chunk = False
                        if "reasoning" in generation_chunk.message.additional_kwargs:
                            has_reasoning = True
                        yield generation_chunk
        except openai.BadRequestError as e:
            _handle_openai_bad_request(e)
        except openai.APIError as e:
            _handle_openai_api_error(e)

    async def _astream_responses(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        kwargs["stream"] = True
        payload = self._get_request_payload(messages, stop=stop, **kwargs)
        headers: dict = {}
        base_generation_info: dict = {}
        try:
            if self.include_response_headers or self._uses_gateway:
                raw_context_manager = (
                    await self.root_async_client.with_raw_response.responses.create(
                        **payload
                    )
                )
                context_manager = raw_context_manager.parse()
                if self.include_response_headers:
                    headers = {"headers": dict(raw_context_manager.headers)}
                _add_gateway_metadata(base_generation_info, raw_context_manager)
            else:
                context_manager = await self.root_async_client.responses.create(
                    **payload
                )
            original_schema_obj = kwargs.get("response_format")

            async with context_manager as response:
                is_first_chunk = True
                current_index = -1
                current_output_index = -1
                current_sub_index = -1
                has_reasoning = False
                item_output_indices: dict[str, int] = {}
                async for chunk in _astream_with_chunk_timeout(
                    response,
                    self.stream_chunk_timeout,
                    model_name=self.model_name,
                ):
                    metadata = headers if is_first_chunk else {}
                    (
                        current_index,
                        current_output_index,
                        current_sub_index,
                        generation_chunk,
                    ) = _convert_responses_chunk_to_generation_chunk(
                        chunk,
                        current_index,
                        current_output_index,
                        current_sub_index,
                        schema=original_schema_obj,
                        metadata=metadata,
                        has_reasoning=has_reasoning,
                        output_version=self.output_version,
                        item_output_indices=item_output_indices,
                    )
                    if generation_chunk:
                        if is_first_chunk and base_generation_info:
                            generation_chunk.generation_info = {
                                **base_generation_info,
                                **(generation_chunk.generation_info or {}),
                            }
                        if run_manager:
                            await run_manager.on_llm_new_token(
                                generation_chunk.text, chunk=generation_chunk
                            )
                        is_first_chunk = False
                        if "reasoning" in generation_chunk.message.additional_kwargs:
                            has_reasoning = True
                        yield generation_chunk
        except openai.BadRequestError as e:
            _handle_openai_bad_request(e)
        except openai.APIError as e:
            _handle_openai_api_error(e)

    def _should_stream_usage(
        self, stream_usage: bool | None = None, **kwargs: Any
    ) -> bool:
        """确定是否在流式输出中包含使用量元数据。

        For backwards compatibility, we check for `stream_options` passed
        explicitly to kwargs or in the `model_kwargs` and override `self.stream_usage`.
        """
        stream_usage_sources = [  # `response_metadata`。
            stream_usage,
            kwargs.get("stream_options", {}).get("include_usage"),
            self.model_kwargs.get("stream_options", {}).get("include_usage"),
            self.stream_usage,
        ]
        for source in stream_usage_sources:
            if isinstance(source, bool):
                return source
        return self.stream_usage or False

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        *,
        stream_usage: bool | None = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        self._ensure_sync_client_available()
        kwargs["stream"] = True
        stream_usage = self._should_stream_usage(stream_usage, **kwargs)
        if stream_usage:
            kwargs["stream_options"] = {"include_usage": stream_usage}
        payload = self._get_request_payload(messages, stop=stop, **kwargs)
        default_chunk_class: type[BaseMessageChunk] = AIMessageChunk
        base_generation_info = {}

        try:
            if "response_format" in payload:
                if self.include_response_headers:
                    warnings.warn(
                        "Cannot currently include response headers when "
                        "response_format is specified."
                    )
                payload.pop("stream")
                response_stream = self.root_client.beta.chat.completions.stream(
                    **payload
                )
                context_manager = response_stream
            else:
                if self.include_response_headers or self._uses_gateway:
                    raw_response = self.client.with_raw_response.create(**payload)
                    response = raw_response.parse()
                    if self.include_response_headers:
                        base_generation_info = {"headers": dict(raw_response.headers)}
                    _add_gateway_metadata(base_generation_info, raw_response)
                else:
                    response = self.client.create(**payload)
                context_manager = response
            with context_manager as response:
                is_first_chunk = True
                for chunk in response:
                    if not isinstance(chunk, dict):
                        chunk = chunk.model_dump()
                    generation_chunk = self._convert_chunk_to_generation_chunk(
                        chunk,
                        default_chunk_class,
                        base_generation_info if is_first_chunk else {},
                    )
                    if generation_chunk is None:
                        continue
                    default_chunk_class = generation_chunk.message.__class__
                    logprobs = (generation_chunk.generation_info or {}).get("logprobs")
                    if run_manager:
                        run_manager.on_llm_new_token(
                            generation_chunk.text,
                            chunk=generation_chunk,
                            logprobs=logprobs,
                        )
                    is_first_chunk = False
                    yield generation_chunk
        except openai.BadRequestError as e:
            _handle_openai_bad_request(e)
        except openai.APIError as e:
            _handle_openai_api_error(e)
        if hasattr(response, "get_final_completion") and "response_format" in payload:
            final_completion = response.get_final_completion()
            generation_chunk = self._get_generation_chunk_from_completion(
                final_completion
            )
            if run_manager:
                run_manager.on_llm_new_token(
                    generation_chunk.text, chunk=generation_chunk
                )
            yield generation_chunk

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        self._ensure_sync_client_available()
        payload = self._get_request_payload(messages, stop=stop, **kwargs)
        generation_info = None
        raw_response = None
        try:
            if "response_format" in payload:
                payload.pop("stream")
                raw_response = (
                    self.root_client.chat.completions.with_raw_response.parse(**payload)
                )
                response = raw_response.parse()
            elif self._use_responses_api(payload):
                original_schema_obj = kwargs.get("response_format")
                if original_schema_obj and _is_pydantic_class(original_schema_obj):
                    raw_response = self.root_client.responses.with_raw_response.parse(
                        **payload
                    )
                else:
                    raw_response = self.root_client.responses.with_raw_response.create(
                        **payload
                    )
                response = raw_response.parse()
                if self.include_response_headers:
                    generation_info = {"headers": dict(raw_response.headers)}
                generation_info = generation_info or {}
                _add_gateway_metadata(generation_info, raw_response)
                                # `parse()` 在端点返回非 JSON 正文时生成 `str`，
                                # 例如重定向后返回的 HTML 错误页面。
                gateway_metadata = generation_info.pop(
                    GATEWAY_METADATA_RESPONSE_KEY, None
                )
                result = _construct_lc_result_from_responses_api(
                    response,
                    schema=original_schema_obj,
                    metadata=generation_info,
                    output_version=self.output_version,
                )
                if gateway_metadata is not None:
                    for generation in result.generations:
                        generation.generation_info = generation.generation_info or {}
                        generation.generation_info[GATEWAY_METADATA_RESPONSE_KEY] = (
                            gateway_metadata
                        )
                return result
            else:
                raw_response = self.client.with_raw_response.create(**payload)
                response = raw_response.parse()
        except openai.BadRequestError as e:
            _handle_openai_bad_request(e)
        except openai.APIError as e:
            _handle_openai_api_error(e)
        except Exception as e:
            if raw_response is not None and hasattr(raw_response, "http_response"):
                e.response = raw_response.http_response  # type: ignore[attr-defined]
            raise e
        if (
            self.include_response_headers
            and raw_response is not None
            and hasattr(raw_response, "headers")
        ):
            generation_info = {"headers": dict(raw_response.headers)}
        generation_info = generation_info or {}
        _add_gateway_metadata(generation_info, raw_response)
        return self._create_chat_result(response, generation_info)

    def _use_responses_api(self, payload: dict) -> bool:
        if isinstance(self.use_responses_api, bool):
            return self.use_responses_api
        if (
            self.output_version == "responses/v1"
            or self.context_management is not None
            or self.include is not None
            or self.reasoning is not None
            or self.truncation is not None
            or self.use_previous_response_id
            or _model_prefers_responses_api(self.model_name)
            or (
                (self.model_name or "").lower().startswith("gpt-6")
                and payload.get("tools")
            )
        ):
            return True
        return _use_responses_api(payload)

    def _get_request_payload(
        self,
        input_: LanguageModelInput,
        *,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> dict:
        messages = self._convert_input(input_).to_messages()
        if stop is not None:
            kwargs["stop"] = stop

        payload = {**self._default_params, **kwargs}

        if self._use_responses_api(payload):
            if self.use_previous_response_id:
                last_messages, previous_response_id = _get_last_messages(messages)
                payload_to_use = last_messages if previous_response_id else messages
                if previous_response_id:
                    payload["previous_response_id"] = previous_response_id
                payload = _construct_responses_api_payload(payload_to_use, payload)
            else:
                payload = _construct_responses_api_payload(messages, payload)
        else:
            payload["messages"] = [
                _convert_message_to_dict(_convert_from_v1_to_chat_completions(m))
                if isinstance(m, AIMessage)
                else _convert_message_to_dict(m)
                for m in messages
            ]
        return payload

    def _create_chat_result(
        self,
        response: dict | openai.BaseModel,
        generation_info: dict | None = None,
    ) -> ChatResult:
        generations = []

        if not isinstance(response, dict | openai.BaseModel):
                        # `parsed` 可能包含来自结构化输出的任意 Pydantic 模型。
                        # 从本次转储中排除它，并从下方的类型化响应中复制。
            preview = repr(response)
            if len(preview) > 200:
                preview = f"{preview[:200]}..."
            msg = (
                "Unexpected response type from OpenAI-compatible endpoint. "
                "Expected a dict or openai.BaseModel, got "
                f"{type(response).__name__}: {preview}"
            )
            raise ValueError(msg)

        response_dict = (
            response
            if isinstance(response, dict)
                        # 有时 AI 模型调用会出错，我们应该抛出该错误（这通常紧随 `choices` 为 null
            # 值，我们将在下面单独处理）。
                        # 对返回格式错误响应的非 OpenAI 聊天补全 API 抛出信息性错误消息
            else response.model_dump(
                exclude={"choices": {"__all__": {"message": {"parsed"}}}},
                warnings=False,
            )
        )
                # 某些 OpenAI 兼容 API（如
        # vLLM）可能在响应格式不匹配或发生错误但未填充错误字段时返回 null
        # choices。提供更友好的错误提示。
                # 网关元数据应位于 `generation_info`，而非由 `metadata` 填充的 message
                # `response_metadata`。
        if response_dict.get("error"):
            raise ValueError(response_dict.get("error"))

                # 从内置远程 MCP 工具调用中清除凭证
                # tiktoken 不支持 Python 3.7 或更低版本
        try:
            choices = response_dict["choices"]
        except KeyError as e:
            msg = f"Response missing 'choices' key: {response_dict.keys()}"
            raise KeyError(msg) from e

        if choices is None:
                        # TODO: 将绑定工具计入输入。
                        # 每条消息遵循 <im_start>{role/name}\n{content}<im_end>\n
                        # 如果有 name，则省略 role
            msg = (
                "Received response with null value for 'choices'. "
                "This can happen when using OpenAI-compatible APIs (e.g., vLLM) "
                "that return a response in an unexpected format. "
                f"Full response keys: {list(response_dict.keys())}"
            )
            raise TypeError(msg)

        token_usage = response_dict.get("usage")
        service_tier = response_dict.get("service_tier")

        for res in choices:
            message = _convert_dict_to_message(res["message"])
            if token_usage and isinstance(message, AIMessage):
                message.usage_metadata = _create_usage_metadata(
                    token_usage, service_tier
                )
            generation_info = generation_info or {}
            generation_info["finish_reason"] = (
                res.get("finish_reason")
                if res.get("finish_reason") is not None
                else generation_info.get("finish_reason")
            )
            if "logprobs" in res:
                generation_info["logprobs"] = res["logprobs"]
            gen = ChatGeneration(message=message, generation_info=generation_info)
            generations.append(gen)
        llm_output = {
            "token_usage": token_usage,
            "model_provider": "openai",
            "model_name": response_dict.get("model", self.model_name),
            "system_fingerprint": response_dict.get("system_fingerprint", ""),
        }
        if "id" in response_dict:
            llm_output["id"] = response_dict["id"]
        if service_tier:
            llm_output["service_tier"] = service_tier

        if isinstance(response, openai.BaseModel) and getattr(
            response, "choices", None
        ):
            message = response.choices[0].message  # type: ignore[attr-defined]
            if hasattr(message, "parsed"):
                generations[0].message.additional_kwargs["parsed"] = message.parsed
            if hasattr(message, "refusal"):
                generations[0].message.additional_kwargs["refusal"] = message.refusal

        return ChatResult(generations=generations, llm_output=llm_output)

    async def _astream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        *,
        stream_usage: bool | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        kwargs["stream"] = True
        stream_usage = self._should_stream_usage(stream_usage, **kwargs)
        if stream_usage:
            kwargs["stream_options"] = {"include_usage": stream_usage}
        payload = self._get_request_payload(messages, stop=stop, **kwargs)
        default_chunk_class: type[BaseMessageChunk] = AIMessageChunk
        base_generation_info = {}

        try:
            if "response_format" in payload:
                if self.include_response_headers:
                    warnings.warn(
                        "Cannot currently include response headers when "
                        "response_format is specified."
                    )
                payload.pop("stream")
                response_stream = self.root_async_client.beta.chat.completions.stream(
                    **payload
                )
                context_manager = response_stream
            else:
                if self.include_response_headers or self._uses_gateway:
                    raw_response = await self.async_client.with_raw_response.create(
                        **payload
                    )
                    response = raw_response.parse()
                    if self.include_response_headers:
                        base_generation_info = {"headers": dict(raw_response.headers)}
                    _add_gateway_metadata(base_generation_info, raw_response)
                else:
                    response = await self.async_client.create(**payload)
                context_manager = response
            async with context_manager as response:
                is_first_chunk = True
                async for chunk in _astream_with_chunk_timeout(
                    response,
                    self.stream_chunk_timeout,
                    model_name=self.model_name,
                ):
                    if not isinstance(chunk, dict):
                        chunk = chunk.model_dump()
                    generation_chunk = self._convert_chunk_to_generation_chunk(
                        chunk,
                        default_chunk_class,
                        base_generation_info if is_first_chunk else {},
                    )
                    if generation_chunk is None:
                        continue
                    default_chunk_class = generation_chunk.message.__class__
                    logprobs = (generation_chunk.generation_info or {}).get("logprobs")
                    if run_manager:
                        await run_manager.on_llm_new_token(
                            generation_chunk.text,
                            chunk=generation_chunk,
                            logprobs=logprobs,
                        )
                    is_first_chunk = False
                    yield generation_chunk
        except openai.BadRequestError as e:
            _handle_openai_bad_request(e)
        except openai.APIError as e:
            _handle_openai_api_error(e)
        if hasattr(response, "get_final_completion") and "response_format" in payload:
            final_completion = await response.get_final_completion()
            generation_chunk = self._get_generation_chunk_from_completion(
                final_completion
            )
            if run_manager:
                await run_manager.on_llm_new_token(
                    generation_chunk.text, chunk=generation_chunk
                )
            yield generation_chunk

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        payload = self._get_request_payload(messages, stop=stop, **kwargs)
        generation_info = {}
        raw_response = None
        try:
            if "response_format" in payload:
                payload.pop("stream")
                raw_response = await self.root_async_client.chat.completions.with_raw_response.parse(  # noqa: E501
                    **payload
                )
                response = raw_response.parse()
            elif self._use_responses_api(payload):
                original_schema_obj = kwargs.get("response_format")
                if original_schema_obj and _is_pydantic_class(original_schema_obj):
                    raw_response = (
                        await self.root_async_client.responses.with_raw_response.parse(
                            **payload
                        )
                    )
                else:
                    raw_response = (
                        await self.root_async_client.responses.with_raw_response.create(
                            **payload
                        )
                    )
                response = raw_response.parse()
                if self.include_response_headers:
                    generation_info = {"headers": dict(raw_response.headers)}
                _add_gateway_metadata(generation_info, raw_response)
                                # 这是推断的近似值。OpenAI 未文档化如何计算工具消息的 token 数量。
                                # content 或 tool calls
                gateway_metadata = generation_info.pop(
                    GATEWAY_METADATA_RESPONSE_KEY, None
                )
                result = _construct_lc_result_from_responses_api(
                    response,
                    schema=original_schema_obj,
                    metadata=generation_info,
                    output_version=self.output_version,
                )
                if gateway_metadata is not None:
                    for generation in result.generations:
                        generation.generation_info = generation.generation_info or {}
                        generation.generation_info[GATEWAY_METADATA_RESPONSE_KEY] = (
                            gateway_metadata
                        )
                return result
            else:
                raw_response = await self.async_client.with_raw_response.create(
                    **payload
                )
                response = raw_response.parse()
        except openai.BadRequestError as e:
            _handle_openai_bad_request(e)
        except openai.APIError as e:
            _handle_openai_api_error(e)
        except Exception as e:
            if raw_response is not None and hasattr(raw_response, "http_response"):
                e.response = raw_response.http_response  # type: ignore[attr-defined]
            raise e
        if (
            self.include_response_headers
            and raw_response is not None
            and hasattr(raw_response, "headers")
        ):
            generation_info = {"headers": dict(raw_response.headers)}
        _add_gateway_metadata(generation_info, raw_response)
        return await run_in_executor(
            None, self._create_chat_result, response, generation_info
        )

    @property
    def _identifying_params(self) -> dict[str, Any]:
        """获取标识参数。"""
        return {"model_name": self.model_name, **self._default_params}

    def _get_invocation_params(
        self, stop: list[str] | None = None, **kwargs: Any
    ) -> dict[str, Any]:
        """获取调用模型时使用的参数。"""
        params = {
            "model": self.model_name,
            **super()._get_invocation_params(stop=stop),
            **self._default_params,
            **kwargs,
        }
                # 工具/函数调用的 token 计算未被 OpenAI 文档化。
        if (tools := params.get("tools")) and isinstance(tools, list):
            sensitive_fields = ("headers", "authorization")
            params["tools"] = [
                {
                    **tool,
                    **{
                        field: "**REDACTED**"
                        for field in sensitive_fields
                        if field in tool
                    },
                }
                if isinstance(tool, dict) and tool.get("type") == "mcp"
                else tool
                for tool in tools
            ]

        return params

    def _get_ls_params(
        self, stop: list[str] | None = None, **kwargs: Any
    ) -> LangSmithParams:
        """获取用于追踪的标准参数。"""
        params = self._get_invocation_params(stop=stop, **kwargs)
        ls_params = LangSmithParams(
            ls_provider="openai",
            ls_model_name=params.get("model", self.model_name),
            ls_model_type="chat",
            ls_temperature=params.get("temperature", self.temperature),
        )
        if ls_max_tokens := params.get("max_tokens", self.max_tokens) or params.get(
            "max_completion_tokens", self.max_tokens
        ):
            ls_params["ls_max_tokens"] = ls_max_tokens
        if ls_stop := stop or params.get("stop", None):
            ls_params["ls_stop"] = ls_stop
        return ls_params

    @property
    def _llm_type(self) -> str:
        """返回聊天模型的返回类型。

        无论具体模型名称如何，始终返回 `'openai-chat'`。
        """
        return "openai-chat"

    def _get_encoding_model(self) -> tuple[str, tiktoken.Encoding]:
        if self.tiktoken_model_name is not None:
            model = self.tiktoken_model_name
        else:
            model = self.model_name

        try:
            encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            model_lower = model.lower()
            encoder = "cl100k_base"
            if model_lower.startswith(("gpt-4o", "gpt-4.1", "gpt-5", "gpt-6")):
                encoder = "o200k_base"
            encoding = tiktoken.get_encoding(encoder)
        return model, encoding

    def get_token_ids(self, text: str) -> list[int]:
        """使用 tiktoken 包获取文本中的 token。"""
        if self.custom_get_token_ids is not None:
            return self.custom_get_token_ids(text)
                # 这是近似值。
        if sys.version_info[1] <= 7:
            return super().get_token_ids(text)
        _, encoding_model = self._get_encoding_model()
        return encoding_model.encode(text)

    def get_num_tokens_from_messages(
        self,
        messages: Sequence[BaseMessage],
        tools: Sequence[dict[str, Any] | type | Callable | BaseTool] | None = None,
        *,
        allow_fetching_images: bool = True,
    ) -> int:
        """计算支持的 OpenAI 聊天模型的 token 数量。

        !!! warning
            You must have the `pillow` installed if you want to count image tokens if
            you are specifying the image as a base64 string, and you must have both
            `pillow` and `httpx` installed if you are specifying the image as a URL. If
            these aren't installed image inputs will be ignored in token counting.

        [OpenAI 参考](https://github.com/openai/openai-cookbook/blob/main/examples/How_to_format_inputs_to_ChatGPT_models.ipynb)。

        Args:
            messages: The message inputs to tokenize.
            tools: If provided, sequence of `dict`, `BaseModel`, function, or `BaseTool`
                to be converted to tool schemas.
            allow_fetching_images: Whether to allow fetching images for token counting.
        """
                # 将 str(value) 强制转换，以防消息值不是字符串
        if tools is not None:
            warnings.warn(
                "Counting tokens in tool schemas is not yet supported. Ignoring tools."
            )
        if sys.version_info[1] <= 7:
            return super().get_num_tokens_from_messages(messages)
        model, encoding = self._get_encoding_model()
        if model.startswith("gpt-3.5-turbo-0301"):
                        # 这在函数消息中会发生
            tokens_per_message = 4
                        # 每条回复都以 <im_start>assistant 开头
            tokens_per_name = -1
        elif model.startswith(
            ("gpt-3.5-turbo", "gpt-4", "gpt-5", "gpt-6", "o1", "o3", "o4")
        ):
            tokens_per_message = 3
            tokens_per_name = 1
        else:
            msg = (
                f"get_num_tokens_from_messages() is not presently implemented "
                f"for model {model}. See "
                "https://platform.openai.com/docs/guides/text-generation/managing-tokens"
                " for information on how messages are converted to tokens."
            )
            raise NotImplementedError(msg)
        num_tokens = 0
        messages_dict = [_convert_message_to_dict(m) for m in messages]
        for message in messages_dict:
            num_tokens += tokens_per_message
            for key, value in message.items():
                                # 当通过聊天补全 API 提供 response_format 时，OpenAI
                                # 要求所有函数工具必须严格。默认 strict=True，除非
                if key == "tool_call_id":
                    num_tokens += 3
                    continue
                if isinstance(value, list):
                                        # 内容或工具调用
                    for val in value:
                        if isinstance(val, str) or val["type"] == "text":
                            text = val["text"] if isinstance(val, dict) else val
                            num_tokens += len(encoding.encode(text))
                        elif val["type"] == "image_url":
                            if val["image_url"].get("detail") == "low":
                                num_tokens += 85
                            elif allow_fetching_images:
                                image_size = _url_to_size(val["image_url"]["url"])
                                if not image_size:
                                    continue
                                num_tokens += _count_image_tokens(*image_size)
                            else:
                                pass
                                                # 工具/函数调用的 token 计数未在 OpenAI 的文档中说明。
                                                # 这是一个近似值。
                        elif val["type"] == "function":
                            num_tokens += len(
                                encoding.encode(val["function"]["arguments"])
                            )
                            num_tokens += len(encoding.encode(val["function"]["name"]))
                        elif val["type"] == "file":
                            warnings.warn(
                                "Token counts for file inputs are not supported. "
                                "Ignoring file inputs."
                            )
                        else:
                            msg = f"Unrecognized content block type\n\n{val}"
                            raise ValueError(msg)
                elif not value:
                    continue
                else:
                                        # 在 message 值不是字符串的情况下，将其转换为 str(value)
                                        # 这会在函数消息中发生
                    num_tokens += len(encoding.encode(str(value)))
                if key == "name":
                    num_tokens += tokens_per_name
                # 每次回复都以 <im_start>assistant 开头
        num_tokens += 3
        return num_tokens

    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type | Callable | BaseTool],
        *,
        tool_choice: dict | str | bool | None = None,
        strict: bool | None = None,
        parallel_tool_calls: bool | None = None,
        response_format: _DictOrPydanticClass | None = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, AIMessage]:
        """将工具类对象绑定到此聊天模型。

        假设模型与 OpenAI 工具调用 API 兼容。

        Args:
            tools: 要绑定到此聊天模型的工具定义列表。

                支持由 [`convert_to_openai_tool`][langchain_core.utils.function_calling.convert_to_openai_tool] 处理的任何工具定义。
            tool_choice: 要求模型调用的工具。选项包括：

                - `str` of the form `'<<tool_name>>'`: calls `<<tool_name>>` tool.
                - `'auto'`: automatically selects a tool (including no tool).
                - `'none'`: does not call a tool.
                - `'any'` or `'required'` or `True`: force at least one tool to be called.
                - `dict` of the form `{"type": "function", "function": {"name": <<tool_name>>}}`: calls `<<tool_name>>` tool.
                - `False` or `None`: no effect, default OpenAI behavior.
            strict: If `True`, model output is guaranteed to exactly match the JSON Schema
                provided in the tool definition. The input schema will also be validated according to the
                [supported schemas](https://platform.openai.com/docs/guides/structured-outputs/supported-schemas?api-mode=responses#supported-schemas).
                If `False`, input schema will not be validated and model output will not
                be validated. If `None`, `strict` argument will not be passed to the model.
            parallel_tool_calls: Set to `False` to disable parallel tool use.
                Defaults to `None` (no specification, which allows parallel tool use).
            response_format: Optional schema to format model response. If provided
                and the model does not call a tool, the model will generate a
                [structured response](https://platform.openai.com/docs/guides/structured-outputs).
            kwargs: Any additional parameters are passed directly to `bind`.
        """  # noqa: E501
        if parallel_tool_calls is not None:
            kwargs["parallel_tool_calls"] = parallel_tool_calls
                # 当通过聊天补全 API 提供 response_format 时，OpenAI
                # 要求所有函数工具都严格。默认 strict=True，除非
                # 调用方显式传递了 strict=False。Responses API 不需要此参数。
                # tool_choice 是工具/函数名称
        if response_format and strict is not False and not self.use_responses_api:
            strict = True
        formatted_tools = [
            convert_to_openai_tool(tool, strict=strict) for tool in tools
        ]
        for original, formatted in zip(tools, formatted_tools, strict=False):
            if (
                isinstance(original, BaseTool)
                and hasattr(original, "extras")
                and isinstance(original.extras, dict)
            ):
                for key in _TOOL_EXTRAS_PASSTHROUGH:
                    if key in original.extras:
                        formatted[key] = original.extras[key]
        tool_names = []
        for tool in formatted_tools:
            if "function" in tool:
                tool_names.append(tool["function"]["name"])
            elif "name" in tool:
                tool_names.append(tool["name"])
            else:
                pass
        if tool_choice:
            if isinstance(tool_choice, str):
                                # 'any' 在 OpenAI API 中不被原生支持。
                if tool_choice in tool_names:
                    tool_choice = {
                        "type": "function",
                        "function": {"name": tool_choice},
                    }
                elif tool_choice in WellKnownTools:
                    tool_choice = {"type": tool_choice}
                                # 我们支持 'any'，因为其他模型使用此选项代替 'required'。
                                # 与 langchain.agents.create_agent response_format 兼容，这是 OpenAI
                # 格式的近似实现
                elif tool_choice == "any":
                    tool_choice = "required"
                else:
                    pass
            elif isinstance(tool_choice, bool):
                tool_choice = "required"
            elif isinstance(tool_choice, dict):
                pass
            else:
                msg = (
                    f"Unrecognized tool_choice type. Expected str, bool or dict. "
                    f"Received: {tool_choice}"
                )
                raise ValueError(msg)
            kwargs["tool_choice"] = tool_choice

        if response_format:
            if (
                isinstance(response_format, dict)
                and response_format.get("type") == "json_schema"
                and "schema" in response_format.get("json_schema", {})
            ):
                                # 检查 Pydantic BaseModel V1
                                # 检查不兼容的模型
                strict = response_format["json_schema"].get("strict", None)
                response_format = cast(dict, response_format["json_schema"]["schema"])
            kwargs["response_format"] = _convert_to_openai_response_format(
                response_format, strict=strict
            )
        return super().bind(tools=formatted_tools, **kwargs)

    def with_structured_output(
        self,
        schema: _DictOrPydanticClass | None = None,
        *,
        method: Literal[
            "function_calling", "json_mode", "json_schema"
        ] = "function_calling",
        include_raw: bool = False,
        strict: bool | None = None,
        tools: list | None = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, _DictOrPydantic]:
        """将输出格式化为匹配给定模式的模型包装器。

        Args:
            schema: 输出模式。可作为以下形式传入：

                - OpenAI 函数/工具 schema，
                - JSON Schema，
                - `TypedDict` 类，
                - 或 Pydantic 类。

                If `schema` is a Pydantic class then the model output will be a
                Pydantic instance of that class, and the model-generated fields will be
                validated by the Pydantic class. Otherwise the model output will be a
                dict and will not be validated.

                See `langchain_core.utils.function_calling.convert_to_openai_tool` for
                more on how to properly specify types and descriptions of schema fields
                when specifying a Pydantic or `TypedDict` class.

            method: 控制模型生成的方法，可选之一：

                - `'function_calling'`:
                    Uses OpenAI's [tool-calling API](https://platform.openai.com/docs/guides/function-calling)
                    (formerly called function calling)
                - `'json_schema'`:
                    Uses OpenAI's [Structured Output API](https://platform.openai.com/docs/guides/structured-outputs)
                - `'json_mode'`:
                    Uses OpenAI's [JSON mode](https://platform.openai.com/docs/guides/structured-outputs/json-mode).
                    Note that if using JSON mode then you must include instructions for
                    formatting the output into the desired schema into the model call

            include_raw:
                如果为 `False`，则仅返回解析后的结构化输出。

                如果在模型输出解析过程中发生错误，将抛出异常。

                如果为 `True`，则同时返回原始模型响应（一个 `BaseMessage`）和
                解析后的模型响应。

                如果在输出解析过程中发生错误，该错误将被捕获并一同返回。

                The final output is always a `dict` with keys `'raw'`, `'parsed'`, and
                `'parsing_error'`.
            strict:

                - `True`:
                    Model output is guaranteed to exactly match the schema.
                    The input schema will also be validated according to the
                    [supported schemas](https://platform.openai.com/docs/guides/structured-outputs/supported-schemas?api-mode=responses#supported-schemas).
                - `False`:
                    Input schema will not be validated and model output will not be
                    validated.
                - `None`:
                    `strict` argument will not be passed to the model.

            tools:
                要绑定到聊天模型的一组工具类对象。要求满足：

                - `method` 为 `'json_schema'`（默认值）。
                - `strict=True`
                - `include_raw=True`

                如果模型选择调用某个工具，则 `'raw'` 中的 `AIMessage`
                将包含工具调用信息。

                ??? example

                    ```python
                    from langchain.chat_models import init_chat_model
                    from pydantic import BaseModel


                    class ResponseSchema(BaseModel):
                        response: str


                    def get_weather(location: str) -> str:
                        \"\"\"Get weather at a location.\"\"\"
                        pass

                    model = init_chat_model("openai:gpt-5.5")

                    structured_model = model.with_structured_output(
                        ResponseSchema,
                        tools=[get_weather],
                        strict=True,
                        include_raw=True,
                    )

                    structured_model.invoke("What's the weather in Boston?")
                    ```

                    ```python
                    {
                        "raw": AIMessage(content="", tool_calls=[...], ...),
                        "parsing_error": None,
                        "parsed": None,
                    }
                    ```

            kwargs: 其他关键字参数将直接传递给模型。

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

        !!! warning "`langchain-openai` 0.3.12 版本中行为已更改"

            已添加对 `tools` 的支持。

        !!! warning "`langchain-openai` 0.3.21 版本中行为已更改"

            将 `kwargs` 传递给模型。
        """
        if strict is not None and method == "json_mode":
            msg = "Argument `strict` is not supported with `method`='json_mode'"
            raise ValueError(msg)
        is_pydantic_schema = _is_pydantic_class(schema)

        if method == "json_schema":
                        # 跳过参数
            if (
                is_pydantic_schema and issubclass(schema, BaseModelV1)  # type: ignore[arg-type]
            ):
                warnings.warn(
                    "Received a Pydantic BaseModel V1 schema. This is not supported by "
                    'method="json_schema". Please use method="function_calling" '
                    "or specify schema via JSON Schema or Pydantic V2 BaseModel. "
                    'Overriding to method="function_calling".'
                )
                method = "function_calling"
                        # 保留参数
            if self.model_name and (
                self.model_name.startswith("gpt-3")
                or self.model_name.startswith("gpt-4-")
                or self.model_name == "gpt-4"
            ):
                warnings.warn(
                    f"Cannot use method='json_schema' with model {self.model_name} "
                    f"since it doesn't support OpenAI's Structured Output API. You can "
                    f"see supported models here: "
                    f"https://platform.openai.com/docs/guides/structured-outputs#supported-models. "  # noqa: E501
                    "To fix this warning, set `method='function_calling'. "
                    "Overriding to method='function_calling'."
                )
                method = "function_calling"

        if method == "function_calling":
            if schema is None:
                msg = (
                    "schema must be specified when method is not 'json_mode'. "
                    "Received None."
                )
                raise ValueError(msg)
            tool_name = convert_to_openai_tool(schema)["function"]["name"]
            bind_kwargs = self._filter_disabled_params(
                **{
                    "tool_choice": tool_name,
                    "parallel_tool_calls": False,
                    "strict": strict,
                    "ls_structured_output_format": {
                        "kwargs": {"method": method, "strict": strict},
                        "schema": schema,
                    },
                    **kwargs,
                }
            )

            llm = self.bind_tools([schema], **bind_kwargs)
            if is_pydantic_schema:
                output_parser: Runnable = PydanticToolsParser(
                    tools=[schema],  # type: ignore[list-item]
                    first_tool_only=True,  # type: ignore[list-item]
                )
            else:
                output_parser = JsonOutputKeyToolsParser(
                    key_name=tool_name, first_tool_only=True
                )
        elif method == "json_mode":
            llm = self.bind(
                **{
                    "response_format": {"type": "json_object"},
                    "ls_structured_output_format": {
                        "kwargs": {"method": method},
                        "schema": schema,
                    },
                    **kwargs,
                }
            )
            output_parser = (
                PydanticOutputParser(pydantic_object=schema)  # type: ignore[arg-type]
                if is_pydantic_schema
                else JsonOutputParser()
            )
        elif method == "json_schema":
            if schema is None:
                msg = (
                    "schema must be specified when method is not 'json_mode'. "
                    "Received None."
                )
                raise ValueError(msg)
            response_format = _convert_to_openai_response_format(schema, strict=strict)
            bind_kwargs = {
                **dict(
                    response_format=response_format,
                    ls_structured_output_format={
                        "kwargs": {"method": method, "strict": strict},
                        "schema": convert_to_openai_tool(schema),
                    },
                    **kwargs,
                )
            }
            if tools:
                bind_kwargs["tools"] = [
                    convert_to_openai_tool(t, strict=strict) for t in tools
                ]
            llm = self.bind(**bind_kwargs)
            if is_pydantic_schema:
                output_parser = RunnableLambda(
                    partial(_oai_structured_outputs_parser, schema=cast(type, schema))
                ).with_types(output_type=cast(type, schema))
            else:
                output_parser = JsonOutputParser()
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

    def _filter_disabled_params(self, **kwargs: Any) -> dict[str, Any]:
        if not self.disabled_params:
            return kwargs
        filtered = {}
        for k, v in kwargs.items():
                        # 跳过 tool_calls，已作为分块发送
            if k in self.disabled_params and (
                self.disabled_params[k] is None or v in self.disabled_params[k]
            ):
                continue
                        # max_tokens 已被 max_completion_tokens 取代
            filtered[k] = v
        return filtered

    def _get_generation_chunk_from_completion(
        self, completion: openai.BaseModel
    ) -> ChatGenerationChunk:
        """从完成结果中获取分块（例如，流式输出的最终完成部分）。"""
        chat_result = self._create_chat_result(completion)
        chat_message = chat_result.generations[0].message
        if isinstance(chat_message, AIMessage):
            usage_metadata = chat_message.usage_metadata
                        # 在 2024 年 9 月发布版中
            if "tool_calls" in chat_message.additional_kwargs:
                chat_message.additional_kwargs.pop("tool_calls")
        else:
            usage_metadata = None
        message = AIMessageChunk(
            content="",
            additional_kwargs=chat_message.additional_kwargs,
            usage_metadata=usage_metadata,
        )
        return ChatGenerationChunk(
            message=message, generation_info=chat_result.llm_output
        )


class ChatOpenAI(BaseChatOpenAI):  # type: ignore[override]
    r"""OpenAI 聊天模型 API 的接口。

    !!! warning "API 范围"

        `ChatOpenAI` 仅针对 [官方 OpenAI API
        规范](https://github.com/openai/openai-openapi)
        。第三方提供者添加的非标准响应字段（如
        `reasoning_content`、`reasoning_details`）**不会**被提取或保留。如果您将
        `base_url` 指向 OpenRouter、vLLM 或 DeepSeek 等提供者，请改用对应的特定提供者
        LangChain 包（例如 `ChatDeepSeek`、`ChatOpenRouter`）。

    ???+ info "设置"

        安装 `langchain-openai` 并设置环境变量 `OPENAI_API_KEY`。

        ```bash
        pip install -U langchain-openai

        # or using uv
        uv add langchain-openai
        ```

        ```bash
        export OPENAI_API_KEY="your-api-key"
        ```

    ??? info "关键初始化参数 — 补全参数"

        | Param               | Type          | Description                                                                                                 |
        | ------------------- | ------------- | ----------------------------------------------------------------------------------------------------------- |
        | `model`             | `str`         | 要使用的 OpenAI 模型名称。                                                                                |
        | `temperature`       | `float`       | 采样温度。                                                                                                  |
        | `max_tokens`        | `int | None`  | 最大生成 token 数量。                                                                                       |
        | `logprobs`          | `bool | None` | 是否返回 logprobs。                                                                                         |
        | `stream_options`    | `dict`        | 配置流式输出，例如在流式传输时是否返回 token 使用情况（`{"include_usage": True}`）。                      |
        | `use_responses_api` | `bool | None` | 是否使用 responses API。                                                                                    |

        查看下方支持的完整初始化参数及其描述列表。

    ??? info "关键初始化参数 — 客户端参数"

        | Param          | Type                                       | Description                                                                         |
        | -------------- | ------------------------------------------ | ----------------------------------------------------------------------------------- |
        | `timeout`      | `float | Tuple[float, float] | Any | None` | 请求超时时间。                                                                      |
        | `max_retries`  | `int | None`                               | 最大重试次数。                                                                      |
        | `api_key`      | `str | None`                               | OpenAI API 密钥。若未传入，则从环境变量 `OPENAI_API_KEY` 中读取。                    |
        | `base_url`     | `str | None`                               | API 请求的基础 URL。仅在使用代理或服务模拟器时指定。默认回退至环境变量 `OPENAI_API_BASE`，然后是 `OPENAI_BASE_URL`（由底层 SDK 客户端读取）。 |
        | `organization` | `str | None`                               | OpenAI 组织 ID。若未传入，则从环境变量 `OPENAI_ORG_ID` 中读取。                      |

        查看下方支持的完整初始化参数及其描述列表。

    ??? info "实例化"

        使用所需参数创建模型实例。例如：

        ```python
        from langchain_openai import ChatOpenAI

        model = ChatOpenAI(
            model="...",
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=2,
            # api_key="...",
            # base_url="...",
            # organization="...",
            # other params...
        )
        ```

        查看所有可用参数。

        !!! tip "保留的参数"
            任何未明确支持的参数都将直接传递给
            [`openai.OpenAI.chat.completions.create(...)`](https://platform.openai.com/docs/api-reference/chat/create)
            每次调用模型时。例如：

            ```python
            from langchain_openai import ChatOpenAI
            import openai

            ChatOpenAI(..., frequency_penalty=0.2).invoke(...)

            # Results in underlying API call of:

            openai.OpenAI(..).chat.completions.create(..., frequency_penalty=0.2)

            # Which is also equivalent to:

            ChatOpenAI(...).invoke(..., frequency_penalty=0.2)
            ```

    ??? info "调用"

        从模型生成一个响应：

        ```python
        messages = [
            (
                "system",
                "You are a helpful translator. Translate the user sentence to French.",
            ),
            ("human", "I love programming."),
        ]
        model.invoke(messages)
        ```

        结果是一个 `AIMessage` 响应：

        ```python
        AIMessage(
            content="J'adore la programmation.",
            response_metadata={
                "token_usage": {
                    "completion_tokens": 5,
                    "prompt_tokens": 31,
                    "total_tokens": 36,
                },
                "model_name": "gpt-4.1-mini",
                "system_fingerprint": "fp_43dfabdef1",
                "finish_reason": "stop",
                "logprobs": None,
            },
            id="run-012cffe2-5d3d-424d-83b5-51c6d4a593d1-0",
            usage_metadata={"input_tokens": 31, "output_tokens": 5, "total_tokens": 36},
        )
        ```

    ??? info "流式输出"

        从模型流式输出一个响应：

        ```python
        for chunk in model.stream(messages):
            print(chunk.text, end="")
        ```

        结果是一系列带有部分内容的 `AIMessageChunk` 对象：

        ```python
        AIMessageChunk(content="", id="run-9e1517e3-12bf-48f2-bb1b-2e824f7cd7b0")
        AIMessageChunk(content="J", id="run-9e1517e3-12bf-48f2-bb1b-2e824f7cd7b0")
        AIMessageChunk(content="'adore", id="run-9e1517e3-12bf-48f2-bb1b-2e824f7cd7b0")
        AIMessageChunk(content=" la", id="run-9e1517e3-12bf-48f2-bb1b-2e824f7cd7b0")
        AIMessageChunk(
            content=" programmation", id="run-9e1517e3-12bf-48f2-bb1b-2e824f7cd7b0"
        )
        AIMessageChunk(content=".", id="run-9e1517e3-12bf-48f2-bb1b-2e824f7cd7b0")
        AIMessageChunk(
            content="",
            response_metadata={"finish_reason": "stop"},
            id="run-9e1517e3-12bf-48f2-bb1b-2e824f7cd7b0",
        )
        ```

        要收集完整消息，可以拼接这些块：

        ```python
        stream = model.stream(messages)
        full = next(stream)
        for chunk in stream:
            full += chunk
        ```

        ```python
        full = AIMessageChunk(
            content="J'adore la programmation.",
            response_metadata={"finish_reason": "stop"},
            id="run-bf917526-7f58-4683-84f7-36a6b671d140",
        )
        ```

    ??? info "异步操作"

        `invoke`、`stream` 和 `batch` 的异步等价方法也可用：

        ```python
        # Invoke
        await model.ainvoke(messages)

        # Stream
        async for chunk in (await model.astream(messages))

        # Batch
        await model.abatch([messages])
        ```

        结果是一个 `AIMessage` 响应：

        ```python
        AIMessage(
            content="J'adore la programmation.",
            response_metadata={
                "token_usage": {
                    "completion_tokens": 5,
                    "prompt_tokens": 31,
                    "total_tokens": 36,
                },
                "model_name": "gpt-4.1-mini",
                "system_fingerprint": "fp_43dfabdef1",
                "finish_reason": "stop",
                "logprobs": None,
            },
            id="run-012cffe2-5d3d-424d-83b5-51c6d4a593d1-0",
            usage_metadata={
                "input_tokens": 31,
                "output_tokens": 5,
                "total_tokens": 36,
            },
        )
        ```

        对于批处理调用，结果是一个 `list[AIMessage]`。

    ??? info "工具调用"

        ```python
        from pydantic import BaseModel, Field


        class GetWeather(BaseModel):
            '''Get the current weather in a given location'''

            location: str = Field(
                ..., description="The city and state, e.g. San Francisco, CA"
            )


        class GetPopulation(BaseModel):
            '''Get the current population in a given location'''

            location: str = Field(
                ..., description="The city and state, e.g. San Francisco, CA"
            )


        model_with_tools = model.bind_tools(
            [GetWeather, GetPopulation]
            # strict = True  # Enforce tool args schema is respected
        )
        ai_msg = model_with_tools.invoke(
            "Which city is hotter today and which is bigger: LA or NY?"
        )
        ai_msg.tool_calls
        ```

        ```python
        [
            {
                "name": "GetWeather",
                "args": {"location": "Los Angeles, CA"},
                "id": "call_6XswGD5Pqk8Tt5atYr7tfenU",
            },
            {
                "name": "GetWeather",
                "args": {"location": "New York, NY"},
                "id": "call_ZVL15vA8Y7kXqOy3dtmQgeCi",
            },
            {
                "name": "GetPopulation",
                "args": {"location": "Los Angeles, CA"},
                "id": "call_49CFW8zqC9W7mh7hbMLSIrXw",
            },
            {
                "name": "GetPopulation",
                "args": {"location": "New York, NY"},
                "id": "call_6ghfKxV264jEfe1mRIkS3PE7",
            },
        ]
        ```

        !!! note "并行工具调用" [`openai >= 1.32`](https://pypi.org/project/openai/)
        支持一个 `parallel_tool_calls` 参数，默认值为 `True`。可将其设为 `False`
        以禁用并行工具调用：

            ```python
            ai_msg = model_with_tools.invoke(
                "What is the weather in LA and NY?", parallel_tool_calls=False
            )
            ai_msg.tool_calls
            ```

            ```python
            [
                {
                    "name": "GetWeather",
                    "args": {"location": "Los Angeles, CA"},
                    "id": "call_4OoY0ZR99iEvC7fevsH8Uhtz",
                }
            ]
            ```

        Like other runtime parameters, `parallel_tool_calls` can be bound to a model
        using `model.bind(parallel_tool_calls=False)` or during instantiation by
        setting `model_kwargs`.

        详见 `bind_tools`。

    ??? info "对话过程中的工具添加"

        ```python
        from langchain_core.messages import HumanMessage, SystemMessage
        from langchain_openai import ChatOpenAI

        model = ChatOpenAI(model="gpt-6-astra", use_responses_api=True)
        model.invoke(
            [
                HumanMessage("What time is it?"),
                SystemMessage(
                    [
                        {
                            "type": "additional_tools",
                            "role": "developer",
                            "tools": [
                                {
                                    "type": "function",
                                    "name": "get_time",
                                    "description": "Get the current time.",
                                    "parameters": {
                                        "type": "object",
                                        "properties": {},
                                    },
                                }
                            ],
                        }
                    ]
                ),
            ]
        )
        ```

    ??? info "内置（服务端）工具"

        You can access [built-in tools](https://platform.openai.com/docs/guides/tools?api-mode=responses)
        supported by the OpenAI Responses API. See [LangChain docs](https://docs.langchain.com/oss/python/integrations/chat/openai#responses-api)
        for more detail.

        ```python
        from langchain_openai import ChatOpenAI

        model = ChatOpenAI(model="...", output_version="responses/v1")

        tool = {"type": "web_search"}
        model_with_tools = model.bind_tools([tool])

        response = model_with_tools.invoke("What was a positive news story from today?")
        response.content
        ```

        ```python
        [
            {
                "type": "text",
                "text": "Today, a heartwarming story emerged from ...",
                "annotations": [
                    {
                        "end_index": 778,
                        "start_index": 682,
                        "title": "Title of story",
                        "type": "url_citation",
                        "url": "<url of story>",
                    }
                ],
            }
        ]
        ```

        !!! version-added "在 `langchain-openai` 0.3.9 中新增"

        !!! version-added "Added in `langchain-openai` 0.3.26: Updated `AIMessage` format"
            [`langchain-openai >= 0.3.26`](https://pypi.org/project/langchain-openai/#history)
            allows users to opt-in to an updated `AIMessage` format when using the
            Responses API. Setting `ChatOpenAI(..., output_version="responses/v1")` will
            format output from reasoning summaries, built-in tool invocations, and other
            response items into the message's `content` field, rather than
            `additional_kwargs`. We recommend this format for new applications.

    ??? info "管理对话状态"

        OpenAI's Responses API supports management of [conversation state](https://platform.openai.com/docs/guides/conversation-state?api-mode=responses).
        Passing in response IDs from previous messages will continue a conversational
        thread.

        ```python
        from langchain_openai import ChatOpenAI

        model = ChatOpenAI(
            model="...",
            use_responses_api=True,
            output_version="responses/v1",
        )
        response = model.invoke("Hi, I'm Bob.")
        response.text
        ```

        ```txt
        "Hi Bob! How can I assist you today?"
        ```

        ```python
        second_response = model.invoke(
            "What is my name?",
            previous_response_id=response.response_metadata["id"],
        )
        second_response.text
        ```

        ```txt
        "Your name is Bob. How can I help you today, Bob?"
        ```

        !!! version-added "在 `langchain-openai` 0.3.9 中新增"

        !!! version-added "在 `langchain-openai` 0.3.26 中新增"

            You can also initialize `ChatOpenAI` with `use_previous_response_id`.
            Input messages up to the most recent response will then be dropped from request
            payloads, and `previous_response_id` will be set using the ID of the most
            recent response.

            ```python
            model = ChatOpenAI(model="...", use_previous_response_id=True)
            ```

        !!! note "OpenAI 兼容端点"

            Some OpenAI-compatible providers/proxies may not support forwarding
            reasoning blocks in request history. If you see request-format
            errors while using reasoning + Responses API, prefer
            `use_previous_response_id=True` (so the server keeps
            conversation state).

    ??? info "推理输出"

        OpenAI's Responses API supports [reasoning models](https://platform.openai.com/docs/guides/reasoning?api-mode=responses)
        that expose a summary of internal reasoning processes.

        ```python
        from langchain_openai import ChatOpenAI

        reasoning = {
            "effort": "medium",  # 'low', 'medium', or 'high'
            "summary": "auto",  # 'detailed', 'auto', or None
        }

        model = ChatOpenAI(
            model="...", reasoning=reasoning, output_version="responses/v1"
        )
        response = model.invoke("What is 3^3?")

        # Response text
        print(f"Output: {response.text}")

        # Reasoning summaries
        for block in response.content:
            if block["type"] == "reasoning":
                for summary in block["summary"]:
                    print(summary["text"])
        ```

        ```txt
        Output: 3³ = 27
        Reasoning: The user wants to know...
        ```

        !!! version-added "Added in `langchain-openai` 0.3.26: Updated `AIMessage` format"
            [`langchain-openai >= 0.3.26`](https://pypi.org/project/langchain-openai/#history)
            allows users to opt-in to an updated `AIMessage` format when using the
            Responses API. Setting `ChatOpenAI(..., output_version="responses/v1")` will
            format output from reasoning summaries, built-in tool invocations, and other
            response items into the message's `content` field, rather than
            `additional_kwargs`. We recommend this format for new applications.

        !!! note "Troubleshooting with non-OpenAI backends"
            When using a non-OpenAI endpoint via `base_url`, request handling for
            reasoning history can differ. If agent loops fail after tool calls, use:
            `ChatOpenAI(..., use_responses_api=True, use_previous_response_id=True)`.

    ??? info "结构化输出"

        ```python
        from pydantic import BaseModel, Field


        class Joke(BaseModel):
            '''Joke to tell user.'''

            setup: str = Field(description="The setup of the joke")
            punchline: str = Field(description="The punchline to the joke")
            rating: int | None = Field(
                description="How funny the joke is, from 1 to 10"
            )


        structured_model = model.with_structured_output(Joke)
        structured_model.invoke("Tell me a joke about cats")
        ```

        ```python
        Joke(
            setup="Why was the cat sitting on the computer?",
            punchline="To keep an eye on the mouse!",
            rating=None,
        )
        ```

        请参阅 `with_structured_output` 获取更多信息。

    ??? info "JSON 模式"

        ```python
        json_model = model.bind(response_format={"type": "json_object"})
        ai_msg = json_model.invoke(
            "Return a JSON object with key 'random_ints' and a value of 10 random ints in [0-99]"
        )
        ai_msg.content
        ```

        ```txt
        '\\n{\\n  "random_ints": [23, 87, 45, 12, 78, 34, 56, 90, 11, 67]\\n}'
        ```

    ??? info "图像输入"

        ```python
        import base64
        import httpx
        from langchain.messages import HumanMessage

        image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
        image_data = base64.b64encode(httpx.get(image_url).content).decode("utf-8")
        message = HumanMessage(
            content=[
                {"type": "text", "text": "describe the weather in this image"},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_data}"},
                },
            ]
        )

        ai_msg = model.invoke([message])
        ai_msg.content
        ```

        ```txt
        "The weather in the image appears to be clear and pleasant. The sky is mostly blue with scattered, light clouds, suggesting a sunny day with minimal cloud cover. There is no indication of rain or strong winds, and the overall scene looks bright and calm. The lush green grass and clear visibility further indicate good weather conditions."
        ```

    ??? info "Token 使用量"

        ```python
        ai_msg = model.invoke(messages)
        ai_msg.usage_metadata

        ```txt
        {"input_tokens": 28, "output_tokens": 5, "total_tokens": 33}
        ```

        When streaming, set the `stream_usage` kwarg:

        ```python
        stream = model.stream(messages, stream_usage=True)
        full = next(stream)
        for chunk in stream:
            full += chunk
        full.usage_metadata
        ```

        ```txt
        {"input_tokens": 28, "output_tokens": 5, "total_tokens": 33}
        ```

    ??? info "Logprobs"

        ```python
        logprobs_model = model.bind(logprobs=True)
        ai_msg = logprobs_model.invoke(messages)
        ai_msg.response_metadata["logprobs"]
        ```

        ```txt
        {
            "content": [
                {
                    "token": "J",
                    "bytes": [74],
                    "logprob": -4.9617593e-06,
                    "top_logprobs": [],
                },
                {
                    "token": "'adore",
                    "bytes": [39, 97, 100, 111, 114, 101],
                    "logprob": -0.25202933,
                    "top_logprobs": [],
                },
                {
                    "token": " la",
                    "bytes": [32, 108, 97],
                    "logprob": -0.20141791,
                    "top_logprobs": [],
                },
                {
                    "token": " programmation",
                    "bytes": [
                        32,
                        112,
                        114,
                        111,
                        103,
                        114,
                        97,
                        109,
                        109,
                        97,
                        116,
                        105,
                        111,
                        110,
                    ],
                    "logprob": -1.9361265e-07,
                    "top_logprobs": [],
                },
                {
                    "token": ".",
                    "bytes": [46],
                    "logprob": -1.2233183e-05,
                    "top_logprobs": [],
                },
            ]
        }
        ```

    ??? info "Response metadata"

        ```python
        ai_msg = model.invoke(messages)
        ai_msg.response_metadata
        ```

        ```txt
        {
            "token_usage": {
                "completion_tokens": 5,
                "prompt_tokens": 28,
                "total_tokens": 33,
            },
            "model_name": "gpt-4.1-mini",
            "system_fingerprint": "fp_319be4768e",
            "finish_reason": "stop",
            "logprobs": None,
        }
        ```

    ??? info "Flex processing"

        OpenAI offers a variety of [service tiers](https://platform.openai.com/docs/guides/flex-processing?api-mode=responses).
        The "flex" tier offers cheaper pricing for requests, with the trade-off that
        responses may take longer and resources might not always be available.
        This approach is best suited for non-critical tasks, including model testing,
        data enhancement, or jobs that can be run asynchronously.

        To use it, initialize the model with `service_tier="flex"`:

        ```python
        from langchain_openai import ChatOpenAI

        model = ChatOpenAI(model="...", service_tier="flex")
        ```

        Note that this is a beta feature that is only available for a subset of models.
        See OpenAI [flex processing docs](https://platform.openai.com/docs/guides/flex-processing?api-mode=responses)
        for more detail.

    ??? info "OpenAI-compatible APIs"

        `ChatOpenAI` can be used with OpenAI-compatible APIs like
        [LM Studio](https://lmstudio.ai/), [vLLM](https://github.com/vllm-project/vllm),
        [Ollama](https://ollama.com/), and others.

        To use custom parameters specific to these providers, use the `extra_body` parameter.

        !!! example "LM Studio example with TTL (auto-eviction)"

            ```python
            from langchain_openai import ChatOpenAI

            model = ChatOpenAI(
                base_url="http://localhost:1234/v1",
                api_key="lm-studio",  # Can be any string
                model="mlx-community/QwQ-32B-4bit",
                temperature=0,
                extra_body={
                    "ttl": 300
                },  # Auto-evict model after 5 minutes of inactivity
            )
            ```

        !!! example "vLLM example with custom parameters"

            ```python
            model = ChatOpenAI(
                base_url="http://localhost:8000/v1",
                api_key="EMPTY",
                model="meta-llama/Llama-2-7b-chat-hf",
                extra_body={"use_beam_search": True, "best_of": 4},
            )
            ```

        !!! warning "Model name can trigger Responses API routing"

            The choice between the Chat Completions API (`/v1/chat/completions`)
            and the Responses API (`/v1/responses`) is inferred in part from the
            model name, independent of `base_url`.

            `use_responses_api` should generally be set explicitly to avoid ambiguity,
            especially when using OpenAI-compatible providers:

            ```python
            model = ChatOpenAI(
                base_url="http://localhost:8000/v1",
                api_key="EMPTY",
                model="codex-7b-instruct",
                use_responses_api=False,
            )
            ```

    ??? info "`model_kwargs` vs `extra_body`"

        Use the correct parameter for different types of API arguments:

        **Use `model_kwargs` for:**

        - Standard OpenAI API parameters not explicitly defined as class parameters
        - Parameters that should be flattened into the top-level request payload
        - Examples: `max_completion_tokens`, `stream_options`, `modalities`, `audio`

        ```python
        # 标准 OpenAI 参数
        model = ChatOpenAI(
            model="...",
            model_kwargs={
                "stream_options": {"include_usage": True},
                "max_completion_tokens": 300,
                "modalities": ["text", "audio"],
                "audio": {"voice": "alloy", "format": "wav"},
            },
        )
        ```

        **Use `extra_body` for:**

        - Custom parameters specific to OpenAI-compatible providers (vLLM, LM Studio,
            OpenRouter, etc.)
        - Parameters that need to be nested under `extra_body` in the request
        - Any non-standard OpenAI API parameters

        ```python
        # 自定义提供方参数
        model = ChatOpenAI(
            base_url="http://localhost:8000/v1",
            model="custom-model",
            extra_body={
                "use_beam_search": True,  # vLLM 参数
                "best_of": 4,  # vLLM 参数
                "ttl": 300,  # LM Studio 参数
            },
        )
        ```

        **Key Differences:**

        - `model_kwargs`: Parameters are **merged into top-level** request payload
        - `extra_body`: Parameters are **nested under `extra_body`** key in request

        !!! warning
            Always use `extra_body` for custom parameters, **not** `model_kwargs`.
            Using `model_kwargs` for non-OpenAI parameters will cause API errors.

    ??? info "Prompt caching optimization"

        OpenAI prompt caching is automatic for eligible prompts. For high-volume
        applications with repetitive prompts, use `prompt_cache_key`
        per invocation to improve cache hit rates and reduce costs:

        ```python
        model = ChatOpenAI(model="...")

        response = model.invoke(
            messages,
            prompt_cache_key="example-key-a",  # 路由到同一台机器以命中缓存
        )

        customer_response = model.invoke(messages, prompt_cache_key="example-key-b")
        support_response = model.invoke(messages, prompt_cache_key="example-key-c")

        # 基于上下文的动态缓存键
        cache_key = f"example-key-{dynamic_suffix}"
        response = model.invoke(messages, prompt_cache_key=cache_key)
        ```

        The default `"implicit"` mode keeps OpenAI's automatic breakpoint and
        also uses explicit breakpoints. The `"explicit"` mode uses only the
        breakpoints you provide.

        For models that support explicit cache breakpoints, pass
        request-level cache options and mark supported content blocks
        with `prompt_cache_breakpoint`:

        ```python
        model = ChatOpenAI(model="gpt-5.6-sol")
        response = model.invoke(
            [
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": "稳定的指令和示例...",
                            "prompt_cache_breakpoint": {"mode": "explicit"},
                        }
                    ],
                },
                {"role": "user", "content": "当前请求"},
            ],
            prompt_cache_key="tenant:acme:support-v1",
            prompt_cache_options={"mode": "explicit", "ttl": "30m"},
        )
        ```

        Set `prompt_cache_options` per invocation, as above, or persist it on
        the model:

        ```python
        model = ChatOpenAI(
            model="gpt-5.6-sol",
            prompt_cache_options={"mode": "explicit", "ttl": "30m"},
        )
        ```

        `prompt_cache_options["mode"]` can be `"implicit"` or `"explicit"`.
        OpenAI limits how many breakpoints can write to the cache in a single
        request. In `"implicit"` mode, the implicit breakpoint on the latest
        message uses one write slot, so up to three explicit breakpoints can
        write. In `"explicit"` mode, up to four explicit breakpoints can write.
        For reads, OpenAI considers up to the latest 50 breakpoints.

        For models before the GPT-5.6 family that support legacy prompt cache
        retention, pass `prompt_cache_retention`. See OpenAI's
        [prompt caching docs](https://platform.openai.com/docs/guides/prompt-caching)
        for the current model support list and retention semantics.

        ```python
        response = model.invoke(messages, prompt_cache_retention="24h")
        ```

        Cache keys help ensure requests with the same prompt prefix are routed
        to machines with existing cache, providing cost reduction and
        latency improvement on cached tokens. Cache reads are available as
        `response.usage_metadata["input_token_details"]["cache_read"]`; cache
        writes are available as `"cache_creation"` when the OpenAI response
        includes `cache_write_tokens`. On the `"priority"` and `"flex"`
        service tiers these keys are prefixed with the tier name
        (e.g. `"priority_cache_read"`).
    """  # noqa: E501

    max_tokens: int | None = Field(default=None, alias="max_completion_tokens")
    """生成的最大 token 数量。"""

    @property
    def lc_secrets(self) -> dict[str, str]:
        """密钥环境变量的映射。"""
        return {"openai_api_key": "OPENAI_API_KEY"}

    @classmethod
    def get_lc_namespace(cls) -> list[str]:
        """获取 LangChain 对象的命名空间。

        Returns:
            `["langchain", "chat_models", "openai"]`
        """
        return ["langchain", "chat_models", "openai"]

    @property
    def lc_attributes(self) -> dict[str, Any]:
        """获取 langchain 对象的属性。"""
        attributes: dict[str, Any] = {}

        if self.openai_organization:
            attributes["openai_organization"] = self.openai_organization

        if self.openai_api_base:
            attributes["openai_api_base"] = self.openai_api_base

        if self.openai_proxy:
            attributes["openai_proxy"] = self.openai_proxy

        return attributes

    @classmethod
    def is_lc_serializable(cls) -> bool:
        """返回该模型是否可被 LangChain 序列化。"""
        return True

    @property
    def _default_params(self) -> dict[str, Any]:
        """获取调用 OpenAI API 的默认参数。"""
        params = super()._default_params
        if "max_tokens" in params:
            params["max_completion_tokens"] = params.pop("max_tokens")

        return params

    def _get_request_payload(
        self,
        input_: LanguageModelInput,
        *,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> dict:
        payload = super()._get_request_payload(input_, stop=stop, **kwargs)
                # 将系统消息角色修改为 "developer" 以适配 o 系列模型
                # 设置合理的限制以防止资源耗尽
        if "max_tokens" in payload:
            payload["max_completion_tokens"] = payload.pop("max_tokens")

                # 超时可防止因慢速或恶意服务器导致的无限挂起
        if self.model_name and re.match(r"^o\d", self.model_name):
            for message in payload.get("messages", []):
                if message["role"] == "system":
                    message["role"] = "developer"
        return payload

    def _stream(self, *args: Any, **kwargs: Any) -> Iterator[ChatGenerationChunk]:
        """路由到 Chat Completions 或 Responses API。"""
        if self._use_responses_api({**kwargs, **self.model_kwargs}):
            return super()._stream_responses(*args, **kwargs)
        return super()._stream(*args, **kwargs)

    async def _astream(
        self, *args: Any, **kwargs: Any
    ) -> AsyncIterator[ChatGenerationChunk]:
        """路由到 Chat Completions 或 Responses API。"""
        if self._use_responses_api({**kwargs, **self.model_kwargs}):
            async for chunk in super()._astream_responses(*args, **kwargs):
                yield chunk
        else:
            async for chunk in super()._astream(*args, **kwargs):
                yield chunk

    def with_structured_output(
        self,
        schema: _DictOrPydanticClass | None = None,
        *,
        method: Literal["function_calling", "json_mode", "json_schema"] = "json_schema",
        include_raw: bool = False,
        strict: bool | None = None,
        tools: list | None = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, _DictOrPydantic]:
        r"""模型包装器，返回格式化为匹配给定模式的输出。

        Args:
            schema: 输出模式。可作为以下形式传入：

                - OpenAI 函数/工具模式，
                - JSON Schema，
                - `TypedDict` 类，
                - 或 Pydantic 类。

                If `schema` is a Pydantic class then the model output will be a
                Pydantic instance of that class, and the model-generated fields will be
                validated by the Pydantic class. Otherwise the model output will be a
                dict and will not be validated.

                有关如何在指定 Pydantic 或 `TypedDict`
                类时正确说明模式字段的类型和描述，请参见
                `langchain_core.utils.function_calling.convert_to_openai_tool`。

            method: 控制模型生成的方法，可选之一：

                - `'json_schema'`:
                    Uses OpenAI's [Structured Output API](https://platform.openai.com/docs/guides/structured-outputs).
                    See the docs for [supported models](https://platform.openai.com/docs/guides/structured-outputs#supported-models).
                - `'function_calling'`:
                    Uses OpenAI's [tool-calling API](https://platform.openai.com/docs/guides/function-calling)
                    (formerly called function calling).
                - `'json_mode'`:
                    Uses OpenAI's [JSON mode](https://platform.openai.com/docs/guides/structured-outputs#json-mode).
                    Note that if using JSON mode then you must include instructions for
                    formatting the output into the desired schema into the model call.

                了解更多关于 [方法之间的差异](https://platform.openai.com/docs/guides/structured-outputs#function-calling-vs-response-format)。

            include_raw:
                如果为 `False`，则仅返回解析后的结构化输出。

                如果模型输出解析过程中发生错误，将引发异常。

                如果为 `True`，则会返回原始模型响应（一个
                `BaseMessage`）和解析后的模型响应。

                如果在输出解析过程中发生错误，该错误将被捕获并一同返回。

                The final output is always a `dict` with keys `'raw'`, `'parsed'`, and
                `'parsing_error'`.
            strict:

                - `True`: 模型输出将保证与模式完全匹配。 输入模式也将根据
                [支持的模式](https://platform.openai.com/docs/guides/structured-outputs#supported-schemas)
                进行验证。 - `False`: 不会验证输入模式，也不会验证模型输出。 - `None`:
                不会将 `strict` 参数传递给模型。

                If schema is specified via `TypedDict` or JSON schema, `strict` is not
                enabled by default. Pass `strict=True` to enable it.

                !!! note
                    `strict` 仅在 `method` 为 `'json_schema'` 或 `'function_calling'` 时才可非空。
            tools:
                要绑定到聊天模型的工具类对象列表。要求满足：

                - `method` 为 `'json_schema'`（默认值）。
                - `strict=True`
                - `include_raw=True`

                如果模型选择调用某个工具，则 `'raw'` 中的 `AIMessage`
                将包含工具调用信息。

                ??? example

                    ```python
                    from langchain.chat_models import init_chat_model
                    from pydantic import BaseModel


                    class ResponseSchema(BaseModel):
                        response: str


                    def get_weather(location: str) -> str:
                        \"\"\"Get weather at a location.\"\"\"
                        pass

                    model = init_chat_model("openai:gpt-5.5")

                    structured_model = model.with_structured_output(
                        ResponseSchema,
                        tools=[get_weather],
                        strict=True,
                        include_raw=True,
                    )

                    structured_model.invoke("What's the weather in Boston?")
                    ```

                    ```python
                    {
                        "raw": AIMessage(content="", tool_calls=[...], ...),
                        "parsing_error": None,
                        "parsed": None,
                    }
                    ```

            kwargs: 额外的关键字参数将透传至模型。

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

        !!! warning "Behavior changed in `langchain-openai` 0.3.0"

            `method` 默认值从 `"function_calling"` 更改为 `"json_schema"`。

        !!! warning "Behavior changed in `langchain-openai` 0.3.12"

            增加对 `tools` 的支持。

        !!! warning "Behavior changed in `langchain-openai` 0.3.21"

            透传 `kwargs` 至模型。

        ??? note "Example: `schema=Pydantic` 类，`method='json_schema'`，`include_raw=False`，`strict=True`"

            Note, OpenAI has a number of restrictions on what types of schemas can be
            provided if `strict = True`. When using Pydantic, our model cannot
            specify any Field metadata (like min/max constraints) and fields cannot
            have default values.

            参见 [所有约束](https://platform.openai.com/docs/guides/structured-outputs#supported-schemas)。

            ```python
            from langchain_openai import ChatOpenAI
            from pydantic import BaseModel, Field


            class AnswerWithJustification(BaseModel):
                '''An answer to the user question along with justification for the answer.'''

                answer: str
                justification: str | None = Field(
                    default=..., description="A justification for the answer."
                )


            model = ChatOpenAI(model="...", temperature=0)
            structured_model = model.with_structured_output(AnswerWithJustification)

            structured_model.invoke(
                "What weighs more a pound of bricks or a pound of feathers"
            )
            ```

            ```python
            AnswerWithJustification(
                answer="They weigh the same",
                justification="Both a pound of bricks and a pound of feathers weigh one pound. The weight is the same, but the volume or density of the objects may differ.",
            )
            ```

        ??? note "示例：`schema=Pydantic` 类，`method='function_calling'`，`include_raw=False`，`strict=False`"

            ```python
            from langchain_openai import ChatOpenAI
            from pydantic import BaseModel, Field


            class AnswerWithJustification(BaseModel):
                '''An answer to the user question along with justification for the answer.'''

                answer: str
                justification: str | None = Field(
                    default=..., description="A justification for the answer."
                )


            model = ChatOpenAI(model="...", temperature=0)
            structured_model = model.with_structured_output(
                AnswerWithJustification, method="function_calling"
            )

            structured_model.invoke(
                "What weighs more a pound of bricks or a pound of feathers"
            )
            ```

            ```python
            AnswerWithJustification(
                answer="They weigh the same",
                justification="Both a pound of bricks and a pound of feathers weigh one pound. The weight is the same, but the volume or density of the objects may differ.",
            )
            ```

        ??? note "示例：`schema=Pydantic` 类，`method='json_schema'`，`include_raw=True`"

            ```python
            from langchain_openai import ChatOpenAI
            from pydantic import BaseModel


            class AnswerWithJustification(BaseModel):
                '''An answer to the user question along with justification for the answer.'''

                answer: str
                justification: str


            model = ChatOpenAI(model="...", temperature=0)
            structured_model = model.with_structured_output(
                AnswerWithJustification, include_raw=True
            )

            structured_model.invoke(
                "What weighs more a pound of bricks or a pound of feathers"
            )
            ```

            ```python
            {
                "raw": AIMessage(
                    content="",
                    additional_kwargs={
                        "tool_calls": [
                            {
                                "id": "call_Ao02pnFYXD6GN1yzc0uXPsvF",
                                "function": {
                                    "arguments": '{"answer":"They weigh the same.","justification":"Both a pound of bricks and a pound of feathers weigh one pound. The weight is the same, but the volume or density of the objects may differ."}',
                                    "name": "AnswerWithJustification",
                                },
                                "type": "function",
                            }
                        ]
                    },
                ),
                "parsed": AnswerWithJustification(
                    answer="They weigh the same.",
                    justification="Both a pound of bricks and a pound of feathers weigh one pound. The weight is the same, but the volume or density of the objects may differ.",
                ),
                "parsing_error": None,
            }
            ```

        ??? note "示例：`schema=TypedDict` 类，`method='json_schema'`，`include_raw=False`，`strict=False`"

            ```python
            from typing_extensions import Annotated, TypedDict

            from langchain_openai import ChatOpenAI


            class AnswerWithJustification(TypedDict):
                '''An answer to the user question along with justification for the answer.'''

                answer: str
                justification: Annotated[
                    str | None, None, "A justification for the answer."
                ]


            model = ChatOpenAI(model="...", temperature=0)
            structured_model = model.with_structured_output(AnswerWithJustification)

            structured_model.invoke(
                "What weighs more a pound of bricks or a pound of feathers"
            )
            ```

            ```python
            {
                "answer": "They weigh the same",
                "justification": "Both a pound of bricks and a pound of feathers weigh one pound. The weight is the same, but the volume and density of the two substances differ.",
            }
            ```

        ??? note "示例：`schema=OpenAI` 函数模式，`method='json_schema'`，`include_raw=False`"

            ```python
            from langchain_openai import ChatOpenAI

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

            model = ChatOpenAI(model="...", temperature=0)
            structured_model = model.with_structured_output(oai_schema)

            structured_model.invoke(
                "What weighs more a pound of bricks or a pound of feathers"
            )
            ```

            ```python
            {
                "answer": "They weigh the same",
                "justification": "Both a pound of bricks and a pound of feathers weigh one pound. The weight is the same, but the volume and density of the two substances differ.",
            }
            ```

        ??? note "示例：`schema=Pydantic` 类，`method='json_mode'`，`include_raw=True`"

            ```python
            from langchain_openai import ChatOpenAI
            from pydantic import BaseModel


            class AnswerWithJustification(BaseModel):
                answer: str
                justification: str


            model = ChatOpenAI(model="...", temperature=0)
            structured_model = model.with_structured_output(
                AnswerWithJustification, method="json_mode", include_raw=True
            )

            structured_model.invoke(
                "Answer the following question. "
                "Make sure to return a JSON blob with keys 'answer' and 'justification'.\\n\\n"
                "What's heavier a pound of bricks or a pound of feathers?"
            )
            ```

            ```python
            {
                "raw": AIMessage(
                    content='{\\n    "answer": "They are both the same weight.",\\n    "justification": "Both a pound of bricks and a pound of feathers weigh one pound. The difference lies in the volume and density of the materials, not the weight." \\n}'
                ),
                "parsed": AnswerWithJustification(
                    answer="They are both the same weight.",
                    justification="Both a pound of bricks and a pound of feathers weigh one pound. The difference lies in the volume and density of the materials, not the weight.",
                ),
                "parsing_error": None,
            }
            ```

        ??? note "示例：`schema=None`，`method='json_mode'`，`include_raw=True`"

            ```python
            structured_model = model.with_structured_output(
                method="json_mode", include_raw=True
            )

            structured_model.invoke(
                "Answer the following question. "
                "Make sure to return a JSON blob with keys 'answer' and 'justification'.\\n\\n"
                "What's heavier a pound of bricks or a pound of feathers?"
            )
            ```

            ```python
            {
                "raw": AIMessage(
                    content='{\\n    "answer": "They are both the same weight.",\\n    "justification": "Both a pound of bricks and a pound of feathers weigh one pound. The difference lies in the volume and density of the materials, not the weight." \\n}'
                ),
                "parsed": {
                    "answer": "They are both the same weight.",
                    "justification": "Both a pound of bricks and a pound of feathers weigh one pound. The difference lies in the volume and density of the materials, not the weight.",
                },
                "parsing_error": None,
            }
            ```

        """  # noqa: E501
        return super().with_structured_output(
            schema,
            method=method,
            include_raw=include_raw,
            strict=strict,
            tools=tools,
            **kwargs,
        )


def _is_pydantic_class(obj: Any) -> bool:
    return isinstance(obj, type) and is_basemodel_subclass(obj)


def _lc_tool_call_to_openai_tool_call(tool_call: ToolCall) -> dict:
    return {
        "type": "function",
        "id": tool_call["id"],
        "function": {
            "name": tool_call["name"],
            "arguments": json.dumps(tool_call["args"], ensure_ascii=False),
        },
    }


def _lc_invalid_tool_call_to_openai_tool_call(
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


def _url_to_size(image_source: str) -> tuple[int, int] | None:
    try:
        from PIL import Image  # type: ignore[import]
    except ImportError:
        logger.info(
            "Unable to count image tokens. To count image tokens please install "
            "`pip install -U pillow httpx`."
        )
        return None
    if _is_url(image_source):
        import httpx

                # 秒
                # 最大尺寸匹配 OpenAI 的 50 MB 负载限制
        timeout = 5.0  # 50 MB
                # 在加载到内存前检查响应大小
        max_size = 50 * 1024 * 1024  # 同时检查实际内容大小

        try:
            response = _get_ssrf_safe_client().get(image_source, timeout=timeout)
            response.raise_for_status()

                        # 参考：https://platform.openai.com/docs/guides/vision/calculating-costs
            content_length = response.headers.get("content-length")
            if content_length and int(content_length) > max_size:
                logger.warning(
                    "Image URL exceeds maximum size limit of %d bytes", max_size
                )
                return None

                        # 较大边长必须 <= 2048
            if len(response.content) > max_size:
                logger.warning(
                    "Image URL exceeds maximum size limit of %d bytes", max_size
                )
                return None

            with Image.open(BytesIO(response.content)) as img:
                width, height = img.size
            return width, height
        except httpx.TimeoutException:
            logger.warning("Image URL request timed out after %s seconds", timeout)
            return None
        except httpx.HTTPStatusError as e:
            logger.warning("Image URL returned HTTP error: %s", e)
            return None
        except Exception as e:
            logger.warning("Failed to fetch or process image from URL: %s", e)
            return None

    if _is_b64(image_source):
        _, encoded = image_source.split(",", 1)
        data = base64.b64decode(encoded)
        with Image.open(BytesIO(data)) as img:
            width, height = img.size
        return width, height
    return None


def _count_image_tokens(width: int, height: int) -> int:
        # 较小边长必须 <= 768
    width, height = _resize(width, height)
    h = ceil(height / 512)
    w = ceil(width / 512)
    return (170 * h * w) + 85


def _is_url(s: str) -> bool:
    try:
        result = urlparse(s)
        return all([result.scheme, result.netloc])
    except Exception as e:
        logger.debug("Unable to parse URL: %s", e)
        return False


def _is_b64(s: str) -> bool:
    return s.startswith("data:image")


def _resize(width: int, height: int) -> tuple[int, int]:
        # 避免将缓存读取和推理 token 计入服务层级
    if width > 2048 or height > 2048:
        if width > height:
            height = (height * 2048) // width
            width = 2048
        else:
            width = (width * 2048) // height
            height = 2048
        # token 数量，因为服务层级 token 已经有不同定价
    if width > 768 and height > 768:
        if width > height:
            width = (width * 768) // height
            height = 768
        else:
            height = (height * 768) // width
            width = 768
    return width, height


def _convert_to_openai_response_format(
    schema: dict[str, Any] | type, *, strict: bool | None = None
) -> dict | TypeBaseModel:
    if isinstance(schema, type) and is_basemodel_subclass(schema):
        return schema

    if (
        isinstance(schema, dict)
        and "json_schema" in schema
        and schema.get("type") == "json_schema"
    ):
        response_format = schema
    elif isinstance(schema, dict) and "name" in schema and "schema" in schema:
        response_format = {"type": "json_schema", "json_schema": schema}
    else:
        if strict is None:
            if isinstance(schema, dict) and isinstance(schema.get("strict"), bool):
                strict = schema["strict"]
            else:
                strict = False
        function = convert_to_openai_function(schema, strict=strict)
        function["schema"] = function.pop("parameters")
        response_format = {"type": "json_schema", "json_schema": function}

    if (
        strict is not None
        and strict is not response_format["json_schema"].get("strict")
        and isinstance(schema, dict)
        and "strict" in schema.get("json_schema", {})
    ):
        msg = (
            f"Output schema already has 'strict' value set to "
            f"{schema['json_schema']['strict']} but 'strict' also passed in to "
            f"with_structured_output as {strict}. Please make sure that "
            f"'strict' is only specified in one place."
        )
        raise ValueError(msg)
    return response_format


def _oai_structured_outputs_parser(
    ai_msg: AIMessage, schema: type[_BM]
) -> PydanticBaseModel | None:
    if (parsed := ai_msg.additional_kwargs.get("parsed")) is not None:
        if isinstance(parsed, dict):
            return schema(**parsed)
        return parsed
    if ai_msg.additional_kwargs.get("refusal"):
        raise OpenAIRefusalError(ai_msg.additional_kwargs["refusal"])
    if any(
        isinstance(block, dict)
        and block.get("type") == "non_standard"
        and "refusal" in block["value"]  # type: ignore[typeddict-item]
        for block in ai_msg.content_blocks
    ):
        refusal = next(
            block["value"]["refusal"]
            for block in ai_msg.content_blocks
            if isinstance(block, dict)
            and block["type"] == "non_standard"
            and "refusal" in block["value"]
        )
        raise OpenAIRefusalError(refusal)
    if ai_msg.tool_calls:
        return None
    msg = (
        "Structured Output response does not have a 'parsed' field nor a 'refusal' "
        f"field. Received message:\n\n{ai_msg}"
    )
    raise ValueError(msg)


class OpenAIRefusalError(Exception):
    """当 OpenAI 结构化输出 API 返回拒绝时抛出的错误。

    When using OpenAI's Structured Outputs API with user-generated input, the model
    may occasionally refuse to fulfill the request for safety reasons.

    详见 [关于拒绝的更多说明](https://platform.openai.com/docs/guides/structured-outputs/refusals)。
    """


def _create_usage_metadata(
    oai_token_usage: dict, service_tier: str | None = None
) -> UsageMetadata:
    _input = oai_token_usage.get("prompt_tokens")
    input_tokens = _input if _input is not None else 0
    _output = oai_token_usage.get("completion_tokens")
    output_tokens = _output if _output is not None else 0
    _total = oai_token_usage.get("total_tokens")
    total_tokens = _total if _total is not None else input_tokens + output_tokens
    if service_tier not in {"priority", "flex"}:
        service_tier = None
    service_tier_prefix = f"{service_tier}_" if service_tier else ""
    prompt_tokens_details = oai_token_usage.get("prompt_tokens_details") or {}
    input_token_details: dict = {
        "audio": prompt_tokens_details.get("audio_tokens"),
        f"{service_tier_prefix}cache_read": prompt_tokens_details.get("cached_tokens"),
        f"{service_tier_prefix}cache_creation": prompt_tokens_details.get(
            "cache_write_tokens"
        ),
    }
    output_token_details: dict = {
        "audio": (oai_token_usage.get("completion_tokens_details") or {}).get(
            "audio_tokens"
        ),
        f"{service_tier_prefix}reasoning": (
            oai_token_usage.get("completion_tokens_details") or {}
        ).get("reasoning_tokens"),
    }
    if service_tier is not None:
                # 避免将缓存读取和推理 token 计入服务层级
                # token 数量，因为服务层级 token 已经有不同定价
        input_token_details[service_tier] = input_tokens - (
            input_token_details.get(f"{service_tier_prefix}cache_read", 0) or 0
        )
        output_token_details[service_tier] = output_tokens - (
            output_token_details.get(f"{service_tier_prefix}reasoning", 0) or 0
        )
    return UsageMetadata(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        input_token_details=InputTokenDetails(
            **{k: v for k, v in input_token_details.items() if v is not None}
        ),
        output_token_details=OutputTokenDetails(
            **{k: v for k, v in output_token_details.items() if v is not None}
        ),
    )


def _create_usage_metadata_responses(
    oai_token_usage: dict, service_tier: str | None = None
) -> UsageMetadata:
    _input = oai_token_usage.get("input_tokens")
    input_tokens = _input if _input is not None else 0
    _output = oai_token_usage.get("output_tokens")
    output_tokens = _output if _output is not None else 0
    _total = oai_token_usage.get("total_tokens")
    total_tokens = _total if _total is not None else input_tokens + output_tokens
    if service_tier not in {"priority", "flex"}:
        service_tier = None
    service_tier_prefix = f"{service_tier}_" if service_tier else ""
    output_token_details: dict = {
        f"{service_tier_prefix}reasoning": (
            oai_token_usage.get("output_tokens_details") or {}
        ).get("reasoning_tokens")
    }
    input_tokens_details = oai_token_usage.get("input_tokens_details") or {}
    input_token_details: dict = {
        f"{service_tier_prefix}cache_read": input_tokens_details.get("cached_tokens"),
        f"{service_tier_prefix}cache_creation": input_tokens_details.get(
            "cache_write_tokens"
        ),
    }
    if service_tier is not None:
                # 继续搜索具有有效 response_id 的 AIMessage
                # 重命名旧参数
        output_token_details[service_tier] = output_tokens - (
            output_token_details.get(f"{service_tier_prefix}reasoning", 0) or 0
        )
        input_token_details[service_tier] = input_tokens - (
            input_token_details.get(f"{service_tier_prefix}cache_read", 0) or 0
        )
    return UsageMetadata(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        input_token_details=InputTokenDetails(
            **{k: v for k, v in input_token_details.items() if v is not None}
        ),
        output_token_details=OutputTokenDetails(
            **{k: v for k, v in output_token_details.items() if v is not None}
        ),
    )


def _is_builtin_tool(tool: dict) -> bool:
    return "type" in tool and tool["type"] != "function"


def _use_responses_api(payload: dict) -> bool:
    uses_builtin_tools = "tools" in payload and any(
        _is_builtin_tool(tool) for tool in payload["tools"]
    )
    responses_only_args = {
        "context_management",
        "include",
        "previous_response_id",
        "reasoning",
        "text",
        "truncation",
    }
    return bool(uses_builtin_tools or responses_only_args.intersection(payload))


def _get_last_messages(
    messages: Sequence[BaseMessage],
) -> tuple[Sequence[BaseMessage], str | None]:
    """获取最后一条 `AIMessage` 之后的对话部分。

    将返回：

    1. 在最近一条具有非空 `response_metadata["id"]` 的 `AIMessage`
    之后的所有消息（可能为空列表）， 2. 该 `id`。

    If the most-recent `AIMessage` does not have an `id` (or there is no
    `AIMessage` at all) the entire conversation is returned together with `None`.
    """
    for i in range(len(messages) - 1, -1, -1):
        msg = messages[i]
        if isinstance(msg, AIMessage):
            response_id = msg.response_metadata.get("id")
            if response_id and response_id.startswith("resp_"):
                return messages[i + 1 :], response_id
                        # Responses API 无 `stop` 参数（Chat Completions
            # 有）；为避免请求被拒绝，将其删除。

    return messages, None


def _construct_responses_api_payload(
    messages: Sequence[BaseMessage], payload: dict
) -> dict:
        # 对于 Responses API 不支持 temperature 的模型，移除该参数
    for legacy_token_param in ["max_tokens", "max_completion_tokens"]:
        if legacy_token_param in payload:
            payload["max_output_tokens"] = payload.pop(legacy_token_param)
    if "reasoning_effort" in payload and "reasoning" not in payload:
        payload["reasoning"] = {"effort": payload.pop("reasoning_effort")}
        # gpt-5-chat 支持 temperature，且 reasoning.effort='none' 的 gpt-5 模型也支持
        # gpt-5-chat 支持
    payload.pop("stop", None)

        # chat api: {"type": "function", "function": {"name": "...", "description": "...",
    # "parameters": {...}, "strict": ...}}  # noqa: E501
        # responses api: {"type": "function", "name": "...", "description": "...",
    # "parameters": {...}, "strict": ...}  # noqa: E501
        # 处理部分图像（尚未支持）
    model = payload.get("model") or ""
    if (
        model.startswith("gpt-5")
        and ("chat" not in model)  # OpenAI 要求设置此参数；我们在流式传输期间忽略它。
        and (payload.get("reasoning") or {}).get("effort") != "none"
    ):
        payload.pop("temperature", None)

    payload["input"] = _construct_responses_api_input(
        messages, store=payload.get("store")
    )
    if tools := payload.pop("tools", None):
        new_tools: list = []
        for tool in tools:
                        # chat api: {"type": "function", "function": {"name": "..."}}
                        # responses api: {"type": "function", "name": "..."}
            if tool["type"] == "function" and "function" in tool:
                extra = {k: v for k, v in tool.items() if k not in ("type", "function")}
                new_tools.append({"type": "function", **tool["function"], **extra})
            else:
                if tool["type"] == "image_generation":
                                        # 结构化输出
                    if "partial_images" in tool:
                        msg = (
                            "Partial image generation is not yet supported "
                            "via the LangChain ChatOpenAI client. Please "
                            "drop the 'partial_images' key from the image_generation "
                            "tool."
                        )
                        raise NotImplementedError(msg)
                    if payload.get("stream") and "partial_images" not in tool:
                                                # 对于 pydantic + 非流式情况，我们使用 responses.parse。
                                                # 否则，我们使用 responses.create。
                        tool = {**tool, "partial_images": 1}
                    else:
                        pass

                new_tools.append(tool)

        payload["tools"] = new_tools
    if tool_choice := payload.pop("tool_choice", None):
                # JSON 模式
                # langchain-core 保留 `"index"` 键用于流式聚合。
        if (
            isinstance(tool_choice, dict)
            and tool_choice["type"] == "function"
            and "function" in tool_choice
        ):
            payload["tool_choice"] = {"type": "function", **tool_choice["function"]}
        else:
            payload["tool_choice"] = tool_choice

    if isinstance(payload.get("text"), dict):
        payload["text"] = payload["text"].copy()

        # 此处进行重命名。
    if schema := payload.pop("response_format", None):
                # chat api: {"type": "text", "text": "..."}
                # responses api: {"type": "input_text", "text": "..."}
        strict = payload.pop("strict", None)
        if not payload.get("stream") and _is_pydantic_class(schema):
            payload["text_format"] = schema
        else:
            if _is_pydantic_class(schema):
                schema_dict = schema.model_json_schema()
                strict = True
            else:
                schema_dict = schema
            if schema_dict == {"type": "json_object"}:  # chat api: {"type": "image_url", "image_url": {"url": "...", "detail": "..."}}  # noqa: E501
                if "text" in payload and isinstance(payload["text"], dict):
                    payload["text"]["format"] = {"type": "json_object"}
                else:
                    payload["text"] = {"format": {"type": "json_object"}}
            elif (
                (
                    response_format := _convert_to_openai_response_format(
                        schema_dict, strict=strict
                    )
                )
                and (isinstance(response_format, dict))
                and (response_format["type"] == "json_schema")
            ):
                format_value = {"type": "json_schema", **response_format["json_schema"]}
                if "text" in payload and isinstance(payload["text"], dict):
                    payload["text"]["format"] = format_value
                else:
                    payload["text"] = {"format": format_value}
            else:
                pass

    verbosity = payload.pop("verbosity", None)
    if verbosity is not None:
        if "text" in payload and isinstance(payload["text"], dict):
            payload["text"]["verbosity"] = verbosity
        else:
            payload["text"] = {"verbosity": verbosity}

    return payload


def _format_annotation_to_lc(annotation: dict[str, Any]) -> dict[str, Any]:
        # responses api: {"type": "input_image", "image_url": "...", "detail": "..."}  #
    # noqa: E501
        # 使用第一个 input_image 块
    if annotation.get("type") == "file_citation" and "index" in annotation:
        new_annotation = annotation.copy()
        new_annotation["file_index"] = new_annotation.pop("index")
        return new_annotation
    return annotation


def _format_annotation_from_lc(annotation: dict[str, Any]) -> dict[str, Any]:
    if annotation.get("type") == "file_citation" and "file_index" in annotation:
        new_annotation = annotation.copy()
        new_annotation["index"] = new_annotation.pop("file_index")
        return new_annotation
    return annotation


def _convert_chat_completions_blocks_to_responses(
    block: dict[str, Any],
) -> dict[str, Any]:
    """将聊天补全的内容块转换为响应 API 格式。

    仅处理文本、图像、文件块。其他类型直接通过。
    """
    if block["type"] == "text":
                # 字符串，假设为 image_url
                # 从非标准 content blocks 获取内容
        new_block = {"type": "input_text", "text": block["text"]}
        if "prompt_cache_breakpoint" in block:
            new_block["prompt_cache_breakpoint"] = block["prompt_cache_breakpoint"]
        return new_block
    if block["type"] == "image_url":
                # "name" 参数不受支持
                # 聚合单个消息的内容块
        new_block = {
            "type": "input_image",
            "image_url": block["image_url"]["url"],
        }
        if block["image_url"].get("detail"):
            new_block["detail"] = block["image_url"]["detail"]
        if "prompt_cache_breakpoint" in block:
            new_block["prompt_cache_breakpoint"] = block["prompt_cache_breakpoint"]
        return new_block
    if block["type"] == "file":
        new_block = {"type": "input_file", **block["file"]}
        if "prompt_cache_breakpoint" in block:
            new_block["prompt_cache_breakpoint"] = block["prompt_cache_breakpoint"]
        return new_block
    return block


def _ensure_valid_tool_message_content(tool_output: Any) -> str | list[dict]:
    if isinstance(tool_output, str):
        return tool_output
    if isinstance(tool_output, list) and all(
        isinstance(block, dict)
        and block.get("type")
        in (
            "input_text",
            "input_image",
            "input_file",
            "text",
            "image_url",
            "file",
        )
        for block in tool_output
    ):
        return [
            _convert_chat_completions_blocks_to_responses(block)
            for block in tool_output
        ]
    return _stringify(tool_output)


def _make_computer_call_output_from_message(
    message: ToolMessage,
) -> dict[str, Any] | None:
    computer_call_output: dict[str, Any] | None = None
    if isinstance(message.content, list):
        for block in message.content:
            if (
                message.additional_kwargs.get("type") == "computer_call_output"
                and isinstance(block, dict)
                and block.get("type") == "input_image"
            ):
                                # 防御性检查：block 可能没有 "text" 键
                computer_call_output = {
                    "call_id": message.tool_call_id,
                    "type": "computer_call_output",
                    "output": block,
                }
                break
            if (
                isinstance(block, dict)
                and block.get("type") == "non_standard"
                and block.get("value", {}).get("type") == "computer_call_output"
            ):
                computer_call_output = block["value"]
                break
    elif message.additional_kwargs.get("type") == "computer_call_output":
                # 跳过无文本内容的 block
        computer_call_output = {
            "call_id": message.tool_call_id,
            "type": "computer_call_output",
            "output": {"type": "input_image", "image_url": message.content},
        }
    if (
        computer_call_output is not None
        and "acknowledged_safety_checks" in message.additional_kwargs
    ):
        computer_call_output["acknowledged_safety_checks"] = message.additional_kwargs[
            "acknowledged_safety_checks"
        ]
    return computer_call_output


def _make_custom_tool_output_from_message(message: ToolMessage) -> dict | None:
    custom_tool_output = None
    for block in message.content:
        if isinstance(block, dict) and block.get("type") == "custom_tool_call_output":
            custom_tool_output = {
                "type": "custom_tool_call_output",
                "call_id": message.tool_call_id,
                "output": block.get("output") or "",
            }
            break
        if (
            isinstance(block, dict)
            and block.get("type") == "non_standard"
            and block.get("value", {}).get("type") == "custom_tool_call_output"
        ):
            custom_tool_output = block["value"]
            break

    return custom_tool_output


def _pop_index_and_sub_index(block: dict) -> dict:
    """流式传输时，`langchain-core` 使用 `index` 聚合文本块。

    OpenAI API 不支持此键，因此需要将其移除。
    """
    new_block = {k: v for k, v in block.items() if k != "index"}
    if "summary" in new_block and isinstance(new_block["summary"], list):
        new_summary = []
        for sub_block in new_block["summary"]:
            new_sub_block = {k: v for k, v in sub_block.items() if k != "index"}
            new_summary.append(new_sub_block)
        new_block["summary"] = new_summary
    return new_block


def _construct_responses_api_input(
    messages: Sequence[BaseMessage], *, store: bool | None = None
) -> list:
    """构造 OpenAI Responses API 的输入。

    Args:
        messages: 要序列化为 Responses API 输入项的对话历史。 store: 对应请求的 `store`
        标志，控制无状态重播行为。

            当 `False` 时，服务器不持久化响应项，因此在下一轮中无法解析先前返回的项
            ID。 因此，助手消息 ID（`msg_*`）被省略，除非推理块携带
            `encrypted_content`（可无需服务端存储进行重播）。

            当 `True` 或 `None`（默认值，与服务器默认持久化行为一致），项 ID
            和推理块会被保留。

    Returns:
        从 `messages` 派生的 Responses API 输入项列表。
    """
    input_ = []
    for lc_msg in messages:
        if isinstance(lc_msg, AIMessage):
            lc_msg = _convert_from_v03_ai_message(lc_msg)
            msg = _convert_message_to_dict(lc_msg, api="responses")
            if isinstance(msg.get("content"), list) and all(
                isinstance(block, dict) for block in msg["content"]
            ):
                tcs: list[types.ToolCall] = [
                    {
                        "type": "tool_call",
                        "name": tool_call["name"],
                        "args": tool_call["args"],
                        "id": tool_call.get("id"),
                    }
                    for tool_call in lc_msg.tool_calls
                ]
                msg["content"] = _convert_from_v1_to_responses(msg["content"], tcs)
        else:
            msg = _convert_message_to_dict(lc_msg, api="responses")
                        # 如果存在相同 ID 的 block，则追加到其后
            if isinstance(msg["content"], list):
                for i, block in enumerate(msg["content"]):
                    if isinstance(block, dict) and block.get("type") == "non_standard":
                        msg["content"][i] = block["value"]
                # 如果不存在相同 ID 的 block，则创建新 block
        if "name" in msg:
            msg.pop("name")
        if msg["role"] == "tool":
            tool_output = msg["content"]
            computer_call_output = _make_computer_call_output_from_message(
                cast(ToolMessage, lc_msg)
            )
            custom_tool_output = _make_custom_tool_output_from_message(lc_msg)  # type: ignore[arg-type]
            if computer_call_output:
                input_.append(computer_call_output)
            elif custom_tool_output:
                input_.append(custom_tool_output)
            else:
                tool_output = _ensure_valid_tool_message_content(tool_output)
                function_call_output = {
                    "type": "function_call_output",
                    "output": tool_output,
                    "call_id": msg["tool_call_id"],
                }
                input_.append(function_call_output)
        elif msg["role"] == "assistant":
            if isinstance(msg.get("content"), list):
                for block in msg["content"]:
                    if isinstance(block, dict) and (block_type := block.get("type")):
                                                # LangChain 自动生成（`lc_`）的 ID 不是
                        if block_type in ("text", "output_text", "refusal"):
                            msg_id = block.get("id")
                            phase = block.get("phase")
                            if block_type in ("text", "output_text"):
                                                                # 由提供方分配；Responses API 会拒绝
                                text = block.get("text")
                                if text is None:
                                                                        # 跳过无文本内容的块
                                    continue
                                new_block = {
                                    "type": "output_text",
                                    "text": text,
                                    "annotations": [
                                        _format_annotation_from_lc(annotation)
                                        for annotation in block.get("annotations") or []
                                    ],
                                }
                            elif block_type == "refusal":
                                new_block = {
                                    "type": "refusal",
                                    "refusal": block["refusal"],
                                }
                            for item in input_:
                                if (item_id := item.get("id")) and item_id == msg_id:
                                                                        # 如果已存在此 ID 的区块，则追加到该区块
                                    if "content" not in item:
                                        item["content"] = []
                                    item["content"].append(new_block)
                                    if phase is not None:
                                        item["phase"] = phase
                                    break
                            else:
                                                                # 如果不存在此 ID 的块，则创建一个新的块
                                new_item: dict = {
                                    "type": "message",
                                    "content": [new_block],
                                    "role": "assistant",
                                }
                                if (
                                    store is not False
                                    and msg_id
                                    and not msg_id.startswith(LC_AUTO_PREFIX)
                                ):
                                                                        # LangChain 自动生成的（`lc_`）ID 不
                                                                        # 由提供方颁发；Responses API 会拒绝
                                                                        # 它们（期望 `msg_...`），因此不要转发。
                                    new_item["id"] = msg_id
                                if phase is not None:
                                    new_item["phase"] = phase
                                input_.append(new_item)
                        elif block_type == "reasoning":
                            if store is not False or block.get("encrypted_content"):
                                input_.append(_pop_index_and_sub_index(block))
                        elif block_type in (
                            "compaction",
                            "web_search_call",
                            "file_search_call",
                            "function_call",
                            "computer_call",
                            "custom_tool_call",
                            "code_interpreter_call",
                            "mcp_call",
                            "mcp_list_tools",
                            "mcp_approval_request",
                            "tool_search_call",
                            "tool_search_output",
                            "apply_patch_call",
                            "apply_patch_call_output",
                        ):
                            input_.append(_pop_index_and_sub_index(block))
                        elif block_type == "image_generation_call":
                                                        # 先前的图像生成调用可通过 ID 引用
                            if store is not False:
                                input_.append(
                                    {"type": "image_generation_call", "id": block["id"]}
                                )
                            else:
                                                                # 仅含 ID 的引用需要存储的服务器状态。对于
                                                                # 无状态请求，重新播放完整项并丢弃无法解析的
                                                                # 裸引用，后端无法解析此类引用。
                                image_generation_call = _pop_index_and_sub_index(block)
                                if set(image_generation_call) - {"type", "id"}:
                                    input_.append(image_generation_call)
                        else:
                            pass
            elif isinstance(msg.get("content"), str):
                input_.append(
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [
                            {
                                "type": "output_text",
                                "text": msg["content"],
                                "annotations": [],
                            }
                        ],
                    }
                )

                        # 如果函数调用不存在，则从工具调用中添加函数调用
            if tool_calls := msg.pop("tool_calls", None):
                content_call_ids = {
                    block["call_id"]
                    for block in input_
                    if block.get("type") in ("function_call", "custom_tool_call")
                    and "call_id" in block
                }
                for tool_call in tool_calls:
                    if tool_call["id"] not in content_call_ids:
                        function_call = {
                            "type": "function_call",
                            "name": tool_call["function"]["name"],
                            "arguments": tool_call["function"]["arguments"],
                            "call_id": tool_call["id"],
                        }
                        input_.append(function_call)

        elif msg["role"] in ("user", "system", "developer"):
            if isinstance(msg["content"], list):
                new_blocks = []
                non_message_item_types = (
                    "mcp_approval_response",
                    "tool_search_output",
                    "apply_patch_call_output",
                    "configuration_update",
                )
                for block in msg["content"]:
                    if block["type"] in ("text", "image_url", "file"):
                        new_blocks.append(
                            _convert_chat_completions_blocks_to_responses(block)
                        )
                    elif block["type"] in ("input_text", "input_image", "input_file"):
                        new_blocks.append(block)
                    elif block["type"] in non_message_item_types:
                        input_.append(block)
                    elif block["type"] == _ADDITIONAL_TOOLS_BLOCK_TYPE:
                        if isinstance(lc_msg, SystemMessage):
                            input_.append(block)
                    elif _is_system_role(msg["role"]):
                                                # 系统内容在此处是封闭集，因此不识别的
                                                # 块是错误而非需要转发的内容。
                                                # 用户内容保持其长期存在的静默丢弃，其中
                                                # 集合是开放的，警告会成为噪声。
                        warnings.warn(
                            f"Content block {block['type']!r} was dropped from a "
                            "system message: the Responses API has no input item "
                            "of that type, so it cannot be placed in the request.",
                            UserWarning,
                            stacklevel=2,
                        )
                    else:
                        pass
                msg["content"] = new_blocks
                if msg["content"]:
                    msg["type"] = "message"
                    input_.append(msg)
            else:
                msg["type"] = "message"
                input_.append(msg)
        else:
            input_.append(msg)

    return input_


def _get_output_text(response: Response) -> str:
    """安全地提取输出文本。

    Context: 上下文：OpenAI SDK 在 `1.99.2` 版本中短暂删除了 `response.output_text`。
    """
    if hasattr(response, "output_text"):
        return response.output_text
    texts = [
        content.text
        for output in response.output
        if output.type == "message"
        for content in output.content
        if content.type == "output_text"
    ]
    return "".join(texts)


def _construct_lc_result_from_responses_api(
    response: Response,
    schema: type[_BM] | None = None,
    metadata: dict | None = None,
    output_version: str | None = None,
) -> ChatResult:
    """从 OpenAI Response API 响应构建 `ChatResponse`。"""
    if response.error:
        raise ValueError(response.error)

    if output_version is None:
                # None 作为哨兵值，让我们知道是否显式设置了 output_version。
                # 显式设置 `output_version="responses/v1"` 可单独启用
                # Responses API。
        output_version = "responses/v1"

    response_metadata = {
        k: v
                # warnings=False 是因为 https://github.com/openai/openai-python/issues/2872
        for k, v in response.model_dump(
            exclude_none=True, mode="json", warnings=False
        ).items()
        if k
        in (
            "created_at",
                        # 向后兼容性：在 response_metadata 中保留 response ID，同时也在
                        # 顶层-id 中保留
            "id",
            "incomplete_details",
            "metadata",
            "object",
            "status",
            "user",
            "model",
            "service_tier",
        )
    }
    if metadata:
        response_metadata.update(metadata)
        # 以兼容聊天完成调用。
    response_metadata["model_provider"] = "openai"
    response_metadata["model_name"] = response_metadata.get("model")
    if response.usage:
        usage_metadata = _create_usage_metadata_responses(
            response.usage.model_dump(), response.service_tier
        )
    else:
        usage_metadata = None

    content_blocks: list = []
    tool_calls = []
    invalid_tool_calls = []
    additional_kwargs: dict = {}
    for output in response.output:
        if output.type == "message":
            phase = getattr(output, "phase", None)
            for content in output.content:
                if content.type == "output_text":
                    block = {
                        "type": "text",
                        "text": content.text,
                        "annotations": [
                            _format_annotation_to_lc(annotation.model_dump())
                            for annotation in content.annotations
                        ]
                        if isinstance(content.annotations, list)
                        else [],
                        "id": output.id,
                    }
                    if phase is not None:
                        block["phase"] = phase
                    content_blocks.append(block)
                    if hasattr(content, "parsed"):
                        additional_kwargs["parsed"] = content.parsed
                if content.type == "refusal":
                    refusal_block = {
                        "type": "refusal",
                        "refusal": content.refusal,
                        "id": output.id,
                    }
                    if phase is not None:
                        refusal_block["phase"] = phase
                    content_blocks.append(refusal_block)
        elif output.type == "function_call":
            function_call_block = output.model_dump(exclude_none=True, mode="json")
                        # SDK 将保留字命名为 `async_`；内容块携带
                        # 网络名称，以便在下一次请求中往返传输。
            if "async_" in function_call_block:
                function_call_block["async"] = function_call_block.pop("async_")
            content_blocks.append(function_call_block)
            try:
                args = json.loads(output.arguments, strict=False)
                error = None
            except JSONDecodeError as e:
                args = output.arguments
                error = str(e)
            if error is None:
                tool_call = {
                    "type": "tool_call",
                    "name": output.name,
                    "args": args,
                    "id": output.call_id,
                }
                tool_calls.append(tool_call)
            else:
                tool_call = {
                    "type": "invalid_tool_call",
                    "name": output.name,
                    "args": args,
                    "id": output.call_id,
                    "error": error,
                }
                invalid_tool_calls.append(tool_call)
        elif output.type == "custom_tool_call":
            content_blocks.append(output.model_dump(exclude_none=True, mode="json"))
            tool_call = {
                "type": "tool_call",
                "name": output.name,
                "args": {"__arg1": output.input},
                "id": output.call_id,
            }
            tool_calls.append(tool_call)
        elif output.type in (
            "reasoning",
            "compaction",
            "web_search_call",
            "file_search_call",
            "computer_call",
            "code_interpreter_call",
            "mcp_call",
            "mcp_list_tools",
            "mcp_approval_request",
            "image_generation_call",
            "tool_search_call",
            "tool_search_output",
            "apply_patch_call",
            "apply_patch_call_output",
        ):
            content_blocks.append(output.model_dump(exclude_none=True, mode="json"))

        # 用于处理流式情况下结构化输出的变通方法。
        #    from openai import OpenAI
        #    from pydantic import BaseModel

        #    class Foo(BaseModel):
        #        response: str

        #    client = OpenAI()

        #    client.responses.parse(
        #        model="...",
        #        input=[{"content": "how are ya", "role": "user"}],
        #        text_format=Foo,
        #        stream=True,  # <-- 错误
    #    )
    output_text = _get_output_text(response)
    if (
        schema is not None
        and "parsed" not in additional_kwargs
        and output_text  # 工具调用可能生成空输出文本
        and response.text
        and (text_config := response.text.model_dump())
        and (format_ := text_config.get("format", {}))
        and (format_.get("type") == "json_schema")
    ):
        try:
            parsed_dict = json.loads(output_text)
            if schema and _is_pydantic_class(schema):
                parsed = schema(**parsed_dict)
            else:
                parsed = parsed_dict
            additional_kwargs["parsed"] = parsed
        except json.JSONDecodeError:
            pass

    message = AIMessage(
        content=content_blocks,
        id=response.id,
        usage_metadata=usage_metadata,
        response_metadata=response_metadata,
        additional_kwargs=additional_kwargs,
        tool_calls=tool_calls,
        invalid_tool_calls=invalid_tool_calls,
    )
    if output_version == "v0":
        message = _convert_to_v03_ai_message(message)

    return ChatResult(generations=[ChatGeneration(message=message)])


def _coerce_chunk_response(resp: Any) -> Any:
        # 在流事件中观察到 dict `response` 项
    if isinstance(resp, dict):
        from openai.types.responses import Response

        try:
            return Response.model_validate(resp)
        except ValidationError as e:
                        # API 有时会领先于已安装 SDK 的 Literal
                        # 声明。回退到非验证构造，使流仍能完成，并暴露漂移情况，以便操作员升级。
                        # 内容中的索引
            logger.warning(
                "OpenAI Responses payload failed SDK validation "
                "(response id=%s); falling back to non-validating construct. "
                "This usually means the OpenAI API has drifted ahead of the "
                "installed `openai` package. Details: %s",
                resp.get("id"),
                e,
            )
            return Response.model_construct(**resp)
    return resp


def _convert_responses_chunk_to_generation_chunk(
    chunk: Any,
    current_index: int,  # Response 输出中的索引
    current_output_index: int,  # 输出项中内容块的索引
    current_sub_index: int,  # item id -> `output_index`
    schema: type[_BM] | None = None,
    metadata: dict | None = None,
    has_reasoning: bool = False,
    output_version: str | None = None,
    item_output_indices: dict[str, int] | None = None,  # 在多次内置工具调用后，API 可能在某项的 `done` 事件中报告比 `added` 事件更高的 `output_index`。将每个项目固定到首次看到的 `output_index`，确保其事件始终落在同一内容块上。
) -> tuple[int, int, int, ChatGenerationChunk | None]:
        # None 作为哨兵值，让我们知道是否显式设置了 output_version。
        # 显式设置 `output_version="responses/v1"` 可单独启用
        # Responses API。
    item_id = getattr(getattr(chunk, "item", None), "id", None) or getattr(
        chunk, "item_id", None
    )
    if (
        item_output_indices is not None
        and item_id is not None
        and (stream_output_index := getattr(chunk, "output_index", None)) is not None
    ):
        item_output_indices.setdefault(item_id, stream_output_index)

    def _advance(output_idx: int, sub_idx: int | None = None) -> None:
        """流式处理期间跟踪的索引。

        Example: we stream a response item of the form:

        ```python
        {
            "type": "message",  # output_index 0
            "role": "assistant",
            "id": "msg_123",
            "content": [
                {"type": "output_text", "text": "foo"},  # sub_index 0
                {"type": "output_text", "text": "bar"},  # sub_index 1
            ],
        }
        ```

        This is a single item with a shared `output_index` and two sub-indexes, one
        for each content block.

        这将被处理为包含两个文本块的 `AIMessage`：

        ```python
        AIMessage(
            [
                {"type": "text", "text": "foo", "id": "msg_123"},  # index 0
                {"type": "text", "text": "bar", "id": "msg_123"},  # index 1
            ]
        )
        ```

        此函数仅识别输出或子索引中的更新，并相应地递增当前索引。
        """
        nonlocal current_index, current_output_index, current_sub_index
        if item_output_indices is not None and item_id in item_output_indices:
            output_idx = item_output_indices[item_id]
        if sub_idx is None:
            if current_output_index != output_idx:
                current_index += 1
        else:
            if (current_output_index != output_idx) or (current_sub_index != sub_idx):
                current_index += 1
            current_sub_index = sub_idx
        current_output_index = output_idx

    if output_version is None:
                # 似乎在 openai==1.82.0 中是一个破坏性变更
                # 向后兼容性
                # SDK 将保留字暴露为 `async_`；旧版本则保留在 extras 中。
        output_version = "responses/v1"

    content = []
    tool_call_chunks: list = []
    additional_kwargs: dict = {}
    response_metadata = metadata or {}
    response_metadata["model_provider"] = "openai"
    usage_metadata = None
    chunk_position: Literal["last"] | None = None
    id = None
    if chunk.type == "response.output_text.delta":
        _advance(chunk.output_index, chunk.content_index)
        content.append({"type": "text", "text": chunk.delta, "index": current_index})
    elif chunk.type == "response.output_text.annotation.added":
        _advance(chunk.output_index, chunk.content_index)
        if isinstance(chunk.annotation, dict):
                        # 加密内容仅在对应的 `done` 事件中才完整，该事件
            annotation = chunk.annotation
        else:
            annotation = chunk.annotation.model_dump(exclude_none=True, mode="json")

        content.append(
            {
                "type": "text",
                "annotations": [_format_annotation_to_lc(annotation)],
                "index": current_index,
            }
        )
    elif chunk.type == "response.output_text.done":
        _advance(chunk.output_index, chunk.content_index)
        content.append(
            {
                "type": "text",
                "text": "",
                "id": chunk.item_id,
                "index": current_index,
            }
        )
    elif chunk.type == "response.created":
        response = _coerce_chunk_response(chunk.response)
        id = response.id
        response_metadata["id"] = response.id  # 在下方发出。在此处丢弃它，防止两个事件合并成重复的
    elif chunk.type == "response.failed":
        response = _coerce_chunk_response(chunk.response)
        error_msg = str(response.error or f"Response {response.id} failed.")
        raise ValueError(error_msg)
    elif chunk.type == "error":
        error_msg = f"{chunk.code}: {chunk.message}" if chunk.code else chunk.message
        raise ValueError(error_msg)
    elif chunk.type in ("response.completed", "response.incomplete"):
        response = _coerce_chunk_response(chunk.response)
        msg = cast(
            AIMessage,
            (
                _construct_lc_result_from_responses_api(
                    response, schema=schema, output_version=output_version
                )
                .generations[0]
                .message
            ),
        )
        if parsed := msg.additional_kwargs.get("parsed"):
            additional_kwargs["parsed"] = parsed
        usage_metadata = msg.usage_metadata
        response_metadata = {
            k: v for k, v in msg.response_metadata.items() if k != "id"
        }
        chunk_position = "last"
    elif chunk.type == "response.output_item.added" and chunk.item.type == "message":
        if output_version == "v0":
            id = chunk.item.id
        elif phase := getattr(chunk.item, "phase", None):
            _advance(chunk.output_index, 0)
            content.append(
                {
                    "type": "text",
                    "text": "",
                    "phase": phase,
                    "index": current_index,
                }
            )
        else:
            pass
    elif (
        chunk.type == "response.output_item.added"
        and chunk.item.type == "function_call"
    ):
        _advance(chunk.output_index)
        tool_call_chunks.append(
            {
                "type": "tool_call_chunk",
                "name": chunk.item.name,
                "args": chunk.item.arguments,
                "id": chunk.item.call_id,
                "index": current_index,
            }
        )
        function_call_content: dict = {
            "type": "function_call",
            "name": chunk.item.name,
            "arguments": chunk.item.arguments,
            "call_id": chunk.item.call_id,
            "id": chunk.item.id,
            "index": current_index,
        }
        if getattr(chunk.item, "namespace", None) is not None:
            function_call_content["namespace"] = chunk.item.namespace
                # 字符串（共享 `index` 的块通过拼接合并）。
        async_flag = getattr(chunk.item, "async_", None)
        if async_flag is None:
            async_flag = getattr(chunk.item, "async", None)
        if async_flag is not None:
            function_call_content["async"] = async_flag
        content.append(function_call_content)
    elif chunk.type == "response.output_item.done" and chunk.item.type in (
        "compaction",
        "web_search_call",
        "file_search_call",
        "computer_call",
        "code_interpreter_call",
        "mcp_call",
        "mcp_list_tools",
        "mcp_approval_request",
        "image_generation_call",
        "tool_search_call",
        "tool_search_output",
        "apply_patch_call",
        "apply_patch_call_output",
    ):
        _advance(chunk.output_index)
        tool_output = chunk.item.model_dump(exclude_none=True, mode="json")
        tool_output["index"] = current_index
        content.append(tool_output)
    elif (
        chunk.type == "response.output_item.done"
        and chunk.item.type == "custom_tool_call"
    ):
        _advance(chunk.output_index)
        tool_output = chunk.item.model_dump(exclude_none=True, mode="json")
        tool_output["index"] = current_index
        content.append(tool_output)
        tool_call_chunks.append(
            {
                "type": "tool_call_chunk",
                "name": chunk.item.name,
                "args": json.dumps({"__arg1": chunk.item.input}),
                "id": chunk.item.call_id,
                "index": current_index,
            }
        )
    elif chunk.type == "response.function_call_arguments.delta":
        _advance(chunk.output_index)
        tool_call_chunks.append(
            {"type": "tool_call_chunk", "args": chunk.delta, "index": current_index}
        )
        content.append(
            {"type": "function_call", "arguments": chunk.delta, "index": current_index}
        )
    elif chunk.type == "response.refusal.done":
        content.append({"type": "refusal", "refusal": chunk.refusal})
    elif chunk.type == "response.output_item.added" and chunk.item.type == "reasoning":
        _advance(chunk.output_index)
        current_sub_index = 0
        reasoning = chunk.item.model_dump(exclude_none=True, mode="json")
                # langchain-core 使用 `index` 键来聚合文本块。
                # 部分图像暂不支持。
                # 字符串（具有相同 `index` 的块将通过拼接合并）。
        reasoning.pop("encrypted_content", None)
        reasoning["index"] = current_index
        content.append(reasoning)
    elif (
        chunk.type == "response.output_item.done"
        and chunk.item.type == "reasoning"
        and chunk.item.encrypted_content
    ):
        _advance(chunk.output_index)
        content.append(
            {
                "type": "reasoning",
                "id": chunk.item.id,
                "summary": [],
                "encrypted_content": chunk.item.encrypted_content,
                "index": current_index,
            }
        )
    elif chunk.type == "response.reasoning_summary_part.added":
        _advance(chunk.output_index)
        content.append(
            {
                                # langchain-core 使用 `index` 键来聚合文本块。
                "summary": [
                    {"index": chunk.summary_index, "type": "summary_text", "text": ""}
                ],
                "index": current_index,
                "type": "reasoning",
                "id": chunk.item_id,
            }
        )
    elif chunk.type == "response.image_generation_call.partial_image":
                # 部分图像暂不支持。
        pass
    elif chunk.type == "response.reasoning_summary_text.delta":
        _advance(chunk.output_index)
        content.append(
            {
                "summary": [
                    {
                        "index": chunk.summary_index,
                        "type": "summary_text",
                        "text": chunk.delta,
                    }
                ],
                "index": current_index,
                "type": "reasoning",
            }
        )
    else:
        return current_index, current_output_index, current_sub_index, None

    message = AIMessageChunk(
        content=content,  # type: ignore[arg-type]
        tool_call_chunks=tool_call_chunks,
        usage_metadata=usage_metadata,
        response_metadata=response_metadata,
        additional_kwargs=additional_kwargs,
        id=id,
        chunk_position=chunk_position,
    )
    if output_version == "v0":
        message = cast(
            AIMessageChunk,
            _convert_to_v03_ai_message(message, has_reasoning=has_reasoning),
        )

    return (
        current_index,
        current_output_index,
        current_sub_index,
        ChatGenerationChunk(message=message),
    )

"""Perplexity API 的封装。"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator, Callable, Iterator, Mapping, Sequence
from operator import itemgetter
from typing import Any, Literal, TypeAlias, cast

from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models import (
    LanguageModelInput,
    ModelProfile,
    ModelProfileRegistry,
)
from langchain_core.language_models.chat_models import (
    BaseChatModel,
    agenerate_from_stream,
    generate_from_stream,
)
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    BaseMessageChunk,
    ChatMessage,
    ChatMessageChunk,
    FunctionMessageChunk,
    HumanMessage,
    HumanMessageChunk,
    SystemMessage,
    SystemMessageChunk,
    ToolMessage,
    ToolMessageChunk,
)
from langchain_core.messages.ai import (
    InputTokenDetails,
    OutputTokenDetails,
    UsageMetadata,
    subtract_usage,
)
from langchain_core.messages.tool import tool_call_chunk
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.runnables import Runnable, RunnableMap, RunnablePassthrough
from langchain_core.tools import BaseTool
from langchain_core.utils import get_pydantic_field_names, secret_from_env
from langchain_core.utils.function_calling import (
    convert_to_json_schema,
    convert_to_openai_tool,
)
from langchain_core.utils.pydantic import is_basemodel_subclass
from perplexity import AsyncPerplexity, Perplexity
from pydantic import BaseModel, ConfigDict, Field, SecretStr, model_validator
from typing_extensions import Self

from langchain_perplexity._version import __version__
from langchain_perplexity.data._profiles import _PROFILES
from langchain_perplexity.output_parsers import (
    ReasoningJsonOutputParser,
    ReasoningStructuredOutputParser,
)
from langchain_perplexity.types import MediaResponse, WebSearchOptions

_DictOrPydanticClass: TypeAlias = dict[str, Any] | type[BaseModel]
_DictOrPydantic: TypeAlias = dict | BaseModel

logger = logging.getLogger(__name__)


_MODEL_PROFILES = cast("ModelProfileRegistry", _PROFILES)


def _get_default_model_profile(model_name: str) -> ModelProfile:
    default = _MODEL_PROFILES.get(model_name) or {}
    return default.copy()


def _is_pydantic_class(obj: Any) -> bool:
    return isinstance(obj, type) and is_basemodel_subclass(obj)


def _create_usage_metadata(token_usage: dict) -> UsageMetadata:
    """从 Perplexity token 使用数据创建 UsageMetadata。

    Args:
        token_usage: 包含来自 Perplexity API 的 token 使用信息的字典。

    Returns:
        具有正确结构的 token 数量和详细信息的 UsageMetadata。
    """
    input_tokens = token_usage.get("prompt_tokens", 0)
    output_tokens = token_usage.get("completion_tokens", 0)
    total_tokens = token_usage.get("total_tokens", input_tokens + output_tokens)

        # 构建 output_token_details 以支持 Perplexity 特有的字段
    output_token_details: OutputTokenDetails = {}
    if (reasoning := token_usage.get("reasoning_tokens")) is not None:
        output_token_details["reasoning"] = reasoning
    if (citation_tokens := token_usage.get("citation_tokens")) is not None:
        output_token_details["citation_tokens"] = citation_tokens  # type: ignore[typeddict-unknown-key]

    return UsageMetadata(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        output_token_details=output_token_details,
    )


_RESPONSES_ONLY_ARGS = frozenset(
    {"include", "input", "instructions", "previous_response_id"}
)
"""Top-level keys that exist only on Perplexity's Agent (Responses) API.

The presence of any of these triggers auto-routing through Responses, since
the Chat Completions endpoint would silently reject them.
"""

_RESPONSES_PASSTHROUGH_KEYS = frozenset(
    {
        "model",
        "models",
        "tools",
        "instructions",
        "language_preference",
        "max_steps",
        "preset",
        "reasoning",
        "response_format",
        "stream",
        "extra_body",
        "extra_headers",
        "extra_query",
        "timeout",
    }
)
"""Keys the Perplexity Responses SDK accepts natively.

Mirrors `perplexity.resources.responses.ResponsesResource.create`. Anything
outside this set (other than known renames and drops) is routed through
`extra_body` so the SDK forwards it without breaking strict typing.
"""

_RESPONSES_DROP_KEYS = frozenset({"temperature", "top_p", "top_k", "stop", "metadata"})
"""Chat-Completions-only sampling/control knobs the Responses (Agent) API does
not accept.

Forwarding them would raise `TypeError` from the typed SDK signature in
`perplexity.resources.responses.ResponsesResource.create`, so they are dropped
at the boundary. Every drop emits a `WARNING`-level log on each call, except
the class-default `temperature`, which is suppressed because `_default_params`
injects `self.temperature` on every call regardless of user intent. A
user-supplied `temperature` (via init, `invoke(temperature=...)`, or `.bind`)
still warns.

`tool_choice` is *not* in this set: it is a control-flow primitive
(forced/required tool selection) and is rejected with `ValueError` rather than
silently dropped, since downstream agent loops cannot recover.
"""


def _is_builtin_tool(tool: dict) -> bool:
    """如果 `tool` 是 Responses-API 内置（非 `function`）工具，则返回 True。

    Perplexity 的 Agent API 提供内置工具（例如 `web_search`、
    `code_interpreter`），这些工具通过 `type` 值而非 `"function"` 进行标识。Chat
    Completions 仅接受 function 工具，因此任何未通过此检查的工具都会强制使用 Responses
    路由。
    """
    return "type" in tool and tool["type"] != "function"


def _flatten_responses_tool(tool: dict) -> dict:
    """将 Chat-Completions 函数工具（嵌套在 `function` 下）展平为 Responses-API
的扁平格式。内置工具（例如 `web_search`）保持不变。
    """
    if tool.get("type") == "function" and isinstance(tool.get("function"), dict):
        fn = tool["function"]
        flat: dict[str, Any] = {"type": "function", "name": fn.get("name")}
        for key in ("description", "parameters", "strict"):
            if key in fn:
                flat[key] = fn[key]
        return flat
    return tool


def _content_to_text(content: Any) -> str:
    """Concatenate text from a string or list-of-blocks content, dropping
    non-text blocks (e.g. a `tool_call`/`tool_use` block) that the Responses API
    can't take on a tool turn.

    此处仅构建助手工具回合的可选纯文本前言；调用本身由 `_translate_responses_input`
    重新生成为 `function_call` 项，因此不会丢失任何可操作内容。
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        return "".join(parts)
    if content is not None:
                # 如果内容形状意外（非
        # str/list/None），则直接丢弃而非猜测；记录该情况以便诊断内容形状漂移问题。
                # 助手文本（如有）变为普通消息；后续调用作为 `function_call` 项处理。
        logger.debug("Dropping unexpected content type %s on tool turn.", type(content))
    return ""


def _translate_responses_input(message_dicts: list[dict[str, Any]]) -> list[Any]:
    """将 Chat-Completions 消息字典转换为 Responses-API 输入项。

    Responses API 没有 `tool` 角色：助手回合的 `tool_calls` 变为 `function_call` 项，而
    `tool` 消息变为 `function_call_output`。其他消息直接通过。

    `name`, `id`, and `tool_call_id` are the fields that pair a call with its
    result; `_convert_message_to_dict` always populates them, so a missing one
    here signals upstream drift or a hand-built message and is logged at
    `WARNING` rather than silently coerced.
    """
    translated: list[Any] = []
    for message in message_dicts:
        if not isinstance(message, dict):
            translated.append(message)
            continue
        role = message.get("role")
        if role == "assistant" and message.get("tool_calls"):
                        # Responses API
            # 将整个函数调用作为一个条目交付（无参数增量），因此将其作为单个工具调用块发出。
                        # `response.failed` 是 SDK 的标准事件名称；`response.error` 作为备用，在
            # API 传输期间出现时仍可使用。
            text = _content_to_text(message.get("content"))
            if text:
                translated.append(
                    {"type": "message", "role": "assistant", "content": text}
                )
            for tool_call in message["tool_calls"]:
                function = tool_call.get("function", {})
                call_id = tool_call.get("id")
                name = function.get("name", "")
                if not name or not call_id:
                    logger.warning(
                        "Assistant tool_call missing identity field "
                        "(name=%r, id=%r); the Responses API may reject this "
                        "turn or fail to pair the call with its output.",
                        name,
                        call_id,
                    )
                translated.append(
                    {
                        "type": "function_call",
                        "call_id": call_id,
                        "name": name,
                        "arguments": function.get("arguments", "") or "",
                    }
                )
        elif role == "tool":
            content = message.get("content", "")
            output = content if isinstance(content, str) else json.dumps(content)
            call_id = message.get("tool_call_id")
            if not call_id:
                logger.warning(
                    "Tool message missing tool_call_id; the Responses API "
                    "cannot pair this function_call_output with its call."
                )
            translated.append(
                {
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": output,
                }
            )
        elif role in {"assistant", "system", "user", "developer"}:
            translated.append({**message, "type": "message"})
        else:
            translated.append(message)
    return translated


def _use_responses_api(payload: dict) -> bool:
    """确定是否将负载路由到 Responses API。

    The Agent (Responses) API is required for built-in tools and accepts
    fields that Chat Completions would reject — so callers must be routed
    there transparently when those signals appear.

    Returns True if the payload contains a built-in tool (any element of
    `tools` whose `type` is not `"function"`) or any Responses-only field
    (`input`, `include`, `instructions`, `previous_response_id`).
    """
    uses_builtin_tools = "tools" in payload and any(
        _is_builtin_tool(tool) for tool in payload["tools"]
    )
    matched_fields = _RESPONSES_ONLY_ARGS.intersection(payload)
    if uses_builtin_tools or matched_fields:
        reason = (
            "payload contains a built-in tool (Chat Completions accepts only "
            "function tools)"
            if uses_builtin_tools
            else (
                f"payload sets Responses-only field(s) {sorted(matched_fields)} "
                "(Chat Completions would reject these)"
            )
        )
        logger.debug(
            "Routing through Perplexity Responses API: %s. "
            "Set use_responses_api=False to force Chat Completions.",
            reason,
        )
        return True
    return False


def _set_model_name_alias(response_metadata: dict[str, Any]) -> None:
    """Mirror `model` into `model_name`, which langchain-core usage callbacks
    read for cost tracking (the Chat Completions path already sets it).
    """
    if "model" in response_metadata:
        response_metadata["model_name"] = response_metadata["model"]


def _get_attr(obj: Any, name: str, default: Any = None) -> Any:
    """安全地从 SDK 对象或字典中获取属性。

    Responses SDK 负载以 Pydantic 类似 SDK
    对象（服务器响应）或普通字典（调用方提前序列化或测试时传递）形式到达。此辅助函数统一两种形态，使模块其余部分无需特殊处理。
    """
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _convert_responses_usage(usage: Any) -> UsageMetadata | None:
    """从 Responses API 使用负载构建 `UsageMetadata`。

    Returns `None` if `usage` itself is missing or if either token field is
    absent — emitting zeroed `UsageMetadata` would silently undercount usage
    in downstream cost dashboards.

    Cache hits and cache writes reported under `input_tokens_details` are mapped
    onto the standard `InputTokenDetails` slots so downstream consumers can tell
    cached input from fresh input.
    """
    if usage is None:
        return None
    input_tokens = _get_attr(usage, "input_tokens", None)
    output_tokens = _get_attr(usage, "output_tokens", None)
    if input_tokens is None or output_tokens is None:
        return None
    total_tokens = _get_attr(usage, "total_tokens", None)
    if total_tokens is None:
        total_tokens = input_tokens + output_tokens

    input_token_details: InputTokenDetails = {}
    details = _get_attr(usage, "input_tokens_details", None)
    if details is not None:
        cache_read = _get_attr(details, "cache_read_input_tokens", None)
        if cache_read is not None:
            input_token_details["cache_read"] = cache_read
        cache_creation = _get_attr(details, "cache_creation_input_tokens", None)
        if cache_creation is not None:
            input_token_details["cache_creation"] = cache_creation

    return UsageMetadata(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        input_token_details=input_token_details,
    )


def _extract_responses_text(response: Any) -> str:
    """从 Responses API 响应中提取助手文本内容。

    优先使用 `response.output_text`，否则遍历 `output[*].content[*].text`。
    """
    text = _get_attr(response, "output_text", None)
    if isinstance(text, str) and text:
        return text
    output = _get_attr(response, "output", None) or []
    parts: list[str] = []
    for item in output:
        item_type = _get_attr(item, "type", None)
        if item_type and item_type != "message":
            continue
        content_blocks = _get_attr(item, "content", None) or []
        for block in content_blocks:
            block_text = _get_attr(block, "text", None)
            if isinstance(block_text, str):
                parts.append(block_text)
    return "".join(parts)


def _convert_responses_to_chat_result(response: Any) -> ChatResult:
    """将 Responses API 响应对象转换为 `ChatResult`。

    Maps `output_text`/`output[*].content[*].text` to `AIMessage.content` and
    surfaces `function_call` items as `tool_calls`. Perplexity-specific fields
    (`citations`, `images`, `related_questions`, `search_results`, `videos`,
    `reasoning_steps`) are placed on `additional_kwargs` to match the shape
    produced by the Chat Completions branch, while transport-level fields
    (`id`, `model`, `status`, `object`) land on `response_metadata`.
    """
    content = _extract_responses_text(response)

    tool_calls: list[dict[str, Any]] = []
    output = _get_attr(response, "output", None) or []
    for item in output:
        item_type = _get_attr(item, "type", None)
        if item_type == "function_call":
            raw_args = _get_attr(item, "arguments", "") or ""
            try:
                parsed_args = json.loads(raw_args) if raw_args else {}
            except (TypeError, ValueError):
                logger.warning(
                    "Failed to parse Perplexity function_call arguments as JSON "
                    "for tool %r; preserving raw payload under __raw_arguments__.",
                    _get_attr(item, "name", ""),
                    exc_info=True,
                )
                parsed_args = {"__raw_arguments__": raw_args}
            tool_calls.append(
                {
                    "name": _get_attr(item, "name", ""),
                    "args": parsed_args,
                    "id": _get_attr(item, "call_id", None)
                    or _get_attr(item, "id", None),
                    "type": "tool_call",
                }
            )
        elif item_type and item_type != "message":
            logger.debug("Ignoring unhandled Responses output item type: %s", item_type)

    usage_metadata = _convert_responses_usage(_get_attr(response, "usage", None))

    additional_kwargs: dict[str, Any] = {}
    for key in (
        "citations",
        "images",
        "related_questions",
        "search_results",
        "videos",
        "reasoning_steps",
    ):
        value = _get_attr(response, key, None)
        if value:
            additional_kwargs[key] = value

    response_metadata: dict[str, Any] = {}
    for key in ("id", "model", "status", "object"):
        value = _get_attr(response, key, None)
        if value is not None:
            response_metadata[key] = value
    _set_model_name_alias(response_metadata)

    message = AIMessage(
        content=content,
        additional_kwargs=additional_kwargs,
        tool_calls=tool_calls,  # type: ignore[arg-type]
        usage_metadata=usage_metadata,
        response_metadata=response_metadata,
    )
    return ChatResult(generations=[ChatGeneration(message=message)])


class PerplexityResponsesStreamError(RuntimeError):
    """当 Perplexity Responses（智能体）API 流在中途失败时抛出。

    携带 API
    表面的结构化错误字段（`code`、`type`、`param`、`request_id`）以及原始事件负载，以便可观测性管道可以程序化地检查它们，而不是通过正则表达式解析消息字符串。
    """

    def __init__(
        self,
        message: str,
        *,
        code: str | None = None,
        error_type: str | None = None,
        param: str | None = None,
        request_id: str | None = None,
        raw_event: Any = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.error_type = error_type
        self.param = param
        self.request_id = request_id
        self.raw_event = raw_event


def _convert_responses_stream_event_to_chunk(
    event: Any,
) -> ChatGenerationChunk | None:
    """将 Responses API 流式事件转换为 `ChatGenerationChunk`。

    Handles `response.output_text.delta` (text chunk), `response.output_item.done`
    carrying a `function_call` (surfaced as a tool-call chunk), `response.completed`
    (final usage + metadata), and `response.failed` / `response.error`
    (raises `PerplexityResponsesStreamError`). Returns `None` for any other
    event type; unrecognized event types are logged at `DEBUG` so SDK drift is
    diagnosable without flooding logs.
    """
    event_type = _get_attr(event, "type", None)
    if event_type == "response.output_text.delta":
        delta = _get_attr(event, "delta", "") or ""
        return ChatGenerationChunk(message=AIMessageChunk(content=delta))
    if event_type == "response.output_item.done":
        item = _get_attr(event, "item", None)
        if item is not None and _get_attr(item, "type", None) == "function_call":
                        # 若无此分支，流式传输中服务器端失败会导致零个块返回，并在
            # `BaseChatModel.stream` 中表现为“未返回任何生成块”，掩盖真实错误。
                        # OpenAI 兼容 API 会拒绝空字符串内容和 tool_calls；应发送 `null`
            # 而非空字符串。
            return ChatGenerationChunk(
                message=AIMessageChunk(
                    content="",
                    tool_call_chunks=[
                        tool_call_chunk(
                            name=_get_attr(item, "name", None),
                            args=_get_attr(item, "arguments", None),
                            id=_get_attr(item, "call_id", None)
                            or _get_attr(item, "id", None),
                            index=_get_attr(event, "output_index", 0),
                        )
                    ],
                )
            )
        return None
    if event_type == "response.completed":
        response = _get_attr(event, "response", None)
        usage_metadata = _convert_responses_usage(_get_attr(response, "usage", None))
        response_metadata: dict[str, Any] = {}
        additional_kwargs: dict[str, Any] = {}
        if response is not None:
            for key in ("id", "model", "status", "object"):
                value = _get_attr(response, key, None)
                if value is not None:
                    response_metadata[key] = value
            _set_model_name_alias(response_metadata)
            for key in (
                "citations",
                "images",
                "related_questions",
                "search_results",
                "videos",
                "reasoning_steps",
            ):
                value = _get_attr(response, key, None)
                if value:
                    additional_kwargs[key] = value
        return ChatGenerationChunk(
            message=AIMessageChunk(
                content="",
                additional_kwargs=additional_kwargs,
                usage_metadata=usage_metadata,
                response_metadata=response_metadata,
            )
        )
    if event_type in ("response.failed", "response.error"):
                # 收集被丢弃的值，以便警告信息能指出具体项。
                # 对类默认值 `temperature` 抑制警告，因为 `_default_params`
        # 在每次调用时注入该值，否则会频繁打扰从未主动请求它的用户。
                # 函数工具必须展平为 Responses-API 格式；内置工具（如 web_search
        # 等）保持不变通过。
                # 未知或 Perplexity 特有的键：路由至 extra_body 以便后续处理。
                # 来自 `BaseChatModel.stream`，掩盖了真实错误。
        error = _get_attr(event, "error", None)
        message = (
            _get_attr(error, "message", None)
            if error is not None
            else _get_attr(event, "message", None)
        ) or "Perplexity Responses API stream error"
        code = _get_attr(error, "code", None) if error is not None else None
        error_type = _get_attr(error, "type", None) if error is not None else None
        param = _get_attr(error, "param", None) if error is not None else None
        request_id = _get_attr(event, "request_id", None)
        details: list[str] = []
        for label, value in (
            ("code", code),
            ("type", error_type),
            ("param", param),
            ("request_id", request_id),
        ):
            if value is not None:
                details.append(f"{label}={value}")
        if details:
            message = f"{message} ({', '.join(details)})"
        logger.error(
            "Perplexity Responses stream failure: %s",
            message,
            extra={
                "perplexity_error_code": code,
                "perplexity_error_type": error_type,
                "perplexity_error_param": param,
                "perplexity_request_id": request_id,
            },
        )
        raise PerplexityResponsesStreamError(
            message,
            code=code,
            error_type=error_type,
            param=param,
            request_id=request_id,
            raw_event=event,
        )
    logger.debug("Ignoring unhandled Perplexity stream event type: %s", event_type)
    return None


class ChatPerplexity(BaseChatModel):
    """`Perplexity AI` 聊天模型 API。

    Setup:
        To use, you should have the environment variable `PPLX_API_KEY` set to your API key.
        Any parameters that are valid to be passed to the perplexity.create call
        can be passed in, even if not explicitly saved on this class.

        ```bash
        export PPLX_API_KEY=your_api_key
        ```

        关键初始化参数 - 完成参数：
            model:
                要使用的模型名称。例如 "sonar"
            temperature:
                使用的采样温度。
            max_tokens:
                生成的最大 token 数量。
            streaming:
                是否流式输出结果。

        关键初始化参数 - 客户端参数：
            pplx_api_key:
                PerplexityChat API 的 API 密钥。
            request_timeout:
                对 PerplexityChat 完成 API 请求的超时时间。
            max_retries:
                生成时最大重试次数。

        有关支持的初始化参数及其描述的完整列表，请参见参数部分。

        实例化：

        ```python
        from langchain_perplexity import ChatPerplexity

        model = ChatPerplexity(model="sonar", temperature=0.7)
        ```

        调用：

        ```python
        messages = [("system", "You are a chatbot."), ("user", "Hello!")]
        model.invoke(messages)
        ```

        带结构化输出的调用：

        ```python
        from pydantic import BaseModel


        class StructuredOutput(BaseModel):
            role: str
            content: str


        model.with_structured_output(StructuredOutput)
        model.invoke(messages)
        ```

        流式：
        ```python
        for chunk in model.stream(messages):
            print(chunk.content)
        ```

        token 使用情况：
        ```python
        response = model.invoke(messages)
        response.usage_metadata
        ```

        响应元数据：
        ```python
        response = model.invoke(messages)
        response.response_metadata
        ```

        智能体 API（Responses）：

        Set `use_responses_api=True` to route requests through Perplexity's Agent
        API (the Perplexity-flavored Responses API), or leave it unset to have it
        auto-detected when a built-in tool (e.g. `web_search`) or any
        Responses-only field (`previous_response_id`, `instructions`, `input`,
        `include`) is supplied.

        The Agent API uses its own model catalog (see
        https://docs.perplexity.ai/docs/agent-api/models). Select routing by
        passing either an explicit `model=` such as `"openai/gpt-5.6-sol"`, or a
        Perplexity `preset` (`"fast"`, `"low"`, `"medium"`, `"high"`, `"xhigh"`,
        or `"wide-research"`) through
        `model_kwargs`. Perplexity's built-in tools and Agent-API-only fields
        also go through `model_kwargs`:

        ```python
        from langchain_perplexity import ChatPerplexity

        model = ChatPerplexity(
            use_responses_api=True,
            model_kwargs={
                "preset": "medium",
                "tools": [{"type": "web_search"}],
            },
        )
        model.invoke("What did Perplexity announce most recently?")
        ```

        自动检测示例（由于存在内置的 `web_search` 工具，路由至智能体 API）：

        ```python
        model = ChatPerplexity(model="openai/gpt-5.6-sol")
        model.invoke(
            "Find recent news about AI.",
            tools=[{"type": "web_search"}],
        )
        ```
    """  # noqa: E501

    client: Any = Field(default=None, exclude=True)
    async_client: Any = Field(default=None, exclude=True)

    model: str = "sonar"
    """模型名称。"""

    temperature: float = 0.7
    """采样温度。"""

    model_kwargs: dict[str, Any] = Field(default_factory=dict)
    """保存 `create` 调用中有效但未明确指定的任何模型参数。"""

    pplx_api_key: SecretStr | None = Field(
        default_factory=secret_from_env("PPLX_API_KEY", default=None), alias="api_key"
    )
    """Perplexity API 密钥。"""

    request_timeout: float | tuple[float, float] | None = Field(None, alias="timeout")
    """对 PerplexityChat 补全 API 请求的超时时间。"""

    max_retries: int = 6
    """生成时最大重试次数。"""

    streaming: bool = False
    """是否流式输出结果。"""

    max_tokens: int | None = None
    """最大生成 token 数量。"""

    use_responses_api: bool | None = None
    """是否使用 Responses（智能体）API 而非聊天补全 API。

    如果不指定，则将根据调用参数推断。具体而言，当负载包含内置工具（任意 `tools[*]` 其
    `type` 不为 `"function"`）或以下任一仅 Responses 字段时，请求将路由至 Responses
    API：`previous_response_id`、`instructions`、`input`、`include`。

    显式设置为 `True` 以始终使用 Responses API，或设置为 `False` 以始终使用聊天补全
    API。

    !!! warning "Responses（智能体）API 上禁用的参数"

        Perplexity 智能体 API 不接受仅聊天补全使用的参数。当通过 Responses
        路由（无论是显式还是推断）时：

        - `temperature`, `top_p`, `top_k`, `stop`, and `metadata` are dropped
          at the boundary with a `WARNING` log so the behavior change is
          discoverable. The class default `temperature` is dropped silently
          (it would otherwise spam every call), but a user-supplied
          `temperature` (init, `invoke(temperature=...)`, or `.bind`) still
          warns.
        - `tool_choice` raises `ValueError` rather than being dropped, since
          downstream agent loops cannot recover from a silently-disabled
          forced tool call.
        - Supplying a `preset` causes `model` to be dropped because the Agent
          API rejects bare Chat-Completions model names when `model` is
          provided. If `model` was explicitly set by the user, a `WARNING` is
          logged so the override is discoverable.

        Use `use_responses_api=False` if you need any of these parameters to
        take effect.
    """

    search_mode: Literal["academic", "sec", "web"] | None = None
    """Search mode for specialized content: 用于专业内容的搜索模式："academic"、"sec" 或 "web"。"""

    reasoning_effort: Literal["low", "medium", "high"] | None = None
    """Reasoning effort: 推理努力程度："low"、"medium" 或 "high"（默认）。"""

    language_preference: str | None = None
    """语言偏好："""

    search_domain_filter: list[str] | None = None
    """Search domain filter: 搜索域名过滤：要过滤搜索结果的域名列表（最多 20 个）。"""

    return_images: bool = False
    """是否在响应中返回图片。"""

    return_related_questions: bool = False
    """是否在响应中返回相关问题。"""

    search_recency_filter: Literal["day", "week", "month", "year"] | None = None
    """Filter search results by recency: 按时效性过滤搜索结果："day"、"week"、"month" 或 "year"。"""

    search_after_date_filter: str | None = None
    """Search after date filter: 搜索后日期过滤：日期格式为 "MM/DD/YYYY"（默认）。"""

    search_before_date_filter: str | None = None
    """仅返回此日期之前的搜索结果（格式：MM/DD/YYYY）。"""

    last_updated_after_filter: str | None = None
    """仅返回在此日期之后更新的搜索结果（格式：MM/DD/YYYY）。"""

    last_updated_before_filter: str | None = None
    """仅返回在此日期之前更新的搜索结果（格式：MM/DD/YYYY）。"""

    disable_search: bool = False
    """是否完全禁用网络搜索。"""

    enable_search_classifier: bool = False
    """是否启用搜索分类器。"""

    web_search_options: WebSearchOptions | None = None
    """配置网络搜索行为，包括 Pro Search。"""

    media_response: MediaResponse | None = None
    """Media response: 媒体响应："images"、"videos" 或 "none"（默认）。"""

    model_config = ConfigDict(populate_by_name=True)

    @property
    def lc_secrets(self) -> dict[str, str]:
        return {"pplx_api_key": "PPLX_API_KEY"}

    @model_validator(mode="before")
    @classmethod
    def build_extra(cls, values: dict[str, Any]) -> Any:
        """从传入的额外参数中构建额外的 kwargs。"""
        all_required_field_names = get_pydantic_field_names(cls)
        extra = values.get("model_kwargs", {})
        for field_name in list(values):
            if field_name in extra:
                raise ValueError(f"Found {field_name} supplied twice.")
            if field_name not in all_required_field_names:
                logger.warning(
                    f"""WARNING! {field_name} is not a default parameter.
                    {field_name} was transferred to model_kwargs.
                    Please confirm that {field_name} is what you intended."""
                )
                extra[field_name] = values.pop(field_name)

        invalid_model_kwargs = all_required_field_names.intersection(extra.keys())
        if invalid_model_kwargs:
            raise ValueError(
                f"Parameters {invalid_model_kwargs} should be specified explicitly. "
                f"Instead they were passed in as part of `model_kwargs` parameter."
            )

        values["model_kwargs"] = extra
        return values

    @model_validator(mode="after")
    def _set_perplexity_version(self) -> Self:
        """在元数据中设置包版本。"""
        self._add_version("langchain-perplexity", __version__)
        return self

    @model_validator(mode="after")
    def validate_environment(self) -> Self:
        """验证 API 密钥和 Python 包是否存在于环境中。"""
        pplx_api_key = (
            self.pplx_api_key.get_secret_value() if self.pplx_api_key else None
        )

        client_params: dict[str, Any] = {
            "api_key": pplx_api_key,
            "max_retries": self.max_retries,
        }
        if self.request_timeout is not None:
            client_params["timeout"] = self.request_timeout

        if not self.client:
            self.client = Perplexity(**client_params)

        if not self.async_client:
            self.async_client = AsyncPerplexity(**client_params)

        return self

    def _resolve_model_profile(self) -> ModelProfile | None:
        return _get_default_model_profile(self.model) or None

    @property
    def _default_params(self) -> dict[str, Any]:
        """获取调用 PerplexityChat API 的默认参数。"""
        params: dict[str, Any] = {
            "max_tokens": self.max_tokens,
            "stream": self.streaming,
            "temperature": self.temperature,
        }
        if self.search_mode:
            params["search_mode"] = self.search_mode
        if self.reasoning_effort:
            params["reasoning_effort"] = self.reasoning_effort
        if self.language_preference:
            params["language_preference"] = self.language_preference
        if self.search_domain_filter:
            params["search_domain_filter"] = self.search_domain_filter
        if self.return_images:
            params["return_images"] = self.return_images
        if self.return_related_questions:
            params["return_related_questions"] = self.return_related_questions
        if self.search_recency_filter:
            params["search_recency_filter"] = self.search_recency_filter
        if self.search_after_date_filter:
            params["search_after_date_filter"] = self.search_after_date_filter
        if self.search_before_date_filter:
            params["search_before_date_filter"] = self.search_before_date_filter
        if self.last_updated_after_filter:
            params["last_updated_after_filter"] = self.last_updated_after_filter
        if self.last_updated_before_filter:
            params["last_updated_before_filter"] = self.last_updated_before_filter
        if self.disable_search:
            params["disable_search"] = self.disable_search
        if self.enable_search_classifier:
            params["enable_search_classifier"] = self.enable_search_classifier
        if self.web_search_options:
            params["web_search_options"] = self.web_search_options.model_dump(
                exclude_none=True
            )
        if self.media_response:
            if "extra_body" not in params:
                params["extra_body"] = {}
            params["extra_body"]["media_response"] = self.media_response.model_dump(
                exclude_none=True
            )

        return {**params, **self.model_kwargs}

    def _convert_message_to_dict(self, message: BaseMessage) -> dict[str, Any]:
        message_dict: dict[str, Any]
        if isinstance(message, ChatMessage):
            message_dict = {"role": message.role, "content": message.content}
        elif isinstance(message, SystemMessage):
            message_dict = {"role": "system", "content": message.content}
        elif isinstance(message, HumanMessage):
            message_dict = {"role": "user", "content": message.content}
        elif isinstance(message, AIMessage):
            message_dict = {"role": "assistant", "content": message.content}
            if message.tool_calls or message.invalid_tool_calls:
                message_dict["tool_calls"] = [
                    {
                        "id": tool_call["id"],
                        "type": "function",
                        "function": {
                            "name": tool_call["name"],
                            "arguments": json.dumps(
                                tool_call["args"], ensure_ascii=False
                            ),
                        },
                    }
                    for tool_call in message.tool_calls
                ] + [
                    {
                        "id": tool_call["id"],
                        "type": "function",
                        "function": {
                            "name": tool_call["name"],
                            "arguments": tool_call["args"],
                        },
                    }
                    for tool_call in message.invalid_tool_calls
                ]
                                # OpenAI 兼容的 API 会拒绝空字符串内容与
                                # tool_calls; 请发送 null 替代。
                message_dict["content"] = message_dict["content"] or None
        elif isinstance(message, ToolMessage):
            message_dict = {
                "role": "tool",
                "content": message.content,
                "tool_call_id": message.tool_call_id,
            }
        else:
            raise TypeError(f"Got unknown type {message}")
        return message_dict

    def _create_message_dicts(
        self, messages: list[BaseMessage], stop: list[str] | None
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        params = dict(self._invocation_params)
        if stop is not None:
            if "stop" in params:
                raise ValueError("`stop` found in both the input and default params.")
            params["stop"] = stop
        message_dicts = [self._convert_message_to_dict(m) for m in messages]
        return message_dicts, params

    def _use_responses_api(self, payload: dict) -> bool:
        """如果 `payload` 应通过 Responses API 路由，则返回 True。

        当显式设置 `self.use_responses_api` 时，尊重该值；否则委托给模块级别的
        `_use_responses_api` 启发式判断。
        """
        if isinstance(self.use_responses_api, bool):
            return self.use_responses_api
        return _use_responses_api(payload)

    def _to_responses_payload(
        self,
        message_dicts: list[dict[str, Any]],
        params: dict[str, Any],
        *,
        user_set_keys: set[str] | None = None,
    ) -> dict[str, Any]:
        """将 Chat Completions 风格的 payload 转换为 Responses API 格式。

        Renames `messages` to `input` and `max_tokens` to `max_output_tokens`.
        `None`-valued params are dropped. Chat-Completions-only sampling/control
        parameters that the Perplexity Responses (Agent) API does not accept
        (`temperature`, `top_p`, `top_k`, `stop`, `metadata`) are dropped at
        the boundary because the typed SDK signature would otherwise raise a
        `TypeError`; every drop emits a `WARNING`-level log on each call,
        except the class-default `temperature`, which is suppressed because
        `_default_params` injects it on every call regardless of user intent.

        `tool_choice` is rejected with `ValueError` rather than dropped: it is
        a control-flow primitive (forced/required tool selection) that agent
        loops depend on, so silently disabling it would produce wrong
        completions while returning HTTP 200.

        未知或 Perplexity 特有的键（包括 `previous_response_id` 和
        `include`，这些是文档中记录的 Perplexity 功能，但当前类型化 SDK
        签名未暴露）将通过 `extra_body` 转发。

        未知或 Perplexity 特有的键（包括 `previous_response_id` 和 `include`，这些是
        Perplexity 文档中记录的功能，但当前类型化的 SDK 签名尚未暴露）会通过
        `extra_body` 传递。

        Args:
            message_dicts: Chat messages already serialized to the Chat
                Completions shape; promoted to `payload["input"]`.
            params: Merged invocation params from `_default_params` and the
                per-call `kwargs`.
            user_set_keys: Keys the user explicitly supplied for this call
                (typically `set(kwargs)`). Used in combination with
                `self.model_fields_set` to distinguish class defaults from
                explicit user intent for `temperature` and `model`.

        Raises:
            ValueError: 如果提供了 `tool_choice` —— Responses API 无法支持。 TypeError:
            如果调用者提供了非 `dict` 类型的 `extra_body` ——
            静默丢弃后续参数会掩盖用户设置的搜索/过滤选项。
        """
        payload: dict[str, Any] = {"input": _translate_responses_input(message_dicts)}
        runtime_keys = user_set_keys or set()
        user_set_temperature = (
            "temperature" in self.model_fields_set or "temperature" in runtime_keys
        )
        user_set_model = "model" in self.model_fields_set or "model" in runtime_keys
                # 收集被丢弃的值，以便警告信息可以指出它们。
        dropped_for_warning: dict[str, Any] = {}
        for key, value in params.items():
            if value is None:
                continue
            if key == "messages":
                continue
            if key in _RESPONSES_DROP_KEYS:
                                # 抑制类默认值 `temperature` 的警告,
                                # 在每次调用时注入的 `_default_params`
                                # 否则会向从未请求过此服务的用户发送垃圾信息。
                if key != "temperature" or user_set_temperature:
                    dropped_for_warning[key] = value
                continue
            if key == "tool_choice":
                msg = (
                    "Perplexity Responses (Agent) API does not support "
                    "`tool_choice`. Forced tool selection is unavailable on "
                    "this route. Set `use_responses_api=False` to use Chat "
                    "Completions, or remove `tool_choice` to let the model "
                    "decide."
                )
                raise ValueError(msg)
            if key == "max_tokens":
                payload["max_output_tokens"] = value
                continue
            if key == "tools":
                                # 函数工具必须展平为 Responses-API 格式；
                                # 内置工具（web_search 等）原样传递。
                payload["tools"] = [_flatten_responses_tool(tool) for tool in value]
                continue
            if key in _RESPONSES_PASSTHROUGH_KEYS:
                payload[key] = dict(value) if isinstance(value, dict) else value
                continue
                        # 未知 / Perplexity 特有键：通过 extra_body 路由
                        # SDK 会将它们转发到智能体 API，而不会破坏严格的类型定义。
            extra_body = payload.setdefault("extra_body", {})
            if not isinstance(extra_body, dict):
                msg = (
                    "`extra_body` must be a dict to forward Perplexity-specific "
                    f"parameters to the Responses API, got "
                    f"{type(extra_body).__name__}={extra_body!r}; cannot merge "
                    f"user-set key {key!r}."
                )
                raise TypeError(msg)
            extra_body[key] = value
                # 当调用方选择了预设时，将模型选择推迟给该预设：
                # 智能体 API 会直接拒绝仅包含 Chat-Completions 模型名称（如 `sonar-pro`）的情况
                # 当设置了 `model` 时，即使同时存在预设也是如此。
        if "preset" in payload:
            dropped_model = payload.pop("model", None)
            if user_set_model and dropped_model is not None:
                logger.warning(
                    "Perplexity Agent API rejects `model` when `preset` is "
                    "set; dropping explicit model=%r in favor of preset=%r.",
                    dropped_model,
                    payload["preset"],
                )
        if dropped_for_warning:
            logger.warning(
                "Perplexity Responses (Agent) API does not accept %s; the "
                "following values were dropped: %s. Use the Chat Completions "
                "API (set `use_responses_api=False`) if you need them.",
                sorted(dropped_for_warning),
                dropped_for_warning,
            )
        return payload

    def _convert_delta_to_message_chunk(
        self, _dict: Mapping[str, Any], default_class: type[BaseMessageChunk]
    ) -> BaseMessageChunk:
        role = _dict.get("role")
        content = _dict.get("content") or ""
        additional_kwargs: dict = {}
        if _dict.get("function_call"):
            function_call = dict(_dict["function_call"])
            if "name" in function_call and function_call["name"] is None:
                function_call["name"] = ""
            additional_kwargs["function_call"] = function_call
        if _dict.get("tool_calls"):
            additional_kwargs["tool_calls"] = _dict["tool_calls"]

        if role == "user" or default_class == HumanMessageChunk:
            return HumanMessageChunk(content=content)
        elif role == "assistant" or default_class == AIMessageChunk:
            return AIMessageChunk(content=content, additional_kwargs=additional_kwargs)
        elif role == "system" or default_class == SystemMessageChunk:
            return SystemMessageChunk(content=content)
        elif role == "function" or default_class == FunctionMessageChunk:
            return FunctionMessageChunk(content=content, name=_dict["name"])
        elif role == "tool" or default_class == ToolMessageChunk:
            return ToolMessageChunk(content=content, tool_call_id=_dict["tool_call_id"])
        elif role or default_class == ChatMessageChunk:
            return ChatMessageChunk(content=content, role=role)  # type: ignore[arg-type]
        else:
            return default_class(content=content)  # type: ignore[call-arg]

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        message_dicts, params = self._create_message_dicts(messages, stop)
        runtime_keys = set(kwargs)
        if stop is not None:
            runtime_keys.add("stop")
        params = {**params, **kwargs}
        default_chunk_class = AIMessageChunk
        params.pop("stream", None)
        if self._use_responses_api({**params, "messages": message_dicts}):
            responses_payload = self._to_responses_payload(
                message_dicts, params, user_set_keys=runtime_keys
            )
            responses_payload["stream"] = True
            stream_events = self.client.responses.create(**responses_payload)
                        # 信任 SDK 的 SSE 解码（perplexityai>=0.34.1，上游问题
            # perplexityai-python#53）。
                        # _convert_responses_stream_event_to_chunk
                        # 已通过 `_get_attr` 处理 SDK 对象和字典两种情况。
            for event in stream_events:
                response_chunk = _convert_responses_stream_event_to_chunk(event)
                if response_chunk is None:
                    continue
                if run_manager:
                    run_manager.on_llm_new_token(
                        response_chunk.text, chunk=response_chunk
                    )
                yield response_chunk
            return
        if stop:
            params["stop_sequences"] = stop
        stream_resp = self.client.chat.completions.create(
            messages=message_dicts, stream=True, **params
        )
        first_chunk = True
        prev_total_usage: UsageMetadata | None = None

        added_model_name: bool = False
        added_search_queries: bool = False
        added_search_context_size: bool = False
        for chunk in stream_resp:
            if not isinstance(chunk, dict):
                chunk = chunk.model_dump()
                        # 收集标准的使用元数据（从聚合形式转换为增量形式）
            if total_usage := chunk.get("usage"):
                lc_total_usage = _create_usage_metadata(total_usage)
                if prev_total_usage:
                    usage_metadata: UsageMetadata | None = subtract_usage(
                        lc_total_usage, prev_total_usage
                    )
                else:
                    usage_metadata = lc_total_usage
                prev_total_usage = lc_total_usage
            else:
                usage_metadata = None
            generation_info = {}
            if (model_name := chunk.get("model")) and not added_model_name:
                generation_info["model_name"] = model_name
                added_model_name = True
            if total_usage := chunk.get("usage"):
                if num_search_queries := total_usage.get("num_search_queries"):
                    if not added_search_queries:
                        generation_info["num_search_queries"] = num_search_queries
                        added_search_queries = True
                if not added_search_context_size:
                    if search_context_size := total_usage.get("search_context_size"):
                        generation_info["search_context_size"] = search_context_size
                        added_search_context_size = True

            choices = chunk.get("choices") or []
            if len(choices) == 0:
                                # 仅包含使用信息或为空的块：仍需生成，以确保流不为空，下游调用者能接收到使用元数据。
                                # 参见同步版本的 `_stream` 以了解 SDK
                # 信任的依据（perplexityai>=0.34.1）。
                message = AIMessageChunk(content="", usage_metadata=usage_metadata)
                yield ChatGenerationChunk(
                    message=message, generation_info=generation_info or None
                )
                continue
            choice = choices[0]

            additional_kwargs = {}
            if first_chunk:
                additional_kwargs["citations"] = chunk.get("citations", [])
                for attr in ["images", "related_questions", "search_results"]:
                    if attr in chunk:
                        additional_kwargs[attr] = chunk[attr]

                if chunk.get("videos"):
                    additional_kwargs["videos"] = chunk["videos"]

                if chunk.get("reasoning_steps"):
                    additional_kwargs["reasoning_steps"] = chunk["reasoning_steps"]

            chunk = self._convert_delta_to_message_chunk(
                choice["delta"], default_chunk_class
            )

            if isinstance(chunk, AIMessageChunk) and usage_metadata:
                chunk.usage_metadata = usage_metadata

            if first_chunk:
                chunk.additional_kwargs |= additional_kwargs
                first_chunk = False

            if finish_reason := choice.get("finish_reason"):
                generation_info["finish_reason"] = finish_reason

            default_chunk_class = chunk.__class__
            chunk = ChatGenerationChunk(message=chunk, generation_info=generation_info)
            if run_manager:
                run_manager.on_llm_new_token(chunk.text, chunk=chunk)
            yield chunk

    async def _astream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        message_dicts, params = self._create_message_dicts(messages, stop)
        runtime_keys = set(kwargs)
        if stop is not None:
            runtime_keys.add("stop")
        params = {**params, **kwargs}
        default_chunk_class = AIMessageChunk
        params.pop("stream", None)
        if self._use_responses_api({**params, "messages": message_dicts}):
            responses_payload = self._to_responses_payload(
                message_dicts, params, user_set_keys=runtime_keys
            )
            responses_payload["stream"] = True
            stream_events = await self.async_client.responses.create(
                **responses_payload
            )
                        # 仅包含使用信息或为空的块：仍需生成，以确保流不为空，下游调用者能接收到使用元数据。
            async for event in stream_events:
                response_chunk = _convert_responses_stream_event_to_chunk(event)
                if response_chunk is None:
                    continue
                if run_manager:
                    await run_manager.on_llm_new_token(
                        response_chunk.text, chunk=response_chunk
                    )
                yield response_chunk
            return
        if stop:
            params["stop_sequences"] = stop
        stream_resp = await self.async_client.chat.completions.create(
            messages=message_dicts, stream=True, **params
        )
        first_chunk = True
        prev_total_usage: UsageMetadata | None = None

        added_model_name: bool = False
        added_search_queries: bool = False
        added_search_context_size: bool = False
        async for chunk in stream_resp:
            if not isinstance(chunk, dict):
                chunk = chunk.model_dump()
            if total_usage := chunk.get("usage"):
                lc_total_usage = _create_usage_metadata(total_usage)
                if prev_total_usage:
                    usage_metadata: UsageMetadata | None = subtract_usage(
                        lc_total_usage, prev_total_usage
                    )
                else:
                    usage_metadata = lc_total_usage
                prev_total_usage = lc_total_usage
            else:
                usage_metadata = None
            generation_info = {}
            if (model_name := chunk.get("model")) and not added_model_name:
                generation_info["model_name"] = model_name
                added_model_name = True
            if total_usage := chunk.get("usage"):
                if num_search_queries := total_usage.get("num_search_queries"):
                    if not added_search_queries:
                        generation_info["num_search_queries"] = num_search_queries
                        added_search_queries = True
                if not added_search_context_size:
                    if search_context_size := total_usage.get("search_context_size"):
                        generation_info["search_context_size"] = search_context_size
                        added_search_context_size = True

            choices = chunk.get("choices") or []
            if len(choices) == 0:
                                # 'any' 不是 OpenAI 模式原生支持的；将其映射为 'required'
                                # 以与使用 'any' 的其他提供方保持一致。
                message = AIMessageChunk(content="", usage_metadata=usage_metadata)
                yield ChatGenerationChunk(
                    message=message, generation_info=generation_info or None
                )
                continue
            choice = choices[0]

            additional_kwargs = {}
            if first_chunk:
                additional_kwargs["citations"] = chunk.get("citations", [])
                for attr in ["images", "related_questions", "search_results"]:
                    if attr in chunk:
                        additional_kwargs[attr] = chunk[attr]

                if chunk.get("videos"):
                    additional_kwargs["videos"] = chunk["videos"]

                if chunk.get("reasoning_steps"):
                    additional_kwargs["reasoning_steps"] = chunk["reasoning_steps"]

            chunk = self._convert_delta_to_message_chunk(
                choice["delta"], default_chunk_class
            )

            if isinstance(chunk, AIMessageChunk) and usage_metadata:
                chunk.usage_metadata = usage_metadata

            if first_chunk:
                chunk.additional_kwargs |= additional_kwargs
                first_chunk = False

            if finish_reason := choice.get("finish_reason"):
                generation_info["finish_reason"] = finish_reason

            default_chunk_class = chunk.__class__
            chunk = ChatGenerationChunk(message=chunk, generation_info=generation_info)
            if run_manager:
                await run_manager.on_llm_new_token(chunk.text, chunk=chunk)
            yield chunk

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        if self.streaming:
            stream_iter = self._stream(
                messages, stop=stop, run_manager=run_manager, **kwargs
            )
            if stream_iter:
                return generate_from_stream(stream_iter)
        message_dicts, params = self._create_message_dicts(messages, stop)
        runtime_keys = set(kwargs)
        if stop is not None:
            runtime_keys.add("stop")
        params = {**params, **kwargs}
        if self._use_responses_api({**params, "messages": message_dicts}):
            responses_payload = self._to_responses_payload(
                message_dicts, params, user_set_keys=runtime_keys
            )
            responses_payload.pop("stream", None)
            response = self.client.responses.create(**responses_payload)
            return _convert_responses_to_chat_result(response)
        response = self.client.chat.completions.create(messages=message_dicts, **params)

        if hasattr(response, "usage") and response.usage:
            usage_dict = response.usage.model_dump()
            usage_metadata = _create_usage_metadata(usage_dict)
        else:
            usage_metadata = None
            usage_dict = {}

        additional_kwargs = {}
        for attr in ["citations", "images", "related_questions", "search_results"]:
            if hasattr(response, attr) and getattr(response, attr):
                additional_kwargs[attr] = getattr(response, attr)

        if hasattr(response, "videos") and response.videos:
            additional_kwargs["videos"] = [
                v.model_dump() if hasattr(v, "model_dump") else v
                for v in response.videos
            ]

        if hasattr(response, "reasoning_steps") and response.reasoning_steps:
            additional_kwargs["reasoning_steps"] = [
                r.model_dump() if hasattr(r, "model_dump") else r
                for r in response.reasoning_steps
            ]

        response_metadata: dict[str, Any] = {
            "model_name": getattr(response, "model", self.model)
        }
        if num_search_queries := usage_dict.get("num_search_queries"):
            response_metadata["num_search_queries"] = num_search_queries
        if search_context_size := usage_dict.get("search_context_size"):
            response_metadata["search_context_size"] = search_context_size

        message = AIMessage(
            content=response.choices[0].message.content,
            additional_kwargs=additional_kwargs,
            usage_metadata=usage_metadata,
            response_metadata=response_metadata,
        )
        return ChatResult(generations=[ChatGeneration(message=message)])

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        if self.streaming:
            stream_iter = self._astream(
                messages, stop=stop, run_manager=run_manager, **kwargs
            )
            if stream_iter:
                return await agenerate_from_stream(stream_iter)
        message_dicts, params = self._create_message_dicts(messages, stop)
        runtime_keys = set(kwargs)
        if stop is not None:
            runtime_keys.add("stop")
        params = {**params, **kwargs}
        if self._use_responses_api({**params, "messages": message_dicts}):
            responses_payload = self._to_responses_payload(
                message_dicts, params, user_set_keys=runtime_keys
            )
            responses_payload.pop("stream", None)
            response = await self.async_client.responses.create(**responses_payload)
            return _convert_responses_to_chat_result(response)
        response = await self.async_client.chat.completions.create(
            messages=message_dicts, **params
        )

        if hasattr(response, "usage") and response.usage:
            usage_dict = response.usage.model_dump()
            usage_metadata = _create_usage_metadata(usage_dict)
        else:
            usage_metadata = None
            usage_dict = {}

        additional_kwargs = {}
        for attr in ["citations", "images", "related_questions", "search_results"]:
            if hasattr(response, attr) and getattr(response, attr):
                additional_kwargs[attr] = getattr(response, attr)

        if hasattr(response, "videos") and response.videos:
            additional_kwargs["videos"] = [
                v.model_dump() if hasattr(v, "model_dump") else v
                for v in response.videos
            ]

        if hasattr(response, "reasoning_steps") and response.reasoning_steps:
            additional_kwargs["reasoning_steps"] = [
                r.model_dump() if hasattr(r, "model_dump") else r
                for r in response.reasoning_steps
            ]

        response_metadata: dict[str, Any] = {
            "model_name": getattr(response, "model", self.model)
        }
        if num_search_queries := usage_dict.get("num_search_queries"):
            response_metadata["num_search_queries"] = num_search_queries
        if search_context_size := usage_dict.get("search_context_size"):
            response_metadata["search_context_size"] = search_context_size

        message = AIMessage(
            content=response.choices[0].message.content,
            additional_kwargs=additional_kwargs,
            usage_metadata=usage_metadata,
            response_metadata=response_metadata,
        )
        return ChatResult(generations=[ChatGeneration(message=message)])

    @property
    def _invocation_params(self) -> Mapping[str, Any]:
        """获取用于调用模型的参数。"""
        pplx_creds: dict[str, Any] = {"model": self.model}
        return {**pplx_creds, **self._default_params}

    @property
    def _llm_type(self) -> str:
        """聊天模型的返回类型。"""
        return "perplexitychat"

    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type | Callable | BaseTool],
        *,
        tool_choice: dict | str | bool | None = None,
        strict: bool | None = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, AIMessage]:
        """将工具类对象绑定到此聊天模型。

        Client-side function tools require the Perplexity Responses (Agent) API:
        construct the model with `use_responses_api=True` and a tool-capable
        model such as `openai/gpt-5`. The `sonar` family does not support
        client-side function tools.

        Args:
            tools: A list of tool definitions to bind to this chat model.
                Supports any tool handled by
                [convert_to_openai_tool][langchain_core.utils.function_calling.convert_to_openai_tool]
                (Pydantic models, `TypedDict` classes, callables, `BaseTool`,
                or OpenAI-format dicts), as well as Perplexity built-in tools such
                as `{"type": "web_search"}`, which are passed through unchanged.
            tool_choice: Which tool the model should use. Normalized here for API
                parity with `langchain-openai` (a tool name, `"auto"`, `"none"`,
                `"any"`/`"required"`/`True`, or an OpenAI-style dict) and stored
                on the binding, but the Perplexity Responses (Agent) API does not
                currently honor it: a non-empty `tool_choice` makes
                `_to_responses_payload` raise `ValueError` at invoke time on the
                Responses route. The restriction can be relaxed if Perplexity
                adds `tool_choice` support.
            strict: If `True`, the tool parameter schema is sent with `strict`
                enabled. If `None` (default), the flag is omitted.
            kwargs: Any additional parameters are passed directly to `bind`.
        """
        formatted_tools = [
            tool
            if isinstance(tool, dict) and _is_builtin_tool(tool)
            else convert_to_openai_tool(tool, strict=strict)
            for tool in tools
        ]
        if tool_choice:
            tool_names = [
                t["function"]["name"] if "function" in t else t.get("name")
                for t in formatted_tools
            ]
            if isinstance(tool_choice, str):
                if tool_choice in tool_names:
                    tool_choice = {
                        "type": "function",
                        "function": {"name": tool_choice},
                    }
                                # ‘any’ 不是 OpenAI 模式原生支持的；将其映射为 'required'
                                # 为与使用 'any' 的提供方保持一致。
                elif tool_choice == "any":
                    tool_choice = "required"
            elif isinstance(tool_choice, bool):
                tool_choice = "required"
            elif isinstance(tool_choice, dict):
                pass
            else:
                msg = (
                    "Unrecognized tool_choice type. Expected str, bool or dict. "
                    f"Received: {tool_choice}"
                )
                raise ValueError(msg)
            kwargs["tool_choice"] = tool_choice
        return super().bind(tools=formatted_tools, **kwargs)

    def with_structured_output(
        self,
        schema: _DictOrPydanticClass | None = None,
        *,
        method: Literal["json_schema"] = "json_schema",
        include_raw: bool = False,
        strict: bool | None = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, _DictOrPydantic]:
        """Model wrapper that returns outputs formatted to match the given schema for Preplexity.
        Currently, Perplexity only supports "json_schema" method for structured output
        as per their [official documentation](https://docs.perplexity.ai/guides/structured-outputs).

        Args:
            schema: 输出模式。可作为以下形式传入：

                - JSON Schema,
                - `TypedDict` 类,
                - 或 Pydantic 类

            method: 用于引导模型生成的方法，目前仅支持：

                - `'json_schema'`: 使用 JSON Schema 解析模型输出


            include_raw:
                如果为 `False`，则仅返回解析后的结构化输出。

                如果在模型输出解析过程中发生错误，将抛出异常。

                如果为 `True`，则同时返回原始模型响应（一个
                `BaseMessage`）和解析后的模型响应。

                如果在输出解析过程中发生错误，该错误将被捕获并一同返回。

                The final output is always a `dict` with keys `'raw'`, `'parsed'`, and
                `'parsing_error'`.
            strict:
                Unsupported: whether to enable strict schema adherence when generating
                the output. This parameter is included for compatibility with other
                chat models, but is currently ignored.

            kwargs: 不支持额外的关键字参数。

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
        """  # noqa: E501
        if method in ("function_calling", "json_mode"):
            method = "json_schema"
        if method == "json_schema":
            if schema is None:
                raise ValueError(
                    "schema must be specified when method is not 'json_schema'. "
                    "Received None."
                )
            is_pydantic_schema = _is_pydantic_class(schema)
            response_format = convert_to_json_schema(schema)
            llm = self.bind(
                response_format={
                    "type": "json_schema",
                    "json_schema": {"schema": response_format},
                },
                ls_structured_output_format={
                    "kwargs": {"method": method},
                    "schema": response_format,
                },
            )
            output_parser = (
                ReasoningStructuredOutputParser(pydantic_object=schema)  # type: ignore[arg-type]
                if is_pydantic_schema
                else ReasoningJsonOutputParser()
            )
        else:
            raise ValueError(
                f"Unrecognized method argument. Expected 'json_schema' Received:\
                    '{method}'"
            )

        if include_raw:
            parser_assign = RunnablePassthrough.assign(
                parsed=itemgetter("raw") | output_parser, parsing_error=lambda _: None
            )
            parser_none = RunnablePassthrough.assign(parsed=lambda _: None)
            parser_with_fallback = parser_assign.with_fallbacks(
                [parser_none], exception_key="parsing_error"
            )
            return RunnableMap(raw=llm) | parser_with_fallback
        else:
            return llm | output_parser

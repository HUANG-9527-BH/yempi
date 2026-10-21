"""OpenRouter 聊天模型。"""

from __future__ import annotations

import json
import warnings
from collections.abc import AsyncIterator, Callable, Iterator, Mapping, Sequence
from operator import itemgetter
from typing import Any, Literal, cast

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
    LangSmithParams,
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
from langchain_core.messages.ai import (
    InputTokenDetails,
    OutputTokenDetails,
    UsageMetadata,
)
from langchain_core.messages.block_translators.openai import (
    convert_to_openai_data_block,
)
from langchain_core.messages.tool import tool_call_chunk
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
from langchain_core.utils import from_env, get_pydantic_field_names, secret_from_env
from langchain_core.utils.function_calling import (
    convert_to_json_schema,
    convert_to_openai_tool,
)
from langchain_core.utils.pydantic import is_basemodel_subclass
from pydantic import BaseModel, ConfigDict, Field, SecretStr, model_validator
from typing_extensions import Self

from langchain_openrouter._version import __version__
from langchain_openrouter.data._profiles import _PROFILES

_MODEL_PROFILES = cast("ModelProfileRegistry", _PROFILES)

# LangChain-内部参数，不得转发至 SDK。
_INTERNAL_KWARGS = frozenset({"ls_structured_output_format"})


def _get_default_model_profile(model_name: str) -> ModelProfile:
    default = _MODEL_PROFILES.get(model_name) or {}
    return default.copy()


def _create_stream_generation_info(
    chunk_dict: dict[str, Any], choice: dict[str, Any], model_name: str
) -> dict[str, Any]:
    generation_info = {"finish_reason": choice["finish_reason"]}
    generation_info["model_name"] = chunk_dict.get("model") or model_name
    if provider := chunk_dict.get("provider"):
        generation_info["provider"] = provider
    if system_fingerprint := chunk_dict.get("system_fingerprint"):
        generation_info["system_fingerprint"] = system_fingerprint
    if native_finish_reason := choice.get("native_finish_reason"):
        generation_info["native_finish_reason"] = native_finish_reason
    if response_id := chunk_dict.get("id"):
        generation_info["id"] = response_id
    if created := chunk_dict.get("created"):
        generation_info["created"] = int(created)
    if object_ := chunk_dict.get("object"):
        generation_info["object"] = object_
    return generation_info


class ChatOpenRouter(BaseChatModel):
    """OpenRouter 聊天模型集成。

    OpenRouter 是一个统一的 API，可访问来自多个提供商（OpenAI、Anthropic、Google、Meta
    等）的数百个模型。

    ???+ info "设置"

        安装 `langchain-openrouter` 并设置环境变量 `OPENROUTER_API_KEY`。

        ```bash
        pip install -U langchain-openrouter
        ```

        ```bash
        export OPENROUTER_API_KEY="your-api-key"
        ```

    ??? info "关键初始化参数 —— 完成参数"

        | 参数 | 类型 | 描述 |
        | ----- | ---- | ----------- |
        | `model` | `str` | 模型名称 |
        | `temperature` | `float | None` | 采样温度。 |
        | `max_tokens` | `int | None` | 最大生成 token 数。 |

    ??? info "关键初始化参数 —— 客户端参数"

        | 参数 | 类型 | 描述 |
        | ----- | ---- | ----------- |
        | `api_key` | `str | None` | OpenRouter API 密钥。 |
        | `base_url` | `str | None` | API 请求的基础 URL。 |
        | `timeout` | `int | None` | 超时时间（毫秒）。 |
        | `app_url` | `str | None` | 应用程序 URL，用于归属标识。 |
        | `app_title` | `str | None` | 应用程序标题，用于归属标识。 |
        | `app_categories` | `list[str] | None` | 市场归属类别。 |
        | `session_id` | `str | None` | 用于可观测性的相关请求分组。 |
        | `trace` | `dict[str, Any] | None` | 广播追踪元数据。 |
        | `default_headers` | `Mapping[str, str] | None` | 额外请求头。 |
        | `max_retries` | `int` | 最大重试次数（默认值为 `2`）。设为 `0` 可禁用。 |

    ??? info "实例化"

        ```python
        from langchain_openrouter import ChatOpenRouter

        model = ChatOpenRouter(
            model="anthropic/claude-sonnet-4-5",
            temperature=0,
            # api_key="...",
            # openrouter_provider={"order": ["Anthropic"]},
        )
        ```

    详见 https://openrouter.ai/docs 获取平台文档。
    """

    client: Any = Field(default=None, exclude=True)
    """底层 SDK 客户端（`openrouter.OpenRouter`）。"""

    openrouter_api_key: SecretStr | None = Field(
        alias="api_key",
        default_factory=secret_from_env("OPENROUTER_API_KEY", default=None),
    )
    """OpenRouter API 密钥。"""

    openrouter_api_base: str | None = Field(
        default_factory=from_env("OPENROUTER_API_BASE", default=None),
        alias="base_url",
    )
    """OpenRouter API 基础 URL。映射到 SDK 的 `server_url`。"""

    app_url: str | None = Field(
        default_factory=from_env(
            "OPENROUTER_APP_URL",
            default="https://docs.langchain.com",
        ),
    )
    """OpenRouter 归属的应用程序 URL。

    映射到 `HTTP-Referer` 头。

    Defaults to LangChain docs URL. Set this to your app's URL to get
    attribution for API usage in the OpenRouter dashboard.

    详见 https://openrouter.ai/docs/app-attribution 了解详情。
    """

    app_title: str | None = Field(
        default_factory=from_env("OPENROUTER_APP_TITLE", default="LangChain"),
    )
    """OpenRouter 归属的应用程序标题。

    映射到 `X-Title` 头。

    Defaults to `'LangChain'`. Set this to your app's name to get attribution
    for API usage in the OpenRouter dashboard.

    详见 https://openrouter.ai/docs/app-attribution 了解详情。
    """

    app_categories: list[str] | None = Field(
        default=None,
    )
    """OpenRouter 归属的市场类别。

    映射到 `X-OpenRouter-Categories` 头。传入小写、连字符分隔的类别字符串列表（每个最多
    30 个字符），例如 `['cli-agent', 'programming-app']`。

    仅接受已识别的类别（未识别的值会被 OpenRouter 静默丢弃）。

    详见 https://openrouter.ai/docs/app-attribution 查看已识别类别。
    """

    request_timeout: int | None = Field(default=None, alias="timeout")
    """请求超时时间（毫秒）。映射到 SDK 的 `timeout_ms`。"""

    max_retries: int = 2
    """最大重试次数。

    Each unit adds ~150 seconds to the backoff window via the SDK's
    `max_elapsed_time` (e.g. `max_retries=2` allows up to ~300 s).

    设为 `0` 可禁用重试。
    """

    model_name: str = Field(alias="model")
    """模型名称，例如 `'anthropic/claude-sonnet-4-5'`。"""

    @property
    def model(self) -> str:
        """与 model_name 相同。"""
        return self.model_name

    temperature: float | None = None
    """采样温度。"""

    max_tokens: int | None = None
    """生成的最大 token 数量。"""

    max_completion_tokens: int | None = None
    """生成的最大补全 token 数量。"""

    top_p: float | None = None
    """核采样参数。"""

    frequency_penalty: float | None = None
    """生成时的频率惩罚。"""

    presence_penalty: float | None = None
    """生成时的存在惩罚。"""

    seed: int | None = None
    """用于可复现性的随机种子。"""

    stop: list[str] | str | None = Field(default=None, alias="stop_sequences")
    """默认停止序列。"""

    n: int = Field(default=1, ge=1)
    """每个提示词生成的聊天补全数量。"""

    streaming: bool = False
    """是否流式输出结果。"""

    stream_usage: bool = True
    """是否在流式输出中包含使用元数据。

    如果为 `True`，流式过程中将生成额外的消息块，包含使用元数据。
    """

    model_kwargs: dict[str, Any] = Field(default_factory=dict)
    """OpenRouter API 的任意额外模型参数。"""

    reasoning: dict[str, Any] | None = None
    """传递给 OpenRouter 的推理设置。

    Controls how many tokens the model allocates for internal chain-of-thought
    reasoning.

    接受一个 `openrouter.components.OpenResponsesReasoningConfig` 或等效的字典。

    支持的键：

    - `effort`：控制推理 token 预算。

        值：`'xhigh'`、`'high'`、`'medium'`、`'low'`、`'minimal'`、`'none'`。 -
        `summary`：控制返回响应中推理摘要的详细程度。

        Values: 值：`'auto'`、`'concise'`、`'detailed'`。

    Example: `{"effort": "high", "summary": "auto"}`

    参见 https://openrouter.ai/docs/guides/best-practices/reasoning-tokens
    """

    openrouter_provider: dict[str, Any] | None = None
    """传递给 OpenRouter 的提供方偏好设置。

    Example: `{"order": ["Anthropic", "OpenAI"]}`
    """

    route: str | None = None
    """OpenRouter 的路由偏好，例如 `'fallback'`。"""

    plugins: list[dict[str, Any]] | None = None
    """OpenRouter 的插件配置。"""

    default_headers: Mapping[str, str] | None = Field(default=None, exclude=True)
    """每次请求 OpenRouter 时要包含的额外 HTTP 头部。

    Headers set here become the underlying httpx client's default headers, so
    they are sent on every request to OpenRouter. Useful for upstream provider
    features that require custom headers — for example, xAI's `x-grok-conv-id`
    for sticky-routing prompt cache hits, region routing, A/B test bucketing
    headers, or provider-specific authentication. Whether a given header is
    forwarded to the upstream provider (versus consumed by OpenRouter itself)
    is determined by OpenRouter; consult its docs for which headers propagate.

    由于这些头部可能包含凭据，因此它们会被排除在 LangChain 序列化之外。

    Example: `{"x-grok-conv-id": "session-abc123"}`

    Headers set via this field are merged with the OpenRouter app-attribution
    headers (`HTTP-Referer`, `X-Title`, `X-OpenRouter-Categories`); on a
    collision the value from `default_headers` takes precedence, matched
    case-insensitively (HTTP header names are case-insensitive).
    """

    session_id: str | None = Field(
        default_factory=from_env("OPENROUTER_SESSION_ID", default=None),
    )
    """OpenRouter 用来将相关请求分组的标识符。

    任何需要多个请求共享可观测性分组的情况都非常有用（例如对话、智能体工作流、批处理作业或
    CI 运行）。等同于在底层请求上设置 `x-session-id` HTTP 头部。OpenRouter 会拒绝超过
    256 个字符的值。

    Falls back to the `OPENROUTER_SESSION_ID` environment variable when
    unset, so callers can group all requests from a process without
    threading the value through application code. Empty strings are
    treated as unset.

    Example: `"conv-2026-04-30-abc"`

    参见 https://openrouter.ai/docs/guides/features/broadcast/overview
    """

    trace: dict[str, Any] | None = None
    """用于可观测性工具（如 Langfuse、LangSmith）的追踪元数据。

    Forwarded by OpenRouter to configured broadcast destinations. Common
    keys include `trace_id`, `trace_name`, `span_name`, `generation_name`,
    and `parent_span_id`; see the OpenRouter broadcast docs for the
    current full set. Unknown keys are forwarded as custom metadata.

    不支持环境变量回退 —— 请按调用设置或在构造函数中设置。

    Example: `{"trace_id": "abc-123", "span_name": "summarize"}`

    参见 https://openrouter.ai/docs/guides/features/broadcast/overview
    """

    model_config = ConfigDict(populate_by_name=True)

    @model_validator(mode="before")
    @classmethod
    def build_extra(cls, values: dict[str, Any]) -> Any:
        """从传入的额外参数中构建额外关键字参数。"""
        all_required_field_names = get_pydantic_field_names(cls)
        extra = values.get("model_kwargs", {})
        for field_name in list(values):
            if field_name in extra:
                msg = f"Found {field_name} supplied twice."
                raise ValueError(msg)
            if field_name not in all_required_field_names:
                warnings.warn(
                    f"""WARNING! {field_name} is not default parameter.
                    {field_name} was transferred to model_kwargs.
                    Please confirm that {field_name} is what you intended.""",
                    stacklevel=2,
                )
                extra[field_name] = values.pop(field_name)

        invalid_model_kwargs = all_required_field_names.intersection(extra.keys())
        if invalid_model_kwargs:
            msg = (
                f"Parameters {invalid_model_kwargs} should be specified explicitly. "
                f"Instead they were passed in as part of `model_kwargs` parameter."
            )
            raise ValueError(msg)

        values["model_kwargs"] = extra
        return values

    def _build_client(self) -> Any:
        """构建并返回一个 `openrouter.OpenRouter` SDK 客户端。

        Returns:
            一个 `openrouter.OpenRouter` SDK 客户端实例。
        """
        import openrouter  # noqa: PLC0415
        from openrouter.utils import (  # noqa: PLC0415
            BackoffStrategy,
            RetryConfig,
        )

        client_kwargs: dict[str, Any] = {
            "api_key": self.openrouter_api_key.get_secret_value(),  # type: ignore[union-attr]
        }
        if self.openrouter_api_base:
            client_kwargs["server_url"] = self.openrouter_api_base
        extra_headers: dict[str, str] = {}
        if self.app_url:
            extra_headers["HTTP-Referer"] = self.app_url
        if self.app_title:
            extra_headers["X-Title"] = self.app_title
        if self.app_categories:
            extra_headers["X-OpenRouter-Categories"] = ",".join(self.app_categories)
                # 用户提供的请求头优先于内置的
                # 应用归属请求头。HTTP
        # 请求头名称不区分大小写，因此在合并前需丢弃任何名称（不区分大小写）
                # 与用户请求头冲突的内置请求头；直接使用字典更新会保留两种拼写形式，而 httpx
        # 会发送重复的请求头。
                # 序列化类方法重写
                # BaseChatModel 方法重写
        if self.default_headers:
            user_header_names = {name.casefold() for name in self.default_headers}
            extra_headers = {
                name: value
                for name, value in extra_headers.items()
                if name.casefold() not in user_header_names
            }
            extra_headers.update(self.default_headers)
        if extra_headers:
            import httpx  # noqa: PLC0415

            client_kwargs["client"] = httpx.Client(
                headers=extra_headers, follow_redirects=True
            )
            client_kwargs["async_client"] = httpx.AsyncClient(
                headers=extra_headers, follow_redirects=True
            )
        if self.request_timeout is not None:
            client_kwargs["timeout_ms"] = self.request_timeout
        if self.max_retries > 0:
            client_kwargs["retry_config"] = RetryConfig(
                strategy="backoff",
                backoff=BackoffStrategy(
                    initial_interval=500,
                    max_interval=60000,
                    exponent=1.5,
                    max_elapsed_time=self.max_retries * 150_000,
                ),
                retry_connection_errors=True,
            )
        return openrouter.OpenRouter(**client_kwargs)

    @model_validator(mode="after")
    def _set_openrouter_version(self) -> Self:
        """在元数据中设置包版本。"""
        self._add_version("langchain-openrouter", __version__)
        return self

    @model_validator(mode="after")
    def validate_environment(self) -> Self:
        """验证配置并构建 SDK 客户端。"""
        if not (self.openrouter_api_key and self.openrouter_api_key.get_secret_value()):
            msg = "OPENROUTER_API_KEY must be set."
            raise ValueError(msg)
        if self.n > 1 and self.streaming:
            msg = "n must be 1 when streaming."
            raise ValueError(msg)

        if not self.client:
            try:
                import openrouter  # noqa: PLC0415, F401

                self.client = self._build_client()
            except ImportError as e:
                msg = (
                    "Could not import the `openrouter` Python SDK. "
                    "Please install it with: pip install openrouter"
                )
                raise ImportError(msg) from e
        return self

    def _resolve_model_profile(self) -> ModelProfile | None:
        return _get_default_model_profile(self.model_name) or None

    #
        # 仅用于使用（无 choices）—— 使用 usage_metadata 发出
    #
    @property
    def lc_secrets(self) -> dict[str, str]:
        """构造函数参数名到密钥 ID 的映射。"""
        return {"openrouter_api_key": "OPENROUTER_API_KEY"}

    @classmethod
    def is_lc_serializable(cls) -> bool:
        """返回此模型是否可被 LangChain 序列化。"""
        return True

    #
        # 仅用于使用（无 choices）—— 使用 usage_metadata 发出
    #
    @property
    def _llm_type(self) -> str:
        """返回聊天模型的返回类型。"""
        return "openrouter-chat"

    @property
    def _identifying_params(self) -> dict[str, Any]:
        """获取标识参数。"""
        return {
            "model": self.model_name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "streaming": self.streaming,
            "reasoning": self.reasoning,
            "openrouter_provider": self.openrouter_provider,
            "route": self.route,
            "model_kwargs": self.model_kwargs,
        }

    def _get_ls_params(
        self,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> LangSmithParams:
        """获取用于追踪的标准参数。"""
        params = self._get_invocation_params(stop=stop, **kwargs)
        ls_params = LangSmithParams(
            ls_provider="openrouter",
            ls_model_name=params.get("model", self.model_name),
            ls_model_type="chat",
            ls_temperature=params.get("temperature", self.temperature),
        )
        if ls_max_tokens := params.get("max_tokens", self.max_tokens):
            ls_params["ls_max_tokens"] = ls_max_tokens
        if ls_stop := stop or params.get("stop", None) or self.stop:
            ls_params["ls_stop"] = ls_stop if isinstance(ls_stop, list) else [ls_stop]
        return ls_params

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
            return generate_from_stream(stream_iter)
        message_dicts, params = self._create_message_dicts(messages, stop)
        params = {**params, **kwargs}
        _strip_internal_kwargs(params)
        response = self.client.chat.send(messages=message_dicts, **params)
        return self._create_chat_result(response)

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
            return await agenerate_from_stream(stream_iter)
        message_dicts, params = self._create_message_dicts(messages, stop)
        params = {**params, **kwargs}
        _strip_internal_kwargs(params)
        response = await self.client.chat.send_async(messages=message_dicts, **params)
        return self._create_chat_result(response)

    def _stream(  # noqa: C901
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        message_dicts, params = self._create_message_dicts(messages, stop)
        params = {**params, **kwargs, "stream": True}
        if self.stream_usage:
            params["stream_options"] = {"include_usage": True}
        _strip_internal_kwargs(params)

        default_chunk_class: type[BaseMessageChunk] = AIMessageChunk
        terminal_generation_info: dict[str, Any] = {}
        for chunk in self.client.chat.send(messages=message_dicts, **params):
            chunk_dict = chunk.model_dump(by_alias=True)
            if not chunk_dict.get("choices"):
                if error := chunk_dict.get("error"):
                    msg = (
                        f"OpenRouter API returned an error during streaming: "
                        f"{error.get('message', str(error))} "
                        f"(code: {error.get('code', 'unknown')})"
                    )
                    raise ValueError(msg)
                                # 内部方法
                if usage := chunk_dict.get("usage"):
                    usage_metadata = _create_usage_metadata(usage)
                    response_metadata: dict[str, Any] = {}
                    if "cost" in usage:
                        response_metadata["cost"] = usage["cost"]
                    if "cost_details" in usage:
                        response_metadata["cost_details"] = usage["cost_details"]
                    usage_chunk = AIMessageChunk(
                        content="",
                        usage_metadata=usage_metadata,
                        response_metadata=response_metadata,
                    )
                    generation_chunk = ChatGenerationChunk(message=usage_chunk)
                    if run_manager:
                        run_manager.on_llm_new_token(
                            generation_chunk.text, chunk=generation_chunk
                        )
                    yield generation_chunk
                continue
            choice = chunk_dict["choices"][0]
            message_chunk = _convert_chunk_to_message_chunk(
                chunk_dict, default_chunk_class
            )
            generation_info: dict[str, Any] = {}
            if choice.get("finish_reason"):
                candidate_generation_info = _create_stream_generation_info(
                    chunk_dict, choice, self.model_name
                )
                generation_info.update(
                    {
                        key: value
                        for key, value in candidate_generation_info.items()
                        if key not in terminal_generation_info
                    }
                )
                terminal_generation_info.update(generation_info)
            logprobs = choice.get("logprobs")
            if logprobs:
                generation_info["logprobs"] = logprobs

            if generation_info:
                generation_info["model_provider"] = "openrouter"
                message_chunk = message_chunk.model_copy(
                    update={
                        "response_metadata": {
                            **message_chunk.response_metadata,
                            **generation_info,
                        }
                    }
                )

            default_chunk_class = message_chunk.__class__
            generation_chunk = ChatGenerationChunk(
                message=message_chunk, generation_info=generation_info or None
            )

            if run_manager:
                run_manager.on_llm_new_token(
                    generation_chunk.text,
                    chunk=generation_chunk,
                    logprobs=logprobs,
                )
            yield generation_chunk

    async def _astream(  # noqa: C901
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        message_dicts, params = self._create_message_dicts(messages, stop)
        params = {**params, **kwargs, "stream": True}
        if self.stream_usage:
            params["stream_options"] = {"include_usage": True}
        _strip_internal_kwargs(params)

        default_chunk_class: type[BaseMessageChunk] = AIMessageChunk
        terminal_generation_info: dict[str, Any] = {}
        async for chunk in await self.client.chat.send_async(
            messages=message_dicts, **params
        ):
            chunk_dict = chunk.model_dump(by_alias=True)
            if not chunk_dict.get("choices"):
                if error := chunk_dict.get("error"):
                    msg = (
                        f"OpenRouter API returned an error during streaming: "
                        f"{error.get('message', str(error))} "
                        f"(code: {error.get('code', 'unknown')})"
                    )
                    raise ValueError(msg)
                                # OpenRouter 特有参数
                if usage := chunk_dict.get("usage"):
                    usage_metadata = _create_usage_metadata(usage)
                    response_metadata = {}
                    if "cost" in usage:
                        response_metadata["cost"] = usage["cost"]
                    if "cost_details" in usage:
                        response_metadata["cost_details"] = usage["cost_details"]
                    usage_chunk = AIMessageChunk(
                        content="",
                        usage_metadata=usage_metadata,
                        response_metadata=response_metadata,
                    )
                    generation_chunk = ChatGenerationChunk(message=usage_chunk)
                    if run_manager:
                        await run_manager.on_llm_new_token(
                            token=generation_chunk.text, chunk=generation_chunk
                        )
                    yield generation_chunk
                continue
            choice = chunk_dict["choices"][0]
            message_chunk = _convert_chunk_to_message_chunk(
                chunk_dict, default_chunk_class
            )
            generation_info: dict[str, Any] = {}
            if choice.get("finish_reason"):
                candidate_generation_info = _create_stream_generation_info(
                    chunk_dict, choice, self.model_name
                )
                generation_info.update(
                    {
                        key: value
                        for key, value in candidate_generation_info.items()
                        if key not in terminal_generation_info
                    }
                )
                terminal_generation_info.update(generation_info)
            logprobs = choice.get("logprobs")
            if logprobs:
                generation_info["logprobs"] = logprobs

            if generation_info:
                generation_info["model_provider"] = "openrouter"
                message_chunk = message_chunk.model_copy(
                    update={
                        "response_metadata": {
                            **message_chunk.response_metadata,
                            **generation_info,
                        }
                    }
                )

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

    #
        # 提取顶层响应元数据
    #
    @property
    def _default_params(self) -> dict[str, Any]:  # noqa: C901, PLR0912
        """获取调用 OpenRouter API 的默认参数。"""
        params: dict[str, Any] = {
            "model": self.model_name,
            "stream": self.streaming,
            **self.model_kwargs,
        }
        if self.temperature is not None:
            params["temperature"] = self.temperature
        if self.max_tokens is not None:
            params["max_tokens"] = self.max_tokens
        if self.max_completion_tokens is not None:
            params["max_completion_tokens"] = self.max_completion_tokens
        if self.top_p is not None:
            params["top_p"] = self.top_p
        if self.frequency_penalty is not None:
            params["frequency_penalty"] = self.frequency_penalty
        if self.presence_penalty is not None:
            params["presence_penalty"] = self.presence_penalty
        if self.seed is not None:
            params["seed"] = self.seed
        if self.n > 1:
            params["n"] = self.n
        if self.stop is not None:
            params["stop"] = self.stop
                # 在 response_metadata 中展示 OpenRouter 成本数据
        if self.reasoning is not None:
            params["reasoning"] = self.reasoning
        if self.openrouter_provider is not None:
            params["provider"] = self.openrouter_provider
        if self.route is not None:
            params["route"] = self.route
        if self.plugins is not None:
            params["plugins"] = self.plugins
        if self.session_id:
            params["session_id"] = self.session_id
        if self.trace is not None:
            params["trace"] = self.trace
        return params

    def _create_message_dicts(
        self, messages: list[BaseMessage], stop: list[str] | None
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        params = self._default_params
        if stop is not None:
            params["stop"] = stop
        message_dicts = [_convert_message_to_dict(m) for m in messages]
        return message_dicts, params

    def _create_chat_result(self, response: Any) -> ChatResult:  # noqa: C901, PLR0912
        """从 OpenRouter SDK 响应中创建一个 `ChatResult`。"""
        if not isinstance(response, dict):
            response = response.model_dump(by_alias=True)

        if error := response.get("error"):
            msg = (
                f"OpenRouter API returned an error: "
                f"{error.get('message', str(error))} "
                f"(code: {error.get('code', 'unknown')})"
            )
            raise ValueError(msg)

        generations = []
        token_usage = response.get("usage") or {}

        choices = response.get("choices", [])
        if not choices:
            msg = (
                "OpenRouter API returned a response with no choices. "
                "This may indicate a problem with the request or model availability."
            )
            raise ValueError(msg)

                # 类型转换辅助函数
        response_model = response.get("model")
        system_fingerprint = response.get("system_fingerprint")
        provider = response.get("provider")

        for res in choices:
            message = _convert_dict_to_message(res["message"])
            if token_usage and isinstance(message, AIMessage):
                message.usage_metadata = _create_usage_metadata(token_usage)
                                # --- resolve file_data
                # ---------------------------------------------------
                if "cost" in token_usage:
                    message.response_metadata["cost"] = token_usage["cost"]
                if "cost_details" in token_usage:
                    message.response_metadata["cost_details"] = token_usage[
                        "cost_details"
                    ]
            if isinstance(message, AIMessage):
                if provider:
                    message.response_metadata["provider"] = provider
                if system_fingerprint:
                    message.response_metadata["system_fingerprint"] = system_fingerprint
                if native_finish_reason := res.get("native_finish_reason"):
                    message.response_metadata["native_finish_reason"] = (
                        native_finish_reason
                    )
            generation_info: dict[str, Any] = {
                "finish_reason": res.get("finish_reason"),
            }
            if "logprobs" in res:
                generation_info["logprobs"] = res["logprobs"]
            gen = ChatGeneration(
                message=message,
                generation_info=generation_info,
            )
            generations.append(gen)

        llm_output: dict[str, Any] = {
            "model_name": response_model or self.model_name,
        }
        if response_id := response.get("id"):
            llm_output["id"] = response_id
        if created := response.get("created"):
            llm_output["created"] = int(created)
        if object_ := response.get("object"):
            llm_output["object"] = object_
        return ChatResult(generations=generations, llm_output=llm_output)

    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type[BaseModel] | Callable | BaseTool],
        *,
        tool_choice: dict | str | bool | None = None,
        strict: bool | None = None,
        parallel_tool_calls: bool | None = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, AIMessage]:
        """将工具类对象绑定到此聊天模型。

        Args:
            tools: 要绑定到此聊天模型的工具定义列表。

                支持由 `langchain_core.utils.function_calling.convert_to_openai_tool`
                处理的任何工具定义。 tool_choice: 要求模型调用的工具。 strict: 如果为
                `True`，模型输出将保证完全匹配工具定义中提供的 JSON Schema。 如果为
                `None`，则不会向模型传递 `strict` 参数。 parallel_tool_calls: 设置为
                `False` 以禁用并行工具调用。 默认为
                `None`（无指定，允许并行工具调用）。 **kwargs: 任意其他参数。

                模型包装器，返回格式与给定模式匹配的输出。
        """
        if parallel_tool_calls is not None:
            kwargs["parallel_tool_calls"] = parallel_tool_calls
        formatted_tools = [
            convert_to_openai_tool(tool, strict=strict) for tool in tools
        ]
        if tool_choice is not None and tool_choice:
            if tool_choice == "any":
                tool_choice = "required"
            if isinstance(tool_choice, str) and (
                tool_choice not in ("auto", "none", "required")
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

    def with_structured_output(  # type: ignore[override]
        self,
        schema: dict | type[BaseModel] | None = None,
        *,
        method: Literal["function_calling", "json_schema"] = "function_calling",
        include_raw: bool = False,
        strict: bool | None = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, dict | BaseModel]:
        """封装模型，返回与给定模式匹配的输出格式。

        Args:
            schema: The output schema as a Pydantic class, TypedDict, JSON Schema,
                or OpenAI function schema.
            method: The method for steering model generation.
            include_raw: If `True` then both the raw model response and the
                parsed model response will be returned.
            strict: If `True`, model output is guaranteed to exactly match the
                JSON Schema provided in the schema definition.

                若为 `None`，则 `strict` 参数不会传递给模型。 **kwargs: 任意其他参数。

        Returns:
            一个与 `BaseChatModel` 输入相同的 `Runnable`。
        """
        if method == "json_mode":
            warnings.warn(
                "Unrecognized structured output method 'json_mode'. "
                "Defaulting to 'json_schema' method.",
                stacklevel=2,
            )
            method = "json_schema"
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
                strict=strict,
                ls_structured_output_format={
                    "kwargs": {"method": "function_calling", "strict": strict},
                    "schema": formatted_tool,
                },
                **kwargs,
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
            json_schema = convert_to_json_schema(schema)
            schema_name = json_schema.get("title", "")
            json_schema_spec: dict[str, Any] = {
                "name": schema_name,
                "schema": json_schema,
            }
            if strict is not None:
                json_schema_spec["strict"] = strict
            response_format = {
                "type": "json_schema",
                "json_schema": json_schema_spec,
            }
            ls_format_info = {
                "kwargs": {"method": "json_schema", "strict": strict},
                "schema": json_schema,
            }
            llm = self.bind(
                response_format=response_format,
                ls_structured_output_format=ls_format_info,
                **kwargs,
            )
            output_parser = (
                PydanticOutputParser(pydantic_object=schema)  # type: ignore[type-var, arg-type]
                if is_pydantic_schema
                else JsonOutputParser()
            )
        else:
            msg = (
                f"Unrecognized method argument. Expected one of 'function_calling' "
                f"or 'json_schema'. Received: '{method}'"
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


def _strip_internal_kwargs(params: dict[str, Any]) -> None:
    """移除 SDK 不接受的 LangChain 内部键。"""
    for key in _INTERNAL_KWARGS:
        params.pop(key, None)


#
# --- resolve filename ----------------------------------------------------
#
def _convert_video_block_to_openrouter(block: dict[str, Any]) -> dict[str, Any]:
    """将 LangChain 视频内容块转换为 OpenRouter 的 `video_url` 格式。

    Args:
        block: 一个 LangChain `VideoContentBlock`。

    Returns:
        一个符合 OpenRouter `video_url` 格式的字典。

    Raises:
        ValueError: 如果未提供视频源。
    """
    if "url" in block:
        return {"type": "video_url", "video_url": {"url": block["url"]}}
    if "base64" in block or block.get("source_type") == "base64":
        base64_data = block["data"] if "source_type" in block else block["base64"]
        mime_type = block.get("mime_type", "video/mp4")
        return {
            "type": "video_url",
            "video_url": {"url": f"data:{mime_type};base64,{base64_data}"},
        }
    msg = "Video block must have either 'url' or 'base64' data."
    raise ValueError(msg)


def _convert_file_block_to_openrouter(block: dict[str, Any]) -> dict[str, Any]:
    """将 LangChain 文件内容块转换为 OpenRouter 的 `file` 格式。

    OpenRouter accepts files as::

        {"type": "file", "file": {"filename": "...", "file_data": "..."}}

    其中 `file_data` 可以是公共 URL 或 `data:` URI。

    Args:
        block: 一个 LangChain 文件内容块。

    Returns:
        一个符合 OpenRouter `file` 格式的字典。

    Raises:
        ValueError: 如果该块既不含 URL，也不含 base64 数据，也不含文件 ID。
    """
    file: dict[str, str] = {}

        # 若无索引，则无法区分流式片段与独立的非流式条目，因此保持原样。对任何上游传入的非字典项也作同样处理。
    if "url" in block:
        file["file_data"] = block["url"]
    elif block.get("source_type") == "base64" or "base64" in block:
        base64_data = block["data"] if "source_type" in block else block["base64"]
        mime_type = block.get("mime_type", "application/octet-stream")
        file["file_data"] = f"data:{mime_type};base64,{base64_data}"
    elif block.get("source_type") == "id" or "file_id" in block:
        msg = "OpenRouter does not support file IDs."
        raise ValueError(msg)
    else:
        msg = "File block must have either 'url' or 'base64' data."
        raise ValueError(msg)

        # 从列表内容中过滤掉非文本块
    if filename := block.get("filename"):
        file["filename"] = filename
    elif ((extras := block.get("extras")) and "filename" in extras) or (
        (extras := block.get("metadata")) and "filename" in extras
    ):
        file["filename"] = extras["filename"]

    return {"type": "file", "file": file}


def _format_message_content(content: Any) -> Any:
    """为 OpenRouter API 格式化消息内容。

    将 LangChain 数据内容块转换为期望格式。

    Args:
        content: 消息内容（字符串或内容块列表）。

    Returns:
        适用于 OpenRouter API 的格式化内容。
    """
    if content and isinstance(content, list):
        formatted: list = []
        for block in content:
            if isinstance(block, dict) and is_data_content_block(block):
                if block.get("type") == "video":
                    formatted.append(_convert_video_block_to_openrouter(block))
                elif block.get("type") == "file":
                    formatted.append(_convert_file_block_to_openrouter(block))
                else:
                    formatted.append(convert_to_openai_data_block(block))
            else:
                formatted.append(block)
        return formatted
    return content


def _merge_reasoning_run(run: list[dict[str, Any]]) -> dict[str, Any]:
    """合并连续的相同 `(type, index)` 推理片段。"""
    merged_entry: dict[str, Any] = {}
    text_parts: list[str] = []
    has_text = False
    for frag in run:
        for k, v in frag.items():
            if k == "text":
                has_text = True
                if v:
                    text_parts.append(v)
            elif v is not None:
                merged_entry[k] = v
    if has_text:
        merged_entry["text"] = "".join(text_parts)
    return merged_entry


def _strip_ephemeral_reasoning_ids(value: Any) -> Any:
    """从传出负载中移除 OpenAI 响应的推理项 ID。"""
    if isinstance(value, list):
        return [_strip_ephemeral_reasoning_ids(item) for item in value]
    if isinstance(value, dict):
        return {
            key: _strip_ephemeral_reasoning_ids(item)
            for key, item in value.items()
            if not (key == "id" and isinstance(item, str) and item.startswith("rs_"))
        }
    return value


def _merge_reasoning_details(
    details: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """合并流式分块拼接产生的碎片化 `reasoning_details`。

    During streaming, `AIMessageChunk.__add__` list-concatenates
    `reasoning_details` in `additional_kwargs`, fragmenting a single entry
    into many. When serialized back to the API via
    `_convert_message_to_dict`, these fragments cause
    `BadRequestResponseError` on multi-turn conversations (the provider
    rejects the malformed thinking block with `Invalid signature`).

    Streaming deltas tag each fragment with the `index` of the entry it
    belongs to in the original (non-streamed) array, so this function groups
    consecutive entries by `(type, index)` and merges each group into one.
    Entries without an `index` are preserved as-is, since non-streaming
    responses can legitimately contain multiple entries.

    在合并组内，`text` 值按顺序拼接。其他元数据字段（如
    `format`、`signature`）采用最后非 `None`
    值优先策略，以保留稳定的提供商元数据，而不拼接重复字符串——Anthropic
    风格的推理流在块末尾仅发出一个带签名的碎片。

    包含零个或一个项目的列表保持不变。
    """
    if not isinstance(details, list) or len(details) <= 1:
        return details

    merged: list[dict[str, Any]] = []
    i = 0
    while i < len(details):
        entry = details[i]
                # 保留多轮对话中的推理内容（例如
                # 工具调用循环）。OpenRouter 将推理文本存储在 `reasoning`
        # 中，结构化片段详情存储在 `reasoning_details`
        # 中；后者在序列化前合并，以还原流式分段。
                # 在 response_metadata 中展示 OpenRouter 成本数据
        if not isinstance(entry, dict) or entry.get("index") is None:
            merged.append(entry)
            i += 1
            continue

        entry_type = entry.get("type", "")
        entry_index = entry["index"]
        run = [entry]
        i += 1
        while i < len(details):
            nxt = details[i]
            if (
                isinstance(nxt, dict)
                and nxt.get("type", "") == entry_type
                and nxt.get("index") == entry_index
            ):
                run.append(nxt)
                i += 1
            else:
                break

        merged.append(entry if len(run) == 1 else _merge_reasoning_run(run))

    return merged


def _convert_message_to_dict(message: BaseMessage) -> dict[str, Any]:  # noqa: C901, PLR0912
    """将 LangChain 消息转换为 OpenRouter 兼容的字典负载。

    Handles role mapping, multimodal content formatting, tool call
    serialization, and reasoning content preservation for multi-turn
    conversations.

    Args:
        message: LangChain 消息。

    Returns:
        适用于 OpenRouter 聊天 API `messages` 参数的字典。
    """
    message_dict: dict[str, Any]
    if isinstance(message, ChatMessage):
        message_dict = {"role": message.role, "content": message.content}
    elif isinstance(message, HumanMessage):
        message_dict = {
            "role": "user",
            "content": _format_message_content(message.content),
        }
    elif isinstance(message, AIMessage):
        message_dict = {"role": "assistant", "content": message.content}
                # 从列表内容中过滤掉非文本块
        if isinstance(message.content, list):
            text_blocks = [
                block
                for block in message.content
                if isinstance(block, dict) and block.get("type") == "text"
            ]
            message_dict["content"] = text_blocks or ""
        if message.tool_calls or message.invalid_tool_calls:
            message_dict["tool_calls"] = [
                _lc_tool_call_to_openrouter_tool_call(tc) for tc in message.tool_calls
            ] + [
                _lc_invalid_tool_call_to_openrouter_tool_call(tc)
                for tc in message.invalid_tool_calls
            ]
            if message_dict["content"] == "" or (
                isinstance(message_dict["content"], list)
                and not message_dict["content"]
            ):
                message_dict["content"] = None
        elif "tool_calls" in message.additional_kwargs:
            message_dict["tool_calls"] = message.additional_kwargs["tool_calls"]
            if message_dict["content"] == "" or (
                isinstance(message_dict["content"], list)
                and not message_dict["content"]
            ):
                message_dict["content"] = None
                # 保留多轮对话中的推理内容（例如。
                # tool-calling loops). OpenRouter 将推理文本存储在 `reasoning` 中
                # 以及 `reasoning_details` 中的结构化片段详情；后者
                # 在序列化之前合并，以撤销流式分段。
        if "reasoning_content" in message.additional_kwargs:
            message_dict["reasoning"] = message.additional_kwargs["reasoning_content"]
        if "reasoning_details" in message.additional_kwargs:
            message_dict["reasoning_details"] = _strip_ephemeral_reasoning_ids(
                _merge_reasoning_details(message.additional_kwargs["reasoning_details"])
            )
    elif isinstance(message, SystemMessage):
        message_dict = {"role": "system", "content": message.content}
    elif isinstance(message, ToolMessage):
        message_dict = {
            "role": "tool",
            "content": message.content,
            "tool_call_id": message.tool_call_id,
        }
    else:
        msg = f"Got unknown type {message}"
        raise TypeError(msg)
    if "name" in message.additional_kwargs:
        message_dict["name"] = message.additional_kwargs["name"]
    return message_dict


def _convert_dict_to_message(_dict: Mapping[str, Any]) -> BaseMessage:  # noqa: C901
    """将 OpenRouter API 响应消息字典转换为 LangChain 消息。

    提取工具调用、推理内容，并将角色映射到相应的 LangChain
    消息类型（`HumanMessage`、`AIMessage`、`SystemMessage`、`ToolMessage` 或
    `ChatMessage`）。

    Args:
        _dict: 来自 API 响应的消息字典。

    Returns:
        对应的 LangChain 消息。
    """
    id_ = _dict.get("id")
    role = _dict.get("role")
    if role == "user":
        return HumanMessage(content=_dict.get("content", ""))
    if role == "assistant":
        content = _dict.get("content", "") or ""
        additional_kwargs: dict = {}
        if reasoning := _dict.get("reasoning"):
            additional_kwargs["reasoning_content"] = reasoning
        if reasoning_details := _dict.get("reasoning_details"):
            additional_kwargs["reasoning_details"] = reasoning_details
        tool_calls = []
        invalid_tool_calls = []
        if raw_tool_calls := _dict.get("tool_calls"):
            for raw_tool_call in raw_tool_calls:
                try:
                    tool_calls.append(parse_tool_call(raw_tool_call, return_id=True))
                except Exception as e:  # noqa: BLE001, PERF203
                    invalid_tool_calls.append(
                        make_invalid_tool_call(raw_tool_call, str(e))
                    )
        return AIMessage(
            content=content,
            id=id_,
            additional_kwargs=additional_kwargs,
            tool_calls=tool_calls,
            invalid_tool_calls=invalid_tool_calls,
            response_metadata={"model_provider": "openrouter"},
        )
    if role == "system":
        return SystemMessage(content=_dict.get("content", ""))
    if role == "tool":
        additional_kwargs = {}
        if "name" in _dict:
            additional_kwargs["name"] = _dict["name"]
        return ToolMessage(
            content=_dict.get("content", ""),
            tool_call_id=_dict.get("tool_call_id"),
            additional_kwargs=additional_kwargs,
        )
    if role is None:
        msg = (
            f"OpenRouter response message is missing the 'role' field. "
            f"Message keys: {list(_dict.keys())}"
        )
        raise ValueError(msg)
    warnings.warn(
        f"Unrecognized message role '{role}' from OpenRouter. "
        f"Falling back to ChatMessage.",
        stacklevel=2,
    )
    return ChatMessage(content=_dict.get("content", ""), role=role)


def _convert_chunk_to_message_chunk(  # noqa: C901, PLR0911, PLR0912
    chunk: Mapping[str, Any], default_class: type[BaseMessageChunk]
) -> BaseMessageChunk:
    """将流式分块字典转换为 LangChain 消息分块。

    Args:
        chunk: 流式分块字典。
        default_class: 默认消息分块类。

    Returns:
        LangChain 消息分块。
    """
    choice = chunk["choices"][0]
    _dict = choice.get("delta", {})
    role = cast("str", _dict.get("role"))
    content = cast("str", _dict.get("content") or "")
    additional_kwargs: dict = {}
    tool_call_chunks: list = []

    if raw_tool_calls := _dict.get("tool_calls"):
        for rtc in raw_tool_calls:
            try:
                tool_call_chunks.append(
                    tool_call_chunk(
                        name=rtc["function"].get("name"),
                        args=rtc["function"].get("arguments"),
                        id=rtc.get("id"),
                        index=rtc["index"],
                    )
                )
            except (KeyError, TypeError, AttributeError):  # noqa: PERF203
                warnings.warn(
                    f"Skipping malformed tool call chunk during streaming: "
                    f"unexpected structure in {rtc!r}.",
                    stacklevel=2,
                )

    if role == "user" or default_class == HumanMessageChunk:
        return HumanMessageChunk(content=content)
    if role == "assistant" or default_class == AIMessageChunk:
        if reasoning := _dict.get("reasoning"):
            additional_kwargs["reasoning_content"] = reasoning
        if reasoning_details := _dict.get("reasoning_details"):
            additional_kwargs["reasoning_details"] = reasoning_details
        usage_metadata = None
        response_metadata: dict[str, Any] = {"model_provider": "openrouter"}
        if usage := chunk.get("usage"):
            usage_metadata = _create_usage_metadata(usage)
                        # 在 response_metadata 中暴露 OpenRouter 成本数据
            if "cost" in usage:
                response_metadata["cost"] = usage["cost"]
            if "cost_details" in usage:
                response_metadata["cost_details"] = usage["cost_details"]
        return AIMessageChunk(
            content=content,
            additional_kwargs=additional_kwargs,
            tool_call_chunks=tool_call_chunks,  # type: ignore[arg-type]
            usage_metadata=usage_metadata,  # type: ignore[arg-type]
            response_metadata=response_metadata,
        )
    if role == "system" or default_class == SystemMessageChunk:
        return SystemMessageChunk(content=content)
    if role == "tool" or default_class == ToolMessageChunk:
        return ToolMessageChunk(
            content=content, tool_call_id=_dict.get("tool_call_id", "")
        )
    if role:
        warnings.warn(
            f"Unrecognized streaming chunk role '{role}' from OpenRouter. "
            f"Falling back to ChatMessageChunk.",
            stacklevel=2,
        )
        return ChatMessageChunk(content=content, role=role)
    if default_class is ChatMessageChunk:
        return ChatMessageChunk(content=content, role=role or "")
    return default_class(content=content)  # type: ignore[call-arg]


def _lc_tool_call_to_openrouter_tool_call(tool_call: ToolCall) -> dict[str, Any]:
    """将 LangChain `ToolCall` 转换为 OpenRouter 工具调用字典。

    通过 `json.dumps` 序列化 `args`（一个字典）。
    """
    return {
        "type": "function",
        "id": tool_call["id"],
        "function": {
            "name": tool_call["name"],
            "arguments": json.dumps(tool_call["args"], ensure_ascii=False),
        },
    }


def _lc_invalid_tool_call_to_openrouter_tool_call(
    invalid_tool_call: InvalidToolCall,
) -> dict[str, Any]:
    """将 LangChain `InvalidToolCall` 转换为 OpenRouter 工具调用字典。

    与有效版本不同，`args` 已经是原始字符串（而非字典），直接原样传递。
    """
    return {
        "type": "function",
        "id": invalid_tool_call["id"],
        "function": {
            "name": invalid_tool_call["name"],
            "arguments": invalid_tool_call["args"],
        },
    }


def _create_usage_metadata(token_usage: dict[str, Any]) -> UsageMetadata:
    """从 OpenRouter token 使用量响应中创建使用量元数据。

    OpenRouter may return token counts as floats rather than ints, so all
    values are explicitly cast to int.

    Args:
        token_usage: 来自 API 响应的 token 使用量字典。

    Returns:
        包含输入/输出 token 详情的使用量元数据。
    """
    _input = token_usage.get("prompt_tokens")
    input_tokens = int(
        _input if _input is not None else (token_usage.get("input_tokens") or 0)
    )
    _output = token_usage.get("completion_tokens")
    output_tokens = int(
        _output if _output is not None else (token_usage.get("output_tokens") or 0)
    )
    _total = token_usage.get("total_tokens")
    total_tokens = int(_total if _total is not None else input_tokens + output_tokens)

    input_details_dict = (
        token_usage.get("prompt_tokens_details")
        or token_usage.get("input_tokens_details")
        or {}
    )
    output_details_dict = (
        token_usage.get("completion_tokens_details")
        or token_usage.get("output_tokens_details")
        or {}
    )

    cache_read = input_details_dict.get("cached_tokens")
    cache_creation = input_details_dict.get("cache_write_tokens")
    input_token_details: dict = {
        "cache_read": int(cache_read) if cache_read is not None else None,
        "cache_creation": int(cache_creation) if cache_creation is not None else None,
    }
    reasoning_tokens = output_details_dict.get("reasoning_tokens")
    output_token_details: dict = {
        "reasoning": int(reasoning_tokens) if reasoning_tokens is not None else None,
    }
    usage_metadata: UsageMetadata = {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
    }

    filtered_input = {k: v for k, v in input_token_details.items() if v is not None}
    if filtered_input:
        usage_metadata["input_token_details"] = InputTokenDetails(**filtered_input)  # type: ignore[typeddict-item]
    filtered_output = {k: v for k, v in output_token_details.items() if v is not None}
    if filtered_output:
        usage_metadata["output_token_details"] = OutputTokenDetails(**filtered_output)  # type: ignore[typeddict-item]
    return usage_metadata

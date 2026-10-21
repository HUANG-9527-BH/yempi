"""DeepSeek 对话模型。"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator, Sequence
from json import JSONDecodeError
from typing import Any, Literal, TypeAlias, cast
from urllib.parse import urlparse

import openai
from langchain_core.callbacks import (
    CallbackManagerForLLMRun,
)
from langchain_core.language_models import (
    LangSmithParams,
    LanguageModelInput,
    ModelProfile,
    ModelProfileRegistry,
)
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain_core.messages.ai import InputTokenDetails, UsageMetadata
from langchain_core.outputs import ChatGenerationChunk, ChatResult
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from langchain_core.utils import from_env, secret_from_env
from langchain_openai.chat_models.base import BaseChatOpenAI
from pydantic import BaseModel, ConfigDict, Field, SecretStr, model_validator
from typing_extensions import Self

from langchain_deepseek._version import __version__
from langchain_deepseek.data._profiles import _PROFILES

DEFAULT_API_BASE = "https://api.deepseek.com/v1"
DEFAULT_BETA_API_BASE = "https://api.deepseek.com/beta"

_DictOrPydanticClass: TypeAlias = dict[str, Any] | type[BaseModel]
_DictOrPydantic: TypeAlias = dict[str, Any] | BaseModel


_MODEL_PROFILES = cast("ModelProfileRegistry", _PROFILES)


def _get_default_model_profile(model_name: str) -> ModelProfile:
    default = _MODEL_PROFILES.get(model_name) or {}
    return default.copy()


def _get_prompt_cache_hit_tokens(response: dict | openai.BaseModel) -> int | None:
    """读取 DeepSeek 的顶层 `prompt_cache_hit_tokens` 使用字段。

    DeepSeek 将上下文缓存使用情况报告为顶层的 `prompt_cache_hit_tokens` 和
    `prompt_cache_miss_tokens` 字段，而非 OpenAI 的嵌套
    `prompt_tokens_details.cached_tokens`，因此该计数对基类不可见。

    Args:
        response: 一个对话完成响应或流式数据块。

    Returns:
        The number of prompt tokens served from the context cache, or `None` if
        the field is absent.
    """
    usage: Any = (
        response.get("usage")
        if isinstance(response, dict)
        else getattr(response, "usage", None)
    )
    if isinstance(usage, openai.BaseModel):
                # Fields DeepSeek adds beyond OpenAI's schema are preserved as extras,
                # which `model_dump` includes.
        usage = usage.model_dump()
    if not isinstance(usage, dict):
        return None
    cache_hit_tokens = usage.get("prompt_cache_hit_tokens")
    if not isinstance(cache_hit_tokens, int):
        return None
    return cache_hit_tokens


def _add_cache_read_tokens(message: BaseMessage, cache_hit_tokens: int) -> None:
    """将缓存的提示 token 记录为消息使用元数据中的 `cache_read`。

    仅记录缓存命中。DeepSeek 定义 `prompt_tokens` 为 `prompt_cache_hit_tokens +
    prompt_cache_miss_tokens`，因此缓存未命中是 普通的未缓存输入
    token，而非缓存写入，将其映射到 `cache_creation` 会错误报告。

    An existing `cache_read` count is left untouched, since DeepSeek served
    through an OpenAI-compatible gateway may report the nested form that the
    base class already handles.

    Args:
        message: 需要更新其 `usage_metadata` 的消息。
        cache_hit_tokens: 从 DeepSeek 上下文缓存中提供的提示 token 数量。
    """
    if not isinstance(message, AIMessage) or message.usage_metadata is None:
        return
    input_token_details = message.usage_metadata.get("input_token_details") or {}
    if "cache_read" in input_token_details:
        return
    usage_metadata: dict[str, Any] = dict(message.usage_metadata)
    usage_metadata["input_token_details"] = cast(
        "InputTokenDetails",
        {**input_token_details, "cache_read": cache_hit_tokens},
    )
    message.usage_metadata = cast("UsageMetadata", usage_metadata)


class ChatDeepSeek(BaseChatOpenAI):
    """DeepSeek 对话模型集成，用于访问 DeepSeek API 托管的模型。

    设置：
        安装 `langchain-deepseek` 并设置环境变量 `DEEPSEEK_API_KEY`。

        ```bash
        pip install -U langchain-deepseek
        export DEEPSEEK_API_KEY="your-api-key"
        ```

    关键初始化参数 —— 完成参数：
        model:
            要使用的 DeepSeek 模型名称，例如 `'deepseek-chat'`。
        temperature:
            采样温度。
        max_tokens:
            生成的最大 token 数量。

    关键初始化参数 —— 客户端参数：
        timeout:
            请求超时时间。
        max_retries:
            最大重试次数。
        api_key:
            DeepSeek API 密钥。若未传入，则从环境变量 `DEEPSEEK_API_KEY` 读取。

    有关支持的初始化参数及其描述的完整列表，请参阅参数部分。

    实例化：
        ```python
        from langchain_deepseek import ChatDeepSeek

        model = ChatDeepSeek(
            model="...",
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=2,
            # api_key="...",
            # other params...
        )
        ```

    调用：
        ```python
        messages = [
            ("system", "You are a helpful translator. Translate the user sentence to French."),
            ("human", "I love programming."),
        ]
        model.invoke(messages)
        ```

    流式：
        ```python
        for chunk in model.stream(messages):
            print(chunk.text, end="")
        ```
        ```python
        stream = model.stream(messages)
        full = next(stream)
        for chunk in stream:
            full += chunk
        full
        ```

    异步：
        ```python
        await model.ainvoke(messages)

        # stream:
        # async for chunk in (await model.astream(messages))

        # batch:
        # await model.abatch([messages])
        ```

    工具调用：
        ```python
        from pydantic import BaseModel, Field


        class GetWeather(BaseModel):
            '''Get the current weather in a given location'''

            location: str = Field(..., description="The city and state, e.g. San Francisco, CA")


        class GetPopulation(BaseModel):
            '''Get the current population in a given location'''

            location: str = Field(..., description="The city and state, e.g. San Francisco, CA")


        model_with_tools = model.bind_tools([GetWeather, GetPopulation])
        ai_msg = model_with_tools.invoke("Which city is hotter today and which is bigger: LA or NY?")
        ai_msg.tool_calls
        ```

        更多信息请参见 `ChatDeepSeek.bind_tools()` 方法。

    结构化输出：
        ```python
        from typing import Optional

        from pydantic import BaseModel, Field


        class Joke(BaseModel):
            '''Joke to tell user.'''

            setup: str = Field(description="The setup of the joke")
            punchline: str = Field(description="The punchline to the joke")
            rating: int | None = Field(description="How funny the joke is, from 1 to 10")


        structured_model = model.with_structured_output(Joke)
        structured_model.invoke("Tell me a joke about cats")
        ```

        更多信息请参见 `ChatDeepSeek.with_structured_output()`。

    Token 使用情况：
        ```python
        ai_msg = model.invoke(messages)
        ai_msg.usage_metadata
        ```
        ```python
        {"input_tokens": 28, "output_tokens": 5, "total_tokens": 33}
        ```

    响应元数据：
        ```python
        ai_msg = model.invoke(messages)
        ai_msg.response_metadata
        ```
    """  # noqa: E501

    model_name: str = Field(alias="model")
    """模型的名称"""
    api_key: SecretStr | None = Field(
        default_factory=secret_from_env("DEEPSEEK_API_KEY", default=None),
    )
    """DeepSeek API 密钥"""
    api_base: str = Field(
        alias="base_url",
        default_factory=from_env("DEEPSEEK_API_BASE", default=DEFAULT_API_BASE),
    )
    """DeepSeek API 基础 URL。

    若未提供，则自动从环境变量 `DEEPSEEK_API_BASE` 读取。
    """

    model_config = ConfigDict(populate_by_name=True)

    @property
    def _is_azure_endpoint(self) -> bool:
        """检查配置的端点是否为 Azure 部署。"""
        hostname = urlparse(self.api_base or "").hostname or ""
        return hostname == "azure.com" or hostname.endswith(".azure.com")

    @property
    def _llm_type(self) -> str:
        """聊天模型的返回类型。"""
        return "chat-deepseek"

    @property
    def lc_secrets(self) -> dict[str, str]:
        """构造函数参数名到密钥 ID 的映射。"""
        return {"api_key": "DEEPSEEK_API_KEY"}

    def _get_ls_params(
        self,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> LangSmithParams:
        ls_params = super()._get_ls_params(stop=stop, **kwargs)
        ls_params["ls_provider"] = "deepseek"
        return ls_params

    @model_validator(mode="after")
    def _set_deepseek_version(self) -> Self:
        """在元数据中设置包版本。

        唯一命名以避免与 `BaseChatOpenAI._set_openai_chat_version` 冲突；
        Pydantic 会替换同名验证器而非链式调用。
        """
        self._add_version("langchain-deepseek", __version__)
        return self

    @model_validator(mode="after")
    def validate_environment(self) -> Self:
        """验证必要的环境变量和客户端参数。"""
        if self.api_base == DEFAULT_API_BASE and not (
            self.api_key and self.api_key.get_secret_value()
        ):
            msg = "If using default api base, DEEPSEEK_API_KEY must be set."
            raise ValueError(msg)
        client_params: dict = {
            k: v
            for k, v in {
                "api_key": self.api_key.get_secret_value() if self.api_key else None,
                "base_url": self.api_base,
                "timeout": self.request_timeout,
                "max_retries": self.max_retries,
                "default_headers": self.default_headers,
                "default_query": self.default_query,
            }.items()
            if v is not None
        }

        if not (self.client or None):
            sync_specific: dict = {"http_client": self.http_client}
            self.root_client = openai.OpenAI(**client_params, **sync_specific)
            self.client = self.root_client.chat.completions
        if not (self.async_client or None):
            async_specific: dict = {"http_client": self.http_async_client}
            self.root_async_client = openai.AsyncOpenAI(
                **client_params,
                **async_specific,
            )
            self.async_client = self.root_async_client.chat.completions
        return self

    def _resolve_model_profile(self) -> ModelProfile | None:
        return _get_default_model_profile(self.model_name) or None

    def _with_beta_api_base(self) -> Self:
        """返回此模型的一个副本，目标为 DeepSeek 的测试版端点。"""
        beta_model = self.model_copy(
            update={
                "api_base": DEFAULT_BETA_API_BASE,
                "client": None,
                "async_client": None,
                "root_client": None,
                "root_async_client": None,
            }
        )
        return beta_model.validate_environment()  # type: ignore[operator]

    def _get_request_payload(
        self,
        input_: LanguageModelInput,
        *,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> dict:
        payload = super()._get_request_payload(input_, stop=stop, **kwargs)
        for message in payload["messages"]:
            if message["role"] == "tool" and isinstance(message["content"], list):
                message["content"] = json.dumps(message["content"])
            elif message["role"] == "assistant" and isinstance(
                message["content"], list
            ):
                                # DeepSeek API expects assistant content to be a string, not a list.
                                # Extract text blocks and join them, or use empty string if none exist.
                text_parts = [
                    block.get("text", "")
                    for block in message["content"]
                    if isinstance(block, dict) and block.get("type") == "text"
                ]
                message["content"] = "".join(text_parts) if text_parts else ""

                # Azure-hosted DeepSeek does not support the dict/object form of
                # tool_choice (e.g. {"type": "function", "function": {"name": "..."}}).
                # It only accepts string values: "none", "auto", or "required".
                # Convert the unsupported dict form to "required", which is the closest
                # string equivalent — it forces the model to call a tool without
                # constraining which one. In the common with_structured_output() case
                # only a single tool is bound, so the behavior is effectively identical.
        if self._is_azure_endpoint and isinstance(payload.get("tool_choice"), dict):
            payload["tool_choice"] = "required"

        return payload

    def _create_chat_result(
        self,
        response: dict | openai.BaseModel,
        generation_info: dict | None = None,
    ) -> ChatResult:
        rtn = super()._create_chat_result(response, generation_info)

        cache_hit_tokens = _get_prompt_cache_hit_tokens(response)
        if cache_hit_tokens is not None:
            for generation in rtn.generations:
                _add_cache_read_tokens(generation.message, cache_hit_tokens)

        if not isinstance(response, openai.BaseModel):
            return rtn

        for generation in rtn.generations:
            if generation.message.response_metadata is None:
                generation.message.response_metadata = {}
            generation.message.response_metadata["model_provider"] = "deepseek"

        choices = getattr(response, "choices", None)
        if choices and hasattr(choices[0].message, "reasoning_content"):
            rtn.generations[0].message.additional_kwargs["reasoning_content"] = choices[
                0
            ].message.reasoning_content
                # Handle use via OpenRouter
        elif choices and hasattr(choices[0].message, "model_extra"):
            model_extra = choices[0].message.model_extra
            if isinstance(model_extra, dict) and (
                reasoning := model_extra.get("reasoning")
            ):
                rtn.generations[0].message.additional_kwargs["reasoning_content"] = (
                    reasoning
                )

        return rtn

    def _convert_chunk_to_generation_chunk(
        self,
        chunk: dict,
        default_chunk_class: type,
        base_generation_info: dict | None,
    ) -> ChatGenerationChunk | None:
        generation_chunk = super()._convert_chunk_to_generation_chunk(
            chunk,
            default_chunk_class,
            base_generation_info,
        )
                # Usage arrives in a trailing chunk that carries no choices, so this
                # cannot be folded into the choices branch below.
        if generation_chunk:
            cache_hit_tokens = _get_prompt_cache_hit_tokens(chunk)
            if cache_hit_tokens is not None:
                _add_cache_read_tokens(generation_chunk.message, cache_hit_tokens)

        if (choices := chunk.get("choices")) and generation_chunk:
            top = choices[0]
            if isinstance(generation_chunk.message, AIMessageChunk):
                generation_chunk.message.response_metadata = {
                    **generation_chunk.message.response_metadata,
                    "model_provider": "deepseek",
                }
                if (
                    reasoning_content := top.get("delta", {}).get("reasoning_content")
                ) is not None:
                    generation_chunk.message.additional_kwargs["reasoning_content"] = (
                        reasoning_content
                    )
                                # Handle use via OpenRouter
                elif (reasoning := top.get("delta", {}).get("reasoning")) is not None:
                    generation_chunk.message.additional_kwargs["reasoning_content"] = (
                        reasoning
                    )

        return generation_chunk

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        try:
            yield from super()._stream(
                messages,
                stop=stop,
                run_manager=run_manager,
                **kwargs,
            )
        except JSONDecodeError as e:
            msg = (
                "DeepSeek API returned an invalid response. "
                "Please check the API status and try again."
            )
            raise JSONDecodeError(
                msg,
                e.doc,
                e.pos,
            ) from e

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        try:
            return super()._generate(
                messages,
                stop=stop,
                run_manager=run_manager,
                **kwargs,
            )
        except JSONDecodeError as e:
            msg = (
                "DeepSeek API returned an invalid response. "
                "Please check the API status and try again."
            )
            raise JSONDecodeError(
                msg,
                e.doc,
                e.pos,
            ) from e

    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type | Callable | BaseTool],
        *,
        tool_choice: dict | str | bool | None = None,
        strict: bool | None = None,
        parallel_tool_calls: bool | None = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, AIMessage]:
        """将工具类对象绑定到此聊天模型。

        重写父类方法，当 `strict=True` 时使用测试版端点。

        Args:
            tools: 要绑定到此聊天模型的工具定义列表。
            tool_choice: 要求模型调用的工具。
            strict: 如果为 True，使用测试版 API 进行严格模式的 schema 验证。
            parallel_tool_calls: 设置为 `False` 以禁用并行工具调用。
            **kwargs: 传递给父级 `bind_tools` 的其他参数。

        Returns:
            一个接受与聊天模型相同输入的 Runnable。
        """
                # If strict mode is enabled and using default API base, switch to beta endpoint
        if strict is True and self.api_base == DEFAULT_API_BASE:
                        # Create a new instance with beta endpoint
            beta_model = self._with_beta_api_base()
            return beta_model.bind_tools(
                tools,
                tool_choice=tool_choice,
                strict=strict,
                parallel_tool_calls=parallel_tool_calls,
                **kwargs,
            )

                # Otherwise use parent implementation
        return super().bind_tools(
            tools,
            tool_choice=tool_choice,
            strict=strict,
            parallel_tool_calls=parallel_tool_calls,
            **kwargs,
        )

    def with_structured_output(
        self,
        schema: _DictOrPydanticClass | None = None,
        *,
        method: Literal[
            "function_calling",
            "json_mode",
            "json_schema",
        ] = "function_calling",
        include_raw: bool = False,
        strict: bool | None = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, _DictOrPydantic]:
        """包装模型，返回格式化为匹配给定 schema 的输出。

        Args:
            schema: 输出 schema。可作为以下形式传入：

                - OpenAI 函数/工具 schema，
                - JSON Schema，
                - `TypedDict` 类，
                - 或 Pydantic 类。

                If `schema` is a Pydantic class then the model output will be a
                Pydantic instance of that class, and the model-generated fields will be
                validated by the Pydantic class. Otherwise the model output will be a
                dict and will not be validated.

                有关如何在指定 Pydantic 或 `TypedDict` 类时正确指定 schema
                字段的类型和描述，请参阅
                `langchain_core.utils.function_calling.convert_to_openai_tool`。

            method: 控制模型生成的方法，可选之一为：

                - `'function_calling'`:
                    使用 DeepSeek 的 [工具调用功能](https://api-docs.deepseek.com/guides/function_calling)。
                - `'json_mode'`:
                    使用 DeepSeek 的 [JSON 模式功能](https://api-docs.deepseek.com/guides/json_mode)。

            include_raw:
                如果为 `False`，则仅返回解析后的结构化输出。

                如果在模型输出解析过程中发生错误，将抛出异常。

                如果为 `True`，则同时返回原始模型响应（一个
                `BaseMessage`）和解析后的模型响应。

                如果在输出解析过程中发生错误，该错误将被捕获并一同返回。

                The final output is always a `dict` with keys `'raw'`, `'parsed'`, and
                `'parsing_error'`.

            strict:
                Whether to enable strict schema adherence when generating the function
                call. When set to `True`, DeepSeek will use the beta API endpoint
                (`https://api.deepseek.com/beta`) for strict schema validation.
                This ensures model outputs exactly match the defined schema.

                !!! note

                    DeepSeek 的严格模式要求 schema 中所有对象属性均标记为必需。

            kwargs: 不支持额外的关键字参数。

        Returns:
            A `Runnable` that takes same inputs as a
                `langchain_core.language_models.chat.BaseChatModel`. If `include_raw` is
                `False` and `schema` is a Pydantic class, `Runnable` outputs an instance
                of `schema` (i.e., a Pydantic object). Otherwise, if `include_raw` is
                `False` then `Runnable` outputs a `dict`.

                如果 `include_raw` 为 `True`，则 `Runnable` 输出包含以下键的 `dict`：

                - `'raw'`: `BaseMessage`
                - `'parsed'`: `None` if there was a parsing error, otherwise the type
                    depends on the `schema` as described above.
                - `'parsing_error'`: `BaseException | None`
        """
                # Some applications require that incompatible parameters (e.g., unsupported
                # methods) be handled.
        if method == "json_schema":
            method = "function_calling"

                # If strict mode is enabled and using default API base, switch to beta endpoint
        if strict is True and self.api_base == DEFAULT_API_BASE:
                        # Create a new instance with beta endpoint
            beta_model = self._with_beta_api_base()
            return beta_model.with_structured_output(
                schema,
                method=method,
                include_raw=include_raw,
                strict=strict,
                **kwargs,
            )

        return super().with_structured_output(
            schema,
            method=method,
            include_raw=include_raw,
            strict=strict,
            **kwargs,
        )

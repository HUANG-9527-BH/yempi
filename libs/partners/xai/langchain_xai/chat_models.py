"""xAI 的聊天补全 API 包装器。"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any, Literal, TypeAlias, cast

import openai
from langchain_core.messages import AIMessageChunk
from langchain_core.utils import from_env, secret_from_env
from langchain_openai.chat_models.base import BaseChatOpenAI
from pydantic import BaseModel, ConfigDict, Field, SecretStr, model_validator
from typing_extensions import Self

from langchain_xai._version import __version__
from langchain_xai.data._profiles import _PROFILES

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Iterator

    from langchain_core.language_models import (
        ModelProfile,
        ModelProfileRegistry,
    )
    from langchain_core.language_models.chat_models import (
        LangSmithParams,
        LanguageModelInput,
    )
    from langchain_core.outputs import ChatGenerationChunk, ChatResult
    from langchain_core.runnables import Runnable

_DictOrPydanticClass: TypeAlias = dict[str, Any] | type[BaseModel] | type
_DictOrPydantic: TypeAlias = dict | BaseModel


_MODEL_PROFILES = cast("ModelProfileRegistry", _PROFILES)


def _get_default_model_profile(model_name: str) -> ModelProfile:
    default = _MODEL_PROFILES.get(model_name) or {}
    return default.copy()


def _model_rejects_stop(model_name: str) -> bool:
    """未经过配置的 xAI 模型是否拒绝 `stop` 参数。

    Used only as a fallback when no model profile is available; profiled models
    defer to their `reasoning_output` flag, which is authoritative. The flag and
    the name cannot be reconciled by string matching alone: the API accepts
    `stop` for `grok-4.20-0309-non-reasoning` but rejects it for
    `grok-4-fast-non-reasoning`, despite both containing `non-reasoning`.

    Every current `grok-3`, `grok-4`, and `grok-code-fast` model that is not in
    the generated profiles rejects `stop`, so the fallback drops it for those
    families. Dropping a `stop` the API would reject degrades more gracefully
    than letting the request fail outright.
    """
    return model_name.startswith(("grok-3", "grok-4")) or "grok-code-fast" in model_name


class ChatXAI(BaseChatOpenAI):  # type: ignore[override]
    r"""ChatXAI 聊天模型。

    Refer to [xAI's documentation](https://docs.x.ai/docs/api-reference#chat-completions)
    for more nuanced details on the API's behavior and supported parameters.

    设置：
        安装 `langchain-xai` 并设置环境变量 `XAI_API_KEY`。

        ```bash
        pip install -U langchain-xai
        export XAI_API_KEY="your-api-key"
        ```

    Key init args — completion params:
        model:
            Name of model to use.
        temperature:
            Sampling temperature between `0` and `2`. Higher values mean more random completions,
            while lower values (like `0.2`) mean more focused and deterministic completions.
            (Default: `1`.)
        max_tokens:
            Max number of tokens to generate. Refer to your [model's documentation](https://docs.x.ai/docs/models#model-pricing)
            for the maximum number of tokens it can generate.
        logprobs:
            Whether to return logprobs.

    关键初始化参数 —— 客户端参数：
        timeout:
            请求超时时间。
        max_retries:
            最大重试次数。
        api_key:
            xAI API 密钥。若未传入，则从环境变量 `XAI_API_KEY` 中读取。

    实例化：
        ```python
        from langchain_xai import ChatXAI

        model = ChatXAI(
            model="grok-4",
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
            (
                "system",
                "You are a helpful translator. Translate the user sentence to French.",
            ),
            ("human", "I love programming."),
        ]
        model.invoke(messages)
        ```

        ```python
        AIMessage(
            content="J'adore la programmation.",
            response_metadata={
                "token_usage": {
                    "completion_tokens": 9,
                    "prompt_tokens": 32,
                    "total_tokens": 41,
                },
                "model_name": "grok-4",
                "system_fingerprint": None,
                "finish_reason": "stop",
                "logprobs": None,
            },
            id="run-168dceca-3b8b-4283-94e3-4c739dbc1525-0",
            usage_metadata={
                "input_tokens": 32,
                "output_tokens": 9,
                "total_tokens": 41,
            },
        )
        ```

    流式：
        ```python
        for chunk in model.stream(messages):
            print(chunk.text, end="")
        ```

        ```python
        content='J' id='run-1bc996b5-293f-4114-96a1-e0f755c05eb9'
        content="'" id='run-1bc996b5-293f-4114-96a1-e0f755c05eb9'
        content='ad' id='run-1bc996b5-293f-4114-96a1-e0f755c05eb9'
        content='ore' id='run-1bc996b5-293f-4114-96a1-e0f755c05eb9'
        content=' la' id='run-1bc996b5-293f-4114-96a1-e0f755c05eb9'
        content=' programm' id='run-1bc996b5-293f-4114-96a1-e0f755c05eb9'
        content='ation' id='run-1bc996b5-293f-4114-96a1-e0f755c05eb9'
        content='.' id='run-1bc996b5-293f-4114-96a1-e0f755c05eb9'
        content='' response_metadata={'finish_reason': 'stop', 'model_name': 'grok-4'} id='run-1bc996b5-293f-4114-96a1-e0f755c05eb9'

        ```

    异步：
        ```python
        await model.ainvoke(messages)

        # stream:
        # async for chunk in (await model.astream(messages))

        # batch:
        # await model.abatch([messages])
        ```

        ```python
        AIMessage(
            content="J'adore la programmation.",
            response_metadata={
                "token_usage": {
                    "completion_tokens": 9,
                    "prompt_tokens": 32,
                    "total_tokens": 41,
                },
                "model_name": "grok-4",
                "system_fingerprint": None,
                "finish_reason": "stop",
                "logprobs": None,
            },
            id="run-09371a11-7f72-4c53-8e7c-9de5c238b34c-0",
            usage_metadata={
                "input_tokens": 32,
                "output_tokens": 9,
                "total_tokens": 41,
            },
        )
        ```

    Reasoning:
        [Certain xAI models](https://docs.x.ai/docs/models#model-pricing) support reasoning,
        which allows the model to provide reasoning content along with the response.

        若提供，推理内容将返回在 `AIMessage` 或 `AIMessageChunk` 的 `additional_kwargs`
        字段中。

        若支持，推理努力程度可通过模型构造函数的 `extra_body`
        参数指定，这将控制模型进行推理的强度。该值可以是 `'low'` 或 `'high'` 之一。

        ```python
        model = ChatXAI(
            model="grok-3-mini",
            extra_body={"reasoning_effort": "high"},
        )
        ```

        !!! note 截至 2025-07-10，`reasoning_content` 仅在 Grok 3 模型中返回，例如
        [Grok 3 Mini](https://docs.x.ai/docs/models/grok-3-mini)。

        !!! note 在 [Grok 4](https://docs.x.ai/docs/models/grok-4-0709) 中，截至
        2025-07-10，推理内容未在 `reasoning_content` 中暴露（除初始的 `'Thinking...'`
        文本外），推理无法禁用，且 `reasoning_effort` 无法指定。

    工具调用 / 函数调用：

    ```python
    from pydantic import BaseModel, Field

    model = ChatXAI(model="grok-4")


    class GetWeather(BaseModel):
        '''Get the current weather in a given location'''

        location: str = Field(..., description="The city and state, e.g. San Francisco, CA")


    class GetPopulation(BaseModel):
        '''Get the current population in a given location'''

        location: str = Field(..., description="The city and state, e.g. San Francisco, CA")


    model_with_tools = model.bind_tools([GetWeather, GetPopulation])
    ai_msg = model_with_tools.invoke("Which city is bigger: LA or NY?")
    ai_msg.tool_calls
    ```

    ```python
    [
        {
            "name": "GetPopulation",
            "args": {"location": "NY"},
            "id": "call_m5tstyn2004pre9bfuxvom8x",
            "type": "tool_call",
        },
        {
            "name": "GetPopulation",
            "args": {"location": "LA"},
            "id": "call_0vjgq455gq1av5sp9eb1pw6a",
            "type": "tool_call",
        },
    ]
    ```

    !!! note
    使用流式响应时，工具/函数调用将作为一个整体在单个数据块中返回，而不是跨多个数据块流式传输。

    工具选择可通过在模型构造函数的 `extra_body` 参数中设置 `tool_choice`
    参数来控制。例如，禁用工具/函数调用：

    ```python
    model = ChatXAI(model="grok-4", extra_body={"tool_choice": "none"})
    ```
    若要强制模型始终调用工具/函数，请将 `tool_choice` 设置为 `'required'`：

    ```python
    model = ChatXAI(model="grok-4", extra_body={"tool_choice": "required"})
    ```

    若要指定要调用的工具/函数，请将 `tool_choice` 设置为工具/函数的名称：

    ```python
    from pydantic import BaseModel, Field

    model = ChatXAI(
        model="grok-4",
        extra_body={
            "tool_choice": {"type": "function", "function": {"name": "GetWeather"}}
        },
    )

    class GetWeather(BaseModel):
        \"\"\"Get the current weather in a given location\"\"\"

        location: str = Field(..., description='The city and state, e.g. San Francisco, CA')


    class GetPopulation(BaseModel):
        \"\"\"Get the current population in a given location\"\"\"

        location: str = Field(..., description='The city and state, e.g. San Francisco, CA')


    model_with_tools = model.bind_tools([GetWeather, GetPopulation])
    ai_msg = model_with_tools.invoke(
        "Which city is bigger: LA or NY?",
    )
    ai_msg.tool_calls
    ```

    生成的工具调用结果为：

    ```python
    [
        {
            "name": "GetWeather",
            "args": {"location": "Los Angeles, CA"},
            "id": "call_81668711",
            "type": "tool_call",
        }
    ]
    ```

    并行工具调用/并行函数调用：
    默认情况下，并行工具/函数调用已启用，因此您可以在一次请求/响应周期内处理多个函数调用。当需要两个或更多工具调用时，所有工具调用请求都将包含在响应体中。

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

        ```python
        Joke(
            setup="Why was the cat sitting on the computer?",
            punchline="To keep an eye on the mouse!",
            rating=7,
        )
        ```

    Web search:
        **Live Search** (the legacy `search_parameters` option) has been deprecated by xAI.
        Use `bind_tools` with compatible tool definitions when using the OpenAI-compatible
        Responses API instead. If you pass `search_parameters` to `ChatXAI`, a
        `DeprecationWarning` is emitted and the parameter is ignored; requests otherwise
        succeed without search.

    token 使用量：
        ```python
        ai_msg = model.invoke(messages)
        ai_msg.usage_metadata
        ```

        ```python
        {"input_tokens": 37, "output_tokens": 6, "total_tokens": 43}
        ```

    日志概率：
        ```python
        logprobs_model = model.bind(logprobs=True)
        messages = [("human", "Say Hello World! Do not return anything else.")]
        ai_msg = logprobs_model.invoke(messages)
        ai_msg.response_metadata["logprobs"]
        ```

        ```python
        {
            "content": None,
            "token_ids": [22557, 3304, 28808, 2],
            "tokens": [" Hello", " World", "!", "</s>"],
            "token_logprobs": [-4.7683716e-06, -5.9604645e-07, 0, -0.057373047],
        }
        ```

    响应元数据：

    ```python
    ai_msg = model.invoke(messages)
    ai_msg.response_metadata
    ```

    ```python
    {
        "token_usage": {
            "completion_tokens": 4,
            "prompt_tokens": 19,
            "total_tokens": 23,
        },
        "model_name": "grok-4",
        "system_fingerprint": None,
        "finish_reason": "stop",
        "logprobs": None,
    }
    ```
    """  # noqa: E501

    model_name: str = Field(default="grok-4", alias="model")
    """模型名称。"""

    xai_api_key: SecretStr | None = Field(
        alias="api_key",
        default_factory=secret_from_env("XAI_API_KEY", default=None),
    )
    """xAI API 密钥。

    若未提供，则自动从环境变量 `XAI_API_KEY` 读取。
    """

    xai_api_base: str = Field(
        alias="base_url",
        default_factory=from_env("XAI_API_BASE", default="https://api.x.ai/v1/"),
    )
    """API 请求的基础 URL 路径。

    若未提供，则自动从环境变量 `XAI_API_BASE` 读取。
    """

    search_parameters: dict[str, Any] | None = None
    """**已弃用。** 请改用网络搜索工具：

    ```python
    ChatXAI(model="...").bind_tools([{"type": "web_search"}])
    ```
    """

    openai_api_key: SecretStr | None = None

    openai_api_base: str | None = None

    model_config = ConfigDict(
        populate_by_name=True,
    )

    @property
    def lc_secrets(self) -> dict[str, str]:
        """构造函数参数名到密钥 ID 的映射。

        例如，`{"xai_api_key": "XAI_API_KEY"}`
        """
        return {"xai_api_key": "XAI_API_KEY"}

    @classmethod
    def get_lc_namespace(cls) -> list[str]:
        """获取 LangChain 对象的命名空间。

        Returns:
            `["langchain_xai", "chat_models"]`
        """
        return ["langchain_xai", "chat_models"]

    @classmethod
    def is_lc_serializable(cls) -> bool:
        """返回此模型是否可由 LangChain 序列化。"""
        return True

    @property
    def _llm_type(self) -> str:
        """聊天模型的返回类型。"""
        return "xai-chat"

    def _get_ls_params(
        self,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> LangSmithParams:
        """获取调用模型所使用的参数。"""
        params = super()._get_ls_params(stop=stop, **kwargs)
        params["ls_provider"] = "xai"
        return params

    @model_validator(mode="after")
    def _set_xai_version(self) -> Self:
        """在元数据中设置包版本。

        命名唯一以避免与 `BaseChatOpenAI._set_openai_chat_version` 冲突；
        Pydantic 会替换同名验证器而非链式调用。
        """
        self._add_version("langchain-xai", __version__)
        return self

    @model_validator(mode="after")
    def _warn_search_parameters_deprecated(self) -> Self:
        """若使用 search_parameters（实时搜索），则发出弃用警告。"""
        if self.search_parameters:
            warnings.warn(
                "search_parameters (Live Search) is deprecated by xAI and is ignored. "
                'Use `ChatXAI(model="...").bind_tools([{"type": "web_search"}])` '
                "instead.",
                DeprecationWarning,
                stacklevel=2,
            )
        return self

    @model_validator(mode="after")
    def validate_environment(self) -> Self:
        """验证 API 密钥和 Python 包是否存在于环境中。"""
        if self.n is not None and self.n < 1:
            msg = "n must be at least 1."
            raise ValueError(msg)
        if self.n is not None and self.n > 1 and self.streaming:
            msg = "n must be 1 when streaming."
            raise ValueError(msg)

        client_params: dict = {
            "api_key": (
                self.xai_api_key.get_secret_value() if self.xai_api_key else None
            ),
            "base_url": self.xai_api_base,
            "timeout": self.request_timeout,
            "default_headers": self.default_headers,
            "default_query": self.default_query,
        }
        if self.max_retries is not None:
            client_params["max_retries"] = self.max_retries

        if client_params["api_key"] is None:
            msg = (
                "xAI API key is not set. Please set it in the `xai_api_key` field or "
                "in the `XAI_API_KEY` environment variable."
            )
            raise ValueError(msg)

        if not (self.client or None):
            sync_specific: dict = {"http_client": self.http_client}
            self.client = openai.OpenAI(
                **client_params, **sync_specific
            ).chat.completions
            self.root_client = openai.OpenAI(**client_params, **sync_specific)
        if not (self.async_client or None):
            async_specific: dict = {"http_client": self.http_async_client}
            self.async_client = openai.AsyncOpenAI(
                **client_params, **async_specific
            ).chat.completions
            self.root_async_client = openai.AsyncOpenAI(
                **client_params,
                **async_specific,
            )

                # Enable streaming usage metadata by default
        if self.stream_usage is not False:
            self.stream_usage = True

        return self

    def _resolve_model_profile(self) -> ModelProfile | None:
        return _get_default_model_profile(self.model_name) or None

    def _get_request_payload(
        self,
        input_: LanguageModelInput,
        *,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> dict:
        payload = super()._get_request_payload(input_, stop=stop, **kwargs)
        if payload.get("stop") is not None:
                        # xAI 会拒绝推理模型的 `stop`
            # 参数。模型配置文件是权威的（当可用时，能准确区分仅通过配置文件不同而不同的模型，而非名称差异）；基于名称的检查是针对未出现在生成配置文件中的别名的后备方案。
                        # authoritative when available (it correctly distinguishes models
                        # that only differ from each other by profile, not by name); the
                        # name-based check is a fallback for aliases absent from the
                        # generated profiles.
            model_profile = self._resolve_model_profile()
            rejects_stop = (
                bool(model_profile.get("reasoning_output"))
                if model_profile is not None
                else _model_rejects_stop(self.model_name)
            )
            if rejects_stop:
                payload.pop("stop", None)

                # xAI 要求 `reasoning_effort` 出现在 `extra_body` 中，而非顶层字段。
                # 将其移至 `extra_body` 中，以确保其能到达 API。
        reasoning_effort = payload.pop("reasoning_effort", None)
        if reasoning_effort is not None:
            extra_body = payload.get("extra_body")
            if not isinstance(extra_body, dict):
                extra_body = {}
            payload["extra_body"] = {**extra_body, "reasoning_effort": reasoning_effort}

        return payload

    def _stream(self, *args: Any, **kwargs: Any) -> Iterator[ChatGenerationChunk]:
        """路由至聊天补全或 Responses API。"""
        if self._use_responses_api({**kwargs, **self.model_kwargs}):
            return super()._stream_responses(*args, **kwargs)
        return super()._stream(*args, **kwargs)

    async def _astream(
        self, *args: Any, **kwargs: Any
    ) -> AsyncIterator[ChatGenerationChunk]:
        """路由至聊天补全或 Responses API。"""
        if self._use_responses_api({**kwargs, **self.model_kwargs}):
            async for chunk in super()._astream_responses(*args, **kwargs):
                yield chunk
        else:
            async for chunk in super()._astream(*args, **kwargs):
                yield chunk

    def _create_chat_result(
        self,
        response: dict | openai.BaseModel,
        generation_info: dict | None = None,
    ) -> ChatResult:
        rtn = super()._create_chat_result(response, generation_info)

        for generation in rtn.generations:
            generation.message.response_metadata["model_provider"] = "xai"

        if not isinstance(response, openai.BaseModel):
            return rtn

        if hasattr(response.choices[0].message, "reasoning_content"):  # type: ignore[attr-defined]
            rtn.generations[0].message.additional_kwargs["reasoning_content"] = (
                response.choices[0].message.reasoning_content  # type: ignore[attr-defined]
            )

        if hasattr(response, "citations"):
            rtn.generations[0].message.additional_kwargs["citations"] = (
                response.citations
            )

                # 与 OpenAI 不同，xAI 报告的推理 token 数量 < 完成 token 数量。因此我们假设
                # 它们未被计入输出 token 中，此处进行补充添加。
        if (
            (not self._use_responses_api({}))
            and (usage_metadata := rtn.generations[0].message.usage_metadata)  # type: ignore[attr-defined]
            and (
                reasoning_tokens := usage_metadata.get("output_token_details", {}).get(
                    "reasoning"
                )
            )
        ):
            rtn.generations[0].message.usage_metadata["output_tokens"] += (  # type: ignore[attr-defined]
                reasoning_tokens
            )
                        # 保持已记录的 UsageMetadata 不变性
                        # total_tokens == input_tokens + output_tokens (gh #39634)。
            usage_metadata["total_tokens"] = (
                usage_metadata["input_tokens"] + usage_metadata["output_tokens"]
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

        if generation_chunk:
            generation_chunk.message.response_metadata["model_provider"] = "xai"

        if (choices := chunk.get("choices")) and generation_chunk:
            top = choices[0]
            if isinstance(generation_chunk.message, AIMessageChunk) and (
                reasoning_content := top.get("delta", {}).get("reasoning_content")
            ):
                generation_chunk.message.additional_kwargs["reasoning_content"] = (
                    reasoning_content
                )

        if (
            (citations := chunk.get("citations"))
            and generation_chunk
            and isinstance(generation_chunk.message, AIMessageChunk)
            and not chunk.get("usage")  # 引用内容在最终使用统计块中重复出现
        ):
            generation_chunk.message.additional_kwargs["citations"] = citations

                # 与 OpenAI 不同，xAI 报告的推理 token 数量 < 完成 token 数量。因此我们假设
                # 它们未被计入输出 token 中，此处进行补充添加。
        if (
            generation_chunk
            and (not self._use_responses_api({}))
            and (usage_metadata := generation_chunk.message.usage_metadata)  # type: ignore[attr-defined]
            and (
                reasoning_tokens := usage_metadata.get("output_token_details", {}).get(
                    "reasoning"
                )
            )
        ):
            generation_chunk.message.usage_metadata["output_tokens"] += reasoning_tokens  # type: ignore[attr-defined]
                        # 保持已记录的 UsageMetadata 不变性
                        # total_tokens == input_tokens + output_tokens (gh #39634)。
            usage_metadata["total_tokens"] = (
                usage_metadata["input_tokens"] + usage_metadata["output_tokens"]
            )
        return generation_chunk

    def with_structured_output(
        self,
        schema: _DictOrPydanticClass | None = None,
        *,
        method: Literal[
            "function_calling", "json_mode", "json_schema"
        ] = "function_calling",
        include_raw: bool = False,
        strict: bool | None = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, _DictOrPydantic]:
        """返回格式匹配给定模式的输出的模型包装器。

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

                参见 `langchain_core.utils.function_calling.convert_to_openai_tool`
                以了解如何在使用 Pydantic 或 `TypedDict`
                类时正确指定模式字段的类型和描述。

            method: 控制模型生成的方法，可选值为：

                - `'function_calling'`:
                    使用 xAI 的 [工具调用功能](https://docs.x.ai/docs/guides/function-calling)。
                - `'json_schema'`:
                    使用 xAI 的 [结构化输出功能](https://docs.x.ai/docs/guides/structured-outputs)。
                - `'json_mode'`:
                    使用 xAI 的 JSON 模式功能。

            include_raw:
                若为 `False`，则仅返回解析后的结构化输出。

                若模型输出解析过程中发生错误，则会抛出异常。

                若为 `True`，则同时返回原始模型响应（一个
                `BaseMessage`）和解析后的模型响应。

                若解析过程中发生错误，该错误将被捕获并一同返回。

                The final output is always a `dict` with keys `'raw'`, `'parsed'`, and
                `'parsing_error'`.
            strict:
                - `True`:
                    Model output is guaranteed to exactly match the schema.
                    The input schema will also be validated according to the [supported schemas](https://docs.x.ai/docs/guides/structured-outputs).
                - `False`:
                    Input schema will not be validated and model output will not be
                    validated.
                - `None`:
                    `strict` argument will not be passed to the model.

            kwargs: 不支持额外的关键字参数。

        Returns:
            A `Runnable` that takes same inputs as a
                `langchain_core.language_models.chat.BaseChatModel`. If `include_raw` is
                `False` and `schema` is a Pydantic class, `Runnable` outputs an instance
                of `schema` (i.e., a Pydantic object). Otherwise, if `include_raw` is
                `False` then `Runnable` outputs a `dict`.

                若 `include_raw` 为 `True`，则 `Runnable` 输出一个包含以下键的 `dict`：

                - `'raw'`: `BaseMessage`
                - `'parsed'`: `None` if there was a parsing error, otherwise the type
                    depends on the `schema` as described above.
                - `'parsing_error'`: `BaseException | None`
        """
                # 某些应用要求对不兼容参数（例如不支持的
                # 方法）进行处理。
        if method == "function_calling" and strict:
            strict = None
        return super().with_structured_output(
            schema, method=method, include_raw=include_raw, strict=strict, **kwargs
        )

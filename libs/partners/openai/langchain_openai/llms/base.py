"""OpenAI 大语言模型的基础类。聊天模型位于 `chat_models/` 中。"""

from __future__ import annotations

import logging
import sys
from collections.abc import AsyncIterator, Callable, Collection, Iterator, Mapping
from typing import Any, Literal

import openai
import tiktoken
from langchain_core._api.deprecation import deprecated
from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models.llms import BaseLLM
from langchain_core.outputs import Generation, GenerationChunk, LLMResult
from langchain_core.utils import get_pydantic_field_names
from langchain_core.utils.utils import _build_model_kwargs, from_env, secret_from_env
from pydantic import ConfigDict, Field, SecretStr, model_validator
from typing_extensions import Self

from langchain_openai._version import __version__
from langchain_openai.data._profiles import _PROFILES

logger = logging.getLogger(__name__)


def _update_token_usage(
    keys: set[str], response: dict[str, Any], token_usage: dict[str, Any]
) -> None:
    """更新 token 使用量。"""
    _keys_to_use = keys.intersection(response["usage"])
    for _key in _keys_to_use:
        if _key not in token_usage:
            token_usage[_key] = response["usage"][_key]
        else:
            token_usage[_key] += response["usage"][_key]


def _stream_response_to_generation_chunk(
    stream_response: dict[str, Any],
) -> GenerationChunk:
    """将流式响应转换为生成块。"""
    if not stream_response["choices"]:
        return GenerationChunk(text="")
    return GenerationChunk(
        text=stream_response["choices"][0]["text"] or "",
        generation_info={
            "finish_reason": stream_response["choices"][0].get("finish_reason", None),
            "logprobs": stream_response["choices"][0].get("logprobs", None),
        },
    )


class BaseOpenAI(BaseLLM):
    """基础 OpenAI 大语言模型类。

    设置：
        安装 `langchain-openai` 并设置环境变量 `OPENAI_API_KEY`。

        ```bash
        pip install -U langchain-openai
        export OPENAI_API_KEY="your-api-key"
        ```

    关键初始化参数 —— 完成参数：
        model_name:
            要使用的 OpenAI 模型名称。
        temperature:
            采样温度。
        max_tokens:
            生成的最大 token 数量。
        top_p:
            每一步考虑的 token 总概率质量。
        frequency_penalty:
            根据频率惩罚重复 token。
        presence_penalty:
            惩罚重复 token。
        n:
            为每个提示生成多少个完成结果。
        best_of:
            在服务器端生成 best_of 个完成结果，并返回 "最佳" 的一个。
        logit_bias:
            调整特定 token 被生成的概率。
        seed:
            生成的随机种子。
        logprobs:
            包含 logprobs 最可能输出 token 的对数概率。
        streaming:
            是否流式返回结果。

    Key init args — client params:
        openai_api_key:
            OpenAI API key. If not passed in will be read from env var
            `OPENAI_API_KEY`.
        openai_api_base:
            Base URL path for API requests, leave blank if not using a proxy or
            service emulator. Falls back to env var `OPENAI_API_BASE`, then to
            `OPENAI_BASE_URL` (read by the underlying SDK client).
        openai_organization:
            OpenAI organization ID. If not passed in will be read from env
            var `OPENAI_ORG_ID`.
        request_timeout:
            Timeout for requests to OpenAI completion API.
        max_retries:
            Maximum number of retries to make when generating.
        batch_size:
            Batch size to use when passing multiple documents to generate.

    有关支持的初始化参数及其描述的完整列表，请参见参数部分。

    实例化：
        ```python
        from langchain_openai.llms.base import BaseOpenAI

        model = BaseOpenAI(
            model_name="gpt-3.5-turbo-instruct",
            temperature=0.7,
            max_tokens=256,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            # openai_api_key="...",
            # openai_api_base="...",
            # openai_organization="...",
            # other params...
        )
        ```

    调用：
        ```python
        input_text = "The meaning of life is "
        response = model.invoke(input_text)
        print(response)
        ```

        ```txt
        "a philosophical question that has been debated by thinkers and
        scholars for centuries."
        ```

    流式：
        ```python
        for chunk in model.stream(input_text):
            print(chunk, end="")
        ```
        ```txt
        a philosophical question that has been debated by thinkers and
        scholars for centuries.
        ```

    异步：
        ```python
        response = await model.ainvoke(input_text)

        # stream:
        # async for chunk in model.astream(input_text):
        #     print(chunk, end="")

        # batch:
        # await model.abatch([input_text])
        ```
        ```
        "a philosophical question that has been debated by thinkers and
        scholars for centuries."
        ```

    """

    client: Any = Field(default=None, exclude=True)

    async_client: Any = Field(default=None, exclude=True)

    model_name: str = Field(default="gpt-3.5-turbo-instruct", alias="model")
    """要使用的模型名称。"""

    temperature: float = 0.7
    """要使用的采样温度。"""

    max_tokens: int = 256
    """生成完成结果时允许的最大 token 数量。 -1
表示在提示和模型最大上下文大小限制下尽可能多地返回 token。"""

    top_p: float = 1
    """每一步考虑的 token 总概率质量。"""

    frequency_penalty: float = 0
    """根据频率惩罚重复 token。"""

    presence_penalty: float = 0
    """惩罚重复 token。"""

    n: int = 1
    """为每个提示生成的完成数量。"""

    best_of: int = 1
    """在服务器端生成 best_of 个完成结果，并返回 "最佳" 的一个。"""

    model_kwargs: dict[str, Any] = Field(default_factory=dict)
    """包含 `create` 调用中有效的任何模型参数，但未显式指定的参数。"""

    openai_api_key: SecretStr | None | Callable[[], str] = Field(
        alias="api_key", default_factory=secret_from_env("OPENAI_API_KEY", default=None)
    )
    """若未提供，则自动从环境变量 `OPENAI_API_KEY` 推断。"""

    openai_api_base: str | None = Field(
        alias="base_url", default_factory=from_env("OPENAI_API_BASE", default=None)
    )
    """Base URL path for API requests, leave blank if not using a proxy or service
    emulator.

    解析顺序（首个匹配项生效）：

    1. 显式指定 `base_url`（或 `openai_api_base`）关键字参数。
    2. 环境变量 `OPENAI_API_BASE`（LangChain 在初始化时读取）。
    3. 环境变量 `OPENAI_BASE_URL`（由底层的 `openai` SDK 客户端读取）。
    """

    openai_organization: str | None = Field(
        alias="organization",
        default_factory=from_env(
            ["OPENAI_ORG_ID", "OPENAI_ORGANIZATION"], default=None
        ),
    )
    """如果未提供，将自动从环境变量 `OPENAI_ORG_ID` 推断。"""

        # 支持为 OpenAI 显式配置代理
    openai_proxy: str | None = Field(
        default_factory=from_env("OPENAI_PROXY", default=None)
    )

    batch_size: int = 20
    """在传递多个文档生成时使用的批处理大小。"""

    request_timeout: float | tuple[float, float] | Any | None = Field(
        default=None, alias="timeout"
    )
    """Timeout for requests to OpenAI completion API. Can be float, `httpx.Timeout` or
    None."""

    logit_bias: dict[str, float] | None = None
    """调整特定 token 被生成的概率。"""

    max_retries: int = 2
    """生成时最大重试次数。"""

    seed: int | None = None
    """生成的随机种子。"""

    logprobs: int | None = None
    """包含最可能输出 token 的 logprobs，以及所选 token 的 logprobs。"""

    streaming: bool = False
    """是否启用流式输出。"""

    allowed_special: Literal["all"] | set[str] = set()
    """允许的特殊 token 集合。"""

    disallowed_special: Literal["all"] | Collection[str] = "all"
    """不允许的特殊 token 集合。"""

    tiktoken_model_name: str | None = None
    """使用此类时传递给 tiktoken 的模型名称。

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

        # 配置自定义 httpx 客户端。详见
        # [httpx 文档](https://www.python-httpx.org/api/#client) 获取更多详情。
    http_client: Any | None = None
    """可选的 `httpx.Client`。

    Only used for sync invocations. Must specify `http_async_client` as well if you'd
    like a custom client for async invocations.
    """

    http_async_client: Any | None = None
    """可选的 `httpx.AsyncClient`。

    Only used for async invocations. Must specify `http_client` as well if you'd like a
    custom client for sync invocations.
    """

    extra_body: Mapping[str, Any] | None = None
    """可选的额外 JSON 属性，在向兼容 OpenAI 的 API（如 vLLM）发送请求时包含在请求参数中。"""

    model_config = ConfigDict(populate_by_name=True)

    @model_validator(mode="before")
    @classmethod
    def build_extra(cls, values: dict[str, Any]) -> Any:
        """从传入的额外参数中构建额外的关键字参数。"""
        all_required_field_names = get_pydantic_field_names(cls)
        return _build_model_kwargs(values, all_required_field_names)

    @model_validator(mode="after")
    def _set_openai_version(self) -> Self:
        """在元数据中设置包版本。"""
        self._add_version("langchain-openai", __version__)
        return self

    @model_validator(mode="after")
    def validate_environment(self) -> Self:
        """验证 API 密钥和 Python 包是否存在于环境中。"""
        if self.n < 1:
            msg = "n must be at least 1."
            raise ValueError(msg)
        if self.streaming and self.n > 1:
            msg = "Cannot stream results when n > 1."
            raise ValueError(msg)
        if self.streaming and self.best_of > 1:
            msg = "Cannot stream results when best_of > 1."
            raise ValueError(msg)

                # 从 SecretStr 或 Callable 中解析 API 密钥
        api_key_value: str | Callable[[], str] | None = None
        if self.openai_api_key is not None:
            if isinstance(self.openai_api_key, SecretStr):
                api_key_value = self.openai_api_key.get_secret_value()
            elif callable(self.openai_api_key):
                api_key_value = self.openai_api_key

        client_params: dict = {
            "api_key": api_key_value,
            "organization": self.openai_organization,
            "base_url": self.openai_api_base,
            "timeout": self.request_timeout,
            "max_retries": self.max_retries,
            "default_headers": self.default_headers,
            "default_query": self.default_query,
        }
        if not self.client:
            sync_specific = {"http_client": self.http_client}
            self.client = openai.OpenAI(**client_params, **sync_specific).completions  # type: ignore[arg-type]
        if not self.async_client:
            async_specific = {"http_client": self.http_async_client}
            self.async_client = openai.AsyncOpenAI(
                **client_params,
                **async_specific,  # type: ignore[arg-type]
            ).completions

        return self

    @property
    def _default_params(self) -> dict[str, Any]:
        """获取调用 OpenAI API 的默认参数。"""
        normal_params: dict[str, Any] = {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "frequency_penalty": self.frequency_penalty,
            "presence_penalty": self.presence_penalty,
            "n": self.n,
            "seed": self.seed,
            "logprobs": self.logprobs,
        }

        if self.logit_bias is not None:
            normal_params["logit_bias"] = self.logit_bias

        if self.max_tokens is not None:
            normal_params["max_tokens"] = self.max_tokens

        if self.extra_body is not None:
            normal_params["extra_body"] = self.extra_body

                # Azure gpt-35-turbo 不支持 best_of
                # 如果 best_of 为 1，则不要指定
        if self.best_of > 1:
            normal_params["best_of"] = self.best_of

        return {**normal_params, **self.model_kwargs}

    def _stream(
        self,
        prompt: str,
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> Iterator[GenerationChunk]:
        params = {**self._invocation_params, **kwargs, "stream": True}
        self.get_sub_prompts(params, [prompt], stop)  # 此操作会修改 params
        for stream_resp in self.client.create(prompt=prompt, **params):
            if not isinstance(stream_resp, dict):
                stream_resp = stream_resp.model_dump()
            chunk = _stream_response_to_generation_chunk(stream_resp)

            if run_manager:
                run_manager.on_llm_new_token(
                    chunk.text,
                    chunk=chunk,
                    verbose=self.verbose,
                    logprobs=(
                        chunk.generation_info["logprobs"]
                        if chunk.generation_info
                        else None
                    ),
                )
            yield chunk

    async def _astream(
        self,
        prompt: str,
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[GenerationChunk]:
        params = {**self._invocation_params, **kwargs, "stream": True}
        self.get_sub_prompts(params, [prompt], stop)  # 此操作会修改 params
        async for stream_resp in await self.async_client.create(
            prompt=prompt, **params
        ):
            if not isinstance(stream_resp, dict):
                stream_resp = stream_resp.model_dump()
            chunk = _stream_response_to_generation_chunk(stream_resp)

            if run_manager:
                await run_manager.on_llm_new_token(
                    chunk.text,
                    chunk=chunk,
                    verbose=self.verbose,
                    logprobs=(
                        chunk.generation_info["logprobs"]
                        if chunk.generation_info
                        else None
                    ),
                )
            yield chunk

    def _generate(
        self,
        prompts: list[str],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> LLMResult:
        """使用 k 个唯一提示调用 OpenAI 的端点。

        Args:
            prompts: 提示词：要传递给模型的提示。
            stop: 可选的停止词列表，用于生成时。
            run_manager: 可选的回调管理器，用于该调用。

        Returns:
            完整的 LLM 输出。

        Example:
            ```python
            response = openai.generate(["Tell me a joke."])
            ```
        """
                # TODO: 为此编写单元测试
        params = self._invocation_params
        params = {**params, **kwargs}
        sub_prompts = self.get_sub_prompts(params, prompts, stop)
        choices = []
        token_usage: dict[str, int] = {}
                # 从响应中获取 token 使用情况。
                # 包含提示、完成和总共使用的 token。
        _keys = {"completion_tokens", "prompt_tokens", "total_tokens"}
        system_fingerprint: str | None = None
        for _prompts in sub_prompts:
            if self.streaming:
                if len(_prompts) > 1:
                    msg = "Cannot stream results with multiple prompts."
                    raise ValueError(msg)

                generation: GenerationChunk | None = None
                for chunk in self._stream(_prompts[0], stop, run_manager, **kwargs):
                    if generation is None:
                        generation = chunk
                    else:
                        generation += chunk
                if generation is None:
                    msg = "Generation is empty after streaming."
                    raise ValueError(msg)
                choices.append(
                    {
                        "text": generation.text,
                        "finish_reason": (
                            generation.generation_info.get("finish_reason")
                            if generation.generation_info
                            else None
                        ),
                        "logprobs": (
                            generation.generation_info.get("logprobs")
                            if generation.generation_info
                            else None
                        ),
                    }
                )
            else:
                response = self.client.create(prompt=_prompts, **params)
                if not isinstance(response, dict):
                                        # V1 客户端以 PyDantic 对象形式返回响应，而非
                                        # 字典。在过渡期间，我们将其深度转换为字典。
                    response = response.model_dump()

                                # 有时 AI 模型调用会出错，我们应该抛出异常。
                                # 否则，后续代码 'choices.extend(response["choices"])'
                                # 将引发 "TypeError: 'NoneType' object is not iterable" 错误
                                # 掩盖真实错误。因为 'response["choices"]' 是 None。
                if response.get("error"):
                    raise ValueError(response.get("error"))

                choices.extend(response["choices"])
                _update_token_usage(_keys, response, token_usage)
                if not system_fingerprint:
                    system_fingerprint = response.get("system_fingerprint")
        return self.create_llm_result(
            choices, prompts, params, token_usage, system_fingerprint=system_fingerprint
        )

    async def _agenerate(
        self,
        prompts: list[str],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> LLMResult:
        """使用 k 个唯一提示异步调用 OpenAI 的端点。"""
        params = self._invocation_params
        params = {**params, **kwargs}
        sub_prompts = self.get_sub_prompts(params, prompts, stop)
        choices = []
        token_usage: dict[str, int] = {}
                # 从响应中获取 token 使用量。
                # 包含提示词、完成内容和总共使用的 token 数量。
        _keys = {"completion_tokens", "prompt_tokens", "total_tokens"}
        system_fingerprint: str | None = None
        for _prompts in sub_prompts:
            if self.streaming:
                if len(_prompts) > 1:
                    msg = "Cannot stream results with multiple prompts."
                    raise ValueError(msg)

                generation: GenerationChunk | None = None
                async for chunk in self._astream(
                    _prompts[0], stop, run_manager, **kwargs
                ):
                    if generation is None:
                        generation = chunk
                    else:
                        generation += chunk
                if generation is None:
                    msg = "Generation is empty after streaming."
                    raise ValueError(msg)
                choices.append(
                    {
                        "text": generation.text,
                        "finish_reason": (
                            generation.generation_info.get("finish_reason")
                            if generation.generation_info
                            else None
                        ),
                        "logprobs": (
                            generation.generation_info.get("logprobs")
                            if generation.generation_info
                            else None
                        ),
                    }
                )
            else:
                response = await self.async_client.create(prompt=_prompts, **params)
                if not isinstance(response, dict):
                    response = response.model_dump()
                choices.extend(response["choices"])
                _update_token_usage(_keys, response, token_usage)
        return self.create_llm_result(
            choices, prompts, params, token_usage, system_fingerprint=system_fingerprint
        )

    def get_sub_prompts(
        self,
        params: dict[str, Any],
        prompts: list[str],
        stop: list[str] | None = None,
    ) -> list[list[str]]:
        """获取 LLM 调用的子提示词。"""
        if stop is not None:
            params["stop"] = stop
        if params["max_tokens"] == -1:
            if len(prompts) != 1:
                msg = "max_tokens set to -1 not supported for multiple inputs."
                raise ValueError(msg)
            params["max_tokens"] = self.max_tokens_for_prompt(prompts[0])
        return [
            prompts[i : i + self.batch_size]
            for i in range(0, len(prompts), self.batch_size)
        ]

    def create_llm_result(
        self,
        choices: Any,
        prompts: list[str],
        params: dict[str, Any],
        token_usage: dict[str, int],
        *,
        system_fingerprint: str | None = None,
    ) -> LLMResult:
        """根据选择项和提示词创建 LLMResult。"""
        generations = []
        n = params.get("n", self.n)
        for i, _ in enumerate(prompts):
            sub_choices = choices[i * n : (i + 1) * n]
            generations.append(
                [
                    Generation(
                        text=choice["text"],
                        generation_info={
                            "finish_reason": choice.get("finish_reason"),
                            "logprobs": choice.get("logprobs"),
                        },
                    )
                    for choice in sub_choices
                ]
            )
        llm_output = {"token_usage": token_usage, "model_name": self.model_name}
        if system_fingerprint:
            llm_output["system_fingerprint"] = system_fingerprint
        return LLMResult(generations=generations, llm_output=llm_output)

    @property
    def _invocation_params(self) -> dict[str, Any]:
        """获取调用模型所使用的参数。"""
        return self._default_params

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """获取标识参数。"""
        return {"model_name": self.model_name, **self._default_params}

    @property
    def _llm_type(self) -> str:
        """返回 LLM 的返回类型。"""
        return "openai"

    def get_token_ids(self, text: str) -> list[int]:
        """使用 tiktoken 包获取 token ID。"""
        if self.custom_get_token_ids is not None:
            return self.custom_get_token_ids(text)
                # tiktoken 不支持 Python < 3.8
        if sys.version_info[1] < 8:
            return super().get_num_tokens(text)

        model_name = self.tiktoken_model_name or self.model_name
        try:
            enc = tiktoken.encoding_for_model(model_name)
        except KeyError:
            enc = tiktoken.get_encoding("cl100k_base")

        return enc.encode(
            text,
            allowed_special=self.allowed_special,
            disallowed_special=self.disallowed_special,
        )

    @staticmethod
    @deprecated(
        since="1.2",
        removal="2.0",
        alternative=(
            "the model profile's `max_input_tokens` field "
            "(e.g. `ChatOpenAI(model=...).profile['max_input_tokens']`)"
        ),
    )
    def modelname_to_contextsize(modelname: str) -> int:
        """返回模型的最大输入上下文大小。

        优先使用模型的配置文件（`max_input_tokens`），若无则回退到
        一个包含旧版模型的映射，这些模型没有配置文件。

        !!! warning "1.2 版本变更"

            Now returns `max_input_tokens` from the model profile, which is the
            input context window. Earlier releases returned a hand-maintained
            number that for some newer models (e.g. `gpt-5`) reflected the
            *total* context (input + output). Callers using the result as an
            input-token budget are unaffected; callers using it as a combined
            input+output budget should switch to the profile fields directly.

        Args:
            modelname: 我们想要了解上下文大小的模型名称。

        Returns:
            最大输入上下文大小。

        Example:
            ```python
            max_tokens = openai.modelname_to_contextsize("gpt-3.5-turbo-instruct")
            ```
        """
                # 没有模型配置文件的旧版模型。
        legacy_token_mapping = {
            "gpt-4-0314": 8192,
            "gpt-4-0613": 8192,
            "gpt-4-32k": 32768,
            "gpt-4-32k-0314": 32768,
            "gpt-4-32k-0613": 32768,
            "gpt-4o-2024-05-13": 128_000,
            "gpt-3.5-turbo-0301": 4096,
            "gpt-3.5-turbo-0613": 4096,
            "gpt-3.5-turbo-16k": 16385,
            "gpt-3.5-turbo-16k-0613": 16385,
            "gpt-3.5-turbo-instruct": 4096,
            "text-ada-001": 2049,
            "ada": 2049,
            "text-babbage-001": 2040,
            "babbage": 2049,
            "text-curie-001": 2049,
            "curie": 2049,
            "davinci": 2049,
            "text-davinci-003": 4097,
            "text-davinci-002": 4097,
            "code-davinci-002": 8001,
            "code-davinci-001": 8001,
            "code-cushman-002": 2048,
            "code-cushman-001": 2048,
        }

                # 处理微调模型
        if "ft-" in modelname:
            modelname = modelname.split(":", maxsplit=1)[0]

        profile = _PROFILES.get(modelname)
        context_size = profile.get("max_input_tokens") if profile else None
        if profile is not None and context_size is None:
            logger.warning(
                "Profile for model %s is missing `max_input_tokens`; "
                "falling back to legacy mapping.",
                modelname,
            )
        if context_size is None:
            context_size = legacy_token_mapping.get(modelname)

        if context_size is None:
            known = sorted({*_PROFILES.keys(), *legacy_token_mapping.keys()})
            msg = (
                f"Unknown model: {modelname}. Please provide a valid OpenAI model "
                "name, or read `max_input_tokens` from the model profile directly. "
                "Known models are: " + ", ".join(known)
            )
            raise ValueError(msg)

        return context_size

    @property
    def max_context_size(self) -> int:
        """获取此模型的最大上下文大小。"""
        return self.modelname_to_contextsize(self.model_name)

    def max_tokens_for_prompt(self, prompt: str) -> int:
        """计算针对提示词可能生成的最大 token 数量。

        Args:
            prompt: 要传递给模型的提示词。

        Returns:
            针对提示词可生成的最大 token 数量。

        Example:
            ```python
            max_tokens = openai.max_tokens_for_prompt("Tell me a joke.")
            ```
        """
        num_tokens = self.get_num_tokens(prompt)
        return self.max_context_size - num_tokens


class OpenAI(BaseOpenAI):
    """OpenAI 补全模型集成。

    设置：
        安装 `langchain-openai` 并设置环境变量 `OPENAI_API_KEY`。

        ```bash
        pip install -U langchain-openai
        export OPENAI_API_KEY="your-api-key"
        ```

    Key init args — completion params:
        model:
            Name of OpenAI model to use.
        temperature:
            Sampling temperature.
        max_tokens:
            Max number of tokens to generate.
        logprobs:
            Whether to return logprobs.
        stream_options:
            Configure streaming outputs, like whether to return token usage when
            streaming (`{"include_usage": True}`).

    Key init args — client params:
        timeout:
            Timeout for requests.
        max_retries:
            Max number of retries.
        api_key:
            OpenAI API key. If not passed in will be read from env var `OPENAI_API_KEY`.
        base_url:
            Base URL for API requests. Only specify if using a proxy or service
            emulator.
        organization:
            OpenAI organization ID. If not passed in will be read from env
            var `OPENAI_ORG_ID`.

    参见参数部分中支持的初始化参数及其描述的完整列表。

    实例化：
        ```python
        from langchain_openai import OpenAI

        model = OpenAI(
            model="gpt-3.5-turbo-instruct",
            temperature=0,
            max_retries=2,
            # api_key="...",
            # base_url="...",
            # organization="...",
            # other params...
        )
        ```

    调用：
        ```python
        input_text = "The meaning of life is "
        model.invoke(input_text)
        ```
        ```txt
        "a philosophical question that has been debated by thinkers and scholars for centuries."
        ```

    流式：
        ```python
        for chunk in model.stream(input_text):
            print(chunk, end="|")
        ```
        ```txt
        a| philosophical| question| that| has| been| debated| by| thinkers| and| scholars| for| centuries|.
        ```

        ```python
        "".join(model.stream(input_text))
        ```
        ```txt
        "a philosophical question that has been debated by thinkers and scholars for centuries."
        ```

    异步：
        ```python
        await model.ainvoke(input_text)

        # stream:
        # async for chunk in (await model.astream(input_text)):
        #    print(chunk)

        # batch:
        # await model.abatch([input_text])
        ```
        ```txt
        "a philosophical question that has been debated by thinkers and scholars for centuries."
        ```
    """  # noqa: E501

    @classmethod
    def get_lc_namespace(cls) -> list[str]:
        """获取 LangChain 对象的命名空间。

        Returns:
            `["langchain", "llms", "openai"]`
        """
        return ["langchain", "llms", "openai"]

    @classmethod
    def is_lc_serializable(cls) -> bool:
        """返回此模型是否可以被 LangChain 序列化。"""
        return True

    @property
    def _invocation_params(self) -> dict[str, Any]:
        return {"model": self.model_name, **super()._invocation_params}

    @property
    def lc_secrets(self) -> dict[str, str]:
        """密钥与环境变量的映射关系。"""
        return {"openai_api_key": "OPENAI_API_KEY"}

    @property
    def lc_attributes(self) -> dict[str, Any]:
        """此类的 LangChain 属性。"""
        attributes: dict[str, Any] = {}
        if self.openai_api_base:
            attributes["openai_api_base"] = self.openai_api_base

        if self.openai_organization:
            attributes["openai_organization"] = self.openai_organization

        if self.openai_proxy:
            attributes["openai_proxy"] = self.openai_proxy

        return attributes

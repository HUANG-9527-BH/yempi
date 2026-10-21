"""Anthropic 大语言模型封装。聊天模型位于 `chat_models.py` 中。"""

from __future__ import annotations

import re
from collections.abc import AsyncIterator, Callable, Iterator, Mapping
from typing import Any

import anthropic
from langchain_core._api.deprecation import deprecated
from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models import BaseLanguageModel, LangSmithParams
from langchain_core.language_models.llms import LLM
from langchain_core.outputs import GenerationChunk
from langchain_core.prompt_values import PromptValue
from langchain_core.utils import get_pydantic_field_names
from langchain_core.utils.utils import _build_model_kwargs, from_env, secret_from_env
from pydantic import ConfigDict, Field, SecretStr, model_validator
from typing_extensions import Self

from langchain_anthropic._sdk_compat import _route_unsupported_sampling_params


class _AnthropicCommon(BaseLanguageModel):
    client: Any = None

    async_client: Any = None

    model: str = Field(default="claude-sonnet-4-5", alias="model_name")
    """要使用的模型名称。"""

    max_tokens: int = Field(default=1024, alias="max_tokens_to_sample")
    """表示每次生成预测的 token 数量。"""

    temperature: float | None = None
    """一个非负浮点数，用于调节生成过程中的随机程度。"""

    top_k: int | None = None
    """在每一步中考虑的最可能 token 的数量。"""

    top_p: float | None = None
    """在每一步中考虑的 token 的总概率质量。"""

    streaming: bool = False
    """是否流式输出结果。"""

    default_request_timeout: float | None = None
    """对 Anthropic Completion API 请求的超时时间。默认为 600 秒。"""

    max_retries: int = 2
    """发送到 Anthropic Completion API 的请求允许重试次数。"""

    anthropic_api_url: str | None = Field(
        alias="base_url",
        default_factory=from_env(
            "ANTHROPIC_API_URL",
            default="https://api.anthropic.com",
        ),
    )
    """API 请求的基础 URL。仅在使用代理或服务模拟器时指定。

    如果未传入值，将尝试从 `ANTHROPIC_API_URL` 读取。 如果未设置，则使用默认值
    `https://api.anthropic.com` 。
    """

    anthropic_api_key: SecretStr = Field(
        alias="api_key",
        default_factory=secret_from_env("ANTHROPIC_API_KEY", default=""),
    )
    """如果未提供，将自动从环境变量 `ANTHROPIC_API_KEY` 读取。"""

    HUMAN_PROMPT: str | None = None

    AI_PROMPT: str | None = None

    count_tokens: Callable[[str], int] | None = None

    model_kwargs: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def build_extra(cls, values: dict) -> Any:
        all_required_field_names = get_pydantic_field_names(cls)
        return _build_model_kwargs(values, all_required_field_names)

    @model_validator(mode="after")
    def validate_environment(self) -> Self:
        """验证 API 密钥和 Python 包是否存在于环境中。"""
        self.client = anthropic.Anthropic(
            base_url=self.anthropic_api_url,
            api_key=self.anthropic_api_key.get_secret_value(),
            timeout=self.default_request_timeout,
            max_retries=self.max_retries,
        )
        self.async_client = anthropic.AsyncAnthropic(
            base_url=self.anthropic_api_url,
            api_key=self.anthropic_api_key.get_secret_value(),
            timeout=self.default_request_timeout,
            max_retries=self.max_retries,
        )
                # 保留以保持向后兼容性，但在 Messages API 中不再使用
        self.HUMAN_PROMPT = getattr(anthropic, "HUMAN_PROMPT", None)
        self.AI_PROMPT = getattr(anthropic, "AI_PROMPT", None)
        return self

    @property
    def _default_params(self) -> Mapping[str, Any]:
        """获取调用 Anthropic API 的默认参数。"""
        d = {
            "max_tokens": self.max_tokens,
            "model": self.model,
        }
        if self.temperature is not None:
            d["temperature"] = self.temperature
        if self.top_k is not None:
            d["top_k"] = self.top_k
        if self.top_p is not None:
            d["top_p"] = self.top_p
        return {**d, **self.model_kwargs}

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """获取标识参数。"""
        return {**self._default_params}

    def _get_anthropic_stop(self, stop: list[str] | None = None) -> list[str]:
        if stop is None:
            stop = []
        return stop


@deprecated(since="0.1.0", removal="2.0.0", alternative="ChatAnthropic")
class AnthropicLLM(LLM, _AnthropicCommon):
    """Anthropic 文本补全大型语言模型（旧版 LLM）。

    要使用，您应将环境变量 `ANTHROPIC_API_KEY`
    设置为您的 API 密钥，或将其作为命名参数传递给构造函数。

    Example:
        ```python
        from langchain_anthropic import AnthropicLLM

        model = AnthropicLLM(model="claude-sonnet-4-5")
        ```
    """

    model_config = ConfigDict(
        populate_by_name=True,
        arbitrary_types_allowed=True,
    )

    @property
    def _llm_type(self) -> str:
        """返回 LLM 的返回类型。"""
        return "anthropic-llm"

    @property
    def lc_secrets(self) -> dict[str, str]:
        """返回密钥与环境变量的映射关系。"""
        return {"anthropic_api_key": "ANTHROPIC_API_KEY"}

    @classmethod
    def is_lc_serializable(cls) -> bool:
        """该类是否可被 langchain 序列化。"""
        return True

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
            "default_request_timeout": self.default_request_timeout,
            "max_retries": self.max_retries,
        }

    def _get_ls_params(
        self,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> LangSmithParams:
        """获取标准追踪参数。"""
        params = super()._get_ls_params(stop=stop, **kwargs)
        identifying_params = self._identifying_params
        if max_tokens := kwargs.get(
            "max_tokens",
            identifying_params.get("max_tokens"),
        ):
            params["ls_max_tokens"] = max_tokens
        return params

    def _format_messages(self, prompt: str) -> list[dict[str, str]]:
        """将提示词转换为 Messages API 格式。"""
        messages = []

                # 处理可能包含 HUMAN_PROMPT/AI_PROMPT 标记的旧版提示词
        if self.HUMAN_PROMPT and self.HUMAN_PROMPT in prompt:
                        # 按人类/助手轮次分割
            parts = prompt.split(self.HUMAN_PROMPT)

            for _, part in enumerate(parts):
                if not part.strip():
                    continue

                if self.AI_PROMPT and self.AI_PROMPT in part:
                                        # 分割人类和助手部分
                    human_part, assistant_part = part.split(self.AI_PROMPT, 1)
                    if human_part.strip():
                        messages.append({"role": "user", "content": human_part.strip()})
                    if assistant_part.strip():
                        messages.append(
                            {"role": "assistant", "content": assistant_part.strip()}
                        )
                                # 仅人类内容
                elif part.strip():
                    messages.append({"role": "user", "content": part.strip()})
        else:
                        # 处理现代格式或纯文本
                        # 清理提示词以适配 Messages API
            content = re.sub(r"^\n*Human:\s*", "", prompt)
            content = re.sub(r"\n*Assistant:\s*.*$", "", content)
            if content.strip():
                messages.append({"role": "user", "content": content.strip()})

                # 确保至少有一个消息
        if not messages:
            messages = [{"role": "user", "content": prompt.strip() or "Hello"}]

        return messages

    def _call(
        self,
        prompt: str,
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> str:
        r"""调用 Anthropic 的补全端点。

        Args:
            prompt: 传递给模型的提示词。
            stop: 可选的停止词列表，用于生成时。
            run_manager: 可选的 LLM 运行回调管理器。
            kwargs: 额外的关键字参数，将传递给模型。

        Returns:
            模型生成的字符串。

        Example:
            ```python
            prompt = "What are the biggest risks facing humanity?"
            prompt = f"\n\nHuman: {prompt}\n\nAssistant:"
            response = model.invoke(prompt)
            ```
        """
        if self.streaming:
            completion = ""
            for chunk in self._stream(
                prompt=prompt,
                stop=stop,
                run_manager=run_manager,
                **kwargs,
            ):
                completion += chunk.text
            return completion

        stop = self._get_anthropic_stop(stop)
        params = {**self._default_params, **kwargs}

                # 移除 Messages API 不支持的参数
        params = {k: v for k, v in params.items() if k != "max_tokens_to_sample"}
        params = _route_unsupported_sampling_params(params)

        response = self.client.messages.create(
            messages=self._format_messages(prompt),
            stop_sequences=stop if stop else None,
            **params,
        )
        return response.content[0].text

    def convert_prompt(self, prompt: PromptValue) -> str:
        """将 `PromptValue` 转换为字符串。"""
        return prompt.to_string()

    async def _acall(
        self,
        prompt: str,
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> str:
        """异步调用 Anthropic 的补全端点。"""
        if self.streaming:
            completion = ""
            async for chunk in self._astream(
                prompt=prompt,
                stop=stop,
                run_manager=run_manager,
                **kwargs,
            ):
                completion += chunk.text
            return completion

        stop = self._get_anthropic_stop(stop)
        params = {**self._default_params, **kwargs}

                # 移除 Messages API 不支持的参数
        params = {k: v for k, v in params.items() if k != "max_tokens_to_sample"}
        params = _route_unsupported_sampling_params(params)

        response = await self.async_client.messages.create(
            messages=self._format_messages(prompt),
            stop_sequences=stop if stop else None,
            **params,
        )
        return response.content[0].text

    def _stream(
        self,
        prompt: str,
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> Iterator[GenerationChunk]:
        r"""调用 Anthropic completion_stream 并返回生成器结果。

        Args:
            prompt: 传递给模型的提示词。
            stop: 可选的停止词列表，用于生成时。
            run_manager: 可选的 LLM 运行回调管理器。
            kwargs: 额外的关键字参数，将传递给模型。

        Returns:
            代表来自 Anthropic 的 token 流的生成器。

        Example:
            ```python
            prompt = "Write a poem about a stream."
            prompt = f"\n\nHuman: {prompt}\n\nAssistant:"
            generator = anthropic.stream(prompt)
            for token in generator:
                yield token
            ```
        """
        stop = self._get_anthropic_stop(stop)
        params = {**self._default_params, **kwargs}

                # 移除 Messages API 不支持的参数
        params = {k: v for k, v in params.items() if k != "max_tokens_to_sample"}
        params = _route_unsupported_sampling_params(params)

        with self.client.messages.stream(
            messages=self._format_messages(prompt),
            stop_sequences=stop if stop else None,
            **params,
        ) as stream:
            for event in stream:
                if event.type == "content_block_delta" and hasattr(event.delta, "text"):
                    chunk = GenerationChunk(text=event.delta.text)
                    if run_manager:
                        run_manager.on_llm_new_token(chunk.text, chunk=chunk)
                    yield chunk

    async def _astream(
        self,
        prompt: str,
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[GenerationChunk]:
        r"""调用 Anthropic completion_stream 并返回生成器结果。

        Args:
            prompt: 传递给模型的提示词。
            stop: 可选的停止词列表，用于生成时。
            run_manager: 可选的 LLM 运行回调管理器。
            kwargs: 额外的关键字参数，将传递给模型。

        Returns:
            代表来自 Anthropic 的 token 流的生成器。

        Example:
            ```python
            prompt = "Write a poem about a stream."
            prompt = f"\n\nHuman: {prompt}\n\nAssistant:"
            generator = anthropic.stream(prompt)
            for token in generator:
                yield token
            ```
        """
        stop = self._get_anthropic_stop(stop)
        params = {**self._default_params, **kwargs}

                # 移除 Messages API 不支持的参数
        params = {k: v for k, v in params.items() if k != "max_tokens_to_sample"}
        params = _route_unsupported_sampling_params(params)

        async with self.async_client.messages.stream(
            messages=self._format_messages(prompt),
            stop_sequences=stop if stop else None,
            **params,
        ) as stream:
            async for event in stream:
                if event.type == "content_block_delta" and hasattr(event.delta, "text"):
                    chunk = GenerationChunk(text=event.delta.text)
                    if run_manager:
                        await run_manager.on_llm_new_token(chunk.text, chunk=chunk)
                    yield chunk

    def get_num_tokens(self, text: str) -> int:
        """计算 token 数量。"""
        msg = (
            "Anthropic's legacy count_tokens method was removed in anthropic 0.39.0 "
            "and langchain-anthropic 0.3.0. Please use "
            "ChatAnthropic.get_num_tokens_from_messages instead."
        )
        raise NotImplementedError(
            msg,
        )

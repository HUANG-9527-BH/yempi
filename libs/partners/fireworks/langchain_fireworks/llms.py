"""Fireworks AI 的 Completion API 包装器。"""

from __future__ import annotations

import logging
from typing import Any

import requests
from aiohttp import ClientSession, ClientTimeout
from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models.llms import LLM
from langchain_core.utils import get_pydantic_field_names
from langchain_core.utils.utils import _build_model_kwargs, secret_from_env
from pydantic import ConfigDict, Field, SecretStr, model_validator
from typing_extensions import Self

from langchain_fireworks._version import __version__

logger = logging.getLogger(__name__)


class Fireworks(LLM):
    """来自 `Fireworks` 的 LLM 模型。

    要使用，您需要一个 [API 密钥](https://fireworks.ai)。该密钥可作为初始化参数
    `fireworks_api_key` 传入，或设置为环境变量 `FIREWORKS_API_KEY`。

    [Fireworks AI API 参考文档](https://readme.fireworks.ai/)

    Example:
        ```python
        response = fireworks.generate(["Tell me a joke."])
        ```
    """

    base_url: str = "https://api.fireworks.ai/inference/v1/completions"
    """基础推理 API URL。"""
    fireworks_api_key: SecretStr = Field(
        alias="api_key",
        default_factory=secret_from_env(
            "FIREWORKS_API_KEY",
            error_message=(
                "You must specify an api key. "
                "You can pass it an argument as `api_key=...` or "
                "set the environment variable `FIREWORKS_API_KEY`."
            ),
        ),
    )
    """Fireworks API 密钥。

    如果未提供，将自动从环境变量 `FIREWORKS_API_KEY` 读取。
    """
    model: str
    """模型名称。[(可用模型)](https://readme.fireworks.ai/)"""
    temperature: float | None = None
    """模型温度。"""
    top_p: float | None = None
    """Used to dynamically adjust the number of choices for each predicted token based
    on the cumulative probabilities. A value of `1` will always yield the same output.
    A temperature less than `1` favors more correctness and is appropriate for
    question answering or summarization. A value greater than `1` introduces more
    randomness in the output.
    """
    model_kwargs: dict[str, Any] = Field(default_factory=dict)
    """保存 `create` 调用中有效但未显式指定的任何模型参数。"""
    top_k: int | None = None
    """Used to limit the number of choices for the next predicted word or token. It
    specifies the maximum number of tokens to consider at each step, based on their
    probability of occurrence. This technique helps to speed up the generation process
    and can improve the quality of the generated text by focusing on the most likely
    options.
    """
    max_tokens: int | None = None
    """要生成的最大 token 数量。"""
    repetition_penalty: float | None = None
    """一个控制生成文本多样性的数值，通过降低重复序列的可能性来实现。数值越高，重复越少。
    """
    logprobs: int | None = None
    """An integer that specifies how many top token log probabilities are included in
    the response for each token generation step.
    """
    timeout: int | None = 30
    """对 Fireworks API 请求的超时时间（秒）。"""

    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
    )

    @model_validator(mode="before")
    @classmethod
    def build_extra(cls, values: dict[str, Any]) -> Any:
        """从传入的额外参数构建额外的 kwargs。"""
        all_required_field_names = get_pydantic_field_names(cls)
        return _build_model_kwargs(values, all_required_field_names)

    @model_validator(mode="after")
    def _set_fireworks_version(self) -> Self:
        """在元数据中设置包版本。"""
        self._add_version("langchain-fireworks", __version__)
        return self

    @property
    def _llm_type(self) -> str:
        """模型的返回类型。"""
        return "fireworks"

    def _format_output(self, output: dict) -> str:
        return output["choices"][0]["text"]

    @staticmethod
    def get_user_agent() -> str:
        return f"langchain-fireworks/{__version__}"

    @property
    def default_params(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "max_tokens": self.max_tokens,
            "repetition_penalty": self.repetition_penalty,
        }

    def _call(
        self,
        prompt: str,
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> str:
        """调用 Fireworks 的文本生成端点。

        Args:
            prompt: 传递给模型的提示词。
            stop: 可选的停止序列列表。
            run_manager: (未使用) 可选的 LLM 运行回调管理器。
            kwargs: 额外参数，传递给模型。

        Returns:
            模型生成的字符串。

        """
        headers = {
            "Authorization": f"Bearer {self.fireworks_api_key.get_secret_value()}",
            "Content-Type": "application/json",
        }
        stop_to_use = stop[0] if stop and len(stop) == 1 else stop
        payload: dict[str, Any] = {
            **self.default_params,
            "prompt": prompt,
            "stop": stop_to_use,
            **kwargs,
        }

                # 过滤掉 None 值，避免将其传递给 HTTP 负载
        payload = {k: v for k, v in payload.items() if v is not None}
        response = requests.post(
            url=self.base_url, json=payload, headers=headers, timeout=self.timeout
        )

        if response.status_code >= 500:
            msg = f"Fireworks Server: Error {response.status_code}"
            raise Exception(msg)
        if response.status_code >= 400:
            msg = f"Fireworks received an invalid payload: {response.text}"
            raise ValueError(msg)
        if response.status_code != 200:
            msg = (
                f"Fireworks returned an unexpected response with status "
                f"{response.status_code}: {response.text}"
            )
            raise Exception(msg)

        data = response.json()
        return self._format_output(data)

    async def _acall(
        self,
        prompt: str,
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> str:
        """调用 Fireworks 模型，根据提示词获取预测结果。

        Args:
            prompt: 传递给模型的提示词。
            stop: 可选的字符串列表，遇到时停止生成。
            run_manager: (未使用) 可选的异步运行回调管理器。
            kwargs: 额外参数，传递给模型。

        Returns:
            模型生成的字符串。

        """
        headers = {
            "Authorization": f"Bearer {self.fireworks_api_key.get_secret_value()}",
            "Content-Type": "application/json",
        }
        stop_to_use = stop[0] if stop and len(stop) == 1 else stop
        payload: dict[str, Any] = {
            **self.default_params,
            "prompt": prompt,
            "stop": stop_to_use,
            **kwargs,
        }

                # 过滤掉 None 值，避免将其传递给 HTTP 负载
        payload = {k: v for k, v in payload.items() if v is not None}
        async with (
            ClientSession() as session,
            session.post(
                self.base_url,
                json=payload,
                headers=headers,
                timeout=ClientTimeout(total=self.timeout),
            ) as response,
        ):
            if response.status >= 500:
                msg = f"Fireworks Server: Error {response.status}"
                raise Exception(msg)
            if response.status >= 400:
                msg = f"Fireworks received an invalid payload: {response.text}"
                raise ValueError(msg)
            if response.status != 200:
                msg = (
                    f"Fireworks returned an unexpected response with status "
                    f"{response.status}: {response.text}"
                )
                raise Exception(msg)

            response_json = await response.json()
            return self._format_output(response_json)

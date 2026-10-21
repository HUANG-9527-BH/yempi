from __future__ import annotations

import inspect
import logging
import os
from collections.abc import AsyncIterator, Iterator, Mapping
from typing import Any
from urllib.parse import urlparse

from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models.llms import LLM
from langchain_core.outputs import GenerationChunk
from langchain_core.utils import from_env, get_pydantic_field_names
from pydantic import ConfigDict, Field, model_validator
from typing_extensions import Self

from langchain_huggingface._version import __version__

logger = logging.getLogger(__name__)


def _is_huggingface_hosted_url(url: str | None) -> bool:
    """如果 url 是 Hugging Face 托管的（huggingface.co 或 hf.space）则为 True。"""
    if not url:
        return False
    hostname = (urlparse(url).hostname or "").lower()
    return (
        hostname == "huggingface.co"
        or hostname == "hf.space"
        or hostname.endswith((".huggingface.co", ".hf.space"))
    )


VALID_TASKS = (
    "text2text-generation",
    "text-generation",
    "summarization",
    "conversational",
)


class HuggingFaceEndpoint(LLM):
    """Hugging Face 推理端点。此功能适用于任何支持文本生成（即文本补全）任务的模型。

    要使用此类，您应安装 `huggingface_hub` 包，并
    设置环境变量 `HUGGINGFACEHUB_API_TOKEN` 以包含您的 API token，
    或作为构造函数的命名参数传入。

    Example:
        ```python
        # Basic Example (no streaming)
        model = HuggingFaceEndpoint(
            endpoint_url="http://localhost:8010/",
            max_new_tokens=512,
            top_k=10,
            top_p=0.95,
            typical_p=0.95,
            temperature=0.01,
            repetition_penalty=1.03,
            huggingfacehub_api_token="my-api-key",
        )
        print(model.invoke("What is Deep Learning?"))

        # Streaming response example
        from langchain_core.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

        callbacks = [StreamingStdOutCallbackHandler()]
        model = HuggingFaceEndpoint(
            endpoint_url="http://localhost:8010/",
            max_new_tokens=512,
            top_k=10,
            top_p=0.95,
            typical_p=0.95,
            temperature=0.01,
            repetition_penalty=1.03,
            callbacks=callbacks,
            streaming=True,
            huggingfacehub_api_token="my-api-key",
        )
        print(model.invoke("What is Deep Learning?"))

        # Basic Example (no streaming) with Mistral-Nemo-Base-2407 model using a third-party provider (Novita).
        model = HuggingFaceEndpoint(
            repo_id="mistralai/Mistral-Nemo-Base-2407",
            provider="novita",
            max_new_tokens=100,
            do_sample=False,
            huggingfacehub_api_token="my-api-key",
        )
        print(model.invoke("What is Deep Learning?"))
        ```
    """  # noqa: E501

    endpoint_url: str | None = None
    """要使用的端点 URL。如果未指定 repo_id，则需要提供此值或
    通过环境变量 `HF_INFERENCE_ENDPOINT` 传递。"""

    repo_id: str | None = None
    """要使用的仓库。如果未指定 endpoint_url，则需要提供此值"""

    provider: str | None = None
    """Name of the provider to use for inference with the model specified in `repo_id`.
        e.g. "cerebras". if not specified, Defaults to "auto" i.e. the first of the
        providers available for the model, sorted by the user's order in https://hf.co/settings/inference-providers.
        available providers can be found in the [huggingface_hub documentation](https://huggingface.co/docs/huggingface_hub/guides/inference#supported-providers-and-tasks)."""

    huggingfacehub_api_token: str | None = Field(
        default_factory=from_env("HUGGINGFACEHUB_API_TOKEN", default=None),
        repr=False,
    )

    max_new_tokens: int = 512
    """生成的最大 token 数量"""

    top_k: int | None = None
    """保留用于 top-k 过滤的最高概率词汇表 token 数量"""

    top_p: float | None = 0.95
    """If set to < 1, only the smallest set of most probable tokens with probabilities
    that add up to `top_p` or higher are kept for generation."""

    typical_p: float | None = 0.95
    """Typical Decoding mass. See [Typical Decoding for Natural Language
    Generation](https://arxiv.org/abs/2202.00666) for more information."""

    temperature: float | None = 0.8
    """用于调节 logits 分布的值。"""

    repetition_penalty: float | None = None
    """重复惩罚参数。1.0 表示无惩罚。
    详情请参见 [这篇论文](https://arxiv.org/pdf/1909.05858.pdf)。"""

    return_full_text: bool = False
    """是否在生成文本前添加提示词"""

    truncate: int | None = None
    """将输入 token 截断至指定大小"""

    stop_sequences: list[str] = Field(default_factory=list)
    """如果生成了 `stop_sequences` 中的任意成员，则停止生成 token"""

    seed: int | None = None
    """随机采样种子"""

    inference_server_url: str = ""
    """text-generation-inference 实例基础 URL"""

    timeout: int = 120
    """超时时间（秒）"""

    streaming: bool = False
    """是否异步流式生成 token"""

    do_sample: bool = False
    """启用 logits 采样"""

    watermark: bool = False
    """使用 [大型语言模型的水印技术]
    (https://arxiv.org/abs/2301.10226)"""

    server_kwargs: dict[str, Any] = Field(default_factory=dict)
    """保存未明确指定的 text-generation-inference 服务器参数"""

    model_kwargs: dict[str, Any] = Field(default_factory=dict)
    """保存 `call` 有效但未明确指定的模型参数"""

    model: str

    client: Any = None

    async_client: Any = None

    task: str | None = None
    """要调用模型的任务。应为返回 `generated_text` 的任务。"""

    model_config = ConfigDict(
        extra="forbid",
    )

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
                logger.warning(
                    f"""WARNING! {field_name} is not default parameter.
                    {field_name} was transferred to model_kwargs.
                    Please make sure that {field_name} is what you intended."""
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

                # 正确创建 InferenceClient 和 AsyncInferenceClient
                # 在 validate_environment 中，我们需要填充 values["model"]。
                # 来自 InferenceClient 文档字符串：
                # model (`str`, `optional`):
                #     要运行推理的模型。可以是托管在 Hugging
                #       Face Hub 上的模型 ID，例如 `bigcode/starcoder`
                #     或部署的推理端点的 URL。默认为 `None`，此时会使用推荐的模型
                #       模型
                #     自动为该任务选择。

                # 此字符串可能出现在三个优先级递减的位置：
                # 2. values["model"] 或 values["endpoint_url"] 或 values["repo_id"]
                #       (优先级相等 - 不允许同时设置)
                # 3. values["HF_INFERENCE_ENDPOINT"] (如果上述任一未设置)

        model = values.get("model")
        endpoint_url = values.get("endpoint_url")
        repo_id = values.get("repo_id")

        if repo_id and repo_id.startswith(("http://", "https://")):
            msg = (
                "`repo_id` must be a HuggingFace repo ID, not a URL. "
                "Use `endpoint_url` for direct endpoints."
            )
            raise ValueError(msg)

        if sum([bool(model), bool(endpoint_url), bool(repo_id)]) > 1:
            msg = (
                "Please specify either a `model` OR an `endpoint_url` OR a `repo_id`,"
                "not more than one."
            )
            raise ValueError(msg)
        values["model"] = (
            model or endpoint_url or repo_id or os.environ.get("HF_INFERENCE_ENDPOINT")
        )
        if not values["model"]:
            msg = (
                "Please specify a `model` or an `endpoint_url` or a `repo_id` for the "
                "model."
            )
            raise ValueError(msg)
        return values

    @model_validator(mode="after")
    def _set_huggingface_version(self) -> Self:
        """在元数据中设置包版本。"""
        self._add_version("langchain-huggingface", __version__)
        return self

    @model_validator(mode="after")
    def validate_environment(self) -> Self:
        """验证包是否已安装且 API token 是否有效。"""
        huggingfacehub_api_token = self.huggingfacehub_api_token or os.getenv(
            "HF_TOKEN"
        )
                # 本地/自定义端点 URL -> 不传递 HF token（避免 401 错误和出站流量）。
        if self.endpoint_url and not _is_huggingface_hosted_url(self.endpoint_url):
            client_api_key: str | None = None
        else:
            client_api_key = huggingfacehub_api_token

        from huggingface_hub import (  # type: ignore[import]
            AsyncInferenceClient,  # type: ignore[import]
            InferenceClient,  # type: ignore[import]
        )

                # 使用支持的 kwargs 实例化客户端
        sync_supported_kwargs = set(inspect.signature(InferenceClient).parameters)
        self.client = InferenceClient(
            model=self.model,
            timeout=self.timeout,
            api_key=client_api_key,
            provider=self.provider,  # type: ignore[arg-type]
            **{
                key: value
                for key, value in self.server_kwargs.items()
                if key in sync_supported_kwargs
            },
        )

        async_supported_kwargs = set(inspect.signature(AsyncInferenceClient).parameters)
        self.async_client = AsyncInferenceClient(
            model=self.model,
            timeout=self.timeout,
            api_key=client_api_key,
            provider=self.provider,  # type: ignore[arg-type]
            **{
                key: value
                for key, value in self.server_kwargs.items()
                if key in async_supported_kwargs
            },
        )
        ignored_kwargs = (
            set(self.server_kwargs.keys())
            - sync_supported_kwargs
            - async_supported_kwargs
        )
        if len(ignored_kwargs) > 0:
            logger.warning(
                f"Ignoring following parameters as they are not supported by the "
                f"InferenceClient or AsyncInferenceClient: {ignored_kwargs}."
            )

        return self

    @property
    def _default_params(self) -> dict[str, Any]:
        """获取调用文本生成推理 API 的默认参数。"""
        return {
            "max_new_tokens": self.max_new_tokens,
            "top_k": self.top_k,
            "top_p": self.top_p,
            "typical_p": self.typical_p,
            "temperature": self.temperature,
            "repetition_penalty": self.repetition_penalty,
            "return_full_text": self.return_full_text,
            "truncate": self.truncate,
            "stop": self.stop_sequences,
            "seed": self.seed,
            "do_sample": self.do_sample,
            "watermark": self.watermark,
            **self.model_kwargs,
        }

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """获取标识参数。"""
        _model_kwargs = self.model_kwargs or {}
        return {
            "endpoint_url": self.endpoint_url,
            "task": self.task,
            "provider": self.provider,
            "model_kwargs": _model_kwargs,
        }

    @property
    def _llm_type(self) -> str:
        """LLM 的返回类型。"""
        return "huggingface_endpoint"

    def _invocation_params(
        self, runtime_stop: list[str] | None, **kwargs: Any
    ) -> dict[str, Any]:
        params = {**self._default_params, **kwargs}
        params["stop"] = params["stop"] + (runtime_stop or [])
        return params

    def _call(
        self,
        prompt: str,
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> str:
        """调用 HuggingFace Hub 的推理端点。"""
        invocation_params = self._invocation_params(stop, **kwargs)
        if self.streaming:
            completion = ""
            for chunk in self._stream(
                prompt, run_manager=run_manager, **invocation_params
            ):
                completion += chunk.text
            return completion

        response_text = self.client.text_generation(
            prompt=prompt,
            model=self.model,
            **invocation_params,
        )

                # 生成可能在某个停止序列处终止：
                # 然后从生成文本末尾移除该停止序列
        for stop_seq in invocation_params["stop"]:
            if response_text[-len(stop_seq) :] == stop_seq:
                response_text = response_text[: -len(stop_seq)]
        return response_text

    async def _acall(
        self,
        prompt: str,
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> str:
        invocation_params = self._invocation_params(stop, **kwargs)
        if self.streaming:
            completion = ""
            async for chunk in self._astream(
                prompt, run_manager=run_manager, **invocation_params
            ):
                completion += chunk.text
            return completion

        response_text = await self.async_client.text_generation(
            prompt=prompt,
            **invocation_params,
            model=self.model,
            stream=False,
        )

                # 生成可能在某个停止序列处终止：
                # 然后从生成文本末尾移除该停止序列
        for stop_seq in invocation_params["stop"]:
            if response_text[-len(stop_seq) :] == stop_seq:
                response_text = response_text[: -len(stop_seq)]
        return response_text

    def _stream(
        self,
        prompt: str,
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> Iterator[GenerationChunk]:
        invocation_params = self._invocation_params(stop, **kwargs)

        for response in self.client.text_generation(
            prompt, **invocation_params, stream=True
        ):
                        # 识别生成文本中的停止序列（如有）
            stop_seq_found: str | None = None
            for stop_seq in invocation_params["stop"]:
                if stop_seq in response:
                    stop_seq_found = stop_seq

                        # 识别要输出的文本
            text: str | None = None
            if stop_seq_found:
                text = response[: response.index(stop_seq_found)]
            else:
                text = response

                        # 如果有，输出文本
            if text:
                chunk = GenerationChunk(text=text)

                if run_manager:
                    run_manager.on_llm_new_token(chunk.text)
                yield chunk

                        # 若发现停止序列则中断
            if stop_seq_found:
                break

    async def _astream(
        self,
        prompt: str,
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[GenerationChunk]:
        invocation_params = self._invocation_params(stop, **kwargs)
        async for response in await self.async_client.text_generation(
            prompt, **invocation_params, stream=True
        ):
                        # 识别生成文本中的停止序列（如有）
            stop_seq_found: str | None = None
            for stop_seq in invocation_params["stop"]:
                if stop_seq in response:
                    stop_seq_found = stop_seq

                        # 识别要输出的文本
            text: str | None = None
            if stop_seq_found:
                text = response[: response.index(stop_seq_found)]
            else:
                text = response

                        # 如果有，输出文本
            if text:
                chunk = GenerationChunk(text=text)

                if run_manager:
                    await run_manager.on_llm_new_token(chunk.text)
                yield chunk

                        # 若发现停止序列则中断
            if stop_seq_found:
                break

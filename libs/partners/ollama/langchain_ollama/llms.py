"""Ollama 大型语言模型。"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator, Mapping
from typing import Any, Literal

from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models import BaseLLM, LangSmithParams
from langchain_core.outputs import GenerationChunk, LLMResult
from ollama import AsyncClient, Client, Options
from pydantic import PrivateAttr, model_validator
from typing_extensions import Self

from langchain_ollama._utils import (
    merge_auth_headers,
    parse_url_with_auth,
    validate_model,
)
from langchain_ollama._version import __version__


class OllamaLLM(BaseLLM):
    """Ollama 大型语言模型。

    设置：
        安装 `langchain-ollama` 并在本地安装/运行 Ollama 服务器：

        ```bash
        pip install -U langchain-ollama
        # Visit https://ollama.com/download to download and install Ollama
        # (Linux users): start the server with `ollama serve`
        ```

        下载要使用的模型：

        ```bash
        ollama pull llama3.1
        ```

    关键初始化参数 —— 生成参数：
        model: str
            要使用的 Ollama 模型名称（例如 `'llama4'`）。
        temperature: float | None
            采样温度。值越高，输出越具创造性。
        num_predict: int | None
            最大预测 token 数量。
        top_k: int | None
            将下一个 token 的选择限制为最可能的 K 个 token。
        top_p: float | None
            核采样参数。值越高，文本多样性越强。
        mirostat: int | None
            启用 Mirostat 采样以控制困惑度。
        seed: int | None
            用于生成可复现性的随机数种子。

    关键初始化参数 —— 客户端参数：
        base_url:
            Ollama 服务器托管的基地址。
        keep_alive:
            模型在内存中保持加载的时间长度。
        format:
            指定输出格式。

    请参阅参数部分中的完整支持初始化参数列表及其说明。

    实例化：
        ```python
        from langchain_ollama import OllamaLLM

        model = OllamaLLM(
            model="llama3.1",
            temperature=0.7,
            num_predict=256,
            # base_url="http://localhost:11434",
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
        "a philosophical question that has been contemplated by humans for
        centuries..."
        ```

    流式：
        ```python
        for chunk in model.stream(input_text):
            print(chunk, end="")
        ```
        ```txt
        a philosophical question that has been contemplated by humans for
        centuries...
        ```

    异步：
        ```python
        response = await model.ainvoke(input_text)

        # stream:
        # async for chunk in model.astream(input_text):
        #     print(chunk, end="")
        ```
    """

    model: str
    """要使用的模型名称。"""

    reasoning: bool | None = None
    """Controls the reasoning/thinking mode for
    [supported models](https://ollama.com/search?c=thinking).

    - `True`: 启用推理模式。模型的推理过程将被捕捉并在响应消息的 `additional_kwargs`
    中单独返回，位于 `reasoning_content` 键下。主响应内容不会包含推理标签。 - `False`:
    禁用推理模式。模型不会进行任何推理，响应也不会包含任何推理内容。 -
    `None`（默认）：模型将使用其默认推理行为。如果模型执行推理，`<think>` 和 `</think>`
    标签将直接出现在主响应内容中。"""

    validate_model_on_init: bool = False
    """是否在初始化时验证模型是否存在于本地 Ollama 中。

    !!! version-added "在 `langchain-ollama` 0.3.4 中新增"
    """

    mirostat: int | None = None
    """启用 Mirostat 采样以控制困惑度。
    （默认：`0`，`0` = 禁用，`1` = Mirostat，`2` = Mirostat 2.0）"""

    mirostat_eta: float | None = None
    """Influences how quickly the algorithm responds to feedback
    from the generated text. A lower learning rate will result in
    slower adjustments, while a higher learning rate will make
    the algorithm more responsive. (Default: `0.1`)"""

    mirostat_tau: float | None = None
    """Controls the balance between coherence and diversity
    of the output. A lower value will result in more focused and
    coherent text. (Default: `5.0`)"""

    num_ctx: int | None = None
    """Sets the size of the context window used to generate the
    next token. (Default: `2048`)"""

    num_gpu: int | None = None
    """要使用的 GPU 数量。在 macOS 上默认为 `1` 以启用 metal 支持，`0` 表示禁用。"""

    num_thread: int | None = None
    """Sets the number of threads to use during computation.
    By default, Ollama will detect this for optimal performance.
    It is recommended to set this value to the number of physical
    CPU cores your system has (as opposed to the logical number of cores)."""

    num_predict: int | None = None
    """生成文本时预测的最大 token 数量。
    （默认：`128`，`-1` = 无限生成，`-2` = 填充上下文）"""

    repeat_last_n: int | None = None
    """Sets how far back for the model to look back to prevent
    repetition. (Default: `64`, `0` = disabled, `-1` = `num_ctx`)"""

    repeat_penalty: float | None = None
    """Sets how strongly to penalize repetitions. A higher value (e.g., `1.5`)
    will penalize repetitions more strongly, while a lower value (e.g., `0.9`)
    will be more lenient. (Default: `1.1`)"""

    temperature: float | None = None
    """The temperature of the model. Increasing the temperature will
    make the model answer more creatively. (Default: `0.8`)"""

    seed: int | None = None
    """Sets the random number seed to use for generation. Setting this
    to a specific number will make the model generate the same text for
    the same prompt."""

    stop: list[str] | None = None
    """设置要使用的停止 token。"""

    tfs_z: float | None = None
    """Tail free sampling is used to reduce the impact of less probable
    tokens from the output. A higher value (e.g., `2.0`) will reduce the
    impact more, while a value of 1.0 disables this setting. (default: `1`)"""

    top_k: int | None = None
    """Reduces the probability of generating nonsense. A higher value (e.g. `100`)
    will give more diverse answers, while a lower value (e.g. `10`)
    will be more conservative. (Default: `40`)"""

    top_p: float | None = None
    """Works together with top-k. A higher value (e.g., `0.95`) will lead
    to more diverse text, while a lower value (e.g., `0.5`) will
    generate more focused and conservative text. (Default: `0.9`)"""

    format: Literal["", "json"] = ""
    """指定输出格式（选项：`'json'`）"""

    keep_alive: int | str | None = None
    """模型在内存中保持加载的时间长度。"""

    base_url: str | None = None
    """模型托管的基础 URL。

    如果未指定，则默认为 Ollama 客户端默认值。

    支持 `userinfo` 认证格式，如 `http://username:password@localhost:11434`。
    当你的 Ollama 服务器位于代理后方时非常有用。

    !!! warning
        `userinfo` is not secure and should only be used for local testing or
        in secure environments. Avoid using it in production or over unsecured
        networks.

    !!! note 使用 `userinfo` 时，请确保 Ollama 服务器已配置为接受并验证这些凭据。

    !!! note
        `userinfo` 头信息会传递给同步和异步客户端。

    """

    client_kwargs: dict | None = {}
    """额外的 kwargs，用于传递给 httpx 客户端。在此处传递 headers。

    这些参数会同时传递给同步和异步客户端。

    使用 `sync_client_kwargs` 和 `async_client_kwargs`
    可分别为同步和异步客户端传递不同的参数。
    """

    async_client_kwargs: dict | None = {}
    """在传递给 httpx 客户端前，与 `client_kwargs` 合并的额外 kwargs。

    这些是异步客户端独有的参数；共享参数请使用 `client_kwargs`。

    完整参数列表，请参阅 [httpx 文档](https://www.python-httpx.org/api/#asyncclient)。
    """

    sync_client_kwargs: dict | None = {}
    """在传递给 httpx 客户端前，与 `client_kwargs` 合并的额外 kwargs。

    这些是同步客户端独有的参数；共享参数请使用 `client_kwargs`。

    有关完整参数列表，请参阅 [httpx 文档](https://www.python-httpx.org/api/#client)。
    """

    _client: Client | None = PrivateAttr(default=None)
    """用于发起请求的客户端。"""

    _async_client: AsyncClient | None = PrivateAttr(default=None)
    """用于发起请求的异步客户端。"""

    def _generate_params(
        self,
        prompt: str,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        if self.stop is not None and stop is not None:
            msg = "`stop` found in both the input and default params."
            raise ValueError(msg)
        if self.stop is not None:
            stop = self.stop

        options_dict = kwargs.pop(
            "options",
            {
                "mirostat": self.mirostat,
                "mirostat_eta": self.mirostat_eta,
                "mirostat_tau": self.mirostat_tau,
                "num_ctx": self.num_ctx,
                "num_gpu": self.num_gpu,
                "num_thread": self.num_thread,
                "num_predict": self.num_predict,
                "repeat_last_n": self.repeat_last_n,
                "repeat_penalty": self.repeat_penalty,
                "temperature": self.temperature,
                "seed": self.seed,
                "stop": self.stop if stop is None else stop,
                "tfs_z": self.tfs_z,
                "top_k": self.top_k,
                "top_p": self.top_p,
            },
        )

        return {
            "prompt": prompt,
            "stream": kwargs.pop("stream", True),
            "model": kwargs.pop("model", self.model),
            "think": kwargs.pop("reasoning", self.reasoning),
            "format": kwargs.pop("format", self.format),
            "options": Options(**options_dict),
            "keep_alive": kwargs.pop("keep_alive", self.keep_alive),
            **kwargs,
        }

    @property
    def _llm_type(self) -> str:
        """LLM 的返回类型。"""
        return "ollama-llm"

    def _get_ls_params(
        self, stop: list[str] | None = None, **kwargs: Any
    ) -> LangSmithParams:
        """获取追踪的标准参数。"""
        params = super()._get_ls_params(stop=stop, **kwargs)
        if max_tokens := kwargs.get("num_predict", self.num_predict):
            params["ls_max_tokens"] = max_tokens
        return params

    @model_validator(mode="after")
    def _set_ollama_llm_version(self) -> Self:
        """在元数据中设置包版本。"""
        self._add_version("langchain-ollama", __version__)
        return self

    @model_validator(mode="after")
    def _set_clients(self) -> Self:
        """设置用于 ollama 的客户端。"""
        client_kwargs = self.client_kwargs or {}

        cleaned_url, auth_headers = parse_url_with_auth(self.base_url)
        merge_auth_headers(client_kwargs, auth_headers)

        sync_client_kwargs = client_kwargs
        if self.sync_client_kwargs:
            sync_client_kwargs = {**sync_client_kwargs, **self.sync_client_kwargs}

        async_client_kwargs = client_kwargs
        if self.async_client_kwargs:
            async_client_kwargs = {**async_client_kwargs, **self.async_client_kwargs}

        self._client = Client(host=cleaned_url, **sync_client_kwargs)
        self._async_client = AsyncClient(host=cleaned_url, **async_client_kwargs)
        if self.validate_model_on_init:
            validate_model(self._client, self.model)
        return self

    async def _acreate_generate_stream(
        self,
        prompt: str,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[Mapping[str, Any] | str]:
        if not self._async_client:
            msg = (
                "Ollama async client is not initialized. "
                "Make sure the model was properly constructed."
            )
            raise RuntimeError(msg)
        async for part in await self._async_client.generate(
            **self._generate_params(prompt, stop=stop, **kwargs)
        ):
            yield part

    def _create_generate_stream(
        self,
        prompt: str,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> Iterator[Mapping[str, Any] | str]:
        if not self._client:
            msg = (
                "Ollama sync client is not initialized. "
                "Make sure the model was properly constructed."
            )
            raise RuntimeError(msg)
        yield from self._client.generate(
            **self._generate_params(prompt, stop=stop, **kwargs)
        )

    async def _astream_with_aggregation(
        self,
        prompt: str,
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        verbose: bool = False,  # noqa: FBT002
        **kwargs: Any,
    ) -> GenerationChunk:
        final_chunk = None
        thinking_content = ""
        async for stream_resp in self._acreate_generate_stream(prompt, stop, **kwargs):
            if not isinstance(stream_resp, str):
                if stream_resp.get("thinking"):
                    thinking_content += stream_resp["thinking"]
                chunk = GenerationChunk(
                    text=stream_resp.get("response", ""),
                    generation_info=(
                        dict(stream_resp) if stream_resp.get("done") is True else None
                    ),
                )
                if final_chunk is None:
                    final_chunk = chunk
                else:
                    final_chunk += chunk
                if run_manager:
                    await run_manager.on_llm_new_token(
                        chunk.text,
                        chunk=chunk,
                        verbose=verbose,
                    )
        if final_chunk is None:
            msg = "No data received from Ollama stream."
            raise ValueError(msg)

        if thinking_content:
            if final_chunk.generation_info:
                final_chunk.generation_info["thinking"] = thinking_content
            else:
                final_chunk.generation_info = {"thinking": thinking_content}

        return final_chunk

    def _stream_with_aggregation(
        self,
        prompt: str,
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        verbose: bool = False,  # noqa: FBT002
        **kwargs: Any,
    ) -> GenerationChunk:
        final_chunk = None
        thinking_content = ""
        for stream_resp in self._create_generate_stream(prompt, stop, **kwargs):
            if not isinstance(stream_resp, str):
                if stream_resp.get("thinking"):
                    thinking_content += stream_resp["thinking"]
                chunk = GenerationChunk(
                    text=stream_resp.get("response", ""),
                    generation_info=(
                        dict(stream_resp) if stream_resp.get("done") is True else None
                    ),
                )
                if final_chunk is None:
                    final_chunk = chunk
                else:
                    final_chunk += chunk
                if run_manager:
                    run_manager.on_llm_new_token(
                        chunk.text,
                        chunk=chunk,
                        verbose=verbose,
                    )
        if final_chunk is None:
            msg = "No data received from Ollama stream."
            raise ValueError(msg)

        if thinking_content:
            if final_chunk.generation_info:
                final_chunk.generation_info["thinking"] = thinking_content
            else:
                final_chunk.generation_info = {"thinking": thinking_content}

        return final_chunk

    def _generate(
        self,
        prompts: list[str],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> LLMResult:
        generations = []
        for prompt in prompts:
            final_chunk = self._stream_with_aggregation(
                prompt,
                stop=stop,
                run_manager=run_manager,
                verbose=self.verbose,
                **kwargs,
            )
            generations.append([final_chunk])
        return LLMResult(generations=generations)

    async def _agenerate(
        self,
        prompts: list[str],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> LLMResult:
        generations = []
        for prompt in prompts:
            final_chunk = await self._astream_with_aggregation(
                prompt,
                stop=stop,
                run_manager=run_manager,
                verbose=self.verbose,
                **kwargs,
            )
            generations.append([final_chunk])
        return LLMResult(generations=generations)

    def _stream(
        self,
        prompt: str,
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> Iterator[GenerationChunk]:
        reasoning = kwargs.get("reasoning", self.reasoning)
        for stream_resp in self._create_generate_stream(prompt, stop, **kwargs):
            if not isinstance(stream_resp, str):
                additional_kwargs = {}
                if reasoning and (thinking_content := stream_resp.get("thinking")):
                    additional_kwargs["reasoning_content"] = thinking_content

                chunk = GenerationChunk(
                    text=(stream_resp.get("response", "")),
                    generation_info={
                        "finish_reason": self.stop,
                        **additional_kwargs,
                        **(
                            dict(stream_resp) if stream_resp.get("done") is True else {}
                        ),
                    },
                )
                if run_manager:
                    run_manager.on_llm_new_token(
                        chunk.text,
                        verbose=self.verbose,
                    )
                yield chunk

    async def _astream(
        self,
        prompt: str,
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[GenerationChunk]:
        reasoning = kwargs.get("reasoning", self.reasoning)
        async for stream_resp in self._acreate_generate_stream(prompt, stop, **kwargs):
            if not isinstance(stream_resp, str):
                additional_kwargs = {}
                if reasoning and (thinking_content := stream_resp.get("thinking")):
                    additional_kwargs["reasoning_content"] = thinking_content

                chunk = GenerationChunk(
                    text=(stream_resp.get("response", "")),
                    generation_info={
                        "finish_reason": self.stop,
                        **additional_kwargs,
                        **(
                            dict(stream_resp) if stream_resp.get("done") is True else {}
                        ),
                    },
                )
                if run_manager:
                    await run_manager.on_llm_new_token(
                        chunk.text,
                        verbose=self.verbose,
                    )
                yield chunk

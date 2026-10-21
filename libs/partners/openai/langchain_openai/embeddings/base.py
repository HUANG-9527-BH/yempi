"""OpenAI 嵌入的基类。"""

from __future__ import annotations

import logging
import warnings
from collections.abc import Awaitable, Callable, Iterable, Mapping, Sequence
from typing import Any, Literal, cast

import openai
import tiktoken
from langchain_core.embeddings import Embeddings
from langchain_core.runnables.config import run_in_executor
from langchain_core.utils import from_env, get_pydantic_field_names, secret_from_env
from pydantic import BaseModel, ConfigDict, Field, SecretStr, model_validator
from typing_extensions import Self

from langchain_openai.chat_models._client_utils import _resolve_sync_and_async_api_keys

logger = logging.getLogger(__name__)

MAX_TOKENS_PER_REQUEST = 300000
"""API limit per request for embedding tokens."""


def _process_batched_chunked_embeddings(
    num_texts: int,
    tokens: list[list[int] | str],
    batched_embeddings: list[list[float]],
    indices: list[int],
    skip_empty: bool,
) -> list[list[float] | None]:
        # for each text, this is the list of embeddings (list of list of floats)
        # corresponding to the chunks of the text
    results: list[list[list[float]]] = [[] for _ in range(num_texts)]

        # for each text, this is the token length of each chunk
        # for transformers tokenization, this is the string length
        # for tiktoken, this is the number of tokens
    num_tokens_in_batch: list[list[int]] = [[] for _ in range(num_texts)]

    for i in range(len(indices)):
        if skip_empty and len(batched_embeddings[i]) == 1:
            continue
        results[indices[i]].append(batched_embeddings[i])
        num_tokens_in_batch[indices[i]].append(len(tokens[i]))

        # for each text, this is the final embedding
    embeddings: list[list[float] | None] = []
    for i in range(num_texts):
                # an embedding for each chunk
        _result: list[list[float]] = results[i]

        if len(_result) == 0:
                        # this will be populated with the embedding of an empty string
                        # in the sync or async code calling this
            embeddings.append(None)
            continue

        if len(_result) == 1:
                        # if only one embedding was produced, use it
            embeddings.append(_result[0])
            continue

                # else we need to weighted average
                # should be same as
                # average = np.average(_result, axis=0, weights=num_tokens_in_batch[i])
        total_weight = sum(num_tokens_in_batch[i])
        average = [
            sum(
                val * weight
                for val, weight in zip(embedding, num_tokens_in_batch[i], strict=False)
            )
            / total_weight
            for embedding in zip(*_result, strict=False)
        ]

                # should be same as
                # embeddings.append((average / np.linalg.norm(average)).tolist())
        magnitude = sum(val**2 for val in average) ** 0.5
        embeddings.append([val / magnitude for val in average])

    return embeddings


class OpenAIEmbeddings(BaseModel, Embeddings):
    """OpenAI 嵌入模型集成。

    设置：
        安装 `langchain_openai` 并设置环境变量 `OPENAI_API_KEY`。

        ```bash
        pip install -U langchain_openai
        export OPENAI_API_KEY="your-api-key"
        ```

    关键初始化参数 —— 嵌入参数：
        model:
            要使用的 OpenAI 模型名称。
        dimensions:
            输出嵌入应具有的维度数。
            仅在 `'text-embedding-3'` 及后续模型中受支持。

    Key init args — client params:
        api_key:
            OpenAI API key.
        organization:
            OpenAI organization ID. If not passed in will be read
            from env var `OPENAI_ORG_ID`.
        max_retries:
            Maximum number of retries to make when generating.
        request_timeout:
            Timeout for requests to OpenAI completion API

    有关支持的初始化参数及其描述的完整列表，请参阅参数部分。

    实例化：
        ```python
        from langchain_openai import OpenAIEmbeddings

        embed = OpenAIEmbeddings(
            model="text-embedding-3-large"
            # With the `text-embedding-3` class
            # of models, you can specify the size
            # of the embeddings you want returned.
            # dimensions=1024
        )
        ```

    嵌入单个文本：
        ```python
        input_text = "The meaning of life is 42"
        vector = embeddings.embed_query("hello")
        print(vector[:3])
        ```
        ```python
        [-0.024603435769677162, -0.007543657906353474, 0.0039630369283258915]
        ```

    嵌入多个文本：
        ```python
        vectors = embeddings.embed_documents(["hello", "goodbye"])
        # Showing only the first 3 coordinates
        print(len(vectors))
        print(vectors[0][:3])
        ```
        ```python
        2
        [-0.024603435769677162, -0.007543657906353474, 0.0039630369283258915]
        ```

    异步：
        ```python
        await embed.aembed_query(input_text)
        print(vector[:3])

        # multiple:
        # await embed.aembed_documents(input_texts)
        ```
        ```python
        [-0.009100092574954033, 0.005071679595857859, -0.0029193938244134188]
        ```

    !!! note "OpenAI 兼容 API（如 OpenRouter、Ollama、vLLM）"

        使用非 OpenAI 提供商时，设置
        `check_embedding_ctx_length=False` 以发送原始文本而非 token
        （许多提供方不支持），并可选地设置
        `encoding_format` 为 `'float'` 以避免 base64 编码问题：

        ```python
        from langchain_openai import OpenAIEmbeddings

        embeddings = OpenAIEmbeddings(
            model="...",
            base_url="...",
            check_embedding_ctx_length=False,
        )
        ```

    """

    client: Any = Field(default=None, exclude=True)

    async_client: Any = Field(default=None, exclude=True)

    model: str = "text-embedding-ada-002"

    dimensions: int | None = None
    """输出嵌入应具有的维度数。

    仅在 `'text-embedding-3'` 及后续模型中受支持。
    """

        # to support Azure OpenAI Service custom deployment names
    deployment: str | None = model

        # TODO: Move to AzureOpenAIEmbeddings.
    openai_api_version: str | None = Field(
        default_factory=from_env("OPENAI_API_VERSION", default=None),
        alias="api_version",
    )
    """要使用的 OpenAI API 版本。

    如果未提供，则自动从环境变量 `OPENAI_API_VERSION` 推断。
    """

        # to support Azure OpenAI Service custom endpoints
    openai_api_base: str | None = Field(
        alias="base_url", default_factory=from_env("OPENAI_API_BASE", default=None)
    )
    """Base URL path for API requests, leave blank if not using a proxy or
    service emulator.

    解析顺序（首个匹配项生效）：

    1. 显式 `base_url`（或 `openai_api_base`）关键字参数。
    2. 环境变量 `OPENAI_API_BASE`（LangChain 在初始化时读取）。
    3. 环境变量 `OPENAI_BASE_URL`（由底层 `openai` SDK 客户端读取）。
    """

        # to support Azure OpenAI Service custom endpoints
    openai_api_type: str | None = Field(
        default_factory=from_env("OPENAI_API_TYPE", default=None)
    )

        # to support explicit proxy for OpenAI
    openai_proxy: str | None = Field(
        default_factory=from_env("OPENAI_PROXY", default=None)
    )

    embedding_ctx_length: int = 8191
    """一次嵌入的最大 token 数量。"""

    openai_api_key: (
        SecretStr | None | Callable[[], str] | Callable[[], Awaitable[str]]
    ) = Field(
        alias="api_key", default_factory=secret_from_env("OPENAI_API_KEY", default=None)
    )
    """用于 API 调用的 API 密钥。

    如果未提供，则自动从环境变量 `OPENAI_API_KEY` 推断。
    """

    openai_organization: str | None = Field(
        alias="organization",
        default_factory=from_env(
            ["OPENAI_ORG_ID", "OPENAI_ORGANIZATION"], default=None
        ),
    )
    """用于 API 调用的 OpenAI 组织 ID。

    如果未提供，则自动从环境变量 `OPENAI_ORG_ID` 推断。
    """

    allowed_special: Literal["all"] | set[str] | None = None

    disallowed_special: Literal["all"] | set[str] | Sequence[str] | None = None

    chunk_size: int = 1000
    """每次批处理嵌入的最大文本数量"""

    max_retries: int = 2
    """生成时的最大重试次数。"""

    request_timeout: float | tuple[float, float] | Any | None = Field(
        default=None, alias="timeout"
    )
    """对 OpenAI 补全 API 请求的超时时间。

    可以为 float、`httpx.Timeout` 或 `None`。
    """

    headers: Any = None

    tiktoken_enabled: bool = True
    """将其设为 False 以使用 HuggingFace `transformers` 的分词方式。

    For non-OpenAI providers (OpenRouter, Ollama, vLLM, etc.), consider setting
    `check_embedding_ctx_length=False` instead, as it bypasses tokenization
    entirely.
    """

    tiktoken_model_name: str | None = None
    """当使用此模块时传递给 tiktoken 的模型名称。

    tiktoken 用于计算文档中的 token 数量，以确保其在一定限制内。

    By default, when set to `None`, this will be the same as the embedding model
    name. However, there are some cases where you may want to use this
    `Embedding` class with a model name not supported by tiktoken. This can
    include when using Azure embeddings or when using one of the many model
    providers that expose an OpenAI-like API but with different models. In those
    cases, in order to avoid erroring when tiktoken is called, you can specify a
    model name to use here.
    """

    show_progress_bar: bool = False
    """嵌入时是否显示进度条。"""

    model_kwargs: dict[str, Any] = Field(default_factory=dict)
    """保存 `create` 调用中有效但未明确指定的任何模型参数。"""

    skip_empty: bool = False
    """嵌入时是否跳过空字符串或抛出错误。"""

    default_headers: Mapping[str, str] | None = None

    default_query: Mapping[str, object] | None = None

        # Configure a custom httpx client. See the
        # [httpx documentation](https://www.python-httpx.org/api/#client) for more details.

    retry_min_seconds: int = 4
    """重试之间最短等待秒数"""

    retry_max_seconds: int = 20
    """重试之间最长等待秒数"""

    http_client: Any | None = None
    """可选的 `httpx.Client`。

    Only used for sync invocations. Must specify `http_async_client` as well if
    you'd like a custom client for async invocations.
    """

    http_async_client: Any | None = None
    """可选的 `httpx.AsyncClient`。

    Only used for async invocations. Must specify `http_client` as well if you'd
    like a custom client for sync invocations.
    """

    check_embedding_ctx_length: bool = True
    """是否检查输入的 token 长度，并自动将超过 `embedding_ctx_length` 的输入拆分。

    Set to `False` to send raw text strings directly to the API instead of
    tokenizing. Useful for many non-OpenAI providers (e.g. OpenRouter, Ollama,
    vLLM).
    """

    model_config = ConfigDict(
        extra="forbid", populate_by_name=True, protected_namespaces=()
    )

    @model_validator(mode="before")
    @classmethod
    def build_extra(cls, values: dict[str, Any]) -> Any:
        """从传入的额外参数中构建额外的 kwargs。"""
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
                    Please confirm that {field_name} is what you intended."""
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

    @model_validator(mode="after")
    def validate_environment(self) -> Self:
        """验证 API 密钥和 Python 包是否存在于环境变量中。"""
        if self.openai_api_type in ("azure", "azure_ad", "azuread"):
            msg = (
                "If you are using Azure, please use the `AzureOpenAIEmbeddings` class."
            )
            raise ValueError(msg)

                # Resolve API key from SecretStr or Callable
        sync_api_key_value: str | Callable[[], str] | None = None
        async_api_key_value: str | Callable[[], Awaitable[str]] | None = None

        if self.openai_api_key is not None:
                        # Because OpenAI and AsyncOpenAI clients support either sync or async
                        # callables for the API key, we need to resolve separate values here.
            sync_api_key_value, async_api_key_value = _resolve_sync_and_async_api_keys(
                self.openai_api_key
            )

        client_params: dict = {
            "organization": self.openai_organization,
            "base_url": self.openai_api_base,
            "timeout": self.request_timeout,
            "max_retries": self.max_retries,
            "default_headers": self.default_headers,
            "default_query": self.default_query,
        }

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
        if not self.client:
            if sync_api_key_value is None:
                                # No valid sync API key, leave client as None and raise informative
                                # error on invocation.
                self.client = None
            else:
                if self.openai_proxy and not self.http_client:
                    try:
                        from langchain_openai._compat import httpx
                    except ImportError as e:
                        msg = (
                            "Could not import the httpx package used by the "
                            "OpenAI SDK. It is normally installed alongside "
                            "`openai`; reinstall with `pip install openai`."
                        )
                        raise ImportError(msg) from e
                    self.http_client = httpx.Client(proxy=self.openai_proxy)
                sync_specific = {
                    "http_client": self.http_client,
                    "api_key": sync_api_key_value,
                }
                self.client = openai.OpenAI(**client_params, **sync_specific).embeddings  # type: ignore[arg-type]
        if not self.async_client:
            if self.openai_proxy and not self.http_async_client:
                try:
                    from langchain_openai._compat import httpx
                except ImportError as e:
                    msg = (
                        "Could not import the httpx package used by the "
                        "OpenAI SDK. It is normally installed alongside "
                        "`openai`; reinstall with `pip install openai`."
                    )
                    raise ImportError(msg) from e
                self.http_async_client = httpx.AsyncClient(proxy=self.openai_proxy)
            async_specific = {
                "http_client": self.http_async_client,
                "api_key": async_api_key_value,
            }
            self.async_client = openai.AsyncOpenAI(
                **client_params,
                **async_specific,  # type: ignore[arg-type]
            ).embeddings
        return self

    @property
    def _invocation_params(self) -> dict[str, Any]:
        params: dict = {"model": self.model, **self.model_kwargs}
        if self.dimensions is not None:
            params["dimensions"] = self.dimensions
        return params

    def _ensure_sync_client_available(self) -> None:
        """检查同步客户端是否可用，如果不可用则抛出错误。"""
        if self.client is None:
            msg = (
                "Sync client is not available. This happens when an async callable "
                "was provided for the API key. Use async methods (ainvoke, astream) "
                "instead, or provide a string or sync callable for the API key."
            )
            raise ValueError(msg)

    def _tokenize(
        self, texts: list[str], chunk_size: int
    ) -> tuple[Iterable[int], list[list[int] | str], list[int], list[int]]:
        """对输入文本进行 token 化并分批处理。

        根据 `embedding_ctx_length` 拆分文本，并将它们按 `chunk_size` 大小分组为批次。

        Args:
            texts: 要 token 化的文本列表。
            chunk_size: 单个批次中包含的最大文本数量。

        Returns:
            A tuple containing:
                1. An iterable of starting indices in the token list for each batch.
                2. A list of tokenized texts (token arrays for tiktoken, strings for
                    HuggingFace).
                3. An iterable mapping each token array to the index of the original
                    text. Same length as the token list.
                4. A list of token counts for each tokenized text.
        """
        tokens: list[list[int] | str] = []
        indices: list[int] = []
        token_counts: list[int] = []
        model_name = self.tiktoken_model_name or self.model

                # If tiktoken flag set to False
        if not self.tiktoken_enabled:
            try:
                from transformers import AutoTokenizer
            except ImportError:
                msg = (
                    "Could not import transformers python package. "
                    "This is needed for OpenAIEmbeddings to work without "
                    "`tiktoken`. Please install it with `pip install transformers`. "
                )
                raise ValueError(msg)

            tokenizer = AutoTokenizer.from_pretrained(
                pretrained_model_name_or_path=model_name
            )
            for i, text in enumerate(texts):
                                # Tokenize the text using HuggingFace transformers
                tokenized: list[int] = tokenizer.encode(text, add_special_tokens=False)

                                # 将 token 按照嵌入上下文长度限制进行分块
                for j in range(0, len(tokenized), self.embedding_ctx_length):
                    token_chunk: list[int] = tokenized[
                        j : j + self.embedding_ctx_length
                    ]

                                        # 将 token ID 转换回字符串
                    chunk_text: str = tokenizer.decode(token_chunk)
                    tokens.append(chunk_text)
                    indices.append(i)
                    token_counts.append(len(token_chunk))
        else:
            try:
                encoding = tiktoken.encoding_for_model(model_name)
            except KeyError:
                encoding = tiktoken.get_encoding("cl100k_base")
            encoder_kwargs: dict[str, Any] = {
                k: v
                for k, v in {
                    "allowed_special": self.allowed_special,
                    "disallowed_special": self.disallowed_special,
                }.items()
                if v is not None
            }
            for i, text in enumerate(texts):
                if self.model.endswith("001"):
                                        # 参见：https://github.com/openai/openai-python/
                                        #      issues/418#issuecomment-1525939500
                                        # 替换换行符，这可能对性能产生负面影响。
                    text = text.replace("\n", " ")

                if encoder_kwargs:
                    token = encoding.encode(text, **encoder_kwargs)
                else:
                    token = encoding.encode_ordinary(text)

                                # 将 token 按照嵌入上下文长度限制进行分块
                for j in range(0, len(token), self.embedding_ctx_length):
                    tokens.append(token[j : j + self.embedding_ctx_length])
                    indices.append(i)
                    token_counts.append(len(token[j : j + self.embedding_ctx_length]))

        if self.show_progress_bar:
            try:
                from tqdm.auto import tqdm

                _iter: Iterable = tqdm(range(0, len(tokens), chunk_size))
            except ImportError:
                _iter = range(0, len(tokens), chunk_size)
        else:
            _iter = range(0, len(tokens), chunk_size)
        return _iter, tokens, indices, token_counts

        # 请参考
        # https://github.com/openai/openai-cookbook/blob/main/examples/Embedding_long_inputs.ipynb
    def _get_len_safe_embeddings(
        self,
        texts: list[str],
        *,
        engine: str,
        chunk_size: int | None = None,
        **kwargs: Any,
    ) -> list[list[float]]:
        """为文本列表生成长度安全的嵌入。

        此方法处理 token 化和嵌入生成，尊重 `embedding_ctx_length` 和
        `chunk_size`。根据 `tiktoken_enabled` 标志，支持 `tiktoken` 和 HuggingFace
        `transformers`。

        Args:
            texts: 要嵌入的文本列表。
            engine: 用于嵌入的引擎或模型。
            chunk_size: 处理嵌入的块大小。

        Returns:
            每个输入文本对应的嵌入列表。
        """
        _chunk_size = chunk_size or self.chunk_size
        client_kwargs = {**self._invocation_params, **kwargs}
        _iter, tokens, indices, token_counts = self._tokenize(texts, _chunk_size)
        batched_embeddings: list[list[float]] = []

                # 按批次处理，遵守 token 限制
        i = 0
        while i < len(tokens):
                        # 确定本批次可以包含多少个分块
            batch_token_count = 0
            batch_end = i

            for j in range(i, min(i + _chunk_size, len(tokens))):
                chunk_tokens = token_counts[j]
                                # 检查添加此分块是否会超出限制
                if batch_token_count + chunk_tokens > MAX_TOKENS_PER_REQUEST:
                    if batch_end == i:
                                                # 单个分块超过限制 - 仍予以处理
                        batch_end = j + 1
                    break
                batch_token_count += chunk_tokens
                batch_end = j + 1

                        # 使用此批次发起 API 调用
            batch_tokens = tokens[i:batch_end]
            response = self.client.create(input=batch_tokens, **client_kwargs)
            if not isinstance(response, dict):
                response = response.model_dump()
            batched_embeddings.extend(r["embedding"] for r in response["data"])

            i = batch_end

        embeddings = _process_batched_chunked_embeddings(
            len(texts), tokens, batched_embeddings, indices, self.skip_empty
        )
        _cached_empty_embedding: list[float] | None = None

        def empty_embedding() -> list[float]:
            nonlocal _cached_empty_embedding
            if _cached_empty_embedding is None:
                average_embedded = self.client.create(input="", **client_kwargs)
                if not isinstance(average_embedded, dict):
                    average_embedded = average_embedded.model_dump()
                _cached_empty_embedding = average_embedded["data"][0]["embedding"]
            return _cached_empty_embedding

        return [e if e is not None else empty_embedding() for e in embeddings]

        # 请参考
        # https://github.com/openai/openai-cookbook/blob/main/examples/Embedding_long_inputs.ipynb
    async def _aget_len_safe_embeddings(
        self,
        texts: list[str],
        *,
        engine: str,
        chunk_size: int | None = None,
        **kwargs: Any,
    ) -> list[list[float]]:
        """异步为文本列表生成长度安全的嵌入。

        该方法处理分词和嵌入生成，尊重 `embedding_ctx_length` 和 `chunk_size`。根据
        `tiktoken_enabled` 标志，支持 `tiktoken` 和 HuggingFace `transformers`。

        Args:
            texts: 要嵌入的文本列表。
            engine: 用于嵌入的引擎或模型。
            chunk_size: 处理嵌入时的块大小。

        Returns:
            每个输入文本对应的嵌入列表。
        """
        _chunk_size = chunk_size or self.chunk_size
        client_kwargs = {**self._invocation_params, **kwargs}
        _iter, tokens, indices, token_counts = await run_in_executor(
            None, self._tokenize, texts, _chunk_size
        )
        batched_embeddings: list[list[float]] = []

                # 按批次处理，遵守 token 限制
        i = 0
        while i < len(tokens):
                        # 确定本批次可以包含多少个分块
            batch_token_count = 0
            batch_end = i

            for j in range(i, min(i + _chunk_size, len(tokens))):
                chunk_tokens = token_counts[j]
                                # 检查添加此分块是否会超出限制
                if batch_token_count + chunk_tokens > MAX_TOKENS_PER_REQUEST:
                    if batch_end == i:
                                                # 单个分块超过限制 - 仍予以处理
                        batch_end = j + 1
                    break
                batch_token_count += chunk_tokens
                batch_end = j + 1

                        # 使用此批次发起 API 调用
            batch_tokens = tokens[i:batch_end]
            response = await self.async_client.create(
                input=batch_tokens, **client_kwargs
            )
            if not isinstance(response, dict):
                response = response.model_dump()
            batched_embeddings.extend(r["embedding"] for r in response["data"])

            i = batch_end

        embeddings = _process_batched_chunked_embeddings(
            len(texts), tokens, batched_embeddings, indices, self.skip_empty
        )
        _cached_empty_embedding: list[float] | None = None

        async def empty_embedding() -> list[float]:
            nonlocal _cached_empty_embedding
            if _cached_empty_embedding is None:
                average_embedded = await self.async_client.create(
                    input="", **client_kwargs
                )
                if not isinstance(average_embedded, dict):
                    average_embedded = average_embedded.model_dump()
                _cached_empty_embedding = average_embedded["data"][0]["embedding"]
            return _cached_empty_embedding

        return [e if e is not None else await empty_embedding() for e in embeddings]

    def embed_documents(
        self, texts: list[str], chunk_size: int | None = None, **kwargs: Any
    ) -> list[list[float]]:
        """调用 OpenAI 的嵌入端点以嵌入搜索文档。

        Args:
            texts: 要嵌入的文本列表。
            chunk_size: 嵌入的块大小。

                如果为 `None`，将使用类中指定的块大小。
            kwargs: 传递给嵌入 API 的额外关键字参数。

        Returns:
            每个文本对应的嵌入列表。
        """
        self._ensure_sync_client_available()
        chunk_size_ = chunk_size or self.chunk_size
        client_kwargs = {**self._invocation_params, **kwargs}
        if not self.check_embedding_ctx_length:
            embeddings: list[list[float]] = []
            for i in range(0, len(texts), chunk_size_):
                response = self.client.create(
                    input=texts[i : i + chunk_size_], **client_kwargs
                )
                if not isinstance(response, dict):
                    response = response.model_dump()
                embeddings.extend(r["embedding"] for r in response["data"])
            return embeddings

                # 无条件调用 _get_len_safe_embeddings 以处理长度安全问题。
                # 当所有文本都足够短时，可优化以避免重复工作。
        engine = cast(str, self.deployment)
        return self._get_len_safe_embeddings(
            texts, engine=engine, chunk_size=chunk_size, **kwargs
        )

    async def aembed_documents(
        self, texts: list[str], chunk_size: int | None = None, **kwargs: Any
    ) -> list[list[float]]:
        """异步调用 OpenAI 的嵌入端点以嵌入搜索文档。

        Args:
            texts: 要嵌入的文本列表。
            chunk_size: 嵌入的块大小。

                如果为 `None`，将使用类中指定的块大小。
            kwargs: 传递给嵌入 API 的额外关键字参数。

        Returns:
            每个文本对应的嵌入列表。
        """
        chunk_size_ = chunk_size or self.chunk_size
        client_kwargs = {**self._invocation_params, **kwargs}
        if not self.check_embedding_ctx_length:
            embeddings: list[list[float]] = []
            for i in range(0, len(texts), chunk_size_):
                response = await self.async_client.create(
                    input=texts[i : i + chunk_size_], **client_kwargs
                )
                if not isinstance(response, dict):
                    response = response.model_dump()
                embeddings.extend(r["embedding"] for r in response["data"])
            return embeddings

                # 无条件调用 _get_len_safe_embeddings 以处理长度安全问题。
                # 当所有文本都足够短时，可优化以避免重复工作。
        engine = cast(str, self.deployment)
        return await self._aget_len_safe_embeddings(
            texts, engine=engine, chunk_size=chunk_size, **kwargs
        )

    def embed_query(self, text: str, **kwargs: Any) -> list[float]:
        """调用 OpenAI 的嵌入端点以嵌入查询文本。

        Args:
            text: 要嵌入的文本。
            kwargs: 传递给嵌入 API 的额外关键字参数。

        Returns:
            文本的嵌入结果。
        """
        self._ensure_sync_client_available()
        return self.embed_documents([text], **kwargs)[0]

    async def aembed_query(self, text: str, **kwargs: Any) -> list[float]:
        """异步调用 OpenAI 的嵌入端点以嵌入查询文本。

        Args:
            text: 要嵌入的文本。
            kwargs: 传递给嵌入 API 的额外关键字参数。

        Returns:
            文本的嵌入结果。
        """
        embeddings = await self.aembed_documents([text], **kwargs)
        return embeddings[0]

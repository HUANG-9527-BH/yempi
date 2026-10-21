import asyncio
import logging
import warnings
from collections.abc import Callable, Iterable

import httpx
from httpx import Response
from langchain_core.embeddings import Embeddings
from langchain_core.utils import (
    secret_from_env,
)
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SecretStr,
    model_validator,
)
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_fixed
from tokenizers import Tokenizer  # type: ignore[import]
from typing_extensions import Self

logger = logging.getLogger(__name__)

MAX_TOKENS = 16_000
"""A batching parameter for the Mistral API. This is NOT the maximum number of tokens
accepted by the embedding model for each document/chunk, but rather the maximum number
of tokens that can be sent in a single request to the Mistral API (across multiple
documents/chunks)"""


def _is_retryable_error(exception: BaseException) -> bool:
    """确定异常是否应触发重试。

    仅在以下情况下重试：
    - 超时异常
    - 429（速率限制）错误
    - 5xx（服务器）错误

    不对 400（错误请求）或其他 4xx 客户端错误进行重试。
    """
    if isinstance(exception, httpx.TimeoutException):
        return True
    if isinstance(exception, httpx.HTTPStatusError):
        status_code = exception.response.status_code
                # 重试速率限制（429）或服务器错误（5xx）
        return status_code == 429 or status_code >= 500
    return False


class DummyTokenizer:
    """当无法访问分词器时的占位分词器（例如通过 Huggingface）。"""

    @staticmethod
    def encode_batch(texts: list[str]) -> list[list[str]]:
        return [list(text) for text in texts]


class MistralAIEmbeddings(BaseModel, Embeddings):
    """MistralAI 嵌入模型集成。

    设置：
        安装 `langchain_mistralai` 并设置环境变量
        `MISTRAL_API_KEY`。

        ```bash
        pip install -U langchain_mistralai
        export MISTRAL_API_KEY="your-api-key"
        ```

    关键初始化参数 —— 完成参数：
        model:
            要使用的 `MistralAI` 模型名称。

    Key init args — client params:
        api_key:
            The API key for the MistralAI API. If not provided, it will be read from the
            environment variable `MISTRAL_API_KEY`.
        max_concurrent_requests: int
        max_retries:
            The number of times to retry a request if it fails.
        timeout:
            The number of seconds to wait for a response before timing out.
        wait_time:
            The number of seconds to wait before retrying a request in case of 429
            error.
        max_concurrent_requests:
            The maximum number of concurrent requests to make to the Mistral API.

    有关支持的初始化参数及其描述的完整列表，请参见参数部分。

    实例化：

        ```python
        from __module_name__ import MistralAIEmbeddings

        embed = MistralAIEmbeddings(
            model="mistral-embed",
            # api_key="...",
            # other params...
        )
        ```

    嵌入单个文本：

        ```python
        input_text = "The meaning of life is 42"
        vector = embed.embed_query(input_text)
        print(vector[:3])
        ```
        ```python
        [-0.024603435769677162, -0.007543657906353474, 0.0039630369283258915]
        ```

    嵌入多个文本：

        ```python
        input_texts = ["Document 1...", "Document 2..."]
        vectors = embed.embed_documents(input_texts)
        print(len(vectors))
        # The first 3 coordinates for the first vector
        print(vectors[0][:3])
        ```
        ```python
        2
        [-0.024603435769677162, -0.007543657906353474, 0.0039630369283258915]
        ```

    异步：

        ```python
        vector = await embed.aembed_query(input_text)
        print(vector[:3])

        # multiple:
        # await embed.aembed_documents(input_texts)
        ```
        ```python
        [-0.009100092574954033, 0.005071679595857859, -0.0029193938244134188]
        ```
    """

        # 客户端和 async_client 的类型被忽略，因为模型初始化后类型不再是
    # Optional，且模型校验器已运行。
        # 模型初始化后类型不再是 Optional，且模型校验器已运行。
        # TODO: 处理重试
    client: httpx.Client = Field(default=None)  # type: ignore[assignment]

    async_client: httpx.AsyncClient = Field(  # type: ignore[assignment]
        default=None
    )

    mistral_api_key: SecretStr = Field(
        alias="api_key",
        default_factory=secret_from_env("MISTRAL_API_KEY", default=""),
    )

    endpoint: str = "https://api.mistral.ai/v1/"

    max_retries: int | None = 5

    timeout: int = 120

    wait_time: int | None = 30

    max_concurrent_requests: int = 64

    tokenizer: Tokenizer = Field(default=None)

    model: str = "mistral-embed"

    model_config = ConfigDict(
        extra="forbid",
        arbitrary_types_allowed=True,
        populate_by_name=True,
    )

    @model_validator(mode="after")
    def validate_environment(self) -> Self:
        """验证配置。"""
        api_key_str = self.mistral_api_key.get_secret_value()
                # TODO: 处理重试和 max_concurrency
        if not self.client:
            self.client = httpx.Client(
                base_url=self.endpoint,
                headers={
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                    "Authorization": f"Bearer {api_key_str}",
                },
                timeout=self.timeout,
            )
                # huggingface_hub GatedRepoError
        if not self.async_client:
            self.async_client = httpx.AsyncClient(
                base_url=self.endpoint,
                headers={
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                    "Authorization": f"Bearer {api_key_str}",
                },
                timeout=self.timeout,
            )
        if self.tokenizer is None:
            try:
                self.tokenizer = Tokenizer.from_pretrained(
                    "mistralai/Mixtral-8x7B-v0.1"
                )
            except OSError:  # 首个批次超出最大 token 数的边缘情况
                warnings.warn(
                    "Could not download mistral tokenizer from Huggingface for "
                    "calculating batch sizes. Set a Huggingface token via the "
                    "HF_TOKEN environment variable to download the real tokenizer. "
                    "Falling back to a dummy tokenizer that uses `len()`.",
                    stacklevel=2,
                )
                self.tokenizer = DummyTokenizer()
        return self

    def _get_batches(self, texts: list[str]) -> Iterable[list[str]]:
        """将文本列表分割为小于 16k token 的批次，以供 Mistral API 使用。"""
        batch: list[str] = []
        batch_tokens = 0

        text_token_lengths = [
            len(encoded) for encoded in self.tokenizer.encode_batch(texts)
        ]

        for text, text_tokens in zip(texts, text_token_lengths, strict=False):
            if batch_tokens + text_tokens > MAX_TOKENS:
                if len(batch) > 0:
                                        # 不应生成空批次。
                                        # 不应生成空批处理。
                    yield batch
                batch = [text]
                batch_tokens = text_tokens
            else:
                batch.append(text)
                batch_tokens += text_tokens
        if batch:
            yield batch

    def _retry(self, func: Callable) -> Callable:
        if self.max_retries is None or self.wait_time is None:
            return func

        return retry(
            retry=retry_if_exception(_is_retryable_error),
            wait=wait_fixed(self.wait_time),
            stop=stop_after_attempt(self.max_retries),
        )(func)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """嵌入一组文档文本。

        Args:
            texts: 要嵌入的文本列表。

        Returns:
            每个文本对应的嵌入列表。

        """
        try:
            batch_responses = []

            @self._retry
            def _embed_batch(batch: list[str]) -> Response:
                response = self.client.post(
                    url="/embeddings",
                    json={
                        "model": self.model,
                        "input": batch,
                    },
                )
                response.raise_for_status()
                return response

            batch_responses = [
                _embed_batch(batch) for batch in self._get_batches(texts)
            ]
            return [
                list(map(float, embedding_obj["embedding"]))
                for response in batch_responses
                for embedding_obj in response.json()["data"]
            ]
        except Exception:
            logger.exception("An error occurred with MistralAI")
            raise

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        """嵌入一组文档文本。

        Args:
            texts: 要嵌入的文本列表。

        Returns:
            每个文本对应的嵌入列表。
        """
        try:

            @self._retry
            async def _aembed_batch(batch: list[str]) -> Response:
                response = await self.async_client.post(
                    url="/embeddings",
                    json={
                        "model": self.model,
                        "input": batch,
                    },
                )
                response.raise_for_status()
                return response

            batch_responses = await asyncio.gather(
                *[_aembed_batch(batch) for batch in self._get_batches(texts)]
            )
            return [
                list(map(float, embedding_obj["embedding"]))
                for response in batch_responses
                for embedding_obj in response.json()["data"]
            ]
        except Exception:
            logger.exception("An error occurred with MistralAI")
            raise

    def embed_query(self, text: str) -> list[float]:
        """嵌入单个查询文本。

        Args:
            text: 要嵌入的文本。

        Returns:
            文本的嵌入结果。

        """
        return self.embed_documents([text])[0]

    async def aembed_query(self, text: str) -> list[float]:
        """嵌入单个查询文本。

        Args:
            text: 要嵌入的文本。

        Returns:
            文本的嵌入结果。

        """
        return (await self.aembed_documents([text]))[0]

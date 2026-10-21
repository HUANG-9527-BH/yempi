"""Perplexity 嵌入 API 的封装。"""

from __future__ import annotations

import base64
import struct
from typing import Any

from langchain_core.embeddings import Embeddings
from langchain_core.utils import secret_from_env
from perplexity import AsyncPerplexity, Perplexity
from pydantic import BaseModel, ConfigDict, Field, SecretStr, model_validator
from typing_extensions import Self


def _decode_int8_embedding(b64: str) -> list[float]:
    """将 `base64_int8` 编码的 Perplexity 嵌入解码为浮点数列表。"""
    raw = base64.b64decode(b64)
    return [float(v) for v in struct.unpack(f"<{len(raw)}b", raw)]


class PerplexityEmbeddings(BaseModel, Embeddings):
    """`Perplexity AI` 嵌入。

    设置：
        安装 `perplexityai` 包，并设置 `PPLX_API_KEY`
        （或 `PERPLEXITY_API_KEY`）环境变量，或通过
        `pplx_api_key`/`api_key` 参数传入密钥。

        ```bash
        pip install -U langchain-perplexity
        export PPLX_API_KEY=your_api_key
        ```

        参见 Perplexity 嵌入 API 参考文档：
        https://docs.perplexity.ai/api-reference/embeddings-post

        实例化：

        ```python
        from langchain_perplexity import PerplexityEmbeddings

        embeddings = PerplexityEmbeddings()
        ```

        嵌入单个查询：

        ```python
        query_vector = embeddings.embed_query("hello world")
        ```

        嵌入文档：

        ```python
        doc_vectors = embeddings.embed_documents(["hello", "world"])
        ```

        选择特定模型：

        ```python
        embeddings = PerplexityEmbeddings(model="pplx-embed-v1-0.6b")
        ```

    !!! note
        Perplexity returns base64-encoded signed int8 embeddings. This class
        decodes them into `list[float]` values in the range [-128, 127]. The
        magnitude is preserved from the API's quantized output; cosine
        similarity is unaffected by the lack of unit-length normalization.
    """

    client: Any = Field(default=None, exclude=True)
    """Perplexity SDK 客户端（自动设置）。"""

    async_client: Any = Field(default=None, exclude=True)
    """异步 Perplexity SDK 客户端（自动设置）。"""

    model: str = "pplx-embed-v1-4b"
    """要使用的 Perplexity 嵌入模型名称。

    See the API reference for available identifiers, including
    `pplx-embed-v1-0.6b` and `pplx-embed-v1-4b`. Contextualized variants are
    served through a separate endpoint and are not exposed by this class.
    """

    pplx_api_key: SecretStr | None = Field(
        default_factory=secret_from_env(
            ["PPLX_API_KEY", "PERPLEXITY_API_KEY"], default=None
        ),
        alias="api_key",
    )
    """Perplexity API 密钥。从 `PPLX_API_KEY` 或 `PERPLEXITY_API_KEY` 读取。"""

    request_timeout: float | tuple[float, float] | None = Field(None, alias="timeout")
    """对 Perplexity 嵌入 API 请求的超时时间。"""

    max_retries: int = 6
    """调用嵌入 API 时的最大重试次数。"""

    model_config = ConfigDict(populate_by_name=True, arbitrary_types_allowed=True)

    @property
    def lc_secrets(self) -> dict[str, str]:
        """将密钥字段名映射到其环境变量名。"""
        return {"pplx_api_key": "PPLX_API_KEY"}

    @model_validator(mode="after")
    def validate_environment(self) -> Self:
        """初始化 Perplexity SDK 客户端。"""
        if not self.pplx_api_key:
            msg = (
                "Perplexity API key not provided. Pass `pplx_api_key` (or "
                "`api_key`) to PerplexityEmbeddings, or set the `PPLX_API_KEY` "
                "or `PERPLEXITY_API_KEY` environment variable."
            )
            raise ValueError(msg)

        api_key = self.pplx_api_key.get_secret_value()
        client_params: dict[str, Any] = {
            "api_key": api_key,
            "max_retries": self.max_retries,
        }
        if self.request_timeout is not None:
            client_params["timeout"] = self.request_timeout

        if self.client is None:
            self.client = Perplexity(**client_params)
        if self.async_client is None:
            self.async_client = AsyncPerplexity(**client_params)
        return self

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """使用 Perplexity 嵌入 API 嵌入文档列表。

        Args:
            texts: 要嵌入的文本列表。

        Returns:
            每个输入文本对应一个嵌入向量的列表。当 `texts` 为空时返回空列表。
        """
        if not texts:
            return []
        response = self.client.embeddings.create(model=self.model, input=texts)
        return [_decode_int8_embedding(item.embedding) for item in response.data]

    def embed_query(self, text: str) -> list[float]:
        """使用 Perplexity 嵌入 API 嵌入单个查询字符串。

        Args:
            text: 要嵌入的文本。

        Returns:
            输入文本的嵌入向量。
        """
        return self.embed_documents([text])[0]

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        """异步嵌入文档列表。

        Args:
            texts: 要嵌入的文本列表。

        Returns:
            每个输入文本对应一个嵌入向量的列表。当 `texts` 为空时返回空列表。
        """
        if not texts:
            return []
        response = await self.async_client.embeddings.create(
            model=self.model, input=texts
        )
        return [_decode_int8_embedding(item.embedding) for item in response.data]

    async def aembed_query(self, text: str) -> list[float]:
        """异步嵌入单个查询字符串。

        Args:
            text: 要嵌入的文本。

        Returns:
            输入文本的嵌入向量。
        """
        result = await self.aembed_documents([text])
        return result[0]

from __future__ import annotations

import os
from typing import Any

from langchain_core.embeddings import Embeddings
from langchain_core.utils import from_env
from pydantic import BaseModel, ConfigDict, Field, model_validator
from typing_extensions import Self

DEFAULT_MODEL = "sentence-transformers/all-mpnet-base-v2"
VALID_TASKS = ("feature-extraction",)


class HuggingFaceEndpointEmbeddings(BaseModel, Embeddings):
    """HuggingFaceHub 嵌入模型。

    To use, you should have the `huggingface_hub` python package installed, and the
    environment variable `HUGGINGFACEHUB_API_TOKEN` set with your API token, or pass
    it as a named parameter to the constructor.

    Example:
        ```python
        from langchain_huggingface import HuggingFaceEndpointEmbeddings

        model = "sentence-transformers/all-mpnet-base-v2"
        hf = HuggingFaceEndpointEmbeddings(
            model=model,
            task="feature-extraction",
            huggingfacehub_api_token="my-api-key",
        )
        ```
    """

    client: Any = None

    async_client: Any = None

    model: str | None = None
    """要使用的模型名称。"""

    provider: str | None = None
    """Name of the provider to use for inference with the model specified in
        `repo_id`. e.g. "sambanova". if not specified, defaults to HF Inference API.
        available providers can be found in the [huggingface_hub documentation](https://huggingface.co/docs/huggingface_hub/guides/inference#supported-providers-and-tasks)."""

    repo_id: str | None = None
    """Huggingfacehub 仓库 ID，用于向后兼容。"""

    task: str | None = "feature-extraction"
    """要调用模型的任务。"""

    model_kwargs: dict | None = None
    """传递给模型的关键字参数。"""

    huggingfacehub_api_token: str | None = Field(
        default_factory=from_env("HUGGINGFACEHUB_API_TOKEN", default=None)
    )

    model_config = ConfigDict(
        extra="forbid",
        protected_namespaces=(),
    )

    @model_validator(mode="after")
    def validate_environment(self) -> Self:
        """验证环境中的 API 密钥和 Python 包是否存在。"""
        for field_name in ("model", "repo_id"):
            value = getattr(self, field_name)
            if value and value.startswith(("http://", "https://")):
                msg = f"`{field_name}` must be a HuggingFace repo ID, not a URL."
                raise ValueError(msg)

        huggingfacehub_api_token = self.huggingfacehub_api_token or os.getenv(
            "HF_TOKEN"
        )

        try:
            from huggingface_hub import (  # type: ignore[import]
                AsyncInferenceClient,
                InferenceClient,
            )

            if self.model:
                self.repo_id = self.model
            elif self.repo_id:
                self.model = self.repo_id
            else:
                self.model = DEFAULT_MODEL
                self.repo_id = DEFAULT_MODEL

            client = InferenceClient(
                model=self.model,
                token=huggingfacehub_api_token,
                provider=self.provider,  # type: ignore[arg-type]
            )

            async_client = AsyncInferenceClient(
                model=self.model,
                token=huggingfacehub_api_token,
                provider=self.provider,  # type: ignore[arg-type]
            )

            if self.task not in VALID_TASKS:
                msg = (
                    f"Got invalid task {self.task}, "
                    f"currently only {VALID_TASKS} are supported"
                )
                raise ValueError(msg)
            self.client = client
            self.async_client = async_client

        except ImportError as e:
            msg = (
                "Could not import huggingface_hub python package. "
                "Please install it with `pip install huggingface_hub`."
            )
            raise ImportError(msg) from e
        return self

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """调用 HuggingFaceHub 的嵌入端点以嵌入搜索文档。

        Args:
            texts: 要嵌入的文本列表。

        Returns:
            每个文本对应的嵌入列表。

        """
                # 替换换行符，因为它们可能对性能产生负面影响。
        texts = [text.replace("\n", " ") for text in texts]
        _model_kwargs = self.model_kwargs or {}
                #  api 文档:
        #  https://huggingface.github.io/text-embeddings-inference/#/Text%20Embeddings%20Inference/embed
        responses = self.client.feature_extraction(text=texts, **_model_kwargs)
        return responses.tolist()

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        """异步调用 HuggingFaceHub 的嵌入端点以嵌入搜索文档。

        Args:
            texts: 要嵌入的文本列表。

        Returns:
            每个文本对应的嵌入列表。

        """
                # 替换换行符，因为它们可能对性能产生负面影响。
        texts = [text.replace("\n", " ") for text in texts]
        _model_kwargs = self.model_kwargs or {}
        responses = await self.async_client.feature_extraction(
            text=texts, **_model_kwargs
        )
        return responses.tolist()

    def embed_query(self, text: str) -> list[float]:
        """调用 HuggingFaceHub 的嵌入端点以嵌入查询文本。

        Args:
            text: 要嵌入的文本。

        Returns:
            该文本的嵌入。

        """
        return self.embed_documents([text])[0]

    async def aembed_query(self, text: str) -> list[float]:
        """异步调用 HuggingFaceHub 的嵌入端点以嵌入查询文本。

        Args:
            text: 要嵌入的文本。

        Returns:
            该文本的嵌入。

        """
        return (await self.aembed_documents([text]))[0]

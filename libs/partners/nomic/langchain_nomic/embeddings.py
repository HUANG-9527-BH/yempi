"""LangChain 的 Nomic 合作集成。"""

from __future__ import annotations

import os
from typing import Literal, overload

import nomic  # type: ignore[import]
from langchain_core.embeddings import Embeddings
from nomic import embed


class NomicEmbeddings(Embeddings):
    """`NomicEmbeddings` 嵌入模型。

    Example:
        ```python
        from langchain_nomic import NomicEmbeddings

        model = NomicEmbeddings()
        ```
    """

    @overload
    def __init__(
        self,
        *,
        model: str,
        nomic_api_key: str | None = ...,
        dimensionality: int | None = ...,
        inference_mode: Literal["remote"] = ...,
    ) -> None: ...

    @overload
    def __init__(
        self,
        *,
        model: str,
        nomic_api_key: str | None = ...,
        dimensionality: int | None = ...,
        inference_mode: Literal["local", "dynamic"],
        device: str | None = ...,
    ) -> None: ...

    @overload
    def __init__(
        self,
        *,
        model: str,
        nomic_api_key: str | None = ...,
        dimensionality: int | None = ...,
        inference_mode: str,
        device: str | None = ...,
    ) -> None: ...

    def __init__(
        self,
        *,
        model: str,
        nomic_api_key: str | None = None,
        dimensionality: int | None = None,
        inference_mode: str = "remote",
        device: str | None = None,
        vision_model: str | None = None,
    ):
        """初始化 `NomicEmbeddings` 模型。

        Args:
            model: Model name
            nomic_api_key: Optionally, set the Nomic API key. Uses the `NOMIC_API_KEY`
                environment variable by default.
            dimensionality: The embedding dimension, for use with Matryoshka-capable
                models. Defaults to full-size.
            inference_mode: How to generate embeddings. One of `'remote'`, `'local'`
                (Embed4All), or `'dynamic'` (automatic).
            device: The device to use for local embeddings. Choices include
                `'cpu'`, `'gpu'`, `'nvidia'`, `'amd'`, or a specific device
                name. See the docstring for `GPT4All.__init__` for more info.

                通常默认为 `'cpu'`。

                !!! warning

                    不要在 macOS 上使用。
            vision_model: 用于图像嵌入的视觉模型。

        """
        _api_key = nomic_api_key or os.environ.get("NOMIC_API_KEY")
        if _api_key:
            nomic.login(_api_key)
        self.model = model
        self.dimensionality = dimensionality
        self.inference_mode = inference_mode
        self.device = device
        self.vision_model = vision_model

    def embed(self, texts: list[str], *, task_type: str) -> list[list[float]]:
        """嵌入文本。

        Args:
            texts: 要嵌入的文本列表
            task_type: 嵌入时使用的任务类型。可选 `'search_query'`、
                `'search_document'`、`'classification'`、`'clustering'`

        """
        output = embed.text(
            texts=texts,
            model=self.model,
            task_type=task_type,
            dimensionality=self.dimensionality,
            inference_mode=self.inference_mode,
            device=self.device,
        )
        return output["embeddings"]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """嵌入搜索文档。

        Args:
            texts: 要作为文档嵌入的文本列表

        """
        return self.embed(
            texts=texts,
            task_type="search_document",
        )

    def embed_query(self, text: str) -> list[float]:
        """嵌入查询文本。

        Args:
            text: 查询文本

        """
        return self.embed(
            texts=[text],
            task_type="search_query",
        )[0]

    def embed_image(self, uris: list[str]) -> list[list[float]]:
        """嵌入图像。

        Args:
            uris: 要嵌入的图像 URI 列表
        """
        return embed.image(
            images=uris,
            model=self.vision_model,
        )["embeddings"]

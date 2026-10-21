from __future__ import annotations

import warnings
from typing import Any

from langchain_core.embeddings import Embeddings
from pydantic import BaseModel, ConfigDict, Field


class HuggingFaceEmbeddings(BaseModel, Embeddings):
    """HuggingFace sentence_transformers 嵌入模型。

    要使用，您应安装 `sentence_transformers` Python 包。

    Example:
        ```python
        from langchain_huggingface import HuggingFaceEmbeddings

        model_name = "sentence-transformers/all-mpnet-base-v2"
        model_kwargs = {"device": "cpu"}
        encode_kwargs = {"normalize_embeddings": False}
        hf = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs,
        )
        ```
    """

    model_name: str = Field(
        default="sentence-transformers/all-mpnet-base-v2", alias="model"
    )
    """要使用的模型名称。"""
    cache_folder: str | None = None
    """存储模型的路径。
    也可通过 SENTENCE_TRANSFORMERS_HOME 环境变量设置。"""
    model_kwargs: dict[str, Any] = Field(default_factory=dict)
    """传递给 Sentence Transformer 模型的关键字参数，例如 `device`、
    `prompts`、`default_prompt_name`、`revision`、`trust_remote_code` 或 `token`。
    See also the Sentence Transformer documentation: 参见 Sentence Transformer 文档：https://sbert.net/docs/package_reference/SentenceTransformer.html#sentence_transformers.SentenceTransformer"""
    encode_kwargs: dict[str, Any] = Field(default_factory=dict)
    """Keyword arguments to pass when calling the `encode` method for the documents of
    the Sentence Transformer model, such as `prompt_name`, `prompt`, `batch_size`,
    `precision`, `normalize_embeddings`, and more.
    See also the Sentence Transformer documentation: https://sbert.net/docs/package_reference/SentenceTransformer.html#sentence_transformers.SentenceTransformer.encode"""
    query_encode_kwargs: dict[str, Any] = Field(default_factory=dict)
    """Keyword arguments to pass when calling the `encode` method for the query of
    the Sentence Transformer model, such as `prompt_name`, `prompt`, `batch_size`,
    `precision`, `normalize_embeddings`, and more.
    See also the Sentence Transformer documentation: https://sbert.net/docs/package_reference/SentenceTransformer.html#sentence_transformers.SentenceTransformer.encode"""
    multi_process: bool = False
    """在多个 GPU 上运行 encode()。"""
    show_progress: bool = False
    """是否显示进度条。"""

    def __init__(self, **kwargs: Any):
        """初始化 sentence_transformer。"""
        super().__init__(**kwargs)
        try:
            import sentence_transformers  # type: ignore[import]
        except ImportError as exc:
            msg = (
                "Could not import sentence_transformers python package. "
                "Please install it with `pip install sentence-transformers`."
            )
            raise ImportError(msg) from exc

        if self.model_kwargs.get("backend", "torch") == "ipex":
            msg = (
                "`backend='ipex'` is no longer supported; the default `torch` "
                "backend will be used instead. Intel GPU acceleration is "
                "natively supported in PyTorch 2.5 and later."
            )
            warnings.warn(msg, UserWarning, stacklevel=2)
            self.model_kwargs["backend"] = "torch"

        model_cls = sentence_transformers.SentenceTransformer

        self._client = model_cls(
            self.model_name, cache_folder=self.cache_folder, **self.model_kwargs
        )

    model_config = ConfigDict(
        extra="forbid",
        protected_namespaces=(),
        populate_by_name=True,
    )

    def _embed(
        self, texts: list[str], encode_kwargs: dict[str, Any]
    ) -> list[list[float]]:
        """使用 HuggingFace 变换器模型嵌入文本。

        Args:
            texts: 要嵌入的文本列表。
            encode_kwargs: 调用
                `encode` 方法时传递给 SentenceTransformer
                encode 方法的关键字参数。

        Returns:
            嵌入列表，每个文本对应一个嵌入。

        """
        import sentence_transformers  # type: ignore[import]

        texts = [x.replace("\n", " ") for x in texts]
        if self.multi_process:
            pool = self._client.start_multi_process_pool()
            embeddings = self._client.encode_multi_process(texts, pool)
            sentence_transformers.SentenceTransformer.stop_multi_process_pool(pool)
        else:
            embeddings = self._client.encode(
                texts,
                show_progress_bar=self.show_progress,
                **encode_kwargs,
            )

        if isinstance(embeddings, list):
            msg = (
                "Expected embeddings to be a Tensor or a numpy array, "
                "got a list instead."
            )
            raise TypeError(msg)

        return embeddings.tolist()  # type: ignore[return-type]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """使用 HuggingFace 变换器模型计算文档嵌入。

        Args:
            texts: 要嵌入的文本列表。

        Returns:
            嵌入列表，每个文本对应一个嵌入。

        """
        return self._embed(texts, self.encode_kwargs)

    def embed_query(self, text: str) -> list[float]:
        """使用 HuggingFace 变换器模型计算查询嵌入。

        Args:
            text: 要嵌入的文本。

        Returns:
            该文本的嵌入。

        """
        embed_kwargs = (
            self.query_encode_kwargs
            if len(self.query_encode_kwargs) > 0
            else self.encode_kwargs
        )
        return self._embed([text], embed_kwargs)[0]

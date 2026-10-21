from __future__ import annotations

from typing import TYPE_CHECKING, Any

from langchain_qdrant.sparse_embeddings import SparseEmbeddings, SparseVector

if TYPE_CHECKING:
    from collections.abc import Sequence


class FastEmbedSparse(SparseEmbeddings):
    """用于与 Qdrant 配合使用的稀疏嵌入模型的接口。"""

    def __init__(
        self,
        model_name: str = "Qdrant/bm25",
        batch_size: int = 256,
        cache_dir: str | None = None,
        threads: int | None = None,
        providers: Sequence[Any] | None = None,
        parallel: int | None = None,
        **kwargs: Any,
    ) -> None:
        """使用 FastEmbed 实现的稀疏编码器。

        使用 [FastEmbed](https://qdrant.github.io/fastembed/) 进行稀疏文本
        嵌入。
        可用模型列表请参见 [Qdrant 文档](https://qdrant.github.io/fastembed/examples/Supported_Models/)。

        Args:
            model_name (str): 要使用的模型名称。
            batch_size (int): 编码时的批处理大小。
            cache_dir (str, optional): 模型缓存目录路径。\
                也可通过设置\
                `FASTEMBED_CACHE_PATH` 环境变量来指定。
            threads (int, optional): onnxruntime 会话可使用的线程数。
            providers (Sequence[Any], optional): ONNX 执行提供者列表。\
            parallel (int, optional): 若值 `>1`，将使用数据并行编码，r\
                推荐用于大规模数据集的编码。\
                若为 `0`，则使用所有可用核心。\
                若为 `None`，则不使用数据并行处理，\
                改用 onnxruntime 默认线程设置。

            kwargs: 传递给 `fastembed.SparseTextEmbedding` 的额外选项。

        Raises:
            ValueError: 如果 `model_name` 在 `SparseTextEmbedding` 中不受支持。
        """
        try:
            from fastembed import (  # type: ignore[import-not-found] # noqa: PLC0415
                SparseTextEmbedding,
            )
        except ImportError as err:
            msg = (
                "The 'fastembed' package is not installed. "
                "Please install it with "
                "`pip install fastembed` or `pip install fastembed-gpu`."
            )
            raise ValueError(msg) from err
        self._batch_size = batch_size
        self._parallel = parallel
        self._model = SparseTextEmbedding(
            model_name=model_name,
            cache_dir=cache_dir,
            threads=threads,
            providers=providers,
            **kwargs,
        )

    def embed_documents(self, texts: list[str]) -> list[SparseVector]:
        results = self._model.embed(
            texts, batch_size=self._batch_size, parallel=self._parallel
        )
        return [
            SparseVector(indices=result.indices.tolist(), values=result.values.tolist())
            for result in results
        ]

    def embed_query(self, text: str) -> SparseVector:
        result = next(self._model.query_embed(text))

        return SparseVector(
            indices=result.indices.tolist(), values=result.values.tolist()
        )

"""内存中 `VectorStore` 实现的内部工具。

!!! warning

    这些属于私有 API 的一部分，用户不应直接使用，因为它们可能随时更改。
"""

from __future__ import annotations

import logging
import warnings
from typing import TYPE_CHECKING

try:
    import numpy as np

    _HAS_NUMPY = True
except ImportError:
    _HAS_NUMPY = False

try:
    import simsimd as simd  # type: ignore[import-not-found]

    _HAS_SIMSIMD = True
except ImportError:
    _HAS_SIMSIMD = False

if TYPE_CHECKING:
    import numpy.typing as npt

    Matrix = (
        list[list[float]] | list[npt.NDArray[np.floating]] | npt.NDArray[np.floating]
    )

logger = logging.getLogger(__name__)


def _cosine_similarity(x: Matrix, y: Matrix) -> npt.NDArray[np.floating]:
    """两个等宽矩阵之间的行向余弦相似度。

    Args:
        x: 一个形状为 `(n, m)` 的矩阵。
        y: 一个形状为 `(k, m)` 的矩阵。

    Returns:
        一个形状为 `(n, k)` 的矩阵，其中每个元素 `(i, j)` 是 `x` 的第 `i` 行与 `y` 的第
        `j` 行之间的余弦相似度。

    Raises:
        ValueError: 如果 `x` 和 `y` 的列数不相同。
        ImportError: 如果未安装 numpy。
    """
    if not _HAS_NUMPY:
        msg = (
            "cosine_similarity requires numpy to be installed. "
            "Please install numpy with `pip install numpy`."
        )
        raise ImportError(msg)

    if len(x) == 0 or len(y) == 0:
        return np.array([[]])

    x = np.array(x)
    y = np.array(y)

        # 检查 NaN
    if np.any(np.isnan(x)) or np.any(np.isnan(y)):
        warnings.warn(
            "NaN found in input arrays, unexpected return might follow",
            category=RuntimeWarning,
            stacklevel=2,
        )

        # 检查 Inf
    if np.any(np.isinf(x)) or np.any(np.isinf(y)):
        warnings.warn(
            "Inf found in input arrays, unexpected return might follow",
            category=RuntimeWarning,
            stacklevel=2,
        )

    if x.shape[1] != y.shape[1]:
        msg = (
            f"Number of columns in X and Y must be the same. X has shape {x.shape} "
            f"and Y has shape {y.shape}."
        )
        raise ValueError(msg)
    if not _HAS_SIMSIMD:
        logger.debug(
            "Unable to import simsimd, defaulting to NumPy implementation. If you want "
            "to use simsimd please install with `pip install simsimd`."
        )
        x_norm = np.linalg.norm(x, axis=1)
        y_norm = np.linalg.norm(y, axis=1)
                # 忽略运行时的除以零警告，因为这些情况已在下方处理。
        with np.errstate(divide="ignore", invalid="ignore"):
            similarity: npt.NDArray[np.floating] = np.dot(x, y.T) / np.outer(
                x_norm, y_norm
            )
        if np.isnan(similarity).all():
            msg = "NaN values found, please remove the NaN values and try again"
            raise ValueError(msg) from None
        similarity[np.isnan(similarity) | np.isinf(similarity)] = 0.0
        return similarity

    x = np.array(x, dtype=np.float32)
    y = np.array(y, dtype=np.float32)
    return 1 - np.array(simd.cdist(x, y, metric="cosine"))


def maximal_marginal_relevance(
    query_embedding: npt.NDArray[np.floating],
    embedding_list: list[list[float]],
    lambda_mult: float = 0.5,
    k: int = 4,
) -> list[int]:
    """计算最大边际相关性（MMR）。

    Args:
        query_embedding: 查询嵌入。
        embedding_list: 嵌入列表。
        lambda_mult: MMR 的 lambda 参数。
        k: 要返回的嵌入数量。

    Returns:
        要返回的嵌入索引列表。

    Raises:
        ImportError: 如果未安装 numpy。
    """
    if not _HAS_NUMPY:
        msg = (
            "maximal_marginal_relevance requires numpy to be installed. "
            "Please install numpy with `pip install numpy`."
        )
        raise ImportError(msg)

    if min(k, len(embedding_list)) <= 0:
        return []
    if query_embedding.ndim == 1:
        query_embedding = np.expand_dims(query_embedding, axis=0)
    similarity_to_query = _cosine_similarity(query_embedding, embedding_list)[0]
    most_similar = int(np.argmax(similarity_to_query))
    idxs = [most_similar]
    selected = np.array([embedding_list[most_similar]])
    while len(idxs) < min(k, len(embedding_list)):
        best_score = -np.inf
        idx_to_add = -1
        similarity_to_selected = _cosine_similarity(embedding_list, selected)
        for i, query_score in enumerate(similarity_to_query):
            if i in idxs:
                continue
            redundant_score = max(similarity_to_selected[i])
            equation_score = (
                lambda_mult * query_score - (1 - lambda_mult) * redundant_score
            )
            if equation_score > best_score:
                best_score = equation_score
                idx_to_add = i
        idxs.append(idx_to_add)
        selected = np.append(selected, [embedding_list[idx_to_add]], axis=0)
    return idxs

from typing import TypeAlias

import numpy as np

Matrix: TypeAlias = list[list[float]] | list[np.ndarray] | np.ndarray


def maximal_marginal_relevance(
    query_embedding: np.ndarray,
    embedding_list: list,
    lambda_mult: float = 0.5,
    k: int = 4,
) -> list[int]:
    """计算最大边际相关性。"""
    if min(k, len(embedding_list)) <= 0:
        return []
    if query_embedding.ndim == 1:
        query_embedding = np.expand_dims(query_embedding, axis=0)
    similarity_to_query = cosine_similarity(query_embedding, embedding_list)[0]
    most_similar = int(np.argmax(similarity_to_query))
    idxs = [most_similar]
    selected = np.array([embedding_list[most_similar]])
    while len(idxs) < min(k, len(embedding_list)):
        best_score = -np.inf
        idx_to_add = -1
        similarity_to_selected = cosine_similarity(embedding_list, selected)
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


def cosine_similarity(X: Matrix, Y: Matrix) -> np.ndarray:  # noqa: N803
    """两个等宽矩阵之间的行向量余弦相似度。"""
    if len(X) == 0 or len(Y) == 0:
        return np.array([])

    x: np.ndarray = np.array(X)
    y: np.ndarray = np.array(Y)
    if x.shape[1] != y.shape[1]:
        msg = (
            f"Number of columns in X and Y must be the same. X has shape {x.shape} "
            f"and Y has shape {y.shape}."
        )
        raise ValueError(msg)
    try:
        import simsimd as simd  # noqa: PLC0415

        x = np.array(x, dtype=np.float32)
        y = np.array(y, dtype=np.float32)
        return 1 - np.array(simd.cdist(x, y, metric="cosine"))
    except ImportError:
        x_norm = np.linalg.norm(x, axis=1)
        y_norm = np.linalg.norm(y, axis=1)
                # 忽略运行时的除以零错误警告，因为这些情况已在下方处理。
        with np.errstate(divide="ignore", invalid="ignore"):
            similarity = np.dot(x, y.T) / np.outer(x_norm, y_norm)
        similarity[np.isnan(similarity) | np.isinf(similarity)] = 0.0
        return similarity

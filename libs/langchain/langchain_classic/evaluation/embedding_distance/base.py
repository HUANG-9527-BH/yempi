"""用于通过嵌入比较两个模型输出的链。"""

import functools
import logging
from enum import Enum
from importlib import util
from typing import Any

from langchain_core.callbacks import Callbacks
from langchain_core.callbacks.manager import (
    AsyncCallbackManagerForChainRun,
    CallbackManagerForChainRun,
)
from langchain_core.embeddings import Embeddings
from langchain_core.utils import pre_init
from pydantic import ConfigDict, Field
from typing_extensions import override

from langchain_classic.chains.base import Chain
from langchain_classic.evaluation.schema import PairwiseStringEvaluator, StringEvaluator
from langchain_classic.schema import RUN_KEY


def _import_numpy() -> Any:
    try:
        import numpy as np
    except ImportError as e:
        msg = "Could not import numpy, please install with `pip install numpy`."
        raise ImportError(msg) from e
    return np


logger = logging.getLogger(__name__)


@functools.lru_cache(maxsize=1)
def _check_numpy() -> bool:
    if bool(util.find_spec("numpy")):
        return True
    logger.warning(
        "NumPy not found in the current Python environment. "
        "langchain will use a pure Python implementation for embedding distance "
        "operations, which may significantly impact performance, especially for large "
        "datasets. For optimal speed and efficiency, consider installing NumPy: "
        "pip install numpy",
    )
    return False


def _embedding_factory() -> Embeddings:
    """创建一个 `Embeddings` 对象。

    Returns:
        创建的 `Embeddings` 对象。
    """
        # 此处用于向后兼容。
        # 通常，我们不希望在 langchain 中看到来自 langchain community 的导入
        # 或合作伙伴包的导入。
    try:
        from langchain_openai import OpenAIEmbeddings
    except ImportError:
        try:
            from langchain_community.embeddings.openai import (  # type: ignore[no-redef,unused-ignore]
                OpenAIEmbeddings,
            )
        except ImportError as e:
            msg = (
                "Could not import OpenAIEmbeddings. Please install the "
                "OpenAIEmbeddings package using `pip install langchain-openai`."
            )
            raise ImportError(msg) from e
    return OpenAIEmbeddings()


class EmbeddingDistance(str, Enum):
    """嵌入距离度量。

    Attributes:
        COSINE: 余弦距离度量。
        EUCLIDEAN: 欧几里得距离度量。
        MANHATTAN: 曼哈顿距离度量。
        CHEBYSHEV: 切比雪夫距离度量。
        HAMMING: 汉明距离度量。
    """

    COSINE = "cosine"
    EUCLIDEAN = "euclidean"
    MANHATTAN = "manhattan"
    CHEBYSHEV = "chebyshev"
    HAMMING = "hamming"


class _EmbeddingDistanceChainMixin(Chain):
    """嵌入距离评估器的共享功能。

    Attributes:
        embeddings: 用于向量化输出的嵌入对象。
        distance_metric: 用于比较嵌入的距离度量。
    """

    embeddings: Embeddings = Field(default_factory=_embedding_factory)
    distance_metric: EmbeddingDistance = Field(default=EmbeddingDistance.COSINE)

    @pre_init
    def _validate_tiktoken_installed(cls, values: dict[str, Any]) -> dict[str, Any]:
        """验证 TikTok 库是否已安装。

        Args:
            values: 需要验证的值。

        Returns:
            验证后的值。
        """
        embeddings = values.get("embeddings")
        types_ = []
        try:
            from langchain_openai import OpenAIEmbeddings

            types_.append(OpenAIEmbeddings)
        except ImportError:
            pass

        try:
            from langchain_community.embeddings.openai import (  # type: ignore[no-redef,unused-ignore]
                OpenAIEmbeddings,
            )

            types_.append(OpenAIEmbeddings)
        except ImportError:
            pass

        if not types_:
            msg = (
                "Could not import OpenAIEmbeddings. Please install the "
                "OpenAIEmbeddings package using `pip install langchain-openai`."
            )
            raise ImportError(msg)

        if isinstance(embeddings, tuple(types_)):
            try:
                import tiktoken  # noqa: F401
            except ImportError as e:
                msg = (
                    "The tiktoken library is required to use the default "
                    "OpenAI embeddings with embedding distance evaluators."
                    " Please either manually select a different Embeddings object"
                    " or install tiktoken using `pip install tiktoken`."
                )
                raise ImportError(msg) from e
        return values

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )

    @property
    def output_keys(self) -> list[str]:
        """返回链的输出键。

        Returns:
            输出键。
        """
        return ["score"]

    def _prepare_output(self, result: dict) -> dict:
        parsed = {"score": result["score"]}
        if RUN_KEY in result:
            parsed[RUN_KEY] = result[RUN_KEY]
        return parsed

    def _get_metric(self, metric: EmbeddingDistance) -> Any:
        """获取给定度量名称对应的度量函数。

        Args:
            metric: 度量名称。

        Returns:
            度量函数。
        """
        metrics = {
            EmbeddingDistance.COSINE: self._cosine_distance,
            EmbeddingDistance.EUCLIDEAN: self._euclidean_distance,
            EmbeddingDistance.MANHATTAN: self._manhattan_distance,
            EmbeddingDistance.CHEBYSHEV: self._chebyshev_distance,
            EmbeddingDistance.HAMMING: self._hamming_distance,
        }
        if metric in metrics:
            return metrics[metric]
        msg = f"Invalid metric: {metric}"
        raise ValueError(msg)

    @staticmethod
    def _cosine_distance(a: Any, b: Any) -> Any:
        """计算两个向量之间的余弦距离。

        Args:
            a (np.ndarray): 第一个向量。
            b (np.ndarray): 第二个向量。

        Returns:
            np.ndarray: 余弦距离。
        """
        try:
            from langchain_core.vectorstores.utils import _cosine_similarity

            return 1.0 - _cosine_similarity(a, b)
        except ImportError:
                        # 如果可用，则回退到 scipy
            try:
                from scipy.spatial.distance import cosine

                return cosine(a.flatten(), b.flatten())
            except ImportError:
                                # 纯 numpy 回退
                if _check_numpy():
                    np = _import_numpy()
                    a_flat = a.flatten()
                    b_flat = b.flatten()
                    dot_product = np.dot(a_flat, b_flat)
                    norm_a = np.linalg.norm(a_flat)
                    norm_b = np.linalg.norm(b_flat)
                    if norm_a == 0 or norm_b == 0:
                        return 0.0
                    return 1.0 - (dot_product / (norm_a * norm_b))
                                # 纯 Python 实现
                a_flat = a if hasattr(a, "__len__") else [a]
                b_flat = b if hasattr(b, "__len__") else [b]
                if hasattr(a, "flatten"):
                    a_flat = a.flatten()
                if hasattr(b, "flatten"):
                    b_flat = b.flatten()

                dot_product = sum(x * y for x, y in zip(a_flat, b_flat, strict=False))
                norm_a = sum(x * x for x in a_flat) ** 0.5
                norm_b = sum(x * x for x in b_flat) ** 0.5
                if norm_a == 0 or norm_b == 0:
                    return 0.0
                return 1.0 - (dot_product / (norm_a * norm_b))

    @staticmethod
    def _euclidean_distance(a: Any, b: Any) -> Any:
        """计算两个向量之间的欧几里得距离。

        Args:
            a (np.ndarray): 第一个向量。
            b (np.ndarray): 第二个向量。

        Returns:
            np.floating: 欧几里得距离。
        """
        try:
            from scipy.spatial.distance import euclidean

            return euclidean(a.flatten(), b.flatten())
        except ImportError:
            if _check_numpy():
                import numpy as np

                return np.linalg.norm(a - b)

            return sum((x - y) * (x - y) for x, y in zip(a, b, strict=False)) ** 0.5

    @staticmethod
    def _manhattan_distance(a: Any, b: Any) -> Any:
        """计算两个向量之间的曼哈顿距离。

        Args:
            a (np.ndarray): 第一个向量。
            b (np.ndarray): 第二个向量。

        Returns:
            np.floating: 曼哈顿距离。
        """
        try:
            from scipy.spatial.distance import cityblock

            return cityblock(a.flatten(), b.flatten())
        except ImportError:
            if _check_numpy():
                np = _import_numpy()
                return np.sum(np.abs(a - b))

            return sum(abs(x - y) for x, y in zip(a, b, strict=False))

    @staticmethod
    def _chebyshev_distance(a: Any, b: Any) -> Any:
        """计算两个向量之间的切比雪夫距离。

        Args:
            a (np.ndarray): 第一个向量。
            b (np.ndarray): 第二个向量。

        Returns:
            np.floating: 切比雪夫距离。
        """
        try:
            from scipy.spatial.distance import chebyshev

            return chebyshev(a.flatten(), b.flatten())
        except ImportError:
            if _check_numpy():
                np = _import_numpy()
                return np.max(np.abs(a - b))

            return max(abs(x - y) for x, y in zip(a, b, strict=False))

    @staticmethod
    def _hamming_distance(a: Any, b: Any) -> Any:
        """计算两个向量之间的汉明距离。

        Args:
            a (np.ndarray): 第一个向量。
            b (np.ndarray): 第二个向量。

        Returns:
            np.floating: 汉明距离。
        """
        try:
            from scipy.spatial.distance import hamming

            return hamming(a.flatten(), b.flatten())
        except ImportError:
            if _check_numpy():
                np = _import_numpy()
                return np.mean(a != b)

            return sum(1 for x, y in zip(a, b, strict=False) if x != y) / len(a)

    def _compute_score(self, vectors: Any) -> float:
        """根据距离度量计算得分。

        Args:
            vectors (np.ndarray): 输入向量。

        Returns:
            计算得到的得分。
        """
        metric = self._get_metric(self.distance_metric)
        if _check_numpy() and isinstance(vectors, _import_numpy().ndarray):
            score = metric(vectors[0].reshape(1, -1), vectors[1].reshape(1, -1)).item()
        else:
            score = metric(vectors[0], vectors[1])
        return float(score)


class EmbeddingDistanceEvalChain(_EmbeddingDistanceChainMixin, StringEvaluator):
    """嵌入距离评估链。

    使用嵌入距离对预测值与参考值之间的语义差异进行评分。

    Examples:
        >>> chain = EmbeddingDistanceEvalChain()
        >>> result = chain.evaluate_strings(prediction="Hello", reference="Hi")
        >>> print(result)
        {'score': 0.5}
    """

    @property
    def requires_reference(self) -> bool:
        """返回链是否需要参考值。

        Returns:
            如果需要参考值则返回 True，否则返回 `False`。
        """
        return True

    @property
    @override
    def evaluation_name(self) -> str:
        return f"embedding_{self.distance_metric.value}_distance"

    @property
    def input_keys(self) -> list[str]:
        """返回链的输入键。

        Returns:
            输入键。
        """
        return ["prediction", "reference"]

    @override
    def _call(
        self,
        inputs: dict[str, Any],
        run_manager: CallbackManagerForChainRun | None = None,
    ) -> dict[str, Any]:
        """计算预测值与参考值之间的得分。

        Args:
            inputs: 输入数据。
            run_manager: 回调管理器。

        Returns:
            计算得到的得分。
        """
        vectors = self.embeddings.embed_documents(
            [inputs["prediction"], inputs["reference"]],
        )
        if _check_numpy():
            np = _import_numpy()
            vectors = np.array(vectors)
        score = self._compute_score(vectors)
        return {"score": score}

    @override
    async def _acall(
        self,
        inputs: dict[str, Any],
        run_manager: AsyncCallbackManagerForChainRun | None = None,
    ) -> dict[str, Any]:
        """异步计算预测值与参考值之间的得分。

        Args:
            inputs: 输入数据。
            run_manager: 回调管理器。

        Returns:
            计算得到的得分。
        """
        vectors = await self.embeddings.aembed_documents(
            [
                inputs["prediction"],
                inputs["reference"],
            ],
        )
        if _check_numpy():
            np = _import_numpy()
            vectors = np.array(vectors)
        score = self._compute_score(vectors)
        return {"score": score}

    @override
    def _evaluate_strings(
        self,
        *,
        prediction: str,
        reference: str | None = None,
        callbacks: Callbacks = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        include_run_info: bool = False,
        **kwargs: Any,
    ) -> dict:
        """评估预测值与参考值之间的嵌入距离。

        Args:
            prediction: 第一个模型的输出字符串。
            reference: 第二个模型的输出字符串。
            callbacks: 使用的回调。
            tags: 应用的标签。
            metadata: 使用的元数据。
            include_run_info: 是否在输出中包含运行信息。
            **kwargs: 额外的关键字参数。

        Returns:
            `dict` 包含：
                - score: 两个预测之间的嵌入距离。
        """
        result = self(
            inputs={"prediction": prediction, "reference": reference},
            callbacks=callbacks,
            tags=tags,
            metadata=metadata,
            include_run_info=include_run_info,
        )
        return self._prepare_output(result)

    @override
    async def _aevaluate_strings(
        self,
        *,
        prediction: str,
        reference: str | None = None,
        callbacks: Callbacks = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        include_run_info: bool = False,
        **kwargs: Any,
    ) -> dict:
        """评估两个预测之间的嵌入距离。

        Args:
            prediction: 第一个模型的输出字符串。
            reference: 第二个模型的输出字符串。
            callbacks: 使用的回调。
            tags: 应用的标签。
            metadata: 使用的元数据。
            include_run_info: 是否在输出中包含运行信息。
            **kwargs: 额外的关键字参数。

        Returns:
            `dict` 包含：
                - score: 两个预测之间的嵌入距离。
        """
        result = await self.acall(
            inputs={"prediction": prediction, "reference": reference},
            callbacks=callbacks,
            tags=tags,
            metadata=metadata,
            include_run_info=include_run_info,
        )
        return self._prepare_output(result)


class PairwiseEmbeddingDistanceEvalChain(
    _EmbeddingDistanceChainMixin,
    PairwiseStringEvaluator,
):
    """使用嵌入距离对两个预测值之间的语义差异进行评分。

    Examples:
    >>> chain = PairwiseEmbeddingDistanceEvalChain()
    >>> result = chain.evaluate_string_pairs(prediction="Hello", prediction_b="Hi")
    >>> print(result)
    {'score': 0.5}
    """

    @property
    def input_keys(self) -> list[str]:
        """返回链的输入键。

        Returns:
            输入键。
        """
        return ["prediction", "prediction_b"]

    @property
    def evaluation_name(self) -> str:
        """返回评估名称。"""
        return f"pairwise_embedding_{self.distance_metric.value}_distance"

    @override
    def _call(
        self,
        inputs: dict[str, Any],
        run_manager: CallbackManagerForChainRun | None = None,
    ) -> dict[str, Any]:
        """计算两个预测的得分。

        Args:
            inputs: 输入数据。
            run_manager: 回调管理器。

        Returns:
            计算得到的得分。
        """
        vectors = self.embeddings.embed_documents(
            [
                inputs["prediction"],
                inputs["prediction_b"],
            ],
        )
        if _check_numpy():
            np = _import_numpy()
            vectors = np.array(vectors)
        score = self._compute_score(vectors)
        return {"score": score}

    @override
    async def _acall(
        self,
        inputs: dict[str, Any],
        run_manager: AsyncCallbackManagerForChainRun | None = None,
    ) -> dict[str, Any]:
        """异步计算两个预测的得分。

        Args:
            inputs: 输入数据。
            run_manager: 回调管理器。

        Returns:
            计算得到的得分。
        """
        vectors = await self.embeddings.aembed_documents(
            [
                inputs["prediction"],
                inputs["prediction_b"],
            ],
        )
        if _check_numpy():
            np = _import_numpy()
            vectors = np.array(vectors)
        score = self._compute_score(vectors)
        return {"score": score}

    @override
    def _evaluate_string_pairs(
        self,
        *,
        prediction: str,
        prediction_b: str,
        callbacks: Callbacks = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        include_run_info: bool = False,
        **kwargs: Any,
    ) -> dict:
        """评估两个预测之间的嵌入距离。

        Args:
            prediction: 第一个模型的输出字符串。
            prediction_b: 第二个模型的输出字符串。
            callbacks: 使用的回调。
            tags: 应用的标签。
            metadata: 使用的元数据。
            include_run_info: 是否在输出中包含运行信息。
            **kwargs: 额外的关键字参数。

        Returns:
            `dict` 包含：
                - score: 两个预测之间的嵌入距离。
        """
        result = self(
            inputs={"prediction": prediction, "prediction_b": prediction_b},
            callbacks=callbacks,
            tags=tags,
            metadata=metadata,
            include_run_info=include_run_info,
        )
        return self._prepare_output(result)

    @override
    async def _aevaluate_string_pairs(
        self,
        *,
        prediction: str,
        prediction_b: str,
        callbacks: Callbacks = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        include_run_info: bool = False,
        **kwargs: Any,
    ) -> dict:
        """异步评估两个预测之间的嵌入距离。

        Args:
            prediction: 第一个模型的输出字符串。
            prediction_b: 第二个模型的输出字符串。
            callbacks: 使用的回调。
            tags: 应用的标签。
            metadata: 使用的元数据。
            include_run_info: 是否在输出中包含运行信息。
            **kwargs: 额外的关键字参数。

        Returns:
            `dict` 包含：
                - score: 两个预测之间的嵌入距离。
        """
        result = await self.acall(
            inputs={"prediction": prediction, "prediction_b": prediction_b},
            callbacks=callbacks,
            tags=tags,
            metadata=metadata,
            include_run_info=include_run_info,
        )
        return self._prepare_output(result)

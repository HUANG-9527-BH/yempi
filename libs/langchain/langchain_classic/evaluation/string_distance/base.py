"""基于 RapidFuzz 库的字符串距离评估器。"""

from collections.abc import Callable
from enum import Enum
from typing import Any

from langchain_core.callbacks import Callbacks
from langchain_core.callbacks.manager import (
    AsyncCallbackManagerForChainRun,
    CallbackManagerForChainRun,
)
from langchain_core.utils import pre_init
from pydantic import Field
from typing_extensions import override

from langchain_classic.chains.base import Chain
from langchain_classic.evaluation.schema import PairwiseStringEvaluator, StringEvaluator
from langchain_classic.schema import RUN_KEY


def _load_rapidfuzz() -> Any:
    """加载 RapidFuzz 库。

    Raises:
        ImportError: 如果未安装 rapidfuzz 库。

    Returns:
        `rapidfuzz.distance` 模块。
    """
    try:
        import rapidfuzz
    except ImportError as e:
        msg = (
            "Please install the rapidfuzz library to use the FuzzyMatchStringEvaluator."
            "Please install it with `pip install rapidfuzz`."
        )
        raise ImportError(msg) from e
    return rapidfuzz.distance


class StringDistance(str, Enum):
    """要使用的距离度量标准。

    Attributes:
        `DAMERAU_LEVENSHTEIN`: Damerau-Levenshtein 距离。
        `LEVENSHTEIN`: Levenshtein 距离。
        `JARO`: Jaro 距离。
        `JARO_WINKLER`: Jaro-Winkler 距离。
        `HAMMING`: Hamming 距离。
        `INDEL`: Indel 距离。
    """

    DAMERAU_LEVENSHTEIN = "damerau_levenshtein"
    LEVENSHTEIN = "levenshtein"
    JARO = "jaro"
    JARO_WINKLER = "jaro_winkler"
    HAMMING = "hamming"
    INDEL = "indel"


class _RapidFuzzChainMixin(Chain):
    """RapidFuzz 字符串距离评估器的共享方法。"""

    distance: StringDistance = Field(default=StringDistance.JARO_WINKLER)
    normalize_score: bool = Field(default=True)
    """是否将分数归一化为 `0` 到 `1` 之间的值。
    仅适用于 Levenshtein 和 Damerau-Levenshtein 距离。"""

    @pre_init
    def validate_dependencies(cls, values: dict[str, Any]) -> dict[str, Any]:
        """验证 rapidfuzz 库是否已安装。

        Args:
            values: 输入值。

        Returns:
            验证后的值。
        """
        _load_rapidfuzz()
        return values

    @property
    def output_keys(self) -> list[str]:
        """获取输出键。

        Returns:
            输出键。
        """
        return ["score"]

    def _prepare_output(self, result: dict[str, Any]) -> dict[str, Any]:
        """准备输出字典。

        Args:
            result: 评估结果。

        Returns:
            准备好的输出字典。
        """
        result = {"score": result["score"]}
        if RUN_KEY in result:
            result[RUN_KEY] = result[RUN_KEY].dict()
        return result

    @staticmethod
    def _get_metric(distance: str, *, normalize_score: bool = False) -> Callable:
        """根据距离类型获取距离度量函数。

        Args:
            distance: 距离类型。
            normalize_score: 是否归一化分数。

        Returns:
            距离度量函数。

        Raises:
            ValueError: 如果距离度量无效。
        """
        from rapidfuzz import distance as rf_distance

        module_map: dict[str, Any] = {
            StringDistance.DAMERAU_LEVENSHTEIN: rf_distance.DamerauLevenshtein,
            StringDistance.LEVENSHTEIN: rf_distance.Levenshtein,
            StringDistance.JARO: rf_distance.Jaro,
            StringDistance.JARO_WINKLER: rf_distance.JaroWinkler,
            StringDistance.HAMMING: rf_distance.Hamming,
            StringDistance.INDEL: rf_distance.Indel,
        }
        if distance not in module_map:
            msg = (
                f"Invalid distance metric: {distance}"
                f"\nMust be one of: {list(StringDistance)}"
            )
            raise ValueError(msg)
        module = module_map[distance]
        if normalize_score:
            return module.normalized_distance
        return module.distance

    @property
    def metric(self) -> Callable:
        """获取距离度量函数。

        Returns:
            距离度量函数。
        """
        return _RapidFuzzChainMixin._get_metric(
            self.distance,
            normalize_score=self.normalize_score,
        )

    def compute_metric(self, a: str, b: str) -> float:
        """计算两个字符串之间的距离。

        Args:
            a: 第一个字符串。
            b: 第二个字符串。

        Returns:
            两个字符串之间的距离。
        """
        return self.metric(a, b)


class StringDistanceEvalChain(StringEvaluator, _RapidFuzzChainMixin):
    """计算预测值与参考值之间的字符串距离。

    Examples:
    ----------
    >>> from langchain_classic.evaluation import StringDistanceEvalChain
    >>> evaluator = StringDistanceEvalChain()
    >>> evaluator.evaluate_strings(
            prediction="Mindy is the CTO",
            reference="Mindy is the CEO",
        )

    使用 `load_evaluator` 函数：

    >>> from langchain_classic.evaluation import load_evaluator
    >>> evaluator = load_evaluator("string_distance")
    >>> evaluator.evaluate_strings(
            prediction="The answer is three",
            reference="three",
        )
    """

    @property
    def requires_input(self) -> bool:
        """此评估器不需要输入。"""
        return False

    @property
    def requires_reference(self) -> bool:
        """此评估器不需要参考值。"""
        return True

    @property
    def input_keys(self) -> list[str]:
        """获取输入键。

        Returns:
            输入键。
        """
        return ["reference", "prediction"]

    @property
    def evaluation_name(self) -> str:
        """获取评估名称。

        Returns:
            评估名称。
        """
        return f"{self.distance.value}_distance"

    @override
    def _call(
        self,
        inputs: dict[str, Any],
        run_manager: CallbackManagerForChainRun | None = None,
    ) -> dict[str, Any]:
        """计算预测值与参考值之间的字符串距离。

        Args:
            inputs: 输入值。
            run_manager: 回调管理器。

        Returns:
            包含分数的评估结果。
        """
        return {"score": self.compute_metric(inputs["reference"], inputs["prediction"])}

    @override
    async def _acall(
        self,
        inputs: dict[str, Any],
        run_manager: AsyncCallbackManagerForChainRun | None = None,
    ) -> dict[str, Any]:
        """计算预测值与参考值之间的字符串距离。

        Args:
            inputs: 输入值。
            run_manager: 回调管理器。

        Returns:
            包含分数的评估结果。
        """
        return {"score": self.compute_metric(inputs["reference"], inputs["prediction"])}

    @override
    def _evaluate_strings(
        self,
        *,
        prediction: str,
        reference: str | None = None,
        input: str | None = None,
        callbacks: Callbacks = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        include_run_info: bool = False,
        **kwargs: Any,
    ) -> dict:
        """评估预测值与参考值之间的字符串距离。

        Args:
            prediction: 预测字符串。
            reference: 参考字符串。
            input: 输入字符串。
            callbacks: 使用的回调。
            tags: 应用的标签。
            metadata: 使用的元数据。
            include_run_info: 是否在输出中包含运行信息。
            **kwargs: 其他关键字参数。

        Returns:
            包含分数的评估结果。
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
        input: str | None = None,
        callbacks: Callbacks = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        include_run_info: bool = False,
        **kwargs: Any,
    ) -> dict:
        """评估预测值与参考值之间的字符串距离。

        Args:
            prediction: 预测字符串。
            reference: 参考字符串。
            input: 输入字符串。
            callbacks: 要使用的回调。
            tags: 要应用的标签。
            metadata: 要应用的元数据。
            include_run_info: 是否在输出中包含运行信息。
            **kwargs: 额外的关键字参数。

        Returns:
            评估结果，包含得分。
        """
        result = await self.acall(
            inputs={"prediction": prediction, "reference": reference},
            callbacks=callbacks,
            tags=tags,
            metadata=metadata,
            include_run_info=include_run_info,
        )
        return self._prepare_output(result)


class PairwiseStringDistanceEvalChain(PairwiseStringEvaluator, _RapidFuzzChainMixin):
    """计算两个预测值之间的字符串编辑距离。"""

    @property
    def input_keys(self) -> list[str]:
        """获取输入键。

        Returns:
            输入键。
        """
        return ["prediction", "prediction_b"]

    @property
    def evaluation_name(self) -> str:
        """获取评估名称。

        Returns:
            评估名称。
        """
        return f"pairwise_{self.distance.value}_distance"

    @override
    def _call(
        self,
        inputs: dict[str, Any],
        run_manager: CallbackManagerForChainRun | None = None,
    ) -> dict[str, Any]:
        """计算两个预测之间的字符串距离。

        Args:
            inputs: 输入值。
            run_manager: 回调管理器。

        Returns:
            评估结果，包含得分。
        """
        return {
            "score": self.compute_metric(inputs["prediction"], inputs["prediction_b"]),
        }

    @override
    async def _acall(
        self,
        inputs: dict[str, Any],
        run_manager: AsyncCallbackManagerForChainRun | None = None,
    ) -> dict[str, Any]:
        """异步计算两个预测之间的字符串距离。

        Args:
            inputs: 输入值。
            run_manager: 回调管理器。

        Returns:
            评估结果，包含得分。
        """
        return {
            "score": self.compute_metric(inputs["prediction"], inputs["prediction_b"]),
        }

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
        """评估两个预测之间的字符串距离。

        Args:
            prediction: 第一个预测字符串。
            prediction_b: 第二个预测字符串。
            callbacks: 要使用的回调。
            tags: 要应用的标签。
            metadata: 要使用的元数据。
            include_run_info: 是否在输出中包含运行信息。
            **kwargs: 额外的关键字参数。

        Returns:
            评估结果，包含得分。
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
        """异步评估两个预测之间的字符串距离。

        Args:
            prediction: 第一个预测字符串。
            prediction_b: 第二个预测字符串。
            callbacks: 要使用的回调。
            tags: 要应用的标签。
            metadata: 要使用的元数据。
            include_run_info: 是否在输出中包含运行信息。
            **kwargs: 额外的关键字参数。

        Returns:
            评估结果，包含得分。
        """
        result = await self.acall(
            inputs={"prediction": prediction, "prediction_b": prediction_b},
            callbacks=callbacks,
            tags=tags,
            metadata=metadata,
            include_run_info=include_run_info,
        )
        return self._prepare_output(result)

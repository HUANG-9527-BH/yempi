import string
from typing import Any

from typing_extensions import override

from langchain_classic.evaluation.schema import StringEvaluator


class ExactMatchStringEvaluator(StringEvaluator):
    """计算预测值与参考值之间的精确匹配。

    Examples:
    ----------
    >>> evaluator = ExactMatchChain()
    >>> evaluator.evaluate_strings(
            prediction="Mindy is the CTO",
            reference="Mindy is the CTO",
        )  # This will return {'score': 1.0}

    >>> evaluator.evaluate_strings(
            prediction="Mindy is the CTO",
            reference="Mindy is the CEO",
        )  # This will return {'score': 0.0}
    """

    def __init__(
        self,
        *,
        ignore_case: bool = False,
        ignore_punctuation: bool = False,
        ignore_numbers: bool = False,
        **_: Any,
    ):
        """初始化 `ExactMatchStringEvaluator`。

        Args:
            ignore_case: 比较字符串时是否忽略大小写。
            ignore_punctuation: 比较字符串时是否忽略标点符号。
            ignore_numbers: 比较字符串时是否忽略数字。
        """
        super().__init__()
        self.ignore_case = ignore_case
        self.ignore_punctuation = ignore_punctuation
        self.ignore_numbers = ignore_numbers

    @property
    def requires_input(self) -> bool:
        """此评估器不需要输入。"""
        return False

    @property
    def requires_reference(self) -> bool:
        """此评估器需要参考值。"""
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
        return "exact_match"

    @override
    def _evaluate_strings(  # type: ignore[override]
        self,
        *,
        prediction: str,
        reference: str,
        **kwargs: Any,
    ) -> dict:
        """评估预测值与参考值之间的精确匹配。

        Args:
            prediction: 预测字符串。
            reference: 参考字符串。
            **kwargs: 额外的关键字参数（未使用）。

        Returns:
            包含得分的评估结果。
        """
        if self.ignore_case:
            prediction = prediction.lower()
            reference = reference.lower()
        if self.ignore_punctuation:
            prediction = prediction.translate(str.maketrans("", "", string.punctuation))
            reference = reference.translate(str.maketrans("", "", string.punctuation))
        if self.ignore_numbers:
            prediction = prediction.translate(str.maketrans("", "", string.digits))
            reference = reference.translate(str.maketrans("", "", string.digits))
        return {"score": int(prediction == reference)}

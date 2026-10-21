import re
from typing import Any

from typing_extensions import override

from langchain_classic.evaluation.schema import StringEvaluator


class RegexMatchStringEvaluator(StringEvaluator):
    """计算预测值与参考值之间的正则匹配。

    Examples:
    ----------
    >>> evaluator = RegexMatchStringEvaluator(flags=re.IGNORECASE)
    >>> evaluator.evaluate_strings(
            prediction="Mindy is the CTO",
            reference="^mindy.*cto$",
        )  # This will return {'score': 1.0} due to the IGNORECASE flag

    >>> evaluator = RegexMatchStringEvaluator()
    >>> evaluator.evaluate_strings(
            prediction="Mindy is the CTO",
            reference="^Mike.*CEO$",
        )  # This will return {'score': 0.0}

    >>> evaluator.evaluate_strings(
            prediction="Mindy is the CTO",
            reference="^Mike.*CEO$|^Mindy.*CTO$",
        )  # This will return {'score': 1.0} as the prediction matches the second pattern in the union
    """  # noqa: E501

    def __init__(self, *, flags: int = 0, **_: Any):  # 默认为无标志
        """初始化 RegexMatchStringEvaluator。

        Args:
            flags: 用于正则匹配的标志。默认为无标志。
        """
        super().__init__()
        self.flags = flags

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
        return "regex_match"

    @override
    def _evaluate_strings(  # type: ignore[override]
        self,
        *,
        prediction: str,
        reference: str,
        **kwargs: Any,
    ) -> dict:
        """评估预测值与参考值之间的正则匹配。

        Args:
            prediction: 预测字符串。
            reference: 参考正则表达式模式。
            **kwargs: 额外的关键字参数（不使用）。

        Returns:
            包含得分的评估结果。
        """
        match = re.match(reference, prediction, flags=self.flags)
        return {"score": int(bool(match))}

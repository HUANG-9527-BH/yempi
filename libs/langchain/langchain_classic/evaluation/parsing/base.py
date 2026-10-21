"""用于解析字符串的评估器。"""

import json
import logging
from collections.abc import Callable
from operator import eq
from typing import Any, cast

from langchain_core.utils.json import parse_json_markdown
from typing_extensions import override

from langchain_classic.evaluation.schema import StringEvaluator

_logger = logging.getLogger(__name__)


class JsonValidityEvaluator(StringEvaluator):
    """评估预测是否为有效 JSON。

    This evaluator checks if the prediction is a valid JSON string. It does not
        require any input or reference.

    Attributes:
        requires_input: 该评估器是否需要输入字符串。始终为 False。 requires_reference:
        该评估器是否需要参考字符串。始终为 False。 evaluation_name: 评估指标的名称。
        始终为 "json"。

    Examples:
        >>> evaluator = JsonValidityEvaluator()
        >>> prediction = '{"name": "John", "age": 30, "city": "New York"}'
        >>> evaluator.evaluate(prediction)
        {'score': 1}

        >>> prediction = '{"name": "John", "age": 30, "city": "New York",}'
        >>> evaluator.evaluate(prediction)
        {'score': 0, 'reasoning': 'Expecting property name enclosed in double quotes'}
    """

    def __init__(self, **_: Any) -> None:
        """初始化 JsonValidityEvaluator。"""
        super().__init__()

    @property
    @override
    def requires_input(self) -> bool:
        return False

    @property
    @override
    def requires_reference(self) -> bool:
        return False

    @property
    @override
    def evaluation_name(self) -> str:
        return "json_validity"

    @override
    def _evaluate_strings(
        self,
        prediction: str,
        **kwargs: Any,
    ) -> dict:
        """评估预测字符串。

        Args:
            prediction: 要评估的预测字符串。
            **kwargs: 额外的关键字参数（未使用）。

        Returns:
            包含评估分数的 `dict`。如果预测为有效 JSON，则分数为 `1`，否则为 `0`。

                If the prediction is not valid JSON, the dictionary also contains
                a `reasoning` field with the error message.

        """
        try:
            parse_json_markdown(prediction, parser=json.loads)
        except json.JSONDecodeError as e:
            return {"score": 0, "reasoning": str(e)}
        except Exception as e:
            _logger.exception("Passing JSON failed with unexpected error.")
            return {"score": 0, "reasoning": str(e)}
        return {"score": 1}


class JsonEqualityEvaluator(StringEvaluator):
    """JSON 等价性评估器。

    评估预测在解析为 JSON 后是否与参考值相等。 此评估器检查预测在解析为 JSON
    后是否等于参考值， 参考值同样被解析为 JSON。它不需要输入字符串。

    This evaluator checks if the prediction, after parsing as JSON, is equal
        to the reference,
    which is also parsed as JSON. It does not require an input string.

    Attributes:
        requires_input: 该评估器是否需要输入字符串。始终为 False。 requires_reference:
        该评估器是否需要参考字符串。始终为 True。 evaluation_name: 评估指标的名称。
        始终为 "parsed_equality"。

    Examples:
        >>> evaluator = JsonEqualityEvaluator()
        >>> evaluator.evaluate_strings('{"a": 1}', reference='{"a": 1}')
        {'score': True}
        >>> evaluator.evaluate_strings('{"a": 1}', reference='{"a": 2}')
        {'score': False}

        >>> evaluator = JsonEqualityEvaluator(operator=lambda x, y: x["a"] == y["a"])
        >>> evaluator.evaluate_strings('{"a": 1}', reference='{"a": 1}')
        {'score': True}
        >>> evaluator.evaluate_strings('{"a": 1}', reference='{"a": 2}')
        {'score': False}

    """

    def __init__(self, operator: Callable | None = None, **_: Any) -> None:
        """初始化 JsonEqualityEvaluator。

        Args:
            operator: 用于比较解析后的 JSON 对象的自定义操作符。
                默认为等价性（`eq`）。
        """
        super().__init__()
        self.operator = operator or eq

    @property
    @override
    def requires_input(self) -> bool:
        return False

    @property
    @override
    def requires_reference(self) -> bool:
        return True

    @property
    @override
    def evaluation_name(self) -> str:
        return "json_equality"

    def _parse_json(
        self,
        string: Any,
    ) -> dict | list | None | float | bool | int | str:
        if isinstance(string, str):
            return parse_json_markdown(string)
        return string

    @override
    def _evaluate_strings(
        self,
        prediction: str,
        reference: str | None = None,
        **kwargs: Any,
    ) -> dict:
        """评估预测字符串。

        Args:
            prediction: 要评估的预测字符串。
            reference: 用于对比的参考字符串。
            **kwargs: 额外的关键字参数（未使用）。

        Returns:
            包含评估分数的 `dict`。
        """
        parsed = self._parse_json(prediction)
        label = self._parse_json(cast("str", reference))
        if isinstance(label, list):
            if not isinstance(parsed, list):
                return {"score": 0}
            parsed = sorted(parsed, key=str)
            label = sorted(label, key=str)
        return {"score": self.operator(parsed, label)}

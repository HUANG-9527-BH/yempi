from typing import Any

from langchain_core.utils.json import parse_json_markdown
from typing_extensions import override

from langchain_classic.evaluation.schema import StringEvaluator


class JsonSchemaEvaluator(StringEvaluator):
    """一个用于验证 JSON 预测是否符合 JSON 模式参考的评估器。

    此评估器检查给定的 JSON 预测是否符合提供的 JSON 模式。
    如果预测有效，得分为 True（无错误）。否则，得分为 False（发生错误）。

    Attributes:
        requires_input: 评估器是否需要输入。
        requires_reference: 评估器是否需要参考。
        evaluation_name: 评估的名称。

    Examples:
        evaluator = JsonSchemaEvaluator()
        result = evaluator.evaluate_strings(
            prediction='{"name": "John", "age": 30}',
            reference={
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "age": {"type": "integer"}
                }
            }
        )
        assert result["score"] is not None

    """  # noqa: E501

    def __init__(self, **_: Any) -> None:
        """初始化 JsonSchemaEvaluator。

        Raises:
            ImportError: 如果未安装 jsonschema 包。
        """
        super().__init__()
        try:
            import jsonschema  # noqa: F401
        except ImportError as e:
            msg = (
                "The JsonSchemaEvaluator requires the jsonschema package."
                " Please install it with `pip install jsonschema`."
            )
            raise ImportError(msg) from e

    @property
    def requires_input(self) -> bool:
        """返回评估器是否需要输入。"""
        return False

    @property
    def requires_reference(self) -> bool:
        """返回评估器是否需要参考。"""
        return True

    @property
    def evaluation_name(self) -> str:
        """返回评估的名称。"""
        return "json_schema_validation"

    def _parse_json(self, node: Any) -> dict | list | None | float | bool | int | str:
        if isinstance(node, str):
            return parse_json_markdown(node)
        if hasattr(node, "model_json_schema") and callable(node.model_json_schema):
                        # Pydantic v2 model
            return node.model_json_schema()
        if hasattr(node, "schema") and callable(node.schema):
                        # Pydantic v1 model
            return node.schema()
        return node

    def _validate(self, prediction: Any, schema: Any) -> dict:
        from jsonschema import ValidationError, validate

        try:
            validate(instance=prediction, schema=schema)
        except ValidationError as e:
            return {"score": False, "reasoning": repr(e)}
        return {"score": True}

    @override
    def _evaluate_strings(
        self,
        prediction: str | Any,
        input: str | Any = None,
        reference: str | Any = None,
        **kwargs: Any,
    ) -> dict:
        parsed_prediction = self._parse_json(prediction)
        schema = self._parse_json(reference)
        return self._validate(parsed_prediction, schema)

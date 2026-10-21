"""JSON 输出的解析器。"""

from __future__ import annotations

import json
from json import JSONDecodeError
from typing import Annotated, Any, TypeVar

import jsonpatch  # type: ignore[import-untyped]
import pydantic
from pydantic import SkipValidation
from pydantic.v1 import BaseModel
from typing_extensions import override

from langchain_core.exceptions import OutputParserException
from langchain_core.output_parsers.format_instructions import JSON_FORMAT_INSTRUCTIONS
from langchain_core.output_parsers.transform import BaseCumulativeTransformOutputParser
from langchain_core.outputs import Generation
from langchain_core.utils.json import (
    parse_and_check_json_markdown,
    parse_json_markdown,
    parse_partial_json,
)

# Union 类型需要最后赋值给 PydanticBaseModel 以让 mypy 满意。
PydanticBaseModel = BaseModel | pydantic.BaseModel

TBaseModel = TypeVar("TBaseModel", bound=PydanticBaseModel)


class JsonOutputParser(BaseCumulativeTransformOutputParser[Any]):
    """将 LLM 调用的输出解析为 JSON 对象。

    Probably the most reliable output parser for getting structured data that does *not*
    use function calling.

    When used in streaming mode, it will yield partial JSON objects containing all the
    keys that have been returned so far.

    In streaming, if `diff` is set to `True`, yields `JSONPatch` operations describing
    the difference between the previous and the current object.
    """

    pydantic_object: Annotated[type[TBaseModel] | None, SkipValidation()] = None  # type: ignore[valid-type]
    """用于验证的 Pydantic 对象。

    如果为 `None`，则不执行验证。
    """

    @override
    def _diff(self, prev: Any | None, next: Any) -> Any:
        return jsonpatch.make_patch(prev, next).patch

    @staticmethod
    def _get_schema(pydantic_object: type[TBaseModel]) -> dict[str, Any]:
        if issubclass(pydantic_object, pydantic.BaseModel):
            return pydantic_object.model_json_schema()
        return pydantic_object.schema()

    @override
    def parse_result(self, result: list[Generation], *, partial: bool = False) -> Any:
        """将 LLM 调用的结果解析为 JSON 对象。

        Args:
            result: LLM 调用的结果。
            partial: 是否解析部分 JSON 对象。

                如果为 `True`，输出将是一个包含所有已返回键的 JSON 对象。

                如果为 `False`，输出将是完整的 JSON 对象。

        Returns:
            解析后的 JSON 对象。

        Raises:
            OutputParserException: 如果输出不是有效的 JSON。
        """
        text = result[0].text
        text = text.strip()
        if partial:
            try:
                return parse_json_markdown(text)
            except JSONDecodeError:
                return None
        else:
            try:
                return parse_json_markdown(text)
            except JSONDecodeError as e:
                msg = f"Invalid json output: {text}"
                raise OutputParserException(msg, llm_output=text) from e

    def parse(self, text: str) -> Any:
        """将 LLM 调用的输出解析为 JSON 对象。

        Args:
            text: LLM 调用的输出。

        Returns:
            解析后的 JSON 对象。
        """
        return self.parse_result([Generation(text=text)])

    def get_format_instructions(self) -> str:
        """返回 JSON 输出的格式说明。

        Returns:
            JSON 输出的格式说明。
        """
        if self.pydantic_object is None:
            return "Return a JSON object."
                # 复制模式以避免修改原始 Pydantic 模式。
        schema = dict(self._get_schema(self.pydantic_object).items())

                # 移除多余字段。
        reduced_schema = schema
        if "title" in reduced_schema:
            del reduced_schema["title"]
        if "type" in reduced_schema:
            del reduced_schema["type"]
                # 确保上下文中的 json 使用双引号，格式正确。
        schema_str = json.dumps(reduced_schema, ensure_ascii=False)
        return JSON_FORMAT_INSTRUCTIONS.format(schema=schema_str)

    @property
    def _type(self) -> str:
        return "simple_json_output_parser"


# 为向后兼容性保留
SimpleJsonOutputParser = JsonOutputParser


__all__ = [
    "JsonOutputParser",
    "SimpleJsonOutputParser",  # 为向后兼容性保留
    "parse_and_check_json_markdown",  # 为向后兼容性保留
    "parse_partial_json",  # 为向后兼容性保留
]

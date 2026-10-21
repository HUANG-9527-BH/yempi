"""使用 Pydantic 的输出解析器。"""

import json
from typing import Annotated, Any, Generic, Literal, overload

import pydantic
from pydantic import SkipValidation
from typing_extensions import override

from langchain_core.exceptions import OutputParserException
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.outputs import Generation
from langchain_core.utils.pydantic import (
    PydanticBaseModel,
    TBaseModel,
)


class PydanticOutputParser(JsonOutputParser, Generic[TBaseModel]):
    """使用 Pydantic 模型解析输出。"""

    pydantic_object: Annotated[type[TBaseModel], SkipValidation()]
    """要解析的 Pydantic 模型。"""

    def _parse_obj(self, obj: Any) -> TBaseModel:
        try:
            if issubclass(self.pydantic_object, pydantic.BaseModel):
                return self.pydantic_object.model_validate(obj)
            if issubclass(self.pydantic_object, pydantic.v1.BaseModel):
                return self.pydantic_object.parse_obj(obj)
            msg = (  # type: ignore[unreachable]
                "Unsupported model version for PydanticOutputParser: "
                f"{self.pydantic_object.__class__}"
            )
            raise OutputParserException(msg)
        except (pydantic.ValidationError, pydantic.v1.ValidationError) as e:
            raise self._parser_exception(e, obj) from e

    def _parser_exception(
        self, e: Exception, json_object: Any
    ) -> OutputParserException:
        json_string = json.dumps(json_object, ensure_ascii=False)
        name = self.pydantic_object.__name__
        msg = f"Failed to parse {name} from completion {json_string}. Got: {e}"
        return OutputParserException(msg, llm_output=json_string)

    @overload
    def parse_result(
        self, result: list[Generation], *, partial: Literal[False] = False
    ) -> TBaseModel: ...

    @overload
    def parse_result(
        self, result: list[Generation], *, partial: bool = False
    ) -> TBaseModel | None: ...

    def parse_result(
        self, result: list[Generation], *, partial: bool = False
    ) -> TBaseModel | None:
        """将 LLM 调用的结果解析为 Pydantic 对象。

        Args:
            result: LLM 调用的结果。
            partial: 是否解析部分 JSON 对象。

                如果为 `True`，输出将是一个包含目前已返回所有键的 JSON 对象。

        Raises:
            OutputParserException: 如果结果不是有效的 JSON 或不符合 Pydantic 模型。

        Returns:
            解析后的 Pydantic 对象。
        """
        try:
            json_object = super().parse_result(result, partial=partial)
            return self._parse_obj(json_object)
        except OutputParserException:
            if partial:
                return None
            raise

    def parse(self, text: str) -> TBaseModel:
        """将 LLM 调用的输出解析为 Pydantic 对象。

        Args:
            text: LLM 调用的输出。

        Returns:
            解析后的 Pydantic 对象。
        """
        return self.parse_result([Generation(text=text)])

    def get_format_instructions(self) -> str:
        """返回 JSON 输出的格式说明。

        Returns:
            JSON 输出的格式说明。
        """
                # 复制模式以避免修改原始 Pydantic 模式。
        schema = dict(self._get_schema(self.pydantic_object).items())

                # 移除多余的字段。
        reduced_schema = schema
        if "title" in reduced_schema:
            del reduced_schema["title"]
        if "type" in reduced_schema:
            del reduced_schema["type"]
                # 确保上下文中的 JSON 使用双引号，格式正确。
        schema_str = json.dumps(reduced_schema, ensure_ascii=False)

        return _PYDANTIC_FORMAT_INSTRUCTIONS.format(schema=schema_str)

    @property
    def _type(self) -> str:
        return "pydantic"

    @property
    @override
    def OutputType(self) -> type[TBaseModel]:
        """返回 Pydantic 模型。"""
        return self.pydantic_object


_PYDANTIC_FORMAT_INSTRUCTIONS = """The output should be formatted as a JSON instance that conforms to the JSON schema below.

As an example, for the schema {{"properties": {{"foo": {{"title": "Foo", "description": "a list of strings", "type": "array", "items": {{"type": "string"}}}}}}, "required": ["foo"]}}
the object {{"foo": ["bar", "baz"]}} is a well-formatted instance of the schema. The object {{"properties": {{"foo": ["bar", "baz"]}}}} is not well-formatted.

Here is the output schema:
```
{schema}
```"""  # noqa: E501

# 重新导出类型以保证向后兼容
__all__ = [
    "PydanticBaseModel",
    "PydanticOutputParser",
    "TBaseModel",
]

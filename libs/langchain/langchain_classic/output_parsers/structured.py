from __future__ import annotations

from typing import Any

from langchain_core.output_parsers import BaseOutputParser
from langchain_core.output_parsers.json import parse_and_check_json_markdown
from pydantic import BaseModel
from typing_extensions import override

from langchain_classic.output_parsers.format_instructions import (
    STRUCTURED_FORMAT_INSTRUCTIONS,
    STRUCTURED_FORMAT_SIMPLE_INSTRUCTIONS,
)

line_template = '\t"{name}": {type}  // {description}'


class ResponseSchema(BaseModel):
    """结构化输出解析器的响应模式。"""

    name: str
    """模式的名称。"""
    description: str
    """模式的描述。"""
    type: str = "string"
    """响应的类型。"""


def _get_sub_string(schema: ResponseSchema) -> str:
    return line_template.format(
        name=schema.name,
        description=schema.description,
        type=schema.type,
    )


class StructuredOutputParser(BaseOutputParser[dict[str, Any]]):
    """将 LLM 调用的输出解析为结构化输出。"""

    response_schemas: list[ResponseSchema]
    """响应的模式列表。"""

    @classmethod
    def from_response_schemas(
        cls,
        response_schemas: list[ResponseSchema],
    ) -> StructuredOutputParser:
        """从 ResponseSchema 列表创建 StructuredOutputParser。

        Args:
            response_schemas: 响应的模式列表。

        Returns:
            StructuredOutputParser 的实例。
        """
        return cls(response_schemas=response_schemas)

    def get_format_instructions(
        self,
        only_json: bool = False,  # noqa: FBT001,FBT002
    ) -> str:
        """获取输出解析器的格式说明。

        Example:
        ```python
        from langchain_classic.output_parsers.structured import (
            StructuredOutputParser, ResponseSchema
        )

        response_schemas = [
            ResponseSchema(
                name="foo",
                description="a list of strings",
                type="List[string]"
                ),
            ResponseSchema(
                name="bar",
                description="a string",
                type="string"
                ),
        ]

        parser = StructuredOutputParser.from_response_schemas(response_schemas)

        print(parser.get_format_instructions())  # noqa: T201

        output:
        # The output should be a Markdown code snippet formatted in the following
        # schema, including the leading and trailing "```json" and "```":
        #
        # ```json
        # {
        #     "foo": List[string]  // a list of strings
        #     "bar": string  // a string
        # }
        # ```

        Args:
            only_json: If `True`, only the json in the Markdown code snippet
                will be returned, without the introducing text.
        """
        schema_str = "\n".join(
            [_get_sub_string(schema) for schema in self.response_schemas],
        )
        if only_json:
            return STRUCTURED_FORMAT_SIMPLE_INSTRUCTIONS.format(format=schema_str)
        return STRUCTURED_FORMAT_INSTRUCTIONS.format(format=schema_str)

    @override
    def parse(self, text: str) -> dict[str, Any]:
        expected_keys = [rs.name for rs in self.response_schemas]
        return parse_and_check_json_markdown(text, expected_keys)

    @property
    def _type(self) -> str:
        return "structured"

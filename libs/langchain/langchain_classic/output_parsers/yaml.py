import json
import re
from typing import TypeVar

import yaml
from langchain_core.exceptions import OutputParserException
from langchain_core.output_parsers import BaseOutputParser
from pydantic import BaseModel, ValidationError
from typing_extensions import override

from langchain_classic.output_parsers.format_instructions import (
    YAML_FORMAT_INSTRUCTIONS,
)

T = TypeVar("T", bound=BaseModel)


class YamlOutputParser(BaseOutputParser[T]):
    """使用 Pydantic 模型解析 YAML 输出。"""

    pydantic_object: type[T]
    """用于解析的 Pydantic 模型。"""
    pattern: re.Pattern = re.compile(
        r"^```(?:ya?ml)?(?P<yaml>[^`]*)",
        re.MULTILINE | re.DOTALL,
    )
    """Regex pattern to match yaml code blocks
    within triple backticks with optional yaml or yml prefix."""

    @override
    def parse(self, text: str) -> T:
        try:
                        # 贪婪搜索第一个 YAML 候选内容。
            match = re.search(self.pattern, text.strip())
                        # 如果没有出现反引号，尝试将整个输出作为 YAML 解析。
            yaml_str = match.group("yaml") if match else text

            json_object = yaml.safe_load(yaml_str)
            return self.pydantic_object.model_validate(json_object)

        except (yaml.YAMLError, ValidationError) as e:
            name = self.pydantic_object.__name__
            msg = f"Failed to parse {name} from completion {text}. Got: {e}"
            raise OutputParserException(msg, llm_output=text) from e

    @override
    def get_format_instructions(self) -> str:
                # 复制模式以避免修改原始 Pydantic 模式。
        schema = dict(self.pydantic_object.model_json_schema().items())

                # 移除多余字段。
        reduced_schema = schema
        if "title" in reduced_schema:
            del reduced_schema["title"]
        if "type" in reduced_schema:
            del reduced_schema["type"]
                # 确保上下文中的 YAML 格式正确，使用双引号。
        schema_str = json.dumps(reduced_schema)

        return YAML_FORMAT_INSTRUCTIONS.format(schema=schema_str)

    @property
    def _type(self) -> str:
        return "yaml"

    @property
    @override
    def OutputType(self) -> type[T]:
        return self.pydantic_object

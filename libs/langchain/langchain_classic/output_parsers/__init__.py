"""OutputParser 类用于解析 LLM 调用的输出。"""

from typing import TYPE_CHECKING, Any

from langchain_core.output_parsers import (
    CommaSeparatedListOutputParser,
    ListOutputParser,
    MarkdownListOutputParser,
    NumberedListOutputParser,
    PydanticOutputParser,
    XMLOutputParser,
)
from langchain_core.output_parsers.openai_tools import (
    JsonOutputKeyToolsParser,
    JsonOutputToolsParser,
    PydanticToolsParser,
)

from langchain_classic._api import create_importer
from langchain_classic.output_parsers.boolean import BooleanOutputParser
from langchain_classic.output_parsers.combining import CombiningOutputParser
from langchain_classic.output_parsers.datetime import DatetimeOutputParser
from langchain_classic.output_parsers.enum import EnumOutputParser
from langchain_classic.output_parsers.fix import OutputFixingParser
from langchain_classic.output_parsers.pandas_dataframe import (
    PandasDataFrameOutputParser,
)
from langchain_classic.output_parsers.regex import RegexParser
from langchain_classic.output_parsers.regex_dict import RegexDictParser
from langchain_classic.output_parsers.retry import (
    RetryOutputParser,
    RetryWithErrorOutputParser,
)
from langchain_classic.output_parsers.structured import (
    ResponseSchema,
    StructuredOutputParser,
)
from langchain_classic.output_parsers.yaml import YamlOutputParser

if TYPE_CHECKING:
    from langchain_community.output_parsers.rail_parser import GuardrailsOutputParser

# 创建一种动态查找已弃用导入的方式。
# 用于统一处理弃用警告的逻辑并
# 处理可选导入。
DEPRECATED_LOOKUP = {
    "GuardrailsOutputParser": "langchain_community.output_parsers.rail_parser",
}

_import_attribute = create_importer(__package__, deprecated_lookups=DEPRECATED_LOOKUP)


def __getattr__(name: str) -> Any:
    """动态查找属性。"""
    return _import_attribute(name)


__all__ = [
    "BooleanOutputParser",
    "CombiningOutputParser",
    "CommaSeparatedListOutputParser",
    "DatetimeOutputParser",
    "EnumOutputParser",
    "GuardrailsOutputParser",
    "JsonOutputKeyToolsParser",
    "JsonOutputToolsParser",
    "ListOutputParser",
    "MarkdownListOutputParser",
    "NumberedListOutputParser",
    "OutputFixingParser",
    "PandasDataFrameOutputParser",
    "PydanticOutputParser",
    "PydanticToolsParser",
    "RegexDictParser",
    "RegexParser",
    "ResponseSchema",
    "RetryOutputParser",
    "RetryWithErrorOutputParser",
    "StructuredOutputParser",
    "XMLOutputParser",
    "YamlOutputParser",
]

import re
from typing import Any, Generic

from langchain_core.output_parsers import JsonOutputParser, PydanticOutputParser
from langchain_core.outputs import Generation
from langchain_core.utils.pydantic import TBaseModel


def strip_think_tags(text: str) -> str:
    """从文本中移除所有<think>...</think> 标签及其内容。

    此函数会移除所有 think 标签的出现，保留标签前后的文本。
    它还处理 markdown 代码块标记。

    Args:
        text: 可能包含 think 标签的输入文本。

    Returns:
        移除所有 `thinks>...<think>` 块后的文本。
    """
        # 使用正则表达式移除所有<think>...</think> 块
        # 该模式匹配 <think> 后跟任意内容（非贪婪）直到 </think>
    result = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)

        # 如果存在，移除 markdown 代码块标记
    result = result.strip()
    if result.startswith("```json"):
        result = result[len("```json") :].strip()
    elif result.startswith("```"):
        result = result[3:].strip()

    if result.endswith("```"):
        result = result[:-3].strip()

    return result


class ReasoningJsonOutputParser(JsonOutputParser):
    """一个在解析前清除推理标签的 JSON 输出解析器。

    此解析器会在将输入文本传递给父级 JsonOutputParser 进行 JSON 解析之前，
    移除任何被 <think> 标签包围的内容。

    """

    def parse_result(self, result: list[Generation], *, partial: bool = False) -> Any:
        """将 LLM 调用的结果解析为 JSON 对象。

        Args:
            result: LLM 调用的结果。
            partial: 是否解析部分 JSON 对象。
                如果为 `True`，输出将是一个包含
                已返回所有键的 JSON 对象。
                如果为 `False`，输出将是完整的 JSON 对象。

        Returns:
            解析后的 JSON 对象。

        Raises:
            OutputParserException: 如果输出不是有效的 JSON。
        """
        text = result[0].text
        text = strip_think_tags(text)
        return super().parse_result([Generation(text=text)], partial=partial)


class ReasoningStructuredOutputParser(
    PydanticOutputParser[TBaseModel], Generic[TBaseModel]
):
    """一个在解析前清除推理标签的结构化输出解析器。

    此解析器会在将输入文本传递给父级 PydanticOutputParser 进行结构化解析之前，
    移除任何被 <think> 标签包围的内容。
    """

    def parse_result(self, result: list[Generation], *, partial: bool = False) -> Any:
        """将 LLM 调用的结果解析为 Pydantic 对象。

        Args:
            result: LLM 调用的结果。
            partial: 是否解析部分 JSON 对象。
                如果为 `True`，输出将是一个包含
                已返回所有键的 JSON 对象。
                如果为 `False`，输出将是完整的 JSON 对象。
        """
        text = result[0].text
        text = strip_think_tags(text)
        return super().parse_result([Generation(text=text)], partial=partial)

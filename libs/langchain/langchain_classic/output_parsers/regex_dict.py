from __future__ import annotations

import re

from langchain_core.output_parsers import BaseOutputParser


class RegexDictParser(BaseOutputParser[dict[str, str]]):
    """使用正则表达式将 LLM 调用的输出解析为字典。"""

    regex_pattern: str = r"{}:\s?([^.'\n']*)\.?"
    """用于解析输出的正则表达式模式。"""
    output_key_to_format: dict[str, str]
    """用于输出的键。"""
    no_update_value: str | None = None
    """用于输出的默认键。"""

    @property
    def _type(self) -> str:
        """返回类型键。"""
        return "regex_dict_parser"

    def parse(self, text: str) -> dict[str, str]:
        """解析 LLM 调用的输出。"""
        result = {}
        for output_key, expected_format in self.output_key_to_format.items():
            specific_regex = self.regex_pattern.format(re.escape(expected_format))
            matches = re.findall(specific_regex, text)
            if not matches:
                msg = (
                    f"No match found for output key: {output_key} with expected format \
                        {expected_format} on text {text}"
                )
                raise ValueError(msg)
            if len(matches) > 1:
                msg = f"Multiple matches found for output key: {output_key} with \
                        expected format {expected_format} on text {text}"
                raise ValueError(msg)
            if self.no_update_value is not None and matches[0] == self.no_update_value:
                continue
            result[output_key] = matches[0]
        return result

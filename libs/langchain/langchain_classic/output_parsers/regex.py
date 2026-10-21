from __future__ import annotations

import re

from langchain_core.output_parsers import BaseOutputParser
from typing_extensions import override


class RegexParser(BaseOutputParser[dict[str, str]]):
    """使用正则表达式解析 LLM 调用的输出。"""

    @classmethod
    @override
    def is_lc_serializable(cls) -> bool:
        return True

    regex: str
    """用于解析输出的正则表达式。"""
    output_keys: list[str]
    """用于输出的键。"""
    default_output_key: str | None = None
    """用于输出的默认键。"""

    @property
    def _type(self) -> str:
        """返回类型键。"""
        return "regex_parser"

    def parse(self, text: str) -> dict[str, str]:
        """解析 LLM 调用的输出。"""
        match = re.search(self.regex, text)
        if match:
            return {key: match.group(i + 1) for i, key in enumerate(self.output_keys)}
        if self.default_output_key is None:
            msg = f"Could not parse output: {text}"
            raise ValueError(msg)
        return {
            key: text if key == self.default_output_key else ""
            for key in self.output_keys
        }

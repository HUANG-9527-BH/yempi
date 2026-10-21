from __future__ import annotations

from typing import Any

from langchain_core.output_parsers import BaseOutputParser
from langchain_core.utils import pre_init
from typing_extensions import override

_MIN_PARSERS = 2


class CombiningOutputParser(BaseOutputParser[dict[str, Any]]):
    """将多个输出解析器合并为一个。"""

    parsers: list[BaseOutputParser]

    @classmethod
    @override
    def is_lc_serializable(cls) -> bool:
        return True

    @pre_init
    def validate_parsers(cls, values: dict[str, Any]) -> dict[str, Any]:
        """验证解析器。"""
        parsers = values["parsers"]
        if len(parsers) < _MIN_PARSERS:
            msg = "Must have at least two parsers"
            raise ValueError(msg)
        for parser in parsers:
            if parser._type == "combining":  # noqa: SLF001
                msg = "Cannot nest combining parsers"
                raise ValueError(msg)
            if parser._type == "list":  # noqa: SLF001
                msg = "Cannot combine list parsers"
                raise ValueError(msg)
        return values

    @property
    def _type(self) -> str:
        """返回类型键。"""
        return "combining"

    def get_format_instructions(self) -> str:
        """关于 LLM 输出应如何格式化的说明。"""
        initial = f"For your first output: {self.parsers[0].get_format_instructions()}"
        subsequent = "\n".join(
            f"Complete that output fully. Then produce another output, separated by two newline characters: {p.get_format_instructions()}"  # noqa: E501
            for p in self.parsers[1:]
        )
        return f"{initial}\n{subsequent}"

    def parse(self, text: str) -> dict[str, Any]:
        """解析 LLM 调用的输出。"""
        texts = text.split("\n\n")
        output = {}
        for txt, parser in zip(texts, self.parsers, strict=False):
            output.update(parser.parse(txt.strip()))
        return output

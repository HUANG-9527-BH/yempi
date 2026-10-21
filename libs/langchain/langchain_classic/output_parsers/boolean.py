import re

from langchain_core.output_parsers import BaseOutputParser


class BooleanOutputParser(BaseOutputParser[bool]):
    """将 LLM 调用的输出解析为布尔值。"""

    true_val: str = "YES"
    """应解析为 True 的字符串值。"""
    false_val: str = "NO"
    """应解析为 False 的字符串值。"""

    def parse(self, text: str) -> bool:
        """将 LLM 调用的输出解析为布尔值。

        Args:
            text: 语言模型的输出

        Returns:
            boolean
        """
        regexp = rf"\b({self.true_val}|{self.false_val})\b"

        truthy = {
            val.upper()
            for val in re.findall(regexp, text, flags=re.IGNORECASE | re.MULTILINE)
        }
        if self.true_val.upper() in truthy:
            if self.false_val.upper() in truthy:
                msg = (
                    f"Ambiguous response. Both {self.true_val} and {self.false_val} "
                    f"in received: {text}."
                )
                raise ValueError(msg)
            return True
        if self.false_val.upper() in truthy:
            if self.true_val.upper() in truthy:
                msg = (
                    f"Ambiguous response. Both {self.true_val} and {self.false_val} "
                    f"in received: {text}."
                )
                raise ValueError(msg)
            return False
        msg = (
            f"BooleanOutputParser expected output value to include either "
            f"{self.true_val} or {self.false_val}. Received {text}."
        )
        raise ValueError(msg)

    @property
    def _type(self) -> str:
        """输出解析器类型的蛇形命名标识符。"""
        return "boolean_output_parser"

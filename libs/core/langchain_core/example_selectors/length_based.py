"""根据长度选择示例。"""

import re
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, Field, model_validator
from typing_extensions import Self

from langchain_core.example_selectors.base import BaseExampleSelector
from langchain_core.prompts.prompt import PromptTemplate


def _get_length_based(text: str) -> int:
    return len(re.split(r"\n| ", text))


class LengthBasedExampleSelector(BaseExampleSelector, BaseModel):
    r"""根据长度选择示例。

    Example:
        ```python
        from langchain_core.example_selectors import LengthBasedExampleSelector
        from langchain_core.prompts import PromptTemplate

        # Define examples
        examples = [
            {"input": "happy", "output": "sad"},
            {"input": "tall", "output": "short"},
            {"input": "fast", "output": "slow"},
        ]

        # Create prompt template
        example_prompt = PromptTemplate(
            input_variables=["input", "output"],
            template="Input: {input}\nOutput: {output}",
        )

        # Create selector with max length constraint
        selector = LengthBasedExampleSelector(
            examples=examples,
            example_prompt=example_prompt,
            max_length=50,  # Maximum prompt length
        )

        # Select examples for a new input
        selected = selector.select_examples({"input": "large", "output": "tiny"})
        # Returns examples that fit within max_length constraint
        ```
    """

    examples: list[dict[str, Any]]
    """提示词模板期望的示例列表。"""

    example_prompt: PromptTemplate
    """用于格式化示例的提示词模板。"""

    get_text_length: Callable[[str], int] = _get_length_based
    """衡量提示词长度的函数。默认为词数统计。"""

    max_length: int = 2048
    """提示词的最大长度，超过此长度的示例将被截断。"""

    example_text_lengths: list[int] = Field(default_factory=list)
    """每个示例的长度。"""

    def add_example(self, example: dict[str, str]) -> None:
        """向列表中添加新示例。

        Args:
            example: A dictionary with keys as input variables
                and values as their values.
        """
        self.examples.append(example)
        string_example = self.example_prompt.format(**example)
        self.example_text_lengths.append(self.get_text_length(string_example))

    async def aadd_example(self, example: dict[str, str]) -> None:
        """异步向列表中添加新示例。

        Args:
            example: A dictionary with keys as input variables
                and values as their values.
        """
        self.add_example(example)

    @model_validator(mode="after")
    def post_init(self) -> Self:
        """验证示例格式是否正确。"""
        if self.example_text_lengths:
            return self
        string_examples = [self.example_prompt.format(**eg) for eg in self.examples]
        self.example_text_lengths = [self.get_text_length(eg) for eg in string_examples]
        return self

    def select_examples(self, input_variables: dict[str, str]) -> list[dict[str, Any]]:
        """根据输入长度选择要使用的示例。

        Args:
            input_variables: A dictionary with keys as input variables
                and values as their values.

        Returns:
            要包含在提示词中的示例列表。
        """
        inputs = " ".join(input_variables.values())
        remaining_length = self.max_length - self.get_text_length(inputs)
        i = 0
        examples = []
        while remaining_length > 0 and i < len(self.examples):
            new_length = remaining_length - self.example_text_lengths[i]
            if new_length < 0:
                break
            examples.append(self.examples[i])
            remaining_length = new_length
            i += 1
        return examples

    async def aselect_examples(
        self, input_variables: dict[str, str]
    ) -> list[dict[str, Any]]:
        """异步根据输入长度选择要使用的示例。

        Args:
            input_variables: A dictionary with keys as input variables
                and values as their values.

        Returns:
            要包含在提示词中的示例列表。
        """
        return self.select_examples(input_variables)

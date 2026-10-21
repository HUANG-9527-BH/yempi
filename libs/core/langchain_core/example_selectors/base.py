"""用于在提示词中选择示例的接口。"""

from abc import ABC, abstractmethod
from typing import Any

from langchain_core.runnables import run_in_executor


class BaseExampleSelector(ABC):
    """用于在提示词中选择示例的接口。"""

    @abstractmethod
    def add_example(self, example: dict[str, str]) -> Any:
        """向存储中添加新示例。

        Args:
            example: A dictionary with keys as input variables
                and values as their values.

        Returns:
            任意返回值。
        """

    async def aadd_example(self, example: dict[str, str]) -> Any:
        """异步向存储中添加新示例。

        Args:
            example: A dictionary with keys as input variables
                and values as their values.

        Returns:
            任意返回值。
        """
        return await run_in_executor(None, self.add_example, example)

    @abstractmethod
    def select_examples(self, input_variables: dict[str, str]) -> list[dict[str, Any]]:
        """根据输入选择要使用的示例。

        Args:
            input_variables: A dictionary with keys as input variables
                and values as their values.

        Returns:
            一个示例列表。
        """

    async def aselect_examples(
        self, input_variables: dict[str, str]
    ) -> list[dict[str, Any]]:
        """异步根据输入选择要使用的示例。

        Args:
            input_variables: A dictionary with keys as input variables
                and values as their values.

        Returns:
            一个示例列表。
        """
        return await run_in_executor(None, self.select_examples, input_variables)

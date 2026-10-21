import warnings
from typing import Any

from pydantic import field_validator

from langchain_classic.base_memory import BaseMemory
from langchain_classic.memory.chat_memory import BaseChatMemory


class CombinedMemory(BaseMemory):
    """合并多个记忆的数据。"""

    memories: list[BaseMemory]
    """用于追踪所有应被访问的记忆。"""

    @field_validator("memories")
    @classmethod
    def _check_repeated_memory_variable(
        cls,
        value: list[BaseMemory],
    ) -> list[BaseMemory]:
        all_variables: set[str] = set()
        for val in value:
            overlap = all_variables.intersection(val.memory_variables)
            if overlap:
                msg = (
                    f"The same variables {overlap} are found in multiple"
                    "memory object, which is not allowed by CombinedMemory."
                )
                raise ValueError(msg)
            all_variables |= set(val.memory_variables)

        return value

    @field_validator("memories")
    @classmethod
    def check_input_key(cls, value: list[BaseMemory]) -> list[BaseMemory]:
        """检查记忆是否为 BaseChatMemory 类型，且输入键存在。"""
        for val in value:
            if isinstance(val, BaseChatMemory) and val.input_key is None:
                warnings.warn(
                    "When using CombinedMemory, "
                    "input keys should be so the input is known. "
                    f" Was not set on {val}",
                    stacklevel=5,
                )
        return value

    @property
    def memory_variables(self) -> list[str]:
        """此实例提供的所有记忆变量。"""
        """Collected from the all the linked memories."""

        memory_variables = []

        for memory in self.memories:
            memory_variables.extend(memory.memory_variables)

        return memory_variables

    def load_memory_variables(self, inputs: dict[str, Any]) -> dict[str, str]:
        """从子记忆中加载所有变量。"""
        memory_data: dict[str, Any] = {}

                # 从所有子记忆中收集变量
        for memory in self.memories:
            data = memory.load_memory_variables(inputs)
            for key, value in data.items():
                if key in memory_data:
                    msg = f"The variable {key} is repeated in the CombinedMemory."
                    raise ValueError(msg)
                memory_data[key] = value

        return memory_data

    def save_context(self, inputs: dict[str, Any], outputs: dict[str, str]) -> None:
        """为每个记忆保存本次会话的上下文。"""
                # 为所有子记忆保存上下文
        for memory in self.memories:
            memory.save_context(inputs, outputs)

    def clear(self) -> None:
        """为每个记忆清除本次会话的上下文。"""
        for memory in self.memories:
            memory.clear()

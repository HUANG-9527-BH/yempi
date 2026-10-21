from typing import Any

from langchain_classic.base_memory import BaseMemory


class ReadOnlySharedMemory(BaseMemory):
    """只读的内存包装器，无法更改。"""

    memory: BaseMemory

    @property
    def memory_variables(self) -> list[str]:
        """返回内存变量。"""
        return self.memory.memory_variables

    def load_memory_variables(self, inputs: dict[str, Any]) -> dict[str, str]:
        """从内存中加载内存变量。"""
        return self.memory.load_memory_variables(inputs)

    def save_context(self, inputs: dict[str, Any], outputs: dict[str, str]) -> None:
        """不应保存或更改任何内容。"""

    def clear(self) -> None:
        """无需清除，内存如同保险库。"""

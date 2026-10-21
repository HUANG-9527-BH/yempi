from typing import Any

from typing_extensions import override

from langchain_classic.base_memory import BaseMemory


class SimpleMemory(BaseMemory):
    """简单记忆。

    Simple memory for storing context or other information that shouldn't
    ever change between prompts.
    """

    memories: dict[str, Any] = {}

    @property
    @override
    def memory_variables(self) -> list[str]:
        return list(self.memories.keys())

    @override
    def load_memory_variables(self, inputs: dict[str, Any]) -> dict[str, str]:
        return self.memories

    def save_context(self, inputs: dict[str, Any], outputs: dict[str, str]) -> None:
        """不会保存或更改任何内容，我的记忆坚如磐石。"""

    def clear(self) -> None:
        """无需清除，我的记忆如同保险库。"""

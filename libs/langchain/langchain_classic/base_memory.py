"""Memory 维护链的状态，包含过去运行的上下文。

此模块包含 LangChain v0.0.x 中的内存抽象。

这些抽象现已弃用，将在 LangChain v1.0.0 中移除。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from langchain_core._api import deprecated
from langchain_core.load.serializable import Serializable
from langchain_core.runnables import run_in_executor
from pydantic import ConfigDict


@deprecated(
    since="0.3.3",
    removal="2.0.0",
    alternative="langchain.agents.create_agent",
    addendum=(
        "For agents that need to remember prior interactions, use "
        "`create_agent` with checkpointing or the `Store` API. See "
        "https://docs.langchain.com/oss/python/langchain/short-term-memory and "
        "https://docs.langchain.com/oss/python/langchain/long-term-memory"
    ),
)
class BaseMemory(Serializable, ABC):
    """链中内存的抽象基类。

    Memory refers to state in Chains. Memory can be used to store information about
        past executions of a Chain and inject that information into the inputs of
        future executions of the Chain. For example, for conversational Chains Memory
        can be used to store conversations and automatically add them to future model
        prompts so that the model has the necessary context to respond coherently to
        the latest input.

    Example:
        ```python
        class SimpleMemory(BaseMemory):
            memories: dict[str, Any] = dict()

            @property
            def memory_variables(self) -> list[str]:
                return list(self.memories.keys())

            def load_memory_variables(self, inputs: dict[str, Any]) -> dict[str, str]:
                return self.memories

            def save_context(
                self, inputs: dict[str, Any], outputs: dict[str, str]
            ) -> None:
                pass

            def clear(self) -> None:
                pass
        ```
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )

    @property
    @abstractmethod
    def memory_variables(self) -> list[str]:
        """此内存类将添加到链输入中的字符串键。"""

    @abstractmethod
    def load_memory_variables(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """根据链的文本输入返回键值对。

        Args:
            inputs: 链的输入。

        Returns:
            键值对的字典。
        """

    async def aload_memory_variables(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """异步返回给定链文本输入的键值对。

        Args:
            inputs: 链的输入。

        Returns:
            键值对的字典。
        """
        return await run_in_executor(None, self.load_memory_variables, inputs)

    @abstractmethod
    def save_context(self, inputs: dict[str, Any], outputs: dict[str, str]) -> None:
        """将此链运行的上下文保存到内存中。

        Args:
            inputs: 链的输入。
            outputs: 链的输出。
        """

    async def asave_context(
        self, inputs: dict[str, Any], outputs: dict[str, str]
    ) -> None:
        """异步将此链运行的上下文保存到内存中。

        Args:
            inputs: 链的输入。
            outputs: 链的输出。
        """
        await run_in_executor(None, self.save_context, inputs, outputs)

    @abstractmethod
    def clear(self) -> None:
        """清除内存内容。"""

    async def aclear(self) -> None:
        """异步清除内存内容。"""
        await run_in_executor(None, self.clear)

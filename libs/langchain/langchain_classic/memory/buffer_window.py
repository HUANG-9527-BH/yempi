from typing import Any

from langchain_core._api import deprecated
from langchain_core.messages import BaseMessage, get_buffer_string
from typing_extensions import override

from langchain_classic.memory.chat_memory import BaseChatMemory


@deprecated(
    since="0.3.1",
    removal="2.0.0",
    alternative="langchain.agents.create_agent",
    addendum=(
        "For agents that need to remember prior interactions, use "
        "`create_agent` with checkpointing or the `Store` API. See "
        "https://docs.langchain.com/oss/python/langchain/short-term-memory and "
        "https://docs.langchain.com/oss/python/langchain/long-term-memory"
    ),
)
class ConversationBufferWindowMemory(BaseChatMemory):
    """用于跟踪对话的最后 k 轮消息。

    如果对话中的消息数量超过最大保留消息数，最旧的消息将被丢弃。
    """

    human_prefix: str = "Human"
    ai_prefix: str = "AI"
    memory_key: str = "history"
    k: int = 5
    """要存储在缓冲区中的消息数量。"""

    @property
    def buffer(self) -> str | list[BaseMessage]:
        """内存的字符串缓冲区。"""
        return self.buffer_as_messages if self.return_messages else self.buffer_as_str

    @property
    def buffer_as_str(self) -> str:
        """当 return_messages 为 False 时，将缓冲区作为字符串暴露。"""
        messages = self.chat_memory.messages[-self.k * 2 :] if self.k > 0 else []
        return get_buffer_string(
            messages,
            human_prefix=self.human_prefix,
            ai_prefix=self.ai_prefix,
        )

    @property
    def buffer_as_messages(self) -> list[BaseMessage]:
        """当 return_messages 为 True 时，将缓冲区作为消息列表暴露。"""
        return self.chat_memory.messages[-self.k * 2 :] if self.k > 0 else []

    @property
    def memory_variables(self) -> list[str]:
        """始终返回消息变量的列表。"""
        return [self.memory_key]

    @override
    def load_memory_variables(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """返回历史缓冲区。"""
        return {self.memory_key: self.buffer}

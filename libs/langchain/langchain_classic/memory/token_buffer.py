from typing import Any

from langchain_core._api import deprecated
from langchain_core.language_models import BaseLanguageModel
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
class ConversationTokenBufferMemory(BaseChatMemory):
    """带有 token 限制的对话聊天内存。

    在对话中仅保留最近的消息，同时确保对话的总 token 数不超过指定限制。
    """

    human_prefix: str = "Human"
    ai_prefix: str = "AI"
    llm: BaseLanguageModel
    memory_key: str = "history"
    max_token_limit: int = 2000

    @property
    def buffer(self) -> Any:
        """字符串形式的内存缓冲区。"""
        return self.buffer_as_messages if self.return_messages else self.buffer_as_str

    @property
    def buffer_as_str(self) -> str:
        """当 return_messages 为 False 时，将缓冲区作为字符串暴露。"""
        return get_buffer_string(
            self.chat_memory.messages,
            human_prefix=self.human_prefix,
            ai_prefix=self.ai_prefix,
        )

    @property
    def buffer_as_messages(self) -> list[BaseMessage]:
        """当 return_messages 为 True 时，将缓冲区作为消息列表暴露。"""
        return self.chat_memory.messages

    @property
    def memory_variables(self) -> list[str]:
        """始终返回消息变量的列表。"""
        return [self.memory_key]

    @override
    def load_memory_variables(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """返回历史缓冲区。"""
        return {self.memory_key: self.buffer}

    def save_context(self, inputs: dict[str, Any], outputs: dict[str, str]) -> None:
        """将本次对话的上下文保存到缓冲区中，并进行修剪。"""
        super().save_context(inputs, outputs)
                # 如果缓冲区超过最大 token 限制，则进行修剪
        buffer = self.chat_memory.messages
        curr_buffer_length = self.llm.get_num_tokens_from_messages(buffer)
        if curr_buffer_length > self.max_token_limit:
            pruned_memory = []
            while curr_buffer_length > self.max_token_limit:
                pruned_memory.append(buffer.pop(0))
                curr_buffer_length = self.llm.get_num_tokens_from_messages(buffer)

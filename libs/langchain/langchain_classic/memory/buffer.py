from typing import Any

from langchain_core._api import deprecated
from langchain_core.messages import BaseMessage, get_buffer_string
from langchain_core.utils import pre_init
from typing_extensions import override

from langchain_classic.base_memory import BaseMemory
from langchain_classic.memory.chat_memory import BaseChatMemory
from langchain_classic.memory.utils import get_prompt_input_key


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
class ConversationBufferMemory(BaseChatMemory):
    """一种基础的记忆实现，仅存储对话历史。

    此实现将整个对话历史存储在内存中，不进行任何额外处理。

    注意：当对话历史过长，超出模型上下文窗口容量时，可能需要额外处理。
    """

    human_prefix: str = "Human"
    ai_prefix: str = "AI"
    memory_key: str = "history"

    @property
    def buffer(self) -> Any:
        """记忆的字符串缓冲区。"""
        return self.buffer_as_messages if self.return_messages else self.buffer_as_str

    async def abuffer(self) -> Any:
        """记忆的字符串缓冲区。"""
        return (
            await self.abuffer_as_messages()
            if self.return_messages
            else await self.abuffer_as_str()
        )

    def _buffer_as_str(self, messages: list[BaseMessage]) -> str:
        return get_buffer_string(
            messages,
            human_prefix=self.human_prefix,
            ai_prefix=self.ai_prefix,
        )

    @property
    def buffer_as_str(self) -> str:
        """当 return_messages 为 True 时，将缓冲区暴露为字符串。"""
        return self._buffer_as_str(self.chat_memory.messages)

    async def abuffer_as_str(self) -> str:
        """当 return_messages 为 True 时，将缓冲区暴露为字符串。"""
        messages = await self.chat_memory.aget_messages()
        return self._buffer_as_str(messages)

    @property
    def buffer_as_messages(self) -> list[BaseMessage]:
        """当 return_messages 为 False 时，将缓冲区暴露为消息列表。"""
        return self.chat_memory.messages

    async def abuffer_as_messages(self) -> list[BaseMessage]:
        """当 return_messages 为 False 时，将缓冲区暴露为消息列表。"""
        return await self.chat_memory.aget_messages()

    @property
    def memory_variables(self) -> list[str]:
        """始终返回记忆变量的列表。"""
        return [self.memory_key]

    @override
    def load_memory_variables(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """返回历史缓冲区。"""
        return {self.memory_key: self.buffer}

    @override
    async def aload_memory_variables(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """根据链的文本输入，返回键值对。"""
        buffer = await self.abuffer()
        return {self.memory_key: buffer}


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
class ConversationStringBufferMemory(BaseMemory):
    """一种基础的记忆实现，仅存储对话历史。

    此实现将整个对话历史存储在内存中，不进行任何额外处理。

    Equivalent to ConversationBufferMemory but tailored more specifically
    for string-based conversations rather than chat models.

    注意：当对话历史过长，超出模型上下文窗口容量时，可能需要额外处理。
    """

    human_prefix: str = "Human"
    ai_prefix: str = "AI"
    """用于 AI 生成回复的前缀。"""
    buffer: str = ""
    output_key: str | None = None
    input_key: str | None = None
    memory_key: str = "history"

    @pre_init
    def validate_chains(cls, values: dict) -> dict:
        """验证 return_messages 不为 True。"""
        if values.get("return_messages", False):
            msg = "return_messages must be False for ConversationStringBufferMemory"
            raise ValueError(msg)
        return values

    @property
    def memory_variables(self) -> list[str]:
        """始终返回记忆变量的列表。"""
        return [self.memory_key]

    @override
    def load_memory_variables(self, inputs: dict[str, Any]) -> dict[str, str]:
        """返回历史缓冲区。"""
        return {self.memory_key: self.buffer}

    async def aload_memory_variables(self, inputs: dict[str, Any]) -> dict[str, str]:
        """返回历史缓冲区。"""
        return self.load_memory_variables(inputs)

    def save_context(self, inputs: dict[str, Any], outputs: dict[str, str]) -> None:
        """将本次对话的上下文保存到缓冲区。"""
        if self.input_key is None:
            prompt_input_key = get_prompt_input_key(inputs, self.memory_variables)
        else:
            prompt_input_key = self.input_key
        if self.output_key is None:
            if len(outputs) != 1:
                msg = f"One output key expected, got {outputs.keys()}"
                raise ValueError(msg)
            output_key = next(iter(outputs.keys()))
        else:
            output_key = self.output_key
        human = f"{self.human_prefix}: " + inputs[prompt_input_key]
        ai = f"{self.ai_prefix}: " + outputs[output_key]
        self.buffer += f"\n{human}\n{ai}"

    async def asave_context(
        self,
        inputs: dict[str, Any],
        outputs: dict[str, str],
    ) -> None:
        """将本次对话的上下文保存到缓冲区。"""
        return self.save_context(inputs, outputs)

    def clear(self) -> None:
        """清除记忆内容。"""
        self.buffer = ""

    @override
    async def aclear(self) -> None:
        self.clear()

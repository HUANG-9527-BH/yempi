from typing import Any

from langchain_core._api import deprecated
from langchain_core.messages import BaseMessage, get_buffer_string
from langchain_core.utils import pre_init
from typing_extensions import override

from langchain_classic.memory.chat_memory import BaseChatMemory
from langchain_classic.memory.summary import SummarizerMixin


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
class ConversationSummaryBufferMemory(BaseChatMemory, SummarizerMixin):
    """用于存储对话记忆的缓冲区，带有摘要功能。

    Provides a running summary of the conversation together with the most recent
    messages in the conversation under the constraint that the total number of
    tokens in the conversation does not exceed a certain limit.
    """

    max_token_limit: int = 2000
    moving_summary_buffer: str = ""
    memory_key: str = "history"

    @property
    def buffer(self) -> str | list[BaseMessage]:
        """记忆的字符串缓冲区。"""
        return self.load_memory_variables({})[self.memory_key]

    async def abuffer(self) -> str | list[BaseMessage]:
        """异步记忆缓冲区。"""
        memory_variables = await self.aload_memory_variables({})
        return memory_variables[self.memory_key]

    @property
    def memory_variables(self) -> list[str]:
        """始终返回记忆变量列表。"""
        return [self.memory_key]

    @override
    def load_memory_variables(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """返回历史缓冲区。"""
        buffer = self.chat_memory.messages
        if self.moving_summary_buffer != "":
            first_messages: list[BaseMessage] = [
                self.summary_message_cls(content=self.moving_summary_buffer),
            ]
            buffer = first_messages + buffer
        if self.return_messages:
            final_buffer: Any = buffer
        else:
            final_buffer = get_buffer_string(
                buffer,
                human_prefix=self.human_prefix,
                ai_prefix=self.ai_prefix,
            )
        return {self.memory_key: final_buffer}

    @override
    async def aload_memory_variables(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """异步返回给定链输入文本的关键值对。"""
        buffer = await self.chat_memory.aget_messages()
        if self.moving_summary_buffer != "":
            first_messages: list[BaseMessage] = [
                self.summary_message_cls(content=self.moving_summary_buffer),
            ]
            buffer = first_messages + buffer
        if self.return_messages:
            final_buffer: Any = buffer
        else:
            final_buffer = get_buffer_string(
                buffer,
                human_prefix=self.human_prefix,
                ai_prefix=self.ai_prefix,
            )
        return {self.memory_key: final_buffer}

    @pre_init
    def validate_prompt_input_variables(cls, values: dict) -> dict:
        """验证提示词输入变量的一致性。"""
        prompt_variables = values["prompt"].input_variables
        expected_keys = {"summary", "new_lines"}
        if expected_keys != set(prompt_variables):
            msg = (
                "Got unexpected prompt input variables. The prompt expects "
                f"{prompt_variables}, but it should have {expected_keys}."
            )
            raise ValueError(msg)
        return values

    def save_context(self, inputs: dict[str, Any], outputs: dict[str, str]) -> None:
        """将本次对话的上下文保存到缓冲区。"""
        super().save_context(inputs, outputs)
        self.prune()

    async def asave_context(
        self,
        inputs: dict[str, Any],
        outputs: dict[str, str],
    ) -> None:
        """异步将本次对话的上下文保存到缓冲区。"""
        await super().asave_context(inputs, outputs)
        await self.aprune()

    def prune(self) -> None:
        """如果缓冲区超出最大 token 限制，则进行修剪。"""
        buffer = self.chat_memory.messages
        curr_buffer_length = self.llm.get_num_tokens_from_messages(buffer)
        if curr_buffer_length > self.max_token_limit:
            pruned_memory = []
            while curr_buffer_length > self.max_token_limit:
                pruned_memory.append(buffer.pop(0))
                curr_buffer_length = self.llm.get_num_tokens_from_messages(buffer)
            self.moving_summary_buffer = self.predict_new_summary(
                pruned_memory,
                self.moving_summary_buffer,
            )

    async def aprune(self) -> None:
        """如果缓冲区超出最大 token 限制，则异步进行修剪。"""
        buffer = self.chat_memory.messages
        curr_buffer_length = self.llm.get_num_tokens_from_messages(buffer)
        if curr_buffer_length > self.max_token_limit:
            pruned_memory = []
            while curr_buffer_length > self.max_token_limit:
                pruned_memory.append(buffer.pop(0))
                curr_buffer_length = self.llm.get_num_tokens_from_messages(buffer)
            self.moving_summary_buffer = await self.apredict_new_summary(
                pruned_memory,
                self.moving_summary_buffer,
            )

    def clear(self) -> None:
        """清除内存内容。"""
        super().clear()
        self.moving_summary_buffer = ""

    async def aclear(self) -> None:
        """异步清除内存内容。"""
        await super().aclear()
        self.moving_summary_buffer = ""

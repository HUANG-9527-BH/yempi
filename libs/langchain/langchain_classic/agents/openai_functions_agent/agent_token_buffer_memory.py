"""用于保存智能体输出和中间步骤的内存。"""

from typing import Any

from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import BaseMessage, get_buffer_string
from typing_extensions import override

from langchain_classic.agents.format_scratchpad import (
    format_to_openai_function_messages,
    format_to_tool_messages,
)
from langchain_classic.memory.chat_memory import BaseChatMemory


class AgentTokenBufferMemory(BaseChatMemory):
    """用于保存智能体输出和中间步骤的内存。

    Args:
        human_prefix: Prefix for human messages.
        ai_prefix: Prefix for AI messages.
        llm: Language model.
        memory_key: Key to save memory under.
        max_token_limit: Maximum number of tokens to keep in the buffer.
            Once the buffer exceeds this many tokens, the oldest
            messages will be pruned.
        return_messages: Whether to return messages.
        output_key: Key to save output under.
        intermediate_steps_key: Key to save intermediate steps under.
        format_as_tools: Whether to format as tools.
    """

    human_prefix: str = "Human"
    ai_prefix: str = "AI"
    llm: BaseLanguageModel
    memory_key: str = "history"
    max_token_limit: int = 12000
    """缓冲区中保留的最大 token 数量。
    一旦缓冲区超过此数量的 token，最旧的消息将被修剪。"""
    return_messages: bool = True
    output_key: str = "output"
    intermediate_steps_key: str = "intermediate_steps"
    format_as_tools: bool = False

    @property
    def buffer(self) -> list[BaseMessage]:
        """字符串形式的内存缓冲区。"""
        return self.chat_memory.messages

    @property
    def memory_variables(self) -> list[str]:
        """始终返回内存变量列表。"""
        return [self.memory_key]

    @override
    def load_memory_variables(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """返回历史缓冲区。

        Args:
            inputs: 智能体的输入。

        Returns:
            一个包含历史缓冲区的字典。
        """
        if self.return_messages:
            final_buffer: Any = self.buffer
        else:
            final_buffer = get_buffer_string(
                self.buffer,
                human_prefix=self.human_prefix,
                ai_prefix=self.ai_prefix,
            )
        return {self.memory_key: final_buffer}

    def save_context(self, inputs: dict[str, Any], outputs: dict[str, Any]) -> None:
        """将本次对话的上下文保存到缓冲区。已修剪。

        Args:
            inputs: 智能体的输入。
            outputs: 智能体的输出。
        """
        input_str, output_str = self._get_input_output(inputs, outputs)
        self.chat_memory.add_messages(input_str)  # type: ignore[arg-type]
        format_to_messages = (
            format_to_tool_messages
            if self.format_as_tools
            else format_to_openai_function_messages
        )
        steps = format_to_messages(outputs[self.intermediate_steps_key])
        for msg in steps:
            self.chat_memory.add_message(msg)
        self.chat_memory.add_messages(output_str)  # type: ignore[arg-type]
                # 如果缓冲区超过最大 token 限制，则修剪
        buffer = self.chat_memory.messages
        curr_buffer_length = self.llm.get_num_tokens_from_messages(buffer)
        if curr_buffer_length > self.max_token_limit:
            while curr_buffer_length > self.max_token_limit:
                buffer.pop(0)
                curr_buffer_length = self.llm.get_num_tokens_from_messages(buffer)

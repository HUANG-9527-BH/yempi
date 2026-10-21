from __future__ import annotations

from typing import Any

from langchain_core._api import deprecated
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import BaseMessage, SystemMessage, get_buffer_string
from langchain_core.prompts import BasePromptTemplate
from langchain_core.utils import pre_init
from pydantic import BaseModel
from typing_extensions import override

from langchain_classic.chains.llm import LLMChain
from langchain_classic.memory.chat_memory import BaseChatMemory
from langchain_classic.memory.prompt import SUMMARY_PROMPT


@deprecated(
    since="0.2.12",
    removal="2.0.0",
    addendum=(
        "For agents, summarize conversation history with `create_agent` and "
        "summarization middleware. See "
        "https://docs.langchain.com/oss/python/langchain/short-term-memory and "
        "https://docs.langchain.com/oss/python/langgraph/add-memory#summarize-messages"
    ),
)
class SummarizerMixin(BaseModel):
    """摘要生成器的混合类。"""

    human_prefix: str = "Human"
    ai_prefix: str = "AI"
    llm: BaseLanguageModel
    prompt: BasePromptTemplate = SUMMARY_PROMPT
    summary_message_cls: type[BaseMessage] = SystemMessage

    def predict_new_summary(
        self,
        messages: list[BaseMessage],
        existing_summary: str,
    ) -> str:
        """基于消息和现有摘要预测新的摘要。

        Args:
            messages: 需要摘要的消息列表。
            existing_summary: 用于构建的新摘要。

        Returns:
            一个新的摘要字符串。
        """
        new_lines = get_buffer_string(
            messages,
            human_prefix=self.human_prefix,
            ai_prefix=self.ai_prefix,
        )

        chain = LLMChain(llm=self.llm, prompt=self.prompt)
        return chain.predict(summary=existing_summary, new_lines=new_lines)

    async def apredict_new_summary(
        self,
        messages: list[BaseMessage],
        existing_summary: str,
    ) -> str:
        """基于消息和现有摘要预测新的摘要。

        Args:
            messages: 需要摘要的消息列表。
            existing_summary: 用于构建的新摘要。

        Returns:
            一个新的摘要字符串。
        """
        new_lines = get_buffer_string(
            messages,
            human_prefix=self.human_prefix,
            ai_prefix=self.ai_prefix,
        )

        chain = LLMChain(llm=self.llm, prompt=self.prompt)
        return await chain.apredict(summary=existing_summary, new_lines=new_lines)


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
class ConversationSummaryMemory(BaseChatMemory, SummarizerMixin):
    """持续总结对话历史。

    每次对话轮次后都会更新摘要。
    实现返回对话历史的摘要，
    可用于为模型提供上下文。
    """

    buffer: str = ""
    memory_key: str = "history"

    @classmethod
    def from_messages(
        cls,
        llm: BaseLanguageModel,
        chat_memory: BaseChatMessageHistory,
        *,
        summarize_step: int = 2,
        **kwargs: Any,
    ) -> ConversationSummaryMemory:
        """从消息列表创建 ConversationSummaryMemory。

        Args:
            llm: 用于摘要的语言模型。
            chat_memory: 需要摘要的聊天历史。
            summarize_step: 一次摘要的消息数量。
            **kwargs: 传递给类的额外关键字参数。

        Returns:
            包含已摘要历史的 ConversationSummaryMemory 实例。
        """
        obj = cls(llm=llm, chat_memory=chat_memory, **kwargs)
        for i in range(0, len(obj.chat_memory.messages), summarize_step):
            obj.buffer = obj.predict_new_summary(
                obj.chat_memory.messages[i : i + summarize_step],
                obj.buffer,
            )
        return obj

    @property
    def memory_variables(self) -> list[str]:
        """始终返回内存变量列表。"""
        return [self.memory_key]

    @override
    def load_memory_variables(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """返回历史缓冲区。"""
        if self.return_messages:
            buffer: Any = [self.summary_message_cls(content=self.buffer)]
        else:
            buffer = self.buffer
        return {self.memory_key: buffer}

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
        self.buffer = self.predict_new_summary(
            self.chat_memory.messages[-2:],
            self.buffer,
        )

    def clear(self) -> None:
        """清除内存内容。"""
        super().clear()
        self.buffer = ""

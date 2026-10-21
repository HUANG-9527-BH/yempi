"""用于测试目的的假聊天模型。"""

import asyncio
import re
import time
from collections.abc import AsyncIterator, Iterator
from typing import Any, Literal, cast

from typing_extensions import override

from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models.chat_models import BaseChatModel, SimpleChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.runnables import RunnableConfig


class FakeMessagesListChatModel(BaseChatModel):
    """用于测试目的的假聊天模型。"""

    responses: list[BaseMessage]
    """要按顺序**循环**的响应列表。"""
    sleep: float | None = None
    """每次响应之间的睡眠时间（秒）。"""
    i: int = 0
    """每次模型调用后内部递增。"""

    @override
    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        if self.sleep is not None:
            time.sleep(self.sleep)
        response = self.responses[self.i]
        if self.i < len(self.responses) - 1:
            self.i += 1
        else:
            self.i = 0
        generation = ChatGeneration(message=response)
        return ChatResult(generations=[generation])

    @property
    @override
    def _llm_type(self) -> str:
        return "fake-messages-list-chat-model"


class FakeListChatModelError(Exception):
    """用于测试目的的假错误。"""


class FakeListChatModel(SimpleChatModel):
    """用于测试目的的假聊天模型。"""

    responses: list[str]
    """要按顺序**循环**的响应列表。"""
    sleep: float | None = None
    i: int = 0
    """每次模型调用后内部递增。"""
    error_on_chunk_number: int | None = None
    """如果设置，在流式传输的指定块编号处抛出错误。"""

    @property
    @override
    def _llm_type(self) -> str:
        return "fake-list-chat-model"

    @override
    def _call(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> str:
        """返回列表中的下一个响应。

        如果已到末尾，则循环回到开头。
        """
        if self.sleep is not None:
            time.sleep(self.sleep)
        response = self.responses[self.i]
        if self.i < len(self.responses) - 1:
            self.i += 1
        else:
            self.i = 0
        return response

    @override
    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        response = self.responses[self.i]
        if self.i < len(self.responses) - 1:
            self.i += 1
        else:
            self.i = 0
        for i_c, c in enumerate(response):
            if self.sleep is not None:
                time.sleep(self.sleep)
            if (
                self.error_on_chunk_number is not None
                and i_c == self.error_on_chunk_number
            ):
                raise FakeListChatModelError

            chunk_position: Literal["last"] | None = (
                "last" if i_c == len(response) - 1 else None
            )
            yield ChatGenerationChunk(
                message=AIMessageChunk(content=c, chunk_position=chunk_position)
            )

    @override
    async def _astream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        response = self.responses[self.i]
        if self.i < len(self.responses) - 1:
            self.i += 1
        else:
            self.i = 0
        for i_c, c in enumerate(response):
            if self.sleep is not None:
                await asyncio.sleep(self.sleep)
            if (
                self.error_on_chunk_number is not None
                and i_c == self.error_on_chunk_number
            ):
                raise FakeListChatModelError
            chunk_position: Literal["last"] | None = (
                "last" if i_c == len(response) - 1 else None
            )
            yield ChatGenerationChunk(
                message=AIMessageChunk(content=c, chunk_position=chunk_position)
            )

    @property
    @override
    def _identifying_params(self) -> dict[str, Any]:
        return {"responses": self.responses}

    @override
        # 手动覆盖 batch 以保持批处理顺序且无并发
    def batch(
        self,
        inputs: list[Any],
        config: RunnableConfig | list[RunnableConfig] | None = None,
        *,
        return_exceptions: bool = False,
        **kwargs: Any,
    ) -> list[AIMessage]:
        if isinstance(config, list):
            return [
                self.invoke(m, c, **kwargs)
                for m, c in zip(inputs, config, strict=False)
            ]
        return [self.invoke(m, config, **kwargs) for m in inputs]

    @override
    async def abatch(
        self,
        inputs: list[Any],
        config: RunnableConfig | list[RunnableConfig] | None = None,
        *,
        return_exceptions: bool = False,
        **kwargs: Any,
    ) -> list[AIMessage]:
        if isinstance(config, list):
                        # 不要在此处使用异步迭代器，因为需要显式顺序
            return [
                await self.ainvoke(m, c, **kwargs)
                for m, c in zip(inputs, config, strict=False)
            ]
                # 不要在此处使用异步迭代器，因为需要显式顺序
        return [await self.ainvoke(m, config, **kwargs) for m in inputs]


class FakeChatModel(SimpleChatModel):
    """用于测试目的的假聊天模型包装器。"""

    @override
    def _call(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> str:
        return "fake response"

    @override
    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        output_str = "fake response"
        message = AIMessage(content=output_str)
        generation = ChatGeneration(message=message)
        return ChatResult(generations=[generation])

    @property
    def _llm_type(self) -> str:
        return "fake-chat-model"

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return {"key": "fake"}


class GenericFakeChatModel(BaseChatModel):
    """通用的假聊天模型，可用于测试聊天模型接口。

    * Chat model should be usable in both sync and async tests
    * Invokes `on_llm_new_token` to allow for testing of callback related code for new
        tokens.
    * Includes logic to break messages into message chunk to facilitate testing of
        streaming.

    """

    messages: Iterator[AIMessage | str]
    """获取消息的迭代器。

    此处可扩展以接受其他类型，如可调用对象 / 字典 / 字符串
    以在需要时使接口更通用。

    !!! note
        如果你想传入一个列表，可以使用 `iter` 将其转换为迭代器。

    !!! warning 流式传输尚未实现。我们应尝试在未来通过委托给 invoke
    并将结果输出拆分为消息块来实现。

    """

    @override
    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        message = next(self.messages)
        message_ = AIMessage(content=message) if isinstance(message, str) else message
        generation = ChatGeneration(message=message_)
        return ChatResult(generations=[generation])

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        chat_result = self._generate(
            messages, stop=stop, run_manager=run_manager, **kwargs
        )
        if not isinstance(chat_result, ChatResult):
            msg = (  # type: ignore[unreachable]
                f"Expected generate to return a ChatResult, "
                f"but got {type(chat_result)} instead."
            )
            raise ValueError(msg)  # noqa: TRY004

        message = chat_result.generations[0].message

        if not isinstance(message, AIMessage):
            msg = (
                f"Expected invoke to return an AIMessage, "
                f"but got {type(message)} instead."
            )
            raise ValueError(msg)  # noqa: TRY004

        content = message.content

        if content:
                        # 使用正则表达式按空白字符分割并捕获分组
                        # 以便在输出中保留空白字符。
            if not isinstance(content, str):
                msg = "Expected content to be a string."
                raise ValueError(msg)

            content_chunks = cast("list[str]", re.split(r"(\s)", content))

            for idx, token in enumerate(content_chunks):
                chunk = ChatGenerationChunk(
                    message=AIMessageChunk(content=token, id=message.id)
                )
                if (
                    idx == len(content_chunks) - 1
                    and isinstance(chunk.message, AIMessageChunk)
                    and not message.additional_kwargs
                ):
                    chunk.message.chunk_position = "last"
                if run_manager:
                    run_manager.on_llm_new_token(token, chunk=chunk)
                yield chunk

        if message.additional_kwargs:
            for key, value in message.additional_kwargs.items():
                                # 我们应该进一步将额外的 kwargs 拆分为块
                                # 函数调用的特殊情况
                if key == "function_call":
                    for fkey, fvalue in value.items():
                        if isinstance(fvalue, str):
                                                        # 按 `,` 拆分函数调用
                            fvalue_chunks = cast("list[str]", re.split(r"(,)", fvalue))
                            for fvalue_chunk in fvalue_chunks:
                                chunk = ChatGenerationChunk(
                                    message=AIMessageChunk(
                                        id=message.id,
                                        content="",
                                        additional_kwargs={
                                            "function_call": {fkey: fvalue_chunk}
                                        },
                                    )
                                )
                                if run_manager:
                                    run_manager.on_llm_new_token(
                                        "",
                                        chunk=chunk,  # 函数调用没有 token
                                    )
                                yield chunk
                        else:
                            chunk = ChatGenerationChunk(
                                message=AIMessageChunk(
                                    id=message.id,
                                    content="",
                                    additional_kwargs={"function_call": {fkey: fvalue}},
                                )
                            )
                            if run_manager:
                                run_manager.on_llm_new_token(
                                    "",
                                    chunk=chunk,  # 函数调用没有 token
                                )
                            yield chunk
                else:
                    chunk = ChatGenerationChunk(
                        message=AIMessageChunk(
                            id=message.id, content="", additional_kwargs={key: value}
                        )
                    )
                    if run_manager:
                        run_manager.on_llm_new_token(
                            "",
                            chunk=chunk,  # 函数调用没有 token
                        )
                    yield chunk

    @property
    def _llm_type(self) -> str:
        return "generic-fake-chat-model"


class ParrotFakeChatModel(BaseChatModel):
    """通用的假聊天模型，可用于测试聊天模型接口。

    * 聊天模型应在同步和异步测试中均可使用

    """

    @override
    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        if not messages:
            msg = "messages list cannot be empty."
            raise ValueError(msg)
        return ChatResult(generations=[ChatGeneration(message=messages[-1])])

    @property
    def _llm_type(self) -> str:
        return "parrot-fake-chat-model"

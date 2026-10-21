"""LangChain 的基础回调处理器。"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Sequence
    from uuid import UUID

    from langchain_protocol.protocol import MessagesData
    from tenacity import RetryCallState
    from typing_extensions import Self

    from langchain_core.agents import AgentAction, AgentFinish
    from langchain_core.documents import Document
    from langchain_core.messages import BaseMessage
    from langchain_core.outputs import ChatGenerationChunk, GenerationChunk, LLMResult

_LOGGER = logging.getLogger(__name__)


class RetrieverManagerMixin:
    """用于 `Retriever` 回调的混合类。"""

    def on_retriever_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> Any:
        """当 `Retriever` 出错时运行。

        Args:
            error: 发生的错误。
            run_id: 当前运行的 ID。
            parent_run_id: 父运行的 ID。
            **kwargs: 额外的关键字参数。
        """

    def on_retriever_end(
        self,
        documents: Sequence[Document],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> Any:
        """当 `Retriever` 运行结束时运行。

        Args:
            documents: 检索到的文档。
            run_id: 当前运行的 ID。
            parent_run_id: 父运行的 ID。
            **kwargs: 额外的关键字参数。
        """


class LLMManagerMixin:
    """用于 LLM 回调的混合类。"""

    def on_llm_new_token(
        self,
        token: str | list[str | dict[str, Any]],
        *,
        chunk: GenerationChunk | ChatGenerationChunk | None = None,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        **kwargs: Any,
    ) -> Any:
        """在新输出 token 生成时运行。

        仅在启用流式传输时可用。

        适用于聊天模型和非聊天模型（旧版文本补全 LLM）。

        Args:
            token: 新的 token，或内容块列表。
            chunk: 新生成的块，包含内容和其他信息。
            run_id: 当前运行的 ID。
            parent_run_id: 父运行的 ID。
            tags: 标签。
            **kwargs: 额外的关键字参数。
        """

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        **kwargs: Any,
    ) -> Any:
        """当 LLM 运行结束时运行。

        Args:
            response: 生成的响应。
            run_id: 当前运行的 ID。
            parent_run_id: 父运行的 ID。
            tags: 标签。
            **kwargs: 额外的关键字参数。
        """

    def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        **kwargs: Any,
    ) -> Any:
        """当 LLM 出错时运行。

        Args:
            error: 发生的错误。
            run_id: 当前运行的 ID。
            parent_run_id: 父运行的 ID。
            tags: 标签。
            **kwargs: 额外的关键字参数。
        """

    def on_stream_event(
        self,
        event: MessagesData,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        **kwargs: Any,
    ) -> Any:
        """在 `stream_events(version="v3")` 的每个协议事件上触发。

        也适用于异步等价方法
        (`astream_events(version="v3")`)。

        每个 `MessagesData` 事件触发一次 —— `message-start`，每个块的
        `content-block-start` / `content-block-delta` / `content-block-finish`，以及
        `message-finish`。类似于 v1 流式中的
        `on_llm_new_token`，但粒度为事件而非块：单个块可能对应多个事件（例如一个
        `content-block-start` 加上其首个
        `content-block-delta`），且生命周期边界是明确的。

        无论提供方通过 `_stream_chat_model_events`
        原生发出事件，还是通过块到事件的兼容桥接方式，都会统一触发。观察者看到的事件流始终一致，与底层模型如何生成输出无关。

        Not fired from v1 `stream()` / `astream()`; for those, keep using
        `on_llm_new_token`. Purely additive — `on_chat_model_start`,
        `on_llm_end`, and `on_llm_error` still fire around a v2 call as
        they do around a v1 call.

        Args:
            event: 协议事件。
            run_id: 当前运行的 ID。
            parent_run_id: 父运行的 ID。
            tags: 标签。
            **kwargs: 额外的关键字参数。
        """


class ChainManagerMixin:
    """用于链回调的混合类。"""

    def on_chain_end(
        self,
        outputs: dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> Any:
        """链运行结束时调用。

        Args:
            outputs: 链的输出。
            run_id: 当前运行的 ID。
            parent_run_id: 父运行的 ID。
            **kwargs: 额外的关键字参数。
        """

    def on_chain_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> Any:
        """链出错时调用。

        Args:
            error: 发生的错误。
            run_id: 当前运行的 ID。
            parent_run_id: 父运行的 ID。
            **kwargs: 额外的关键字参数。
        """

    def on_agent_action(
        self,
        action: AgentAction,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> Any:
        """智能体执行动作时调用。

        Args:
            action: 智能体动作。
            run_id: 当前运行的 ID。
            parent_run_id: 父运行的 ID。
            **kwargs: 额外的关键字参数。
        """

    def on_agent_finish(
        self,
        finish: AgentFinish,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> Any:
        """智能体运行结束时调用。

        Args:
            finish: 智能体完成状态。
            run_id: 当前运行的 ID。
            parent_run_id: 父运行的 ID。
            **kwargs: 额外的关键字参数。
        """


class ToolManagerMixin:
    """用于工具回调的混合类。"""

    def on_tool_end(
        self,
        output: Any,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> Any:
        """工具运行结束时调用。

        Args:
            output: 工具的输出。
            run_id: 当前运行的 ID。
            parent_run_id: 父运行的 ID。
            **kwargs: 额外的关键字参数。
        """

    def on_tool_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> Any:
        """工具出错时调用。

        Args:
            error: 发生的错误。
            run_id: 当前运行的 ID。
            parent_run_id: 父运行的 ID。
            **kwargs: 额外的关键字参数。
        """


class CallbackManagerMixin:
    """用于回调管理器的混合类。"""

    def on_llm_start(
        self,
        serialized: dict[str, Any],
        prompts: list[str],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        """大语言模型开始运行时调用。

        !!! warning

            This method is called for non-chat models (regular text completion LLMs). If
            you're implementing a handler for a chat model, you should use
            `on_chat_model_start` instead.

        Args:
            serialized: 序列化的 LLM。
            prompts: 提示词。
            run_id: 当前运行的 ID。
            parent_run_id: 父运行的 ID。
            tags: 标签。
            metadata: 元数据。
            **kwargs: 额外的关键字参数。
        """

    def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[BaseMessage]],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        """聊天模型开始运行时调用。

        !!! warning

            This method is called for chat models. If you're implementing a handler for
            a non-chat model, you should use `on_llm_start` instead.

        !!! note

            When overriding this method, the signature **must** include the two
            required positional arguments `serialized` and `messages`.  Avoid
            using `*args` in your override — doing so causes an `IndexError`
            in the fallback path when the callback system converts `messages`
            to prompt strings for `on_llm_start`.  Always declare the
            signature explicitly:

            .. code-block:: python

                def on_chat_model_start(
                    self,
                    serialized: dict[str, Any],
                    messages: list[list[BaseMessage]],
                    **kwargs: Any,
                ) -> None:
                    raise NotImplementedError  # triggers fallback to on_llm_start

        Args:
            serialized: 已序列化的聊天模型。 messages:
            消息列表。必须是消息列表的列表——这是必需的位置参数，任何重写都必须包含此参数。
            run_id: 当前运行的 ID。 parent_run_id: 父运行的 ID。 tags: 标签。 metadata:
            元数据。 **kwargs: 额外的关键字参数。
        """
                # NotImplementedError 是故意抛出的
                # 如果抛出此异常，回调处理器将回退到 on_llm_start
        msg = f"{self.__class__.__name__} does not implement `on_chat_model_start`"
        raise NotImplementedError(msg)

    def on_retriever_start(
        self,
        serialized: dict[str, Any],
        query: str,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        """当 `Retriever` 开始运行时调用。

        Args:
            serialized: 已序列化的 `Retriever`。
            query: 查询。
            run_id: 当前运行的 ID。
            parent_run_id: 父运行的 ID。
            tags: 标签。
            metadata: 元数据。
            **kwargs: 额外的关键字参数。
        """

    def on_chain_start(
        self,
        serialized: dict[str, Any],
        inputs: dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        """当链开始运行时调用。

        Args:
            serialized: 已序列化的链。
            inputs: 输入。
            run_id: 当前运行的 ID。
            parent_run_id: 父运行的 ID。
            tags: 标签。
            metadata: 元数据。
            **kwargs: 额外的关键字参数。
        """

    def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        inputs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        """当工具开始运行时调用。

        Args:
            serialized: 已序列化的链。
            input_str: 输入字符串。
            run_id: 当前运行的 ID。
            parent_run_id: 父运行的 ID。
            tags: 标签。
            metadata: 元数据。
            inputs: 输入。
            **kwargs: 额外的关键字参数。
        """


class RunManagerMixin:
    """用于运行管理器的混合类。"""

    def on_text(
        self,
        text: str,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> Any:
        """对任意文本执行操作。

        Args:
            text: 文本。
            run_id: 当前运行的 ID。
            parent_run_id: 父运行的 ID。
            **kwargs: 额外的关键字参数。
        """

    def on_retry(
        self,
        retry_state: RetryCallState,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> Any:
        """在重试事件上执行。

        Args:
            retry_state: 重试状态。
            run_id: 当前运行的 ID。
            parent_run_id: 父运行的 ID。
            **kwargs: 额外的关键字参数。
        """

    def on_custom_event(
        self,
        name: str,
        data: Any,
        *,
        run_id: UUID,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        """重写以定义自定义事件的处理器。

        Args:
            name: 自定义事件的名称。
            data: 自定义事件的数据。

                Format will match the format specified by the user.
            run_id: The ID of the run.
            tags: The tags associated with the custom event (includes inherited tags).
            metadata: The metadata associated with the custom event (includes inherited
                metadata).
        """


class BaseCallbackHandler(
    LLMManagerMixin,
    ChainManagerMixin,
    ToolManagerMixin,
    RetrieverManagerMixin,
    CallbackManagerMixin,
    RunManagerMixin,
):
    """基础回调处理器。"""

    raise_error: bool = False
    """如果发生异常是否抛出错误。"""

    run_inline: bool = False
    """是否内联运行回调。"""

    @property
    def ignore_llm(self) -> bool:
        """是否忽略 LLM 回调。"""
        return False

    @property
    def ignore_retry(self) -> bool:
        """是否忽略重试回调。"""
        return False

    @property
    def ignore_chain(self) -> bool:
        """是否忽略链回调。"""
        return False

    @property
    def ignore_agent(self) -> bool:
        """是否忽略智能体回调。"""
        return False

    @property
    def ignore_retriever(self) -> bool:
        """是否忽略检索器回调。"""
        return False

    @property
    def ignore_chat_model(self) -> bool:
        """是否忽略聊天模型回调。"""
        return False

    @property
    def ignore_custom_event(self) -> bool:
        """忽略自定义事件。"""
        return False


class AsyncCallbackHandler(BaseCallbackHandler):
    """基础异步回调处理器。"""

    async def on_llm_start(
        self,
        serialized: dict[str, Any],
        prompts: list[str],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """当模型开始运行时调用。

        !!! warning

            This method is called for non-chat models (regular text completion LLMs). If
            you're implementing a handler for a chat model, you should use
            `on_chat_model_start` instead.

        Args:
            serialized: 已序列化的 LLM。
            prompts: 提示词。
            run_id: 当前运行的 ID。
            parent_run_id: 父运行的 ID。
            tags: 标签。
            metadata: 元数据。
            **kwargs: 额外的关键字参数。
        """

    async def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[BaseMessage]],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        """智能体开始运行时触发。

        !!! warning

            This method is called for chat models. If you're implementing a handler for
            a non-chat model, you should use `on_llm_start` instead.

        !!! note

            When overriding this method, the signature **must** include the two
            required positional arguments `serialized` and `messages`.  Avoid
            using `*args` in your override — doing so causes an `IndexError`
            in the fallback path when the callback system converts `messages`
            to prompt strings for `on_llm_start`.  Always declare the
            signature explicitly:

            .. code-block:: python

                async def on_chat_model_start(
                    self,
                    serialized: dict[str, Any],
                    messages: list[list[BaseMessage]],
                    **kwargs: Any,
                ) -> None:
                    raise NotImplementedError  # 触发回退到 on_llm_start

        Args:
            serialized: 已序列化的聊天模型。 messages:
            消息。必须是消息列表的列表——这是必需的位置参数，任何重写都必须存在。
            run_id: 当前运行的 ID。 parent_run_id: 父运行的 ID。 tags: 标签。 metadata:
            元数据。 **kwargs: 额外的关键字参数。
        """
                # NotImplementedError 是故意抛出的
                # 如果抛出此异常，回调处理器将回退到 on_llm_start
        msg = f"{self.__class__.__name__} does not implement `on_chat_model_start`"
        raise NotImplementedError(msg)

    async def on_llm_new_token(
        self,
        token: str | list[str | dict[str, Any]],
        *,
        chunk: GenerationChunk | ChatGenerationChunk | None = None,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        """在生成新输出 token 时触发。仅在启用流式传输时可用。

        适用于聊天模型和非聊天模型（旧版文本补全 LLM）。

        Args:
            token: 新的 token，或内容块列表。
            chunk: 新生成的片段，包含内容和其他信息。
            run_id: 当前运行的 ID。
            parent_run_id: 父运行的 ID。
            tags: 标签。
            **kwargs: 额外的关键字参数。
        """

    async def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        """模型结束运行时触发。

        Args:
            response: 生成的响应。
            run_id: 当前运行的 ID。
            parent_run_id: 父运行的 ID。
            tags: 标签。
            **kwargs: 额外的关键字参数。
        """

    async def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        """LLM 出错时触发。

        Args:
            error: 发生的错误。
            run_id: 当前运行的 ID。
            parent_run_id: 父运行的 ID。
            tags: 标签。
            **kwargs: 额外的关键字参数。

                - response (LLMResult): 在错误发生前生成的响应。
        """

    async def on_stream_event(
        self,
        event: MessagesData,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        """在 `astream_events(version="v3")` 产生的每个协议事件上运行。

        See :meth:`LLMManagerMixin.on_stream_event` for the full contract.
        Fires once per `MessagesData` event at event granularity, uniformly
        across native and compat-bridge providers, and is purely additive
        to the existing `on_chat_model_start` / `on_llm_end` /
        `on_llm_error` callbacks.

        Args:
            event: 协议事件。
            run_id: 当前运行的 ID。
            parent_run_id: 父运行的 ID。
            tags: 标签。
            **kwargs: 额外的关键字参数。
        """

    async def on_chain_start(
        self,
        serialized: dict[str, Any],
        inputs: dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """链开始运行时运行。

        Args:
            serialized: 序列化的链。
            inputs: 输入。
            run_id: 当前运行的 ID。
            parent_run_id: 父运行的 ID。
            tags: 标签。
            metadata: 元数据。
            **kwargs: 额外的关键字参数。
        """

    async def on_chain_end(
        self,
        outputs: dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        """链结束运行时运行。

        Args:
            outputs: 链的输出。
            run_id: 当前运行的 ID。
            parent_run_id: 父运行的 ID。
            tags: 标签。
            **kwargs: 额外的关键字参数。
        """

    async def on_chain_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        """链出错时运行。

        Args:
            error: 发生的错误。
            run_id: 当前运行的 ID。
            parent_run_id: 父运行的 ID。
            tags: 标签。
            **kwargs: 额外的关键字参数。
        """

    async def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        inputs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """工具开始运行时运行。

        Args:
            serialized: 序列化的工具。
            input_str: 输入字符串。
            run_id: 当前运行的 ID。
            parent_run_id: 父运行的 ID。
            tags: 标签。
            metadata: 元数据。
            inputs: 输入。
            **kwargs: 额外的关键字参数。
        """

    async def on_tool_end(
        self,
        output: Any,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        """工具结束运行时运行。

        Args:
            output: 工具的输出。
            run_id: 当前运行的 ID。
            parent_run_id: 父运行的 ID。
            tags: 标签。
            **kwargs: 额外的关键字参数。
        """

    async def on_tool_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        """工具出错时运行。

        Args:
            error: 发生的错误。
            run_id: 当前运行的 ID。
            parent_run_id: 父运行的 ID。
            tags: 标签。
            **kwargs: 额外的关键字参数。
        """

    async def on_text(
        self,
        text: str,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        """在任意文本上运行。

        Args:
            text: 文本。
            run_id: 当前运行的 ID。
            parent_run_id: 父运行的 ID。
            tags: 标签。
            **kwargs: 额外的关键字参数。
        """

    async def on_retry(
        self,
        retry_state: RetryCallState,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> Any:
        """在重试事件上运行。

        Args:
            retry_state: 重试状态。
            run_id: 当前运行的 ID。
            parent_run_id: 父运行的 ID。
            **kwargs: 额外的关键字参数。
        """

    async def on_agent_action(
        self,
        action: AgentAction,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        """在智能体动作上运行。

        Args:
            action: 智能体动作。
            run_id: 当前运行的 ID。
            parent_run_id: 父运行的 ID。
            tags: 标签。
            **kwargs: 额外的关键字参数。
        """

    async def on_agent_finish(
        self,
        finish: AgentFinish,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        """在智能体端运行。

        Args:
            finish: 智能体完成。
            run_id: 当前运行的 ID。
            parent_run_id: 父运行的 ID。
            tags: 标签。
            **kwargs: 额外的关键字参数。
        """

    async def on_retriever_start(
        self,
        serialized: dict[str, Any],
        query: str,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """在检索器开始时运行。

        Args:
            serialized: 序列化的检索器。
            query: 查询。
            run_id: 当前运行的 ID。
            parent_run_id: 父运行的 ID。
            tags: 标签。
            metadata: 元数据。
            **kwargs: 额外的关键字参数。
        """

    async def on_retriever_end(
        self,
        documents: Sequence[Document],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        """在检索器结束时运行。

        Args:
            documents: 检索到的文档。
            run_id: 当前运行的 ID。
            parent_run_id: 父运行的 ID。
            tags: 标签。
            **kwargs: 额外的关键字参数。
        """

    async def on_retriever_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        """在检索器出错时运行。

        Args:
            error: 发生的错误。
            run_id: 当前运行的 ID。
            parent_run_id: 父运行的 ID。
            tags: 标签。
            **kwargs: 额外的关键字参数。
        """

    async def on_custom_event(
        self,
        name: str,
        data: Any,
        *,
        run_id: UUID,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """重写以定义自定义事件的处理器。

        Args:
            name: 自定义事件的名称。
            data: 自定义事件的数据。

                格式将与用户指定的格式匹配。
            run_id: 运行的 ID。
            tags: 与自定义事件关联的标签（包含继承的标签）。
            metadata: 与自定义事件关联的元数据（包含继承的
                元数据）。
        """


class BaseCallbackManager(CallbackManagerMixin):
    """基础回调管理器。"""

    def __init__(
        self,
        handlers: list[BaseCallbackHandler],
        inheritable_handlers: list[BaseCallbackHandler] | None = None,
        parent_run_id: UUID | None = None,
        *,
        tags: list[str] | None = None,
        inheritable_tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        inheritable_metadata: dict[str, Any] | None = None,
    ) -> None:
        """初始化回调管理器。

        Args:
            handlers: 处理器。
            inheritable_handlers: 可继承的处理器。
            parent_run_id: 父运行 ID。
            tags: 标签。
            inheritable_tags: 可继承的标签。
            metadata: 元数据。
            inheritable_metadata: 可继承的元数据。
        """
        self.handlers: list[BaseCallbackHandler] = handlers
        self.inheritable_handlers: list[BaseCallbackHandler] = (
            inheritable_handlers or []
        )
        self.parent_run_id: UUID | None = parent_run_id
        self.tags = tags or []
        self.inheritable_tags = inheritable_tags or []
        self.metadata = metadata or {}
        self.inheritable_metadata = inheritable_metadata or {}

    def copy(self) -> Self:
        """返回回调管理器的副本。"""
        return self.__class__(
            handlers=self.handlers.copy(),
            inheritable_handlers=self.inheritable_handlers.copy(),
            parent_run_id=self.parent_run_id,
            tags=self.tags.copy(),
            inheritable_tags=self.inheritable_tags.copy(),
            metadata=self.metadata.copy(),
            inheritable_metadata=self.inheritable_metadata.copy(),
        )

    def merge(self, other: BaseCallbackManager) -> Self:
        """将回调管理器与另一个回调管理器合并。

        可在子类中重写。

        主要用于内部 `merge_configs` 中。

        Returns:
            与当前对象类型相同的合并后的回调管理器。

        Example:
            ```python
            # Merging two callback managers`
            from langchain_core.callbacks.manager import (
                CallbackManager,
                trace_as_chain_group,
            )
            from langchain_core.callbacks.stdout import StdOutCallbackHandler

            manager = CallbackManager(handlers=[StdOutCallbackHandler()], tags=["tag2"])
            with trace_as_chain_group("My Group Name", tags=["tag1"]) as group_manager:
                merged_manager = group_manager.merge(manager)
                print(merged_manager.handlers)
                # [
                #    <langchain_core.callbacks.stdout.StdOutCallbackHandler object at ...>,
                #    <langchain_core.callbacks.streaming_stdout.StreamingStdOutCallbackHandler object at ...>,
                # ]

                print(merged_manager.tags)
                #    ['tag2', 'tag1']
            ```
        """  # noqa: E501
                # 分别合并 handlers 和 inheritable_handlers，使用集合
                # 以去重（不保证顺序）
        combined_handlers = list(set(self.handlers) | set(other.handlers))
        combined_inheritable = list(
            set(self.inheritable_handlers) | set(other.inheritable_handlers)
        )

        return self.__class__(
            parent_run_id=self.parent_run_id or other.parent_run_id,
            handlers=combined_handlers,
            inheritable_handlers=combined_inheritable,
            tags=list(set(self.tags + other.tags)),
            inheritable_tags=list(set(self.inheritable_tags + other.inheritable_tags)),
            metadata={
                **self.metadata,
                **other.metadata,
            },
            inheritable_metadata={
                **self.inheritable_metadata,
                **other.inheritable_metadata,
            },
        )

    @property
    def is_async(self) -> bool:
        """回调管理器是否为异步。"""
        return False

    def add_handler(
        self,
        handler: BaseCallbackHandler,
        inherit: bool = True,  # noqa: FBT001,FBT002
    ) -> None:
        """向回调管理器添加处理器。

        Args:
            handler: 要添加的处理器。
            inherit: 是否继承该处理器。
        """
        if handler not in self.handlers:
            self.handlers.append(handler)
        if inherit and handler not in self.inheritable_handlers:
            self.inheritable_handlers.append(handler)

    def remove_handler(self, handler: BaseCallbackHandler) -> None:
        """从回调管理器中移除处理器。

        Args:
            handler: 要移除的处理器。
        """
        if handler in self.handlers:
            self.handlers.remove(handler)
        if handler in self.inheritable_handlers:
            self.inheritable_handlers.remove(handler)

    def set_handlers(
        self,
        handlers: list[BaseCallbackHandler],
        inherit: bool = True,  # noqa: FBT001,FBT002
    ) -> None:
        """将处理器设置为回调管理器上唯一的处理器。

        Args:
            handlers: 要设置的处理器。
            inherit: 是否继承这些处理器。
        """
        self.handlers = []
        self.inheritable_handlers = []
        for handler in handlers:
            self.add_handler(handler, inherit=inherit)

    def set_handler(
        self,
        handler: BaseCallbackHandler,
        inherit: bool = True,  # noqa: FBT001,FBT002
    ) -> None:
        """将处理器设置为回调管理器上唯一的处理器。

        Args:
            handler: 要设置的处理器。
            inherit: 是否继承该处理器。
        """
        self.set_handlers([handler], inherit=inherit)

    def add_tags(
        self,
        tags: list[str],
        inherit: bool = True,  # noqa: FBT001,FBT002
    ) -> None:
        """向回调管理器添加标签。

        Args:
            tags: 要添加的标签。
            inherit: 是否继承标签。
        """
        for tag in tags:
            if tag in self.tags:
                self.remove_tags([tag])
        self.tags.extend(tags)
        if inherit:
            self.inheritable_tags.extend(tags)

    def remove_tags(self, tags: list[str]) -> None:
        """从回调管理器中移除标签。

        Args:
            tags: 要移除的标签。
        """
        for tag in tags:
            if tag in self.tags:
                self.tags.remove(tag)
            if tag in self.inheritable_tags:
                self.inheritable_tags.remove(tag)

    def add_metadata(
        self,
        metadata: dict[str, Any],
        inherit: bool = True,  # noqa: FBT001,FBT002
    ) -> None:
        """向回调管理器添加元数据。

        Args:
            metadata: 要添加的元数据。
            inherit: 是否继承元数据。
        """
        from langchain_core.runnables.config import (  # noqa: PLC0415
            _merge_metadata_dicts,
        )

        self.metadata = _merge_metadata_dicts(self.metadata, metadata)
        if inherit:
            self.inheritable_metadata = _merge_metadata_dicts(
                self.inheritable_metadata, metadata
            )

    def remove_metadata(self, keys: list[str]) -> None:
        """从回调管理器中移除元数据。

        Args:
            keys: 要移除的键。
        """
        for key in keys:
            self.metadata.pop(key, None)
            self.inheritable_metadata.pop(key, None)


Callbacks = list[BaseCallbackHandler] | BaseCallbackManager | None

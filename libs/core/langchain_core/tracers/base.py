"""追踪运行的基本接口。"""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import (
    TYPE_CHECKING,
    Any,
)

from typing_extensions import override

from langchain_core.callbacks.base import AsyncCallbackHandler, BaseCallbackHandler
from langchain_core.exceptions import TracerException  # noqa: F401
from langchain_core.tracers.core import _TracerCore

if TYPE_CHECKING:
    from collections.abc import Sequence
    from uuid import UUID

    from tenacity import RetryCallState

    from langchain_core.documents import Document
    from langchain_core.messages import BaseMessage
    from langchain_core.outputs import ChatGenerationChunk, GenerationChunk, LLMResult
    from langchain_core.tracers.schemas import Run

logger = logging.getLogger(__name__)


class BaseTracer(_TracerCore, BaseCallbackHandler, ABC):
    """追踪器的基本接口。"""

    @abstractmethod
    def _persist_run(self, run: Run) -> None:
        """持久化一个运行。"""

    def _start_trace(self, run: Run) -> None:
        """为一个运行启动追踪。"""
        super()._start_trace(run)
        self._on_run_create(run)

    def _end_trace(self, run: Run) -> None:
        """结束一个运行的追踪。"""
        if not run.parent_run_id:
            self._persist_run(run)
        self.run_map.pop(str(run.id))
                # 如果此运行的父运行是从外部追踪上下文注入的
                # (例如 langsmith @traceable)，则减少其子引用计数并
                # 在最后一个子运行完成后，将其从 run_map 中移除。
        parent_id = str(run.parent_run_id) if run.parent_run_id else None
        if parent_id and parent_id in self._external_run_ids:
            self._external_run_ids[parent_id] -= 1
            if self._external_run_ids[parent_id] <= 0:
                self.run_map.pop(parent_id, None)
                del self._external_run_ids[parent_id]
        self._on_run_update(run)

    def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[BaseMessage]],
        *,
        run_id: UUID,
        tags: list[str] | None = None,
        parent_run_id: UUID | None = None,
        metadata: dict[str, Any] | None = None,
        name: str | None = None,
        **kwargs: Any,
    ) -> Run:
        """为一个聊天模型运行启动追踪。

        Note:
            命名可能令人困惑：存在 `on_chat_model_start`，但没有对应的
            `on_chat_model_end` 回调。聊天模型的完成通过 `on_llm_end` / `_on_llm_end`
            路由，这些回调与文本 LLM 运行共享。

        Args:
            serialized: 序列化的模型。
            messages: 用于开始聊天的消息。
            run_id: 运行 ID。
            tags: 运行的标签。
            parent_run_id: 父运行 ID。
            metadata: 运行的元数据。
            name: 运行的名称。
            **kwargs: 额外参数。

        Returns:
            运行。
        """
        chat_model_run = self._create_chat_model_run(
            serialized=serialized,
            messages=messages,
            run_id=run_id,
            parent_run_id=parent_run_id,
            tags=tags,
            metadata=metadata,
            name=name,
            **kwargs,
        )
        self._start_trace(chat_model_run)
        self._on_chat_model_start(chat_model_run)
        return chat_model_run

    def on_llm_start(
        self,
        serialized: dict[str, Any],
        prompts: list[str],
        *,
        run_id: UUID,
        tags: list[str] | None = None,
        parent_run_id: UUID | None = None,
        metadata: dict[str, Any] | None = None,
        name: str | None = None,
        **kwargs: Any,
    ) -> Run:
        """为一个 LLM 运行启动追踪。

        Args:
            serialized: 序列化的模型。
            prompts: 用于启动 LLM 的提示词。
            run_id: 运行 ID。
            tags: 运行的标签。
            parent_run_id: 父运行 ID。
            metadata: 运行的元数据。
            name: 运行的名称。
            **kwargs: 额外参数。

        Returns:
            运行。
        """
        llm_run = self._create_llm_run(
            serialized=serialized,
            prompts=prompts,
            run_id=run_id,
            parent_run_id=parent_run_id,
            tags=tags,
            metadata=metadata,
            name=name,
            **kwargs,
        )
        self._start_trace(llm_run)
        self._on_llm_start(llm_run)
        return llm_run

    @override
    def on_llm_new_token(
        self,
        token: str | list[str | dict[str, Any]],
        *,
        chunk: GenerationChunk | ChatGenerationChunk | None = None,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> Run:
        """在新 LLM token 上运行。

        仅在启用流式传输时可用。

        Args:
            token: token，或结构化输出的内容块列表。
            chunk: 块。
            run_id: 运行 ID。
            parent_run_id: 父运行 ID。
            **kwargs: 额外参数。

        Returns:
            运行。
        """
                # "chat_model" 仅用于实验性的 new streaming_events 格式。
                # 此更改不应影响任何现有的追踪器。
        llm_run = self._llm_run_with_token_event(
            token=token,
            run_id=run_id,
            chunk=chunk,
            parent_run_id=parent_run_id,
        )
        self._on_llm_new_token(llm_run, token, chunk)
        return llm_run

    @override
    def on_retry(
        self,
        retry_state: RetryCallState,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> Run:
        """在重试时运行。

        Args:
            retry_state: 重试状态。
            run_id: 运行 ID。
            **kwargs: 额外参数。

        Returns:
            运行。
        """
        return self._llm_run_with_retry_event(
            retry_state=retry_state,
            run_id=run_id,
        )

    @override
    def on_llm_end(self, response: LLMResult, *, run_id: UUID, **kwargs: Any) -> Run:
        """结束一个 LLM 或聊天模型运行的追踪。

        Note:
            This is the end callback for both run types. Chat models start with
            `on_chat_model_start`, but there is no `on_chat_model_end`;
            completion is routed here for callback API compatibility.

        Args:
            response: 响应。
            run_id: 运行 ID。
            **kwargs: 额外参数。

        Returns:
            运行。
        """
                # "chat_model" 仅用于实验性的 new streaming_events 格式。
                # 此更改不应影响任何现有的追踪器。
        llm_run = self._complete_llm_run(
            response=response,
            run_id=run_id,
        )
        self._end_trace(llm_run)
        self._on_llm_end(llm_run)
        return llm_run

    def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> Run:
        """处理一个 LLM 运行的错误。

        Args:
            error: 错误。
            run_id: 运行 ID。
            **kwargs: 额外参数。

        Returns:
            运行。
        """
                # "chat_model" 仅用于实验性的 new streaming_events 格式。
                # 此更改不应影响任何现有的追踪器。
        llm_run = self._errored_llm_run(
            error=error, run_id=run_id, response=kwargs.pop("response", None)
        )
        self._end_trace(llm_run)
        self._on_llm_error(llm_run)
        return llm_run

    @override
    def on_chain_start(
        self,
        serialized: dict[str, Any],
        inputs: dict[str, Any],
        *,
        run_id: UUID,
        tags: list[str] | None = None,
        parent_run_id: UUID | None = None,
        metadata: dict[str, Any] | None = None,
        run_type: str | None = None,
        name: str | None = None,
        **kwargs: Any,
    ) -> Run:
        """为一个链运行启动追踪。

        Args:
            serialized: 序列化的链。
            inputs: 链的输入。
            run_id: 运行 ID。
            tags: 运行的标签。
            parent_run_id: 父运行 ID。
            metadata: 运行的元数据。
            run_type: 运行类型。
            name: 运行的名称。
            **kwargs: 额外参数。

        Returns:
            运行。
        """
        chain_run = self._create_chain_run(
            serialized=serialized,
            inputs=inputs,
            run_id=run_id,
            tags=tags,
            parent_run_id=parent_run_id,
            metadata=metadata,
            run_type=run_type,
            name=name,
            **kwargs,
        )
        self._start_trace(chain_run)
        self._on_chain_start(chain_run)
        return chain_run

    @override
    def on_chain_end(
        self,
        outputs: dict[str, Any],
        *,
        run_id: UUID,
        inputs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Run:
        """结束一个链运行的追踪。

        Args:
            outputs: 链的输出。
            run_id: 运行 ID。
            inputs: 链的输入。
            **kwargs: 额外参数。

        Returns:
            运行。
        """
        chain_run = self._complete_chain_run(
            outputs=outputs,
            run_id=run_id,
            inputs=inputs,
        )
        self._end_trace(chain_run)
        self._on_chain_end(chain_run)
        return chain_run

    @override
    def on_chain_error(
        self,
        error: BaseException,
        *,
        inputs: dict[str, Any] | None = None,
        run_id: UUID,
        **kwargs: Any,
    ) -> Run:
        """处理一个链运行的错误。

        Args:
            error: 错误。
            inputs: 链的输入。
            run_id: 运行 ID。
            **kwargs: 额外参数。

        Returns:
            运行。
        """
        chain_run = self._errored_chain_run(
            error=error,
            run_id=run_id,
            inputs=inputs,
        )
        self._end_trace(chain_run)
        self._on_chain_error(chain_run)
        return chain_run

    def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        tags: list[str] | None = None,
        parent_run_id: UUID | None = None,
        metadata: dict[str, Any] | None = None,
        name: str | None = None,
        inputs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Run:
        """为工具运行启动追踪。

        Args:
            serialized: 序列化的工具。
            input_str: 输入字符串。
            run_id: 运行 ID。
            tags: 运行标签。
            parent_run_id: 父运行 ID。
            metadata: 运行元数据。
            name: 运行名称。
            inputs: 工具的输入。
            **kwargs: 额外参数。

        Returns:
            运行。
        """
        tool_run = self._create_tool_run(
            serialized=serialized,
            input_str=input_str,
            run_id=run_id,
            tags=tags,
            parent_run_id=parent_run_id,
            metadata=metadata,
            name=name,
            inputs=inputs,
            **kwargs,
        )
        self._start_trace(tool_run)
        self._on_tool_start(tool_run)
        return tool_run

    @override
    def on_tool_end(self, output: Any, *, run_id: UUID, **kwargs: Any) -> Run:
        """结束工具运行的追踪。

        Args:
            output: 工具的输出。
            run_id: 运行 ID。
            **kwargs: 额外参数。

        Returns:
            运行。
        """
        tool_run = self._complete_tool_run(
            output=output,
            run_id=run_id,
        )
        self._end_trace(tool_run)
        self._on_tool_end(tool_run)
        return tool_run

    @override
    def on_tool_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> Run:
        """处理工具运行的错误。

        Args:
            error: 错误。
            run_id: 运行 ID。
            **kwargs: 额外参数。

        Returns:
            运行。
        """
        tool_run = self._errored_tool_run(
            error=error,
            run_id=run_id,
        )
        self._end_trace(tool_run)
        self._on_tool_error(tool_run)
        return tool_run

    def on_retriever_start(
        self,
        serialized: dict[str, Any],
        query: str,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        name: str | None = None,
        **kwargs: Any,
    ) -> Run:
        """当 `Retriever` 开始运行时调用。

        Args:
            serialized: 序列化的检索器。
            query: 查询。
            run_id: 运行 ID。
            parent_run_id: 父运行 ID。
            tags: 运行标签。
            metadata: 运行元数据。
            name: 运行名称。
            **kwargs: 额外参数。

        Returns:
            运行。
        """
        retrieval_run = self._create_retrieval_run(
            serialized=serialized,
            query=query,
            run_id=run_id,
            parent_run_id=parent_run_id,
            tags=tags,
            metadata=metadata,
            name=name,
            **kwargs,
        )
        self._start_trace(retrieval_run)
        self._on_retriever_start(retrieval_run)
        return retrieval_run

    @override
    def on_retriever_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> Run:
        """当 `Retriever` 出错时调用。

        Args:
            error: 错误。
            run_id: 运行 ID。
            **kwargs: 额外参数。

        Returns:
            运行。
        """
        retrieval_run = self._errored_retrieval_run(
            error=error,
            run_id=run_id,
        )
        self._end_trace(retrieval_run)
        self._on_retriever_error(retrieval_run)
        return retrieval_run

    @override
    def on_retriever_end(
        self, documents: Sequence[Document], *, run_id: UUID, **kwargs: Any
    ) -> Run:
        """当 `Retriever` 结束运行时调用。

        Args:
            documents: 文档。
            run_id: 运行 ID。
            **kwargs: 额外参数。

        Returns:
            运行。
        """
        retrieval_run = self._complete_retrieval_run(
            documents=documents,
            run_id=run_id,
        )
        self._end_trace(retrieval_run)
        self._on_retriever_end(retrieval_run)
        return retrieval_run

    def __deepcopy__(self, memo: dict[int, Any] | None = None) -> BaseTracer:
        """返回自身。"""
        return self

    def __copy__(self) -> BaseTracer:
        """返回自身。"""
        return self


class AsyncBaseTracer(_TracerCore, AsyncCallbackHandler, ABC):
    """追踪器的异步基础接口。"""

    @abstractmethod
    @override
    async def _persist_run(self, run: Run) -> None:
        """持久化一次运行。"""

    @override
    async def _start_trace(self, run: Run) -> None:
        """为一次运行启动追踪。

        Starting a trace will run concurrently with each `_on_[run_type]_start` method.
        No `_on_[run_type]_start` callback should depend on operations in
        `_start_trace`.
        """
        super()._start_trace(run)
        await self._on_run_create(run)

    @override
    async def _end_trace(self, run: Run) -> None:
        """结束一次运行的追踪。

        结束追踪将与每个 `_on_[run_type]_end` 方法并发运行。
        不应让 `_on_[run_type]_end` 回调依赖于 `_end_trace` 中的操作。
        """
        if not run.parent_run_id:
            await self._persist_run(run)
        self.run_map.pop(str(run.id))
                # 如果此运行的父运行是从外部追踪上下文注入的
                # (例如 langsmith @traceable)，则减少其子引用计数，并
                # 在最后一个子运行完成后将其从 run_map 中移除。
        parent_id = str(run.parent_run_id) if run.parent_run_id else None
        if parent_id and parent_id in self._external_run_ids:
            self._external_run_ids[parent_id] -= 1
            if self._external_run_ids[parent_id] <= 0:
                self.run_map.pop(parent_id, None)
                del self._external_run_ids[parent_id]
        await self._on_run_update(run)

    @override
    async def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[BaseMessage]],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        name: str | None = None,
        **kwargs: Any,
    ) -> Any:
        chat_model_run = self._create_chat_model_run(
            serialized=serialized,
            messages=messages,
            run_id=run_id,
            parent_run_id=parent_run_id,
            tags=tags,
            metadata=metadata,
            name=name,
            **kwargs,
        )
        tasks = [
            self._start_trace(chat_model_run),
            self._on_chat_model_start(chat_model_run),
        ]
        await asyncio.gather(*tasks)
        return chat_model_run

    @override
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
        llm_run = self._create_llm_run(
            serialized=serialized,
            prompts=prompts,
            run_id=run_id,
            parent_run_id=parent_run_id,
            tags=tags,
            metadata=metadata,
            **kwargs,
        )
        tasks = [self._start_trace(llm_run), self._on_llm_start(llm_run)]
        await asyncio.gather(*tasks)

    @override
    async def on_llm_new_token(
        self,
        token: str | list[str | dict[str, Any]],
        *,
        chunk: GenerationChunk | ChatGenerationChunk | None = None,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        llm_run = self._llm_run_with_token_event(
            token=token,
            run_id=run_id,
            chunk=chunk,
            parent_run_id=parent_run_id,
        )
        await self._on_llm_new_token(llm_run, token, chunk)

    @override
    async def on_retry(
        self,
        retry_state: RetryCallState,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        self._llm_run_with_retry_event(
            retry_state=retry_state,
            run_id=run_id,
        )

    @override
    async def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        """结束 LLM 或聊天模型运行的追踪。

        Note:
            This async callback also handles both run types. Async chat models
            start with `on_chat_model_start`, but there is no
            `on_chat_model_end`; completion is routed here for callback API
            compatibility.
        """
        llm_run = self._complete_llm_run(
            response=response,
            run_id=run_id,
        )
        tasks = [self._on_llm_end(llm_run), self._end_trace(llm_run)]
        await asyncio.gather(*tasks)

    @override
    async def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        llm_run = self._errored_llm_run(
            error=error,
            run_id=run_id,
        )
        tasks = [self._on_llm_error(llm_run), self._end_trace(llm_run)]
        await asyncio.gather(*tasks)

    @override
    async def on_chain_start(
        self,
        serialized: dict[str, Any],
        inputs: dict[str, Any],
        *,
        run_id: UUID,
        tags: list[str] | None = None,
        parent_run_id: UUID | None = None,
        metadata: dict[str, Any] | None = None,
        run_type: str | None = None,
        name: str | None = None,
        **kwargs: Any,
    ) -> None:
        chain_run = self._create_chain_run(
            serialized=serialized,
            inputs=inputs,
            run_id=run_id,
            tags=tags,
            parent_run_id=parent_run_id,
            metadata=metadata,
            run_type=run_type,
            name=name,
            **kwargs,
        )
        tasks = [self._start_trace(chain_run), self._on_chain_start(chain_run)]
        await asyncio.gather(*tasks)

    @override
    async def on_chain_end(
        self,
        outputs: dict[str, Any],
        *,
        run_id: UUID,
        inputs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        chain_run = self._complete_chain_run(
            outputs=outputs,
            run_id=run_id,
            inputs=inputs,
        )
        tasks = [self._end_trace(chain_run), self._on_chain_end(chain_run)]
        await asyncio.gather(*tasks)

    @override
    async def on_chain_error(
        self,
        error: BaseException,
        *,
        inputs: dict[str, Any] | None = None,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        chain_run = self._errored_chain_run(
            error=error,
            inputs=inputs,
            run_id=run_id,
        )
        tasks = [self._end_trace(chain_run), self._on_chain_error(chain_run)]
        await asyncio.gather(*tasks)

    @override
    async def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        tags: list[str] | None = None,
        parent_run_id: UUID | None = None,
        metadata: dict[str, Any] | None = None,
        name: str | None = None,
        inputs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        tool_run = self._create_tool_run(
            serialized=serialized,
            input_str=input_str,
            run_id=run_id,
            tags=tags,
            parent_run_id=parent_run_id,
            metadata=metadata,
            inputs=inputs,
            **kwargs,
        )
        tasks = [self._start_trace(tool_run), self._on_tool_start(tool_run)]
        await asyncio.gather(*tasks)

    @override
    async def on_tool_end(
        self,
        output: Any,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        tool_run = self._complete_tool_run(
            output=output,
            run_id=run_id,
        )
        tasks = [self._end_trace(tool_run), self._on_tool_end(tool_run)]
        await asyncio.gather(*tasks)

    @override
    async def on_tool_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        tool_run = self._errored_tool_run(
            error=error,
            run_id=run_id,
        )
        tasks = [self._end_trace(tool_run), self._on_tool_error(tool_run)]
        await asyncio.gather(*tasks)

    @override
    async def on_retriever_start(
        self,
        serialized: dict[str, Any],
        query: str,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        name: str | None = None,
        **kwargs: Any,
    ) -> None:
        retriever_run = self._create_retrieval_run(
            serialized=serialized,
            query=query,
            run_id=run_id,
            parent_run_id=parent_run_id,
            tags=tags,
            metadata=metadata,
            name=name,
        )
        tasks = [
            self._start_trace(retriever_run),
            self._on_retriever_start(retriever_run),
        ]
        await asyncio.gather(*tasks)

    @override
    async def on_retriever_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        retrieval_run = self._errored_retrieval_run(
            error=error,
            run_id=run_id,
        )
        tasks = [
            self._end_trace(retrieval_run),
            self._on_retriever_error(retrieval_run),
        ]
        await asyncio.gather(*tasks)

    @override
    async def on_retriever_end(
        self,
        documents: Sequence[Document],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        retrieval_run = self._complete_retrieval_run(
            documents=documents,
            run_id=run_id,
        )
        tasks = [self._end_trace(retrieval_run), self._on_retriever_end(retrieval_run)]
        await asyncio.gather(*tasks)

    async def _on_run_create(self, run: Run) -> None:
        """在创建运行时进行处理。"""

    async def _on_run_update(self, run: Run) -> None:
        """在更新运行时进行处理。"""

    async def _on_llm_start(self, run: Run) -> None:
        """在 LLM 运行开始时进行处理。"""

    async def _on_llm_end(self, run: Run) -> None:
        """处理 LLM/聊天模型运行的完成。"""

    async def _on_llm_error(self, run: Run) -> None:
        """在 LLM 运行出错时进行处理。"""

    async def _on_llm_new_token(
        self,
        run: Run,
        token: str | list[str | dict[str, Any]],
        chunk: GenerationChunk | ChatGenerationChunk | None,
    ) -> None:
        """处理新的 LLM token。"""

    async def _on_chain_start(self, run: Run) -> None:
        """在链运行开始时进行处理。"""

    async def _on_chain_end(self, run: Run) -> None:
        """处理链运行。"""

    async def _on_chain_error(self, run: Run) -> None:
        """在链运行出错时进行处理。"""

    async def _on_tool_start(self, run: Run) -> None:
        """在工具运行开始时进行处理。"""

    async def _on_tool_end(self, run: Run) -> None:
        """处理工具运行。"""

    async def _on_tool_error(self, run: Run) -> None:
        """在工具运行出错时进行处理。"""

    async def _on_chat_model_start(self, run: Run) -> None:
        """在聊天模型运行开始时进行处理。"""

    async def _on_retriever_start(self, run: Run) -> None:
        """在检索器运行开始时进行处理。"""

    async def _on_retriever_end(self, run: Run) -> None:
        """处理检索器运行。"""

    async def _on_retriever_error(self, run: Run) -> None:
        """在检索器运行出错时进行处理。"""

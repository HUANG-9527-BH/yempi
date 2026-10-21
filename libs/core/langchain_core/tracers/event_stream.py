"""用于支持事件流 API 的内部追踪器。"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from typing import (
    TYPE_CHECKING,
    Any,
    TypedDict,
    TypeVar,
    cast,
)

from typing_extensions import NotRequired, override

from langchain_core.callbacks.base import AsyncCallbackHandler, BaseCallbackManager
from langchain_core.messages import AIMessageChunk, BaseMessage, BaseMessageChunk
from langchain_core.outputs import (
    ChatGenerationChunk,
    GenerationChunk,
    LLMResult,
)
from langchain_core.runnables import ensure_config
from langchain_core.runnables.schema import (
    CustomStreamEvent,
    EventData,
    StandardStreamEvent,
    StreamEvent,
)
from langchain_core.runnables.utils import (
    Input,
    Output,
    _RootEventFilter,
)
from langchain_core.tracers._streaming import _StreamingCallbackHandler
from langchain_core.tracers.log_stream import (
    LogStreamCallbackHandler,
    RunLog,
    _astream_log_implementation,
)
from langchain_core.tracers.memory_stream import _get_or_create_loop, _MemoryStream
from langchain_core.utils.aiter import aclosing
from langchain_core.utils.uuid import uuid7

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Iterator, Sequence
    from uuid import UUID

    from langchain_core.documents import Document
    from langchain_core.runnables import Runnable, RunnableConfig
    from langchain_core.tracers.log_stream import LogEntry

logger = logging.getLogger(__name__)


class RunInfo(TypedDict):
    """关于一次运行的信息。

    用于跟踪与一次运行相关的元数据。
    """

    name: str
    """运行的名称。"""

    tags: list[str]
    """与运行相关联的标签。"""

    metadata: dict[str, Any]
    """与运行相关联的元数据。"""

    run_type: str
    """运行的类型。"""

    inputs: NotRequired[Any]
    """运行的输入。"""

    parent_run_id: UUID | None
    """父运行的 ID。"""

    tool_call_id: NotRequired[str | None]
    """与运行关联的工具调用 ID。"""


def _assign_name(name: str | None, serialized: dict[str, Any] | None) -> str:
    """为运行分配一个名称。"""
    if name is not None:
        return name
    if serialized is not None:
        if "name" in serialized:
            return cast("str", serialized["name"])
        if "id" in serialized:
            return cast("str", serialized["id"][-1])
    return "Unnamed"


T = TypeVar("T")


class _AstreamEventsCallbackHandler(
    AsyncCallbackHandler, _StreamingCallbackHandler[Any]
):
    """异步回调处理器的实现，用于 astream 事件。"""

    def __init__(
        self,
        *args: Any,
        include_names: Sequence[str] | None = None,
        include_types: Sequence[str] | None = None,
        include_tags: Sequence[str] | None = None,
        exclude_names: Sequence[str] | None = None,
        exclude_types: Sequence[str] | None = None,
        exclude_tags: Sequence[str] | None = None,
        **kwargs: Any,
    ) -> None:
        """初始化追踪器。"""
        super().__init__(*args, **kwargs)
                # 运行 ID 到运行信息的映射。
                # 对应于给定运行 ID 的条目在每次相应运行结束时被清理。
                # 当每个对应运行结束时进行清理。
        self.run_map: dict[UUID, RunInfo] = {}
                # 与父运行结束对应的回调事件
                # 可能在子运行结束对应的回调事件之前被调用，
                # 这会导致 run_map 的清理。
                # 因此我们单独维护一个子运行 ID 与父运行 ID 之间的映射关系。
                # 该容器在追踪器被垃圾回收时一并被回收。
        self.parent_map: dict[UUID, UUID | None] = {}

        self.is_tapped: dict[UUID, Any] = {}

                # 过滤将通过队列发送的事件。
        self.root_event_filter = _RootEventFilter(
            include_names=include_names,
            include_types=include_types,
            include_tags=include_tags,
            exclude_names=exclude_names,
            exclude_types=exclude_types,
            exclude_tags=exclude_tags,
        )

        memory_stream = _MemoryStream[StreamEvent](_get_or_create_loop(self))
        self.send_stream = memory_stream.get_send_stream()
        self.receive_stream = memory_stream.get_receive_stream()

    def _get_parent_ids(self, run_id: UUID) -> list[str]:
        """获取运行的父 ID（非递归），并转换为字符串。"""
        parent_ids = []

        while parent_id := self.parent_map.get(run_id):
            str_parent_id = str(parent_id)
            if str_parent_id in parent_ids:
                msg = (
                    f"Parent ID {parent_id} is already in the parent_ids list. "
                    f"This should never happen."
                )
                raise AssertionError(msg)
            parent_ids.append(str_parent_id)
            run_id = parent_id

                # 以逆序返回父运行 ID，使得第一个父运行 ID 为根，最后一个为直接父运行 ID。
                # 以逆序返回父运行 ID，使得第一个父运行 ID 为根，最后一个为直接父运行 ID。
        return parent_ids[::-1]

    def _send(self, event: StreamEvent, event_type: str) -> None:
        """向流发送一个事件。"""
        if self.root_event_filter.include_event(event, event_type):
            self.send_stream.send_nowait(event)

    def __aiter__(self) -> AsyncIterator[Any]:
        """遍历接收流。

        Returns:
            接收流的异步迭代器。
        """
        return self.receive_stream.__aiter__()

    async def tap_output_aiter(
        self, run_id: UUID, output: AsyncIterator[T]
    ) -> AsyncIterator[T]:
        """对输出异步迭代器进行采样。

        此方法用于对生成异步迭代器的 `Runnable` 的输出进行采样。
        用于为 `Runnable` 的输出生成流事件。

        Args:
            run_id: 运行的 ID。
            output: `Runnable` 的输出。

        Yields:
            `Runnable` 的输出。
        """
        sentinel = object()
                # 原子性的检查与设置
        tap = self.is_tapped.setdefault(run_id, sentinel)
                # 等待首个数据块
        first = await anext(output, sentinel)
        if first is sentinel:
            return
                # 获取运行信息
        run_info = self.run_map.get(run_id)
        if run_info is None:
                        # 运行已结束，不再发出任何流式事件
            yield cast("T", first)
            return
        if tap is sentinel:
                        # 如果这是首个监听者，发出流式事件
            event: StandardStreamEvent = {
                "event": f"on_{run_info['run_type']}_stream",
                "run_id": str(run_id),
                "name": run_info["name"],
                "tags": run_info["tags"],
                "metadata": run_info["metadata"],
                "data": {},
                "parent_ids": self._get_parent_ids(run_id),
            }
            self._send({**event, "data": {"chunk": first}}, run_info["run_type"])
            yield cast("T", first)
                        # 消耗剩余输出
            async for chunk in output:
                self._send(
                    {**event, "data": {"chunk": chunk}},
                    run_info["run_type"],
                )
                yield chunk
        else:
                        # 否则直接传递
            yield cast("T", first)
                        # 消耗剩余输出
            async for chunk in output:
                yield chunk

    def tap_output_iter(self, run_id: UUID, output: Iterator[T]) -> Iterator[T]:
        """对输出迭代器进行采样。

        Args:
            run_id: 运行的 ID。
            output: `Runnable` 的输出。

        Yields:
            `Runnable` 的输出。
        """
        sentinel = object()
                # 原子性检查并设置
        tap = self.is_tapped.setdefault(run_id, sentinel)
                # 等待第一个数据块
        first = next(output, sentinel)
        if first is sentinel:
            return
                # 获取运行信息
        run_info = self.run_map.get(run_id)
        if run_info is None:
                        # 运行已完成，不发出任何流事件
            yield cast("T", first)
            return
        if tap is sentinel:
                        # 如果这是首个监听者，发出流事件
            event: StandardStreamEvent = {
                "event": f"on_{run_info['run_type']}_stream",
                "run_id": str(run_id),
                "name": run_info["name"],
                "tags": run_info["tags"],
                "metadata": run_info["metadata"],
                "data": {},
                "parent_ids": self._get_parent_ids(run_id),
            }
            self._send({**event, "data": {"chunk": first}}, run_info["run_type"])
            yield cast("T", first)
                        # 消耗剩余输出
            for chunk in output:
                self._send(
                    {**event, "data": {"chunk": chunk}},
                    run_info["run_type"],
                )
                yield chunk
        else:
                        # 否则直接传递
            yield cast("T", first)
                        # 消耗剩余输出
            for chunk in output:
                yield chunk

    def _write_run_start_info(
        self,
        run_id: UUID,
        *,
        tags: list[str] | None,
        metadata: dict[str, Any] | None,
        parent_run_id: UUID | None,
        name_: str,
        run_type: str,
        **kwargs: Any,
    ) -> None:
        """更新运行信息。"""
        info: RunInfo = {
            "tags": tags or [],
            "metadata": metadata or {},
            "name": name_,
            "run_type": run_type,
            "parent_run_id": parent_run_id,
        }

        if "inputs" in kwargs:
                        # 特殊处理输入以允许输入为
                        # 可选提供的值，并区分缺失值
                        # 与 None 值。
            info["inputs"] = kwargs["inputs"]

        if "tool_call_id" in kwargs:
                        # 在运行信息中存储 tool_call_id 以关联错误到工具调用
            info["tool_call_id"] = kwargs["tool_call_id"]

        self.run_map[run_id] = info
        self.parent_map[run_id] = parent_run_id

    @override
    async def on_chat_model_start(
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
    ) -> None:
        """为聊天模型运行启动追踪。"""
        name_ = _assign_name(name, serialized)
        run_type = "chat_model"

        self._write_run_start_info(
            run_id,
            tags=tags,
            metadata=metadata,
            parent_run_id=parent_run_id,
            name_=name_,
            run_type=run_type,
            inputs={"messages": messages},
        )

        self._send(
            {
                "event": "on_chat_model_start",
                "data": {
                    "input": {"messages": messages},
                },
                "name": name_,
                "tags": tags or [],
                "run_id": str(run_id),
                "metadata": metadata or {},
                "parent_ids": self._get_parent_ids(run_id),
            },
            run_type,
        )

    @override
    async def on_llm_start(
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
    ) -> None:
        """为（非聊天模型）LLM 运行启动追踪。"""
        name_ = _assign_name(name, serialized)
        run_type = "llm"

        self._write_run_start_info(
            run_id,
            tags=tags,
            metadata=metadata,
            parent_run_id=parent_run_id,
            name_=name_,
            run_type=run_type,
            inputs={"prompts": prompts},
        )

        self._send(
            {
                "event": "on_llm_start",
                "data": {
                    "input": {
                        "prompts": prompts,
                    }
                },
                "name": name_,
                "tags": tags or [],
                "run_id": str(run_id),
                "metadata": metadata or {},
                "parent_ids": self._get_parent_ids(run_id),
            },
            run_type,
        )

    @override
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
        """生成自定义的流事件。"""
        event = CustomStreamEvent(
            event="on_custom_event",
            run_id=str(run_id),
            name=name,
            tags=tags or [],
            metadata=metadata or {},
            data=data,
            parent_ids=self._get_parent_ids(run_id),
        )
        self._send(event, name)

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
        """在新输出 token 上运行。

        仅在启用流式传输时可用。

        适用于聊天模型和非聊天模型（旧版文本补全 LLM）。

        Raises:
            ValueError: 如果运行类型不是 `llm` 或 `chat_model`。
            AssertionError: 如果运行 ID 在运行映射中未找到。
        """
        run_info = self.run_map.get(run_id)
        chunk_: GenerationChunk | BaseMessageChunk

        if run_info is None:
            msg = f"Run ID {run_id} not found in run map."
            raise AssertionError(msg)
        if self.is_tapped.get(run_id):
            return
        if run_info["run_type"] == "chat_model":
            event = "on_chat_model_stream"

            if chunk is None:
                chunk_ = AIMessageChunk(content=token)
            else:
                chunk_ = cast("ChatGenerationChunk", chunk).message

        elif run_info["run_type"] == "llm":
            event = "on_llm_stream"
            if chunk is None:
                text = token if isinstance(token, str) else ""
                chunk_ = GenerationChunk(text=text)
            else:
                chunk_ = cast("GenerationChunk", chunk)
        else:
            msg = f"Unexpected run type: {run_info['run_type']}"
            raise ValueError(msg)

        self._send(
            {
                "event": event,
                "data": {
                    "chunk": chunk_,
                },
                "run_id": str(run_id),
                "name": run_info["name"],
                "tags": run_info["tags"],
                "metadata": run_info["metadata"],
                "parent_ids": self._get_parent_ids(run_id),
            },
            run_info["run_type"],
        )

    @override
    async def on_llm_end(
        self, response: LLMResult, *, run_id: UUID, **kwargs: Any
    ) -> None:
        """结束模型运行的追踪。

        适用于聊天模型和非聊天模型（旧版文本补全 LLM）。

        Raises:
            ValueError: 如果运行类型不是 `'llm'` 或 `'chat_model'`。
        """
        run_info = self.run_map.pop(run_id)
        inputs_ = run_info.get("inputs")

        generations: list[list[GenerationChunk]] | list[list[ChatGenerationChunk]]
        output: dict[str, Any] | BaseMessage = {}

        if run_info["run_type"] == "chat_model":
            generations = cast("list[list[ChatGenerationChunk]]", response.generations)
            for gen in generations:
                if output != {}:
                    break
                for chunk in gen:
                    output = chunk.message
                    break

            event = "on_chat_model_end"
        elif run_info["run_type"] == "llm":
            generations = cast("list[list[GenerationChunk]]", response.generations)
            output = {
                "generations": [
                    [
                        {
                            "text": chunk.text,
                            "generation_info": chunk.generation_info,
                            "type": chunk.type,
                        }
                        for chunk in gen
                    ]
                    for gen in generations
                ],
                "llm_output": response.llm_output,
            }
            event = "on_llm_end"
        else:
            msg = f"Unexpected run type: {run_info['run_type']}"
            raise ValueError(msg)

        self._send(
            {
                "event": event,
                "data": {"output": output, "input": inputs_},
                "run_id": str(run_id),
                "name": run_info["name"],
                "tags": run_info["tags"],
                "metadata": run_info["metadata"],
                "parent_ids": self._get_parent_ids(run_id),
            },
            run_info["run_type"],
        )

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
        """开始链运行的追踪。"""
        name_ = _assign_name(name, serialized)
        run_type_ = run_type or "chain"

        data: EventData = {}

                # 绕过 Runnable 核心代码在某些情况下未发送输入的问题
                # 绕过 Runnable 核心代码在某些情况下未发送输入的问题
        if inputs != {"input": ""}:
            data["input"] = inputs
            kwargs["inputs"] = inputs

        self._write_run_start_info(
            run_id,
            tags=tags,
            metadata=metadata,
            parent_run_id=parent_run_id,
            name_=name_,
            run_type=run_type_,
            **kwargs,
        )

        self._send(
            {
                "event": f"on_{run_type_}_start",
                "data": data,
                "name": name_,
                "tags": tags or [],
                "run_id": str(run_id),
                "metadata": metadata or {},
                "parent_ids": self._get_parent_ids(run_id),
            },
            run_type_,
        )

    @override
    async def on_chain_end(
        self,
        outputs: dict[str, Any],
        *,
        run_id: UUID,
        inputs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """结束链运行的追踪。"""
        run_info = self.run_map.pop(run_id)
        run_type = run_info["run_type"]

        event = f"on_{run_type}_end"

        inputs = inputs or run_info.get("inputs") or {}

        data: EventData = {
            "output": outputs,
            "input": inputs,
        }

        self._send(
            {
                "event": event,
                "data": data,
                "run_id": str(run_id),
                "name": run_info["name"],
                "tags": run_info["tags"],
                "metadata": run_info["metadata"],
                "parent_ids": self._get_parent_ids(run_id),
            },
            run_type,
        )

    def _get_tool_run_info_with_inputs(self, run_id: UUID) -> tuple[RunInfo, Any]:
        """获取工具的运行信息并提取输入，包含验证。

        Args:
            run_id: 工具的运行 ID。

        Returns:
            一个元组 `(run_info, inputs)`。

        Raises:
            AssertionError: 如果运行 ID 是工具调用但没有输入。
        """
        run_info = self.run_map.pop(run_id)
        if "inputs" not in run_info:
            msg = (
                f"Run ID {run_id} is a tool call and is expected to have "
                f"inputs associated with it."
            )
            raise AssertionError(msg)
        inputs = run_info["inputs"]
        return run_info, inputs

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
        """开始工具运行的追踪。"""
        name_ = _assign_name(name, serialized)

        self._write_run_start_info(
            run_id,
            tags=tags,
            metadata=metadata,
            parent_run_id=parent_run_id,
            name_=name_,
            run_type="tool",
            inputs=inputs,
            tool_call_id=kwargs.get("tool_call_id"),
        )

        self._send(
            {
                "event": "on_tool_start",
                "data": {
                    "input": inputs or {},
                },
                "name": name_,
                "tags": tags or [],
                "run_id": str(run_id),
                "metadata": metadata or {},
                "parent_ids": self._get_parent_ids(run_id),
            },
            "tool",
        )

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
        """工具出错时运行。"""
                # 如果直接传入，则从 kwargs 中提取 tool_call_id，否则从 run_info 中提取
                # (在 on_tool_start 期间存储)
        tool_call_id = kwargs.get("tool_call_id")
        run_info, inputs = self._get_tool_run_info_with_inputs(run_id)
        if tool_call_id is None:
            tool_call_id = run_info.get("tool_call_id")

        event: StandardStreamEvent = {
            "event": "on_tool_error",
            "data": {
                "error": error,
                "input": inputs,
                "tool_call_id": tool_call_id,
            },
            "run_id": str(run_id),
            "name": run_info["name"],
            "tags": run_info["tags"],
            "metadata": run_info["metadata"],
            "parent_ids": self._get_parent_ids(run_id),
        }
        self._send(event, "tool")

    @override
    async def on_tool_end(self, output: Any, *, run_id: UUID, **kwargs: Any) -> None:
        """结束工具运行的追踪。"""
        run_info, inputs = self._get_tool_run_info_with_inputs(run_id)

        self._send(
            {
                "event": "on_tool_end",
                "data": {
                    "output": output,
                    "input": inputs,
                },
                "run_id": str(run_id),
                "name": run_info["name"],
                "tags": run_info["tags"],
                "metadata": run_info["metadata"],
                "parent_ids": self._get_parent_ids(run_id),
            },
            "tool",
        )

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
        """当 `Retriever` 开始运行时触发。"""
        name_ = _assign_name(name, serialized)
        run_type = "retriever"

        self._write_run_start_info(
            run_id,
            tags=tags,
            metadata=metadata,
            parent_run_id=parent_run_id,
            name_=name_,
            run_type=run_type,
            inputs={"query": query},
        )

        self._send(
            {
                "event": "on_retriever_start",
                "data": {
                    "input": {
                        "query": query,
                    }
                },
                "name": name_,
                "tags": tags or [],
                "run_id": str(run_id),
                "metadata": metadata or {},
                "parent_ids": self._get_parent_ids(run_id),
            },
            run_type,
        )

    @override
    async def on_retriever_end(
        self, documents: Sequence[Document], *, run_id: UUID, **kwargs: Any
    ) -> None:
        """当 `Retriever` 结束运行时触发。"""
        run_info = self.run_map.pop(run_id)

        self._send(
            {
                "event": "on_retriever_end",
                "data": {
                    "output": documents,
                    "input": run_info.get("inputs"),
                },
                "run_id": str(run_id),
                "name": run_info["name"],
                "tags": run_info["tags"],
                "metadata": run_info["metadata"],
                "parent_ids": self._get_parent_ids(run_id),
            },
            run_info["run_type"],
        )

    def __deepcopy__(
        self, memo: dict[int, Any] | None = None
    ) -> _AstreamEventsCallbackHandler:
        """返回自身。"""
        return self

    def __copy__(self) -> _AstreamEventsCallbackHandler:
        """返回自身。"""
        return self


async def _astream_events_implementation_v1(
    runnable: Runnable[Input, Output],
    value: Any,
    config: RunnableConfig | None = None,
    *,
    include_names: Sequence[str] | None = None,
    include_types: Sequence[str] | None = None,
    include_tags: Sequence[str] | None = None,
    exclude_names: Sequence[str] | None = None,
    exclude_types: Sequence[str] | None = None,
    exclude_tags: Sequence[str] | None = None,
    **kwargs: Any,
) -> AsyncIterator[StandardStreamEvent]:
    stream = LogStreamCallbackHandler(
        auto_close=False,
        include_names=include_names,
        include_types=include_types,
        include_tags=include_tags,
        exclude_names=exclude_names,
        exclude_types=exclude_types,
        exclude_tags=exclude_tags,
        _schema_format="streaming_events",
    )

    run_log = RunLog(state=None)  # type: ignore[arg-type]
    encountered_start_event = False

    root_event_filter = _RootEventFilter(
        include_names=include_names,
        include_types=include_types,
        include_tags=include_tags,
        exclude_names=exclude_names,
        exclude_types=exclude_types,
        exclude_tags=exclude_tags,
    )

    config = ensure_config(config)
    root_tags = config.get("tags", [])
    root_metadata = config.get("metadata", {})
    root_name = config.get("run_name", runnable.get_name())

    async for log in _astream_log_implementation(
        runnable,
        value,
        config=config,
        stream=stream,
        diff=True,
        with_streamed_output_list=True,
        **kwargs,
    ):
        run_log += log

        if not encountered_start_event:
                        # 生成根 Runnable 的开始事件
            encountered_start_event = True
            state = run_log.state.copy()

            event = StandardStreamEvent(
                event=f"on_{state['type']}_start",
                run_id=state["id"],
                name=root_name,
                tags=root_tags,
                metadata=root_metadata,
                data={
                    "input": value,
                },
                parent_ids=[],  # v1 中不支持
            )

            if root_event_filter.include_event(event, state["type"]):
                yield event

        paths = {
            op["path"].split("/")[2]
            for op in log.ops
            if op["path"].startswith("/logs/")
        }
                # 集合中的元素应按现代 Python 版本中插入的顺序迭代
                # 集合中的元素应按现代 Python 版本中插入的顺序迭代
        for path in paths:
            data: EventData = {}
            log_entry: LogEntry = run_log.state["logs"][path]
            if log_entry["end_time"] is None:
                event_type = "stream" if log_entry["streamed_output"] else "start"
            else:
                event_type = "end"

            if event_type == "start":
                                # 如果可用，将输入包含在开始事件中
                                # 通常对于操作流的组件而言，这些输入不会可用
                                # 因为这些组件会流式传输输入
                                # 且直到流结束才知道其最终值
                inputs = log_entry.get("inputs")
                if inputs is not None:
                    data["input"] = inputs

            if event_type == "end":
                inputs = log_entry.get("inputs")
                if inputs is not None:
                    data["input"] = inputs

                                # None 是结束事件的有效输出
                data["output"] = log_entry["final_output"]

            if event_type == "stream":
                num_chunks = len(log_entry["streamed_output"])
                if num_chunks != 1:
                    msg = (
                        f"Expected exactly one chunk of streamed output, "
                        f"got {num_chunks} instead. This is impossible. "
                        f"Encountered in: {log_entry['name']}"
                    )
                    raise AssertionError(msg)

                data = {"chunk": log_entry["streamed_output"][0]}
                                # 清理流，我们不再需要它
                                # 这也能避免重复！
                log_entry["streamed_output"] = []

            yield StandardStreamEvent(
                event=f"on_{log_entry['type']}_{event_type}",
                name=log_entry["name"],
                run_id=log_entry["id"],
                tags=log_entry["tags"],
                metadata=log_entry["metadata"],
                data=data,
                parent_ids=[],  # v1 中不支持
            )

                # 最后，我们处理根链可能产生的流式输出
                # 如果存在的话
        state = run_log.state
        if state["streamed_output"]:
            num_chunks = len(state["streamed_output"])
            if num_chunks != 1:
                msg = (
                    f"Expected exactly one chunk of streamed output, "
                    f"got {num_chunks} instead. This is impossible. "
                    f"Encountered in: {state['name']}"
                )
                raise AssertionError(msg)

            data = {"chunk": state["streamed_output"][0]}
                        # 清理流，我们不再需要它
            state["streamed_output"] = []

            event = StandardStreamEvent(
                event=f"on_{state['type']}_stream",
                run_id=state["id"],
                tags=root_tags,
                metadata=root_metadata,
                name=root_name,
                data=data,
                parent_ids=[],  # v1 中不支持
            )
            if root_event_filter.include_event(event, state["type"]):
                yield event

    state = run_log.state

        # 最后生成根 Runnable 的结束事件
    event = StandardStreamEvent(
        event=f"on_{state['type']}_end",
        name=root_name,
        run_id=state["id"],
        tags=root_tags,
        metadata=root_metadata,
        data={
            "output": state["final_output"],
        },
        parent_ids=[],  # v1 中不支持
    )
    if root_event_filter.include_event(event, state["type"]):
        yield event


async def _astream_events_implementation_v2(
    runnable: Runnable[Input, Output],
    value: Any,
    config: RunnableConfig | None = None,
    *,
    include_names: Sequence[str] | None = None,
    include_types: Sequence[str] | None = None,
    include_tags: Sequence[str] | None = None,
    exclude_names: Sequence[str] | None = None,
    exclude_types: Sequence[str] | None = None,
    exclude_tags: Sequence[str] | None = None,
    **kwargs: Any,
) -> AsyncIterator[StandardStreamEvent]:
    """v2 Runnable 的 astream 事件 API 的实现。"""
    event_streamer = _AstreamEventsCallbackHandler(
        include_names=include_names,
        include_types=include_types,
        include_tags=include_tags,
        exclude_names=exclude_names,
        exclude_types=exclude_types,
        exclude_tags=exclude_tags,
    )

        # 将流处理器分配给配置
    config = ensure_config(config)
    if "run_id" in config:
        run_id = cast("UUID", config["run_id"])
    else:
        run_id = uuid7()
        config["run_id"] = run_id
    callbacks = config.get("callbacks")
    if callbacks is None:
        config["callbacks"] = [event_streamer]
    elif isinstance(callbacks, list):
        config["callbacks"] = [*callbacks, event_streamer]
    elif isinstance(callbacks, BaseCallbackManager):
        callbacks = callbacks.copy()
        callbacks.add_handler(event_streamer, inherit=True)
        config["callbacks"] = callbacks
    else:
        msg = (  # type: ignore[unreachable]
            f"Unexpected type for callbacks: {callbacks}."
            "Expected None, list or AsyncCallbackManager."
        )
        raise ValueError(msg)

        # 以流式模式调用 Runnable
        # 将每个数据块添加到输出流中
    async def consume_astream() -> None:
        try:
                        # 如果 astream 也调用了 tap_output_aiter，这将是无操作
            async with aclosing(runnable.astream(value, config, **kwargs)) as stream:
                async for _ in event_streamer.tap_output_aiter(run_id, stream):
                                        # 所有内容都会被拾取
                    pass
        finally:
            await event_streamer.send_stream.aclose()

        # 在任务中启动 Runnable，以便开始消费输出
    task = asyncio.create_task(consume_astream())

    first_event_sent = False
    first_event_run_id = None

    try:
        async for event in event_streamer:
            if not first_event_sent:
                first_event_sent = True
                                # 这是对一个已知问题的临时解决方案：链的输入直到整个输入被消耗后才可用
                                # 作为临时方案，我们将输入修改为传入链的输入
                                # 如果这是对应于根 Runnable 的结束事件
                                # 我们不在事件中包含输入，因为它保证会在第一个事件中出现
                event["data"]["input"] = value
                first_event_run_id = event["run_id"]
                yield event
                continue

                        # 如果任务仍在运行则取消它
                        # 如果任务仍在运行则取消它
                        # 仍然等待，以执行任何清理代码，并传播任何异常
            if (
                event["run_id"] == first_event_run_id
                and event["event"].endswith("_end")
                and "input" in event["data"]
            ):
                del event["data"]["input"]

            yield event
    except asyncio.CancelledError as exc:
                # 如果任务仍在运行，则取消它
        task.cancel(exc.args[0] if exc.args else None)
        raise
    finally:
                # 如果任务仍在运行，则取消它
        task.cancel()
                # 仍然等待它，以运行任何清理代码，并传播任何异常
        with contextlib.suppress(asyncio.CancelledError):
            await task

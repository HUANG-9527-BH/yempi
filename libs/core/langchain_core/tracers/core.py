"""根监听器的工具函数。"""

from __future__ import annotations

import logging
import traceback
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    cast,
)

from langchain_core.exceptions import TracerException
from langchain_core.load import dumpd
from langchain_core.tracers.schemas import Run
from langchain_core.utils._gateway import GATEWAY_METADATA_RESPONSE_KEY

if TYPE_CHECKING:
    from collections.abc import Coroutine, Sequence
    from uuid import UUID

    from tenacity import RetryCallState

    from langchain_core.documents import Document
    from langchain_core.messages import BaseMessage
    from langchain_core.outputs import (
        ChatGeneration,
        ChatGenerationChunk,
        GenerationChunk,
        LLMResult,
    )

logger = logging.getLogger(__name__)

SCHEMA_FORMAT_TYPE = Literal["original", "streaming_events"]

# 用于在追踪时将网关元数据附加到运行元数据的键。
_GATEWAY_RUN_METADATA_KEY = "ls_gateway_info"


def _extract_gateway_metadata(response: LLMResult) -> dict[str, Any] | None:
    """从模型响应中提取 LangSmith 网关元数据。

    当请求通过 LangSmith 网关路由时，网关会返回
    元数据（例如解析后的提供方和模型），这些元数据由集成在生成的 `generation_info`
    中，位于 `GATEWAY_METADATA_RESPONSE_KEY` 下。

    Args:
        response: 模型生成的 `LLMResult`。

    Returns:
        网关元数据，如果响应中不包含则为 `None`。
    """
    for generations in response.generations:
        for generation in generations:
            generation_info = getattr(generation, "generation_info", None) or {}
            gateway_metadata = generation_info.get(GATEWAY_METADATA_RESPONSE_KEY)
            if isinstance(gateway_metadata, dict):
                return gateway_metadata
    return None


class _TracerCore(ABC):
    """追踪器的抽象基类。

    该类提供通用方法以及可重用的追踪器方法。
    """

    log_missing_parent: bool = True

    def __init__(
        self,
        *,
        _schema_format: Literal[
            "original", "streaming_events", "original+chat"
        ] = "original",
        run_map: dict[str, Run] | None = None,
        order_map: dict[UUID, tuple[UUID, str]] | None = None,
        _external_run_ids: dict[str, int] | None = None,
        **kwargs: Any,
    ) -> None:
        """初始化追踪器。

        Args:
            _schema_format: 主要改变输入和输出的处理方式。

                仅供内部使用。此 API 将发生变化。

                - `'original'` 是当前所有追踪器使用的格式。
                此格式在输入和输出处理上略有不一致。 - `'streaming_events'`
                用于支持流式事件，仅限内部使用。未来可能更改，或被专门用于流式事件的异步追踪器取代。
                - `'original+chat'` 的格式与 `'original'` 相同，但不会在
                `on_chat_model_start` 时抛出属性错误

                    This format is slightly inconsistent with respect to inputs and
                    outputs.
                - `'streaming_events'` is used for supporting streaming events, for
                    internal usage. It will likely change in the future, or be
                    deprecated entirely in favor of a dedicated async tracer for
                    streaming events.
                - `'original+chat'` is a format that is the same as `'original'` except
                    it does NOT raise an attribute error `on_chat_model_start`
            run_map: Optional shared map of run ID to run.
            order_map: Optional shared map of run ID to trace ordering data.
            _external_run_ids: Optional shared set of externally injected run IDs.
            **kwargs: Additional keyword arguments that will be passed to the
                superclass.
        """
        super().__init__(**kwargs)

        self._schema_format = _schema_format  # 仅供内部使用，API 将发生变化。

        self.run_map = run_map if run_map is not None else {}
        """Map of run ID to run. Cleared on run end."""

        self.order_map = order_map if order_map is not None else {}
        """Map of run ID to (trace_id, dotted_order). Cleared when tracer GCed."""

        self._external_run_ids: dict[str, int] = (
            _external_run_ids if _external_run_ids is not None else {}
        )
        """Refcount of active children per externally-injected run ID.

        These runs are added to `run_map` so child runs can find their parent,
        but they are not managed by the tracer's callback lifecycle.  When
        the last child finishes the entry is evicted to avoid memory leaks.
        """

    @abstractmethod
    def _persist_run(self, run: Run) -> Coroutine[Any, Any, None] | None:
        """持久化一个运行。"""

    @staticmethod
    def _add_child_run(
        parent_run: Run,
        child_run: Run,
    ) -> None:
        """向链运行或工具运行添加子运行。"""
        parent_run.child_runs.append(child_run)

    @staticmethod
    def _get_stacktrace(error: BaseException) -> str:
        """获取父错误的堆栈跟踪。"""
        msg = repr(error)
        try:
            tb = traceback.format_exception(error)
            return (msg + "\n\n".join(tb)).strip()
        except Exception:
            return msg

    def _start_trace(self, run: Run) -> Coroutine[Any, Any, None] | None:  # type: ignore[return]
        current_dotted_order = run.start_time.strftime("%Y%m%dT%H%M%S%fZ") + str(run.id)
        if run.parent_run_id:
            if parent := self.order_map.get(run.parent_run_id):
                run.trace_id, run.dotted_order = parent
                run.dotted_order += "." + current_dotted_order
                if parent_run := self.run_map.get(str(run.parent_run_id)):
                    self._add_child_run(parent_run, run)
                    parent_key = str(run.parent_run_id)
                    if parent_key in self._external_run_ids:
                        self._external_run_ids[parent_key] += 1
            else:
                if self.log_missing_parent:
                    logger.debug(
                        "Parent run %s not found for run %s. Treating as a root run.",
                        run.parent_run_id,
                        run.id,
                    )
                run.parent_run_id = None
                run.trace_id = run.id
                run.dotted_order = current_dotted_order
        else:
            run.trace_id = run.id
            run.dotted_order = current_dotted_order
        self.order_map[run.id] = (run.trace_id, run.dotted_order)
        self.run_map[str(run.id)] = run

    def _get_run_kwargs(
        self,
        parent_run_id: UUID | None,  # noqa: ARG002
        extra: dict[str, Any],
    ) -> dict[str, Any]:
        """返回运行的构造参数。"""
        return {"extra": extra}

    def _get_run(self, run_id: UUID, run_type: str | set[str] | None = None) -> Run:
        try:
            run = self.run_map[str(run_id)]
        except KeyError as exc:
            msg = f"No indexed run ID {run_id}."
            raise TracerException(msg) from exc

        if isinstance(run_type, str):
            run_types: set[str] | None = {run_type}
        else:
            run_types = run_type
        if run_types is not None and run.run_type not in run_types:
            msg = (
                f"Found {run.run_type} run at ID {run_id}, "
                f"but expected {run_types} run."
            )
            raise TracerException(msg)
        return run

    def _create_chat_model_run(
        self,
        serialized: dict[str, Any],
        messages: list[list[BaseMessage]],
        run_id: UUID,
        tags: list[str] | None = None,
        parent_run_id: UUID | None = None,
        metadata: dict[str, Any] | None = None,
        name: str | None = None,
        **kwargs: Any,
    ) -> Run:
        """创建一个聊天模型运行。"""
        if self._schema_format not in {"streaming_events", "original+chat"}:
                        # 请保持此方法未实现以保证向后兼容性。
                        # 当它未实现时，旧版追踪器若使用 "original" 格式
                        # 会回退到 on_llm_start 方法的实现，如果它们
                        # 发现 on_chat_model_start 方法未实现的话。
                        # 最终可通过编写一个 "现代" 追踪器来清理此问题
                        # 该追踪器包含与 "streaming_events" 格式对应的
                        # 所有更新的模式变更。
            msg = (
                f"Chat model tracing is not supported in "
                f"for {self._schema_format} format."
            )
            raise NotImplementedError(msg)
        start_time = datetime.now(timezone.utc)
        if metadata:
            kwargs.update({"metadata": metadata})
        return Run(
            id=run_id,
            parent_run_id=parent_run_id,
            serialized=serialized,
            inputs={"messages": [[dumpd(msg) for msg in batch] for batch in messages]},
            **self._get_run_kwargs(parent_run_id, kwargs),
            events=[{"name": "start", "time": start_time}],
            start_time=start_time,
                        # 警告：此内容仅对 streaming_events 有效。
                        # run_type="llm" 是几乎所有追踪器使用的值。
                        # 将其更改为 "chat_model" 可能会导致 on_llm_start 触发失败
            run_type="chat_model",
            tags=tags,
            name=name,
        )

    def _create_llm_run(
        self,
        serialized: dict[str, Any],
        prompts: list[str],
        run_id: UUID,
        tags: list[str] | None = None,
        parent_run_id: UUID | None = None,
        metadata: dict[str, Any] | None = None,
        name: str | None = None,
        **kwargs: Any,
    ) -> Run:
        """创建一个 LLM 运行。"""
        start_time = datetime.now(timezone.utc)
        if metadata:
            kwargs.update({"metadata": metadata})
        return Run(
            id=run_id,
            parent_run_id=parent_run_id,
            serialized=serialized,
                        # TODO: 解决如何在此处暴露 kwargs
            inputs={"prompts": prompts},
            **self._get_run_kwargs(parent_run_id, kwargs),
            events=[{"name": "start", "time": start_time}],
            start_time=start_time,
            run_type="llm",
            tags=tags or [],
            name=name,
        )

    def _llm_run_with_token_event(
        self,
        token: str | list[str | dict[str, Any]],
        run_id: UUID,
        chunk: GenerationChunk | ChatGenerationChunk | None = None,
        parent_run_id: UUID | None = None,
    ) -> Run:
        """将 token 事件追加到 LLM 运行并返回该运行。"""
        _ = parent_run_id
        llm_run = self._get_run(run_id, run_type={"llm", "chat_model"})
        event_kwargs: dict[str, Any] = {"token": token}
        if chunk:
            event_kwargs["chunk"] = chunk
        llm_run.events.append(
            {
                "name": "new_token",
                "time": datetime.now(timezone.utc),
                "kwargs": event_kwargs,
            },
        )
        return llm_run

    def _llm_run_with_retry_event(
        self,
        retry_state: RetryCallState,
        run_id: UUID,
    ) -> Run:
        llm_run = self._get_run(run_id)
        retry_d: dict[str, Any] = {
            "slept": retry_state.idle_for,
            "attempt": retry_state.attempt_number,
        }
        if retry_state.outcome is None:
            retry_d["outcome"] = "N/A"
        elif retry_state.outcome.failed:
            retry_d["outcome"] = "failed"
            exception = retry_state.outcome.exception()
            retry_d["exception"] = str(exception)
            retry_d["exception_type"] = exception.__class__.__name__
        else:
            retry_d["outcome"] = "success"
            retry_d["result"] = str(retry_state.outcome.result())
        llm_run.events.append(
            {
                "name": "retry",
                "time": datetime.now(timezone.utc),
                "kwargs": retry_d,
            },
        )
        return llm_run

    def _complete_llm_run(self, response: LLMResult, run_id: UUID) -> Run:
        llm_run = self._get_run(run_id, run_type={"llm", "chat_model"})
        if getattr(llm_run, "outputs", None) is None:
            llm_run.outputs = {}
        else:
            llm_run.outputs = cast("dict[str, Any]", llm_run.outputs)
        if not llm_run.extra.get("__omit_auto_outputs", False):
            llm_run.outputs.update(response.model_dump())
        for i, generations in enumerate(response.generations):
            for j, generation in enumerate(generations):
                output_generation = llm_run.outputs["generations"][i][j]
                if "message" in output_generation:
                    output_generation["message"] = dumpd(
                        cast("ChatGeneration", generation).message
                    )
        llm_run.end_time = datetime.now(timezone.utc)
        llm_run.events.append({"name": "end", "time": llm_run.end_time})

        tool_call_count = 0
        for generations in response.generations:
            for generation in generations:
                if hasattr(generation, "message"):
                    msg = generation.message
                    if hasattr(msg, "tool_calls") and msg.tool_calls:
                        tool_call_count += len(msg.tool_calls)
        if tool_call_count > 0:
            llm_run.extra["tool_call_count"] = tool_call_count

        self._attach_gateway_metadata(llm_run, response)

        return llm_run

    def _attach_gateway_metadata(self, run: Run, response: LLMResult) -> None:
        """将 LangSmith 网关元数据从响应提升至运行元数据。"""
        gateway_metadata = _extract_gateway_metadata(response)
        if gateway_metadata is None:
            return
        metadata = run.extra.setdefault("metadata", {})
        metadata[_GATEWAY_RUN_METADATA_KEY] = gateway_metadata
                # 解析后的网关身份优先于请求时间的元数据。
        for source, target in (("model", "ls_model_name"), ("provider", "ls_provider")):
            value = gateway_metadata.get(source)
            if isinstance(value, str) and (value := value.strip()):
                metadata[target] = value

    def _errored_llm_run(
        self, error: BaseException, run_id: UUID, response: LLMResult | None = None
    ) -> Run:
        llm_run = self._get_run(run_id, run_type={"llm", "chat_model"})
        llm_run.error = self._get_stacktrace(error)
        if response:
            if getattr(llm_run, "outputs", None) is None:
                llm_run.outputs = {}
            else:
                llm_run.outputs = cast("dict[str, Any]", llm_run.outputs)
            if not llm_run.extra.get("__omit_auto_outputs", False):
                llm_run.outputs.update(response.model_dump())
            for i, generations in enumerate(response.generations):
                for j, generation in enumerate(generations):
                    output_generation = llm_run.outputs["generations"][i][j]
                    if "message" in output_generation:
                        output_generation["message"] = dumpd(
                            cast("ChatGeneration", generation).message
                        )
            self._attach_gateway_metadata(llm_run, response)
        llm_run.end_time = datetime.now(timezone.utc)
        llm_run.events.append({"name": "error", "time": llm_run.end_time})

        return llm_run

    def _create_chain_run(
        self,
        serialized: dict[str, Any],
        inputs: dict[str, Any],
        run_id: UUID,
        tags: list[str] | None = None,
        parent_run_id: UUID | None = None,
        metadata: dict[str, Any] | None = None,
        run_type: str | None = None,
        name: str | None = None,
        **kwargs: Any,
    ) -> Run:
        """创建一个链运行。"""
        start_time = datetime.now(timezone.utc)
        if metadata:
            kwargs.update({"metadata": metadata})
        return Run(
            id=run_id,
            parent_run_id=parent_run_id,
            serialized=serialized,
            inputs=self._get_chain_inputs(inputs),
            **self._get_run_kwargs(parent_run_id, kwargs),
            events=[{"name": "start", "time": start_time}],
            start_time=start_time,
            child_runs=[],
            run_type=run_type or "chain",
            name=name,
            tags=tags or [],
        )

    def _get_chain_inputs(self, inputs: Any) -> Any:
        """获取链运行的输入。"""
        if self._schema_format in {"original", "original+chat"}:
            return inputs if isinstance(inputs, dict) else {"input": inputs}
        if self._schema_format == "streaming_events":
            return {
                "input": inputs,
            }
        msg = f"Invalid format: {self._schema_format}"
        raise ValueError(msg)

    def _get_chain_outputs(self, outputs: Any) -> Any:
        """获取链运行的输出。"""
        if self._schema_format in {"original", "original+chat"}:
            return outputs if isinstance(outputs, dict) else {"output": outputs}
        if self._schema_format == "streaming_events":
            return {
                "output": outputs,
            }
        msg = f"Invalid format: {self._schema_format}"
        raise ValueError(msg)

    def _complete_chain_run(
        self,
        outputs: dict[str, Any],
        run_id: UUID,
        inputs: dict[str, Any] | None = None,
    ) -> Run:
        """使用输出和结束时间更新链运行。"""
        chain_run = self._get_run(run_id)
        if getattr(chain_run, "outputs", None) is None:
            chain_run.outputs = {}
        if not chain_run.extra.get("__omit_auto_outputs", False):
            cast("dict[str, Any]", chain_run.outputs).update(
                self._get_chain_outputs(outputs)
            )
        chain_run.end_time = datetime.now(timezone.utc)
        chain_run.events.append({"name": "end", "time": chain_run.end_time})
        if inputs is not None:
            chain_run.inputs = self._get_chain_inputs(inputs)
        return chain_run

    def _errored_chain_run(
        self,
        error: BaseException,
        inputs: dict[str, Any] | None,
        run_id: UUID,
    ) -> Run:
        chain_run = self._get_run(run_id)
        chain_run.error = self._get_stacktrace(error)
        chain_run.end_time = datetime.now(timezone.utc)
        chain_run.events.append({"name": "error", "time": chain_run.end_time})
        if inputs is not None:
            chain_run.inputs = self._get_chain_inputs(inputs)
        return chain_run

    def _create_tool_run(
        self,
        serialized: dict[str, Any],
        input_str: str,
        run_id: UUID,
        tags: list[str] | None = None,
        parent_run_id: UUID | None = None,
        metadata: dict[str, Any] | None = None,
        name: str | None = None,
        inputs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Run:
        """创建一个工具运行。"""
        start_time = datetime.now(timezone.utc)
        if metadata:
            kwargs.update({"metadata": metadata})

        if self._schema_format in {"original", "original+chat"}:
            inputs = inputs if isinstance(inputs, dict) else {"input": input_str}
        elif self._schema_format == "streaming_events":
            inputs = {"input": inputs}
        else:
            msg = f"Invalid format: {self._schema_format}"
            raise AssertionError(msg)

        return Run(
            id=run_id,
            parent_run_id=parent_run_id,
            serialized=serialized,
                        # 由于 Run 需要字典对象，因此用 dict 包装。
            inputs=inputs,
            **self._get_run_kwargs(parent_run_id, kwargs),
            events=[{"name": "start", "time": start_time}],
            start_time=start_time,
            child_runs=[],
            run_type="tool",
            tags=tags or [],
            name=name,
        )

    def _complete_tool_run(
        self,
        output: dict[str, Any],
        run_id: UUID,
    ) -> Run:
        """使用输出和结束时间更新工具运行。"""
        tool_run = self._get_run(run_id, run_type="tool")
        if getattr(tool_run, "outputs", None) is None:
            tool_run.outputs = {}
        if not tool_run.extra.get("__omit_auto_outputs", False):
            cast("dict[str, Any]", tool_run.outputs).update({"output": output})
        tool_run.end_time = datetime.now(timezone.utc)
        tool_run.events.append({"name": "end", "time": tool_run.end_time})
        return tool_run

    def _errored_tool_run(
        self,
        error: BaseException,
        run_id: UUID,
    ) -> Run:
        """使用错误和结束时间更新工具运行。"""
        tool_run = self._get_run(run_id, run_type="tool")
        tool_run.error = self._get_stacktrace(error)
        tool_run.end_time = datetime.now(timezone.utc)
        tool_run.events.append({"name": "error", "time": tool_run.end_time})
        return tool_run

    def _create_retrieval_run(
        self,
        serialized: dict[str, Any],
        query: str,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        name: str | None = None,
        **kwargs: Any,
    ) -> Run:
        """创建一个检索运行。"""
        start_time = datetime.now(timezone.utc)
        if metadata:
            kwargs.update({"metadata": metadata})
        return Run(
            id=run_id,
            name=name or "Retriever",
            parent_run_id=parent_run_id,
            serialized=serialized,
            inputs={"query": query},
            **self._get_run_kwargs(parent_run_id, kwargs),
            events=[{"name": "start", "time": start_time}],
            start_time=start_time,
            tags=tags,
            child_runs=[],
            run_type="retriever",
        )

    def _complete_retrieval_run(
        self,
        documents: Sequence[Document],
        run_id: UUID,
    ) -> Run:
        """使用输出和结束时间更新检索运行。"""
        retrieval_run = self._get_run(run_id, run_type="retriever")
        if getattr(retrieval_run, "outputs", None) is None:
            retrieval_run.outputs = {}
        if not retrieval_run.extra.get("__omit_auto_outputs", False):
            cast("dict[str, Any]", retrieval_run.outputs).update(
                {"documents": documents}
            )
        retrieval_run.end_time = datetime.now(timezone.utc)
        retrieval_run.events.append({"name": "end", "time": retrieval_run.end_time})
        return retrieval_run

    def _errored_retrieval_run(
        self,
        error: BaseException,
        run_id: UUID,
    ) -> Run:
        retrieval_run = self._get_run(run_id, run_type="retriever")
        retrieval_run.error = self._get_stacktrace(error)
        retrieval_run.end_time = datetime.now(timezone.utc)
        retrieval_run.events.append({"name": "error", "time": retrieval_run.end_time})
        return retrieval_run

    def __deepcopy__(self, memo: dict[int, Any] | None = None) -> _TracerCore:
        """返回自身深度拷贝。"""
        return self

    def __copy__(self) -> _TracerCore:
        """返回自身拷贝。"""
        return self

    def _end_trace(self, run: Run) -> Coroutine[Any, Any, None] | None:
        """结束一个运行的追踪。

        Args:
            run: 该运行。
        """
        _ = run
        return None

    def _on_run_create(self, run: Run) -> Coroutine[Any, Any, None] | None:
        """在运行创建时进行处理。

        Args:
            run: 创建的运行。
        """
        _ = run
        return None

    def _on_run_update(self, run: Run) -> Coroutine[Any, Any, None] | None:
        """在运行更新时进行处理。

        Args:
            run: 更新后的运行。
        """
        _ = run
        return None

    def _on_llm_start(self, run: Run) -> Coroutine[Any, Any, None] | None:
        """在 LLM 运行开始时进行处理。

        Args:
            run: LLM 运行。
        """
        _ = run
        return None

    def _on_llm_new_token(
        self,
        run: Run,
        token: str | list[str | dict[str, Any]],
        chunk: GenerationChunk | ChatGenerationChunk | None,
    ) -> Coroutine[Any, Any, None] | None:
        """处理新的 LLM token。

        Args:
            run: LLM 运行。
            token: 新的 token，或内容块列表。
            chunk: 可选的分块。
        """
        _ = (run, token, chunk)
        return None

    def _on_llm_end(self, run: Run) -> Coroutine[Any, Any, None] | None:
        """处理 LLM 运行。

        Args:
            run: LLM 运行。
        """
        _ = run
        return None

    def _on_llm_error(self, run: Run) -> Coroutine[Any, Any, None] | None:
        """在 LLM 运行出错时进行处理。

        Args:
            run: LLM 运行。
        """
        _ = run
        return None

    def _on_chain_start(self, run: Run) -> Coroutine[Any, Any, None] | None:
        """在链运行开始时进行处理。

        Args:
            run: 链运行。
        """
        _ = run
        return None

    def _on_chain_end(self, run: Run) -> Coroutine[Any, Any, None] | None:
        """处理链运行。

        Args:
            run: 链运行。
        """
        _ = run
        return None

    def _on_chain_error(self, run: Run) -> Coroutine[Any, Any, None] | None:
        """在链运行出错时进行处理。

        Args:
            run: 链运行。
        """
        _ = run
        return None

    def _on_tool_start(self, run: Run) -> Coroutine[Any, Any, None] | None:
        """在启动时处理工具运行。

        Args:
            run: 工具运行。
        """
        _ = run
        return None

    def _on_tool_end(self, run: Run) -> Coroutine[Any, Any, None] | None:
        """处理工具运行。

        Args:
            run: 工具运行。
        """
        _ = run
        return None

    def _on_tool_error(self, run: Run) -> Coroutine[Any, Any, None] | None:
        """在出错时处理工具运行。

        Args:
            run: 工具运行。
        """
        _ = run
        return None

    def _on_chat_model_start(self, run: Run) -> Coroutine[Any, Any, None] | None:
        """在启动时处理聊天模型运行。

        Args:
            run: 聊天模型运行。
        """
        _ = run
        return None

    def _on_retriever_start(self, run: Run) -> Coroutine[Any, Any, None] | None:
        """在启动时处理检索器运行。

        Args:
            run: 检索器运行。
        """
        _ = run
        return None

    def _on_retriever_end(self, run: Run) -> Coroutine[Any, Any, None] | None:
        """处理检索器运行。

        Args:
            run: 检索器运行。
        """
        _ = run
        return None

    def _on_retriever_error(self, run: Run) -> Coroutine[Any, Any, None] | None:
        """在出错时处理检索器运行。

        Args:
            run: 检索器运行。
        """
        _ = run
        return None

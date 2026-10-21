"""将运行日志流式传输到流中的追踪器。"""

from __future__ import annotations

import asyncio
import contextlib
import copy
import threading
from collections import defaultdict
from pprint import pformat
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    TypeVar,
    overload,
)

import jsonpatch  # type: ignore[import-untyped]
from typing_extensions import NotRequired, TypedDict, override

from langchain_core.callbacks.base import BaseCallbackManager
from langchain_core.load import dumps
from langchain_core.load.load import load
from langchain_core.outputs import ChatGenerationChunk, GenerationChunk
from langchain_core.runnables import RunnableConfig, ensure_config
from langchain_core.tracers._streaming import _StreamingCallbackHandler
from langchain_core.tracers.base import BaseTracer
from langchain_core.tracers.memory_stream import _get_or_create_loop, _MemoryStream

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Iterator, Sequence
    from uuid import UUID

    from langchain_core.runnables import Runnable
    from langchain_core.runnables.utils import Input, Output
    from langchain_core.tracers.schemas import Run


class LogEntry(TypedDict):
    """运行日志中的单个条目。"""

    id: str
    """子运行的 ID。"""

    name: str
    """正在运行的对象的名称。"""

    type: str
    """正在运行的对象的类型，例如 prompt、chain、llm 等。"""

    tags: list[str]
    """运行的标签列表。"""

    metadata: dict[str, Any]
    """运行的元数据键值对。"""

    start_time: str
    """运行开始的 ISO-8601 时间戳。"""

    streamed_output_str: list[str]
    """此运行生成的 LLM token 列表（如适用）。"""

    streamed_output: list[Any]
    """此运行生成的输出块列表（如可用）。"""

    inputs: NotRequired[Any | None]
    """此运行的输入。目前通过 `astream_log` 不可用。"""

    final_output: Any | None
    """此运行的最终输出。

    仅在运行成功完成后可用。
    """

    end_time: str | None
    """运行结束的 ISO-8601 时间戳。

    仅在运行完成后可用。
    """


class RunState(TypedDict):
    """运行状态。"""

    id: str
    """运行的 ID。"""

    streamed_output: list[Any]
    """`Runnable.stream()` 生成的输出块列表。"""

    final_output: Any | None
    """运行的最终输出，通常为对 `streamed_output` 进行聚合（`+`）的结果。

    在支持的情况下，运行过程中持续更新。
    """

    name: str
    """正在运行的对象的名称。"""

    type: str
    """正在运行的对象的类型，例如 prompt、chain、llm 等。"""

        # 我们是否希望在根运行上添加标签/元数据？客户端在大多数情况下其实已知晓。
        # tags: list[str]

    logs: dict[str, LogEntry]
    """运行名称到子运行的映射。

    如果提供了过滤条件，该列表将仅包含匹配过滤条件的运行。
    """


class RunLogPatch:
    """运行日志的补丁。"""

    ops: list[dict[str, Any]]
    """List of `JSONPatch` operations, which describe how to create the run state
    from an empty dict.

    This is the minimal representation of the log, designed to be serialized as JSON and
    sent over the wire to reconstruct the log on the other side. Reconstruction of the
    state can be done with any JSONPatch-compliant library, see https://jsonpatch.com
    for more information.
    """

    def __init__(self, *ops: dict[str, Any]) -> None:
        """创建一个 RunLogPatch。

        Args:
            *ops: 要应用于状态的操作。
        """
        self.ops = list(ops)

    def __add__(self, other: RunLogPatch | Any) -> RunLog:
        """合并两个 `RunLogPatch` 实例。

        Args:
            other: 要合并的另一个 `RunLogPatch`。

        Raises:
            TypeError: 如果其他对象不是 `RunLogPatch`。

        Returns:
            代表两个实例组合的新 `RunLog`。
        """
        if type(other) is RunLogPatch:
            ops = self.ops + other.ops
            state = jsonpatch.apply_patch(None, copy.deepcopy(ops))
            return RunLog(*ops, state=state)

        msg = f"unsupported operand type(s) for +: '{type(self)}' and '{type(other)}'"
        raise TypeError(msg)

    @override
    def __repr__(self) -> str:
                # 1:-1 以去除列表周围的 []
        return f"RunLogPatch({pformat(self.ops)[1:-1]})"

    @override
    def __eq__(self, other: object) -> bool:
        return isinstance(other, RunLogPatch) and self.ops == other.ops

    __hash__ = None  # type: ignore[assignment]


class RunLog(RunLogPatch):
    """运行日志。"""

    state: RunState
    """通过按顺序应用所有操作获得的日志当前状态。"""

    def __init__(self, *ops: dict[str, Any], state: RunState) -> None:
        """创建一个 RunLog。

        Args:
            *ops: 要应用于状态的操作。
            state: 运行日志的初始状态。
        """
        super().__init__(*ops)
        self.state = state

    def __add__(self, other: RunLogPatch | Any) -> RunLog:
        """合并两个 `RunLog` 对象。

        Args:
            other: 要合并的另一个 `RunLog` 或 `RunLogPatch`。

        Raises:
            TypeError: 如果其他对象不是 `RunLog` 或 `RunLogPatch`。

        Returns:
            代表两个实例组合的新 `RunLog`。
        """
        if type(other) is RunLogPatch:
            ops = self.ops + other.ops
            state = jsonpatch.apply_patch(self.state, other.ops)
            return RunLog(*ops, state=state)

        msg = f"unsupported operand type(s) for +: '{type(self)}' and '{type(other)}'"
        raise TypeError(msg)

    @override
    def __repr__(self) -> str:
        return f"RunLog({pformat(self.state)})"

    @override
    def __eq__(self, other: object) -> bool:
        """检查两个 `RunLog` 是否相等。

        Args:
            other: 要比较的另一个 `RunLog`。

        Returns:
            如果 `RunLog` 相等则返回 `True`，否则返回 `False`。
        """
                # 首先比较状态是否相同
        if not isinstance(other, RunLog):
            return False
        if self.state != other.state:
            return False
                # 然后比较操作是否相同
        return super().__eq__(other)

    __hash__ = None


T = TypeVar("T")


class LogStreamCallbackHandler(BaseTracer, _StreamingCallbackHandler[Any]):
    """将运行日志流式传输到流中的追踪器。"""

    def __init__(
        self,
        *,
        auto_close: bool = True,
        include_names: Sequence[str] | None = None,
        include_types: Sequence[str] | None = None,
        include_tags: Sequence[str] | None = None,
        exclude_names: Sequence[str] | None = None,
        exclude_types: Sequence[str] | None = None,
        exclude_tags: Sequence[str] | None = None,
                # 模式格式仅用于内部使用。
        _schema_format: Literal["original", "streaming_events"] = "streaming_events",
    ) -> None:
        """一个将运行日志流式传输到流中的追踪器。

        Args:
            auto_close: Whether to close the stream when the root run finishes.
            include_names: Only include runs from `Runnable` objects with matching
                names.
            include_types: Only include runs from `Runnable` objects with matching
                types.
            include_tags: Only include runs from `Runnable` objects with matching tags.
            exclude_names: Exclude runs from `Runnable` objects with matching names.
            exclude_types: Exclude runs from `Runnable` objects with matching types.
            exclude_tags: Exclude runs from `Runnable` objects with matching tags.
            _schema_format: Primarily changes how the inputs and outputs are handled.

                **仅供内部使用。此 API 将更改。**

                - `'original'` is the format used by all current tracers. This format is
                    slightly inconsistent with respect to inputs and outputs.
                - 'streaming_events' is used for supporting streaming events, for
                    internal usage. It will likely change in the future,
                    or be deprecated entirely in favor of a dedicated async
                    tracer for streaming events.

        Raises:
            ValueError: 如果提供了无效的 schema format（仅供内部使用）。
        """
        if _schema_format not in {"original", "streaming_events"}:
            msg = (
                f"Invalid schema format: {_schema_format}. "
                f"Expected one of 'original', 'streaming_events'."
            )
            raise ValueError(msg)
        super().__init__(_schema_format=_schema_format)

        self.auto_close = auto_close
        self.include_names = include_names
        self.include_types = include_types
        self.include_tags = include_tags
        self.exclude_names = exclude_names
        self.exclude_types = exclude_types
        self.exclude_tags = exclude_tags

        memory_stream = _MemoryStream[RunLogPatch](_get_or_create_loop(self))
        self.lock = threading.Lock()
        self.send_stream = memory_stream.get_send_stream()
        self.receive_stream = memory_stream.get_receive_stream()
        self._key_map_by_run_id: dict[UUID, str] = {}
        self._counter_map_by_name: dict[str, int] = defaultdict(int)
        self.root_id: UUID | None = None

    def __aiter__(self) -> AsyncIterator[RunLogPatch]:
        """遍历运行日志的流。

        Returns:
            一个异步迭代器，用于遍历运行日志补丁。
        """
        return self.receive_stream.__aiter__()

    def send(self, *ops: dict[str, Any]) -> bool:
        """向流发送一个补丁，如果流已关闭则返回 `False`。

        Args:
            *ops: 要发送到流的操作。

        Returns:
            如果补丁成功发送则返回 `True`，如果流已关闭则返回 `False`。
        """
                # 我们可能需要在某个时刻将其包裹在 try / except 中
                # 以处理运行时可能出现的异常。
                # 目前我们让异常向上冒泡，并始终返回
                # 在正常路径下返回 True。
        self.send_stream.send_nowait(RunLogPatch(*ops))
        return True

    async def tap_output_aiter(
        self, run_id: UUID, output: AsyncIterator[T]
    ) -> AsyncIterator[T]:
        """将输出异步迭代器接入，将其值流式传输到日志中。

        Args:
            run_id: 运行的 ID。
            output: 输出异步迭代器。

        Yields:
            输出值。
        """
        async for chunk in output:
                        # 根运行由 .astream_log() 处理
                        # 如果找不到运行则静默忽略
                        # 例如因为该运行未包含在日志中
            if (
                run_id != self.root_id
                and (key := self._key_map_by_run_id.get(run_id))
                and (
                    not self.send(
                        {
                            "op": "add",
                            "path": f"/logs/{key}/streamed_output/-",
                            "value": chunk,
                        }
                    )
                )
            ):
                break

            yield chunk

    def tap_output_iter(self, run_id: UUID, output: Iterator[T]) -> Iterator[T]:
        """将输出迭代器接入，将其值流式传输到日志中。

        Args:
            run_id: 运行的 ID。
            output: 输出迭代器。

        Yields:
            输出值。
        """
        for chunk in output:
                        # 根运行由 .astream_log() 处理
                        # 如果找不到运行则静默忽略
                        # 例如因为该运行未包含在日志中
            if (
                run_id != self.root_id
                and (key := self._key_map_by_run_id.get(run_id))
                and (
                    not self.send(
                        {
                            "op": "add",
                            "path": f"/logs/{key}/streamed_output/-",
                            "value": chunk,
                        }
                    )
                )
            ):
                break

            yield chunk

    def include_run(self, run: Run) -> bool:
        """检查一个 `Run` 是否应包含在日志中。

        Args:
            run: 要检查的 `Run`。

        Returns:
            如果应包含 `Run`，则为 `True`，否则为 `False`。
        """
        if run.id == self.root_id:
            return False

        run_tags = run.tags or []

        if (
            self.include_names is None
            and self.include_types is None
            and self.include_tags is None
        ):
            include = True
        else:
            include = False

        if self.include_names is not None:
            include = include or run.name in self.include_names
        if self.include_types is not None:
            include = include or run.run_type in self.include_types
        if self.include_tags is not None:
            include = include or any(tag in self.include_tags for tag in run_tags)

        if self.exclude_names is not None:
            include = include and run.name not in self.exclude_names
        if self.exclude_types is not None:
            include = include and run.run_type not in self.exclude_types
        if self.exclude_tags is not None:
            include = include and all(tag not in self.exclude_tags for tag in run_tags)

        return include

    def _persist_run(self, run: Run) -> None:
                # 这是遗留方法，仅在整个运行树上调用一次
                # 因此在此处无用
        pass

    def _on_run_create(self, run: Run) -> None:
        """启动一个运行。"""
        if self.root_id is None:
            self.root_id = run.id
            if not self.send(
                {
                    "op": "replace",
                    "path": "",
                    "value": RunState(
                        id=str(run.id),
                        streamed_output=[],
                        final_output=None,
                        logs={},
                        name=run.name,
                        type=run.run_type,
                    ),
                }
            ):
                return

        if not self.include_run(run):
            return

                # 确定前一个索引，加 1
        with self.lock:
            self._counter_map_by_name[run.name] += 1
            count = self._counter_map_by_name[run.name]
            self._key_map_by_run_id[run.id] = (
                run.name if count == 1 else f"{run.name}:{count}"
            )

        entry = LogEntry(
            id=str(run.id),
            name=run.name,
            type=run.run_type,
            tags=run.tags or [],
            metadata=(run.extra or {}).get("metadata", {}),
            start_time=run.start_time.isoformat(timespec="milliseconds"),
            streamed_output=[],
            streamed_output_str=[],
            final_output=None,
            end_time=None,
        )

        if self._schema_format == "streaming_events":
                        # 如果使用流式事件，让我们也添加输入
            entry["inputs"] = _get_standardized_inputs(run, self._schema_format)

                # 将运行添加到流中
        self.send(
            {
                "op": "add",
                "path": f"/logs/{self._key_map_by_run_id[run.id]}",
                "value": entry,
            }
        )

    def _on_run_update(self, run: Run) -> None:
        """完成一个 `Run`。"""
        try:
            index = self._key_map_by_run_id.get(run.id)

            if index is None:
                return

            ops = []

            if self._schema_format == "streaming_events":
                ops.append(
                    {
                        "op": "replace",
                        "path": f"/logs/{index}/inputs",
                        "value": _get_standardized_inputs(run, self._schema_format),
                    }
                )

            ops.extend(
                [
                                        # 用最终输入替换 'inputs'
                                        # 这是因为许多情况下输入直到运行结束后才可知
                                        # 且整个输入流已被 Runnable 处理完毕。
                                        # 为了撤销某些 Runnable / tracer / 等造成的 dumpd
                    {
                        "op": "add",
                        "path": f"/logs/{index}/final_output",
                                                # 新样式链
                        "value": _get_standardized_outputs(run, self._schema_format),
                    },
                    {
                        "op": "add",
                        "path": f"/logs/{index}/end_time",
                        "value": run.end_time.isoformat(timespec="milliseconds")
                        if run.end_time is not None
                        else None,
                    },
                ]
            )

            self.send(*ops)
        finally:
            if run.id == self.root_id and self.auto_close:
                self.send_stream.close()

    def _on_llm_new_token(
        self,
        run: Run,
        token: str | list[str | dict[str, Any]],
        chunk: GenerationChunk | ChatGenerationChunk | None,
    ) -> None:
        """处理新的 LLM token。"""
        index = self._key_map_by_run_id.get(run.id)

        if index is None:
            return

        self.send(
            {
                "op": "add",
                "path": f"/logs/{index}/streamed_output_str/-",
                "value": token,
            },
            {
                "op": "add",
                "path": f"/logs/{index}/streamed_output/-",
                "value": chunk.message
                if isinstance(chunk, ChatGenerationChunk)
                else token,
            },
        )


def _get_standardized_inputs(
    run: Run, schema_format: Literal["original", "streaming_events"]
) -> Any:
    """从 `Run` 中提取标准化输入。

    根据所使用的 Runnable 类型对输入进行标准化。

    Args:
        run: `Run` 对象
        schema_format: 使用的模式格式。

    Returns:
        有效输入仅为 dict。根据约定，输入始终以命名参数形式表示。`None`
        表示输入尚不可知！
    """
    if schema_format == "original":
        msg = (
            "Do not assign inputs with original schema drop the key for now."
            "When inputs are added to astream_log they should be added with "
            "standardized schema for streaming events."
        )
        raise NotImplementedError(msg)

    inputs = load(run.inputs, allowed_objects="messages")

    if run.run_type in {"retriever", "llm", "chat_model"}:
        return inputs

        # 这些在 'inputs' 内部嵌套了额外的 'input' 键，以确保
        # 输入始终为字典。我们需要展开并使用内部值。
        # 我们应该尝试在 Runnables 和回调/追踪器中修复此问题
    inputs = inputs["input"]
        # Runnables 在此处应使用 None 类型而非占位符
        # dict。
        # 针对 Runnables 未使用 None 的临时解决方案
    if inputs == {"input": ""}:  # 输入未知，因此不分配 data['input']
                # 这些之前是在 tracer 之前被 dump 的。
        return None
    return inputs


def _get_standardized_outputs(
    run: Run, schema_format: Literal["original", "streaming_events", "original+chat"]
) -> Any | None:
    """从运行中提取标准化输出。

    根据所使用的 Runnable 类型对输出进行标准化。

    Args:
        run: 运行对象。
        schema_format: 使用的模式格式。

    Returns:
        返回输出，否则为 `None`。
    """
    outputs = load(run.outputs, allowed_objects="messages")
    if schema_format == "original":
        if run.run_type == "prompt" and "output" in outputs:
                        # 现在我们无需再对它们做任何操作。
                        # 返回旧的模式，不进行任何标准化
            return outputs["output"]
                # 将流处理器分配给配置
        return outputs

    if run.run_type in {"retriever", "llm", "chat_model"}:
        return outputs

    if isinstance(outputs, dict):
        return outputs.get("output", None)

    return None


@overload
def _astream_log_implementation(
    runnable: Runnable[Input, Output],
    value: Any,
    config: RunnableConfig | None = None,
    *,
    stream: LogStreamCallbackHandler,
    diff: Literal[True] = True,
    with_streamed_output_list: bool = True,
    **kwargs: Any,
) -> AsyncIterator[RunLogPatch]: ...


@overload
def _astream_log_implementation(
    runnable: Runnable[Input, Output],
    value: Any,
    config: RunnableConfig | None = None,
    *,
    stream: LogStreamCallbackHandler,
    diff: Literal[False],
    with_streamed_output_list: bool = True,
    **kwargs: Any,
) -> AsyncIterator[RunLog]: ...


async def _astream_log_implementation(
    runnable: Runnable[Input, Output],
    value: Any,
    config: RunnableConfig | None = None,
    *,
    stream: LogStreamCallbackHandler,
    diff: bool = True,
    with_streamed_output_list: bool = True,
    **kwargs: Any,
) -> AsyncIterator[RunLogPatch] | AsyncIterator[RunLog]:
    """针对给定 Runnable 的 `astream_log` 实现。

    该实现已被拆分（至少暂时如此），因为 `astream_log` 和 `astream_events` 都依赖它。

    Args:
        runnable: The runnable to run in streaming mode.
        value: The input to the runnable.
        config: The config to pass to the runnable.
        stream: The stream to send the run logs to.
        diff: Whether to yield run log patches (`True`) or full run logs (`False`).
        with_streamed_output_list: Whether to include a list of all streamed outputs in
            each patch. If `False`, only the final output will be included in the
            patches.
        **kwargs: Additional keyword arguments to pass to the `Runnable`.

    Raises:
        ValueError: 如果配置中的回调类型不符合预期。

    Yields:
        运行日志补丁或状态，具体取决于 `diff` 的值。
    """
        # 以流式模式调用 Runnable，
    config = ensure_config(config)
    callbacks = config.get("callbacks")
    if callbacks is None:
        config["callbacks"] = [stream]
    elif isinstance(callbacks, list):
        config["callbacks"] = [*callbacks, stream]
    elif isinstance(callbacks, BaseCallbackManager):
        callbacks = callbacks.copy()
        callbacks.add_handler(stream, inherit=True)
        config["callbacks"] = callbacks
    else:
        msg = (  # type: ignore[unreachable]
            f"Unexpected type for callbacks: {callbacks}."
            "Expected None, list or AsyncCallbackManager."
        )
        raise ValueError(msg)

        # 将每个块添加到输出流中
        # chunk 不能在
    async def consume_astream() -> None:
        try:
            prev_final_output: Output | None = None
            final_output: Output | None = None

            async for chunk in runnable.astream(value, config, **kwargs):
                prev_final_output = final_output
                if final_output is None:
                    final_output = chunk
                else:
                    try:
                        final_output = final_output + chunk  # type: ignore[operator]
                    except TypeError:
                        prev_final_output = None
                        final_output = chunk
                patches: list[dict[str, Any]] = []
                if with_streamed_output_list:
                    patches.append(
                        {
                            "op": "add",
                            "path": "/streamed_output/-",
                                                        # streamed_output 和 final_output 之间共享
                                                        # 否则 jsonpatch.apply 会
                                                        # 同时修改两者
                                                        # 在任务中启动 Runnable，以便开始消费输出
                            "value": copy.deepcopy(chunk),
                        }
                    )
                patches.extend(
                    {**op, "path": f"/final_output{op['path']}"}
                    for op in jsonpatch.JsonPatch.from_diff(
                        prev_final_output, final_output, dumps=dumps
                    )
                )
                await stream.send_stream.send(RunLogPatch(*patches))
        finally:
            await stream.send_stream.aclose()

        # 从输出流中逐个生成块
    task = asyncio.create_task(consume_astream())
    try:
                # 等待 Runnable 完成，除非被取消（例如由 break 触发）
        if diff:
            async for log in stream:
                yield log
        else:
            state = RunLog(state=None)  # type: ignore[arg-type]
            async for log in stream:
                state += log
                yield state
    finally:
                # 等待 Runnable 完成，除非被取消（例如由 break 引起）
        with contextlib.suppress(asyncio.CancelledError):
            await task

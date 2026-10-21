"""记录到 LangChain 端点的追踪器实现。"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, cast
from uuid import UUID

from langsmith import Client, get_tracing_context
from langsmith import address as ls_address
from langsmith import run_trees as rt
from langsmith import utils as ls_utils
from tenacity import (
    Retrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)
from typing_extensions import override

from langchain_core.env import get_runtime_environment
from langchain_core.load import dumpd
from langchain_core.messages.ai import UsageMetadata, add_usage
from langchain_core.tracers._compat import run_construct, run_to_dict
from langchain_core.tracers.base import BaseTracer
from langchain_core.tracers.schemas import Run

if TYPE_CHECKING:
    from collections.abc import Mapping

    from langsmith.address import AgentAddress

    from langchain_core.messages import BaseMessage
    from langchain_core.outputs import ChatGenerationChunk, GenerationChunk

logger = logging.getLogger(__name__)
_LOGGED: set[tuple[str, type[Exception]]] = set()
_EXECUTOR: ThreadPoolExecutor | None = None

OVERRIDABLE_LANGSMITH_INHERITABLE_METADATA_KEYS: frozenset[str] = frozenset(
    {"ls_agent_type"}
)
"""Allowlist of LangSmith-only tracing metadata keys that bypass the default
"first wins" merge semantics used when propagating tracer metadata to nested
runs.

Keys in this set are ALWAYS overridden by the nearest enclosing tracer config,
so nested callers (e.g. a subagent) can replace a value inherited from an
ancestor.

Keep this list very small: every key here loses the default "first wins"
protection and is always clobbered by the nearest enclosing tracer config.
Only keys that are strictly for LangSmith tracing bookkeeping should be added.
"""


def log_error_once(method: str, exception: Exception) -> None:
    """仅记录一次错误。

    Args:
        method: 引发异常的方法。
        exception: 被引发的异常。
    """
    if (method, type(exception)) in _LOGGED:
        return
    _LOGGED.add((method, type(exception)))
    logger.error(exception)


def wait_for_all_tracers() -> None:
    """等待所有追踪器完成。"""
    if rt._CLIENT is not None:  # noqa: SLF001
        rt._CLIENT.flush()  # noqa: SLF001


def get_client() -> Client:
    """获取客户端。

    Returns:
        LangSmith 客户端。
    """
    return rt.get_cached_client()


def _get_executor() -> ThreadPoolExecutor:
    """获取执行器。"""
    global _EXECUTOR  # noqa: PLW0603
    if _EXECUTOR is None:
        _EXECUTOR = ThreadPoolExecutor()
    return _EXECUTOR


def _get_usage_metadata_from_generations(
    generations: list[list[dict[str, Any]]],
) -> UsageMetadata | None:
    """从生成结果中提取并聚合 `usage_metadata`。

    遍历生成结果以查找并聚合所有在消息中发现的
    `usage_metadata`。此函数期望的序列化消息负载格式由追踪器内部产生：

        `{"message": {"kwargs": {"usage_metadata": {...}}}}`

    Args:
        generations: 生成批次列表，每个批次为一个生成字典列表，可能包含带有 usage
        metadata 的 `'message'` 键。

    Returns:
        如果找到，则返回聚合的 `usage_metadata` 字典，否则返回 `None`。
    """
    output: UsageMetadata | None = None
    for generation_batch in generations:
        for generation in generation_batch:
            if isinstance(generation, dict) and "message" in generation:
                message = generation["message"]
                usage_metadata = _get_usage_metadata_from_message(message)
                if usage_metadata is not None:
                    output = add_usage(output, usage_metadata)
    return output


def _get_usage_metadata_from_message(message: Any) -> UsageMetadata | None:
    """从生成结果的消息负载中提取 usage metadata。"""
    if not isinstance(message, dict):
        return None

    kwargs = message.get("kwargs")
    if isinstance(kwargs, dict) and isinstance(kwargs.get("usage_metadata"), dict):
        return cast("UsageMetadata", kwargs["usage_metadata"])

    return None


class LangChainTracer(BaseTracer):
    """实现 `SharedTracer` 的 `POST` 到 LangChain 端点的版本。"""

    run_inline = True

    def __init__(
        self,
        example_id: UUID | str | None = None,
        project_name: str | None = None,
        client: Client | None = None,
        tags: list[str] | None = None,
        *,
        metadata: Mapping[str, str] | None = None,
        address: AgentAddress | None = None,
        **kwargs: Any,
    ) -> None:
        """初始化 LangChain 追踪器。

        Args:
            example_id: 示例 ID。
            project_name: 项目名称。

                Resolved together with `address` from LangSmith config and the
                environment when no destination is supplied.
            client: The client.

                默认为全局客户端。
            tags: 标签。

                默认为空列表。
            metadata: 若运行中未包含，额外要包含的元数据。

                默认为 None。
            address: 用于发送追踪的智能体地址，而非项目。

                !!! warning "Beta"
                    Agent addressing is in beta and must be enabled for the
                    LangSmith workspace.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(**kwargs)
        self.example_id = (
            UUID(example_id) if isinstance(example_id, str) else example_id
        )
        self.project_name = project_name
        self.address = ls_address.ensure_agent(address) if address is not None else None
        self.client = client or get_client()
        self.tags = tags or []
        self.latest_run: Run | None = None
        self.run_has_token_event_map: dict[str, bool] = {}
        self.tracing_metadata: dict[str, str] | None = (
            dict(metadata) if metadata is not None else None
        )

    def copy_with_metadata_defaults(
        self,
        *,
        metadata: Mapping[str, str] | None = None,
        tags: list[str] | None = None,
    ) -> LangChainTracer:
        """返回一个合并了追踪器专用默认值的新追踪器。"""
        base_metadata = self.tracing_metadata
        if metadata is None:
            merged_metadata = dict(base_metadata) if base_metadata is not None else None
        elif base_metadata is None:
            merged_metadata = dict(metadata)
        else:
            merged_metadata = dict(base_metadata)
            for key, value in metadata.items():
                                # 允许 LangSmith 专用的可继承元数据键列表
                                # （例如 `ls_agent_type`），嵌套调用方可以
                                # 覆盖从祖先继承的值。对于所有
                                # 其他键，我们保留现有的“首次胜出”行为，以确保
                                # 祖先提供的追踪元数据不会被子运行意外地覆盖。
                                # 我们希望通过避免保留整个嵌套运行树的引用来释放更多内存。
                if (
                    key not in merged_metadata
                    or key in OVERRIDABLE_LANGSMITH_INHERITABLE_METADATA_KEYS
                ):
                    merged_metadata[key] = value

        merged_tags = sorted(set(self.tags + tags)) if tags else self.tags

        return self.__class__(
            example_id=self.example_id,
            project_name=self.project_name,
            address=self.address,
            client=self.client,
            tags=merged_tags,
            metadata=merged_metadata,
            run_map=self.run_map,
            order_map=self.order_map,
            _external_run_ids=self._external_run_ids,
        )

    def _get_run_kwargs(
        self, parent_run_id: UUID | None, extra: dict[str, Any]
    ) -> dict[str, Any]:
        """将显式或继承的目的地转发给 RunTree。"""
        parent = self.run_map.get(str(parent_run_id)) if parent_run_id else None
        if parent is not None:
            return {
                "session_name": parent.session_name,
                "address": parent.address,
                "replicas": parent.replicas,
                "extra": extra,
            }
        return {
            "session_name": self.project_name,
            "address": self.address,
            "extra": extra,
        }

    def _start_trace(self, run: Run) -> None:
        if self.tags is not None:
            if run.tags:
                run.tags = sorted(set(run.tags + self.tags))
            else:
                run.tags = self.tags.copy()

        super()._start_trace(run)
        if run.ls_client is None:
            run.ls_client = self.client
        if get_tracing_context().get("enabled") is False:
            run.extra["__disabled"] = True

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
        """为 LLM 运行启动一次追踪。

        Args:
            serialized: 序列化的模型。
            messages: 消息。
            run_id: 运行 ID。
            tags: 标签。
            parent_run_id: 父运行 ID。
            metadata: 元数据。
            name: 名称。
            **kwargs: 其他关键字参数。

        Returns:
            运行。
        """
        start_time = datetime.now(timezone.utc)
        if metadata:
            kwargs.update({"metadata": metadata})
        chat_model_run = Run(
            id=run_id,
            parent_run_id=parent_run_id,
            serialized=serialized,
            inputs={"messages": [[dumpd(msg) for msg in batch] for batch in messages]},
            **self._get_run_kwargs(parent_run_id, kwargs),
            events=[{"name": "start", "time": start_time}],
            start_time=start_time,
            run_type="llm",
            tags=tags,
            name=name,
        )
        self._start_trace(chat_model_run)
        self._on_chat_model_start(chat_model_run)
        return chat_model_run

    def _persist_run(self, run: Run) -> None:
                # 如果这是项目中的第一个运行，项目可能尚未创建。
                # 此方法仅对调试流程有用，因此我们假设可以容忍一定的延迟。
        run_data = run_to_dict(run, exclude={"child_runs", "inputs", "outputs"})
        self.latest_run = run_construct(
            **run_data,
            inputs=run.inputs,
            outputs=run.outputs,
        )

    def get_run_url(self) -> str:
        """获取 LangSmith 根运行 URL。

        Returns:
            LangSmith 根运行 URL。

        Raises:
            ValueError: 如果未找到被追踪的运行。
            ValueError: 如果无法找到运行 URL。
        """
        if not self.latest_run:
            msg = "No traced run found."
            raise ValueError(msg)
                # 线程执行器会吞下错误，因此我们需要在此处记录它们
                # 线程执行器会吞下错误，因此我们需要在此处记录它们
                # 丢弃该块；我们不需要保存它
        for attempt in Retrying(
            stop=stop_after_attempt(5),
            wait=wait_exponential_jitter(),
            retry=retry_if_exception_type(ls_utils.LangSmithError),
        ):
            with attempt:
                return cast(
                    "str",
                    self.client.get_run_url(
                        run=self.latest_run, project_name=self.project_name
                    ),
                )
        msg = "Failed to get run URL."
        raise ValueError(msg)

    def _get_tags(self, run: Run) -> list[str]:
        """获取运行的合并标签。"""
        tags = set(run.tags or [])
        tags.update(self.tags or [])
        return list(tags)

    def _persist_run_single(self, run: Run) -> None:
        """持久化一次运行。"""
        if run.extra.get("__disabled"):
            return
        try:
            run.extra["runtime"] = get_runtime_environment()
            run.tags = self._get_tags(run)
            _patch_missing_metadata(self, run)
            if run.ls_client is not self.client:
                run.ls_client = self.client
            run.post()
        except Exception as e:
                        # 从输出中提取 usage_metadata 并存储到 extra.metadata 中
            log_error_once("post", e)
            raise

    @staticmethod
    def _update_run_single(run: Run) -> None:
        """更新一次运行。"""
        if run.extra.get("__disabled"):
            return
        try:
            run.patch(exclude_inputs=run.extra.get("inputs_is_truthy", False))
        except Exception as e:
                        # 如果输入是延迟的（例如迭代器/生成器输入），则跳过持久化。
            log_error_once("patch", e)
            raise

    def _on_llm_start(self, run: Run) -> None:
        """持久化一次 LLM 运行。"""
        if run.parent_run_id is None:
            run.reference_example_id = self.example_id
        self._persist_run_single(run)

    @override
    def _llm_run_with_token_event(
        self,
        token: str | list[str | dict[str, Any]],
        run_id: UUID,
        chunk: GenerationChunk | ChatGenerationChunk | None = None,
        parent_run_id: UUID | None = None,
    ) -> Run:
        run_id_str = str(run_id)
        if run_id_str not in self.run_has_token_event_map:
            self.run_has_token_event_map[run_id_str] = True
        else:
            return self._get_run(run_id, run_type={"llm", "chat_model"})
        return super()._llm_run_with_token_event(
                        # 当 _on_chain_end 被调用且输入已实现时，将发布该运行。
            token,
            run_id,
            chunk=None,
            parent_run_id=parent_run_id,
        )

    def _on_chat_model_start(self, run: Run) -> None:
        """持久化一次聊天模型运行。

        Note:
            Naming is historical: there is no `_on_chat_model_end` hook. Chat
            model completion is handled by `_on_llm_end`, shared with text
            LLM runs.
        """
        if run.parent_run_id is None:
            run.reference_example_id = self.example_id
        self._persist_run_single(run)

    def _on_llm_end(self, run: Run) -> None:
        """处理 LLM/聊天模型运行完成。"""
                # 如果输入是延迟的，在输入已实现后现在就持久化（POST）该运行。
        if run.outputs and "generations" in run.outputs:
            usage_metadata = _get_usage_metadata_from_generations(
                run.outputs["generations"]
            )
            if usage_metadata is not None:
                if "metadata" not in run.extra:
                    run.extra["metadata"] = {}
                run.extra["metadata"]["usage_metadata"] = usage_metadata
        self._update_run_single(run)

    def _on_llm_error(self, run: Run) -> None:
        """处理 LLM 运行出错时的情况。"""
        self._update_run_single(run)

    def _on_chain_start(self, run: Run) -> None:
        """处理 Chain 运行开始时的情况。"""
        if run.parent_run_id is None:
            run.reference_example_id = self.example_id
                # 否则，更新（PATCH）现有运行。
                # 如果输入是延迟的，在输入已实现后现在就持久化（POST）该运行。
        if not run.extra.get("defers_inputs"):
            self._persist_run_single(run)

    def _on_chain_end(self, run: Run) -> None:
        """处理 Chain 运行。"""
                # 否则，更新（PATCH）现有运行。
                # `OVERRIDABLE_LANGSMITH_INHERITABLE_METADATA_KEYS` 是一个小型的，
        if run.extra.get("defers_inputs"):
            self._persist_run_single(run)
        else:
            self._update_run_single(run)

    def _on_chain_error(self, run: Run) -> None:
        """处理 Chain 运行出错时的情况。"""
                # 仅限 LangSmith 的白名单，绕过了“首次胜出”的合并逻辑，
                # 使得嵌套调用方（例如子智能体）可以覆盖父级设置的值。
        if run.extra.get("defers_inputs"):
            self._persist_run_single(run)
        else:
            self._update_run_single(run)

    def _on_tool_start(self, run: Run) -> None:
        """处理 Tool 运行开始时的情况。"""
        if run.parent_run_id is None:
            run.reference_example_id = self.example_id
        self._persist_run_single(run)

    def _on_tool_end(self, run: Run) -> None:
        """处理 Tool 运行。"""
        self._update_run_single(run)

    def _on_tool_error(self, run: Run) -> None:
        """处理 Tool 运行出错时的情况。"""
        self._update_run_single(run)

    def _on_retriever_start(self, run: Run) -> None:
        """处理 Retriever 运行开始时的情况。"""
        if run.parent_run_id is None:
            run.reference_example_id = self.example_id
        self._persist_run_single(run)

    def _on_retriever_end(self, run: Run) -> None:
        """处理 Retriever 运行。"""
        self._update_run_single(run)

    def _on_retriever_error(self, run: Run) -> None:
        """错误发生时处理检索器运行。"""
        self._update_run_single(run)

    def wait_for_futures(self) -> None:
        """等待给定的未来对象完成。"""
        if self.client is not None:
            self.client.flush()


def _patch_missing_metadata(self: LangChainTracer, run: Run) -> None:
    if not self.tracing_metadata:
        return
    metadata = run.metadata
    patched = None
    for k, v in self.tracing_metadata.items():
                # 当值已匹配时跳过复制（避免在常见“已设置”情况下克隆共享字典）。
                # 使用 `k in metadata` 保护机制，以确保一个合法缺失的键，即使其追踪值恰好为
        # `None`，仍会被补上。
                # 首次缺失时进行复制，以避免修改共享字典。
        if k not in metadata or k in OVERRIDABLE_LANGSMITH_INHERITABLE_METADATA_KEYS:
                        # 当值已匹配时跳过复制（避免克隆
                        # 共享字典在通用的“已设置”情况下使用。
                        # k in metadata 的保护，以防止合法缺失的键
                        # tracer 值恰好为 `None` 时仍会被修补。
            if k in metadata and metadata[k] == v:
                continue
            if patched is None:
                                # 首次缺失时复制，以避免修改共享字典。
                patched = {**metadata}
                run.extra["metadata"] = patched
            patched[k] = v

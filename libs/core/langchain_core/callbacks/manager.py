"""运行管理器。"""

from __future__ import annotations

import asyncio
import atexit
import functools
import inspect
import logging
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager, contextmanager
from contextvars import copy_context
from typing import TYPE_CHECKING, Any, TypeVar, cast

from typing_extensions import Self, override

from langchain_core.callbacks.base import (
    BaseCallbackHandler,
    BaseCallbackManager,
    Callbacks,
    ChainManagerMixin,
    LLMManagerMixin,
    RetrieverManagerMixin,
    RunManagerMixin,
    ToolManagerMixin,
)
from langchain_core.callbacks.stdout import StdOutCallbackHandler
from langchain_core.globals import get_debug
from langchain_core.messages import BaseMessage, get_buffer_string
from langchain_core.utils.env import env_var_is_set
from langchain_core.utils.uuid import uuid7

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Coroutine, Generator, Sequence
    from uuid import UUID

    from langchain_protocol.protocol import MessagesData
    from langsmith.address import AgentAddress
    from tenacity import RetryCallState

    from langchain_core.agents import AgentAction, AgentFinish
    from langchain_core.documents import Document
    from langchain_core.outputs import ChatGenerationChunk, GenerationChunk, LLMResult
    from langchain_core.runnables.config import RunnableConfig
    from langchain_core.tracers.schemas import Run

logger = logging.getLogger(__name__)


def _get_debug() -> bool:
    return get_debug()


@contextmanager
def trace_as_chain_group(
    group_name: str,
    callback_manager: CallbackManager | None = None,
    *,
    inputs: dict[str, Any] | None = None,
    project_name: str | None = None,
    address: AgentAddress | None = None,
    example_id: str | UUID | None = None,
    run_id: UUID | None = None,
    tags: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> Generator[CallbackManagerForChainGroup, None, None]:
    """在上下文管理器中获取链组的回调管理器。

    Useful for grouping different calls together as a single run even if they aren't
    composed in a single chain.

    Args:
        group_name: 链组的名称。 callback_manager: 要使用的回调管理器。 inputs:
        链组的输入。 project_name: 项目的名称。 address:
        代替项目向代理地址发送追踪信息。 !!! warning "测试版" 必须在 LangSmith
        工作区启用代理地址功能。 example_id: 示例的 ID。 run_id: 运行的 ID。 tags:
        应用于所有运行的可继承标签。 metadata: 应用于所有运行的元数据。

            !!! warning "Beta"
                Agent addressing must be enabled for the LangSmith workspace.
        example_id: The ID of the example.
        run_id: The ID of the run.
        tags: The inheritable tags to apply to all runs.
        metadata: The metadata to apply to all runs.

    必须将 `LANGCHAIN_TRACING_V2` 环境变量设为 true 才能在 LangSmith 中查看追踪信息。

        链组的回调管理器。

    Yields:
        在上下文管理器中获取链组的异步回调管理器。

    Example:
        ```python
        llm_input = "Foo"
        with trace_as_chain_group("group_name", inputs={"input": llm_input}) as manager:
            # Use the callback manager for the chain group
            res = llm.invoke(llm_input, {"callbacks": manager})
            manager.on_chain_end({"output": res})
        ```
    """
    from langchain_core.tracers.context import (  # noqa: PLC0415 -- deferred to avoid importing langsmith at module level
        _get_trace_callbacks,
    )

    cb = _get_trace_callbacks(
        project_name, example_id, callback_manager=callback_manager, address=address
    )
    cm = CallbackManager.configure(
        inheritable_callbacks=cb,
        inheritable_tags=tags,
        inheritable_metadata=metadata,
    )

    run_manager = cm.on_chain_start({"name": group_name}, inputs or {}, run_id=run_id)
    child_cm = run_manager.get_child()
    group_cm = CallbackManagerForChainGroup(
        child_cm.handlers,
        child_cm.inheritable_handlers,
        child_cm.parent_run_id,
        parent_run_manager=run_manager,
        tags=child_cm.tags,
        inheritable_tags=child_cm.inheritable_tags,
        metadata=child_cm.metadata,
        inheritable_metadata=child_cm.inheritable_metadata,
    )
    try:
        yield group_cm
    except BaseException as e:
        if not group_cm.ended:
            run_manager.on_chain_error(e)
        raise
    else:
        if not group_cm.ended:
            run_manager.on_chain_end({})


@asynccontextmanager
async def atrace_as_chain_group(
    group_name: str,
    callback_manager: AsyncCallbackManager | None = None,
    *,
    inputs: dict[str, Any] | None = None,
    project_name: str | None = None,
    address: AgentAddress | None = None,
    example_id: str | UUID | None = None,
    run_id: UUID | None = None,
    tags: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> AsyncGenerator[AsyncCallbackManagerForChainGroup, None]:
    """用于将不同的异步调用分组为单个运行，即使它们未在单一链中组合。

    Useful for grouping different async calls together as a single run even if they
    aren't composed in a single chain.

    Args:
        group_name: The name of the chain group.
        callback_manager: The async callback manager to use, which manages tracing and
            other callback behavior.
        inputs: The inputs to the chain group.
        project_name: The name of the project.
        address: The agent address to send traces to instead of a project.

            !!! warning "Beta"
                Agent addressing must be enabled for the LangSmith workspace.
        example_id: The ID of the example.
        run_id: The ID of the run.
        tags: The inheritable tags to apply to all runs.
        metadata: The metadata to apply to all runs.

    Yields:
        必须将 `LANGCHAIN_TRACING_V2` 环境变量设为 true 才能在 LangSmith 中查看追踪信息。

    确保可等待的方法始终免于取消。

        func: 要屏蔽的函数。

    Example:
        ```python
        llm_input = "Foo"
        async with atrace_as_chain_group(
            "group_name", inputs={"input": llm_input}
        ) as manager:
            # Use the async callback manager for the chain group
            res = await llm.ainvoke(llm_input, {"callbacks": manager})
            await manager.on_chain_end({"output": res})
        ```
    """
    from langchain_core.tracers.context import (  # noqa: PLC0415 -- deferred to avoid importing langsmith at module level
        _get_trace_callbacks,
    )

    cb = _get_trace_callbacks(
        project_name, example_id, callback_manager=callback_manager, address=address
    )
    cm = AsyncCallbackManager.configure(
        inheritable_callbacks=cb, inheritable_tags=tags, inheritable_metadata=metadata
    )

    run_manager = await cm.on_chain_start(
        {"name": group_name}, inputs or {}, run_id=run_id
    )
    child_cm = run_manager.get_child()
    group_cm = AsyncCallbackManagerForChainGroup(
        child_cm.handlers,
        child_cm.inheritable_handlers,
        child_cm.parent_run_id,
        parent_run_manager=run_manager,
        tags=child_cm.tags,
        inheritable_tags=child_cm.inheritable_tags,
        metadata=child_cm.metadata,
        inheritable_metadata=child_cm.inheritable_metadata,
    )
    try:
        yield group_cm
    except BaseException as e:
        if not group_cm.ended:
            await run_manager.on_chain_error(e)
        raise
    else:
        if not group_cm.ended:
            await run_manager.on_chain_end({})


Func = TypeVar("Func", bound=Callable[..., Any])


def shielded(func: Func) -> Func:
    """屏蔽后的函数

    Args:
        func: 将异步 `on_chat_model_start` 协程包装为带有回退机制的版本。

    Returns:
        捕获 `NotImplementedError` 并触发 `on_llm_start` 回退。 此操作覆盖从 **同步**
        `handle_event` 调用中调用的异步处理器， 其中协程被收集到 `coros` 中，并稍后由
        `_run_coros` 执行。若无此包装， `NotImplementedError` 将被 `_run_coros`
        通用捕获，导致追踪丢失。

    """

    @functools.wraps(func)
    async def wrapped(*args: Any, **kwargs: Any) -> Any:
                # 捕获当前上下文以保留上下文变量
        ctx = copy_context()

                # 创建协程
        coro = func(*args, **kwargs)

                # 对于 Python 3.11+，使用显式上下文创建任务
                # 对于旧版本，回退到原始行为
        try:
                        # 使用捕获的上下文创建任务以保留上下文变量
            task = asyncio.create_task(coro, context=ctx)  # type: ignore[call-arg, unused-ignore]
                        # `call-arg` 用于避免在 3.9 或 3.10 测试中失败
            return await asyncio.shield(task)
        except TypeError:
                        # Python < 3.11 回退 - 正常创建任务后屏蔽
                        # 这不能完美保留上下文，但总比没有好
            task = asyncio.create_task(coro)
            return await asyncio.shield(task)

    return cast("Func", wrapped)


async def _achat_model_start_fallback(
    coro: Coroutine[Any, Any, Any],
    handler: BaseCallbackHandler,
    *args: Any,
    **kwargs: Any,
) -> None:
    """适用于 `CallbackManager` 的通用事件处理器。

    Catches `NotImplementedError` and triggers the `on_llm_start` fallback.
    This covers async handlers invoked from a **sync** `handle_event` call,
    where the coroutine is collected into `coros` and executed later by
    `_run_coros`. Without this wrapper the `NotImplementedError` would be
    caught generically by `_run_coros` and the trace would be lost.
    """
    try:
        await coro
    except NotImplementedError:
        message_strings = [get_buffer_string(m) for m in args[1]]
        await _ahandle_event_for_handler(
            handler,
            "on_llm_start",
            "ignore_llm",
            args[0],
            message_strings,
            *args[2:],
            **kwargs,
        )


def handle_event(
    handlers: list[BaseCallbackHandler],
    event_name: str,
    ignore_condition_name: str | None,
    *args: Any,
    **kwargs: Any,
) -> None:
    """适用于 `AsyncCallbackManager` 的异步通用事件处理器。

    Args:
        handlers: 将处理事件的处理器列表。
        event_name: 事件名称（例如 `'on_llm_start'`）。
        ignore_condition_name: 定义在处理器上的属性名称，若其值为 `True`
            则会导致该处理器跳过当前事件的处理。
        *args: 传递给事件处理器的参数。
        **kwargs: 传递给事件处理器的关键字参数。

    """
    coros: list[Coroutine[Any, Any, Any]] = []

    try:
        message_strings: list[str] | None = None
        for handler in handlers:
            try:
                if ignore_condition_name is None or not getattr(
                    handler, ignore_condition_name
                ):
                    event = getattr(handler, event_name)(*args, **kwargs)
                    if asyncio.iscoroutine(event):
                        if event_name == "on_chat_model_start":
                            event = _achat_model_start_fallback(
                                event, handler, *args, **kwargs
                            )
                        coros.append(event)
            except NotImplementedError as e:
                if event_name == "on_chat_model_start":
                    if message_strings is None:
                        message_strings = [get_buffer_string(m) for m in args[1]]
                    handle_event(
                        [handler],
                        "on_llm_start",
                        "ignore_llm",
                        args[0],
                        message_strings,
                        *args[2:],
                        **kwargs,
                    )
                else:
                    handler_name = handler.__class__.__name__
                    logger.warning(
                        "NotImplementedError in %s.%s callback: %s",
                        handler_name,
                        event_name,
                        repr(e),
                    )
            except Exception as e:
                logger.warning(
                    "Error in %s.%s callback: %s",
                    handler.__class__.__name__,
                    event_name,
                    repr(e),
                )
                if handler.raise_error:
                    raise
    finally:
        if coros:
            try:
                                # 如果没有当前事件循环，则引发 RuntimeError。
                asyncio.get_running_loop()
                loop_running = True
            except RuntimeError:
                loop_running = False

            if loop_running:
                                # 如果我们尝试将此协程提交给正在运行的循环
                                # 我们会陷入死锁，因为我们是从一个正在运行的协程中到达此处的，而我们无法中断它来运行这个协程。
                                # 解决方案是将同步函数在全局共享的
                                # 线程池执行器上运行，以避免阻塞主事件循环。
                                # 如果没有运行中的循环，我们可以直接运行协程。
                _executor().submit(copy_context().run, _run_coros, coros).result()
            else:
                                # 注意：这些协程引发的异常始终会被记录并吞下
                _run_coros(coros)


def _run_coros(coros: list[Coroutine[Any, Any, Any]]) -> None:
        # 无论处理程序的 `raise_error` 设置如何。通过同步 `handle_event`
    # 驱动的异步处理程序错误永远不会传播，与同步处理程序的错误不同（后者尊重
    # `raise_error`）。这是同步和异步回调路径之间已存在的不对称性。
        # Python 3.11+
        # 在新事件循环中运行协程，需注意
        # - 安装信号处理器
        # - 运行由 `coros` 调度的待处理任务
    if hasattr(asyncio, "Runner"):
                # - 关闭异步生成器和执行器
                # - 关闭循环
                # 运行协程，获取结果
                # 运行由 coros 调度的待处理任务，直到全部完成
                # 在 Python 3.11 之前，我们需要在新事件循环中运行每个协程
                # 因为 Runner API 不可用。
        with asyncio.Runner() as runner:
                        # 不能有相同运行 ID 的重复运行（如果提供了）
            for coro in coros:
                try:
                    runner.run(coro)
                except Exception as e:
                    logger.warning("Error in callback coroutine: %s", repr(e))

                        # 按顺序运行内联任务
            while pending := asyncio.all_tasks(runner.get_loop()):
                runner.run(asyncio.wait(pending))
    else:
                # 并发运行非内联任务
                # 按顺序运行内联任务
        for coro in coros:
            try:
                asyncio.run(coro)
            except Exception as e:
                logger.warning("Error in callback coroutine: %s", repr(e))


async def _ahandle_event_for_handler(
    handler: BaseCallbackHandler,
    event_name: str,
    ignore_condition_name: str | None,
    *args: Any,
    **kwargs: Any,
) -> None:
    try:
        if ignore_condition_name is None or not getattr(handler, ignore_condition_name):
            event = getattr(handler, event_name)
            if inspect.iscoroutinefunction(event):
                await event(*args, **kwargs)
            elif handler.run_inline:
                event(*args, **kwargs)
            else:
                await asyncio.get_running_loop().run_in_executor(
                    None,
                    functools.partial(copy_context().run, event, *args, **kwargs),
                )
    except NotImplementedError as e:
        if event_name == "on_chat_model_start":
            message_strings = [get_buffer_string(m) for m in args[1]]
            await _ahandle_event_for_handler(
                handler,
                "on_llm_start",
                "ignore_llm",
                args[0],
                message_strings,
                *args[2:],
                **kwargs,
            )
        else:
            logger.warning(
                "NotImplementedError in %s.%s callback: %s",
                handler.__class__.__name__,
                event_name,
                repr(e),
            )
    except Exception as e:
        logger.warning(
            "Error in %s.%s callback: %s",
            handler.__class__.__name__,
            event_name,
            repr(e),
        )
        if handler.raise_error:
            raise


async def ahandle_event(
    handlers: list[BaseCallbackHandler],
    event_name: str,
    ignore_condition_name: str | None,
    *args: Any,
    **kwargs: Any,
) -> None:
    """AsyncCallbackManager 的通用异步事件处理器。

    Args:
        handlers: The list of handlers that will handle the event.
        event_name: The name of the event (e.g., `'on_llm_start'`).
        ignore_condition_name: Name of the attribute defined on handler that if `True`
            will cause the handler to be skipped for the given event.
        *args: The arguments to pass to the event handler.
        **kwargs: The keyword arguments to pass to the event handler.

    """
    for handler in [h for h in handlers if h.run_inline]:
        await _ahandle_event_for_handler(
            handler, event_name, ignore_condition_name, *args, **kwargs
        )
    await asyncio.gather(
        *(
            _ahandle_event_for_handler(
                handler,
                event_name,
                ignore_condition_name,
                *args,
                **kwargs,
            )
            for handler in handlers
            if not handler.run_inline
        )
    )


class BaseRunManager(RunManagerMixin):
    """运行管理器的基础类（绑定的回调管理器）。"""

    def __init__(
        self,
        *,
        run_id: UUID,
        handlers: list[BaseCallbackHandler],
        inheritable_handlers: list[BaseCallbackHandler],
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        inheritable_tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        inheritable_metadata: dict[str, Any] | None = None,
    ) -> None:
        """初始化运行管理器。

        Args:
            run_id: 运行的 ID。
            handlers: 处理程序列表。
            inheritable_handlers: 可继承处理程序列表。
            parent_run_id: 父运行的 ID。
            tags: 标签列表。
            inheritable_tags: 可继承标签列表。
            metadata: 元数据。
            inheritable_metadata: 可继承元数据。

        """
        self.run_id = run_id
        self.handlers = handlers
        self.inheritable_handlers = inheritable_handlers
        self.parent_run_id = parent_run_id
        self.tags = tags or []
        self.inheritable_tags = inheritable_tags or []
        self.metadata = metadata or {}
        self.inheritable_metadata = inheritable_metadata or {}

    @classmethod
    def get_noop_manager(cls) -> Self:
        """返回一个不执行任何操作的管理器。

        Returns:
            无操作管理器。

        """
        return cls(
            run_id=uuid7(),
            handlers=[],
            inheritable_handlers=[],
            tags=[],
            inheritable_tags=[],
            metadata={},
            inheritable_metadata={},
        )


class RunManager(BaseRunManager):
    """同步运行管理器。"""

    def on_text(
        self,
        text: str,
        **kwargs: Any,
    ) -> None:
        """在接收到文本时运行。

        Args:
            text: 接收到的文本。
            **kwargs: 额外的关键字参数。
        """
        if not self.handlers:
            return
        handle_event(
            self.handlers,
            "on_text",
            None,
            text,
            run_id=self.run_id,
            parent_run_id=self.parent_run_id,
            tags=self.tags,
            **kwargs,
        )

    def on_retry(
        self,
        retry_state: RetryCallState,
        **kwargs: Any,
    ) -> None:
        """在接收到重试时运行。

        Args:
            retry_state: 重试状态。
            **kwargs: 额外的关键字参数。

        """
        if not self.handlers:
            return
        handle_event(
            self.handlers,
            "on_retry",
            "ignore_retry",
            retry_state,
            run_id=self.run_id,
            parent_run_id=self.parent_run_id,
            tags=self.tags,
            **kwargs,
        )


class ParentRunManager(RunManager):
    """同步父级运行管理器。"""

    def get_child(self, tag: str | None = None) -> CallbackManager:
        """获取子回调管理器。

        Args:
            tag: 子回调管理器的标签。

        Returns:
            子回调管理器。

        """
        manager = CallbackManager(handlers=[], parent_run_id=self.run_id)
        manager.set_handlers(self.inheritable_handlers)
        manager.add_tags(self.inheritable_tags)
        manager.add_metadata(self.inheritable_metadata)
        if tag is not None:
            manager.add_tags([tag], inherit=False)
        return manager


class AsyncRunManager(BaseRunManager, ABC):
    """异步运行管理器。"""

    @abstractmethod
    def get_sync(self) -> RunManager:
        """获取等价的同步 `RunManager`。

        Returns:
            同步 `RunManager`。

        """

    async def on_text(
        self,
        text: str,
        **kwargs: Any,
    ) -> None:
        """在接收到文本时运行。

        Args:
            text: 接收到的文本。
            **kwargs: 额外的关键字参数。
        """
        if not self.handlers:
            return
        await ahandle_event(
            self.handlers,
            "on_text",
            None,
            text,
            run_id=self.run_id,
            parent_run_id=self.parent_run_id,
            tags=self.tags,
            **kwargs,
        )

    async def on_retry(
        self,
        retry_state: RetryCallState,
        **kwargs: Any,
    ) -> None:
        """异步在接收到重试时运行。

        Args:
            retry_state: 重试状态。
            **kwargs: 额外的关键字参数。

        """
        if not self.handlers:
            return
        await ahandle_event(
            self.handlers,
            "on_retry",
            "ignore_retry",
            retry_state,
            run_id=self.run_id,
            parent_run_id=self.parent_run_id,
            tags=self.tags,
            **kwargs,
        )


class AsyncParentRunManager(AsyncRunManager):
    """异步父级运行管理器。"""

    def get_child(self, tag: str | None = None) -> AsyncCallbackManager:
        """获取子回调管理器。

        Args:
            tag: 子回调管理器的标签。

        Returns:
            子回调管理器。

        """
        manager = AsyncCallbackManager(handlers=[], parent_run_id=self.run_id)
        manager.set_handlers(self.inheritable_handlers)
        manager.add_tags(self.inheritable_tags)
        manager.add_metadata(self.inheritable_metadata)
        if tag is not None:
            manager.add_tags([tag], inherit=False)
        return manager


class CallbackManagerForLLMRun(RunManager, LLMManagerMixin):
    """LLM 运行的回调管理器。"""

    def on_llm_new_token(
        self,
        token: str | list[str | dict[str, Any]],
        *,
        chunk: GenerationChunk | ChatGenerationChunk | None = None,
        **kwargs: Any,
    ) -> None:
        """在 LLM 生成新 token 时运行。

        Args:
            token: 新的 token，或内容块列表。
            chunk: 数据块。
            **kwargs: 额外的关键字参数。

        """
        if not self.handlers:
            return
        handle_event(
            self.handlers,
            "on_llm_new_token",
            "ignore_llm",
            token=token,
            run_id=self.run_id,
            parent_run_id=self.parent_run_id,
            tags=self.tags,
            chunk=chunk,
            **kwargs,
        )

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """在 LLM 运行结束时运行。

        Args:
            response: LLM 的结果。
            **kwargs: 额外的关键字参数。

        """
        if not self.handlers:
            return
        handle_event(
            self.handlers,
            "on_llm_end",
            "ignore_llm",
            response,
            run_id=self.run_id,
            parent_run_id=self.parent_run_id,
            tags=self.tags,
            **kwargs,
        )

    def on_llm_error(
        self,
        error: BaseException,
        **kwargs: Any,
    ) -> None:
        """在 LLM 出错时运行。

        Args:
            error: 错误。
            **kwargs: 额外的关键字参数。

                - response (LLMResult): 错误发生前生成的响应。
        """
        if not self.handlers:
            return
        handle_event(
            self.handlers,
            "on_llm_error",
            "ignore_llm",
            error,
            run_id=self.run_id,
            parent_run_id=self.parent_run_id,
            tags=self.tags,
            **kwargs,
        )

    def on_stream_event(self, event: MessagesData, **kwargs: Any) -> None:
        """在 `stream_events(version="v3")` 的每个协议事件上运行。

        Args:
            event: 协议事件。
            **kwargs: 额外的关键字参数。
        """
        if not self.handlers:
            return
        handle_event(
            self.handlers,
            "on_stream_event",
            "ignore_llm",
            event,
            run_id=self.run_id,
            parent_run_id=self.parent_run_id,
            tags=self.tags,
            **kwargs,
        )


class AsyncCallbackManagerForLLMRun(AsyncRunManager, LLMManagerMixin):
    """LLM 运行的异步回调管理器。"""

    def get_sync(self) -> CallbackManagerForLLMRun:
        """获取等价的同步 `RunManager`。

        Returns:
            同步 `RunManager`。

        """
        return CallbackManagerForLLMRun(
            run_id=self.run_id,
            handlers=self.handlers,
            inheritable_handlers=self.inheritable_handlers,
            parent_run_id=self.parent_run_id,
            tags=self.tags,
            inheritable_tags=self.inheritable_tags,
            metadata=self.metadata,
            inheritable_metadata=self.inheritable_metadata,
        )

    async def on_llm_new_token(
        self,
        token: str | list[str | dict[str, Any]],
        *,
        chunk: GenerationChunk | ChatGenerationChunk | None = None,
        **kwargs: Any,
    ) -> None:
        """在 LLM 生成新 token 时运行。

        Args:
            token: 新的 token，或内容块列表。
            chunk: 数据块。
            **kwargs: 额外的关键字参数。

        """
        if not self.handlers:
            return
        await ahandle_event(
            self.handlers,
            "on_llm_new_token",
            "ignore_llm",
            token,
            chunk=chunk,
            run_id=self.run_id,
            parent_run_id=self.parent_run_id,
            tags=self.tags,
            **kwargs,
        )

    @shielded
    async def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """在 LLM 运行结束时运行。

        Args:
            response: LLM 的结果。
            **kwargs: 额外的关键字参数。

        """
        if not self.handlers:
            return
        await ahandle_event(
            self.handlers,
            "on_llm_end",
            "ignore_llm",
            response,
            run_id=self.run_id,
            parent_run_id=self.parent_run_id,
            tags=self.tags,
            **kwargs,
        )

    @shielded
    async def on_llm_error(
        self,
        error: BaseException,
        **kwargs: Any,
    ) -> None:
        """在 LLM 出错时运行。

        Args:
            error: 错误。
            **kwargs: 额外的关键字参数。

                - response (LLMResult): 错误发生前生成的响应。

        """
        if not self.handlers:
            return
        await ahandle_event(
            self.handlers,
            "on_llm_error",
            "ignore_llm",
            error,
            run_id=self.run_id,
            parent_run_id=self.parent_run_id,
            tags=self.tags,
            **kwargs,
        )

    async def on_stream_event(self, event: MessagesData, **kwargs: Any) -> None:
        """在 `astream_events(version="v3")` 的每个协议事件上运行。

        Args:
            event: 协议事件。
            **kwargs: 额外的关键字参数。
        """
        if not self.handlers:
            return
        await ahandle_event(
            self.handlers,
            "on_stream_event",
            "ignore_llm",
            event,
            run_id=self.run_id,
            parent_run_id=self.parent_run_id,
            tags=self.tags,
            **kwargs,
        )


class CallbackManagerForChainRun(ParentRunManager, ChainManagerMixin):
    """链运行的回调管理器。"""

    def on_chain_end(self, outputs: dict[str, Any] | Any, **kwargs: Any) -> None:
        """在链运行结束时运行。

        Args:
            outputs: 链的输出。
            **kwargs: 额外的关键字参数。

        """
        if not self.handlers:
            return
        handle_event(
            self.handlers,
            "on_chain_end",
            "ignore_chain",
            outputs,
            run_id=self.run_id,
            parent_run_id=self.parent_run_id,
            tags=self.tags,
            **kwargs,
        )

    def on_chain_error(
        self,
        error: BaseException,
        **kwargs: Any,
    ) -> None:
        """在链出错时运行。

        Args:
            error: 错误。
            **kwargs: 额外的关键字参数。

        """
        if not self.handlers:
            return
        handle_event(
            self.handlers,
            "on_chain_error",
            "ignore_chain",
            error,
            run_id=self.run_id,
            parent_run_id=self.parent_run_id,
            tags=self.tags,
            **kwargs,
        )

    def on_agent_action(self, action: AgentAction, **kwargs: Any) -> None:
        """智能体动作接收时运行。

        Args:
            action: 智能体动作。
            **kwargs: 额外的关键字参数。
        """
        if not self.handlers:
            return
        handle_event(
            self.handlers,
            "on_agent_action",
            "ignore_agent",
            action,
            run_id=self.run_id,
            parent_run_id=self.parent_run_id,
            tags=self.tags,
            **kwargs,
        )

    def on_agent_finish(self, finish: AgentFinish, **kwargs: Any) -> None:
        """智能体完成接收时运行。

        Args:
            finish: 智能体完成。
            **kwargs: 额外的关键字参数。
        """
        if not self.handlers:
            return
        handle_event(
            self.handlers,
            "on_agent_finish",
            "ignore_agent",
            finish,
            run_id=self.run_id,
            parent_run_id=self.parent_run_id,
            tags=self.tags,
            **kwargs,
        )


class AsyncCallbackManagerForChainRun(AsyncParentRunManager, ChainManagerMixin):
    """链运行的异步回调管理器。"""

    def get_sync(self) -> CallbackManagerForChainRun:
        """获取对应的同步 `RunManager`。

        Returns:
            对应的同步 `RunManager`。
        """
        return CallbackManagerForChainRun(
            run_id=self.run_id,
            handlers=self.handlers,
            inheritable_handlers=self.inheritable_handlers,
            parent_run_id=self.parent_run_id,
            tags=self.tags,
            inheritable_tags=self.inheritable_tags,
            metadata=self.metadata,
            inheritable_metadata=self.inheritable_metadata,
        )

    @shielded
    async def on_chain_end(self, outputs: dict[str, Any] | Any, **kwargs: Any) -> None:
        """链运行结束时运行。

        Args:
            outputs: 链的输出。
            **kwargs: 额外的关键字参数。

        """
        if not self.handlers:
            return
        await ahandle_event(
            self.handlers,
            "on_chain_end",
            "ignore_chain",
            outputs,
            run_id=self.run_id,
            parent_run_id=self.parent_run_id,
            tags=self.tags,
            **kwargs,
        )

    @shielded
    async def on_chain_error(
        self,
        error: BaseException,
        **kwargs: Any,
    ) -> None:
        """链出错时运行。

        Args:
            error: 错误。
            **kwargs: 额外的关键字参数。

        """
        if not self.handlers:
            return
        await ahandle_event(
            self.handlers,
            "on_chain_error",
            "ignore_chain",
            error,
            run_id=self.run_id,
            parent_run_id=self.parent_run_id,
            tags=self.tags,
            **kwargs,
        )

    async def on_agent_action(self, action: AgentAction, **kwargs: Any) -> None:
        """智能体动作接收时运行。

        Args:
            action: 智能体动作。
            **kwargs: 额外的关键字参数。
        """
        if not self.handlers:
            return
        await ahandle_event(
            self.handlers,
            "on_agent_action",
            "ignore_agent",
            action,
            run_id=self.run_id,
            parent_run_id=self.parent_run_id,
            tags=self.tags,
            **kwargs,
        )

    async def on_agent_finish(self, finish: AgentFinish, **kwargs: Any) -> None:
        """智能体完成接收时运行。

        Args:
            finish: 智能体完成。
            **kwargs: 额外的关键字参数。
        """
        if not self.handlers:
            return
        await ahandle_event(
            self.handlers,
            "on_agent_finish",
            "ignore_agent",
            finish,
            run_id=self.run_id,
            parent_run_id=self.parent_run_id,
            tags=self.tags,
            **kwargs,
        )


class CallbackManagerForToolRun(ParentRunManager, ToolManagerMixin):
    """工具运行的回调管理器。"""

    def on_tool_end(
        self,
        output: Any,
        **kwargs: Any,
    ) -> None:
        """工具运行结束时运行。

        Args:
            output: 工具的输出。
            **kwargs: 传递给事件处理器的关键字参数

        """
        if not self.handlers:
            return
        handle_event(
            self.handlers,
            "on_tool_end",
            "ignore_agent",
            output,
            run_id=self.run_id,
            parent_run_id=self.parent_run_id,
            tags=self.tags,
            **kwargs,
        )

    def on_tool_error(
        self,
        error: BaseException,
        **kwargs: Any,
    ) -> None:
        """工具出错时运行。

        Args:
            error: 错误。
            **kwargs: 额外的关键字参数。

        """
        if not self.handlers:
            return
        handle_event(
            self.handlers,
            "on_tool_error",
            "ignore_agent",
            error,
            run_id=self.run_id,
            parent_run_id=self.parent_run_id,
            tags=self.tags,
            **kwargs,
        )


class AsyncCallbackManagerForToolRun(AsyncParentRunManager, ToolManagerMixin):
    """工具运行的异步回调管理器。"""

    def get_sync(self) -> CallbackManagerForToolRun:
        """获取对应的同步 `RunManager`。

        Returns:
            对应的同步 `RunManager`。
        """
        return CallbackManagerForToolRun(
            run_id=self.run_id,
            handlers=self.handlers,
            inheritable_handlers=self.inheritable_handlers,
            parent_run_id=self.parent_run_id,
            tags=self.tags,
            inheritable_tags=self.inheritable_tags,
            metadata=self.metadata,
            inheritable_metadata=self.inheritable_metadata,
        )

    async def on_tool_end(self, output: Any, **kwargs: Any) -> None:
        """异步工具运行结束时运行。

        Args:
            output: 工具的输出。
            **kwargs: 额外的关键字参数。

        """
        if not self.handlers:
            return
        await ahandle_event(
            self.handlers,
            "on_tool_end",
            "ignore_agent",
            output,
            run_id=self.run_id,
            parent_run_id=self.parent_run_id,
            tags=self.tags,
            **kwargs,
        )

    async def on_tool_error(
        self,
        error: BaseException,
        **kwargs: Any,
    ) -> None:
        """工具出错时运行。

        Args:
            error: 错误。
            **kwargs: 额外的关键字参数。

        """
        if not self.handlers:
            return
        await ahandle_event(
            self.handlers,
            "on_tool_error",
            "ignore_agent",
            error,
            run_id=self.run_id,
            parent_run_id=self.parent_run_id,
            tags=self.tags,
            **kwargs,
        )


class CallbackManagerForRetrieverRun(ParentRunManager, RetrieverManagerMixin):
    """检索器运行的回调管理器。"""

    def on_retriever_end(
        self,
        documents: Sequence[Document],
        **kwargs: Any,
    ) -> None:
        """检索器运行结束时运行。

        Args:
            documents: 检索到的文档。
            **kwargs: 额外的关键字参数。

        """
        if not self.handlers:
            return
        handle_event(
            self.handlers,
            "on_retriever_end",
            "ignore_retriever",
            documents,
            run_id=self.run_id,
            parent_run_id=self.parent_run_id,
            tags=self.tags,
            **kwargs,
        )

    def on_retriever_error(
        self,
        error: BaseException,
        **kwargs: Any,
    ) -> None:
        """检索器出错时运行。

        Args:
            error: 错误。
            **kwargs: 额外的关键字参数。

        """
        if not self.handlers:
            return
        handle_event(
            self.handlers,
            "on_retriever_error",
            "ignore_retriever",
            error,
            run_id=self.run_id,
            parent_run_id=self.parent_run_id,
            tags=self.tags,
            **kwargs,
        )


class AsyncCallbackManagerForRetrieverRun(
    AsyncParentRunManager,
    RetrieverManagerMixin,
):
    """检索器运行的异步回调管理器。"""

    def get_sync(self) -> CallbackManagerForRetrieverRun:
        """获取对应的同步 `RunManager`。

        Returns:
            对应的同步 `RunManager`。

        """
        return CallbackManagerForRetrieverRun(
            run_id=self.run_id,
            handlers=self.handlers,
            inheritable_handlers=self.inheritable_handlers,
            parent_run_id=self.parent_run_id,
            tags=self.tags,
            inheritable_tags=self.inheritable_tags,
            metadata=self.metadata,
            inheritable_metadata=self.inheritable_metadata,
        )

    @shielded
    async def on_retriever_end(
        self, documents: Sequence[Document], **kwargs: Any
    ) -> None:
        """检索器运行结束时运行。

        Args:
            documents: 检索到的文档。
            **kwargs: 额外的关键字参数。

        """
        if not self.handlers:
            return
        await ahandle_event(
            self.handlers,
            "on_retriever_end",
            "ignore_retriever",
            documents,
            run_id=self.run_id,
            parent_run_id=self.parent_run_id,
            tags=self.tags,
            **kwargs,
        )

    @shielded
    async def on_retriever_error(
        self,
        error: BaseException,
        **kwargs: Any,
    ) -> None:
        """检索器出错时运行。

        Args:
            error: 错误。
            **kwargs: 额外的关键字参数。

        """
        if not self.handlers:
            return
        await ahandle_event(
            self.handlers,
            "on_retriever_error",
            "ignore_retriever",
            error,
            run_id=self.run_id,
            parent_run_id=self.parent_run_id,
            tags=self.tags,
            **kwargs,
        )


class CallbackManager(BaseCallbackManager):
    """LangChain 的回调管理器。"""

    def on_llm_start(
        self,
        serialized: dict[str, Any],
        prompts: list[str],
        run_id: UUID | None = None,
        **kwargs: Any,
    ) -> list[CallbackManagerForLLMRun]:
        """LLM 开始运行时运行。

        Args:
            serialized: 序列化的 LLM。
            prompts: 提示词列表。
            run_id: 运行 ID。
            **kwargs: 额外的关键字参数。

        Returns:
            每个提示词作为 LLM 运行的回调管理器。

        """
        managers = []
        for i, prompt in enumerate(prompts):
                        # 并发运行非内联任务
            run_id_ = run_id if i == 0 and run_id is not None else uuid7()
            handle_event(
                self.handlers,
                "on_llm_start",
                "ignore_llm",
                serialized,
                [prompt],
                run_id=run_id_,
                parent_run_id=self.parent_run_id,
                tags=self.tags,
                metadata=self.metadata,
                **kwargs,
            )

            managers.append(
                CallbackManagerForLLMRun(
                    run_id=run_id_,
                    handlers=self.handlers,
                    inheritable_handlers=self.inheritable_handlers,
                    parent_run_id=self.parent_run_id,
                    tags=self.tags,
                    inheritable_tags=self.inheritable_tags,
                    metadata=self.metadata,
                    inheritable_metadata=self.inheritable_metadata,
                )
            )

        return managers

    def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[BaseMessage]],
        run_id: UUID | None = None,
        **kwargs: Any,
    ) -> list[CallbackManagerForLLMRun]:
        """聊天模型开始运行时运行。

        Args:
            serialized: 序列化的 LLM。
            messages: 消息列表。
            run_id: 运行 ID。
            **kwargs: 额外的关键字参数。

        Returns:
            每个消息列表作为 LLM 运行的回调管理器。

        """
        managers = []
        for message_list in messages:
            if run_id is not None:
                run_id_ = run_id
                run_id = None
            else:
                run_id_ = uuid7()
            handle_event(
                self.handlers,
                "on_chat_model_start",
                "ignore_chat_model",
                serialized,
                [message_list],
                run_id=run_id_,
                parent_run_id=self.parent_run_id,
                tags=self.tags,
                metadata=self.metadata,
                **kwargs,
            )

            managers.append(
                CallbackManagerForLLMRun(
                    run_id=run_id_,
                    handlers=self.handlers,
                    inheritable_handlers=self.inheritable_handlers,
                    parent_run_id=self.parent_run_id,
                    tags=self.tags,
                    inheritable_tags=self.inheritable_tags,
                    metadata=self.metadata,
                    inheritable_metadata=self.inheritable_metadata,
                )
            )

        return managers

    def on_chain_start(
        self,
        serialized: dict[str, Any] | None,
        inputs: dict[str, Any] | Any,
        run_id: UUID | None = None,
        **kwargs: Any,
    ) -> CallbackManagerForChainRun:
        """链开始运行时运行。

        Args:
            serialized: 序列化的链。
            inputs: 链的输入。
            run_id: 运行 ID。
            **kwargs: 额外的关键字参数。

        Returns:
            链运行的回调管理器。

        """
        if run_id is None:
            run_id = uuid7()
        handle_event(
            self.handlers,
            "on_chain_start",
            "ignore_chain",
            serialized,
            inputs,
            run_id=run_id,
            parent_run_id=self.parent_run_id,
            tags=self.tags,
            metadata=self.metadata,
            **kwargs,
        )

        return CallbackManagerForChainRun(
            run_id=run_id,
            handlers=self.handlers,
            inheritable_handlers=self.inheritable_handlers,
            parent_run_id=self.parent_run_id,
            tags=self.tags,
            inheritable_tags=self.inheritable_tags,
            metadata=self.metadata,
            inheritable_metadata=self.inheritable_metadata,
        )

    @override
    def on_tool_start(
        self,
        serialized: dict[str, Any] | None,
        input_str: str,
        run_id: UUID | None = None,
        parent_run_id: UUID | None = None,
        inputs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> CallbackManagerForToolRun:
        """工具开始运行时运行。

        Args:
            serialized: 工具的序列化表示。
            input_str: 工具的输入字符串。

                非字符串输入会被转换为字符串。
            run_id: 运行的 ID。
            parent_run_id: 父运行的 ID。
            inputs: 如果提供，为工具的原始输入。

                Recommended for usage instead of input_str when the original input is
                needed.

                如果提供，输入应格式化为字典。键将对应工具中的命名参数。 **kwargs:
                传递给事件处理器的关键字参数

        Returns:
            工具运行的回调管理器。

        """
        if run_id is None:
            run_id = uuid7()

        handle_event(
            self.handlers,
            "on_tool_start",
            "ignore_agent",
            serialized,
            input_str,
            run_id=run_id,
            parent_run_id=self.parent_run_id,
            tags=self.tags,
            metadata=self.metadata,
            inputs=inputs,
            **kwargs,
        )

        return CallbackManagerForToolRun(
            run_id=run_id,
            handlers=self.handlers,
            inheritable_handlers=self.inheritable_handlers,
            parent_run_id=self.parent_run_id,
            tags=self.tags,
            inheritable_tags=self.inheritable_tags,
            metadata=self.metadata,
            inheritable_metadata=self.inheritable_metadata,
        )

    @override
    def on_retriever_start(
        self,
        serialized: dict[str, Any] | None,
        query: str,
        run_id: UUID | None = None,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> CallbackManagerForRetrieverRun:
        """Run when the retriever starts running.

        Args:
            serialized: 序列化的检索器。
            query: 查询。
            run_id: 运行的 ID。
            parent_run_id: 父运行的 ID。
            **kwargs: 其他关键字参数。

        Returns:
            检索器运行的回调管理器。
        """
        if run_id is None:
            run_id = uuid7()

        handle_event(
            self.handlers,
            "on_retriever_start",
            "ignore_retriever",
            serialized,
            query,
            run_id=run_id,
            parent_run_id=self.parent_run_id,
            tags=self.tags,
            metadata=self.metadata,
            **kwargs,
        )

        return CallbackManagerForRetrieverRun(
            run_id=run_id,
            handlers=self.handlers,
            inheritable_handlers=self.inheritable_handlers,
            parent_run_id=self.parent_run_id,
            tags=self.tags,
            inheritable_tags=self.inheritable_tags,
            metadata=self.metadata,
            inheritable_metadata=self.inheritable_metadata,
        )

    def on_custom_event(
        self,
        name: str,
        data: Any,
        run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        """分发临时事件到处理器（异步版本）。

        This event should NOT be used in any internal LangChain code. The event is meant
        specifically for users of the library to dispatch custom events that are
        tailored to their application.

        Args:
            name: 临时事件的名称。
            data: 临时事件的数据。
            run_id: 运行的 ID。

        Raises:
            ValueError: 如果传入了额外的关键字参数。
        """
        if not self.handlers:
            return
        if kwargs:
            msg = (
                "The dispatcher API does not accept additional keyword arguments."
                "Please do not pass any additional keyword arguments, instead "
                "include them in the data field."
            )
            raise ValueError(msg)
        if run_id is None:
            run_id = uuid7()

        handle_event(
            self.handlers,
            "on_custom_event",
            "ignore_custom_event",
            name,
            data,
            run_id=run_id,
            tags=self.tags,
            metadata=self.metadata,
        )

    @classmethod
    def configure(
        cls,
        inheritable_callbacks: Callbacks = None,
        local_callbacks: Callbacks = None,
        verbose: bool = False,  # noqa: FBT001,FBT002
        inheritable_tags: list[str] | None = None,
        local_tags: list[str] | None = None,
        inheritable_metadata: dict[str, Any] | None = None,
        local_metadata: dict[str, Any] | None = None,
        *,
        langsmith_inheritable_metadata: Mapping[str, Any] | None = None,
        langsmith_inheritable_tags: list[str] | None = None,
    ) -> CallbackManager:
        """配置回调管理器。

        Args:
            inheritable_callbacks: 可继承的回调。 local_callbacks: 本地回调。 verbose:
            是否启用详细模式。 inheritable_tags: 可继承的标签。 local_tags: 本地标签。
            inheritable_metadata: 可继承的元数据。 local_metadata: 本地元数据。
            langsmith_inheritable_metadata: 通过 `set_defaults` 应用于任何
            `LangChainTracer` 处理器的默认可继承元数据。 langsmith_inheritable_tags:
            通过 `set_defaults` 应用于任何 `LangChainTracer` 处理器的默认可继承标签。

        Returns:
            配置后的回调管理器。
        """
        return _configure(
            cls,
            inheritable_callbacks,
            local_callbacks,
            inheritable_tags,
            local_tags,
            inheritable_metadata,
            local_metadata,
            verbose=verbose,
            langsmith_inheritable_metadata=langsmith_inheritable_metadata,
            langsmith_inheritable_tags=langsmith_inheritable_tags,
        )


class CallbackManagerForChainGroup(CallbackManager):
    """链组的回调管理器。"""

    def __init__(
        self,
        handlers: list[BaseCallbackHandler],
        inheritable_handlers: list[BaseCallbackHandler] | None = None,
        parent_run_id: UUID | None = None,
        *,
        parent_run_manager: CallbackManagerForChainRun,
        **kwargs: Any,
    ) -> None:
        """初始化回调管理器。

        Args:
            handlers: 处理器列表。
            inheritable_handlers: 可继承的处理器列表。
            parent_run_id: 父运行的 ID。
            parent_run_manager: 父运行管理器。
            **kwargs: 其他关键字参数。

        """
        super().__init__(
            handlers,
            inheritable_handlers,
            parent_run_id,
            **kwargs,
        )
        self.parent_run_manager = parent_run_manager
        self.ended = False

    @override
    def copy(self) -> CallbackManagerForChainGroup:
        return self.__class__(
            handlers=self.handlers.copy(),
            inheritable_handlers=self.inheritable_handlers.copy(),
            parent_run_id=self.parent_run_id,
            tags=self.tags.copy(),
            inheritable_tags=self.inheritable_tags.copy(),
            metadata=self.metadata.copy(),
            inheritable_metadata=self.inheritable_metadata.copy(),
            parent_run_manager=self.parent_run_manager,
        )

    def merge(
        self: CallbackManagerForChainGroup, other: BaseCallbackManager
    ) -> CallbackManagerForChainGroup:
        """将组回调管理器与另一个回调管理器合并。

        Overwrites the merge method in the base class to ensure that the parent run
        manager is preserved. Keeps the `parent_run_manager` from the current object.

        Returns:
            A copy of the current object with the handlers, tags, and other attributes
            merged from the other object.

        Example:
            ```python
            # Merging two callback managers
            from langchain_core.callbacks.manager import (
                CallbackManager,
                trace_as_chain_group,
            )
            from langchain_core.callbacks.stdout import StdOutCallbackHandler

            manager = CallbackManager(handlers=[StdOutCallbackHandler()], tags=["tag2"])
            with trace_as_chain_group("My Group Name", tags=["tag1"]) as group_manager:
                merged_manager = group_manager.merge(manager)
                print(type(merged_manager))
                # <class 'langchain_core.callbacks.manager.CallbackManagerForChainGroup'>

                print(merged_manager.handlers)
                # [
                #    <langchain_core.callbacks.stdout.LangChainTracer object at ...>,
                #    <langchain_core.callbacks.streaming_stdout.StdOutCallbackHandler object at ...>,
                # ]

                print(merged_manager.tags)
                #    ['tag2', 'tag1']
            ```
        """  # noqa: E501
        manager = self.__class__(
            parent_run_id=self.parent_run_id or other.parent_run_id,
            handlers=[],
            inheritable_handlers=[],
            tags=list(set(self.tags + other.tags)),
            inheritable_tags=list(set(self.inheritable_tags + other.inheritable_tags)),
            metadata={
                **self.metadata,
                **other.metadata,
            },
            parent_run_manager=self.parent_run_manager,
        )

        handlers = self.handlers + other.handlers
        inheritable_handlers = self.inheritable_handlers + other.inheritable_handlers

        for handler in handlers:
            manager.add_handler(handler)

        for handler in inheritable_handlers:
            manager.add_handler(handler, inherit=True)
        return manager

    def on_chain_end(self, outputs: dict[str, Any] | Any, **kwargs: Any) -> None:
        """当追踪的链组结束时运行。

        Args:
            outputs: 链的输出。
            **kwargs: 其他关键字参数。

        """
        self.ended = True
        return self.parent_run_manager.on_chain_end(outputs, **kwargs)

    def on_chain_error(
        self,
        error: BaseException,
        **kwargs: Any,
    ) -> None:
        """当链发生错误时运行。

        Args:
            error: 错误。
            **kwargs: 额外的关键字参数。

        """
        self.ended = True
        return self.parent_run_manager.on_chain_error(error, **kwargs)


class AsyncCallbackManager(BaseCallbackManager):
    """处理 LangChain 回调的异步回调管理器。"""

    @property
    def is_async(self) -> bool:
        """返回该处理器是否为异步。"""
        return True

    async def on_llm_start(
        self,
        serialized: dict[str, Any],
        prompts: list[str],
        run_id: UUID | None = None,
        **kwargs: Any,
    ) -> list[AsyncCallbackManagerForLLMRun]:
        """当 LLM 开始运行时调用。

        Args:
            serialized: 序列化的 LLM。
            prompts: 提示词列表。
            run_id: 运行的 ID。
            **kwargs: 额外的关键字参数。

        Returns:
            The list of async callback managers, one for each LLM run corresponding to
            each prompt.
        """
        inline_tasks = []
        non_inline_tasks = []
        inline_handlers = [handler for handler in self.handlers if handler.run_inline]
        non_inline_handlers = [
            handler for handler in self.handlers if not handler.run_inline
        ]
        managers = []

        for prompt in prompts:
            if run_id is not None:
                run_id_ = run_id
                run_id = None
            else:
                run_id_ = uuid7()

            if inline_handlers:
                inline_tasks.append(
                    ahandle_event(
                        inline_handlers,
                        "on_llm_start",
                        "ignore_llm",
                        serialized,
                        [prompt],
                        run_id=run_id_,
                        parent_run_id=self.parent_run_id,
                        tags=self.tags,
                        metadata=self.metadata,
                        **kwargs,
                    )
                )
            else:
                non_inline_tasks.append(
                    ahandle_event(
                        non_inline_handlers,
                        "on_llm_start",
                        "ignore_llm",
                        serialized,
                        [prompt],
                        run_id=run_id_,
                        parent_run_id=self.parent_run_id,
                        tags=self.tags,
                        metadata=self.metadata,
                        **kwargs,
                    )
                )

            managers.append(
                AsyncCallbackManagerForLLMRun(
                    run_id=run_id_,
                    handlers=self.handlers,
                    inheritable_handlers=self.inheritable_handlers,
                    parent_run_id=self.parent_run_id,
                    tags=self.tags,
                    inheritable_tags=self.inheritable_tags,
                    metadata=self.metadata,
                    inheritable_metadata=self.inheritable_metadata,
                )
            )

                # 延迟导入以避免在模块级别导入 langsmith（约 132ms）。
        for inline_task in inline_tasks:
            await inline_task

                # 解决外部追踪上下文与继承上下文之间的优先级冲突
        if non_inline_tasks:
            await asyncio.gather(*non_inline_tasks)

        return managers

    async def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[BaseMessage]],
        run_id: UUID | None = None,
        **kwargs: Any,
    ) -> list[AsyncCallbackManagerForLLMRun]:
        """异步运行时 LLM 开始运行。

        Args:
            serialized: 序列化的 LLM。
            messages: 消息列表。
            run_id: 运行的 ID。
            **kwargs: 额外的关键字参数。

        Returns:
            The list of async callback managers, one for each LLM run corresponding to
            each inner message list.
        """
        inline_tasks = []
        non_inline_tasks = []
        managers = []

        for message_list in messages:
            if run_id is not None:
                run_id_ = run_id
                run_id = None
            else:
                run_id_ = uuid7()

            for handler in self.handlers:
                task = ahandle_event(
                    [handler],
                    "on_chat_model_start",
                    "ignore_chat_model",
                    serialized,
                    [message_list],
                    run_id=run_id_,
                    parent_run_id=self.parent_run_id,
                    tags=self.tags,
                    metadata=self.metadata,
                    **kwargs,
                )
                if handler.run_inline:
                    inline_tasks.append(task)
                else:
                    non_inline_tasks.append(task)

            managers.append(
                AsyncCallbackManagerForLLMRun(
                    run_id=run_id_,
                    handlers=self.handlers,
                    inheritable_handlers=self.inheritable_handlers,
                    parent_run_id=self.parent_run_id,
                    tags=self.tags,
                    inheritable_tags=self.inheritable_tags,
                    metadata=self.metadata,
                    inheritable_metadata=self.inheritable_metadata,
                )
            )

                # 如果 LC 父节点已经反映在运行树中
        for task in inline_tasks:
            await task

                # 我们知道运行树要么是相同的父节点，要么是父节点的子节点。
        if non_inline_tasks:
            await asyncio.gather(*non_inline_tasks)

        return managers

    async def on_chain_start(
        self,
        serialized: dict[str, Any] | None,
        inputs: dict[str, Any] | Any,
        run_id: UUID | None = None,
        **kwargs: Any,
    ) -> AsyncCallbackManagerForChainRun:
        """异步运行时链开始运行。

        Args:
            serialized: 序列化的链。
            inputs: 链的输入。
            run_id: 运行的 ID。
            **kwargs: 额外的关键字参数。

        Returns:
            链运行的异步回调管理器。
        """
        if run_id is None:
            run_id = uuid7()

        await ahandle_event(
            self.handlers,
            "on_chain_start",
            "ignore_chain",
            serialized,
            inputs,
            run_id=run_id,
            parent_run_id=self.parent_run_id,
            tags=self.tags,
            metadata=self.metadata,
            **kwargs,
        )

        return AsyncCallbackManagerForChainRun(
            run_id=run_id,
            handlers=self.handlers,
            inheritable_handlers=self.inheritable_handlers,
            parent_run_id=self.parent_run_id,
            tags=self.tags,
            inheritable_tags=self.inheritable_tags,
            metadata=self.metadata,
            inheritable_metadata=self.inheritable_metadata,
        )

    @override
    async def on_tool_start(
        self,
        serialized: dict[str, Any] | None,
        input_str: str,
        run_id: UUID | None = None,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> AsyncCallbackManagerForToolRun:
        """当工具开始运行时调用。

        Args:
            serialized: 序列化的工具。
            input_str: 工具的输入。
            run_id: 运行的 ID。
            parent_run_id: 父运行的 ID。
            **kwargs: 额外的关键字参数。

        Returns:
            工具运行的异步回调管理器。
        """
        if run_id is None:
            run_id = uuid7()

        await ahandle_event(
            self.handlers,
            "on_tool_start",
            "ignore_agent",
            serialized,
            input_str,
            run_id=run_id,
            parent_run_id=self.parent_run_id,
            tags=self.tags,
            metadata=self.metadata,
            **kwargs,
        )

        return AsyncCallbackManagerForToolRun(
            run_id=run_id,
            handlers=self.handlers,
            inheritable_handlers=self.inheritable_handlers,
            parent_run_id=self.parent_run_id,
            tags=self.tags,
            inheritable_tags=self.inheritable_tags,
            metadata=self.metadata,
            inheritable_metadata=self.inheritable_metadata,
        )

    async def on_custom_event(
        self,
        name: str,
        data: Any,
        run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        """向处理器分发临时事件（异步版本）。

        This event should NOT be used in any internal LangChain code. The event is meant
        specifically for users of the library to dispatch custom events that are
        tailored to their application.

        Args:
            name: 临时事件的名称。
            data: 临时事件的数据。
            run_id: 运行的 ID。

        Raises:
            ValueError: 如果传入了额外的关键字参数。
        """
        if not self.handlers:
            return
        if run_id is None:
            run_id = uuid7()

        if kwargs:
            msg = (
                "The dispatcher API does not accept additional keyword arguments."
                "Please do not pass any additional keyword arguments, instead "
                "include them in the data field."
            )
            raise ValueError(msg)
        await ahandle_event(
            self.handlers,
            "on_custom_event",
            "ignore_custom_event",
            name,
            data,
            run_id=run_id,
            tags=self.tags,
            metadata=self.metadata,
        )

    @override
    async def on_retriever_start(
        self,
        serialized: dict[str, Any] | None,
        query: str,
        run_id: UUID | None = None,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> AsyncCallbackManagerForRetrieverRun:
        """当检索器开始运行时调用。

        Args:
            serialized: 序列化的检索器。
            query: 查询。
            run_id: 运行的 ID。
            parent_run_id: 父运行的 ID。
            **kwargs: 额外的关键字参数。

        Returns:
            检索器运行的异步回调管理器。
        """
        if run_id is None:
            run_id = uuid7()

        await ahandle_event(
            self.handlers,
            "on_retriever_start",
            "ignore_retriever",
            serialized,
            query,
            run_id=run_id,
            parent_run_id=self.parent_run_id,
            tags=self.tags,
            metadata=self.metadata,
            **kwargs,
        )

        return AsyncCallbackManagerForRetrieverRun(
            run_id=run_id,
            handlers=self.handlers,
            inheritable_handlers=self.inheritable_handlers,
            parent_run_id=self.parent_run_id,
            tags=self.tags,
            inheritable_tags=self.inheritable_tags,
            metadata=self.metadata,
            inheritable_metadata=self.inheritable_metadata,
        )

    @classmethod
    def configure(
        cls,
        inheritable_callbacks: Callbacks = None,
        local_callbacks: Callbacks = None,
        verbose: bool = False,  # noqa: FBT001,FBT002
        inheritable_tags: list[str] | None = None,
        local_tags: list[str] | None = None,
        inheritable_metadata: dict[str, Any] | None = None,
        local_metadata: dict[str, Any] | None = None,
        *,
        langsmith_inheritable_metadata: Mapping[str, Any] | None = None,
        langsmith_inheritable_tags: list[str] | None = None,
    ) -> AsyncCallbackManager:
        """配置异步回调管理器。

        Args:
            inheritable_callbacks: 可继承的回调。 local_callbacks: 本地回调。 verbose:
            是否启用详细模式。 inheritable_tags: 可继承的标签。 local_tags: 本地标签。
            inheritable_metadata: 可继承的元数据。 local_metadata: 本地元数据。
            langsmith_inheritable_metadata: 通过 `set_defaults` 应用于任何
            `LangChainTracer` 处理器的默认可继承元数据。 langsmith_inheritable_tags:
            通过 `set_defaults` 应用于任何 `LangChainTracer` 处理器的默认可继承标签。

        Returns:
            配置好的异步回调管理器。
        """
        return _configure(
            cls,
            inheritable_callbacks,
            local_callbacks,
            inheritable_tags,
            local_tags,
            inheritable_metadata,
            local_metadata,
            verbose=verbose,
            langsmith_inheritable_metadata=langsmith_inheritable_metadata,
            langsmith_inheritable_tags=langsmith_inheritable_tags,
        )


class AsyncCallbackManagerForChainGroup(AsyncCallbackManager):
    """链组的异步回调管理器。"""

    def __init__(
        self,
        handlers: list[BaseCallbackHandler],
        inheritable_handlers: list[BaseCallbackHandler] | None = None,
        parent_run_id: UUID | None = None,
        *,
        parent_run_manager: AsyncCallbackManagerForChainRun,
        **kwargs: Any,
    ) -> None:
        """初始化异步回调管理器。

        Args:
            handlers: 处理程序列表。
            inheritable_handlers: 可继承处理程序列表。
            parent_run_id: 父运行的 ID。
            parent_run_manager: 父运行管理器。
            **kwargs: 额外的关键字参数。
        """
        super().__init__(
            handlers,
            inheritable_handlers,
            parent_run_id,
            **kwargs,
        )
        self.parent_run_manager = parent_run_manager
        self.ended = False

    def copy(self) -> AsyncCallbackManagerForChainGroup:
        """返回异步回调管理器的副本。"""
        return self.__class__(
            handlers=self.handlers.copy(),
            inheritable_handlers=self.inheritable_handlers.copy(),
            parent_run_id=self.parent_run_id,
            tags=self.tags.copy(),
            inheritable_tags=self.inheritable_tags.copy(),
            metadata=self.metadata.copy(),
            inheritable_metadata=self.inheritable_metadata.copy(),
            parent_run_manager=self.parent_run_manager,
        )

    def merge(
        self: AsyncCallbackManagerForChainGroup, other: BaseCallbackManager
    ) -> AsyncCallbackManagerForChainGroup:
        """将组回调管理器与另一个回调管理器合并。

        Overwrites the merge method in the base class to ensure that the parent run
        manager is preserved. Keeps the `parent_run_manager` from the current object.

        Returns:
            A copy of the current `AsyncCallbackManagerForChainGroup` with the handlers,
                tags, etc. of the other callback manager merged in.

        Example:
            ```python
            # Merging two callback managers
            from langchain_core.callbacks.manager import (
                CallbackManager,
                atrace_as_chain_group,
            )
            from langchain_core.callbacks.stdout import StdOutCallbackHandler

            manager = CallbackManager(handlers=[StdOutCallbackHandler()], tags=["tag2"])
            async with atrace_as_chain_group(
                "My Group Name", tags=["tag1"]
            ) as group_manager:
                merged_manager = group_manager.merge(manager)
                print(type(merged_manager))
                # <class 'langchain_core.callbacks.manager.AsyncCallbackManagerForChainGroup'>

                print(merged_manager.handlers)
                # [
                #    <langchain_core.callbacks.stdout.LangChainTracer object at ...>,
                #    <langchain_core.callbacks.streaming_stdout.StdOutCallbackHandler object at ...>,
                # ]

                print(merged_manager.tags)
                #    ['tag2', 'tag1']
            ```
        """  # noqa: E501
        manager = self.__class__(
            parent_run_id=self.parent_run_id or other.parent_run_id,
            handlers=[],
            inheritable_handlers=[],
            tags=list(set(self.tags + other.tags)),
            inheritable_tags=list(set(self.inheritable_tags + other.inheritable_tags)),
            metadata={
                **self.metadata,
                **other.metadata,
            },
            parent_run_manager=self.parent_run_manager,
        )

        handlers = self.handlers + other.handlers
        inheritable_handlers = self.inheritable_handlers + other.inheritable_handlers

        for handler in handlers:
            manager.add_handler(handler)

        for handler in inheritable_handlers:
            manager.add_handler(handler, inherit=True)
        return manager

    async def on_chain_end(self, outputs: dict[str, Any] | Any, **kwargs: Any) -> None:
        """当被追踪的链组结束时运行。

        Args:
            outputs: 链的输出。
            **kwargs: 额外的关键字参数。
        """
        self.ended = True
        await self.parent_run_manager.on_chain_end(outputs, **kwargs)

    async def on_chain_error(
        self,
        error: BaseException,
        **kwargs: Any,
    ) -> None:
        """当链出错时运行。

        Args:
            error: 错误信息。
            **kwargs: 额外的关键字参数。
        """
        self.ended = True
        await self.parent_run_manager.on_chain_error(error, **kwargs)


T = TypeVar("T", CallbackManager, AsyncCallbackManager)


def _configure(
    callback_manager_cls: type[T],
    inheritable_callbacks: Callbacks = None,
    local_callbacks: Callbacks = None,
    inheritable_tags: list[str] | None = None,
    local_tags: list[str] | None = None,
    inheritable_metadata: dict[str, Any] | None = None,
    local_metadata: dict[str, Any] | None = None,
    *,
    verbose: bool = False,
    langsmith_inheritable_metadata: Mapping[str, Any] | None = None,
    langsmith_inheritable_tags: list[str] | None = None,
) -> T:
    """配置回调管理器。

    Args:
        callback_manager_cls: 回调管理器类。 inheritable_callbacks: 可继承的回调。
        local_callbacks: 本地回调。 inheritable_tags: 可继承的标签。 local_tags:
        本地标签。 inheritable_metadata: 可继承的元数据。 local_metadata: 本地元数据。
        verbose: 是否启用详细模式。 langsmith_inheritable_metadata:
        默认可继承元数据，通过 `set_defaults` 应用于任何 `LangChainTracer` 处理器。
        langsmith_inheritable_tags: 默认可继承标签，通过 `set_defaults` 应用于任何
        `LangChainTracer` 处理器。

    Raises:
        RuntimeError: 如果设置了 `LANGCHAIN_TRACING` 但未设置 `LANGCHAIN_TRACING_V2`。

    Returns:
        配置好的回调管理器。
    """
        # 否则，我们假设 LC 上下文已推进
    from langsmith.run_helpers import get_tracing_context  # noqa: PLC0415

    from langchain_core.tracers.context import (  # noqa: PLC0415
        _configure_hooks,
        _get_tracer_kwargs,
        _tracing_v2_is_enabled,
        tracing_v2_callback_var,
    )
    from langchain_core.tracers.langchain import LangChainTracer  # noqa: PLC0415
    from langchain_core.tracers.stdout import ConsoleCallbackHandler  # noqa: PLC0415

    tracing_context = get_tracing_context()
    tracing_metadata = tracing_context["metadata"]
    tracing_tags = tracing_context["tags"]
    run_tree: Run | None = tracing_context["parent"]
    parent_run_id = None if run_tree is None else run_tree.id
    callback_manager = callback_manager_cls(
        handlers=[],
        parent_run_id=parent_run_id,
    )
    if inheritable_callbacks or local_callbacks:
        if isinstance(inheritable_callbacks, list) or inheritable_callbacks is None:
            inheritable_callbacks_ = inheritable_callbacks or []
            callback_manager = callback_manager_cls(
                handlers=inheritable_callbacks_.copy(),
                inheritable_handlers=inheritable_callbacks_.copy(),
                parent_run_id=parent_run_id,
            )
        else:
            parent_run_id_ = inheritable_callbacks.parent_run_id
                        # 在外部追踪上下文和继承上下文之间打破歧义
            if parent_run_id is not None and (
                parent_run_id_ is None
                                # 如果 LC 父级已反射
                                # 在运行树中，我们知道运行树要么是
                                # 同级或父级的子级。
                or (run_tree and str(parent_run_id_) in run_tree.dotted_order)
            ):
                parent_run_id_ = parent_run_id
                                # 否则，我们假设 LC 上下文已推进
                                # 超出运行树范围，我们不应继承父级。
            callback_manager = callback_manager_cls(
                handlers=inheritable_callbacks.handlers.copy(),
                inheritable_handlers=inheritable_callbacks.inheritable_handlers.copy(),
                parent_run_id=parent_run_id_,
                tags=inheritable_callbacks.tags.copy(),
                inheritable_tags=inheritable_callbacks.inheritable_tags.copy(),
                metadata=inheritable_callbacks.metadata.copy(),
                inheritable_metadata=inheritable_callbacks.inheritable_metadata.copy(),
            )
        local_handlers_ = (
            local_callbacks
            if isinstance(local_callbacks, list)
            else (local_callbacks.handlers if local_callbacks else [])
        )
        for handler in local_handlers_:
            callback_manager.add_handler(handler, inherit=False)
    if inheritable_tags or local_tags:
        callback_manager.add_tags(inheritable_tags or [])
        callback_manager.add_tags(local_tags or [], inherit=False)
    if inheritable_metadata or local_metadata:
        callback_manager.add_metadata(inheritable_metadata or {})
        callback_manager.add_metadata(local_metadata or {}, inherit=False)
    if tracing_tags:
        callback_manager.add_tags(tracing_tags.copy())

    v1_tracing_enabled_ = env_var_is_set("LANGCHAIN_TRACING") or env_var_is_set(
        "LANGCHAIN_HANDLER"
    )

    tracer_v2 = tracing_v2_callback_var.get()
    tracing_v2_enabled_ = _tracing_v2_is_enabled()

    if v1_tracing_enabled_ and not tracing_v2_enabled_:
                # 如果两者均启用，可静默忽略 v1 追踪器
        msg = (
            "Tracing using LangChainTracerV1 is no longer supported. "
            "Please set the LANGCHAIN_TRACING_V2 environment variable to enable "
            "tracing instead."
        )
        raise RuntimeError(msg)

    debug = _get_debug()
    if verbose or debug or tracing_v2_enabled_:
        if verbose and not any(
            isinstance(handler, StdOutCallbackHandler)
            for handler in callback_manager.handlers
        ):
            if debug:
                pass
            else:
                callback_manager.add_handler(StdOutCallbackHandler(), inherit=False)
        if debug and not any(
            isinstance(handler, ConsoleCallbackHandler)
            for handler in callback_manager.handlers
        ):
            callback_manager.add_handler(ConsoleCallbackHandler())
        if tracing_v2_enabled_ and not any(
            isinstance(handler, LangChainTracer)
            for handler in callback_manager.handlers
        ):
            if tracer_v2:
                callback_manager.add_handler(tracer_v2)
            else:
                try:
                    handler = LangChainTracer(**_get_tracer_kwargs())
                    callback_manager.add_handler(handler)
                except Exception as e:
                    logger.warning(
                        "Unable to load requested LangChainTracer."
                        " To disable this warning,"
                        " unset the LANGCHAIN_TRACING_V2 environment variables.\n"
                        "%s",
                        repr(e),
                    )
        if run_tree is not None:
            for handler in callback_manager.handlers:
                if isinstance(handler, LangChainTracer):
                    handler.order_map[run_tree.id] = (
                        run_tree.trace_id,
                        run_tree.dotted_order,
                    )
                    run_id_str = str(run_tree.id)
                    if run_id_str not in handler.run_map:
                        handler.run_map[run_id_str] = run_tree
                        handler._external_run_ids.setdefault(  # noqa: SLF001
                            run_id_str, 0
                        )
    for var, inheritable, handler_class, env_var in _configure_hooks:
        create_one = (
            env_var is not None
            and env_var_is_set(env_var)
            and handler_class is not None
        )
        if var.get() is not None or create_one:
            var_handler = (
                var.get() or cast("type[BaseCallbackHandler]", handler_class)()
            )
            if handler_class is None:
                if not any(
                    handler is var_handler  # 直接指针比较
                    for handler in callback_manager.handlers
                ):
                    callback_manager.add_handler(var_handler, inheritable)
            elif not any(
                isinstance(handler, handler_class)
                for handler in callback_manager.handlers
            ):
                callback_manager.add_handler(var_handler, inheritable)

    if tracing_metadata:
        langsmith_inheritable_metadata = {
            **tracing_metadata,
            **(langsmith_inheritable_metadata or {}),
        }

    if langsmith_inheritable_metadata or langsmith_inheritable_tags:
        callback_manager.handlers = [
            handler.copy_with_metadata_defaults(
                metadata=langsmith_inheritable_metadata,
                tags=langsmith_inheritable_tags,
            )
            if isinstance(handler, LangChainTracer)
            else handler
            for handler in callback_manager.handlers
        ]
        callback_manager.inheritable_handlers = [
            handler.copy_with_metadata_defaults(
                metadata=langsmith_inheritable_metadata,
                tags=langsmith_inheritable_tags,
            )
            if isinstance(handler, LangChainTracer)
            else handler
            for handler in callback_manager.inheritable_handlers
        ]
    return callback_manager


async def adispatch_custom_event(
    name: str, data: Any, *, config: RunnableConfig | None = None
) -> None:
    """向处理器分发一个临时事件。

    Args:
        name: 临时事件的名称。
        data: 临时事件的数据。

            自由格式数据。理想情况下应为 JSON
            可序列化以避免下游序列化问题，但不强制要求。 config: 可选的配置对象。

            与异步 API 保持一致，但并非严格必需。

    Raises:
        RuntimeError: 如果没有可用的父运行 ID 以关联该事件。

    Example:
        ```python
        from langchain_core.callbacks import (
            AsyncCallbackHandler,
            adispatch_custom_event
        )
        from langchain_core.runnable import RunnableLambda

        class CustomCallbackManager(AsyncCallbackHandler):
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
                print(f"Received custom event: {name} with data: {data}")

        callback = CustomCallbackManager()

        async def foo(inputs):
            await adispatch_custom_event("my_event", {"bar": "buzz})
            return inputs

        foo_ = RunnableLambda(foo)
        await foo_.ainvoke({"a": "1"}, {"callbacks": [CustomCallbackManager()]})
        ```

    Example: Use with astream events

        ```python
        from langchain_core.callbacks import (
            AsyncCallbackHandler,
            adispatch_custom_event
        )
        from langchain_core.runnable import RunnableLambda

        class CustomCallbackManager(AsyncCallbackHandler):
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
                print(f"Received custom event: {name} with data: {data}")

        callback = CustomCallbackManager()

        async def foo(inputs):
            await adispatch_custom_event("event_type_1", {"bar": "buzz})
            await adispatch_custom_event("event_type_2", 5)
            return inputs

        foo_ = RunnableLambda(foo)

        async for event in foo_.ainvoke_stream(
            {"a": "1"},
            version="v2",
            config={"callbacks": [CustomCallbackManager()]}
        ):
            print(event)
        ```

    !!! warning

        If using python 3.10 and async, you MUST specify the `config` parameter or the
        function will raise an error. This is due to a limitation in asyncio for python
        3.10 that prevents LangChain from automatically propagating the config object on
        the user's behalf.
    """
        # 本地导入以防止循环导入。
    from langchain_core.runnables.config import (  # noqa: PLC0415
        ensure_config,
        get_async_callback_manager_for_config,
    )

    config = ensure_config(config)
    callback_manager = get_async_callback_manager_for_config(config)
        # 我们需要获取父运行的回调管理器。
        # 目前这是一个权宜之计，以便能够从工具或 lambda
    # 内部分发临时事件，并将元数据事件关联到父运行，而不是为每次事件生成新的运行 ID。
        # 从工具或 lambda
    # 内部分发临时事件，并将元数据事件关联到父运行，而不是为每次事件生成新的运行 ID。
        # 目前这是一个权宜之计，以便能够从工具或 lambda
    # 内部分发临时事件，并将元数据事件关联到父运行，而不是为每次事件生成新的运行 ID。
    if callback_manager.parent_run_id is None:
        msg = (
            "Unable to dispatch an adhoc event without a parent run id."
            "This function can only be called from within an existing run (e.g.,"
            "inside a tool or a RunnableLambda or a RunnableGenerator.)"
            "If you are doing that and still seeing this error, try explicitly"
            "passing the config parameter to this function."
        )
        raise RuntimeError(msg)

    await callback_manager.on_custom_event(
        name,
        data,
        run_id=callback_manager.parent_run_id,
    )


def dispatch_custom_event(
    name: str, data: Any, *, config: RunnableConfig | None = None
) -> None:
    """分发一个临时事件。

    Args:
        name: 临时事件的名称。
        data: 临时事件的数据。

            自由格式数据。理想情况下应为 JSON
            可序列化以避免下游序列化问题，但不强制要求。 config: 可选的配置对象。

            与异步 API 保持一致，但并非严格必需。

    Raises:
        RuntimeError：当没有可用的父运行 ID 用于关联事件时抛出。

    Example:
        ```python
        from langchain_core.callbacks import BaseCallbackHandler
        from langchain_core.callbacks import dispatch_custom_event
        from langchain_core.runnable import RunnableLambda

        class CustomCallbackManager(BaseCallbackHandler):
            def on_custom_event(
                self,
                name: str,
                data: Any,
                *,
                run_id: UUID,
                tags: list[str] | None = None,
                metadata: dict[str, Any] | None = None,
                **kwargs: Any,
            ) -> None:
                print(f"Received custom event: {name} with data: {data}")

        def foo(inputs):
            dispatch_custom_event("my_event", {"bar": "buzz})
            return inputs

        foo_ = RunnableLambda(foo)
        foo_.invoke({"a": "1"}, {"callbacks": [CustomCallbackManager()]})
        ```
    """
        # 本地导入以防止循环导入。
    from langchain_core.runnables.config import (  # noqa: PLC0415
        ensure_config,
        get_callback_manager_for_config,
    )

    config = ensure_config(config)
    callback_manager = get_callback_manager_for_config(config)
        # 我们需要获取父运行的回调管理器。
        # 目前这是一个权宜之计，以便能够从工具或 lambda
    # 内部分发临时事件，并将元数据事件关联到父运行，而不是为每次事件生成新的运行 ID。
        # 从工具或 lambda
    # 内部分发临时事件，并将元数据事件关联到父运行，而不是为每次事件生成新的运行 ID。
        # 目前这是一个权宜之计，以便能够从工具或 lambda
    # 内部分发临时事件，并将元数据事件关联到父运行，而不是为每次事件生成新的运行 ID。
    if callback_manager.parent_run_id is None:
        msg = (
            "Unable to dispatch an adhoc event without a parent run id."
            "This function can only be called from within an existing run (e.g.,"
            "inside a tool or a RunnableLambda or a RunnableGenerator.)"
            "If you are doing that and still seeing this error, try explicitly"
            "passing the config parameter to this function."
        )
        raise RuntimeError(msg)
    callback_manager.on_custom_event(
        name,
        data,
        run_id=callback_manager.parent_run_id,
    )


@functools.lru_cache(maxsize=1)
def _executor() -> ThreadPoolExecutor:
        # 如果用户在同步上下文中指定了异步回调处理器，且事件循环已在运行中，
        # 我们无法将协程提交给正在运行的循环，因为这会导致死锁。相反，我们必须将其调度到后台线程上。为避免每次创建和关闭新的执行器，我们使用一个延迟创建、共享的执行器。
        # 若使用常规的 langchain 并行处理（如批处理等），通常只需要 1
    # 个工作线程，但目前允许更多，以减少与自定义执行器混合时的性能下降风险。
        # 会导致死锁。因此，我们必须对它们进行调度
        # 在后台线程上。为避免频繁创建和关闭
        # 每次创建新的执行器，我们使用延迟创建的、共享的
        # executor。如果你使用的是 LangChain 的常规并行机制（批处理等）
        # 你只需要 1 个工作进程，但目前我们允许更多，以降低出错概率
        # 如果与自定义执行器混合使用，可能会导致性能下降。
    cutie = ThreadPoolExecutor(max_workers=10)
    atexit.register(cutie.shutdown, wait=True)
    return cutie

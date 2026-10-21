"""调用监听器的追踪器。"""

from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING

from langchain_core.runnables.config import (
    RunnableConfig,
    acall_func_with_variable_args,
    call_func_with_variable_args,
)
from langchain_core.tracers.base import AsyncBaseTracer, BaseTracer
from langchain_core.tracers.schemas import Run

if TYPE_CHECKING:
    from uuid import UUID

Listener = Callable[[Run], None] | Callable[[Run, RunnableConfig], None]
AsyncListener = (
    Callable[[Run], Awaitable[None]] | Callable[[Run, RunnableConfig], Awaitable[None]]
)


class RootListenersTracer(BaseTracer):
    """在运行开始、结束和出错时调用监听器的追踪器。"""

    log_missing_parent = False
    """如果父级缺失，是否记录警告。"""

    def __init__(
        self,
        *,
        config: RunnableConfig,
        on_start: Listener | None,
        on_end: Listener | None,
        on_error: Listener | None,
    ) -> None:
        """初始化追踪器。

        Args:
            config: Runnable 配置。
            on_start: 在运行开始时调用的监听器。
            on_end: 在运行结束时调用的监听器。
            on_error: 在运行出错时调用的监听器
        """
        super().__init__(_schema_format="original+chat")

        self.config = config
        self._arg_on_start = on_start
        self._arg_on_end = on_end
        self._arg_on_error = on_error
        self.root_id: UUID | None = None

    def _persist_run(self, run: Run) -> None:
                # 这是仅在整个运行树中调用一次的旧方法
                # 因此在此处无用
        pass

    def _on_run_create(self, run: Run) -> None:
        if self.root_id is not None:
            return

        self.root_id = run.id

        if self._arg_on_start is not None:
            call_func_with_variable_args(self._arg_on_start, run, self.config)

    def _on_run_update(self, run: Run) -> None:
        if run.id != self.root_id:
            return

        if run.error is None:
            if self._arg_on_end is not None:
                call_func_with_variable_args(self._arg_on_end, run, self.config)
        elif self._arg_on_error is not None:
            call_func_with_variable_args(self._arg_on_error, run, self.config)


class AsyncRootListenersTracer(AsyncBaseTracer):
    """在运行开始、结束和出错时调用监听器的异步追踪器。"""

    log_missing_parent = False
    """如果父级缺失，是否记录警告。"""

    def __init__(
        self,
        *,
        config: RunnableConfig,
        on_start: AsyncListener | None,
        on_end: AsyncListener | None,
        on_error: AsyncListener | None,
    ) -> None:
        """初始化追踪器。

        Args:
            config: Runnable 配置。
            on_start: 在运行开始时调用的监听器。
            on_end: 在运行结束时调用的监听器。
            on_error: 在运行出错时调用的监听器
        """
        super().__init__(_schema_format="original+chat")

        self.config = config
        self._arg_on_start = on_start
        self._arg_on_end = on_end
        self._arg_on_error = on_error
        self.root_id: UUID | None = None

    async def _persist_run(self, run: Run) -> None:
                # 这是仅在整个运行树中调用一次的旧方法
                # 因此在此处无用
        pass

    async def _on_run_create(self, run: Run) -> None:
        if self.root_id is not None:
            return

        self.root_id = run.id

        if self._arg_on_start is not None:
            await acall_func_with_variable_args(self._arg_on_start, run, self.config)

    async def _on_run_update(self, run: Run) -> None:
        if run.id != self.root_id:
            return

        if run.error is None:
            if self._arg_on_end is not None:
                await acall_func_with_variable_args(self._arg_on_end, run, self.config)
        elif self._arg_on_error is not None:
            await acall_func_with_variable_args(self._arg_on_error, run, self.config)

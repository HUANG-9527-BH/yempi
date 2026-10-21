"""收集所有嵌套运行的追踪器，存储在列表中。"""

from typing import Any
from uuid import UUID

from langchain_core.tracers._compat import run_copy
from langchain_core.tracers.base import BaseTracer
from langchain_core.tracers.schemas import Run


class RunCollectorCallbackHandler(BaseTracer):
    """收集所有嵌套运行的追踪器，存储在列表中。

    此追踪器适用于检查和评估目的。
    """

    name: str = "run-collector_callback_handler"

    def __init__(self, example_id: UUID | str | None = None, **kwargs: Any) -> None:
        """初始化 `RunCollectorCallbackHandler`。

        Args:
            example_id: 被追踪示例的 ID。
            **kwargs: 额外的关键字参数。
        """
        super().__init__(**kwargs)
        self.example_id = (
            UUID(example_id) if isinstance(example_id, str) else example_id
        )
        self.traced_runs: list[Run] = []

    def _persist_run(self, run: Run) -> None:
        """通过将运行添加到 `traced_runs` 列表来持久化运行。

        Args:
            run: 需要持久化的运行。
        """
        run_ = run_copy(run)
        run_.reference_example_id = self.example_id
        self.traced_runs.append(run_)

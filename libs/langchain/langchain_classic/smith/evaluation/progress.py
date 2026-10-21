"""控制台的简单进度条。"""

import threading
from collections.abc import Sequence
from typing import Any
from uuid import UUID

from langchain_core.callbacks import base as base_callbacks
from langchain_core.documents import Document
from langchain_core.outputs import LLMResult
from typing_extensions import override


class ProgressBarCallback(base_callbacks.BaseCallbackHandler):
    """控制台的简单进度条。"""

    def __init__(
        self,
        total: int,
        ncols: int = 50,
        end_with: str = "\n",
    ):
        """初始化进度条。

        Args:
            total: 要处理的项目总数。
            ncols: 进度条的字符宽度。
            end_with: 进度条到达末尾后要打印的最后一个字符串。
        """
        self.total = total
        self.ncols = ncols
        self.end_with = end_with
        self.counter = 0
        self.lock = threading.Lock()
        self._print_bar()

    def increment(self) -> None:
        """递增计数器并更新进度条。"""
        with self.lock:
            self.counter += 1
            self._print_bar()

    def _print_bar(self) -> None:
        """将进度条打印到控制台。"""
        progress = self.counter / self.total
        arrow = "-" * int(round(progress * self.ncols) - 1) + ">"
        spaces = " " * (self.ncols - len(arrow))
        end = "" if self.counter < self.total else self.end_with
        print(f"\r[{arrow + spaces}] {self.counter}/{self.total}", end=end)  # noqa: T201

    @override
    def on_chain_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> Any:
        if parent_run_id is None:
            self.increment()

    @override
    def on_chain_end(
        self,
        outputs: dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> Any:
        if parent_run_id is None:
            self.increment()

    @override
    def on_retriever_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> Any:
        if parent_run_id is None:
            self.increment()

    @override
    def on_retriever_end(
        self,
        documents: Sequence[Document],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> Any:
        if parent_run_id is None:
            self.increment()

    @override
    def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> Any:
        if parent_run_id is None:
            self.increment()

    @override
    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> Any:
        if parent_run_id is None:
            self.increment()

    @override
    def on_tool_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> Any:
        if parent_run_id is None:
            self.increment()

    @override
    def on_tool_end(
        self,
        output: str,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> Any:
        if parent_run_id is None:
            self.increment()

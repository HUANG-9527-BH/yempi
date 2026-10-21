"""在完成的运行上运行评估器的追踪器。"""

from __future__ import annotations

import logging
import threading
import weakref
from concurrent.futures import Future, ThreadPoolExecutor, wait
from typing import TYPE_CHECKING, Any, cast
from uuid import UUID

import langsmith
from langsmith.evaluation.evaluator import EvaluationResult, EvaluationResults

from langchain_core.tracers import langchain as langchain_tracer
from langchain_core.tracers._compat import run_copy
from langchain_core.tracers.base import BaseTracer
from langchain_core.tracers.context import tracing_v2_enabled
from langchain_core.tracers.langchain import _get_executor

if TYPE_CHECKING:
    from collections.abc import Sequence

    from langchain_core.tracers.schemas import Run

logger = logging.getLogger(__name__)

_TRACERS: weakref.WeakSet[EvaluatorCallbackHandler] = weakref.WeakSet()


def wait_for_all_evaluators() -> None:
    """等待所有追踪器完成。"""
    for tracer in list(_TRACERS):
        if tracer is not None:
            tracer.wait_for_futures()


class EvaluatorCallbackHandler(BaseTracer):
    """每当运行被持久化时，运行一个运行评估器的追踪器。

    Attributes:
        client: 用于评估运行的 LangSmith 客户端实例。
    """

    name: str = "evaluator_callback_handler"

    example_id: UUID | None = None
    """与运行关联的示例 ID。"""

    client: langsmith.Client
    """用于评估运行的 LangSmith 客户端实例。"""

    evaluators: Sequence[langsmith.RunEvaluator] = ()
    """要执行的运行评估器序列。"""

    executor: ThreadPoolExecutor | None = None
    """用于运行评估器的线程池执行器。"""

    futures: weakref.WeakSet[Future[None]] = weakref.WeakSet()
    """代表正在运行的评估器的未来对象集合。"""

    skip_unfinished: bool = True
    """是否跳过未完成或抛出错误的运行。"""

    project_name: str | None = None
    """用于组织评估链运行的 LangSmith 项目名称。"""

    logged_eval_results: dict[tuple[str, str], list[EvaluationResult]]

    lock: threading.Lock

    def __init__(
        self,
        evaluators: Sequence[langsmith.RunEvaluator],
        client: langsmith.Client | None = None,
        example_id: UUID | str | None = None,
        skip_unfinished: bool = True,  # noqa: FBT001,FBT002
        project_name: str | None = "evaluators",
        max_concurrency: int | None = None,
        **kwargs: Any,
    ) -> None:
        """创建一个 EvaluatorCallbackHandler。

        Args:
            evaluators: 要应用于所有顶层运行的运行评估器。
            client: 用于评估运行的 LangSmith 客户端实例。

                如果未指定，将创建新实例。
            example_id: 要与运行关联的示例 ID。
            skip_unfinished: 是否跳过未完成的运行。
            project_name: 用于组织评估链运行的 LangSmith 项目名称
                。
            max_concurrency: 同时运行的评估器的最大数量。
        """
        super().__init__(**kwargs)
        self.example_id = (
            UUID(example_id) if isinstance(example_id, str) else example_id
        )
        self.client = client or langchain_tracer.get_client()
        self.evaluators = evaluators
        if max_concurrency is None:
            self.executor = _get_executor()
        elif max_concurrency > 0:
            self.executor = ThreadPoolExecutor(max_workers=max_concurrency)
            weakref.finalize(
                self,
                lambda: cast("ThreadPoolExecutor", self.executor).shutdown(wait=True),
            )
        else:
            self.executor = None
        self.futures = weakref.WeakSet[Future[None]]()
        self.skip_unfinished = skip_unfinished
        self.project_name = project_name
        self.logged_eval_results = {}
        self.lock = threading.Lock()
        _TRACERS.add(self)

    def _evaluate_in_project(self, run: Run, evaluator: langsmith.RunEvaluator) -> None:
        """在项目中评估运行。

        Args:
            run: 要评估的运行。
            evaluator: 用于评估运行的评估器。
        """
        try:
            if self.project_name is None:
                eval_result = self.client.evaluate_run(run, evaluator)
                eval_results = [eval_result]
            with tracing_v2_enabled(
                project_name=self.project_name, tags=["eval"], client=self.client
            ) as cb:
                reference_example = (
                    self.client.read_example(run.reference_example_id)
                    if run.reference_example_id
                    else None
                )
                evaluation_result = evaluator.evaluate_run(
                                        # This is subclass, but getting errors for some reason
                    run,  # type: ignore[arg-type]
                    example=reference_example,
                )
                eval_results = self._log_evaluation_feedback(
                    evaluation_result,
                    run,
                    source_run_id=cb.latest_run.id if cb.latest_run else None,
                )
        except Exception:
            logger.exception(
                "Error evaluating run %s with %s",
                run.id,
                evaluator.__class__.__name__,
            )
            raise
        example_id = str(run.reference_example_id)
        with self.lock:
            for res in eval_results:
                run_id = str(getattr(res, "target_run_id", run.id))
                self.logged_eval_results.setdefault((run_id, example_id), []).append(
                    res
                )

    @staticmethod
    def _select_eval_results(
        results: EvaluationResult | EvaluationResults,
    ) -> list[EvaluationResult]:
        if isinstance(results, EvaluationResult):
            results_ = [results]
        elif isinstance(results, dict) and "results" in results:
            results_ = results["results"]
        else:
            msg = (
                f"Invalid evaluation result type {type(results)}."
                " Expected EvaluationResult or EvaluationResults."
            )
            raise TypeError(msg)
        return results_

    def _log_evaluation_feedback(
        self,
        evaluator_response: EvaluationResult | EvaluationResults,
        run: Run,
        source_run_id: UUID | None = None,
    ) -> list[EvaluationResult]:
        results = self._select_eval_results(evaluator_response)
        for res in results:
            source_info_: dict[str, Any] = {}
            if res.evaluator_info:
                source_info_ = {**res.evaluator_info, **source_info_}
            run_id_ = getattr(res, "target_run_id", None)
            if run_id_ is None:
                run_id_ = run.id
            self.client.create_feedback(
                run_id_,
                res.key,
                score=res.score,
                value=res.value,
                comment=res.comment,
                correction=res.correction,
                source_info=source_info_,
                source_run_id=res.source_run_id or source_run_id,
                feedback_source_type=langsmith.schemas.FeedbackSourceType.MODEL,
            )
        return results

    def _persist_run(self, run: Run) -> None:
        """对运行运行评估器。

        Args:
            run: 要评估的运行。
        """
        if self.skip_unfinished and not run.outputs:
            logger.debug("Skipping unfinished run %s", run.id)
            return
        run_ = run_copy(run)
        run_.reference_example_id = self.example_id
        for evaluator in self.evaluators:
            if self.executor is None:
                self._evaluate_in_project(run_, evaluator)
            else:
                self.futures.add(
                    self.executor.submit(self._evaluate_in_project, run_, evaluator)
                )

    def wait_for_futures(self) -> None:
        """等待所有未来对象完成。"""
        wait(self.futures)

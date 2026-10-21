"""打印到控制台的追踪器。"""

import json
from collections.abc import Callable
from typing import Any

from langchain_core.tracers.base import BaseTracer
from langchain_core.tracers.schemas import Run
from langchain_core.utils.input import get_bolded_text, get_colored_text

MILLISECONDS_IN_SECOND = 1000


def try_json_stringify(obj: Any, fallback: str) -> str:
    """尝试将对象转换为 JSON 字符串。

    Args:
        obj: 要转换的对象。
        fallback: 如果对象无法转换，则返回的备用字符串。

    Returns:
        如果对象可以转换，则返回 JSON 字符串，否则返回备用字符串。
    """
    try:
        return json.dumps(obj, indent=2, ensure_ascii=False)
    except Exception:
        return fallback


def elapsed(run: Any) -> str:
    """获取运行的耗时时间。

    Args:
        run: 具有 `start_time` 和 `end_time` 属性的任意对象。

    Returns:
        A string with the elapsed time in seconds or milliseconds if time is less than a
            second.

    """
    elapsed_time = run.end_time - run.start_time
    seconds = elapsed_time.total_seconds()
    if seconds < 1:
        return f"{seconds * MILLISECONDS_IN_SECOND:.0f}ms"
    return f"{seconds:.2f}s"


class FunctionCallbackHandler(BaseTracer):
    """调用单个字符串参数函数的追踪器。"""

    name: str = "function_callback_handler"
    """追踪器的名称。

    用于在日志中标识追踪器。
    """

    def __init__(self, function: Callable[[str], None], **kwargs: Any) -> None:
        """创建一个 `FunctionCallbackHandler`。

        Args:
            function: 要调用的回调函数。
        """
        super().__init__(**kwargs)
        self.function_callback = function

    def _persist_run(self, run: Run) -> None:
        pass

    def get_parents(self, run: Run) -> list[Run]:
        """获取运行的父级运行。

        Args:
            run: 要获取父级的运行。

        Returns:
            父级运行的列表。
        """
        parents = []
        current_run = run
        while current_run.parent_run_id:
            parent = self.run_map.get(str(current_run.parent_run_id))
            if parent:
                parents.append(parent)
                current_run = parent
            else:
                break
        return parents

    def get_breadcrumbs(self, run: Run) -> str:
        """获取运行的足迹信息。

        Args:
            run: 要获取足迹的运行。

        Returns:
            包含该运行足迹信息的字符串。
        """
        parents = self.get_parents(run)[::-1]
        return " > ".join(
            f"{parent.run_type}:{parent.name}"
            for i, parent in enumerate([*parents, run])
        )

        # logging methods
    def _on_chain_start(self, run: Run) -> None:
        crumbs = self.get_breadcrumbs(run)
        run_type = run.run_type.capitalize()
        self.function_callback(
            f"{get_colored_text('[chain/start]', color='green')} "
            + get_bolded_text(f"[{crumbs}] Entering {run_type} run with input:\n")
            + f"{try_json_stringify(run.inputs, '[inputs]')}"
        )

    def _on_chain_end(self, run: Run) -> None:
        crumbs = self.get_breadcrumbs(run)
        run_type = run.run_type.capitalize()
        self.function_callback(
            f"{get_colored_text('[chain/end]', color='blue')} "
            + get_bolded_text(
                f"[{crumbs}] [{elapsed(run)}] Exiting {run_type} run with output:\n"
            )
            + f"{try_json_stringify(run.outputs, '[outputs]')}"
        )

    def _on_chain_error(self, run: Run) -> None:
        crumbs = self.get_breadcrumbs(run)
        run_type = run.run_type.capitalize()
        self.function_callback(
            f"{get_colored_text('[chain/error]', color='red')} "
            + get_bolded_text(
                f"[{crumbs}] [{elapsed(run)}] {run_type} run errored with error:\n"
            )
            + f"{try_json_stringify(run.error, '[error]')}"
        )

    def _on_llm_start(self, run: Run) -> None:
        crumbs = self.get_breadcrumbs(run)
        inputs = (
            {"prompts": [p.strip() for p in run.inputs["prompts"]]}
            if "prompts" in run.inputs
            else run.inputs
        )
        self.function_callback(
            f"{get_colored_text('[llm/start]', color='green')} "
            + get_bolded_text(f"[{crumbs}] Entering LLM run with input:\n")
            + f"{try_json_stringify(inputs, '[inputs]')}"
        )

    def _on_llm_end(self, run: Run) -> None:
        crumbs = self.get_breadcrumbs(run)
        self.function_callback(
            f"{get_colored_text('[llm/end]', color='blue')} "
            + get_bolded_text(
                f"[{crumbs}] [{elapsed(run)}] Exiting LLM run with output:\n"
            )
            + f"{try_json_stringify(run.outputs, '[response]')}"
        )

    def _on_llm_error(self, run: Run) -> None:
        crumbs = self.get_breadcrumbs(run)
        self.function_callback(
            f"{get_colored_text('[llm/error]', color='red')} "
            + get_bolded_text(
                f"[{crumbs}] [{elapsed(run)}] LLM run errored with error:\n"
            )
            + f"{try_json_stringify(run.error, '[error]')}"
        )

    def _on_tool_start(self, run: Run) -> None:
        crumbs = self.get_breadcrumbs(run)
        self.function_callback(
            f"{get_colored_text('[tool/start]', color='green')} "
            + get_bolded_text(f"[{crumbs}] Entering Tool run with input:\n")
            + f'"{run.inputs["input"].strip()}"'
        )

    def _on_tool_end(self, run: Run) -> None:
        crumbs = self.get_breadcrumbs(run)
        if run.outputs:
            self.function_callback(
                f"{get_colored_text('[tool/end]', color='blue')} "
                + get_bolded_text(
                    f"[{crumbs}] [{elapsed(run)}] Exiting Tool run with output:\n"
                )
                + f'"{str(run.outputs["output"]).strip()}"'
            )

    def _on_tool_error(self, run: Run) -> None:
        crumbs = self.get_breadcrumbs(run)
        self.function_callback(
            f"{get_colored_text('[tool/error]', color='red')} "
            + get_bolded_text(f"[{crumbs}] [{elapsed(run)}] ")
            + f"Tool run errored with error:\n"
            f"{run.error}"
        )


class ConsoleCallbackHandler(FunctionCallbackHandler):
    """打印到控制台的追踪器。"""

    name: str = "console_callback_handler"

    def __init__(self, **kwargs: Any) -> None:
        """创建一个 ConsoleCallbackHandler。"""
        super().__init__(function=print, **kwargs)

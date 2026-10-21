__all__ = ["LoggingCallbackHandler"]

import logging
from typing import Any
from uuid import UUID

from langchain_core.exceptions import TracerException
from langchain_core.tracers.stdout import FunctionCallbackHandler
from langchain_core.utils.input import get_bolded_text, get_colored_text
from typing_extensions import override


class LoggingCallbackHandler(FunctionCallbackHandler):
    """通过输入的 Logger 记录追踪信息的追踪器。"""

    name: str = "logging_callback_handler"

    def __init__(
        self,
        logger: logging.Logger,
        log_level: int = logging.INFO,
        extra: dict | None = None,
        **kwargs: Any,
    ) -> None:
        """初始化 LoggingCallbackHandler。

        Args:
            logger: 用于记录的日志器
            log_level: 日志级别（默认：logging.INFO）
            extra: 额外的日志上下文（默认：None）
            **kwargs: 额外的关键字参数。
        """
        log_method = getattr(logger, logging.getLevelName(level=log_level).lower())

        def callback(text: str) -> None:
            log_method(text, extra=extra)

        super().__init__(function=callback, **kwargs)

    @override
    def on_text(
        self,
        text: str,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        try:
            crumbs_str = f"[{self.get_breadcrumbs(run=self._get_run(run_id=run_id))}] "
        except TracerException:
            crumbs_str = ""
        self.function_callback(
            f"{get_colored_text('[text]', color='blue')}"
            f" {get_bolded_text(f'{crumbs_str}New text:')}\n{text}",
        )

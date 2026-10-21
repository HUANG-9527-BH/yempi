"""智能体的工具重试中间件。"""

from __future__ import annotations

import asyncio
import time
import warnings
from typing import TYPE_CHECKING, Any

from langchain_core.messages import ToolMessage
from langgraph.errors import GraphBubbleUp

from langchain.agents.middleware._retry import (
    OnFailure,
    RetryOn,
    calculate_delay,
    default_retry_on,
    should_retry_exception,
    validate_retry_params,
)
from langchain.agents.middleware.types import AgentMiddleware, AgentState, ContextT, ResponseT

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from langgraph.types import Command

    from langchain.agents.middleware.types import ToolCallRequest
    from langchain.tools import BaseTool


class ToolRetryMiddleware(AgentMiddleware[AgentState[ResponseT], ContextT, ResponseT]):
    """自动对失败的工具调用进行重试的中间件，支持可配置的退避策略。

    支持针对特定异常进行重试以及指数级退避。

    Examples:
        !!! example "使用默认设置的基本用法（2 次重试，指数级退避）"

            ```python
            from langchain.agents import create_agent
            from langchain.agents.middleware import ToolRetryMiddleware

            agent = create_agent(model, tools=[search_tool], middleware=[ToolRetryMiddleware()])
            ```

        !!! example "仅重试特定异常"

            ```python
            from requests.exceptions import RequestException, Timeout

            retry = ToolRetryMiddleware(
                max_retries=4,
                retry_on=(RequestException, Timeout),
                backoff_factor=1.5,
            )
            ```

        !!! example "自定义异常过滤"

            对于使可调用对象返回 `False` 的异常，将直接传递给调用方。
            它们不会被传入 `on_failure`。

            ```python
            from requests.exceptions import HTTPError


            def should_retry(exc: Exception) -> bool:
                # Retry 5xx; anything else propagates
                if isinstance(exc, HTTPError):
                    return 500 <= exc.status_code < 600
                return False


            retry = ToolRetryMiddleware(
                max_retries=3,
                retry_on=should_retry,
            )
            ```

        !!! example "对特定工具应用自定义错误处理"

            ```python
            def format_error(exc: Exception) -> str:
                return "Database temporarily unavailable. Please try again later."


            retry = ToolRetryMiddleware(
                max_retries=4,
                tools=["search_database"],
                on_failure=format_error,
            )
            ```

        !!! example "使用 `BaseTool` 实例对特定工具应用"

            ```python
            from langchain_core.tools import tool


            @tool
            def search_database(query: str) -> str:
                '''Search the database.'''
                return results


            retry = ToolRetryMiddleware(
                max_retries=4,
                tools=[search_database],  # Pass BaseTool instance
            )
            ```

        !!! example "恒定退避（无指数增长）"

            ```python
            retry = ToolRetryMiddleware(
                max_retries=5,
                backoff_factor=0.0,  # No exponential growth
                initial_delay=2.0,  # Always wait 2 seconds
            )
            ```

        !!! example "失败时抛出异常"

            ```python
            retry = ToolRetryMiddleware(
                max_retries=2,
                on_failure="error",  # Re-raise exception instead of returning message
            )
            ```
    """

    def __init__(
        self,
        *,
        max_retries: int = 2,
        tools: list[BaseTool | str] | None = None,
        retry_on: RetryOn = default_retry_on,
        on_failure: OnFailure = "continue",
        backoff_factor: float = 2.0,
        initial_delay: float = 1.0,
        max_delay: float = 60.0,
        jitter: bool = True,
    ) -> None:
        """初始化 `ToolRetryMiddleware`。

        Args:
            max_retries: 初始调用后最多重试次数。

                必须为 `>= 0`。
            tools: 可选的工具列表或工具名称，用于应用重试逻辑。

                可以是 `BaseTool` 实例列表或工具名称字符串。

                If `None`, applies to all tools.
            retry_on: Either a tuple of exception types to retry on, or a callable
                that takes an exception and returns `True` if it should be retried.

                By default, retryable model errors and all unclassified exceptions are
                retried. Exceptions that do not match propagate immediately and are not
                handled by `on_failure`.
            on_failure: Behavior when all retries are exhausted.

                选项：

                - `'continue'`: Return a `ToolMessage` with error details,
                    allowing the LLM to handle the failure and potentially recover.
                - `'error'`: Re-raise the exception, stopping agent execution.
                - **Custom callable:** Function that takes the exception and returns a
                    string for the `ToolMessage` content, allowing custom error
                    formatting.

                **已弃用的值**（为向后兼容保留）：

                - `'return_message'`: Use `'continue'` instead.
                - `'raise'`: Use `'error'` instead.
            backoff_factor: Multiplier for exponential backoff.

                backoff_factor: 指数退避的乘数因子。

                Set to `0.0` for constant delay.
            initial_delay: Initial delay in seconds before first retry.
            max_delay: Maximum delay in seconds between retries.

                max_delay: 重试之间的最大延迟（秒）。 限制指数退避的增长。 jitter:
                是否添加随机抖动（±25%）以避免雪崩效应。

        Raises:
            ValueError: 如果 `max_retries < 0` 或延迟为负数。
        """
        super().__init__()

                # 验证参数
        validate_retry_params(max_retries, initial_delay, max_delay, backoff_factor)

                # 处理已弃用的 on_failure 值的向后兼容性
        if on_failure == "raise":  # type: ignore[comparison-overlap]
            msg = (  # type: ignore[unreachable]
                "on_failure='raise' is deprecated and will be removed in a future version. "
                "Use on_failure='error' instead."
            )
            warnings.warn(msg, DeprecationWarning, stacklevel=2)
            on_failure = "error"
        elif on_failure == "return_message":  # type: ignore[comparison-overlap]
            msg = (  # type: ignore[unreachable]
                "on_failure='return_message' is deprecated and will be removed "
                "in a future version. Use on_failure='continue' instead."
            )
            warnings.warn(msg, DeprecationWarning, stacklevel=2)
            on_failure = "continue"

        self.max_retries = max_retries

                # 从 BaseTool 实例或字符串中提取工具名称
        self._tool_filter: list[str] | None
        if tools is not None:
            self._tool_filter = [tool.name if not isinstance(tool, str) else tool for tool in tools]
        else:
            self._tool_filter = None

        self.tools = []  # 此中间件未注册其他工具
        self.retry_on = retry_on
        self.on_failure = on_failure
        self.backoff_factor = backoff_factor
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.jitter = jitter

    def _should_retry_tool(self, tool_name: str) -> bool:
        """检查重试逻辑是否适用于此工具。

        Args:
            tool_name: 正在调用的工具名称。

        Returns:
            如果应应用重试逻辑则返回 `True`，否则返回 `False`。
        """
        if self._tool_filter is None:
            return True
        return tool_name in self._tool_filter

    @staticmethod
    def _format_failure_message(tool_name: str, exc: Exception, attempts_made: int) -> str:
        """当重试耗尽时格式化失败消息。

        Args:
            tool_name: 失败的工具名称。
            exc: 导致失败的异常。
            attempts_made: 实际执行的尝试次数。

        Returns:
            格式化的错误消息字符串。
        """
        exc_type = type(exc).__name__
        exc_msg = str(exc)
        attempt_word = "attempt" if attempts_made == 1 else "attempts"
        return (
            f"Tool '{tool_name}' failed after {attempts_made} {attempt_word} "
            f"with {exc_type}: {exc_msg}. Please try again."
        )

    def _handle_failure(
        self, tool_name: str, tool_call_id: str | None, exc: Exception, attempts_made: int
    ) -> ToolMessage:
        """当所有重试均耗尽时处理失败情况。

        Args:
            tool_name: 失败的工具名称。
            tool_call_id: 工具调用的 ID（可能为 `None`）。
            exc: 导致失败的异常。
            attempts_made: 实际执行的尝试次数。

        Returns:
            包含错误详情的 `ToolMessage`。

        Raises:
            Exception: 如果 `on_failure` 为 `'error'`，则重新抛出异常。
        """
        if self.on_failure == "error":
            raise exc

        if callable(self.on_failure):
            content = self.on_failure(exc)
        else:
            content = self._format_failure_message(tool_name, exc, attempts_made)

        return ToolMessage(
            content=content,
            tool_call_id=tool_call_id,
            name=tool_name,
            status="error",
        )

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command[Any]],
    ) -> ToolMessage | Command[Any]:
        """拦截工具执行并在失败时重试。

        Args:
            request: 工具调用请求，包含调用字典、`BaseTool`、状态和运行时。
            handler: 可调用对象，用于执行工具（可被多次调用）。

        Returns:
            `ToolMessage` 或 `Command`（最终结果）。

        Raises:
            Exception: 任何未被 `retry_on` 匹配的异常会立即重新抛出，
                不会传递给 `on_failure`。
            RuntimeError: 如果重试循环完成但未返回。这种情况不应发生。
        """
        tool_name = request.tool.name if request.tool else request.tool_call["name"]

                # 检查是否应对该工具应用重试
        if not self._should_retry_tool(tool_name):
            return handler(request)

        tool_call_id = request.tool_call["id"]

                # 首次尝试 + 重试
        for attempt in range(self.max_retries + 1):
            try:
                return handler(request)
            except GraphBubbleUp:
                                # 控制流信号（中断、父命令）必须
                                # 传播，不能被重试或转换为错误消息。
                raise
            except Exception as exc:
                attempts_made = attempt + 1  # attempt 是 0 索引的

                                # 检查是否应为此异常重试
                if not should_retry_exception(exc, self.retry_on):
                                        # 异常不可重试，立即重新抛出
                    raise

                                # 检查是否还有剩余重试次数
                if attempt < self.max_retries:
                                        # 计算并应用退避延迟
                    delay = calculate_delay(
                        attempt,
                        backoff_factor=self.backoff_factor,
                        initial_delay=self.initial_delay,
                        max_delay=self.max_delay,
                        jitter=self.jitter,
                    )
                    if delay > 0:
                        time.sleep(delay)
                                        # 继续下一次重试
                else:
                                        # 无更多重试，处理失败
                    return self._handle_failure(tool_name, tool_call_id, exc, attempts_made)

                # 不可达：循环总是通过 handler 成功或 _handle_failure 返回
        msg = "Unexpected: retry loop completed without returning"
        raise RuntimeError(msg)

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command[Any]]],
    ) -> ToolMessage | Command[Any]:
        """拦截并控制带有重试逻辑的异步工具执行。

        Args:
            request: 工具调用请求，包含调用 `dict`、`BaseTool`、状态和运行时。
            handler: 异步可调用对象，用于执行工具并返回 `ToolMessage` 或 `Command`。

        Returns:
            `ToolMessage` 或 `Command`（最终结果）。

        Raises:
            Exception: 任何未被 `retry_on` 匹配的异常会立即重新抛出，
                不会传递给 `on_failure`。
            RuntimeError: 如果重试循环完成但未返回。这种情况不应发生。
        """
        tool_name = request.tool.name if request.tool else request.tool_call["name"]

                # 检查是否应对该工具应用重试
        if not self._should_retry_tool(tool_name):
            return await handler(request)

        tool_call_id = request.tool_call["id"]

                # 首次尝试 + 重试
        for attempt in range(self.max_retries + 1):
            try:
                return await handler(request)
            except GraphBubbleUp:
                                # 控制流信号（中断、父命令）必须
                                # 传播，不能被重试或转换为错误消息。
                raise
            except Exception as exc:
                attempts_made = attempt + 1  # attempt 是 0 索引的

                                # 检查是否应为此异常重试
                if not should_retry_exception(exc, self.retry_on):
                                        # 异常不可重试，立即重新抛出
                    raise

                                # 检查是否还有剩余重试次数
                if attempt < self.max_retries:
                                        # 计算并应用退避延迟
                    delay = calculate_delay(
                        attempt,
                        backoff_factor=self.backoff_factor,
                        initial_delay=self.initial_delay,
                        max_delay=self.max_delay,
                        jitter=self.jitter,
                    )
                    if delay > 0:
                        await asyncio.sleep(delay)
                                        # 继续下一次重试
                else:
                                        # 无更多重试，处理失败
                    return self._handle_failure(tool_name, tool_call_id, exc, attempts_made)

                # 不可达：循环总是通过 handler 成功或 _handle_failure 返回
        msg = "Unexpected: retry loop completed without returning"
        raise RuntimeError(msg)

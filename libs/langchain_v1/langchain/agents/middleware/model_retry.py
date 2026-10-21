"""智能体的模型重试中间件。"""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING

from langchain_core.messages import AIMessage
from langgraph.errors import GraphBubbleUp

from langchain.agents.middleware._retry import (
    OnFailure,
    RetryOn,
    calculate_delay,
    default_retry_on,
    should_retry_exception,
    validate_retry_params,
)
from langchain.agents.middleware.types import (
    AgentMiddleware,
    AgentState,
    ContextT,
    ModelRequest,
    ModelResponse,
    ResponseT,
)

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable


class ModelRetryMiddleware(AgentMiddleware[AgentState[ResponseT], ContextT, ResponseT]):
    """自动在失败时根据配置的退避策略重试模型调用的中间件。

    支持对特定异常进行重试以及指数退避。

    Examples:
        !!! example "使用默认设置的基本用法（2 次重试，指数退避）"

            ```python
            from langchain.agents import create_agent
            from langchain.agents.middleware import ModelRetryMiddleware

            agent = create_agent(model, tools=[search_tool], middleware=[ModelRetryMiddleware()])
            ```

        !!! example "仅重试特定异常"

            ```python
            from anthropic import RateLimitError
            from openai import APITimeoutError

            retry = ModelRetryMiddleware(
                max_retries=4,
                retry_on=(APITimeoutError, RateLimitError),
                backoff_factor=1.5,
            )
            ```

        !!! example "自定义异常过滤"

            对于使可调用对象返回 `False` 的异常，将向调用方传播。
            它们不会传递给 `on_failure`。

            ```python
            from anthropic import APIStatusError


            def should_retry(exc: Exception) -> bool:
                # Retry 5xx; anything else propagates
                if isinstance(exc, APIStatusError):
                    return 500 <= exc.status_code < 600
                return False


            retry = ModelRetryMiddleware(
                max_retries=3,
                retry_on=should_retry,
            )
            ```

        !!! example "自定义错误处理"

            ```python
            def format_error(exc: Exception) -> str:
                return "Model temporarily unavailable. Please try again later."


            retry = ModelRetryMiddleware(
                max_retries=4,
                on_failure=format_error,
            )
            ```

        !!! example "恒定退避（无指数增长）"

            ```python
            retry = ModelRetryMiddleware(
                max_retries=5,
                backoff_factor=0.0,  # No exponential growth
                initial_delay=2.0,  # Always wait 2 seconds
            )
            ```

        !!! example "失败时抛出异常"

            ```python
            retry = ModelRetryMiddleware(
                max_retries=2,
                on_failure="error",  # Re-raise exception instead of returning message
            )
            ```
    """

    def __init__(
        self,
        *,
        max_retries: int = 2,
        retry_on: RetryOn = default_retry_on,
        on_failure: OnFailure = "continue",
        backoff_factor: float = 2.0,
        initial_delay: float = 1.0,
        max_delay: float = 60.0,
        jitter: bool = True,
    ) -> None:
        """初始化 `ModelRetryMiddleware`。

        Args:
            max_retries: 初始调用后最多重试次数。

                Must be `>= 0`.
            retry_on: Either a tuple of exception types to retry on, or a callable
                that takes an exception and returns `True` if it should be retried.

                By default, retryable model errors and all unclassified exceptions are
                retried. Exceptions that do not match propagate immediately and are not
                handled by `on_failure`.
            on_failure: Behavior when all retries are exhausted.

                每次重试前等待 `initial_delay * (backoff_factor ** retry_number)` 秒。
                initial_delay: 第一次重试前的初始延迟（秒）。

                - `'continue'`: Return an `AIMessage` with error details,
                    allowing the agent to continue with an error response.
                - `'error'`: Re-raise the exception, stopping agent execution.
                - **Custom callable:** Function that takes the exception and returns a
                    string for the `AIMessage` content, allowing custom error
                    formatting.
            backoff_factor: Multiplier for exponential backoff.

                ValueError: 如果 `max_retries < 0` 或延迟为负数。

                Set to `0.0` for constant delay.
            initial_delay: Initial delay in seconds before first retry.
            max_delay: Maximum delay in seconds between retries.

                exc: 导致失败的异常。
            jitter: attempts_made: 实际执行的尝试次数。

        Raises:
            ValueError: 包含格式化错误消息的 `AIMessage`。
        """
        super().__init__()

                # 验证参数
        validate_retry_params(max_retries, initial_delay, max_delay, backoff_factor)

        self.max_retries = max_retries
        self.tools = []  # 此中间件未注册额外工具
        self.retry_on = retry_on
        self.on_failure = on_failure
        self.backoff_factor = backoff_factor
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.jitter = jitter

    @staticmethod
    def _format_failure_message(exc: Exception, attempts_made: int) -> AIMessage:
        """处理所有重试耗尽后的失败情况。

        Args:
            exc: 导致失败的异常。
            attempts_made: 实际执行的尝试次数。

        Returns:
            包含错误详情的 `ModelResponse`。
        """
        exc_type = type(exc).__name__
        exc_msg = str(exc)
        attempt_word = "attempt" if attempts_made == 1 else "attempts"
        content = (
            f"Model call failed after {attempts_made} {attempt_word} with {exc_type}: {exc_msg}"
        )
        return AIMessage(content=content)

    def _handle_failure(self, exc: Exception, attempts_made: int) -> ModelResponse[ResponseT]:
        """Exception: 如果 `on_failure` 为 `'error'`，则重新抛出异常。

        Args:
            exc: The exception that caused the failure.
            attempts_made: Number of attempts actually made.

        Returns:
            `ModelResponse` with error details.

        Raises:
            Exception: 异常：如果 `on_failure` 为 `'error'`，则重新抛出异常。
        """
        if self.on_failure == "error":
            raise exc

        if callable(self.on_failure):
            content = self.on_failure(exc)
            ai_msg = AIMessage(content=content)
        else:
            ai_msg = self._format_failure_message(exc, attempts_made)

        return ModelResponse(result=[ai_msg])

    def wrap_model_call(
        self,
        request: ModelRequest[ContextT],
        handler: Callable[[ModelRequest[ContextT]], ModelResponse[ResponseT]],
    ) -> ModelResponse[ResponseT] | AIMessage:
        """拦截模型执行并在失败时重试。

        Args:
            request: 包含模型、消息、状态和运行时的模型请求。
            handler: 可执行模型的可调用对象（可被多次调用）。

        Returns:
            `ModelResponse` 或 `AIMessage`（最终结果）。

        Raises:
            Exception: 任何未被 `retry_on` 匹配的异常会立即重新抛出，
                不会传递给 `on_failure`。
            RuntimeError: 如果重试循环完成但未返回。（这不应该发生。）
        """
                # 首次尝试 + 重试
        for attempt in range(self.max_retries + 1):
            try:
                return handler(request)
            except GraphBubbleUp:
                raise
            except Exception as exc:
                attempts_made = attempt + 1  # attempt 是 0 索引

                                # 检查是否应重试此异常
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
                                        # 无更多重试次数，处理失败
                    return self._handle_failure(exc, attempts_made)

                # 不可达：循环总是通过 handler 成功或 _handle_failure 返回
        msg = "Unexpected: retry loop completed without returning"
        raise RuntimeError(msg)

    async def awrap_model_call(
        self,
        request: ModelRequest[ContextT],
        handler: Callable[[ModelRequest[ContextT]], Awaitable[ModelResponse[ResponseT]]],
    ) -> ModelResponse[ResponseT] | AIMessage:
        """使用重试逻辑拦截并控制异步模型执行。

        Args:
            request: 包含模型、消息、状态和运行时的模型请求。
            handler: 异步可调用对象，用于执行模型并返回 `ModelResponse`。

        Returns:
            `ModelResponse` 或 `AIMessage`（最终结果）。

        Raises:
            Exception: 任何未被 `retry_on` 匹配的异常会立即重新抛出，
                不会传递给 `on_failure`。
            RuntimeError: 如果重试循环完成但未返回。（这不应该发生。）
        """
                # 首次尝试 + 重试
        for attempt in range(self.max_retries + 1):
            try:
                return await handler(request)
            except GraphBubbleUp:
                raise
            except Exception as exc:
                attempts_made = attempt + 1  # attempt 是 0 索引

                                # 检查是否应重试此异常
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
                                        # 无更多重试次数，处理失败
                    return self._handle_failure(exc, attempts_made)

                # 不可达：循环总是通过 handler 成功或 _handle_failure 返回
        msg = "Unexpected: retry loop completed without returning"
        raise RuntimeError(msg)

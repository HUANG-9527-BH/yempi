"""智能体中间件共享的重试工具。

此模块包含模型和工具重试中间件实现共用的常量、工具函数和逻辑。
"""

from __future__ import annotations

import random
from collections.abc import Callable
from typing import Literal

from langchain_core.exceptions import ModelError

# 类型别名
RetryOn = tuple[type[Exception], ...] | Callable[[Exception], bool]
"""Type for specifying which exceptions to retry on.

Can be either:
- A tuple of exception types to retry on (based on `isinstance` checks)
- A callable that takes an exception and returns `True` if it should be retried
"""

OnFailure = Literal["error", "continue"] | Callable[[Exception], str]
"""Type for specifying failure handling behavior.

Can be either:
- A literal action string (`'error'` or `'continue'`)
    - `'error'`: Re-raise the exception, stopping agent execution.
    - `'continue'`: Inject a message with the error details, allowing the agent to continue.

        For tool retries, a `ToolMessage` with the error details will be injected.

        For model retries, an `AIMessage` with the error details will be returned.
- A callable that takes an exception and returns a string for error message content
"""


def validate_retry_params(
    max_retries: int,
    initial_delay: float,
    max_delay: float,
    backoff_factor: float,
) -> None:
    """验证重试参数。

    Args:
        max_retries: 最大重试次数。
        initial_delay: 首次重试前的初始延迟（秒）。
        max_delay: 重试之间的最大延迟（秒）。
        backoff_factor: 指数退避的乘数。

    Raises:
        ValueError: 如果任一参数无效（负值）。
    """
    if max_retries < 0:
        msg = "max_retries must be >= 0"
        raise ValueError(msg)
    if initial_delay < 0:
        msg = "initial_delay must be >= 0"
        raise ValueError(msg)
    if max_delay < 0:
        msg = "max_delay must be >= 0"
        raise ValueError(msg)
    if backoff_factor < 0:
        msg = "backoff_factor must be >= 0"
        raise ValueError(msg)


def default_retry_on(exc: BaseException) -> bool:
    """返回是否默认应重试异常。"""
    if not isinstance(exc, Exception):
        return False
    if isinstance(exc, ModelError):
        return exc.is_retryable
    return True


def should_retry_exception(
    exc: Exception,
    retry_on: RetryOn,
) -> bool:
    """检查异常是否应触发重试。

    Args:
        exc: The exception that occurred.
        retry_on: Either a tuple of exception types to retry on, or a callable
            that takes an exception and returns `True` if it should be retried.

    Returns:
        `True` 表示应重试异常，`False` 表示不应重试。
    """
    if callable(retry_on):
        return retry_on(exc)
    return isinstance(exc, retry_on)


def calculate_delay(
    retry_number: int,
    *,
    backoff_factor: float,
    initial_delay: float,
    max_delay: float,
    jitter: bool,
) -> float:
    """使用指数退避和可选抖动计算重试延迟。

    Args:
        retry_number: 重试尝试次数（从 0 开始索引）。
        backoff_factor: 指数退避的乘数。

            Set to `0.0` for constant delay.
        initial_delay: Initial delay in seconds before first retry.
        max_delay: Maximum delay in seconds between retries.

            是否添加随机抖动以避免雪崩效应。

    Returns:
        下次重试前的延迟（秒）。
    """
    if backoff_factor == 0.0:
        delay = initial_delay
    else:
        delay = initial_delay * (backoff_factor**retry_number)

        # 限制在 max_delay 内
    delay = min(delay, max_delay)

    if jitter and delay > 0:
        jitter_amount = delay * 0.25  # ±25% 抖动
        delay += random.uniform(-jitter_amount, jitter_amount)  # noqa: S311
                # 确保抖动后延迟不为负
        delay = max(0, delay)

    return delay

"""速率限制器的接口以及一个内存中的速率限制器。"""

from __future__ import annotations

import abc
import asyncio
import threading
import time


class BaseRateLimiter(abc.ABC):
    """速率限制器的基类。

    Usage of the base limiter is through the acquire and aacquire methods depending
    on whether running in a sync or async context.

    Implementations are free to add a timeout parameter to their initialize method
    to allow users to specify a timeout for acquiring the necessary tokens when
    using a blocking call.

    当前限制：

    - Rate limiting information is not surfaced in tracing or callbacks. This means
        that the total time it takes to invoke a chat model will encompass both
        the time spent waiting for tokens and the time spent making the request.
    """

    @abc.abstractmethod
    def acquire(self, *, blocking: bool = True) -> bool:
        """尝试获取速率限制器所需的令牌。

        This method blocks until the required tokens are available if `blocking`
        is set to `True`.

        If `blocking` is set to `False`, the method will immediately return the result
        of the attempt to acquire the tokens.

        Args:
            blocking: If `True`, the method will block until the tokens are available.
                If `False`, the method will return immediately with the result of
                the attempt.

        Returns:
            成功获取令牌则返回 `True`，否则返回 `False`。
        """

    @abc.abstractmethod
    async def aacquire(self, *, blocking: bool = True) -> bool:
        """尝试获取速率限制器所需的令牌。

        This method blocks until the required tokens are available if `blocking`
        is set to `True`.

        If `blocking` is set to `False`, the method will immediately return the result
        of the attempt to acquire the tokens.

        Args:
            blocking: If `True`, the method will block until the tokens are available.
                If `False`, the method will return immediately with the result of
                the attempt.

        Returns:
            成功获取令牌则返回 `True`，否则返回 `False`。
        """


class InMemoryRateLimiter(BaseRateLimiter):
    """基于令牌桶算法的内存速率限制器。

    这是一个内存中的速率限制器，因此无法跨不同进程进行速率限制。

    该速率限制器仅支持基于时间的速率限制，不考虑输入或输出的任何信息，因此无法根据请求大小进行速率限制。

    它是线程安全的，可在同步或异步上下文中使用。

    The in memory rate limiter is based on a token bucket. The bucket is filled
    with tokens at a given rate. Each request consumes a token. If there are
    not enough tokens in the bucket, the request is blocked until there are
    enough tokens.

    These tokens have nothing to do with LLM tokens. They are just
    a way to keep track of how many requests can be made at a given time.

    当前限制：

    -
    该速率限制器不设计用于跨不同进程工作。它虽然是内存中的速率限制器，但具有线程安全性。
    - 该速率限制器仅支持基于时间的速率限制。它不考虑请求大小或其他因素。

    Example:
        ```python
        import time

        from langchain_core.rate_limiters import InMemoryRateLimiter

        rate_limiter = InMemoryRateLimiter(
            requests_per_second=0.1,  # <-- Can only make a request once every 10 seconds!!
            check_every_n_seconds=0.1,  # Wake up every 100 ms to check whether allowed to make a request,
            max_bucket_size=10,  # Controls the maximum burst size.
        )

        from langchain_anthropic import ChatAnthropic

        model = ChatAnthropic(model_name="claude-sonnet-5", rate_limiter=rate_limiter)

        for _ in range(5):
            tic = time.time()
            model.invoke("hello")
            toc = time.time()
            print(toc - tic)
        ```
    """  # noqa: E501

    def __init__(
        self,
        *,
        requests_per_second: float = 1,
        check_every_n_seconds: float = 0.1,
        max_bucket_size: float = 1,
    ) -> None:
        """基于令牌桶的速率限制器。

        These tokens have nothing to do with LLM tokens. They are just
        a way to keep track of how many requests can be made at a given time.

        此速率限制器设计用于线程环境。

        It works by filling up a bucket with tokens at a given rate. Each
        request consumes a given number of tokens. If there are not enough
        tokens in the bucket, the request is blocked until there are enough
        tokens.

        Args:
            requests_per_second: 每秒向桶中添加的令牌数。
            令牌代表可用于发起请求的“信用”。 check_every_n_seconds:
            每隔此秒数检查一次令牌是否可用。 可以是浮点数，表示秒的分数。
            max_bucket_size: 桶中最多可容纳的令牌数。 必须至少为 `1`。用于防止请求突发。
        """
                # 每秒可发起的请求次数。
        self.requests_per_second = requests_per_second

                # 桶中的令牌数量。
        self.available_tokens = 0.0

        self.max_bucket_size = max_bucket_size

                # 一个锁，确保同一时间只有一个线程可以消耗令牌
                # 在特定时间。
        self._consume_lock = threading.Lock()

                # 上一次尝试消耗令牌的时间。
        self.last: float | None = None

        self.check_every_n_seconds = check_every_n_seconds

    def _consume(self) -> bool:
        """尝试消耗一个令牌。

        Returns:
            True 表示已成功消耗令牌，调用方可以继续发起请求。False
            表示未消耗令牌，调用方应稍后重试。
        """
        with self._consume_lock:
            now = time.monotonic()

                        # 首次调用时初始化，以避免突发
            if self.last is None:
                self.last = now

            elapsed = now - self.last

            if elapsed * self.requests_per_second >= 1:
                self.available_tokens += elapsed * self.requests_per_second
                self.last = now

                        # 确保不会超过桶的大小。
                        # 用于防止请求突发。
            self.available_tokens = min(self.available_tokens, self.max_bucket_size)

                        # 只要至少有一个令牌，就可以继续。
            if self.available_tokens >= 1:
                self.available_tokens -= 1
                return True

            return False

    def acquire(self, *, blocking: bool = True) -> bool:
        """尝试从速率限制器中获取一个令牌。

        This method blocks until the required tokens are available if `blocking`
        is set to `True`.

        If `blocking` is set to `False`, the method will immediately return the result
        of the attempt to acquire the tokens.

        Args:
            blocking: If `True`, the method will block until the tokens are available.
                If `False`, the method will return immediately with the result of
                the attempt.

        Returns:
            如果成功获取令牌则返回 `True`，否则返回 `False`。
        """
        if not blocking:
            return self._consume()

        while not self._consume():
            time.sleep(self.check_every_n_seconds)

        return True

    async def aacquire(self, *, blocking: bool = True) -> bool:
        """尝试从速率限制器中获取一个令牌。异步版本。

        This method blocks until the required tokens are available if `blocking`
        is set to `True`.

        If `blocking` is set to `False`, the method will immediately return the result
        of the attempt to acquire the tokens.

        Args:
            blocking: If `True`, the method will block until the tokens are available.
                If `False`, the method will return immediately with the result of
                the attempt.

        Returns:
            如果成功获取令牌则返回 `True`，否则返回 `False`。
        """
        if not blocking:
            return self._consume()

        while not self._consume():  # noqa: ASYNC110
                        # 此代码忽略 ASYNC110 警告，因为在此情况下是误报。
                        # 由于令牌由速率限制器自身管理，不存在外部实体可标记事件已完成
                        # 没有外部参与者可以标记该事件已完成
                        # 因为 token 由速率限制器自身管理。
                        # 它需要唤醒以重新填充 tokens。
                        # https://docs.astral.sh/ruff/rules/async-busy-wait/
            await asyncio.sleep(self.check_every_n_seconds)
        return True


__all__ = [
    "BaseRateLimiter",
    "InMemoryRateLimiter",
]

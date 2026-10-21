"""速率限制器的基类抽象及内存实现。

这些速率限制器可用于限制对 API 的请求频率。

速率限制器可与 `BaseChatModel` 一起使用。
"""

from langchain_core.rate_limiters import BaseRateLimiter, InMemoryRateLimiter

__all__ = [
    "BaseRateLimiter",
    "InMemoryRateLimiter",
]

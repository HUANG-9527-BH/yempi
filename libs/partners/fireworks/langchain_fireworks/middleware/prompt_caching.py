"""Fireworks 提示词缓存会话亲和性中间件。

Requires `langchain` for the agent middleware framework. It is imported lazily,
and an `ImportError` with install guidance is raised if it is missing.
`ChatFireworks` is provided by this package (`langchain-fireworks`).
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Awaitable, Callable, Iterator
from contextlib import contextmanager
from typing import Literal
from warnings import warn

from langchain_fireworks.chat_models import _PROMPT_CACHE_AFFINITY, ChatFireworks

try:
    from langchain.agents.middleware.types import (
        AgentMiddleware,
        ModelCallResult,
        ModelRequest,
        ModelResponse,
    )
    from langgraph.config import get_config
except ImportError as e:
    msg = (
        "FireworksPromptCachingMiddleware requires 'langchain' to be installed "
        f"(missing module: {e.name}). This middleware is designed for use with "
        "LangChain agents. Install it with: pip install langchain"
    )
    raise ImportError(msg) from e

logger = logging.getLogger(__name__)

_UNSUPPORTED_MODEL_BEHAVIORS = ("ignore", "warn", "raise")


def _get_thread_id(request: ModelRequest) -> str | None:
    """从请求运行时元数据或 Runnable 配置中返回线程 ID。"""
        # 运行时元数据在无异步配置传播的 Python 3.10 环境下仍可用。
        # 旧版运行时可能无法暴露 execution_info。
    execution_info = getattr(request.runtime, "execution_info", None)
    thread_id = getattr(execution_info, "thread_id", None)
    if isinstance(thread_id, str) and thread_id:
        return thread_id

    try:
        config = get_config()
    except RuntimeError:
        return None
    thread_id = (config.get("configurable") or {}).get("thread_id")
    if isinstance(thread_id, str) and thread_id:
        return thread_id
    return None


@contextmanager
def _session_affinity(request: ModelRequest) -> Iterator[None]:
    """将亲和性默认值限定于此调用范围，包括任何回退尝试。"""
    thread_id = _get_thread_id(request)
    affinity = (
        hashlib.sha256(thread_id.encode("utf-8")).hexdigest()
        if thread_id is not None
        else None
    )
    if affinity is None:
        logger.debug("Fireworks session affinity not applied: no thread_id in config")
    token = _PROMPT_CACHE_AFFINITY.set(affinity)
    try:
        yield
    finally:
        _PROMPT_CACHE_AFFINITY.reset(token)


class FireworksPromptCachingMiddleware(AgentMiddleware):
    """从当前活动线程 ID 设置 Fireworks 提示词缓存会话亲和性。

    Fireworks prompt caching is enabled by default. This middleware improves
    cache hit rate by pinning session affinity to a SHA-256 hash of
    `config.configurable.thread_id`, so related requests route to the same
    replica and reuse its warm cache. The hexadecimal hash keeps affinity values
    safe for HTTP headers, including when thread IDs contain Unicode.

    中间件向 `ChatFireworks` 提供一个作用域默认值，用于在调用 API 时设置
    `prompt_cache_key` 和 `extra_headers["x-session-affinity"]`。 显式指定的 `user` 或
    `prompt_cache_key` 请求体字段（包括 `extra_body` 重写）， 或所选模型或请求上的
    `x-session-affinity` 头部具有更高优先级， 包括在回退模型上也是如此。当未配置线程 ID
    时，不添加亲和性。

    Generated affinity stays out of shared request settings, so it is never
    forwarded to another provider. This works with either ordering of this
    middleware and `ModelFallbackMiddleware` for a Fireworks primary model.
    """

    def __init__(
        self,
        *,
        unsupported_model_behavior: Literal["ignore", "warn", "raise"] = "warn",
    ) -> None:
        """初始化中间件。

        Args:
            unsupported_model_behavior: 当请求模型非 `ChatFireworks` 时的行为。
            `"ignore"` 表示忽略亲和性继续执行， `"warn"` 表示发出警告， `"raise"`
            表示抛出 `ValueError`。

        Raises:
            ValueError: 若 `unsupported_model_behavior` 不是 `"ignore"`、
                `"warn"` 或 `"raise"` 之一。
        """
        if unsupported_model_behavior not in _UNSUPPORTED_MODEL_BEHAVIORS:
            msg = (
                "unsupported_model_behavior must be one of "
                f"{_UNSUPPORTED_MODEL_BEHAVIORS}, got {unsupported_model_behavior!r}."
            )
            raise ValueError(msg)
        self.unsupported_model_behavior = unsupported_model_behavior

    def _should_apply_caching(self, request: ModelRequest) -> bool:
        """返回是否应对请求应用会话亲和性。

        Args:
            request: 待检查的模型请求。

        Returns:
            `True` 表示请求模型为 `ChatFireworks`，否则为 `False`。

        Raises:
            ValueError: 若模型不受支持且
                `unsupported_model_behavior` 为 `"raise"`。
        """
        if isinstance(request.model, ChatFireworks):
            return True

        msg = (
            "FireworksPromptCachingMiddleware only supports ChatFireworks, "
            f"not {type(request.model).__name__}."
        )
        if self.unsupported_model_behavior == "raise":
            raise ValueError(msg)
        if self.unsupported_model_behavior == "warn":
            warn(msg, stacklevel=3)
        return False

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelCallResult:
        """在执行处理器时提供默认 Fireworks 亲和性。

        Args:
            request: 输出的模型请求。
            handler: 执行请求的可调用对象。

        Returns:
            `handler` 产生的结果。
        """
        if not self._should_apply_caching(request):
            return handler(request)
        with _session_affinity(request):
            return handler(request)

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelCallResult:
        """在等待处理器时提供默认 Fireworks 亲和性。

        Args:
            request: 输出的模型请求。
            handler: 异步可调用对象，用于执行请求。

        Returns:
            `handler` 产生的结果。
        """
        if not self._should_apply_caching(request):
            return await handler(request)
        with _session_affinity(request):
            return await handler(request)

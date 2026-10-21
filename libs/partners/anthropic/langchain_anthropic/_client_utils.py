"""用于创建 Anthropic API 客户端的辅助工具。

This module allows for the caching of httpx clients to avoid creating new instances
for each instance of ChatAnthropic.

逻辑主要复制自 anthropic._base_client。
"""

from __future__ import annotations

import asyncio
import os
from functools import lru_cache
from typing import Any

import anthropic

_NOT_GIVEN: Any = object()


class _SyncHttpxClientWrapper(anthropic.DefaultHttpxClient):
    """借用于 anthropic._base_client。"""

    def __del__(self) -> None:
        try:
            if self.is_closed:
                return
            self.close()
        except Exception:  # noqa: S110
            pass


class _AsyncHttpxClientWrapper(anthropic.DefaultAsyncHttpxClient):
    """借用于 anthropic._base_client。"""

    def __del__(self) -> None:
        try:
            if self.is_closed:
                return
                        # TODO(someday): 支持非 asyncio 运行时
            asyncio.get_running_loop().create_task(self.aclose())
        except Exception:  # noqa: S110
            pass


@lru_cache
def _get_default_httpx_client(
    *,
    base_url: str | None,
    timeout: Any = _NOT_GIVEN,
    anthropic_proxy: str | None = None,
) -> _SyncHttpxClientWrapper:
    kwargs: dict[str, Any] = {
        "base_url": base_url
        or os.environ.get("ANTHROPIC_BASE_URL")
        or "https://api.anthropic.com",
    }
    if timeout is not _NOT_GIVEN:
        kwargs["timeout"] = timeout
    if anthropic_proxy is not None:
        kwargs["proxy"] = anthropic_proxy
    return _SyncHttpxClientWrapper(**kwargs)


@lru_cache
def _get_default_async_httpx_client(
    *,
    base_url: str | None,
    timeout: Any = _NOT_GIVEN,
    anthropic_proxy: str | None = None,
) -> _AsyncHttpxClientWrapper:
    kwargs: dict[str, Any] = {
        "base_url": base_url
        or os.environ.get("ANTHROPIC_BASE_URL")
        or "https://api.anthropic.com",
    }
    if timeout is not _NOT_GIVEN:
        kwargs["timeout"] = timeout
    if anthropic_proxy is not None:
        kwargs["proxy"] = anthropic_proxy
    return _AsyncHttpxClientWrapper(**kwargs)

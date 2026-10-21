"""解析与已安装的 `openai` SDK 匹配的 `httpx` 库。

`openai>=3` is backed by `httpx2` (an API-identical drop-in for `httpx`),
`openai<3` by `httpx`. Transport/config objects injected into the SDK client
are built from the same library, so this module re-exports whichever one the
installed SDK version defaults to.

Import `httpx` from here **only** for objects handed to the OpenAI SDK client
(e.g. a custom transport, `Limits`, `Proxy`, or `http_client=`). Standalone HTTP
code should import `httpx` directly.

Note: the SDK's `DefaultHttpxClient`/`DefaultAsyncHttpxClient` classes are *not*
在此重新导出——它们在两个 SDK 版本中均可被继承（`openai>=3` 时基于
`httpx2`），因此请直接继承 `openai.DefaultHttpxClient`。` `DefaultHttpx2Client`
名称是工厂函数，而非类，无法被继承。
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import openai


def _sdk_uses_httpx2() -> bool:
    """已安装的 `openai` SDK 是否默认使用 `httpx2`（即 `openai>=3`）。"""
        # 首次整数运行，因此像 `v3.0.0` 这样的标签字符串仍可解析；
        # 无法解析时回退到经典的 `httpx`（始终存在）。
    match = re.match(r"\D*(\d+)", str(getattr(openai, "__version__", "")))
    return match is not None and int(match.group(1)) >= 3


if TYPE_CHECKING:
        # 对 `httpx` 进行类型检查，其 API 被 `httpx2` 所镜像。这能保持 mypy
        # 在 lint 环境中正常工作（该环境安装的是 `httpx`，而非 `httpx2`）
        # 而运行时选择则会选取正确的模块。
    import httpx
elif _sdk_uses_httpx2():
    import httpx2 as httpx  # type: ignore[import-not-found]
else:
    import httpx

__all__ = ["httpx"]

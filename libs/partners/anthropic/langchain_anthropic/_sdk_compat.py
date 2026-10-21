"""适配已安装的 `anthropic` SDK 主版本。

`langchain-anthropic` 支持 `anthropic` 0.x 和
1.x。两者之间有两点差异影响到本包，此处通过读取已安装 SDK
的实际行为而非其版本字符串来解决：

- **Sampling parameters.** `anthropic>=1` dropped `temperature`, `top_p` and
  `top_k` from its message-creation methods, since current models do not use
  them. The API still honors them for older models, so they are relocated to
  `extra_body`, which the SDK merges into the request JSON as-is. The wire
  payload is identical on both majors.

- **Raw responses.** `.with_raw_response` on the async client returns
  `AsyncAPIResponse`, whose `parse()` is a coroutine; `anthropic<1` returned
  `LegacyAPIResponse` with a synchronous `parse()`.

`anthropic>=1` also moved its HTTP layer from `httpx` to
[`httpx2`](https://github.com/pydantic/httpx2), but that does not surface here:
this package hands the SDK only primitives (a `base_url` string, a float
`timeout`, a `proxy` URL string) and subclasses `anthropic.DefaultHttpxClient`,
which already points at the right library, so no `httpx` type crosses the
boundary. Only the unit tests inject `httpx` objects, and they resolve the
matching library themselves in `tests/unit_tests/_httpx_compat.py`.
"""

from __future__ import annotations

import inspect
import logging
import re
from collections.abc import Mapping
from functools import lru_cache
from typing import Any

import anthropic

logger = logging.getLogger(__name__)

# 已安装 SDK 不再接受作为命名参数的采样参数。
_SAMPLING_PARAMS = ("temperature", "top_p", "top_k")


def _sdk_major_version() -> int | None:
    """已安装的 `anthropic` SDK 的主版本，若无法读取则返回 `None`。

    容忍由标签生成的字符串，如
    `v1.0.0`，这些字符串有时会被构建工具遗漏未清理，通过读取第一个整数序列来处理。
    """
    match = re.match(r"\D*(\d+)", str(getattr(anthropic, "__version__", "")))
    return int(match.group(1)) if match else None


@lru_cache(maxsize=1)
def _unsupported_sampling_params() -> frozenset[str]:
    """已安装 SDK 的消息方法不再接受的采样参数。

    Derived from the signature of `Messages.create`, which `beta.messages` and
    the async resources track. No client is instantiated, so this is safe to
    call during payload construction without resolved credentials.
    """
    try:
        from anthropic.resources.messages import Messages

        accepted = inspect.signature(Messages.create).parameters
    except (ImportError, AttributeError, TypeError, ValueError):
                # SDK
        # 移动了资源或签名无法读取。根据主版本猜测而非失败：错误的猜测只会导致首次请求时报错，而此处异常会破坏所有导入。
                # 一个 SDK 本就会拒绝的值。保持负载不变，而不是将参数移入无法容纳它们的结构中。
                # 首次请求时，此处的异常会导致所有导入失败。
        major = _sdk_major_version()
        logger.warning(
            "Could not introspect the anthropic SDK's `Messages.create` "
            "signature; falling back to its major version (%r) to decide "
            "whether `temperature`/`top_p`/`top_k` are still accepted.",
            major,
        )
        if major is not None and major >= 1:
            return frozenset(_SAMPLING_PARAMS)
        return frozenset()

    return frozenset(p for p in _SAMPLING_PARAMS if p not in accepted)


def _route_unsupported_sampling_params(payload: dict[str, Any]) -> dict[str, Any]:
    """将已安装 SDK 拒绝的采样参数移至 `extra_body`。

    在 `anthropic<1` 时为无操作，因为消息方法仍接受这些参数作为命名参数。调用方已在
    `extra_body` 中设置的值优先，因为这是 SDK 文档明确指出的逃逸通道，属于有意设计。

    Args:
        payload: 请求负载，就地修改。

    Returns:
        相同的负载。
    """
    unsupported = _unsupported_sampling_params()
    if not unsupported:
        return payload

    extra_body = payload.get("extra_body")
    if extra_body is not None and not isinstance(extra_body, Mapping):
                # SDK 会拒绝的值。保持负载不变
                # 而不是将参数移入无法容纳它们的结构中。
        return payload

    relocated = {k: payload.pop(k) for k in unsupported if k in payload}
    if not relocated:
        return payload

    payload["extra_body"] = {**relocated, **(extra_body or {})}
    return payload


async def _aparse(raw_response: Any) -> Any:
    """解析来自异步客户端的原始 SDK 响应。

    `anthropic>=1` 返回 `AsyncAPIResponse`，其 `parse()` 是协程；`anthropic<1` 返回
    `LegacyAPIResponse`，其 `parse()` 是同步的。
    """
    parsed = raw_response.parse()
    if inspect.isawaitable(parsed):
        return await parsed
    return parsed

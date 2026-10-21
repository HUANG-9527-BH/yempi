"""SSRF 保护 - 轻量封装，对内部调用者抛出 ValueError。

所有验证操作委托给 `langchain_core._security._policy`。
"""

import os
import socket
from typing import Annotated, Any
from urllib.parse import urlparse

from pydantic import (
    AnyHttpUrl,
    BeforeValidator,
    HttpUrl,
)

from langchain_core._security._exceptions import SSRFBlockedError
from langchain_core._security._policy import (
    SSRFPolicy,
)
from langchain_core._security._policy import (
    validate_resolved_ip as _validate_resolved_ip,
)
from langchain_core._security._policy import (
    validate_url_sync as _validate_url_sync,
)


def _policy_for(*, allow_private: bool, allow_http: bool) -> SSRFPolicy:
    """从旧版标志接口构建 `SSRFPolicy`。"""
    schemes = frozenset({"http", "https"}) if allow_http else frozenset({"https"})
    return SSRFPolicy(
        allowed_schemes=schemes,
        block_private_ips=not allow_private,
        block_localhost=not allow_private,
        block_cloud_metadata=True,
        block_k8s_internal=True,
    )


def validate_safe_url(
    url: str | AnyHttpUrl,
    *,
    allow_private: bool = False,
    allow_http: bool = True,
) -> str:
    """对 URL 进行 SSRF 保护验证。

    此函数通过阻止对私有网络和云元数据端点的请求，防止服务器端请求伪造（SSRF）攻击。

    Args:
        url: 要验证的 URL（字符串或 Pydantic HttpUrl）。
        allow_private: 若为 `True`，允许私有 IP 和 localhost（用于开发）。
                      云元数据端点始终被阻止。
        allow_http: 若为 `True`，允许 HTTP 和 HTTPS。若为 `False`，仅允许 HTTPS。

    Returns:
        验证后的 URL 作为字符串返回。

    Raises:
        ValueError: 若 URL 无效或可能存在危险。
    """
    url_str = str(url)
    parsed = urlparse(url_str)
    hostname = parsed.hostname or ""

        # 测试环境绕过（保留自原始实现）
    if (
        os.environ.get("LANGCHAIN_ENV") == "local_test"
        and hostname.startswith("test")
        and "server" in hostname
    ):
        return url_str

    policy = _policy_for(allow_private=allow_private, allow_http=allow_http)

        # 同步方案 + 主机名检查
    try:
        _validate_url_sync(url_str, policy)
    except SSRFBlockedError as exc:
        raise ValueError(str(exc)) from exc

        # DNS 解析和 IP 验证
    try:
        addr_info = socket.getaddrinfo(
            hostname,
            parsed.port or (443 if parsed.scheme == "https" else 80),
            socket.AF_UNSPEC,
            socket.SOCK_STREAM,
        )

        for result in addr_info:
            ip_str: str = result[4][0]  # type: ignore[assignment]
            try:
                _validate_resolved_ip(ip_str, policy)
            except SSRFBlockedError as exc:
                raise ValueError(str(exc)) from exc

    except socket.gaierror as e:
        msg = f"Failed to resolve hostname '{hostname}': {e}"
        raise ValueError(msg) from e
    except OSError as e:
        msg = f"Network error while validating URL: {e}"
        raise ValueError(msg) from e

    return url_str


def is_safe_url(
    url: str | AnyHttpUrl,
    *,
    allow_private: bool = False,
    allow_http: bool = True,
) -> bool:
    """不抛异常版本的 `validate_safe_url`。"""
    try:
        validate_safe_url(url, allow_private=allow_private, allow_http=allow_http)
    except ValueError:
        return False
    else:
        return True


def _validate_url_ssrf_strict(v: Any) -> Any:
    """以严格模式验证 URL 的 SSRF 保护。"""
    if isinstance(v, str):
        validate_safe_url(v, allow_private=False, allow_http=True)
    return v


def _validate_url_ssrf_https_only(v: Any) -> Any:
    if isinstance(v, str):
        validate_safe_url(v, allow_private=False, allow_http=False)
    return v


def _validate_url_ssrf_relaxed(v: Any) -> Any:
    """以宽松模式验证 URL 的 SSRF 保护（允许私有 IP）。"""
    if isinstance(v, str):
        validate_safe_url(v, allow_private=True, allow_http=True)
    return v


# 使用 SSRF 保护标注类型
SSRFProtectedUrl = Annotated[HttpUrl, BeforeValidator(_validate_url_ssrf_strict)]
SSRFProtectedUrlRelaxed = Annotated[
    HttpUrl, BeforeValidator(_validate_url_ssrf_relaxed)
]
SSRFProtectedHttpsUrl = Annotated[
    HttpUrl, BeforeValidator(_validate_url_ssrf_https_only)
]
SSRFProtectedHttpsUrlStr = Annotated[
    str, BeforeValidator(_validate_url_ssrf_https_only)
]

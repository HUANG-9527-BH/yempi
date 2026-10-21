"""具备 DNS 解析和 IP 固定的 SSRF 安全 httpx 传输层。"""

import asyncio
import socket

import httpx

from langchain_core._security._exceptions import SSRFBlockedError
from langchain_core._security._policy import (
    DEFAULT_SSRF_POLICY,
    SSRFPolicy,
    _effective_allowed_hosts,
    validate_resolved_ip,
    validate_url_sync,
)

# AsyncHTTPTransport 接受的键（从工厂参数转发）。
_TRANSPORT_KWARGS = frozenset(
    {
        "verify",
        "cert",
        "trust_env",
        "http1",
        "http2",
        "limits",
        "retries",
    }
)


class SSRFSafeTransport(httpx.AsyncBaseTransport):
    """验证 DNS 结果是否符合 SSRF 策略的 httpx 异步传输层。

    对每个出站请求，该传输层执行以下操作： 1. 检查 URL 方案是否在
    `policy.allowed_schemes` 允许范围内。 2. 验证主机名是否匹配被阻止的模式。 3. 解析
    DNS 并验证 **所有** 返回的 IP 地址。 4. 重写请求以连接到第一个有效的
    IP，同时保持原始的 `Host` 头和 TLS SNI 主机名。

    Redirects are re-validated on each hop because `follow_redirects`
    is set on the *client*, causing `handle_async_request` to be called
    again for each redirect target.
    """

    def __init__(
        self,
        policy: SSRFPolicy = DEFAULT_SSRF_POLICY,
        **transport_kwargs: object,
    ) -> None:
        self._policy = policy
        self._inner = httpx.AsyncHTTPTransport(**transport_kwargs)  # type: ignore[arg-type]

    # ------------------------------------------------------------------ #
        # 核心请求处理器
    # ------------------------------------------------------------------ #

    async def handle_async_request(
        self,
        request: httpx.Request,
    ) -> httpx.Response:
        hostname = request.url.host or ""
        scheme = request.url.scheme.lower()

                # 1-3. 方案、主机名和模式检查（复用同步验证器）。
        validate_url_sync(str(request.url), self._policy)

                # 允许主机绕过 - 完全跳过 DNS/IP 验证。
        allowed = {h.lower() for h in _effective_allowed_hosts(self._policy)}
        if hostname.lower() in allowed:
            return await self._inner.handle_async_request(request)

                # 4. DNS 解析
        port = request.url.port or (443 if scheme == "https" else 80)
        try:
            addrinfo = await asyncio.to_thread(
                socket.getaddrinfo,
                hostname,
                port,
                type=socket.SOCK_STREAM,
            )
        except socket.gaierror as exc:
            msg = "DNS resolution failed"
            raise SSRFBlockedError(msg) from exc

        if not addrinfo:
            msg = "DNS resolution returned no results"
            raise SSRFBlockedError(msg)

                # 5. 验证所有解析出的 IP - 任意一个被阻止即拒绝。
        for _family, _type, _proto, _canonname, sockaddr in addrinfo:
            ip_str: str = sockaddr[0]  # type: ignore[assignment]
            validate_resolved_ip(ip_str, self._policy)

                # 6. 固定到首个解析出的 IP。
        pinned_ip = addrinfo[0][4][0]

                # 7. 重写 URL 以使用固定 IP，同时保留 Host 头和 SNI。
        pinned_url = request.url.copy_with(host=pinned_ip)

                # 构建扩展字典，添加 sni_hostname 用于 HTTPS，以便 TLS
                # 证书验证使用原始主机名。
        extensions = dict(request.extensions)
        if scheme == "https":
            extensions["sni_hostname"] = hostname.encode("ascii")

        pinned_request = httpx.Request(
            method=request.method,
            url=pinned_url,
            headers=request.headers,  # Host 头已设置为原始值
            content=request.content,
            extensions=extensions,
        )

        return await self._inner.handle_async_request(pinned_request)

    # ------------------------------------------------------------------ #
        # 生命周期
    # ------------------------------------------------------------------ #

    async def aclose(self) -> None:
        await self._inner.aclose()


# ---------------------------------------------------------------------- #
# 工厂
# ---------------------------------------------------------------------- #


class SSRFSafeSyncTransport(httpx.BaseTransport):
    """验证 DNS 结果是否符合 SSRF 策略的 httpx 同步传输层。

    `SSRFSafeTransport` 的同步对应版本。请参阅该类获取完整文档。
    """

    def __init__(
        self,
        policy: SSRFPolicy = DEFAULT_SSRF_POLICY,
        **transport_kwargs: object,
    ) -> None:
        self._policy = policy
        self._inner = httpx.HTTPTransport(**transport_kwargs)  # type: ignore[arg-type]

    def handle_request(
        self,
        request: httpx.Request,
    ) -> httpx.Response:
        hostname = request.url.host or ""
        scheme = request.url.scheme.lower()

        validate_url_sync(str(request.url), self._policy)

        allowed = {h.lower() for h in _effective_allowed_hosts(self._policy)}
        if hostname.lower() in allowed:
            return self._inner.handle_request(request)

        port = request.url.port or (443 if scheme == "https" else 80)
        try:
            addrinfo = socket.getaddrinfo(
                hostname,
                port,
                type=socket.SOCK_STREAM,
            )
        except socket.gaierror as exc:
            msg = "DNS resolution failed"
            raise SSRFBlockedError(msg) from exc

        if not addrinfo:
            msg = "DNS resolution returned no results"
            raise SSRFBlockedError(msg)

        for _family, _type, _proto, _canonname, sockaddr in addrinfo:
            ip_str: str = sockaddr[0]  # type: ignore[assignment]
            validate_resolved_ip(ip_str, self._policy)

        pinned_ip = addrinfo[0][4][0]
        pinned_url = request.url.copy_with(host=pinned_ip)

        extensions = dict(request.extensions)
        if scheme == "https":
            extensions["sni_hostname"] = hostname.encode("ascii")

        pinned_request = httpx.Request(
            method=request.method,
            url=pinned_url,
            headers=request.headers,
            content=request.content,
            extensions=extensions,
        )

        return self._inner.handle_request(pinned_request)

    def close(self) -> None:
        self._inner.close()


# ---------------------------------------------------------------------- #
# 工厂
# ---------------------------------------------------------------------- #


def ssrf_safe_client(
    policy: SSRFPolicy = DEFAULT_SSRF_POLICY,
    **kwargs: object,
) -> httpx.Client:
    """创建一个带有 SSRF 保护的 `httpx.Client`。"""
    transport_kwargs: dict[str, object] = {}
    client_kwargs: dict[str, object] = {}
    for key, value in kwargs.items():
        if key in _TRANSPORT_KWARGS:
            transport_kwargs[key] = value
        else:
            client_kwargs[key] = value

    transport = SSRFSafeSyncTransport(policy=policy, **transport_kwargs)

    client_kwargs.setdefault("follow_redirects", True)
    client_kwargs.setdefault("max_redirects", 10)

    return httpx.Client(
        transport=transport,
        **client_kwargs,  # type: ignore[arg-type]
    )


def ssrf_safe_async_client(
    policy: SSRFPolicy = DEFAULT_SSRF_POLICY,
    **kwargs: object,
) -> httpx.AsyncClient:
    """创建一个带有 SSRF 保护的 `httpx.AsyncClient`。

    Drop-in replacement for `httpx.AsyncClient(...)` - callers just swap
    the constructor call.  Transport-specific kwargs (`verify`, `cert`,
    `retries`, etc.) are forwarded to the inner `AsyncHTTPTransport`;
    everything else goes to the `AsyncClient`.
    """
    transport_kwargs: dict[str, object] = {}
    client_kwargs: dict[str, object] = {}
    for key, value in kwargs.items():
        if key in _TRANSPORT_KWARGS:
            transport_kwargs[key] = value
        else:
            client_kwargs[key] = value

    transport = SSRFSafeTransport(policy=policy, **transport_kwargs)

        # 仅在未被调用者覆盖时应用默认值。
    client_kwargs.setdefault("follow_redirects", True)
    client_kwargs.setdefault("max_redirects", 10)

    return httpx.AsyncClient(
        transport=transport,
        **client_kwargs,  # type: ignore[arg-type]
    )

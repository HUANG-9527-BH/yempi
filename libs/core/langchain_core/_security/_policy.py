"""带有 IP 验证和 DNS 感知 URL 检查的 SSRF 保护策略。"""

import asyncio
import dataclasses
import ipaddress
import os
import socket
import urllib.parse

from langchain_core._security._exceptions import SSRFBlockedError

# ---------------------------------------------------------------------------
# 阻断列表常量
# ---------------------------------------------------------------------------

_BLOCKED_IPV4_NETWORKS: tuple[ipaddress.IPv4Network, ...] = tuple(
    ipaddress.IPv4Network(n)
    for n in (
        "10.0.0.0/8",  # RFC 1918 - 私有类 A
        "172.16.0.0/12",  # RFC 1918 - 私有类 B
        "192.168.0.0/16",  # RFC 1918 - 私有类 C
        "127.0.0.0/8",  # RFC 1122 - 回环
        "169.254.0.0/16",  # RFC 3927 - 链路本地
        "0.0.0.0/8",  # RFC 1122 - "此网络"
        "100.64.0.0/10",  # RFC 6598 - 共享/运营商级网关地址空间
        "192.0.0.0/24",  # RFC 6890 - IETF 协议分配
        "192.0.2.0/24",  # RFC 5737 - TEST-NET-1（文档）
        "198.18.0.0/15",  # RFC 2544 - 基准测试
        "198.51.100.0/24",  # RFC 5737 - TEST-NET-2（文档）
        "203.0.113.0/24",  # RFC 5737 - TEST-NET-3（文档）
        "224.0.0.0/4",  # RFC 5771 - 组播
        "240.0.0.0/4",  # RFC 1112 - 保留供将来使用
        "255.255.255.255/32",  # RFC 919 - 有限广播
    )
)

_BLOCKED_IPV6_NETWORKS: tuple[ipaddress.IPv6Network, ...] = tuple(
    ipaddress.IPv6Network(n)
    for n in (
        "::1/128",  # RFC 4291 - 回环
        "fc00::/7",  # RFC 4193 - 唯一本地地址（ULA）
        "fe80::/10",  # RFC 4291 - 链路本地
        "ff00::/8",  # RFC 4291 - 组播
        "::ffff:0:0/96",  # RFC 4291 - IPv4 映射的 IPv6 地址
        "::0.0.0.0/96",  # RFC 4291 - IPv4 兼容的 IPv6（已弃用）
        "64:ff9b::/96",  # RFC 6052 - NAT64 专用前缀
        "64:ff9b:1::/48",  # RFC 8215 - NAT64 发现前缀
    )
)

_CLOUD_METADATA_IPS: frozenset[str] = frozenset(
    {
        "169.254.169.254",  # AWS、GCP、Azure、DigitalOcean、Oracle Cloud
        "169.254.170.2",  # AWS ECS 任务元数据
        "169.254.170.23",  # AWS EKS Pod Identity Agent
        "100.100.100.200",  # 阿里云元数据
        "fd00:ec2::254",  # AWS EC2 IMDSv2 通过 IPv6（Nitro 实例）
        "fd00:ec2::23",  # AWS EKS Pod Identity Agent（IPv6）
        "fe80::a9fe:a9fe",  # OpenStack Nova 元数据（IPv6 链路本地）
    }
)

# 当 block_cloud_metadata=True 时始终被阻止的网络范围，
# 与 block_private_ips 无关。整个链路本地范围被各云服务提供商用于元数据服务。
# NAT64 专用前缀
_CLOUD_METADATA_NETWORKS: tuple[ipaddress.IPv4Network | ipaddress.IPv6Network, ...] = (
    ipaddress.IPv4Network("169.254.0.0/16"),
)

_CLOUD_METADATA_HOSTNAMES: frozenset[str] = frozenset(
    {
        "metadata.google.internal",
        "metadata.amazonaws.com",
        "metadata",
        "instance-data",
    }
)

_LOCALHOST_NAMES: frozenset[str] = frozenset(
    {
        "localhost",
        "localhost.localdomain",
        "host.docker.internal",
    }
)

_K8S_SUFFIX = ".svc.cluster.local"

_LOOPBACK_IPV4 = ipaddress.IPv4Network("127.0.0.0/8")
_LOOPBACK_IPV6 = ipaddress.IPv6Address("::1")
_IPV4_COMPATIBLE_PREFIX = ipaddress.IPv6Network("::/96")

# NAT64 专用前缀
_NAT64_PREFIX = ipaddress.IPv6Network("64:ff9b::/96")
_NAT64_DISCOVERY_PREFIX = ipaddress.IPv6Network("64:ff9b:1::/48")


# ---------------------------------------------------------------------------
# SSRFPolicy
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class SSRFPolicy:
    """不可变策略，控制哪些 URL/IP 被视为安全。"""

    allowed_schemes: frozenset[str] = frozenset({"http", "https"})
    block_private_ips: bool = True
    block_localhost: bool = True
    block_cloud_metadata: bool = True
    block_k8s_internal: bool = True
    allowed_hosts: frozenset[str] = frozenset()
    additional_blocked_cidrs: tuple[
        ipaddress.IPv4Network | ipaddress.IPv6Network, ...
    ] = ()


DEFAULT_SSRF_POLICY = SSRFPolicy()


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------


def _extract_embedded_ipv4(
    addr: ipaddress.IPv6Address,
) -> ipaddress.IPv4Address | None:
    """从映射、兼容或 NAT64 的 IPv6 地址中提取 IPv4。"""
        # 首先检查 ipv4_mapped（覆盖 ::ffff:x.x.x.x）
    if addr.ipv4_mapped is not None:
        return addr.ipv4_mapped

        # 检查 NAT64 前缀——嵌入的 IPv4 在最后 4 字节中
    if addr in _NAT64_PREFIX or addr in _NAT64_DISCOVERY_PREFIX:
        raw = addr.packed
        return ipaddress.IPv4Address(raw[-4:])

    if addr in _IPV4_COMPATIBLE_PREFIX and int(addr) > 1:
        return ipaddress.IPv4Address(int(addr))

    return None


def _ip_in_blocked_networks(
    addr: ipaddress.IPv4Address | ipaddress.IPv6Address,
    policy: SSRFPolicy,
) -> str | None:
    """如果 *addr* 属于被阻止范围，则返回原因字符串，否则返回 None。"""
        # NOTE: 如果性能分析显示这是热点路径，可考虑使用
        # @functools.lru_cache（键为 (addr, id(policy))）进行缓存。
    if isinstance(addr, ipaddress.IPv4Address):
        if policy.block_private_ips:
            for blocked_ipv4_net in _BLOCKED_IPV4_NETWORKS:
                if addr in blocked_ipv4_net:
                    return "private IP range"
        for blocked_cidr in policy.additional_blocked_cidrs:
            if isinstance(blocked_cidr, ipaddress.IPv4Network) and addr in blocked_cidr:
                return "blocked CIDR"
    else:
        if policy.block_private_ips:
            for blocked_ipv6_net in _BLOCKED_IPV6_NETWORKS:
                if addr in blocked_ipv6_net:
                    return "private IP range"
        for blocked_cidr in policy.additional_blocked_cidrs:
            if isinstance(blocked_cidr, ipaddress.IPv6Network) and addr in blocked_cidr:
                return "blocked CIDR"

        # 回环检查——独立于 block_private_ips，因此即使允许私有 IP，
        # block_localhost=True 仍能捕获 127.x.x.x / ::1。
        # 云元数据检查——IP 集合 *和* 网络范围（如 169.254.0.0/16）。
    if policy.block_localhost:
        if isinstance(addr, ipaddress.IPv4Address) and (
            addr in _LOOPBACK_IPV4 or addr in ipaddress.IPv4Network("0.0.0.0/8")
        ):
            return "localhost address"
        if isinstance(addr, ipaddress.IPv6Address) and addr == _LOOPBACK_IPV6:
            return "localhost address"

        # 独立于 block_private_ips，因此即使 allow_private=True 仍会阻止
        # 云元数据端点。
        # 公共验证函数
    if policy.block_cloud_metadata:
        if str(addr) in _CLOUD_METADATA_IPS:
            return "cloud metadata endpoint"
        for net in _CLOUD_METADATA_NETWORKS:
            if addr in net:
                return "cloud metadata endpoint"

    return None


# ---------------------------------------------------------------------------
# 公开的验证函数
# ---------------------------------------------------------------------------


def validate_resolved_ip(ip_str: str, policy: SSRFPolicy) -> None:
    """根据 SSRF 策略验证解析后的 IP 地址。

    如果 IP 被阻止，则抛出 SSRFBlockedError。
    """
    try:
        addr = ipaddress.ip_address(ip_str)
    except ValueError as exc:
        msg = "invalid IP address"
        raise SSRFBlockedError(msg) from exc

    if isinstance(addr, ipaddress.IPv6Address):
        addr = ipaddress.IPv6Address(int(addr))
        inner = _extract_embedded_ipv4(addr)
        if inner is not None:
            addr = inner

    reason = _ip_in_blocked_networks(addr, policy)
    if reason is not None:
        raise SSRFBlockedError(reason)


def validate_hostname(hostname: str, policy: SSRFPolicy) -> None:
    """根据 SSRF 策略验证主机名。

    如果主机名被阻止，则抛出 SSRFBlockedError。
    """
    lower = hostname.lower()

    if policy.block_localhost and lower in _LOCALHOST_NAMES:
        msg = "localhost address"
        raise SSRFBlockedError(msg)

    if policy.block_cloud_metadata and lower in _CLOUD_METADATA_HOSTNAMES:
        msg = "cloud metadata endpoint"
        raise SSRFBlockedError(msg)

    if policy.block_k8s_internal and lower.endswith(_K8S_SUFFIX):
        msg = "Kubernetes internal DNS"
        raise SSRFBlockedError(msg)


def _effective_allowed_hosts(policy: SSRFPolicy) -> frozenset[str]:
    """返回允许的主机列表，并为本地环境进行增强。"""
    extra: set[str] = set()
    if os.environ.get("LANGCHAIN_ENV", "").startswith("local"):
        extra.update({"localhost", "testserver"})
    if extra:
        return policy.allowed_hosts | frozenset(extra)
    return policy.allowed_hosts


async def validate_url(url: str, policy: SSRFPolicy = DEFAULT_SSRF_POLICY) -> None:
    """根据 SSRF 策略验证 URL，包括 DNS 解析。

    This is the primary entry-point for async code paths. It delegates
    scheme/hostname/allowed-hosts checks to `validate_url_sync`, then
    resolves DNS and validates every resolved IP.

    Raises:
        SSRFBlockedError: 如果 URL 违反策略。
    """
    parsed = urllib.parse.urlparse(url)
    hostname = parsed.hostname or ""

    validate_url_sync(url, policy)

    allowed = {h.lower() for h in _effective_allowed_hosts(policy)}
    if hostname.lower() in allowed:
        return

    scheme = (parsed.scheme or "").lower()
    port = parsed.port or (443 if scheme == "https" else 80)
    try:
        addrinfo = await asyncio.to_thread(
            socket.getaddrinfo, hostname, port, type=socket.SOCK_STREAM
        )
    except socket.gaierror as exc:
        msg = "DNS resolution failed"
        raise SSRFBlockedError(msg) from exc

    for _family, _type, _proto, _canonname, sockaddr in addrinfo:
        validate_resolved_ip(str(sockaddr[0]), policy)


def validate_url_sync(url: str, policy: SSRFPolicy = DEFAULT_SSRF_POLICY) -> None:
    """同步 URL 验证（无 DNS 解析）。

    Suitable for Pydantic validators and other sync contexts. Checks scheme
    and hostname patterns only - use `validate_url` for full DNS-aware checking.

    Raises:
        SSRFBlockedError: 如果 URL 违反策略。
    """
    parsed = urllib.parse.urlparse(url)

    scheme = (parsed.scheme or "").lower()
    if scheme not in policy.allowed_schemes:
        msg = f"scheme '{scheme}' not allowed"
        raise SSRFBlockedError(msg)

    hostname = parsed.hostname
    if not hostname:
        msg = "missing hostname"
        raise SSRFBlockedError(msg)

    allowed = _effective_allowed_hosts(policy)
    if hostname.lower() in {h.lower() for h in allowed}:
        return

    try:
        ipaddress.ip_address(hostname)
        validate_resolved_ip(hostname, policy)
    except SSRFBlockedError:
        raise
    except ValueError:
        pass
    else:
        return

    validate_hostname(hostname, policy)

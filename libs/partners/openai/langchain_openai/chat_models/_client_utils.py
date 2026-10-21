"""用于 OpenAI httpx 客户端构建、传输调优和流式处理的辅助函数。

Covers cached default client builders, proxy-aware variants for the
`openai_proxy` path, kernel-level TCP keepalive / `TCP_USER_TIMEOUT` socket
options, and the `_astream_with_chunk_timeout` wrapper that bounds per-chunk
wall-clock time on async SSE streams.

客户端构建器样板代码模仿了 `openai._base_client`
中的模式；套接字选项调优和流式超时为本模块原创。
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import os
import socket
import sys
import urllib.request
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from functools import lru_cache
from typing import Any, TypeVar, cast

import openai
from pydantic import SecretStr

from langchain_openai._compat import httpx

logger = logging.getLogger(__name__)

SocketOption = tuple[int, int, int]

# socket.TCP_KEEPIDLE 等常量在 darwin/win32 上不存在；使用原始 UAPI 常量。
_LINUX_TCP_KEEPIDLE = 4
_LINUX_TCP_KEEPINTVL = 5
_LINUX_TCP_KEEPCNT = 6
_LINUX_TCP_USER_TIMEOUT = 18

# macOS：语义相同，但来自 <netinet/tcp.h> 的常量不同。
_DARWIN_TCP_KEEPALIVE = 0x10  # 首次探测前的空闲秒数
_DARWIN_TCP_KEEPINTVL = 0x101
_DARWIN_TCP_KEEPCNT = 0x102

# 与 openai SDK 的池默认值一致。硬编码以避免依赖于 可能在 SDK
# 版本间移动的内部模块路径（openai._constants）。
# 广义捕获是故意的：`pytest_socket` 在 `--disable-socket` 下
_DEFAULT_CONNECTION_LIMITS = httpx.Limits(
    max_connections=1000,
    max_keepalive_connections=100,
    keepalive_expiry=5.0,
)


def _int_env(name: str, default: int, *, allow_negative: bool = False) -> int:
    """以优雅降级方式读取整数环境变量，并提供可发现的警告。

    无法解析或（默认情况下）负值将回退到 `default` 并发出一条
    `WARNING`，指出有问题的变量。配置错误的环境仍可加载，但操作员可在日志中看到回退值，而非静默地获得意外的默认值。
    """
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except (TypeError, ValueError):
        logger.warning(
            "Invalid value for %s=%r (not an int); falling back to %d.",
            name,
            raw,
            default,
        )
        return default
    if not allow_negative and value < 0:
        logger.warning(
            "Invalid value for %s=%r (negative); falling back to %d.",
            name,
            raw,
            default,
        )
        return default
    return value


def _float_env(name: str, default: float, *, allow_negative: bool = False) -> float:
    """以优雅降级方式读取浮点数环境变量，并提供可发现的警告。

    See `_int_env`. Negative values are rejected by default so a typo in
    `LANGCHAIN_OPENAI_STREAM_CHUNK_TIMEOUT_S=-10` can't silently disable the
    wrapper it was meant to configure.
    """
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = float(raw)
    except (TypeError, ValueError):
        logger.warning(
            "Invalid value for %s=%r (not a float); falling back to %s.",
            name,
            raw,
            default,
        )
        return default
    if not allow_negative and value < 0:
        logger.warning(
            "Invalid value for %s=%r (negative); falling back to %s.",
            name,
            raw,
            default,
        )
        return default
    return value


def _filter_supported(opts: list[SocketOption]) -> list[SocketOption]:
    """丢弃运行平台不支持的套接字选项。

    通过 `setsockopt`
    对临时套接字逐一探测每个选项，并仅保留内核接受的选项。这确保了库计算的默认值在不实现所有
    Linux 选项的平台上不会导致致命错误——特别是 `TCP_USER_TIMEOUT` 仅限于 Linux，在
    macOS、某些最小内核和旧版 gVisor 构建中会静默缺失。被丢弃的选项会在 `DEBUG`
    级别记录，以便操作员确认其平台上的内核级开关是否生效。

    如果无法创建探测套接字（沙箱运行时、`pytest-socket` 在 `--disable-socket`
    下、严格的 seccomp
    策略），则返回未过滤的输入列表。这保留了显式用户覆盖所用的透传行为：不支持的选项将在首次真实
    `connect()` 时作为清晰的 `OSError` 显现，而非在 `ChatOpenAI` 构造期间静默丢弃。
    """
    try:
        probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    except Exception:
                # 会抛出 `SocketBlockedError`（一个 `RuntimeError`，而非 `OSError`），且
                # seccomp/沙箱运行时曾观察到抛出其他
                # `OSError` 子类和 `PermissionError`。意图是「任何
                # `OSError` 子类和 `PermissionError`。意图是 "任何"
                # 无法创建探测套接字 -> 直接透传未过滤的连接,"
                # 缩小类型会无声地导致沙箱 CI 出现回归问题。
        return list(opts)
    try:
        supported: list[SocketOption] = []
        dropped: list[SocketOption] = []
        for level, optname, optval in opts:
            try:
                probe.setsockopt(level, optname, optval)
            except OSError:
                dropped.append((level, optname, optval))
                continue
            supported.append((level, optname, optval))
        if dropped:
            logger.debug(
                "Dropped %d unsupported socket option(s) on %s: %s",
                len(dropped),
                sys.platform,
                dropped,
            )
        return supported
    finally:
        probe.close()


def _default_socket_options() -> tuple[SocketOption, ...]:
    """返回默认的 TCP 套接字选项，或在环境变量禁用时返回 `()`。

    Always returns a tuple (never None) so callers and `@lru_cache` keys
    remain uniform: `()` is the single shape for "no options".

    Target behavior on Linux/gVisor with the full option set: silent peers
    are surfaced within ~90-120s via `SO_KEEPALIVE` + `TCP_USER_TIMEOUT`
    (keepalive path gives a ~90s floor at the defaults; `TCP_USER_TIMEOUT`
    caps at 120s). On platforms that reject some options,
    `_filter_supported` drops them and the bound degrades to whatever the
    remaining options provide.
    """
    if os.environ.get("LANGCHAIN_OPENAI_TCP_KEEPALIVE", "1") == "0":
        return ()

    keepidle = _int_env("LANGCHAIN_OPENAI_TCP_KEEPIDLE", 60)
    keepintvl = _int_env("LANGCHAIN_OPENAI_TCP_KEEPINTVL", 10)
    keepcnt = _int_env("LANGCHAIN_OPENAI_TCP_KEEPCNT", 3)
    user_timeout_ms = _int_env("LANGCHAIN_OPENAI_TCP_USER_TIMEOUT_MS", 120000)

    opts: list[SocketOption] = [(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)]
    if sys.platform == "linux":
        opts += [
            (socket.IPPROTO_TCP, _LINUX_TCP_KEEPIDLE, keepidle),
            (socket.IPPROTO_TCP, _LINUX_TCP_KEEPINTVL, keepintvl),
            (socket.IPPROTO_TCP, _LINUX_TCP_KEEPCNT, keepcnt),
            (socket.IPPROTO_TCP, _LINUX_TCP_USER_TIMEOUT, user_timeout_ms),
        ]
    elif sys.platform == "darwin":
        opts += [
            (socket.IPPROTO_TCP, _DARWIN_TCP_KEEPALIVE, keepidle),
            (socket.IPPROTO_TCP, _DARWIN_TCP_KEEPINTVL, keepintvl),
            (socket.IPPROTO_TCP, _DARWIN_TCP_KEEPCNT, keepcnt),
        ]
        # Windows (win32)：仅支持 SO_KEEPALIVE；每选项调优需要使用 WSAIoctl。
    return tuple(_filter_supported(opts))


_PROXY_ENV_VARS = (
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
)
_proxy_env_warning_emitted = False
_proxy_env_bypass_info_emitted = False


def _proxy_env_detected() -> bool:
    """当 httpx 会从环境或系统配置中获取代理时为 True。

    与 httpx 表面读取的逻辑一致（`urllib.request.getproxies()`
    加上大写环境变量名），因此正值表示在预 PR 代码路径中环境代理自动检测处于激活状态。
    """
    if any(os.environ.get(name) for name in _PROXY_ENV_VARS):
        return True
    try:
        return bool(urllib.request.getproxies())
    except Exception:
        return False


def _should_bypass_socket_options_for_proxy_env(
    *,
    http_socket_options: Sequence[SocketOption] | None,
    http_client: Any,
    http_async_client: Any,
    openai_proxy: str | None,
) -> bool:
    """当默认形状 + 环境代理被检测到时为 True → 跳过传输注入。

    Preserves pre-PR behavior for apps relying on httpx's env-proxy
    auto-detection. Only triggers when the user has made no explicit choice
    that would signal they want the custom transport:

    - `http_socket_options` 保持为 `None`（默认值，而非 `()` 或序列）
    - `LANGCHAIN_OPENAI_TCP_KEEPALIVE` 不为 `0`（关闭开关是独立路径）
    - 未提供 `http_client` 或 `http_async_client`
    - 未提供 `openai_proxy`
    - httpx 可见代理环境变量 / 系统代理

    如果上述任一条件被设置，用户已选择进入该传输路径（直接或通过
    `openai_proxy`），则应用正常行为——包括被屏蔽的代理警告。当关闭开关设置时，`_default_socket_options`
    已返回 `()`，因此绕过信息将成噪音；应改走正常路径。
    """
    if http_socket_options is not None:
        return False
    if os.environ.get("LANGCHAIN_OPENAI_TCP_KEEPALIVE", "1") == "0":
        return False
    if http_client is not None or http_async_client is not None:
        return False
    if openai_proxy:
        return False
    return _proxy_env_detected()


def _log_proxy_env_bypass_once() -> None:
    """当代理环境绕过触发时，发出一次性的 INFO 日志。

    Visibility for operators running with a custom log pipeline: the bypass
    is the *safe* outcome (env-proxy auto-detection preserved), but it means
    socket-level keepalive / `TCP_USER_TIMEOUT` aren't applied on this
    instance. INFO-level, since it's not a problem — just a diagnostic.
    """
    global _proxy_env_bypass_info_emitted
    if _proxy_env_bypass_info_emitted:
        return
    _proxy_env_bypass_info_emitted = True
    active = [name for name in _PROXY_ENV_VARS if os.environ.get(name)]
    source = ", ".join(active) if active else "system proxy configuration"
    logger.info(
        "langchain-openai detected %s and no explicit `http_socket_options` / "
        "`http_client` / `http_async_client` / `openai_proxy`; skipping the "
        "custom `httpx` transport so httpx's env-proxy auto-detection applies. "
        "Pass `http_socket_options=[...]` to opt back into kernel-level TCP "
        "keepalive tuning on top of the env proxy.",
        source,
    )


def _warn_if_proxy_env_shadowed(
    socket_options: tuple[SocketOption, ...],
    *,
    openai_proxy: str | None,
) -> None:
    """若自定义传输将屏蔽 httpx 的代理自动检测，仅发出一次警告。

    当 `socket_options` 非空时，我们传递自定义的 `httpx` 传输，这会禁用 httpx
    的原生代理自动检测——包括大写的 `HTTP_PROXY` / `HTTPS_PROXY` / `ALL_PROXY`
    环境变量及其小写等价形式，以及 macOS/Windows 系统代理配置。如果用户显式提供
    `openai_proxy`，则通过它路由，环境变量处理便不再相关。否则，原本透明依赖这些来源的用户在升级后将静默停止使用它们——发出单次警告，使行为变化可被发现。

    检测使用 `urllib.request.getproxies()` —— 与 httpx
    读取的表面一致——因此小写环境变量和 macOS/Windows 系统代理设置均会被捕获。
    """
    global _proxy_env_warning_emitted
    if _proxy_env_warning_emitted or not socket_options or openai_proxy:
        return
    active = [name for name in _PROXY_ENV_VARS if os.environ.get(name)]
    try:
        detected = bool(urllib.request.getproxies())
    except Exception:
        detected = False
    if not active and not detected:
        return
    _proxy_env_warning_emitted = True
    if active:
        source = ", ".join(active) + " set in environment"
    else:
        source = "system proxy configuration detected"
    logger.warning(
        "langchain-openai injected a custom httpx transport to apply "
        "`http_socket_options`, which disables httpx's proxy "
        "auto-detection (%s). Set "
        "`LANGCHAIN_OPENAI_TCP_KEEPALIVE=0` or pass `http_socket_options=()` "
        "to restore default proxy behavior, or supply `openai_proxy` / your "
        "own `http_client` / `http_async_client` to take full control.",
        source,
    )


def _resolve_socket_options(
    value: Sequence[SocketOption] | None,
) -> tuple[SocketOption, ...]:
    """将面向用户的字段规范化为构建器期望的元组形式。

    - `None` => env-driven defaults (may itself be `()` if the user set
        `LANGCHAIN_OPENAI_TCP_KEEPALIVE=0`). This path runs through
        `_filter_supported()` inside `_default_socket_options()` because
        the library-computed option set is aspirational and silent degradation
        is the right posture.
    - Any other sequence (including empty) => retupled for cache hashability.
        An empty tuple is the explicit "disabled" signal. A non-empty sequence
        is passed verbatim — **not** filtered. The user chose these options
        explicitly, so an unsupported constant should surface as a clear
        `OSError` at connect time, not be silently dropped.

    Always returns a tuple — never `None` — so downstream signatures take
    `tuple[SocketOption, ...]` with `()` as the single "no options" shape.
    """
    if value is None:
        return _default_socket_options()
    return tuple(value)


class _SyncHttpxClientWrapper(openai.DefaultHttpxClient):
    """来自 openai._base_client。"""

    def __del__(self) -> None:
        try:
            if self.is_closed:
                return
            self.close()
        except Exception:  # noqa: S110
            pass


class _AsyncHttpxClientWrapper(openai.DefaultAsyncHttpxClient):
    """源自 openai._base_client。"""

    def __del__(self) -> None:
        try:
            if self.is_closed:
                return
                        # TODO(someday): 在此处支持非 asyncio 运行时
            asyncio.get_running_loop().create_task(self.aclose())
        except Exception:  # noqa: S110
            pass


def _build_sync_httpx_client(
    base_url: str | None,
    timeout: Any,
    socket_options: tuple[SocketOption, ...] = (),
) -> _SyncHttpxClientWrapper:
    kwargs: dict[str, Any] = {
        "base_url": base_url
        or os.environ.get("OPENAI_BASE_URL")
        or "https://api.openai.com/v1",
        "timeout": timeout,
    }
    if socket_options:
                # httpx 在提供 transport= 时会忽略 limits=；需显式设置
                # 在 transport 上以避免连接池被无声缩减。
        kwargs["transport"] = httpx.HTTPTransport(
            socket_options=list(socket_options),
            limits=_DEFAULT_CONNECTION_LIMITS,
        )
    return _SyncHttpxClientWrapper(**kwargs)


def _build_async_httpx_client(
    base_url: str | None,
    timeout: Any,
    socket_options: tuple[SocketOption, ...] = (),
) -> _AsyncHttpxClientWrapper:
    kwargs: dict[str, Any] = {
        "base_url": base_url
        or os.environ.get("OPENAI_BASE_URL")
        or "https://api.openai.com/v1",
        "timeout": timeout,
    }
    if socket_options:
                # 参见 _build_sync_httpx_client 了解 limits= 的设计原因。
        kwargs["transport"] = httpx.AsyncHTTPTransport(
            socket_options=list(socket_options),
            limits=_DEFAULT_CONNECTION_LIMITS,
        )
    return _AsyncHttpxClientWrapper(**kwargs)


def _build_proxied_sync_httpx_client(
    proxy: str,
    verify: Any,
    socket_options: tuple[SocketOption, ...] = (),
) -> httpx.Client:
    """openai_proxy 代码路径的 httpx.Client。

    当禁用套接字选项（`()`）时，返回一个普通的
    `httpx.Client(proxy=..., verify=...)`，不注入传输层。
    """
    if not socket_options:
        return httpx.Client(proxy=proxy, verify=verify)
        # 挂载于 `all://`（而非 `transport=`），以便 `Client._mounts` 与
        # httpx 自身 `proxy=` 路径生成的结构保持一致 —— 一个以
        # `URLPattern("all://")` 为键的单条目字典。调用方（以及现有的代理集成
        # 测试）会深入 `_mounts` 以检查代理 URL；仅使用裸的
        # `transport=` 会导致 `_mounts` 为空。
    #
        # `httpx.HTTPTransport(proxy=...)` 对字符串强制转换的要求比
        # `httpx.Client(proxy=...)` 更严格；为保证版本稳定行为，应使用公开的 `httpx.Proxy`
    # 类型进行包装。
        # 在 Python ≤3.10 时，asyncio.TimeoutError 和 builtins.TimeoutError 是不同的
    transport = httpx.HTTPTransport(
        proxy=httpx.Proxy(proxy),
        verify=verify,
        socket_options=list(socket_options),
        limits=_DEFAULT_CONNECTION_LIMITS,
    )
    return httpx.Client(mounts={"all://": transport})


def _build_proxied_async_httpx_client(
    proxy: str,
    verify: Any,
    socket_options: tuple[SocketOption, ...] = (),
) -> httpx.AsyncClient:
    """openai_proxy 代码路径的 httpx.AsyncClient。

    See `_build_proxied_sync_httpx_client` for the opt-out fallback,
    the `mounts={"all://": ...}` shape, and the `httpx.Proxy` wrapping
    rationale.
    """
    if not socket_options:
        return httpx.AsyncClient(proxy=proxy, verify=verify)
    transport = httpx.AsyncHTTPTransport(
        proxy=httpx.Proxy(proxy),
        verify=verify,
        socket_options=list(socket_options),
        limits=_DEFAULT_CONNECTION_LIMITS,
    )
    return httpx.AsyncClient(mounts={"all://": transport})


@lru_cache
def _cached_sync_httpx_client(
    base_url: str | None,
    timeout: Any,
    socket_options: tuple[SocketOption, ...] = (),
) -> _SyncHttpxClientWrapper:
    return _build_sync_httpx_client(base_url, timeout, socket_options)


@lru_cache
def _cached_async_httpx_client(
    base_url: str | None,
    timeout: Any,
    socket_options: tuple[SocketOption, ...] = (),
) -> _AsyncHttpxClientWrapper:
    return _build_async_httpx_client(base_url, timeout, socket_options)


def _get_default_httpx_client(
    base_url: str | None,
    timeout: Any,
    socket_options: tuple[SocketOption, ...] = (),
) -> _SyncHttpxClientWrapper:
    """获取默认的 httpx 客户端。

    除非超时为 `httpx.Timeout`（不可哈希），否则使用缓存的客户端。
    """
    try:
        hash(timeout)
    except TypeError:
        return _build_sync_httpx_client(base_url, timeout, socket_options)
    else:
        return _cached_sync_httpx_client(base_url, timeout, socket_options)


def _get_default_async_httpx_client(
    base_url: str | None,
    timeout: Any,
    socket_options: tuple[SocketOption, ...] = (),
) -> _AsyncHttpxClientWrapper:
    """获取默认的 httpx 客户端。

    除非超时为 `httpx.Timeout`（不可哈希），否则使用缓存的客户端。
    """
    try:
        hash(timeout)
    except TypeError:
        return _build_async_httpx_client(base_url, timeout, socket_options)
    else:
        return _cached_async_httpx_client(base_url, timeout, socket_options)


def _resolve_sync_and_async_api_keys(
    api_key: SecretStr | Callable[[], str] | Callable[[], Awaitable[str]],
) -> tuple[str | None | Callable[[], str], str | Callable[[], Awaitable[str]]]:
    """解析同步和异步 API 密钥值。

    因为 OpenAI 和 AsyncOpenAI 客户端支持同步或异步调用作为
    API 密钥，所以我们需要在此处解析独立的值。
    """
    if isinstance(api_key, SecretStr):
        sync_api_key_value: str | None | Callable[[], str] = api_key.get_secret_value()
        async_api_key_value: str | Callable[[], Awaitable[str]] = (
            api_key.get_secret_value()
        )
    elif callable(api_key):
        if inspect.iscoroutinefunction(api_key):
            async_api_key_value = api_key
            sync_api_key_value = None
        else:
            sync_api_key_value = cast(Callable, api_key)

            async def async_api_key_wrapper() -> str:
                return await asyncio.get_running_loop().run_in_executor(
                    None, cast(Callable, api_key)
                )

            async_api_key_value = async_api_key_wrapper

    return sync_api_key_value, async_api_key_value


T = TypeVar("T")

# 层次结构，因此仅继承 asyncio.TimeoutError 将不会被 `except TimeoutError:` 捕获。在
# Python ≥3.11 时它们是同一对象，因此列出两个基类会引发 TypeError:
# 重复基类。我们在此处类定义时解决该问题。
# 尽力清理；不要掩盖原始异常，
# 但留下 DEBUG 级别的追踪日志，以便在合适的日志级别下仍能发现连接池/传输相关的问题。
# 两个基类都会引发 TypeError: 重复的基类。我们在此处解决此问题
# 类定义时间。
_StreamChunkTimeoutBases: tuple[type, ...] = (
    (asyncio.TimeoutError,)
    if issubclass(asyncio.TimeoutError, TimeoutError)
    else (asyncio.TimeoutError, TimeoutError)
)


class StreamChunkTimeoutError(*_StreamChunkTimeoutBases):  # type: ignore[misc]
    """当在 `stream_chunk_timeout` 内未收到流式数据块时抛出。

    `issubclass(StreamChunkTimeoutError, asyncio.TimeoutError)` and
    `issubclass(StreamChunkTimeoutError, TimeoutError)` both hold on all
    supported Python versions, so existing `except asyncio.TimeoutError:`
    and `except TimeoutError:` handlers keep catching the exception. On
    Python 3.11+ the two exceptions are the same object, so only
    `asyncio.TimeoutError` appears in `__bases__`.

    Structured attributes (`timeout_s`, `model_name`, `chunks_received`)
    mirror the WARNING log's `extra=` payload so diagnostic code doesn't
    need to regex the message.
    """

    def __init__(
        self,
        timeout_s: float,
        *,
        model_name: str | None = None,
        chunks_received: int = 0,
    ) -> None:
        self.timeout_s = timeout_s
        self.model_name = model_name
        self.chunks_received = chunks_received
        context = []
        if model_name:
            context.append(f"model={model_name}")
        context.append(f"chunks_received={chunks_received}")
        suffix = f" ({', '.join(context)})"
        super().__init__(
            f"No streaming chunk received for {timeout_s:.1f}s{suffix}. The "
            f"connection may be alive at the TCP layer but is not producing "
            f"content. Tune or disable via the `stream_chunk_timeout` "
            f"constructor kwarg (set to None or 0 to disable) or the "
            f"`LANGCHAIN_OPENAI_STREAM_CHUNK_TIMEOUT_S` env var. See also "
            f"`http_socket_options` for the kernel-level TCP timeout that "
            f"catches dead TCP peers."
        )


async def _astream_with_chunk_timeout(
    source: AsyncIterator[T],
    timeout: float | None,
    *,
    model_name: str | None = None,
) -> AsyncIterator[T]:
    """从 `source` 中生成，但绑定每个数据块的等待时间。

    If `timeout` is None or <=0, yields directly with no wall-clock bound.
    Otherwise, each `__anext__` is wrapped in
    `asyncio.wait_for(..., timeout)`. A timeout raises
    `StreamChunkTimeoutError` (a `TimeoutError` subclass) whose message
    names the knob, the env-var override, the model, and how many chunks
    were received before the stall. A single-line structured log also
    fires at WARNING so the signal is visible in aggregate logging systems
    even when the exception is caught upstream.

    When the timeout is active, the source iterator is explicitly
    `aclose()`-d on early exit (timeout, consumer break, any exception) so
    the underlying httpx streaming connection is released promptly. The
    pass-through branch (timeout disabled) relies on httpx's GC-driven
    cleanup instead — matching the behavior of unwrapped streams.
    """
    if not timeout or timeout <= 0:
        async for item in source:
            yield item
        return

    chunks_received = 0
    it = source.__aiter__()
    try:
        while True:
            try:
                chunk = await asyncio.wait_for(it.__anext__(), timeout=timeout)
            except StopAsyncIteration:
                return
            except asyncio.TimeoutError as e:
                logger.warning(
                    "langchain_openai.stream_chunk_timeout fired",
                    extra={
                        "source": "stream_chunk_timeout",
                        "timeout_s": timeout,
                        "model_name": model_name,
                        "chunks_received": chunks_received,
                    },
                )
                raise StreamChunkTimeoutError(
                    timeout,
                    model_name=model_name,
                    chunks_received=chunks_received,
                ) from e
            chunks_received += 1
            yield chunk
    finally:
        aclose = getattr(it, "aclose", None)
        if aclose is not None:
            try:
                await aclose()
            except Exception as cleanup_exc:
                                # 尽力清理；不要掩盖原始异常,
                                # 但保留 DEBUG 追踪，以便池/传输错误仍可被发现
                                # 在正确的日志级别下可发现。
                logger.debug(
                    "aclose() during _astream_with_chunk_timeout cleanup "
                    "raised; ignoring",
                    exc_info=cleanup_exc,
                )

"""UUID 工具函数。

This module exports a uuid7 function to generate monotonic, time-ordered UUIDs
for tracing and similar operations.
"""

from __future__ import annotations

import typing
from uuid import UUID

from uuid_utils.compat import uuid7 as _uuid_utils_uuid7

if typing.TYPE_CHECKING:
    from uuid import UUID

_NANOS_PER_SECOND: typing.Final = 1_000_000_000


def _to_timestamp_and_nanos(nanoseconds: int) -> tuple[int, int]:
    """将纳秒级时间戳拆分为秒和剩余纳秒部分。"""
    seconds, nanos = divmod(nanoseconds, _NANOS_PER_SECOND)
    return seconds, nanos


def uuid7(nanoseconds: int | None = None) -> UUID:
    """从 Unix 时间戳（纳秒）和随机位生成 UUID。

    UUIDv7 对象在毫秒内具有单调性。

    Args:
        nanoseconds: 可选的纳秒时间戳。若未提供，则使用当前时间。

    Returns:
        一个 UUIDv7 对象。
    """
    # --- 48 ---   -- 4 --   --- 12 ---   -- 2 --   --- 30 ---   - 32 -
        # unix_ts_ms | version | counter_hi | variant | counter_lo | random
    #
        # 'counter = counter_hi | counter_lo' 是一个 42 位计数器，按 RFC 9562 §6.2 的方法 1
    # 构造，其最高位设为 0。
        # 'random' 是一个 32 位随机值，每次生成新 UUID 时都会重新生成。
    #
        # 如果在同一个毫秒内生成多个 UUID，'counter' 的最低位将加
    # 1。溢出时，时间戳向前推进，计数器重置为一个最高位为 0 的随机 42 位整数。
    #
        # 目前，仅委托给 uuid_utils 实现。
        # 的 'counter' 会加 1。溢出时，时间戳为
        # 高级且计数器重置为一个随机的 42 位整数（最高位为 1）
        # 设置为 0。

        # 目前，仅委托给 uuid_utils 实现
    if nanoseconds is None:
        return _uuid_utils_uuid7()
    seconds, nanos = _to_timestamp_and_nanos(nanoseconds)
    return _uuid_utils_uuid7(timestamp=seconds, nanos=nanos)


__all__ = ["uuid7"]

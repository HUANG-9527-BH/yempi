"""使用工具。"""

from collections.abc import Callable
from typing import Any


def _dict_int_op(
    left: dict[str, Any],
    right: dict[str, Any],
    op: Callable[[int, int], int],
    *,
    default: int = 0,
    depth: int = 0,
    max_depth: int = 100,
) -> dict[str, Any]:
    """对两个字典中对应值应用整数运算。

    递归地合并两个字典，对匹配键处的整数值应用给定的操作。

    支持嵌套字典。

    Args:
        left: 要合并的第一个字典。
        right: 要合并的第二个字典。
        op: 用于整数值的二元操作函数。
        default: 当键在字典中缺失时使用的默认值。
        depth: 当前递归深度（内部使用）。
        max_depth: 最大递归深度（防止无限循环）。

    Returns:
        包含合并值的新字典。

    Raises:
        ValueError: 如果超过 `max_depth` 或值类型不受支持时抛出。
    """
    if depth >= max_depth:
        msg = f"{max_depth=} exceeded, unable to combine dicts."
        raise ValueError(msg)
    combined: dict[str, Any] = {}
    for k in set(left).union(right):
        if isinstance(left.get(k, default), int) and isinstance(
            right.get(k, default), int
        ):
            combined[k] = op(left.get(k, default), right.get(k, default))
        elif isinstance(left.get(k, {}), dict) and isinstance(right.get(k, {}), dict):
            combined[k] = _dict_int_op(
                left.get(k, {}),
                right.get(k, {}),
                op,
                default=default,
                depth=depth + 1,
                max_depth=max_depth,
            )
        else:
            types = [type(d[k]) for d in (left, right) if k in d]
            msg = (
                f"Unknown value types: {types}. Only dict and int values are supported."
            )
            raise ValueError(msg)  # noqa: TRY004
    return combined

"""字符串工具函数。"""

from collections.abc import Iterable
from typing import Any


def stringify_value(val: Any) -> str:
    """将值转换为字符串形式。

    Args:
        val: 待转换的值。

    Returns:
        转换后的字符串。
    """
    if isinstance(val, str):
        return val
    if isinstance(val, dict):
        return "\n" + stringify_dict(val)
    if isinstance(val, list):
        return "\n".join(stringify_value(v) for v in val)
    return str(val)


def stringify_dict(data: dict[Any, Any]) -> str:
    """将字典转换为字符串形式。

    Args:
        data: 待转换的字典。

    Returns:
        转换后的字符串字典。
    """
    return "".join(f"{key}: {stringify_value(value)}\n" for key, value in data.items())


def comma_list(items: Iterable[Any]) -> str:
    """将可迭代对象转换为逗号分隔的字符串。

    Args:
        items: 待转换的可迭代对象。

    Returns:
        逗号分隔的字符串。
    """
    return ", ".join(str(item) for item in items)


def sanitize_for_postgres(text: str, replacement: str = "") -> str:
    r"""通过移除不兼容 PostgreSQL 的 NUL 字节来清理文本。

    PostgreSQL 的 text 字段不能包含 `NUL (0x00)` 字节，这可能导致在插入文档时出现
    `psycopg.DataError`。此函数会移除或替换此类字符以确保兼容性。

    Args:
        text: 待清理的文本。
        replacement: 用于替换 `NUL` 字节的字符串。

    Returns:
        移除或替换 `NUL` 字节后的清理文本。

    Example:
        >>> sanitize_for_postgres("Hello\\x00world")
        'Helloworld'
        >>> sanitize_for_postgres("Hello\\x00world", " ")
        'Hello world'
    """
    return text.replace("\x00", replacement)

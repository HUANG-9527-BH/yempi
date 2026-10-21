"""环境变量相关的工具函数。"""

from __future__ import annotations

import os
from typing import Any


def env_var_is_set(env_var: str) -> bool:
    """检查环境变量是否已设置。

    Args:
        env_var: 环境变量的名称。

    Returns:
        如果环境变量已设置，则返回 `True`，否则返回 `False`。
    """
    return env_var in os.environ and os.environ[env_var] not in {
        "",
        "0",
        "false",
        "False",
    }


def get_from_dict_or_env(
    data: dict[str, Any],
    key: str | list[str],
    env_key: str,
    default: str | None = None,
) -> str:
    """从字典或环境变量中获取值。

    Args:
        data: 要查找键的字典。
        key: 要在字典中查找的键。

            This can be a list of keys to try in order.
        env_key: The environment variable to look up if the key is not
            in the dictionary.
        default: The default value to return if the key is not in the dictionary
            or the environment.

    Returns:
        字典中的值或环境变量中的值。
    """
    if isinstance(key, (list, tuple)):
        for k in key:
            if value := data.get(k):
                return str(value)

    if isinstance(key, str) and key in data and data[key]:
        return str(data[key])

    key_for_err = key[0] if isinstance(key, (list, tuple)) else key

    return get_from_env(key_for_err, env_key, default=default)


def get_from_env(key: str, env_key: str, default: str | None = None) -> str:
    """从字典或环境变量中获取值。

    Args:
        key: The key to look up in the dictionary.
        env_key: The environment variable to look up if the key is not
            in the dictionary.
        default: The default value to return if the key is not in the dictionary
            or the environment.

    Returns:
        键的值。

    Raises:
        ValueError: If the key is not in the dictionary and no default value is
            provided or if the environment variable is not set.
    """
    if env_value := os.getenv(env_key):
        return env_value
    if default is not None:
        return default
    msg = (
        f"Did not find {key}, please add an environment variable"
        f" `{env_key}` which contains it, or pass"
        f" `{key}` as a named parameter."
    )
    raise ValueError(msg)

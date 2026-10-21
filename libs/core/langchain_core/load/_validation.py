"""LangChain 序列化验证工具。

提供基于转义的保护机制，防止序列化对象中的注入攻击。该方法采用白名单设计：仅在反序列化时将
`Serializable.to_json()` 显式生成的字典视为 LC 对象。

## 转义的工作原理

序列化过程中，包含 `'lc'` 键的普通字典（用户数据）会被包裹：

```python
{"lc": 1, ...}  # user data that looks like LC object
# becomes:
{"__lc_escaped__": {"lc": 1, ...}}
```

反序列化过程中，被转义的字典会被解包并返回为普通字典，而非实例化为 LC 对象。
"""

from typing import Any, cast

from langchain_core.load.serializable import (
    Serializable,
    to_json_not_implemented,
)

_LC_ESCAPED_KEY = "__lc_escaped__"
"""Sentinel key used to mark escaped user dicts during serialization.

When a plain dict contains 'lc' key (which could be confused with LC objects),
we wrap it as {"__lc_escaped__": {...original...}}.
"""


def _needs_escaping(obj: dict[str, Any]) -> bool:
    """检查字典是否需要转义以避免与 LC 对象混淆。

    字典需要转义的条件是：

    1. 它包含 `'lc'` 键（可能与 LC 序列化格式混淆）
    2. 它仅包含转义键（会被误认为是被转义的字典）
    """
    return "lc" in obj or (len(obj) == 1 and _LC_ESCAPED_KEY in obj)


def _escape_dict(obj: dict[str, Any]) -> dict[str, Any]:
    """将字典用转义标记包裹。

    Example:
        ```python
        {"key": "value"}  # becomes {"__lc_escaped__": {"key": "value"}}
        ```
    """
    return {_LC_ESCAPED_KEY: obj}


def _is_escaped_dict(obj: dict[str, Any]) -> bool:
    """检查字典是否为被转义的用户字典。

    Example:
        ```python
        {"__lc_escaped__": {...}}  # is an escaped dict
        ```
    """
    return len(obj) == 1 and _LC_ESCAPED_KEY in obj


def _serialize_value(obj: Any) -> Any:
    """使用转义机制序列化一个值。

    Called recursively on kwarg values to escape any plain dicts that could be confused
    with LC objects.

    Args:
        obj: 待序列化的值。

    Returns:
        根据需要转义用户字典后的序列化值。
    """
    if isinstance(obj, Serializable):
                # This is an LC object - serialize it properly (not escaped)
        return _serialize_lc_object(obj)
    if isinstance(obj, dict):
        if not all(isinstance(k, (str, int, float, bool, type(None))) for k in obj):
                        # if keys are not json serializable
            return to_json_not_implemented(obj)
                # Check if dict needs escaping BEFORE recursing into values.
                # If it needs escaping, wrap it as-is - the contents are user data that
                # will be returned as-is during deserialization (no instantiation).
                # This prevents re-escaping of already-escaped nested content.
        if _needs_escaping(obj):
            return _escape_dict(obj)
                # Safe dict (no 'lc' key) - recurse into values
        return {k: _serialize_value(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_serialize_value(item) for item in obj]
    if isinstance(obj, (str, int, float, bool, type(None))):
        return obj

        # 非 JSON 可序列化对象（datetime、自定义对象等）
    return to_json_not_implemented(obj)


def _get_secret_keys(obj: Serializable) -> set[str]:
    """返回构造函数关键字参数中声明为敏感信息的集合。

    与 `Serializable.to_json` 中的 MRO 遍历保持一致，因此此处返回的键与
    `_replace_secrets` 重写为敏感标记的值对应的键匹配。由 `_serialize_lc_object`
    使用，以决定在转义用户数据时跳过哪些参数。
    """
    secrets: dict[str, str] = {}
    model_fields = type(obj).model_fields
    for cls in [None, *obj.__class__.mro()]:
        if cls is Serializable:
            break
        this = cast("Serializable", obj if cls is None else super(cls, obj))
        secrets.update(this.lc_secrets)
        for key in list(secrets):
            if (key in model_fields) and (alias := model_fields[key].alias) is not None:
                secrets[alias] = secrets[key]
    return set(secrets)


def _serialize_lc_object(obj: Any) -> dict[str, Any]:
    """使用转义机制序列化一个 `Serializable` 对象，对关键字参数中的用户数据进行转义。

    Args:
        obj: 待序列化的 `Serializable` 对象。

    Returns:
        根据需要转义用户数据后的序列化字典。

    Note:
        Kwargs values are processed with `_serialize_value` to escape user data
        (like metadata) that contains `'lc'` keys. Secret fields are identified
        by the class's declared `lc_secrets` and skipped because `to_json()`
        already converted their values to secret markers.

        检查基于键名而非结构。基于结构的检查（"此字典看起来像敏感标记"）可被用户数据伪造，导致攻击者控制的自由格式字典绕过转义并进入
        Reviver。
    """
    if not isinstance(obj, Serializable):
        msg = f"Expected Serializable, got {type(obj)}"
        raise TypeError(msg)

    serialized: dict[str, Any] = dict(obj.to_json())

        # 处理 kwargs 以转义用户数据，防止其与 LC 对象混淆。
        # 跳过声明为 secret 的 kwargs - `to_json()` 已通过 `_replace_secrets` 将其值替换为
    # secret 标记。
        # 解包并原样返回用户数据（不再进行进一步反转义）。
    if serialized.get("type") == "constructor" and "kwargs" in serialized:
        secret_keys = _get_secret_keys(obj)
        serialized["kwargs"] = {
            k: v if k in secret_keys else _serialize_value(v)
            for k, v in serialized["kwargs"].items()
        }

    return serialized


def _unescape_value(obj: Any) -> Any:
    """解转义一个值，处理字典值和列表中的转义标记。

    When an escaped dict is encountered (`{"__lc_escaped__": ...}`), it's
    unwrapped and the contents are returned AS-IS (no further processing).
    The contents represent user data that should not be modified.

    对于普通字典和列表，我们递归查找任何嵌套的转义标记。

    Args:
        obj: 待解转义的值。

    Returns:
        解转义后的值。
    """
    if isinstance(obj, dict):
        if _is_escaped_dict(obj):
                        # 内容为用户数据，可能包含更多转义键，
                        # 但这些是用户实际数据的一部分。
                        # 普通字典 - 递归进入值以查找嵌套的转义标记
            return obj[_LC_ESCAPED_KEY]

                # 普通字典 - 递归进入值以查找嵌套的转义标记
        return {k: _unescape_value(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_unescape_value(item) for item in obj]
    return obj

"""将 LangChain 对象序列化为 JSON。

Provides `dumps` (to JSON string) and `dumpd` (to dict) for serializing
`Serializable` objects.

## 转义

During serialization, plain dicts (user data) that contain an `'lc'` key are escaped
by wrapping them: `{"__lc_escaped__": {...original...}}`. This prevents injection
attacks where malicious data could trick the deserializer into instantiating
arbitrary classes. The escape marker is removed during deserialization.

This is an allowlist approach: only dicts explicitly produced by
`Serializable.to_json()` are treated as LC objects; everything else is escaped if it
could be confused with the LC format.
"""

import json
from typing import Any

from pydantic import BaseModel

from langchain_core.load._validation import _serialize_value
from langchain_core.load.serializable import Serializable, to_json_not_implemented
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration


def default(obj: Any) -> Any:
    """为对象返回默认值。

    Args:
        obj: 若为 Serializable 对象，则将其序列化为 json。

    Returns:
        一个可序列化为 JSON 的对象，或一个 SerializedNotImplemented 对象。
    """
    if isinstance(obj, Serializable):
        return obj.to_json()
    return to_json_not_implemented(obj)


def _dump_pydantic_models(obj: Any) -> Any:
    """将嵌套的 Pydantic 模型转换为字典以进行 JSON 序列化。

    Handles the special case where a `ChatGeneration` contains an `AIMessage`
    with a parsed Pydantic model in `additional_kwargs["parsed"]`. Since
    Pydantic models aren't directly JSON serializable, this converts them to
    dicts.

    Args:
        obj: 待处理的对象。

    Returns:
        A copy of the object with nested Pydantic models converted to dicts, or
            the original object unchanged if no conversion was needed.
    """
    if (
        isinstance(obj, ChatGeneration)
        and isinstance(obj.message, AIMessage)
        and (parsed := obj.message.additional_kwargs.get("parsed"))
        and isinstance(parsed, BaseModel)
    ):
        obj_copy = obj.model_copy(deep=True)
        obj_copy.message.additional_kwargs["parsed"] = parsed.model_dump()
        return obj_copy
    return obj


def dumps(obj: Any, *, pretty: bool = False, **kwargs: Any) -> str:
    """返回对象的 JSON 字符串表示形式。

    Note:
        Plain dicts containing an `'lc'` key are automatically escaped to prevent
        confusion with LC serialization format. The escape marker is removed during
        deserialization.

    Args:
        obj: 待导出的对象。
        pretty: 是否美化输出 JSON。

            若为 `True`，JSON 将按 2 个空格或 `indent` 关键字提供的数量进行缩进。
            **kwargs: 传递给 `json.dumps` 的额外参数

    Returns:
        对象的 JSON 字符串表示形式。

    Raises:
        ValueError: 若传入了 `default` 参数。
    """
    if "default" in kwargs:
        msg = "`default` should not be passed to dumps"
        raise ValueError(msg)

    obj = _dump_pydantic_models(obj)
    serialized = _serialize_value(obj)

    if pretty:
        indent = kwargs.pop("indent", 2)
        return json.dumps(serialized, indent=indent, **kwargs)
    return json.dumps(serialized, **kwargs)


def dumpd(obj: Any) -> Any:
    """返回对象的字典表示形式。

    Note:
        Plain dicts containing an `'lc'` key are automatically escaped to prevent
        confusion with LC serialization format. The escape marker is removed during
        deserialization.

    Args:
        obj: 待导出的对象。

    Returns:
        可使用 `json.dumps` 序列化的字典。
    """
    obj = _dump_pydantic_models(obj)
    return _serialize_value(obj)

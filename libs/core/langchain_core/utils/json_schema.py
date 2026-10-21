"""JSON 模式相关的工具。"""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from collections.abc import Sequence


def _retrieve_ref(path: str, schema: dict[str, Any]) -> list[Any] | dict[Any, Any]:
    """使用路径从 JSON 模式中检索引用的对象。

    通过遍历模式结构来解析 JSON 模式引用（例如，`'#/definitions/MyType'`）。

    Args:
        path: 以 `'#'` 开头的引用路径（例如，`'#/definitions/MyType'`）。
        schema: 要搜索的 JSON 模式字典。

    Returns:
        被引用对象的深拷贝（字典或列表）。

    Raises:
        ValueError: 如果路径不以 `'#'` 开头。
        KeyError: 如果引用路径在模式中未找到。
    """
    components = path.split("/")
    if components[0] != "#":
        msg = (
            "ref paths are expected to be URI fragments, meaning they should start "
            "with #."
        )
        raise ValueError(msg)
    out: list[Any] | dict[Any, Any] = schema
    for component in components[1:]:
        if component in out:
            if isinstance(out, list):
                msg = f"Reference '{path}' not found."
                raise KeyError(msg)
            out = out[component]
        elif component.isdigit():
            index = int(component)
            if (isinstance(out, list) and 0 <= index < len(out)) or (
                isinstance(out, dict) and index in out
            ):
                out = out[index]
            else:
                msg = f"Reference '{path}' not found."
                raise KeyError(msg)
        else:
            msg = f"Reference '{path}' not found."
            raise KeyError(msg)
    return deepcopy(out)


def _process_dict_properties(
    properties: dict[str, Any],
    full_schema: dict[str, Any],
    processed_refs: set[str],
    skip_keys: Sequence[str],
    *,
    shallow_refs: bool,
) -> dict[str, Any]:
    """处理字典属性，递归进入嵌套结构。"""
    result: dict[str, Any] = {}
    for key, value in properties.items():
        if key in skip_keys:
                        # 跳过指定键的递归，直接复制其值
            result[key] = deepcopy(value)
        elif isinstance(value, (dict, list)):
                        # 递归处理嵌套的对象和数组
            result[key] = _dereference_refs_helper(
                value, full_schema, processed_refs, skip_keys, shallow_refs=shallow_refs
            )
        else:
                        # 直接复制原始值
            result[key] = value
    return result


def _dereference_refs_helper(
    obj: Any,
    full_schema: dict[str, Any],
    processed_refs: set[str] | None,
    skip_keys: Sequence[str],
    *,
    shallow_refs: bool,
) -> Any:
    """解除 JSON Schema `$ref` 对象的引用，处理纯引用和混合引用。

    此函数通过解析引用并合并其他属性来处理包含 `$ref` 属性的 JSON Schema 对象。它处理：

    - 纯 `$ref` 对象：`{"$ref": "#/path/to/definition"}`
    - 混合 `$ref` 对象：`{"$ref": "#/path", "title": "Custom Title", ...}`
    - 通过打破循环并保留非引用属性来处理循环引用

    Args:
        obj: 要处理的对象（可以是字典、列表或基本类型）
        full_schema: 包含所有定义的完整模式
        processed_refs: 跟踪当前正在处理的引用的集合（用于检测循环）
        skip_keys: 在其下跳过递归的键
        shallow_refs: 如果为 `True`，仅打破循环；如果为 `False`，则深度内联所有引用

    Returns:
        去除 `$ref` 属性并合并其他属性后的对象。
    """
    if processed_refs is None:
        processed_refs = set()

        # 情况 1：对象包含 `$ref` 属性（纯 `$ref` 或与其他属性混合）
    if isinstance(obj, dict) and "$ref" in obj:
        ref_path = obj["$ref"]
        additional_properties = {
            key: value for key, value in obj.items() if key != "$ref"
        }

                # 检测循环引用：如果正在处理此 `$ref`，
                # 则仅返回额外属性以打破循环
        if ref_path in processed_refs:
            return _process_dict_properties(
                additional_properties,
                full_schema,
                processed_refs,
                skip_keys,
                shallow_refs=shallow_refs,
            )

                # 标记此引用为正在处理中（用于循环检测）
        processed_refs.add(ref_path)

                # 获取并递归解析被引用的对象
        referenced_object = deepcopy(_retrieve_ref(ref_path, full_schema))
        resolved_reference = _dereference_refs_helper(
            referenced_object,
            full_schema,
            processed_refs,
            skip_keys,
            shallow_refs=shallow_refs,
        )

                # 清理：在返回前从处理集合中移除
        processed_refs.remove(ref_path)

                # 纯 `$ref` 情况：无额外属性，直接返回解析后的引用
        if not additional_properties:
            return resolved_reference

                # 混合 `$ref` 情况：将解析后的引用与额外属性合并
                # 额外属性优先于已解析的属性
        merged_result = {}
        if isinstance(resolved_reference, dict):
            merged_result.update(resolved_reference)

                # 处理额外属性并合并它们（它们会覆盖已解析的属性）
        processed_additional = _process_dict_properties(
            additional_properties,
            full_schema,
            processed_refs,
            skip_keys,
            shallow_refs=shallow_refs,
        )
        merged_result.update(processed_additional)

        return merged_result

        # 情况 2：普通字典，无 `$ref` - 处理所有属性
    if isinstance(obj, dict):
        return _process_dict_properties(
            obj, full_schema, processed_refs, skip_keys, shallow_refs=shallow_refs
        )

        # 情况 3：列表 - 递归处理每个元素
    if isinstance(obj, list):
        return [
            _dereference_refs_helper(
                item, full_schema, processed_refs, skip_keys, shallow_refs=shallow_refs
            )
            for item in obj
        ]

        # 情况 4：原始值（字符串、数字、布尔值、null）- 原样返回
    return obj


def dereference_refs(
    schema_obj: dict[str, Any],
    *,
    full_schema: dict[str, Any] | None = None,
    skip_keys: Sequence[str] | None = None,
) -> dict[str, Any]:
    """在模式对象中解析并内联 JSON Schema `$ref` 引用。

    This function processes a JSON Schema and resolves all `$ref` references by
    replacing them with the actual referenced content.

    处理简单引用和复杂情况，如循环引用以及包含额外属性的 `$ref` 对象。

    Args:
        schema_obj: 要处理的 JSON Schema 对象或片段。

            这可以是完整的模式，也可以只是其中一部分。
        full_schema: 包含所有定义的完整模式，`$refs` 可能指向这些定义。
            如果未提供，则默认为 `schema_obj`（当模式自包含时有用）。

            如果未提供，将默认为 `schema_obj`（在模式自包含时很有用）。 skip_keys:
            控制递归行为和引用解析深度。

            - If `None` (Default): Only recurse under `'$defs'` and use shallow
                reference resolution (break cycles but don't deep-inline nested refs)
            - If provided (even as `[]`): Recurse under all keys and use deep reference
                resolution (fully inline all nested references)

    Returns:
        所有 $ref 引用已解析并内联后的新字典。

            原始 `schema_obj` 不会被修改。

    Examples:
        基本引用解析：
        >>> schema = {
        ...     "type": "object",
        ...     "properties": {"name": {"$ref": "#/$defs/string_type"}},
        ...     "$defs": {"string_type": {"type": "string"}},
        ... }
        >>> result = dereference_refs(schema)
        >>> result["properties"]["name"]  # {"type": "string"}

        混合 `$ref` 与附加属性：

        >>> schema = {
        ...     "properties": {
        ...         "name": {"$ref": "#/$defs/base", "description": "User name"}
        ...     },
        ...     "$defs": {"base": {"type": "string", "minLength": 1}},
        ... }
        >>> result = dereference_refs(schema)
        >>> result["properties"]["name"]
        # {"type": "string", "minLength": 1, "description": "User name"}

        处理循环引用：

        >>> schema = {
        ...     "properties": {"user": {"$ref": "#/$defs/User"}},
        ...     "$defs": {
        ...         "User": {
        ...             "type": "object",
        ...             "properties": {"friend": {"$ref": "#/$defs/User"}},
        ...         }
        ...     },
        ... }
        >>> result = dereference_refs(schema)  # Won't cause infinite recursion

    !!! note

        - 循环引用会被妥善处理，通过打破循环来避免无限递归
        - 支持混合的 `$ref` 对象（同时包含 `$ref` 和其他属性）
        - 混合 `$refs` 中的额外属性会覆盖已解析的属性
        - 默认情况下，输出中会保留 `$defs` 部分
    """
    full = full_schema or schema_obj
    keys_to_skip = list(skip_keys) if skip_keys is not None else ["$defs"]
    shallow = skip_keys is None
    return cast(
        "dict[str, Any]",
        _dereference_refs_helper(
            schema_obj, full, None, keys_to_skip, shallow_refs=shallow
        ),
    )

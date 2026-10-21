from __future__ import annotations

from typing import Any


def merge_dicts(left: dict[str, Any], *others: dict[str, Any]) -> dict[str, Any]:
    r"""合并字典。

    Merge many dicts, handling specific scenarios where a key exists in both
    dictionaries but has a value of `None` in `'left'`. In such cases, the method uses
    the value from `'right'` for that key in the merged dictionary.

    Args:
        left: 第一个要合并的字典。
        others: 其他要合并的字典。

    Returns:
        合并后的字典。

    Raises:
        TypeError: 如果键同时存在于两个字典中但类型不同。
        TypeError: 如果值具有不受支持的类型。

    Example:
        If `left = {"function_call": {"arguments": None}}` and
        `right = {"function_call": {"arguments": "{\n"}}`, then, after merging, for the
        key `'function_call'`, the value from `'right'` is used, resulting in
        `merged = {"function_call": {"arguments": "{\n"}}`.
    """
    merged = left.copy()
    for right in others:
        for right_k, right_v in right.items():
            if right_k not in merged or (
                right_v is not None and merged[right_k] is None
            ):
                merged[right_k] = right_v
            elif right_v is None:
                continue
            elif type(merged[right_k]) is not type(right_v):
                msg = (
                    f'additional_kwargs["{right_k}"] already exists in this message,'
                    " but with a different type."
                )
                raise TypeError(msg)
            elif isinstance(merged[right_k], str):
                                # TODO: 在 0.3 版本中添加以下对 'type' 键的特殊处理，并移除
                                # merge_lists 'type' 逻辑。
                #
                                # if right_k == "type":
                                #     if merged[right_k] == right_v:
                                #         continue
                                #     else:
                                #         raise ValueError(
                                #             "Unable to merge. Two different values seen for special "
                                #             f"key 'type': {merged[right_k]} and {right_v}. 'type' "
                                #             "should either occur once or have the same value across "
                                #             "all dicts."
                #         )
                if (right_k == "index" and merged[right_k].startswith("lc_")) or (
                    right_k in {"id", "output_version", "model_provider"}
                    and merged[right_k] == right_v
                ):
                    continue
                merged[right_k] += right_v
            elif isinstance(merged[right_k], dict):
                merged[right_k] = merge_dicts(merged[right_k], right_v)
            elif isinstance(merged[right_k], list):
                merged[right_k] = merge_lists(merged[right_k], right_v)
            elif merged[right_k] == right_v:
                continue
            elif isinstance(merged[right_k], bool):
                                # `bool` 是 `int` 的子类，因此若无此检查，不同的
                                # 布尔值会进入下方的 int 分支并被累加，
                                # 静默地将例如 `True + False` 转换为整数 `1`。
                msg = (
                    f"Additional kwargs key {right_k} already exists in left dict and "
                    f"value has unsupported type {type(merged[right_k])}."
                )
                raise TypeError(msg)
            elif isinstance(merged[right_k], int):
                                # 使用最后胜出策略保留标识和时间字段
                                # 而非累加：
                                # - index: 标识某块属于哪个工具调用
                                # - created/timestamp: 时间值不应累积
                if right_k in {"index", "created", "timestamp"}:
                    merged[right_k] = right_v
                else:
                    merged[right_k] += right_v
            else:
                msg = (
                    f"Additional kwargs key {right_k} already exists in left dict and "
                    f"value has unsupported type {type(merged[right_k])}."
                )
                raise TypeError(msg)
    return merged


def merge_lists(left: list[Any] | None, *others: list[Any] | None) -> list[Any] | None:
    """合并多个列表，处理 `None` 值。

    Args:
        left: 第一个要合并的列表。
        others: 其他要合并的列表。

    Returns:
        合并后的列表。
    """
    merged = left.copy() if left is not None else None
    for other in others:
        if other is None:
            continue
        if merged is None:
            merged = other.copy()
        else:
            for e in other:
                if (
                    isinstance(e, dict)
                    and "index" in e
                    and (
                        isinstance(e["index"], int)
                        or (
                            isinstance(e["index"], str) and e["index"].startswith("lc_")
                        )
                    )
                ):
                    to_merge = [
                        i
                        for i, e_left in enumerate(merged)
                        if (
                            isinstance(e_left, dict)
                            and "index" in e_left
                            and e_left["index"] == e["index"]  # index 匹配
                            and (  # ID 不一致
                                e_left.get("id") in {None, ""}
                                or e.get("id") in {None, ""}
                                or e_left.get("id") == e.get("id")
                            )
                        )
                    ]
                    if to_merge:
                                                # TODO: 一旦 merge_dict 更新了对 'type' 的特殊处理，便移除此处
                                                # 处理逻辑。
                        if (left_type := merged[to_merge[0]].get("type")) and (
                            e.get("type") == "non_standard" and "value" in e
                        ):
                            if left_type != "non_standard":
                                                                # standard + non_standard
                                new_e: dict[str, Any] = {
                                    "extras": {
                                        k: v
                                        for k, v in e["value"].items()
                                        if k != "type"
                                    }
                                }
                            else:
                                                                # non_standard + non_standard
                                new_e = {
                                    "value": {
                                        k: v
                                        for k, v in e["value"].items()
                                        if k != "type"
                                    }
                                }
                                if "index" in e:
                                    new_e["index"] = e["index"]
                        else:
                            new_e = (
                                {k: v for k, v in e.items() if k != "type"}
                                if "type" in e
                                else e
                            )
                        merged[to_merge[0]] = merge_dicts(merged[to_merge[0]], new_e)
                    else:
                        merged.append(e)
                else:
                    merged.append(e)
    return merged


def merge_obj(left: Any, right: Any) -> Any:
    """合并两个对象。

    It handles specific scenarios where a key exists in both dictionaries but has a
    value of `None` in `'left'`. In such cases, the method uses the value from `'right'`
    for that key in the merged dictionary.

    Args:
        left: 第一个要合并的对象。
        right: 另一个要合并的对象。

    Returns:
        合并后的对象。

    Raises:
        TypeError: 如果键同时存在于两个字典中但类型不同。
        ValueError: 如果两个对象无法合并。
    """
    if left is None or right is None:
        return left if left is not None else right
    if type(left) is not type(right):
        msg = (
            f"left and right are of different types. Left type:  {type(left)}. Right "
            f"type: {type(right)}."
        )
        raise TypeError(msg)
    if isinstance(left, str):
        return left + right
    if isinstance(left, dict):
        return merge_dicts(left, right)
    if isinstance(left, list):
        return merge_lists(left, right)
    if left == right:
        return left
    msg = (
        f"Unable to merge {left=} and {right=}. Both must be of type str, dict, or "
        f"list, or else be two equal objects."
    )
    raise ValueError(msg)

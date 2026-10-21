"""JSON 相关的实用工具。"""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING, Any

from langchain_core.exceptions import OutputParserException

if TYPE_CHECKING:
    from collections.abc import Callable


def _replace_new_line(match: re.Match[str]) -> str:
    """将正则匹配中的换行符替换为转义序列。

    Args:
        match: 包含待处理字符串的正则匹配对象。

    Returns:
        换行符、回车符、制表符和引号均已正确转义的字符串。
    """
    value = match.group(2)
    value = re.sub(r"\n", r"\\n", value)
    value = re.sub(r"\r", r"\\r", value)
    value = re.sub(r"\t", r"\\t", value)
    value = re.sub(r'(?<!\\)"', r"\"", value)

    return match.group(1) + value + match.group(3)


def _custom_parser(multiline_string: str | bytes | bytearray) -> str:
    r"""多行字符串的自定义解析器。

    The LLM response for `action_input` may be a multiline string containing unescaped
    newlines, tabs or quotes. This function replaces those characters with their escaped
    counterparts. (newlines in JSON must be double-escaped: `\\n`).

    Returns:
        替换换行符、制表符和引号后的修改后字符串。
    """
    if isinstance(multiline_string, (bytes, bytearray)):
        multiline_string = multiline_string.decode()

    return re.sub(
        r'("action_input"\:\s*")(.*?)(")',
        _replace_new_line,
        multiline_string,
        flags=re.DOTALL,
    )


# 改编自
# https://github.com/KillianLucas/open-interpreter/blob/5b6080fae1f8c68938a1e4fa8667e3744084ee21/interpreter/utils/parse_partial_json.py
# MIT License


def parse_partial_json(s: str, *, strict: bool = False) -> Any:
    """解析可能缺少闭合大括号的 JSON 字符串。

    Args:
        s: 要解析的 JSON 字符串。
        strict: 是否使用严格解析模式。

    Returns:
        解析后的 JSON 对象，以 Python 字典形式返回。
    """
        # 尝试直接解析字符串。
    try:
        return json.loads(s, strict=strict)
    except json.JSONDecodeError:
        pass

        # 初始化变量。
    new_chars = []
    stack = []
    is_inside_string = False
    escaped = False

        # 逐个字符处理字符串。
    for char in s:
        new_char = char
        if is_inside_string:
            if char == '"' and not escaped:
                is_inside_string = False
            elif char == "\n" and not escaped:
                new_char = (
                    "\\n"  # 将换行符替换为转义序列。
                )
            elif char == "\\":
                escaped = not escaped
            else:
                escaped = False
        elif char == '"':
            is_inside_string = True
            escaped = False
        elif char == "{":
            stack.append("}")
        elif char == "[":
            stack.append("]")
        elif char in {"}", "]"}:
            if stack and stack[-1] == char:
                stack.pop()
            else:
                                # 闭合字符不匹配；输入格式错误。
                return None

                # 将已处理的字符追加到新字符串中。
        new_chars.append(new_char)

        # 如果处理结束后仍处于字符串内部，
        # 则需要关闭该字符串。
    if is_inside_string:
        if escaped:  # 移除未完成的转义字符
            new_chars.pop()
        new_chars.append('"')

        # 反转栈以获取闭合字符。
    stack.reverse()

        # 尝试对字符串的变体进行解析，直到成功或字符用尽。
    while new_chars:
                # 按相反顺序关闭所有剩余的开放结构，
                # 顺序与它们被打开时相反。
                # 尝试将修改后的字符串解析为 JSON。
        try:
            return json.loads("".join(new_chars + stack), strict=strict)
        except json.JSONDecodeError:
                        # 如果仍无法将字符串解析为 JSON，
                        # 尝试移除最后一个字符
            new_chars.pop()

        # 如果到达此处，说明已无字符可移除
        # 且仍无法将字符串解析为 JSON，因此返回原始字符串的解析错误
        # 尝试在三重反引号内查找 JSON 字符串
    return json.loads(s, strict=strict)


_json_markdown_re = re.compile(r"```(json)?(.*)", re.DOTALL)


def parse_json_markdown(
    json_string: str, *, parser: Callable[[str], Any] = parse_partial_json
) -> Any:
    """从 Markdown 字符串中解析 JSON 字符串。

    Args:
        json_string: Markdown 字符串。
        parser: 要使用的解析器。

    Returns:
        解析后的 JSON 对象，以 Python 字典形式返回。
    """
    try:
        return _parse_json(json_string, parser=parser)
    except json.JSONDecodeError:
                # 如果未找到匹配项，假设整个字符串是 JSON 字符串
        match = _json_markdown_re.search(json_string)

                # 否则，使用反引号内的内容
                # 去除首尾的空白字符、换行符和反引号
        json_str = json_string if match is None else match.group(2)
    return _parse_json(json_str, parser=parser)


_json_strip_chars = " \n\r\t`"


def _parse_json(
    json_str: str, *, parser: Callable[[str], Any] = parse_partial_json
) -> Any:
    """解析 JSON 字符串，处理特殊字符和空白字符。

    去除字符串首尾的空白字符、换行符和反引号，
    然后在解析前处理特殊字符。

    Args:
        json_str: 要解析的 JSON 字符串。
        parser: 可选的自定义解析函数。

    Returns:
        解析后的 JSON 对象。
    """
        # 处理返回值内部的换行符和其他特殊字符
    json_str = json_str.strip(_json_strip_chars)

        # 将 JSON 字符串解析为 Python 字典
    json_str = _custom_parser(json_str)

        # 将 JSON 字符串解析为 Python 字典
    return parser(json_str)


def parse_and_check_json_markdown(
    text: str, expected_keys: list[str]
) -> dict[str, Any]:
    """从 Markdown 字符串中解析并验证 JSON 字符串。

    检查其是否包含预期的键。

    Args:
        text: Markdown 字符串。
        expected_keys: JSON 字符串中预期的键。

    Returns:
        解析后的 JSON 对象，以 Python 字典形式返回。

    Raises:
        OutputParserException: 如果 JSON 字符串无效或不包含
            预期的键。
    """
    try:
        json_obj = parse_json_markdown(text)
    except json.JSONDecodeError as e:
        msg = f"Got invalid JSON object. Error: {e}"
        raise OutputParserException(msg) from e
    if not isinstance(json_obj, dict):
        error_message = (
            f"Expected JSON object (dict), but got: {type(json_obj).__name__}. "
        )
        raise OutputParserException(error_message, llm_output=text)

    for key in expected_keys:
        if key not in json_obj:
            msg = (
                f"Got invalid return object. Expected key `{key}` "
                f"to be present, but got {json_obj}"
            )
            raise OutputParserException(msg)
    return json_obj

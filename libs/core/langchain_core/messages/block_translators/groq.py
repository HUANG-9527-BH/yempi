"""从 Groq 内容派生的标准内容块。"""

import json
import re
from typing import Any

from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.messages import content as types
from langchain_core.messages.base import _extract_reasoning_from_additional_kwargs


def _populate_extras(
    standard_block: types.ContentBlock, block: dict[str, Any], known_fields: set[str]
) -> types.ContentBlock:
    """修改一个内容块，填充额外信息。"""
    if standard_block.get("type") == "non_standard":
        return standard_block

    for key, value in block.items():
        if key not in known_fields:
            if "extras" not in standard_block:
                                # 以下 type-ignores 是因为 mypy 认为非标准内容块可能
                                # 会到达此处，尽管我们在上方已排除了此类情况。
                standard_block["extras"] = {}  # type: ignore[typeddict-unknown-key]
            standard_block["extras"][key] = value  # type: ignore[typeddict-item]

    return standard_block


def _parse_code_json(s: str) -> dict[str, Any]:
    """从 Groq 内置工具内容中提取 Python 代码。

    从如下格式的字符串中提取 'code' 字段的值：
    {"code": some_arbitrary_text_with_unescaped_quotes}

    由于 Groq 可能不会对执行工具中的引号进行转义，例如：
    ```
    '{"code": "import math; print("The square root of 101 is: "); print(math.sqrt(101))"}'
    ```
    """  # noqa: E501
    m = re.fullmatch(r'\s*\{\s*"code"\s*:\s*"(.*)"\s*\}\s*', s, flags=re.DOTALL)
    if not m:
        msg = (
            "Could not extract Python code from Groq tool arguments. "
            "Expected a JSON object with a 'code' field."
        )
        raise ValueError(msg)
    return {"code": m.group(1)}


def _convert_to_v1_from_groq(message: AIMessage) -> list[types.ContentBlock]:
    """将 Groq 消息内容转换为 v1 格式。"""
    content_blocks: list[types.ContentBlock] = []

    if reasoning_block := _extract_reasoning_from_additional_kwargs(message):
        content_blocks.append(reasoning_block)

    if executed_tools := message.additional_kwargs.get("executed_tools"):
        for idx, executed_tool in enumerate(executed_tools):
            args: dict[str, Any] | None = None
            if arguments := executed_tool.get("arguments"):
                try:
                    args = json.loads(arguments)
                except json.JSONDecodeError:
                    if executed_tool.get("type") == "python":
                        try:
                            args = _parse_code_json(arguments)
                        except ValueError:
                            continue
                    elif (
                        executed_tool.get("type") == "function"
                        and executed_tool.get("name") == "python"
                    ):
                                                # GPT-OSS
                        args = {"code": arguments}
                    else:
                        continue
            if isinstance(args, dict):
                name = ""
                if executed_tool.get("type") == "search":
                    name = "web_search"
                elif executed_tool.get("type") == "python" or (
                    executed_tool.get("type") == "function"
                    and executed_tool.get("name") == "python"
                ):
                    name = "code_interpreter"
                server_tool_call: types.ServerToolCall = {
                    "type": "server_tool_call",
                    "name": name,
                    "id": str(idx),
                    "args": args,
                }
                content_blocks.append(server_tool_call)
            if tool_output := executed_tool.get("output"):
                tool_result: types.ServerToolResult = {
                    "type": "server_tool_result",
                    "tool_call_id": str(idx),
                    "output": tool_output,
                    "status": "success",
                }
                known_fields = {"type", "arguments", "index", "output"}
                _populate_extras(tool_result, executed_tool, known_fields)
                content_blocks.append(tool_result)

    if isinstance(message.content, str) and message.content:
        content_blocks.append({"type": "text", "text": message.content})

    content_blocks.extend(
        {
            "type": "tool_call",
            "name": tool_call["name"],
            "args": tool_call["args"],
            "id": tool_call.get("id"),
        }
        for tool_call in message.tool_calls
    )

    return content_blocks


def translate_content(message: AIMessage) -> list[types.ContentBlock]:
    """从包含 Groq 内容的消息派生标准内容块。

    Args:
        message: 需要转换的消息。

    Returns:
        派生出的内容块。
    """
    return _convert_to_v1_from_groq(message)


def translate_content_chunk(message: AIMessageChunk) -> list[types.ContentBlock]:
    """从包含 Groq 内容的消息片段派生标准内容块。

    Args:
        message: 需要转换的消息片段。

    Returns:
        派生出的内容块。
    """
    return _convert_to_v1_from_groq(message)


def _register_groq_translator() -> None:
    """将 Groq 转换器注册到中央注册表中。

    模块导入时自动运行。
    """
    from langchain_core.messages.block_translators import (  # noqa: PLC0415
        register_translator,
    )

    register_translator("groq", translate_content, translate_content_chunk)


_register_groq_translator()

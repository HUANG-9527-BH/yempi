"""Anthropic 对话模型的实验性工具调用支持。"""

from __future__ import annotations

import json
from typing import (
    Any,
)

SYSTEM_PROMPT_FORMAT = """In this environment you have access to a set of tools you can use to answer the user's question.

You may call them like this:
<function_calls>
<invoke>
<tool_name>$TOOL_NAME</tool_name>
<parameters>
<$PARAMETER_NAME>$PARAMETER_VALUE</$PARAMETER_NAME>
...
</parameters>
</invoke>
</function_calls>

Here are the tools available:
<tools>
{formatted_tools}
</tools>"""  # noqa: E501

TOOL_FORMAT = """<tool_description>
<tool_name>{tool_name}</tool_name>
<description>{tool_description}</description>
<parameters>
{formatted_parameters}
</parameters>
</tool_description>"""

TOOL_PARAMETER_FORMAT = """<parameter>
<name>{parameter_name}</name>
<type>{parameter_type}</type>
<description>{parameter_description}</description>
</parameter>"""


def _get_type(parameter: dict[str, Any]) -> str:
    if "type" in parameter:
        return parameter["type"]
    if "anyOf" in parameter:
        return json.dumps({"anyOf": parameter["anyOf"]})
    if "allOf" in parameter:
        return json.dumps({"allOf": parameter["allOf"]})
    return json.dumps(parameter)


def get_system_message(tools: list[dict]) -> str:
    """生成一个系统消息，描述可用的工具。"""
    tools_data: list[dict] = [
        {
            "tool_name": tool["name"],
            "tool_description": tool["description"],
            "formatted_parameters": "\n".join(
                [
                    TOOL_PARAMETER_FORMAT.format(
                        parameter_name=name,
                        parameter_type=_get_type(parameter),
                        parameter_description=parameter.get("description"),
                    )
                    for name, parameter in tool["parameters"]["properties"].items()
                ],
            ),
        }
        for tool in tools
    ]
    tools_formatted = "\n".join(
        [
            TOOL_FORMAT.format(
                tool_name=tool["tool_name"],
                tool_description=tool["tool_description"],
                formatted_parameters=tool["formatted_parameters"],
            )
            for tool in tools_data
        ],
    )
    return SYSTEM_PROMPT_FORMAT.format(formatted_tools=tools_formatted)


def _xml_to_dict(t: Any) -> str | dict[str, Any]:
        # 基础情况：如果元素没有子元素，则返回其文本或空字符串。
    if len(t) == 0:
        return t.text or ""

        # 递归情况：元素有子元素。将其子元素转换为字典。
    d: dict[str, Any] = {}
    for child in t:
        if child.tag not in d:
            d[child.tag] = _xml_to_dict(child)
        else:
                        # 处理具有相同标签的多个子元素
            if not isinstance(d[child.tag], list):
                d[child.tag] = [d[child.tag]]  # 将现有条目转换为列表
            d[child.tag].append(_xml_to_dict(child))
    return d


def _xml_to_function_call(invoke: Any, tools: list[dict]) -> dict[str, Any]:
    name = invoke.find("tool_name").text
    arguments = _xml_to_dict(invoke.find("parameters"))

        # 使参数中的列表元素实际为列表
    filtered_tools = [tool for tool in tools if tool["name"] == name]
    if len(filtered_tools) > 0 and not isinstance(arguments, str):
        tool = filtered_tools[0]
        for key, value in arguments.items():
            if (
                key in tool["parameters"]["properties"]
                and "type" in tool["parameters"]["properties"][key]
            ):
                if tool["parameters"]["properties"][key][
                    "type"
                ] == "array" and not isinstance(value, list):
                    arguments[key] = [value]
                if (
                    tool["parameters"]["properties"][key]["type"] != "object"
                    and isinstance(value, dict)
                    and len(value.keys()) == 1
                ):
                    arguments[key] = next(iter(value.values()))

    return {
        "function": {
            "name": name,
            "arguments": json.dumps(arguments),
        },
        "type": "function",
    }


def _xml_to_tool_calls(elem: Any, tools: list[dict]) -> list[dict[str, Any]]:
    """将 XML 元素及其子元素转换为字典的字典。"""
    invokes = elem.findall("invoke")

    return [_xml_to_function_call(invoke, tools) for invoke in invokes]

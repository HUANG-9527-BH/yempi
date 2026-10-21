"""Anthropic 工具调用的输出解析器。"""

from __future__ import annotations

from typing import Any, cast

from langchain_core.messages import AIMessage, ToolCall
from langchain_core.messages.tool import tool_call
from langchain_core.output_parsers import BaseGenerationOutputParser
from langchain_core.outputs import ChatGeneration, Generation
from pydantic import BaseModel, ConfigDict


class ToolsOutputParser(BaseGenerationOutputParser):
    """工具调用的输出解析器。"""

    first_tool_only: bool = False
    """是否仅返回第一个工具调用。"""
    args_only: bool = False
    """是否仅返回工具调用的参数。"""
    pydantic_schemas: list[type[BaseModel]] | None = None
    """用于将工具调用解析为的 Pydantic 模式。"""

    model_config = ConfigDict(
        extra="forbid",
    )

    def parse_result(self, result: list[Generation], *, partial: bool = False) -> Any:
        """将候选模型生成结果列表解析为特定格式。

        Args:
            result: A list of `Generation` to be parsed. The Generations are assumed
                to be different candidate outputs for a single model input.
            partial: (Not used) Whether the result is a partial result. If `True`, the
                parser may return a partial result, which may not be complete or valid.

        Returns:
            结构化输出。

        """
        if not result or not isinstance(result[0], ChatGeneration):
            return None if self.first_tool_only else []
        message = cast("AIMessage", result[0].message)
        tool_calls: list = [
            dict(tc) for tc in _extract_tool_calls_from_message(message)
        ]
        if isinstance(message.content, list):
                        # 将工具调用 ID 映射到索引
            id_to_index = {
                block["id"]: i
                for i, block in enumerate(message.content)
                if isinstance(block, dict) and block["type"] == "tool_use"
            }
            tool_calls = [{**tc, "index": id_to_index[tc["id"]]} for tc in tool_calls]
        if self.pydantic_schemas:
            tool_calls = [self._pydantic_parse(tc) for tc in tool_calls]
        elif self.args_only:
            tool_calls = [tc["args"] for tc in tool_calls]
        else:
            pass

        if self.first_tool_only:
            return tool_calls[0] if tool_calls else None
        return list(tool_calls)

    def _pydantic_parse(self, tool_call: dict) -> BaseModel:
        cls_ = {schema.__name__: schema for schema in self.pydantic_schemas or []}[
            tool_call["name"]
        ]
        return cls_(**tool_call["args"])


def _extract_tool_calls_from_message(message: AIMessage) -> list[ToolCall]:
    """从内容块列表中提取工具调用。"""
    if message.tool_calls:
        return message.tool_calls
    return extract_tool_calls(message.content)


def extract_tool_calls(content: str | list[str | dict[str, Any]]) -> list[ToolCall]:
    """从内容块列表中提取工具调用。"""
    if isinstance(content, list):
        tool_calls = []
        for block in content:
            if isinstance(block, str):
                continue
            if block["type"] != "tool_use":
                continue
            tool_calls.append(
                tool_call(name=block["name"], args=block["input"], id=block["id"]),
            )
        return tool_calls
    return []

"""解析 OpenAI 工具输出的工具。"""

import copy
import json
import logging
from json import JSONDecodeError
from typing import Annotated, Any

from pydantic import BaseModel, SkipValidation, ValidationError
from pydantic.v1 import BaseModel as BaseModelV1

from langchain_core.exceptions import OutputParserException
from langchain_core.messages import AIMessage, InvalidToolCall
from langchain_core.messages.tool import invalid_tool_call
from langchain_core.messages.tool import tool_call as create_tool_call
from langchain_core.output_parsers.transform import BaseCumulativeTransformOutputParser
from langchain_core.outputs import ChatGeneration, Generation
from langchain_core.utils.json import parse_partial_json
from langchain_core.utils.pydantic import (
    TypeBaseModel,
)

logger = logging.getLogger(__name__)


def parse_tool_call(
    raw_tool_call: dict[str, Any],
    *,
    partial: bool = False,
    strict: bool = False,
    return_id: bool = True,
) -> dict[str, Any] | None:
    """解析单个工具调用。

    Args:
        raw_tool_call: 要解析的原始工具调用。
        partial: 是否解析部分 JSON。
        strict: 是否允许非 JSON 兼容字符串。
        return_id: 是否返回工具调用 ID。

    Returns:
        解析后的工具调用。

    Raises:
        OutputParserException: 如果工具调用不是有效的 JSON。
    """
    if "function" not in raw_tool_call:
        return None

    arguments = raw_tool_call["function"]["arguments"]

    if partial:
        try:
            function_args = parse_partial_json(arguments, strict=strict)
        except (JSONDecodeError, TypeError):  # None args raise TypeError
            return None
        # 处理无参数工具的 None 或空字符串参数
    elif not arguments:
        function_args = {}
    else:
        try:
            function_args = json.loads(arguments, strict=strict)
        except JSONDecodeError as e:
            msg = (
                f"Function {raw_tool_call['function']['name']} arguments:\n\n"
                f"{arguments}\n\nare not valid JSON. "
                f"Received JSONDecodeError {e}"
            )
            raise OutputParserException(msg) from e
    parsed = {
        "name": raw_tool_call["function"]["name"] or "",
        "args": function_args or {},
    }
    if return_id:
        parsed["id"] = raw_tool_call.get("id")
        parsed = create_tool_call(**parsed)  # type: ignore[assignment,arg-type]
    return parsed


def make_invalid_tool_call(
    raw_tool_call: dict[str, Any],
    error_msg: str | None,
) -> InvalidToolCall:
    """从原始工具调用创建一个 `InvalidToolCall`。

    Args:
        raw_tool_call: 原始工具调用。
        error_msg: 错误消息。

    Returns:
        包含错误消息的 `InvalidToolCall` 实例。
    """
    return invalid_tool_call(
        name=raw_tool_call["function"]["name"],
        args=raw_tool_call["function"]["arguments"],
        id=raw_tool_call.get("id"),
        error=error_msg,
    )


def parse_tool_calls(
    raw_tool_calls: list[dict[str, Any]],
    *,
    partial: bool = False,
    strict: bool = False,
    return_id: bool = True,
) -> list[dict[str, Any]]:
    """解析一组工具调用。

    Args:
        raw_tool_calls: 要解析的原始工具调用列表。
        partial: 是否解析部分 JSON。
        strict: 是否允许非 JSON 兼容字符串。
        return_id: 是否返回工具调用 ID。

    Returns:
        解析后的工具调用列表。

    Raises:
        OutputParserException: 如果任意工具调用不是有效的 JSON。
    """
    final_tools: list[dict[str, Any]] = []
    exceptions = []
    for tool_call in raw_tool_calls:
        try:
            parsed = parse_tool_call(
                tool_call, partial=partial, strict=strict, return_id=return_id
            )
            if parsed:
                final_tools.append(parsed)
        except OutputParserException as e:
            exceptions.append(str(e))
            continue
    if exceptions:
        raise OutputParserException("\n\n".join(exceptions))
    return final_tools


class JsonOutputToolsParser(BaseCumulativeTransformOutputParser[Any]):
    """从 OpenAI 响应中解析工具。"""

    strict: bool = False
    """是否允许非 JSON 兼容字符串。

    See: 参见：https://docs.python.org/3/library/json.html#encoders-and-decoders

    在解析输出可能包含 Unicode 字符或换行符时很有用。
    """

    return_id: bool = False
    """是否返回工具调用 ID。"""

    first_tool_only: bool = False
    """是否仅返回第一个工具调用。

    If `False`, the result will be a list of tool calls, or an empty list if no tool
    calls are found.

    如果为 `True`，且找到多个工具调用，则仅返回第一个，其余将被忽略。
    如果未找到工具调用，将返回 `None`。

    如果未找到工具调用，将返回 `None`。
    """

    def parse_result(self, result: list[Generation], *, partial: bool = False) -> Any:
        """将 LLM 调用的结果解析为工具调用列表。

        Args:
            result: LLM 调用的结果。
            partial: 是否解析部分 JSON。

                如果为 `True`，输出将是一个包含已返回所有键的 JSON 对象。

                如果为 `False`，输出将是完整的 JSON 对象。

        Returns:
            解析后的工具调用列表。

        Raises:
            OutputParserException: 如果输出不是有效的 JSON。
        """
        generation = result[0]
        if not isinstance(generation, ChatGeneration):
            msg = "This output parser can only be used with a chat generation."
            raise OutputParserException(msg)
        message = generation.message
        if isinstance(message, AIMessage) and message.tool_calls:
            tool_calls = [dict(tc) for tc in message.tool_calls]
            for tool_call in tool_calls:
                if not self.return_id:
                    _ = tool_call.pop("id")
        else:
            try:
                raw_tool_calls = copy.deepcopy(message.additional_kwargs["tool_calls"])
            except KeyError:
                return []
            tool_calls = parse_tool_calls(
                raw_tool_calls,
                partial=partial,
                strict=self.strict,
                return_id=self.return_id,
            )
                # 为了向后兼容
        for tc in tool_calls:
            tc["type"] = tc.pop("name")

        if self.first_tool_only:
            return tool_calls[0] if tool_calls else None
        return tool_calls

    def parse(self, text: str) -> Any:
        """将 LLM 调用的输出解析为工具调用列表。

        Args:
            text: LLM 调用的输出。

        Returns:
            解析后的工具调用列表。
        """
        raise NotImplementedError


class JsonOutputKeyToolsParser(JsonOutputToolsParser):
    """从 OpenAI 响应中解析工具。"""

    key_name: str
    """要返回的工具类型。"""

    def parse_result(self, result: list[Generation], *, partial: bool = False) -> Any:
        """将 LLM 调用的结果解析为工具调用列表。

        Args:
            result: LLM 调用的结果。
            partial: 是否解析部分 JSON。
                如果为 `True`，输出将是一个包含
                    已返回所有键的 JSON 对象。
                如果为 `False`，输出将是完整的 JSON 对象。

        Raises:
            OutputParserException: 如果生成内容不是聊天生成。

        Returns:
            解析后的工具调用列表。
        """
        generation = result[0]
        if not isinstance(generation, ChatGeneration):
            msg = "This output parser can only be used with a chat generation."
            raise OutputParserException(msg)
        message = generation.message
        if isinstance(message, AIMessage) and message.tool_calls:
            parsed_tool_calls = [dict(tc) for tc in message.tool_calls]
            for tool_call in parsed_tool_calls:
                if not self.return_id:
                    _ = tool_call.pop("id")
        else:
            try:
                                # 此字段仅用于向后兼容 / 缓存的消息
                                # 所有新消息应使用 `message.tool_calls`
                raw_tool_calls = copy.deepcopy(message.additional_kwargs["tool_calls"])
            except KeyError:
                if self.first_tool_only:
                    return None
                return []
            parsed_tool_calls = parse_tool_calls(
                raw_tool_calls,
                partial=partial,
                strict=self.strict,
                return_id=self.return_id,
            )
                # 为了向后兼容
        for tc in parsed_tool_calls:
            tc["type"] = tc.pop("name")
        if self.first_tool_only:
            parsed_result = list(
                filter(lambda x: x["type"] == self.key_name, parsed_tool_calls)
            )
            single_result = (
                parsed_result[0]
                if parsed_result and parsed_result[0]["type"] == self.key_name
                else None
            )
            if self.return_id:
                return single_result
            if single_result:
                return single_result["args"]
            return None
        return (
            [res for res in parsed_tool_calls if res["type"] == self.key_name]
            if self.return_id
            else [
                res["args"] for res in parsed_tool_calls if res["type"] == self.key_name
            ]
        )


# ValidationError 的常见原因是由于 max_tokens 导致的输出截断。
_MAX_TOKENS_ERROR = (
    "Output parser received a `max_tokens` stop reason. "
    "The output is likely incomplete—please increase `max_tokens` "
    "or shorten your prompt."
)


class PydanticToolsParser(JsonOutputToolsParser):
    """从 OpenAI 响应中解析工具。"""

    tools: Annotated[list[TypeBaseModel], SkipValidation()]
    """要解析的工具。"""

        # TODO: 支持更细粒度的对象流式传输。
        # 目前仅在所有 Pydantic 对象字段都存在后才开始流式传输。
    def parse_result(self, result: list[Generation], *, partial: bool = False) -> Any:
        """将 LLM 调用的结果解析为 Pydantic 对象列表。

        Args:
            result: LLM 调用的结果。
            partial: 是否解析部分 JSON。

                如果为 `True`，输出将是一个包含所有已返回键的 JSON 对象。

                如果为 `False`，输出将是完整的 JSON 对象。

        Returns:
            解析后的 Pydantic 对象。

        Raises:
            ValueError: 如果工具调用参数不是字典。 ValidationError:
            如果工具调用参数不符合 Pydantic 模型。
        """
        json_results = super().parse_result(result, partial=partial)
        if not json_results:
            return None if self.first_tool_only else []

        json_results = [json_results] if self.first_tool_only else json_results
        name_dict_v2: dict[str, TypeBaseModel] = {
            tool.model_config.get("title") or tool.__name__: tool
            for tool in self.tools
            if issubclass(tool, BaseModel)
        }
        name_dict_v1: dict[str, TypeBaseModel] = {
            tool.__name__: tool for tool in self.tools if issubclass(tool, BaseModelV1)
        }
        name_dict: dict[str, TypeBaseModel] = {**name_dict_v2, **name_dict_v1}
        pydantic_objects = []
        for res in json_results:
            if not isinstance(res["args"], dict):
                if partial:
                    continue
                msg = (
                    f"Tool arguments must be specified as a dict, received: "
                    f"{res['args']}"
                )
                raise ValueError(msg)

            try:
                tool = name_dict[res["type"]]
            except KeyError as e:
                available = ", ".join(name_dict.keys()) or "<no_tools>"
                msg = (
                    f"Unknown tool type: {res['type']!r}. Available tools: {available}"
                )
                raise OutputParserException(msg) from e

            try:
                pydantic_objects.append(tool(**res["args"]))
            except (ValidationError, ValueError):
                if partial:
                    continue
                has_max_tokens_stop_reason = any(
                    generation.message.response_metadata.get("stop_reason")
                    == "max_tokens"
                    for generation in result
                    if isinstance(generation, ChatGeneration)
                )
                if has_max_tokens_stop_reason:
                    logger.exception(_MAX_TOKENS_ERROR)
                raise
        if self.first_tool_only:
            return pydantic_objects[0] if pydantic_objects else None
        return pydantic_objects

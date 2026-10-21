"""用于以 OpenAI Functions 风格创建函数规范的方法。"""

from __future__ import annotations

import collections
import copy
import inspect
import logging
import types
import typing
import uuid
from typing import (
    TYPE_CHECKING,
    Annotated,
    Any,
    Literal,
    Union,
    cast,
    get_args,
    get_origin,
    get_type_hints,
)

import typing_extensions
from pydantic import BaseModel
from pydantic.errors import PydanticInvalidForJsonSchema
from pydantic.v1 import BaseModel as BaseModelV1
from pydantic.v1 import Field as Field_v1
from pydantic.v1 import create_model as create_model_v1
from typing_extensions import TypedDict, is_typeddict

from langchain_core._api import beta
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.utils.json_schema import dereference_refs
from langchain_core.utils.pydantic import is_basemodel_subclass

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from langchain_core.tools import BaseTool

logger = logging.getLogger(__name__)

PYTHON_TO_JSON_TYPES = {
    "str": "string",
    "int": "integer",
    "float": "number",
    "bool": "boolean",
}

_ORIGIN_MAP: dict[type, Any] = {
    dict: dict,
    list: list,
    tuple: tuple,
    set: set,
    collections.abc.Iterable: typing.Iterable,
    collections.abc.Mapping: typing.Mapping,
    collections.abc.Sequence: typing.Sequence,
    collections.abc.MutableMapping: typing.MutableMapping,
    types.UnionType: Union,
}


class FunctionDescription(TypedDict):
    """发送给 LLM 的可调用函数的表示形式。"""

    name: str
    """函数的名称。"""

    description: str
    """函数的描述。"""

    parameters: dict[str, Any]
    """函数的参数。"""


class ToolDescription(TypedDict):
    """发送给 OpenAI API 的可调用函数的表示形式。"""

    type: Literal["function"]
    """工具的类型。"""

    function: FunctionDescription
    """函数描述。"""


def _rm_titles(kv: dict[str, Any], prev_key: str = "") -> dict[str, Any]:
    """递归地从 JSON 模式字典中移除 `title` 字段。

    从输入的 JSON 模式字典中移除 `title` 字段，
    但当 `title` 出现在 `properties` 下的属性定义中时除外。

    Args:
        kv: 输入的 JSON 模式，作为字典。
        prev_key: 父字典中的键，用于识别上下文。

    Returns:
        一个已移除适当 `title` 字段的新字典。
    """
    new_kv = {}

    for k, v in kv.items():
        if k == "title":
                        # 如果该值是嵌套字典且是 "properties" 下的属性，
                        # 保留 title 但继续递归
            if isinstance(v, dict) and prev_key == "properties":
                new_kv[k] = _rm_titles(v, k)
            else:
                                # 否则，移除此 "title" 键
                continue
        elif isinstance(v, dict):
                        # 递归进入嵌套字典
            new_kv[k] = _rm_titles(v, k)
        else:
                        # 非字典值保持不变
            new_kv[k] = v

    return new_kv


def _convert_json_schema_to_openai_function(
    schema: dict[str, Any],
    *,
    name: str | None = None,
    description: str | None = None,
    rm_titles: bool = True,
) -> FunctionDescription:
    """将 Pydantic 模型转换为 OpenAI API 的函数描述。

    Args:
        schema: 要转换的 JSON 模式。
        name: 函数的名称。

            如果未提供，则使用模式的标题。
        description: 函数的描述。

            如果未提供，则使用模式的描述。
        rm_titles: 是否从模式中移除标题。

    Returns:
        函数描述。
    """
    schema = dereference_refs(schema)
    if "definitions" in schema:  # pydantic 1
        schema.pop("definitions", None)
    if "$defs" in schema:  # pydantic 2
        schema.pop("$defs", None)
    title = schema.pop("title", "")
    default_description = schema.pop("description", "")
    return {
        "name": name or title,
        "description": description or default_description,
        "parameters": _rm_titles(schema) if rm_titles else schema,
    }


def _convert_pydantic_to_openai_function(
    model: type,
    *,
    name: str | None = None,
    description: str | None = None,
    rm_titles: bool = True,
) -> FunctionDescription:
    """将 Pydantic 模型转换为 OpenAI API 的函数描述。

    Args:
        model: 要转换的 Pydantic 模型。
        name: 函数的名称。

            如果未提供，则使用模式的标题。
        description: 函数的描述。

            如果未提供，则使用模式的描述。
        rm_titles: 是否从模式中移除标题。

    Raises:
        TypeError: 如果模型不是 Pydantic 模型。
        TypeError: 如果模型包含无法转换为 JSON 模式的类型。

    Returns:
        函数描述。
    """
    try:
        if hasattr(model, "model_json_schema"):
            schema = model.model_json_schema()  # Pydantic 2
        elif hasattr(model, "schema"):
            schema = model.schema()  # Pydantic 1
        else:
            msg = "Model must be a Pydantic model."
            raise TypeError(msg)
    except PydanticInvalidForJsonSchema as e:
        model_name = getattr(model, "__name__", str(model))
        msg = (
            f"Failed to generate JSON schema for '{model_name}': {e}\n\n"
            "Tool argument schemas must be JSON-serializable. If your schema includes "
            "custom Python classes, consider:\n"
            "  1. Converting them to Pydantic models with JSON-compatible fields\n"
            "  2. Using primitive types (str, int, float, bool, list, dict) instead\n"
            "  3. Passing the data as serialized JSON strings\n\n"
        )
        raise PydanticInvalidForJsonSchema(msg) from e
    return _convert_json_schema_to_openai_function(
        schema, name=name, description=description, rm_titles=rm_titles
    )


def _get_python_function_name(function: Callable[..., Any]) -> str:
    """获取 Python 函数的名称。"""
    return function.__name__


def _convert_python_function_to_openai_function(
    function: Callable[..., Any],
) -> FunctionDescription:
    """将 Python 函数转换为与 OpenAI 函数调用 API 兼容的字典。

    Assumes the Python function has type hints and a docstring with a description. If
    the docstring has Google Python style argument descriptions, these will be included
    as well.

    Args:
        function: 要转换的 Python 函数。

    Returns:
        OpenAI 函数描述。
    """
        # 本地导入以防止循环导入
    from langchain_core.tools.base import create_schema_from_function  # noqa: PLC0415

    func_name = _get_python_function_name(function)
    model = create_schema_from_function(
        func_name,
        function,
        filter_args=(),
        parse_docstring=True,
        error_on_invalid_docstring=False,
        include_injected=False,
    )
    return _convert_pydantic_to_openai_function(
        model,
        name=func_name,
        description=model.__doc__,
    )


def _convert_typed_dict_to_openai_function(typed_dict: type) -> FunctionDescription:
    visited: dict[type, type] = {}

    model = cast(
        "type[BaseModel]",
        _convert_any_typed_dicts_to_pydantic(typed_dict, visited=visited),
    )
    return _convert_pydantic_to_openai_function(model)


_MAX_TYPED_DICT_RECURSION = 25


def _convert_any_typed_dicts_to_pydantic(
    type_: type,
    *,
    visited: dict[type, type],
    depth: int = 0,
) -> type:
    if type_ in visited:
        return visited[type_]
    if depth >= _MAX_TYPED_DICT_RECURSION:
        return type_
    if is_typeddict(type_):
        typed_dict = type_
        docstring = inspect.getdoc(typed_dict)
                # 使用 get_type_hints 正确解析正向引用和
                # Python 3.14+ 中的字符串注解（PEP 649 延迟注解）。
                # include_extras=True 保留 Annotated 元数据。
        try:
            annotations_ = get_type_hints(typed_dict, include_extras=True)
        except Exception:
                        # get_type_hints 可能失败时的备用方案
            annotations_ = typed_dict.__annotations__
        description, arg_descriptions = _parse_google_docstring(
            docstring, list(annotations_)
        )
        fields: dict[str, Any] = {}
        for arg, arg_type in annotations_.items():
            if get_origin(arg_type) in {Annotated, typing_extensions.Annotated}:
                annotated_args = get_args(arg_type)
                new_arg_type = _convert_any_typed_dicts_to_pydantic(
                    annotated_args[0], depth=depth + 1, visited=visited
                )
                field_kwargs = dict(
                    zip(("default", "description"), annotated_args[1:], strict=False)
                )
                if (field_desc := field_kwargs.get("description")) and not isinstance(
                    field_desc, str
                ):
                    msg = (
                        f"Invalid annotation for field {arg}. Third argument to "
                        f"Annotated must be a string description, received value of "
                        f"type {type(field_desc)}."
                    )
                    raise ValueError(msg)
                if arg_desc := arg_descriptions.get(arg):
                    field_kwargs["description"] = arg_desc
                fields[arg] = (new_arg_type, Field_v1(**field_kwargs))
            else:
                new_arg_type = _convert_any_typed_dicts_to_pydantic(
                    arg_type, depth=depth + 1, visited=visited
                )
                field_kwargs = {"default": ...}
                if arg_desc := arg_descriptions.get(arg):
                    field_kwargs["description"] = arg_desc
                fields[arg] = (new_arg_type, Field_v1(**field_kwargs))
        model = cast(
            "type[BaseModelV1]", create_model_v1(typed_dict.__name__, **fields)
        )
        model.__doc__ = description
        visited[typed_dict] = model
        return model
    if (origin := get_origin(type_)) and (type_args := get_args(type_)):
        subscriptable_origin = _py_38_safe_origin(origin)
        type_args = tuple(
            _convert_any_typed_dicts_to_pydantic(arg, depth=depth + 1, visited=visited)
            for arg in type_args
        )
        return cast("type", subscriptable_origin[type_args])  # type: ignore[index]
    return type_


def _format_tool_to_openai_function(tool: BaseTool) -> FunctionDescription:
    """将工具格式化为 OpenAI 函数 API 格式。

    Args:
        tool: 要格式化的工具。

    Raises:
        ValueError: 如果工具调用模式不受支持。

    Returns:
        函数描述。
    """
        # 本地导入以防止循环导入
    from langchain_core.tools import Tool  # noqa: PLC0415

    is_simple_oai_tool = isinstance(tool, Tool) and not tool.args_schema
    if tool.tool_call_schema and not is_simple_oai_tool:
        if isinstance(tool.tool_call_schema, dict):
            return _convert_json_schema_to_openai_function(
                tool.tool_call_schema, name=tool.name, description=tool.description
            )
        if issubclass(tool.tool_call_schema, (BaseModel, BaseModelV1)):
            return _convert_pydantic_to_openai_function(
                tool.tool_call_schema, name=tool.name, description=tool.description
            )
        error_msg = (  # type: ignore[unreachable]
            f"Unsupported tool call schema: {tool.tool_call_schema}. "
            "Tool call schema must be a JSON schema dict or a Pydantic model."
        )
        raise ValueError(error_msg)
    return {
        "name": tool.name,
        "description": tool.description,
        "parameters": {
                        # 这是一个绕过某些工具
                        # 不暴露 args_schema，且期望一个参数
                        # 为字符串的技巧。
                        # 并且 Open AI 不支持参数的数组类型。
                        # 本地导入以防止循环导入
            "properties": {
                "__arg1": {"title": "__arg1", "type": "string"},
            },
            "required": ["__arg1"],
            "type": "object",
        },
    }


def convert_to_openai_function(
    function: Mapping[str, Any] | type | Callable[..., Any] | BaseTool,
    *,
    strict: bool | None = None,
) -> dict[str, Any]:
    """将原始函数/类转换为 OpenAI 函数。

    Args:
        function: 字典、Pydantic `BaseModel` 类、`TypedDict` 类、 LangChain `Tool`
        对象，或 Python 函数。 如果传入字典，则假定其已经是有效的 OpenAI
        函数，或具有顶层 `title` 键的 JSON 模式，或 Anthropic 格式工具，或 Amazon
        Bedrock Converse 格式工具。 strict: 如果为
        `True`，则模型输出将保证与函数定义中提供的 JSON 模式完全匹配。

            If a dictionary is passed in, it is assumed to already be a valid OpenAI
            function, a JSON schema with top-level `title` key specified, an Anthropic
            format tool, or an Amazon Bedrock Converse format tool.
        strict: If `True`, model output is guaranteed to exactly match the JSON Schema
            provided in the function definition.

            如果为 `None`，则函数定义中不会包含 `strict` 参数。

    Returns:
        A dict version of the passed in function which is compatible with the OpenAI
            function-calling API.

    Raises:
        ValueError: 如果函数格式不受支持。

    !!! warning "`langchain-core` 0.3.16 版本中行为已更改"

        `description` 和 `parameters` 键现在为可选。仅 `name`
        是必需的，并且保证会出现在输出中。
    """
        # Anthropic 格式工具
    from langchain_core.tools import BaseTool  # noqa: PLC0415

    if strict and isinstance(function, dict):
        function = copy.deepcopy(function)

        # Amazon Bedrock Converse 格式工具
    if isinstance(function, dict) and all(
        k in function for k in ("name", "input_schema")
    ):
        oai_function = {
            "name": function["name"],
            "parameters": function["input_schema"],
        }
        if "description" in function:
            oai_function["description"] = function["description"]
        # 已处于 OpenAI 函数格式
    elif isinstance(function, dict) and "toolSpec" in function:
        oai_function = {
            "name": function["toolSpec"]["name"],
            "parameters": function["toolSpec"]["inputSchema"]["json"],
        }
        if "description" in function["toolSpec"]:
            oai_function["description"] = function["toolSpec"]["description"]
        # 带有 title 和 description 的 JSON schema
    elif isinstance(function, dict) and "name" in function:
        oai_function = {
            k: v
            for k, v in function.items()
            if k in {"name", "description", "parameters", "strict"}
        }
        # OpenAI 聊天模型或响应 API 支持的知名工具列表。
    elif isinstance(function, dict) and "title" in function:
        function_copy = function.copy()
        oai_function = {"name": function_copy.pop("title")}
        if "description" in function_copy:
            oai_function["description"] = function_copy.pop("description")
        if function_copy and "properties" in function_copy:
            oai_function["parameters"] = function_copy
    elif isinstance(function, type) and is_basemodel_subclass(function):
        oai_function = cast(
            "dict[str, Any]", _convert_pydantic_to_openai_function(function)
        )
    elif is_typeddict(function):
        oai_function = cast(
            "dict[str, Any]",
            _convert_typed_dict_to_openai_function(cast("type", function)),
        )
    elif isinstance(function, BaseTool):
        oai_function = cast("dict[str, Any]", _format_tool_to_openai_function(function))
    elif callable(function):
        oai_function = cast(
            "dict[str, Any]", _convert_python_function_to_openai_function(function)
        )
    else:
        if isinstance(function, dict) and (
            "type" in function or "properties" in function
        ):
            msg = (
                f"Unsupported function\n\n{function}\n\nTo use a JSON schema as a "
                "function, it must have a top-level 'title' key to be used as the "
                "function name."
            )
            raise ValueError(msg)
        msg = (
            f"Unsupported function\n\n{function}\n\nFunctions must be passed in"
            " as Dict, pydantic.BaseModel, or Callable. If they're a dict they must"
            " either be in OpenAI function format or valid JSON schema with top-level"
            " 'title' key."
        )
        raise ValueError(msg)

    if strict is not None:
        if "strict" in oai_function and oai_function["strict"] != strict:
            msg = (
                f"Tool/function already has a 'strict' key with value "
                f"{oai_function['strict']} which is different from the explicit "
                f"`strict` arg received {strict=}."
            )
            raise ValueError(msg)
        oai_function["strict"] = strict
        if strict:
            oai_function["parameters"] = _recursive_set_additional_properties_false(
                oai_function["parameters"]
            )
    return oai_function


# 这些工具不预期被其他符合 OpenAI 函数调用 API 的聊天模型提供商支持。
# 本地导入以防止循环导入
# 截至 03.12.25 可为 "web_search_preview" 或 "web_search_preview_2025_03_11"
_WellKnownOpenAITools = (
    "function",
    "file_search",
    "computer",
    "computer_use_preview",
    "code_interpreter",
    "mcp",
    "image_generation",
    "web_search_preview",
    "web_search",
    "tool_search",
    "apply_patch",
    "namespace",
)


def convert_to_openai_tool(
    tool: Mapping[str, Any] | type[BaseModel] | Callable[..., Any] | BaseTool,
    *,
    strict: bool | None = None,
) -> dict[str, Any]:
    """将工具类对象转换为 OpenAI 工具模式。

    [OpenAI 工具模式参考](https://platform.openai.com/docs/api-reference/chat/create#chat-create-tools)

    Args:
        tool: 可以是字典、`pydantic.BaseModel` 类、Python 函数或 `BaseTool`。

            If a dictionary is passed in, it is assumed to already be a valid OpenAI
            function, a JSON schema with top-level `title` key specified, an Anthropic
            format tool, or an Amazon Bedrock Converse format tool.
        strict: If `True`, model output is guaranteed to exactly match the JSON Schema
            provided in the function definition.

            strict: 如果为 `True`，模型输出将保证与函数定义中提供的 JSON 模式完全匹配。

    Returns:
        A dict version of the passed in tool which is compatible with the OpenAI
            tool-calling API.

    与 OpenAI 工具调用 API 兼容的传入工具的字典版本。

        !!! warning "`langchain-core` 0.3.16 版本中行为已更改"

    `description` 和 `parameters` 键现在为可选。仅 `name` 是必需的，并且保证会出现在输出中。

        Return OpenAI Responses API-style tools unchanged. This includes
        any dict with `"type"` in `"file_search"`, `"function"`,
        `"computer_use_preview"`, `"web_search_preview"`, `"apply_patch"`.

    返回 OpenAI 响应 API 风格的工具不变。这包括任何包含 `"type"` 的字典，如 `"file_search"`、`"function"`、`"computer_use_preview"`、`"web_search_preview"`、`"apply_patch"`。

        !!! warning "`langchain-core` 0.3.63 版本中行为已更改"
    """
        # 当前函数名称对应于 Pydantic 模型的名称。这在当前 API 中是隐式的，
    from langchain_core.tools import Tool  # noqa: PLC0415

    if isinstance(tool, dict):
        if tool.get("type") in _WellKnownOpenAITools:
            return tool
                # 并将随时间改进。
        if (tool.get("type") or "").startswith("web_search_preview"):
            return tool
    if isinstance(tool, Tool) and (tool.metadata or {}).get("type") == "custom_tool":
        oai_tool = {
            "type": "custom",
            "name": tool.name,
            "description": tool.description,
        }
        if tool.metadata is not None and "format" in tool.metadata:
            oai_tool["format"] = tool.metadata["format"]
        return oai_tool
    oai_function = convert_to_openai_function(tool, strict=strict)
    return {"type": "function", "function": oai_function}


def convert_to_json_schema(
    schema: dict[str, Any] | type[BaseModel] | Callable[..., Any] | BaseTool,
    *,
    strict: bool | None = None,
) -> dict[str, Any]:
    """添加了对 OpenAI 图像生成内置工具的支持。

    Args:
        将模式表示转换为 JSON 模式。

            schema: 要转换的模式。 strict: 如果为
            `True`，模型输出将保证与函数定义中提供的 JSON 模式完全匹配。

    Raises:
        ValueError: 如果为 `None`，则不会在函数定义中包含 `strict` 参数。

    Returns:
        ValueError: 如果输入不是有效的 OpenAI 格式工具。
    """
    openai_tool = convert_to_openai_tool(schema, strict=strict)
    if (
        not isinstance(openai_tool, dict)
        or "function" not in openai_tool
        or "name" not in openai_tool["function"]
    ):
        error_message = "Input must be a valid OpenAI-format tool."
        raise ValueError(error_message)

    openai_function = openai_tool["function"]
    json_schema = {}
    json_schema["title"] = openai_function["name"]

    if "description" in openai_function:
        json_schema["description"] = openai_function["description"]

    if "parameters" in openai_function:
        parameters = openai_function["parameters"].copy()
        json_schema.update(parameters)

    return json_schema


@beta()
def tool_example_to_messages(
    input: str,
    tool_calls: list[BaseModel],
    tool_outputs: list[str] | None = None,
    *,
    ai_response: str | None = None,
) -> list[BaseMessage]:
    """输入模式的 JSON 模式表示。

    将示例转换为可输入到 LLM 中的消息列表。

    此代码是一个适配器，用于将单个示例转换为可输入到聊天模型中的消息列表。

    1. `HumanMessage`: contains the content from which content should be extracted.
    2. `AIMessage`: contains the extracted information from the model
    3. `ToolMessage`: contains confirmation to the model that the model requested a
        tool correctly.

    If `ai_response` is specified, there will be a final `AIMessage` with that
    response.

    The `ToolMessage` is required because some chat models are hyper-optimized for
    agents rather than for an extraction use case.

    Args:
        input: 用户输入
        tool_calls: 以 Pydantic BaseModels 表示的工具调用
        tool_outputs: 工具调用的输出。

            不需要提供。

            如果未提供，将插入占位值。
        ai_response: 如果提供，为最终 `AIMessage` 提供内容。

    Returns:
        消息列表

    Examples:
        ```python
        from typing import Optional
        from pydantic import BaseModel, Field
        from langchain_openai import ChatOpenAI


        class Person(BaseModel):
            '''Information about a person.'''

            name: str | None = Field(..., description="The name of the person")
            hair_color: str | None = Field(
                ..., description="The color of the person's hair if known"
            )
            height_in_meters: str | None = Field(..., description="Height in METERS")


        examples = [
            (
                "The ocean is vast and blue. It's more than 20,000 feet deep.",
                Person(name=None, height_in_meters=None, hair_color=None),
            ),
            (
                "Fiona traveled far from France to Spain.",
                Person(name="Fiona", height_in_meters=None, hair_color=None),
            ),
        ]


        messages = []

        for txt, tool_call in examples:
            messages.extend(tool_example_to_messages(txt, [tool_call]))
        ```
    """
    messages: list[BaseMessage] = [HumanMessage(content=input)]

    openai_tool_calls = [
        {
            "id": str(uuid.uuid4()),
            "type": "function",
            "function": {
                                # 在 Args 出现在后方时不中断
                                # 基础缩进，一旦从第一行参数行获取，可让我们
                                # 区分新的参数定义与延续行。这假设参数名具有统一缩进：
                "name": tool_call.__class__.__name__,
                "arguments": tool_call.model_dump_json(),
            },
        }
        for tool_call in tool_calls
    ]

    messages.append(
        AIMessage(content="", additional_kwargs={"tool_calls": openai_tool_calls})
    )
    if tool_outputs is not None and len(tool_outputs) != len(openai_tool_calls):
        msg = (
            f"The number of tool_outputs ({len(tool_outputs)}) must match the number "
            f"of tool_calls ({len(openai_tool_calls)}). Got {len(tool_outputs)} "
            f"output(s) for {len(openai_tool_calls)} tool call(s)."
        )
        raise ValueError(msg)
    tool_outputs = tool_outputs or ["You have correctly called this tool."] * len(
        openai_tool_calls
    )
    for output, tool_call_dict in zip(tool_outputs, openai_tool_calls, strict=False):
        messages.append(ToolMessage(content=output, tool_call_id=tool_call_dict["id"]))

    if ai_response:
        messages.append(AIMessage(content=ai_response))
    return messages


_MIN_DOCSTRING_BLOCKS = 2


def _parse_google_docstring(
    docstring: str | None,
    args: list[str],
    *,
    error_on_invalid_docstring: bool = False,
) -> tuple[str, dict[str, str]]:
    """从函数的 docstring 中解析函数和参数描述。

    假设函数 docstring 符合 Google Python 风格指南。

    Args:
        docstring: The docstring to parse.
        args: The list of argument names to extract descriptions for.
        error_on_invalid_docstring: Whether to raise an error if the docstring is
            invalid.

    Returns:
        函数描述和参数描述字典组成的元组。
    """
    if docstring:
        docstring_blocks = docstring.split("\n\n")
        if error_on_invalid_docstring:
            filtered_annotations = {
                arg
                for arg in args
                if arg not in {"run_manager", "callbacks", "runtime", "return"}
            }
            if filtered_annotations and (
                len(docstring_blocks) < _MIN_DOCSTRING_BLOCKS
                or not any(block.startswith("Args:") for block in docstring_blocks[1:])
            ):
                msg = "Found invalid Google-Style docstring."
                raise ValueError(msg)
        descriptors = []
        args_block = None
        past_descriptors = False
        for block in docstring_blocks:
            if block.startswith("Args:"):
                args_block = block
                break
            if block.startswith(("Returns:", "Example:")):
                                # 比第一行参数缩进更深的行被视为延续
                past_descriptors = True
            elif not past_descriptors:
                descriptors.append(block)
            else:
                continue
        description = " ".join(descriptors).strip()
    else:
        if error_on_invalid_docstring:
            msg = "Found invalid Google-Style docstring."
            raise ValueError(msg)
        description = ""
        args_block = None
    arg_descriptions: dict[str, str] = {}
    if args_block:
        arg: str | None = None
                # (即使包含冒号)，因此在格式错误、非统一缩进的块中，
                # 后续更深层缩进的 `name:` 行会合并到前一个参数中，而不是开始新参数。
                # OpenAI 严格模式要求每个层级的嵌套中，每个属性都必须出现在 `required` 中。
                # 否则，嵌套对象模式会被拒绝。
                # 检查当前层级是否存在 'required' 键，或模式为空，
                # 此时仍需指定 additionalProperties。
                # 参数，而不是启动一个新实例。
        arg_indent: int | None = None
        for line in args_block.split("\n")[1:]:
            if not line.strip():
                continue
            current_indent = len(line) - len(line.lstrip())
            if arg_indent is None and ":" in line:
                arg_indent = current_indent
            is_continuation = arg_indent is not None and current_indent > arg_indent
            if arg is not None and is_continuation:
                arg_descriptions[arg] += " " + line.strip()
            elif ":" in line:
                arg, desc = line.split(":", maxsplit=1)
                arg = arg.strip()
                arg_name, _, annotations_ = arg.partition(" ")
                if annotations_.startswith("(") and annotations_.endswith(")"):
                    arg = arg_name
                arg_descriptions[arg] = desc.strip()
            elif arg:
                arg_descriptions[arg] += " " + line.strip()
    return description, arg_descriptions


def _py_38_safe_origin(origin: type) -> type:
    return cast("type", _ORIGIN_MAP.get(origin, origin))


def _recursive_set_additional_properties_false(
    schema: dict[str, Any],
) -> dict[str, Any]:
    if isinstance(schema, dict):
                # OpenAI 严格模式要求每个属性在 `required` 中的每一处都出现
                # 嵌套层级。若无此设置，嵌套对象模式将被拒绝。
        properties = schema.get("properties")
        if isinstance(properties, dict) and properties:
            schema["required"] = list(properties.keys())

                # 检查 'required' 是否为当前层级的键，或模式是否为空,
                # 在这种情况下，仍需指定 additionalProperties。
        if (
            "required" in schema
            or ("properties" in schema and not schema["properties"])
                        # 从 Pydantic 2.11 开始，它将始终添加 `additionalProperties: True`
                        # 用于任意字典模式
                        # 参见：https://pydantic.dev/articles/pydantic-v2-11-release#changes
                        # 如果已设置为 True，我们需要将其覆盖为 False
            or "additionalProperties" in schema
        ):
            schema["additionalProperties"] = False

                # 递归检查 'properties' 和 'items'（如果存在）
        if "anyOf" in schema:
            for sub_schema in schema["anyOf"]:
                _recursive_set_additional_properties_false(sub_schema)
                # Pydantic <2.9 在字段具有同级键（例如 'description'）时，会使用 'allOf'
        # 包裹引用的模型字段，而不是直接合并它们。
                # 原始 JSON 模式可能在 `$defs` 中保留嵌套对象，并通过 `$ref`
        # 引用；也需要遍历这些定义，使其变为严格模式。
        if "allOf" in schema:
            for sub_schema in schema["allOf"]:
                _recursive_set_additional_properties_false(sub_schema)
        if "properties" in schema:
            for sub_schema in schema["properties"].values():
                _recursive_set_additional_properties_false(sub_schema)
        if "items" in schema:
            _recursive_set_additional_properties_false(schema["items"])
                # 原始 JSON 模式可能将嵌套对象保留在 `$defs` 中，并通过
                # `$ref`; 也遍历这些定义，使其变为严格模式。
        for defs_key in ("$defs", "definitions"):
            if isinstance(schema.get(defs_key), dict):
                for sub_schema in schema[defs_key].values():
                    _recursive_set_additional_properties_false(sub_schema)

    return schema

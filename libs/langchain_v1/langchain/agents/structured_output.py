"""用于设置智能体响应格式的类型。"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, is_dataclass
from types import UnionType
from typing import (
    TYPE_CHECKING,
    Any,
    Generic,
    Literal,
    TypeVar,
    Union,
    get_args,
    get_origin,
)

from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, TypeAdapter
from typing_extensions import Self, is_typeddict

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from langchain_core.messages import AIMessage

# 支持的模式类型：Pydantic 模型、dataclasses、TypedDict、JSON 模式字典
SchemaT = TypeVar("SchemaT")

SchemaKind = Literal["pydantic", "dataclass", "typeddict", "json_schema"]


class StructuredOutputError(Exception):
    """结构化输出错误的基类。"""

    ai_message: AIMessage


class MultipleStructuredOutputsError(StructuredOutputError):
    """当模型返回多个结构化输出工具调用，但仅预期一个时抛出。"""

    def __init__(self, tool_names: list[str], ai_message: AIMessage) -> None:
        """初始化 `MultipleStructuredOutputsError`。

        Args:
            tool_names: 调用结构化输出的工具名称。
            ai_message: 包含无效多工具调用的 AI 消息。
        """
        self.tool_names = tool_names
        self.ai_message = ai_message

        super().__init__(
            "Model incorrectly returned multiple structured responses "
            f"({', '.join(tool_names)}) when only one is expected."
        )


class StructuredOutputValidationError(StructuredOutputError):
    """当结构化输出工具调用参数无法根据模式解析时抛出。"""

    def __init__(self, tool_name: str, source: Exception, ai_message: AIMessage) -> None:
        """初始化 `StructuredOutputValidationError`。

        Args:
            tool_name: 失败的工具名称。
            source: 发生的异常。
            ai_message: 包含无效结构化输出的 AI 消息。
        """
        self.tool_name = tool_name
        self.source = source
        self.ai_message = ai_message
        super().__init__(f"Failed to parse structured output for tool '{tool_name}': {source}.")


def _parse_with_schema(
    schema: type[SchemaT] | dict[str, Any], schema_kind: SchemaKind, data: dict[str, Any]
) -> SchemaT | dict[str, Any]:
    """使用任何支持的模式类型进行数据解析。

    Args:
        schema: 模式类型（Pydantic 模型、`dataclass` 或 `TypedDict`）
        schema_kind: 以下之一：'pydantic'、'dataclass'、'typeddict' 或
            'json_schema'
        data: 要解析的数据

    Returns:
        根据模式类型解析后的实例

    Raises:
        ValueError: 如果解析失败
    """
    if schema_kind == "json_schema":
                # 原始 JSON 模式没有对应的 Python 类型可供实例化。
        return data
    try:
        adapter = TypeAdapter[SchemaT](schema)
        return adapter.validate_python(data)
    except Exception as e:
        schema_name = getattr(schema, "__name__", str(schema))
        msg = f"Failed to parse data to {schema_name}: {e}"
        raise ValueError(msg) from e


@dataclass(init=False)
class _SchemaSpec(Generic[SchemaT]):
    """描述一个结构化输出模式。"""

    schema: type[SchemaT] | dict[str, Any]
    """响应的模式，可以是 Pydantic 模型、`dataclass`、`TypedDict`，
    或 JSON 模式字典。
    """

    name: str
    """模式的名称，用于工具调用。

    如果未提供，名称将为模型/dataclass/TypedDict 的类名，
    或 JSON 模式的 `title` 字段。

    若不可用，则回退到生成的名称。
    """

    description: str
    """模式的自定义描述。

    如果未提供，将使用模型的 docstring。
    """

    schema_kind: SchemaKind
    """模式的类型。"""

    json_schema: dict[str, Any]
    """与模式关联的 JSON 模式。"""

    strict: bool | None = None
    """是否强制执行模式的严格验证。"""

    def __init__(
        self,
        schema: type[SchemaT] | dict[str, Any],
        *,
        name: str | None = None,
        description: str | None = None,
        strict: bool | None = None,
    ) -> None:
        """使用模式和可选参数初始化 `SchemaSpec`。

        Args:
            schema: 要描述的模式。
            name: 模式的可选名称。
            description: 模式的可选描述。
            strict: 是否强制执行模式的严格验证。

        Raises:
            ValueError: 如果模式类型不受支持。
        """
        self.schema = schema

        if name:
            self.name = name
        elif isinstance(schema, dict):
            self.name = str(schema.get("title", f"response_format_{str(uuid.uuid4())[:4]}"))
        else:
            self.name = str(getattr(schema, "__name__", f"response_format_{str(uuid.uuid4())[:4]}"))

        self.description = description or (
            schema.get("description", "")
            if isinstance(schema, dict)
            else getattr(schema, "__doc__", None) or ""
        )

        self.strict = strict

        if isinstance(schema, dict):
            self.schema_kind = "json_schema"
            self.json_schema = schema
        elif isinstance(schema, type) and issubclass(schema, BaseModel):
            self.schema_kind = "pydantic"
            self.json_schema = schema.model_json_schema()
        elif is_dataclass(schema):
            self.schema_kind = "dataclass"
            self.json_schema = TypeAdapter(schema).json_schema()
        elif is_typeddict(schema):
            self.schema_kind = "typeddict"
            self.json_schema = TypeAdapter(schema).json_schema()
        else:
            msg = (
                f"Unsupported schema type: {type(schema)}. "
                f"Supported types: Pydantic models, dataclasses, TypedDicts, and JSON schema dicts."
            )
            raise ValueError(msg)


@dataclass(init=False)
class ToolStrategy(Generic[SchemaT]):
    """对模型响应使用工具调用策略。"""

    schema: type[SchemaT] | UnionType | dict[str, Any]
    """工具调用的模式。"""

    schema_specs: list[_SchemaSpec[Any]]
    """工具调用的模式规范。"""

    tool_message_content: str | None
    """当模型调用人工结构化输出工具时，要返回的工具消息内容。
    """

    handle_errors: (
        bool | str | type[Exception] | tuple[type[Exception], ...] | Callable[[Exception], str]
    )
    """通过 `ToolStrategy` 处理结构化输出错误的策略。

    - `True`: Catch all errors with default error template
    - `str`: Catch all errors with this custom message
    - `type[Exception]`: Only catch this exception type with default message
    - `tuple[type[Exception], ...]`: Only catch these exception types with default
        message
    - `Callable[[Exception], str]`: Custom function that returns error message
    - `False`: No retry, let exceptions propagate

    !!! warning "原始 JSON 模式字典不会被验证"
        当 `schema` 是原始 JSON 模式 `dict`（而非 Pydantic 模型、
        `dataclass` 或 `TypedDict`）时，模型的工具调用参数将直接返回
        **不经过模式验证**。

        As a result, `handle_errors` is effectively inert for dict schemas. To get
        validation and automatic retries, express the schema as a Pydantic model,
        `dataclass`, or `TypedDict` instead.
    """

    def __init__(
        self,
        schema: type[SchemaT] | UnionType | dict[str, Any],
        *,
        tool_message_content: str | None = None,
        handle_errors: bool
        | str
        | type[Exception]
        | tuple[type[Exception], ...]
        | Callable[[Exception], str] = True,
    ) -> None:
        """初始化 `ToolStrategy`。

        Initialize `ToolStrategy` with schemas, tool message content, and error handling
        strategy.
        """
        self.schema = schema
        self.tool_message_content = tool_message_content
        self.handle_errors = handle_errors

        def _iter_variants(schema: Any) -> Iterable[Any]:
            """从 Union 和 JSON Schema oneOf 中生成叶级变体。"""
            if get_origin(schema) in {UnionType, Union}:
                for arg in get_args(schema):
                    yield from _iter_variants(arg)
                return

            if isinstance(schema, dict) and "oneOf" in schema:
                for sub in schema.get("oneOf", []):
                    yield from _iter_variants(sub)
                return

            yield schema

        self.schema_specs = [_SchemaSpec(s) for s in _iter_variants(schema)]


@dataclass(init=False)
class ProviderStrategy(Generic[SchemaT]):
    """使用模型提供商原生的结构化输出方法。"""

    schema: type[SchemaT] | dict[str, Any]
    """原生模式的模式。"""

    schema_spec: _SchemaSpec[SchemaT]
    """原生模式的模式规范。"""

    def __init__(
        self,
        schema: type[SchemaT] | dict[str, Any],
        *,
        strict: bool | None = None,
    ) -> None:
        """使用模式初始化 `ProviderStrategy`。

        Args:
            schema: 通过提供方原生结构化输出强制执行的模式。
            strict: 是否请求提供方端的严格模式强制。
        """
        self.schema = schema
        self.schema_spec = _SchemaSpec(schema, strict=strict)

    def to_model_kwargs(self) -> dict[str, Any]:
        """转换为关键字参数以绑定到模型，强制结构化输出。

        Returns:
            绑定到模型的关键字参数。
        """
                # OpenAI：
                # - 参见 https://platform.openai.com/docs/guides/structured-outputs
        json_schema: dict[str, Any] = {
            "name": self.schema_spec.name,
            "schema": self.schema_spec.json_schema,
        }
        if self.schema_spec.strict:
            json_schema["strict"] = True

        response_format: dict[str, Any] = {
            "type": "json_schema",
            "json_schema": json_schema,
        }
        return {"response_format": response_format}


@dataclass
class OutputToolBinding(Generic[SchemaT]):
    """用于追踪结构化输出工具元数据的信息。

    此信息包含处理通过工具调用生成的结构化响应所需的所有必要信息，包括原始模式、其类型分类，以及工具策略所使用的对应工具实现。
    """

    schema: type[SchemaT] | dict[str, Any]
    """用于结构化输出提供的原始模式（Pydantic 模型、dataclass、
    TypedDict 或 JSON 模式字典）。
    """

    schema_kind: SchemaKind
    """对模式类型进行分类，以正确构建响应。"""

    tool: BaseTool
    """从模型绑定的模式创建 LangChain 工具实例。"""

    @classmethod
    def from_schema_spec(cls, schema_spec: _SchemaSpec[SchemaT]) -> Self:
        """从 `SchemaSpec` 创建一个 `OutputToolBinding` 实例。

        Args:
            schema_spec: 要转换的 `SchemaSpec`

        Returns:
            具有适当工具的 `OutputToolBinding` 实例
        """
        return cls(
            schema=schema_spec.schema,
            schema_kind=schema_spec.schema_kind,
            tool=StructuredTool(
                args_schema=schema_spec.json_schema,
                name=schema_spec.name,
                description=schema_spec.description,
            ),
        )

    def parse(self, tool_args: dict[str, Any]) -> SchemaT | dict[str, Any]:
        """根据模式解析工具参数。

        Args:
            tool_args: 来自工具调用的参数

        Returns:
            根据模式类型解析后的响应

        Raises:
            ValueError: 如果解析失败
        """
        return _parse_with_schema(self.schema, self.schema_kind, tool_args)


@dataclass
class ProviderStrategyBinding(Generic[SchemaT]):
    """用于追踪原生结构化输出元数据的信息。

    This contains all necessary information to handle structured responses generated via
    native provider output, including the original schema, its type classification, and
    parsing logic for provider-enforced JSON.
    """

    schema: type[SchemaT] | dict[str, Any]
    """用于结构化输出的原始模式（Pydantic 模型、`dataclass`、
    `TypedDict` 或 JSON 模式字典）。
    """

    schema_kind: SchemaKind
    """对模式类型进行分类，以正确构建响应。"""

    @classmethod
    def from_schema_spec(cls, schema_spec: _SchemaSpec[SchemaT]) -> Self:
        """从 `SchemaSpec` 创建一个 `ProviderStrategyBinding` 实例。

        Args:
            schema_spec: 要转换的 `SchemaSpec`

        Returns:
            用于解析原生结构化输出的 `ProviderStrategyBinding` 实例
        """
        return cls(
            schema=schema_spec.schema,
            schema_kind=schema_spec.schema_kind,
        )

    def parse(self, response: AIMessage) -> SchemaT | dict[str, Any]:
        """根据模式解析 `AIMessage` 内容。

        Args:
            response: 包含结构化输出的 `AIMessage`

        Returns:
            根据模式解析后的响应

        Raises:
            ValueError: 如果文本提取、JSON 解析或模式验证失败
        """
                # 从 AIMessage 提取文本内容并解析为 JSON
        raw_text = self._extract_text_content_from_message(response)

        try:
            data = json.loads(raw_text)
        except Exception as e:
            schema_name = getattr(self.schema, "__name__", "response_format")
            msg = (
                f"Native structured output expected valid JSON for {schema_name}, "
                f"but parsing failed: {e}."
            )
            raise ValueError(msg) from e

                # 根据模式进行解析
        return _parse_with_schema(self.schema, self.schema_kind, data)

    @staticmethod
    def _extract_text_content_from_message(message: AIMessage) -> str:
        """从 `AIMessage` 中提取文本内容。

        Args:
            message: 要提取文本的 AI 消息

        Returns:
            提取的文本内容
        """
        content = message.content
        if isinstance(content, str):
            return content
        parts: list[str] = []
        for c in content:
            if isinstance(c, dict):
                if c.get("type") == "text" and "text" in c:
                    parts.append(str(c["text"]))
                elif "content" in c and isinstance(c["content"], str):
                    parts.append(c["content"])
            else:
                parts.append(str(c))
        return "".join(parts)


class AutoStrategy(Generic[SchemaT]):
    """自动选择最佳的结构化输出策略。"""

    schema: type[SchemaT] | dict[str, Any]
    """自动模式的模式。"""

    def __init__(
        self,
        schema: type[SchemaT] | dict[str, Any],
    ) -> None:
        """使用模式初始化 `AutoStrategy`。"""
        self.schema = schema


ResponseFormat = ToolStrategy[SchemaT] | ProviderStrategy[SchemaT] | AutoStrategy[SchemaT]
"""Union type for all supported response format strategies."""

"""结构化工具。"""

from __future__ import annotations

import functools
import textwrap
from collections.abc import Awaitable, Callable
from typing import (
    TYPE_CHECKING,
    Annotated,
    Any,
    Literal,
)

from pydantic import Field, PlainSerializer, SkipValidation
from typing_extensions import override

# Cannot move to TYPE_CHECKING as _run/_arun parameter annotations are needed at runtime
from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,  # noqa: TC001
    CallbackManagerForToolRun,  # noqa: TC001
)
from langchain_core.runnables import RunnableConfig, run_in_executor
from langchain_core.tools.base import (
    _EMPTY_SET,
    FILTERED_ARGS,
    ArgsSchema,
    BaseTool,
    _get_injected_args_keys_from_signature,
    _get_runnable_config_param,
    create_schema_from_function,
)
from langchain_core.utils._signature import _signature_parameters
from langchain_core.utils.pydantic import is_basemodel_subclass, model_json_schema

if TYPE_CHECKING:
    from langchain_core.messages import ToolCall
    from langchain_core.utils.pydantic import TypeBaseModel


def _serialize_as_str(value: Any) -> str:
    """将一个没有 JSON 形式的值转换为字符串。

    A plain function rather than the `str` builtin: Pydantic < 2.9 inspects the
    serializer's signature, and `inspect.signature` raises for C builtins.
    """
    return str(value)


@functools.lru_cache(maxsize=256)
def _model_json_schema(args_schema: TypeBaseModel, _rebuild_token: object) -> Any:
    """向 v1 或 v2 模型类请求其 JSON 模式，或回退到其 repr 表示。

    Cached because pydantic does not memoize this per class and it dominates the
    dump -- 1.10 ms against 0.02 ms for an eight-tool payload -- which tracing
    pays on every run.

    `_rebuild_token` is the class's validator, which `model_rebuild()` replaces.
    Keying on it retires the entry for a rebuilt schema, and for one whose
    forward reference was still unresolved when it was first dumped. Mutating a
    class in place without rebuilding it is not detected.
    """
    try:
        return model_json_schema(args_schema)
    except Exception:  # a schema holding an arbitrary type has no JSON schema
        return str(args_schema)


def _serialize_args_schema(args_schema: ArgsSchema) -> Any:
    """将模式类表示为其 JSON 模式，用于序列化为 JSON。

    A Pydantic model class has no JSON form, so leaving it to the default
    serializer raises `PydanticSerializationError`. Its own JSON schema is the
    shape a dict schema already has, and a dict is returned unchanged.
    """
    if isinstance(args_schema, dict):
        return args_schema
    return _model_json_schema(
        args_schema, getattr(args_schema, "__pydantic_validator__", None)
    )


# Attached to the fields via `Annotated` rather than declared with
# `@field_serializer`, which would take the field's one serializer slot and make
# any subclass that declares its own serializer for the same field fail at class
# creation with `PydanticUserError: Multiple field serializer functions ...`.
_JsonSchemaFallback = PlainSerializer(
    _serialize_args_schema, when_used="json-unless-none"
)
_JsonCallableFallback = PlainSerializer(_serialize_as_str, when_used="json-unless-none")


class StructuredTool(BaseTool):
    """可在任意数量输入上操作的工具。"""

    description: str = ""

    args_schema: Annotated[ArgsSchema, SkipValidation(), _JsonSchemaFallback] = Field(
        ..., description="The tool schema."
    )
    """输入参数的模式。"""

    func: Annotated[Callable[..., Any] | None, _JsonCallableFallback] = None
    """调用工具时要运行的函数。"""

    coroutine: Annotated[
        Callable[..., Awaitable[Any]] | None, _JsonCallableFallback
    ] = None
    """函数的异步版本。"""

        # --- Runnable ---

        # TODO: Is this needed?
    @override
    async def ainvoke(
        self,
        input: str | dict[str, Any] | ToolCall,
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> Any:
        if not self.coroutine:
                        # If the tool does not implement async, fall back to default implementation
            return await run_in_executor(config, self.invoke, input, config, **kwargs)

        return await super().ainvoke(input, config, **kwargs)

        # --- Tool ---

    def _run(
        self,
        *args: Any,
        config: RunnableConfig,
        run_manager: CallbackManagerForToolRun | None = None,
        **kwargs: Any,
    ) -> Any:
        """使用该工具。

        Args:
            *args: 传递给工具的位置参数
            config: 运行配置
            run_manager: 可选的回调管理器，用于本次运行
            **kwargs: 传递给工具的关键字参数

        Returns:
            工具执行的结果
        """
        if self.func:
            if run_manager and _signature_parameters(self.func).get("callbacks"):
                kwargs["callbacks"] = run_manager.get_child()
            if config_param := _get_runnable_config_param(self.func):
                kwargs[config_param] = config
            return self.func(*args, **kwargs)
        msg = "StructuredTool does not support sync invocation."
        raise NotImplementedError(msg)

    async def _arun(
        self,
        *args: Any,
        config: RunnableConfig,
        run_manager: AsyncCallbackManagerForToolRun | None = None,
        **kwargs: Any,
    ) -> Any:
        """异步使用该工具。

        Args:
            *args: 传递给工具的位置参数
            config: 运行配置
            run_manager: 可选的回调管理器，用于本次运行
            **kwargs: 传递给工具的关键字参数

        Returns:
            工具执行的结果
        """
        if self.coroutine:
            if run_manager and _signature_parameters(self.coroutine).get("callbacks"):
                kwargs["callbacks"] = run_manager.get_child()
            if config_param := _get_runnable_config_param(self.coroutine):
                kwargs[config_param] = config
            return await self.coroutine(*args, **kwargs)

                # If self.coroutine is None, then this will delegate to the default
                # implementation which is expected to delegate to _run on a separate thread.
        return await super()._arun(
            *args, config=config, run_manager=run_manager, **kwargs
        )

    @classmethod
    def from_function(
        cls,
        func: Callable[..., Any] | None = None,
        coroutine: Callable[..., Awaitable[Any]] | None = None,
        name: str | None = None,
        description: str | None = None,
        return_direct: bool = False,  # noqa: FBT001,FBT002
        args_schema: ArgsSchema | None = None,
        infer_schema: bool = True,  # noqa: FBT001,FBT002
        *,
        response_format: Literal["content", "content_and_artifact"] = "content",
        parse_docstring: bool = False,
        error_on_invalid_docstring: bool = False,
        **kwargs: Any,
    ) -> StructuredTool:
        """从给定函数创建工具。

        帮助从函数创建工具的类方法。

        Args:
            func: 从中创建工具的函数。
            coroutine: 从中创建工具的异步函数。
            name: 工具的名称。

                默认为函数名。
            description: 工具的描述。

                默认为函数文档字符串。
            return_direct: 是否直接返回结果，还是作为回调返回。
            args_schema: 工具输入参数的模式。
            infer_schema: 是否从函数签名推断模式。
            response_format: 工具响应格式。

                If `'content'` then the output of the tool is interpreted as the
                contents of a `ToolMessage`. If `'content_and_artifact'` then the output
                is expected to be a two-tuple corresponding to the `(content, artifact)`
                of a `ToolMessage`.
            parse_docstring: If `infer_schema` and `parse_docstring`, will attempt
                to parse parameter descriptions from Google Style function docstrings.
            error_on_invalid_docstring: if `parse_docstring` is provided, configure
                whether to raise `ValueError` on invalid Google Style docstrings.
            **kwargs: Additional arguments to pass to the tool

        Returns:
            工具。

        Raises:
            ValueError: 若未提供函数。 ValueError: 若函数无文档字符串且未提供
            description。 TypeError: 若 `args_schema` 不是 `BaseModel` 或 dict。

        Examples:
            ```python
            def add(a: int, b: int) -> int:
                \"\"\"Add two numbers\"\"\"
                return a + b
            tool = StructuredTool.from_function(add)
            tool.run(1, 2) # 3

            ```
        """
        if func is not None:
            source_function = func
        elif coroutine is not None:
            source_function = coroutine
        else:
            msg = "Function and/or coroutine must be provided"
            raise ValueError(msg)
        name = name or source_function.__name__
        if args_schema is None and infer_schema:
                        # schema name is appended within function
            args_schema = create_schema_from_function(
                name,
                source_function,
                parse_docstring=parse_docstring,
                error_on_invalid_docstring=error_on_invalid_docstring,
                filter_args=_filter_schema_args(source_function),
            )
        description_ = description
        if description is None and not parse_docstring:
            description_ = source_function.__doc__ or None
        if description_ is None and args_schema:
            if isinstance(args_schema, type) and is_basemodel_subclass(args_schema):
                description_ = args_schema.__doc__
                if (
                    description_
                    and "A base class for creating Pydantic models" in description_
                ):
                    description_ = ""
                elif not description_:
                    description_ = None
            elif isinstance(args_schema, dict):
                description_ = args_schema.get("description")
            else:
                msg = (
                    "Invalid args_schema: expected BaseModel or dict, "
                    f"got {args_schema}"
                )
                raise TypeError(msg)
        if description_ is None:
            msg = "Function must have a docstring if description not provided."
            raise ValueError(msg)
        if description is None:
                        # Only apply if using the function's docstring
            description_ = textwrap.dedent(description_).strip()

                # Description example:
                # search_api(query: str) - Searches the API for the query.
        description_ = f"{description_.strip()}"
        return cls(
            name=name,
            func=func,
            coroutine=coroutine,
            args_schema=args_schema,
            description=description_,
            return_direct=return_direct,
            response_format=response_format,
            **kwargs,
        )

    @functools.cached_property
    def _injected_args_keys(self) -> frozenset[str]:
        fn = self.func or self.coroutine
        if fn is None:
            return _EMPTY_SET
        return _get_injected_args_keys_from_signature(fn)


def _filter_schema_args(func: Callable[..., Any]) -> list[str]:
    filter_args = list(FILTERED_ARGS)
    if config_param := _get_runnable_config_param(func):
        filter_args.append(config_param)
        # filter_args.extend(_get_non_model_params(type_hints))
    return filter_args

"""将函数和 Runnable 转换为工具。"""

import inspect
from collections.abc import Callable
from typing import Any, Literal, cast, get_type_hints, overload

from pydantic import BaseModel, Field, RootModel, create_model

from langchain_core.callbacks import Callbacks
from langchain_core.runnables import Runnable
from langchain_core.tools.base import ArgsSchema, BaseTool
from langchain_core.tools.simple import Tool
from langchain_core.tools.structured import StructuredTool
from langchain_core.utils.pydantic import TypeBaseModel


@overload
def tool(
    *,
    description: str | None = None,
    return_direct: bool = False,
    args_schema: ArgsSchema | None = None,
    infer_schema: bool = True,
    response_format: Literal["content", "content_and_artifact"] = "content",
    parse_docstring: bool = False,
    error_on_invalid_docstring: bool = True,
    extras: dict[str, Any] | None = None,
) -> Callable[[Callable[..., Any] | Runnable[Any, Any]], BaseTool]: ...


@overload
def tool(
    name_or_callable: str,
    runnable: Runnable[Any, Any],
    *,
    description: str | None = None,
    return_direct: bool = False,
    args_schema: ArgsSchema | None = None,
    infer_schema: bool = True,
    response_format: Literal["content", "content_and_artifact"] = "content",
    parse_docstring: bool = False,
    error_on_invalid_docstring: bool = True,
    extras: dict[str, Any] | None = None,
) -> BaseTool: ...


@overload
def tool(
    name_or_callable: Callable[..., Any],
    *,
    description: str | None = None,
    return_direct: bool = False,
    args_schema: ArgsSchema | None = None,
    infer_schema: bool = True,
    response_format: Literal["content", "content_and_artifact"] = "content",
    parse_docstring: bool = False,
    error_on_invalid_docstring: bool = True,
    extras: dict[str, Any] | None = None,
) -> BaseTool: ...


@overload
def tool(
    name_or_callable: str,
    *,
    description: str | None = None,
    return_direct: bool = False,
    args_schema: ArgsSchema | None = None,
    infer_schema: bool = True,
    response_format: Literal["content", "content_and_artifact"] = "content",
    parse_docstring: bool = False,
    error_on_invalid_docstring: bool = True,
    extras: dict[str, Any] | None = None,
) -> Callable[[Callable[..., Any] | Runnable[Any, Any]], BaseTool]: ...


def tool(
    name_or_callable: str | Callable[..., Any] | None = None,
    runnable: Runnable[Any, Any] | None = None,
    *args: Any,
    description: str | None = None,
    return_direct: bool = False,
    args_schema: ArgsSchema | None = None,
    infer_schema: bool = True,
    response_format: Literal["content", "content_and_artifact"] = "content",
    parse_docstring: bool = False,
    error_on_invalid_docstring: bool = True,
    extras: dict[str, Any] | None = None,
) -> BaseTool | Callable[[Callable[..., Any] | Runnable[Any, Any]], BaseTool]:
    """将 Python 函数和 `Runnables` 转换为 LangChain 工具。

    可以作为装饰器使用，带参数或不带参数，从函数创建工具。

    函数可以具有任意签名——除非禁用，否则工具会自动推断输入模式。

    !!! note "要求"

        - Functions should have type hints for proper schema inference.
        - Functions may accept multiple arguments and return types are flexible;
            outputs will be serialized if needed.
        - When using with `Runnable`, a string name must be provided.

    !!! warning "保留的参数名称"

        Avoid naming tool arguments `config`, `run_manager`, or `callbacks`. These
        collide with values LangChain injects into tools at call time. A `config`
        argument is shadowed by the injected `RunnableConfig`, so the tool fails
        at call time with `TypeError: ... missing 1 required positional argument:
        'config'`, while `run_manager` and `callbacks` are silently dropped from
        the generated schema. To access runtime information (state, context, store,
        config), add a `ToolRuntime` parameter instead.

    Args:
        name_or_callable: 工具的可选名称或待转换为工具的 `Callable`。

            覆盖函数的名称。

            必须作为位置参数提供。
        runnable: 待转换为工具的可选 `Runnable`。

            必须作为位置参数提供。
        description: 工具的可选描述。

            工具描述值的优先级如下：

            - This `description` argument (used even if docstring and/or `args_schema`
                are provided)
            - Tool function docstring (used even if `args_schema` is provided)
            - `args_schema` description (used only if `description` and docstring are
                not provided)
        *args: Extra positional arguments.

            Must be empty.
        return_direct: Whether to return directly from the tool rather than continuing
            the agent loop.
        args_schema: Optional argument schema for user to specify.
        infer_schema: Whether to infer the schema of the arguments from the function's
            signature.

            此选项也会使结果工具接受字典输入到其 `run()` 函数中。 response_format:
            工具的响应格式。

            若为 `'content'`，则工具输出被解释为 `ToolMessage` 的内容。

            If `'content_and_artifact'`, then the output is expected to be a two-tuple
            corresponding to the `(content, artifact)` of a `ToolMessage`.
        parse_docstring: If `infer_schema` and `parse_docstring`, will attempt to
            parse parameter descriptions from Google Style function docstrings.
        error_on_invalid_docstring: If `parse_docstring` is provided, configure
            whether to raise `ValueError` on invalid Google Style docstrings.
        extras: Optional provider-specific extra fields for the tool.

            用于传递无法放入标准工具字段中的配置。
            聊天模型应在构建模型负载时处理已知的 extras。

            !!! example

                例如，Anthropic 特有的字段如 `cache_control`、
                `defer_loading` 或 `input_examples`。

    Raises:
        ValueError: If too many positional arguments are provided (e.g. violating the
            `*args` constraint).
        ValueError: If a `Runnable` is provided without a string name. When using `tool`
            with a `Runnable`, a `str` name must be provided as the `name_or_callable`.
        ValueError: If the first argument is not a string or callable with
            a `__name__` attribute.
        ValueError: If the function does not have a docstring and description
            is not provided and `infer_schema` is `False`.
        ValueError: If `parse_docstring` is `True` and the function has an invalid
            Google-style docstring and `error_on_invalid_docstring` is True.
        ValueError: If a `Runnable` is provided that does not have an object schema.

    Returns:
        工具。

    Examples:
        ```python
        @tool
        def search_api(query: str) -> str:
            # Searches the API for the query.
            return


        @tool("search", return_direct=True)
        def search_api(query: str) -> str:
            # Searches the API for the query.
            return


        @tool(response_format="content_and_artifact")
        def search_api(query: str) -> tuple[str, dict]:
            return "partial json of results", {"full": "object of results"}
        ```

        解析 Google 风格文档字符串：

        ```python
        @tool(parse_docstring=True)
        def foo(bar: str, baz: int) -> str:
            \"\"\"The foo.

            Args:
                bar: The bar.
                baz: The baz.
            \"\"\"
            return bar

        foo.args_schema.model_json_schema()
        ```

        ```python
        {
            "title": "foo",
            "description": "The foo.",
            "type": "object",
            "properties": {
                "bar": {
                    "title": "Bar",
                    "description": "The bar.",
                    "type": "string",
                },
                "baz": {
                    "title": "Baz",
                    "description": "The baz.",
                    "type": "integer",
                },
            },
            "required": ["bar", "baz"],
        }
        ```

        Note that parsing by default will raise `ValueError` if the docstring is
        considered invalid. A docstring is considered invalid if it contains arguments
        not in the function signature, or is unable to be parsed into a summary and
        `'Args:'` blocks. Examples below:

        ```python
        # No args section
        def invalid_docstring_1(bar: str, baz: int) -> str:
            \"\"\"The foo.\"\"\"
            return bar

        # Improper whitespace between summary and args section
        def invalid_docstring_2(bar: str, baz: int) -> str:
            \"\"\"The foo.
            Args:
                bar: The bar.
                baz: The baz.
            \"\"\"
            return bar

        # Documented args absent from function signature
        def invalid_docstring_3(bar: str, baz: int) -> str:
            \"\"\"The foo.

            Args:
                banana: The bar.
                monkey: The baz.
            \"\"\"
            return bar

        ```
    """  # noqa: D214, D410, D411  # We're intentionally showing bad formatting in examples

    def _create_tool_factory(
        tool_name: str,
    ) -> Callable[[Callable[..., Any] | Runnable[Any, Any]], BaseTool]:
        """创建一个装饰器，接收一个可调用对象并返回一个工具。

        Args:
            tool_name: 将分配给工具的名称。

        Returns:
            一个接收可调用对象或 `Runnable` 并返回工具的函数。
        """

        def _tool_factory(
            dec_func: Callable[..., Any] | Runnable[Any, Any],
        ) -> BaseTool:
            tool_description = description
            if isinstance(dec_func, Runnable):
                runnable = dec_func

                if runnable.get_input_jsonschema().get("type") != "object":
                    msg = "Runnable must have an object schema."
                    raise ValueError(msg)

                async def ainvoke_wrapper(
                    callbacks: Callbacks | None = None, **kwargs: Any
                ) -> Any:
                    return await runnable.ainvoke(kwargs, {"callbacks": callbacks})

                def invoke_wrapper(
                    callbacks: Callbacks | None = None, **kwargs: Any
                ) -> Any:
                    return runnable.invoke(kwargs, {"callbacks": callbacks})

                coroutine = ainvoke_wrapper
                func = invoke_wrapper
                schema: ArgsSchema | None = runnable.input_schema
                tool_description = description or repr(runnable)
            elif inspect.iscoroutinefunction(dec_func):
                coroutine = dec_func
                func = None
                schema = args_schema
            else:
                coroutine = None
                func = dec_func
                schema = args_schema

            if infer_schema or args_schema is not None:
                return StructuredTool.from_function(
                    func,
                    coroutine,
                    name=tool_name,
                    description=tool_description,
                    return_direct=return_direct,
                    args_schema=schema,
                    infer_schema=infer_schema,
                    response_format=response_format,
                    parse_docstring=parse_docstring,
                    error_on_invalid_docstring=error_on_invalid_docstring,
                    extras=extras,
                )
                        # 如果有人不希望应用模式，我们必须将其视为
                        # 一个简单的 string->string 函数
            tool_description = tool_description or dec_func.__doc__
            if tool_description is None:
                msg = (
                    "Function must have either a docstring or description "
                    "when infer_schema is False."
                )
                raise ValueError(msg)
            return Tool(
                name=tool_name,
                func=func,
                description=tool_description,
                return_direct=return_direct,
                coroutine=coroutine,
                response_format=response_format,
                extras=extras,
            )

        return _tool_factory

    if len(args) != 0:
                # 在用户尝试使用函数签名中不存在的位置参数时触发
                # 例如，@tool("name", runnable, "extra_arg")
                # 此处，"extra_arg" 不是有效参数
                # 工具用作函数
        msg = "Too many arguments for tool decorator. A decorator "
        raise ValueError(msg)

    if runnable is not None:
                # tool 用作函数
                # 例如 tool_from_runnable = tool("name", runnable)
        if not name_or_callable:
            msg = "Runnable without name for tool constructor"
            raise ValueError(msg)
        if not isinstance(name_or_callable, str):
            msg = "Name must be a string for tool constructor"
            raise ValueError(msg)
        return _create_tool_factory(name_or_callable)(runnable)
    if name_or_callable is not None:
        if callable(name_or_callable) and hasattr(name_or_callable, "__name__"):
                        # 用作无参数的装饰器
                        # @tool
                        # def my_tool():
            #    pass
            return _create_tool_factory(name_or_callable.__name__)(name_or_callable)
        if isinstance(name_or_callable, str):
                        # 为工具指定新名称时使用
                        # @tool("search")
                        # def my_tool():
            #    pass
            #
            # or
            #
                        # @tool("search", parse_docstring=True)
                        # def my_tool():
            #    pass
            return _create_tool_factory(name_or_callable)
        msg = (
            f"The first argument must be a string or a callable with a __name__ "
            f"for tool decorator. Got {type(name_or_callable)}"
        )
        raise ValueError(msg)

        # 作为带参数的装饰器使用
        # @tool(parse_docstring=True)
        # def my_tool():
    #    pass
    def _partial(func: Callable[..., Any] | Runnable[Any, Any]) -> BaseTool:
        """部分函数，接收一个 `Callable` 并返回一个工具。"""
        name_ = func.get_name() if isinstance(func, Runnable) else func.__name__
        tool_factory = _create_tool_factory(name_)
        return tool_factory(func)

    return _partial


def _get_description_from_runnable(runnable: Runnable[Any, Any]) -> str:
    """生成 `Runnable` 的占位描述。"""
    input_schema = runnable.get_input_jsonschema()
    return f"Takes {input_schema}."


def _get_schema_from_runnable_and_arg_types(
    runnable: Runnable[Any, Any],
    name: str,
    arg_types: dict[str, type] | None = None,
) -> type[BaseModel]:
    """推断工具的 `args_schema`。"""
    if arg_types is None:
        try:
            arg_types = get_type_hints(runnable.InputType)
        except TypeError as e:
            msg = (
                "Tool input must be str or dict. If dict, dict arguments must be "
                "typed. Either annotate types (e.g., with TypedDict) or pass "
                f"arg_types into `.as_tool` to specify. {e}"
            )
            raise TypeError(msg) from e
    fields = {key: (key_type, Field(...)) for key, key_type in arg_types.items()}
    return cast("type[BaseModel]", create_model(name, **fields))  # type: ignore[call-overload]


def convert_runnable_to_tool(
    runnable: Runnable[Any, Any],
    args_schema: TypeBaseModel | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
    arg_types: dict[str, type] | None = None,
) -> BaseTool:
    """将 `Runnable` 转换为 `BaseTool`。

    Args:
        runnable: 要转换的 `Runnable`。
        args_schema: 工具输入参数的模式。
        name: 工具的名称。
        description: 工具的描述。
        arg_types: 参数的类型。

    Returns:
        工具。
    """
    if args_schema:
        runnable = runnable.with_types(input_type=args_schema)
    description = description or _get_description_from_runnable(runnable)
    name = name or runnable.get_name()

    schema = runnable.get_input_jsonschema()
    if schema.get("type") == "string":
        return Tool(
            name=name,
            func=runnable.invoke,
            coroutine=runnable.ainvoke,
            description=description,
        )

    async def ainvoke_wrapper(callbacks: Callbacks | None = None, **kwargs: Any) -> Any:
        return await runnable.ainvoke(kwargs, config={"callbacks": callbacks})

    def invoke_wrapper(callbacks: Callbacks | None = None, **kwargs: Any) -> Any:
        return runnable.invoke(kwargs, config={"callbacks": callbacks})

    input_schema_cls = runnable.input_schema
        # 检测 `RootModel` 输入模式，避免直接将其用作工具
        # args schema.
    is_root_model = isinstance(input_schema_cls, type) and issubclass(
        input_schema_cls, RootModel
    )
    if (
        arg_types is None
        and schema.get("type") == "object"
        and schema.get("properties")
        and not is_root_model
    ):
        args_schema = input_schema_cls
    else:
        args_schema = _get_schema_from_runnable_and_arg_types(
            runnable, name, arg_types=arg_types
        )

    return StructuredTool.from_function(
        name=name,
        func=invoke_wrapper,
        coroutine=ainvoke_wrapper,
        description=description,
        args_schema=args_schema,
    )

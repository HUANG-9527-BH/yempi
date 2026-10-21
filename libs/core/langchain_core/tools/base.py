"""LangChain 工具的基础类和实用工具。"""

from __future__ import annotations

import functools
import inspect
import json
import logging
import typing
import warnings
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
from typing import (
    TYPE_CHECKING,
    Annotated,
    Any,
    Literal,
    TypeVar,
    cast,
    get_args,
    get_origin,
)

import typing_extensions
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    PydanticDeprecationWarning,
    SkipValidation,
    ValidationError,
    validate_arguments,
)
from pydantic.fields import FieldInfo
from pydantic.v1 import BaseModel as BaseModelV1
from pydantic.v1 import ValidationError as ValidationErrorV1
from pydantic.v1 import validate_arguments as validate_arguments_v1
from typing_extensions import Self, get_type_hints, override

from langchain_core.callbacks import (
    AsyncCallbackManager,
    CallbackManager,
    Callbacks,
)
from langchain_core.messages.tool import ToolCall, ToolMessage, ToolOutputMixin
from langchain_core.runnables import (
    RunnableConfig,
    RunnableSerializable,
    ensure_config,
    patch_config,
    run_in_executor,
)
from langchain_core.runnables.config import set_config_context
from langchain_core.runnables.utils import coro_with_context
from langchain_core.utils._signature import _signature_parameters
from langchain_core.utils.function_calling import (
    _parse_google_docstring,
    _py_38_safe_origin,
)
from langchain_core.utils.pydantic import (
    TypeBaseModel,
    _create_subset_model,
    get_fields,
    is_basemodel_subclass,
    is_pydantic_v1_subclass,
    is_pydantic_v2_subclass,
    model_json_schema,
)

if TYPE_CHECKING:
    import uuid

FILTERED_ARGS = ("run_manager", "callbacks")
TOOL_MESSAGE_BLOCK_TYPES = (
    "text",
    "image_url",
    "image",
    "json",
    "search_result",
    "custom_tool_call_output",
    "document",
    "file",
)

_logger = logging.getLogger(__name__)


class SchemaAnnotationError(TypeError):
    """当 `args_schema` 缺失或类型注解不正确时抛出的异常。"""


def _is_annotated_type(typ: type[Any]) -> bool:
    """检查一个类型是否为 `Annotated` 类型。

    Args:
        typ: 要检查的类型。

    Returns:
        `True` 表示该类型是 `Annotated` 类型，否则为 `False`。
    """
    return get_origin(typ) in {typing.Annotated, typing_extensions.Annotated}


def _get_annotation_description(arg_type: type) -> str | None:
    """从 `Annotated` 类型中提取描述信息。

    检查字符串注解和带有描述的 `FieldInfo` 对象。

    Args:
        arg_type: 要提取描述的类型。

    Returns:
        如果找到描述则返回描述字符串，否则返回 `None`。
    """
    if _is_annotated_type(arg_type):
        annotated_args = get_args(arg_type)
        for annotation in annotated_args[1:]:
            if isinstance(annotation, str):
                return annotation
            if isinstance(annotation, FieldInfo) and annotation.description:
                return annotation.description
    return None


def _parse_python_function_docstring(
    function: Callable[..., Any],
    annotations: dict[str, Any],
    *,
    error_on_invalid_docstring: bool = False,
) -> tuple[str, dict[str, str]]:
    """从 docstring 中解析函数和参数的描述。

    假设函数 docstring 符合 Google Python 风格指南。

    Args:
        function: 要解析 docstring 的函数。
        annotations: 函数参数的类型注解。
        error_on_invalid_docstring: 在 docstring 无效时是否抛出错误。

    Returns:
        包含函数描述和参数描述的元组。
    """
    docstring = inspect.getdoc(function)
    return _parse_google_docstring(
        docstring,
        list(annotations),
        error_on_invalid_docstring=error_on_invalid_docstring,
    )


def _validate_docstring_args_against_annotations(
    arg_descriptions: dict[str, str], annotations: dict[str, Any]
) -> None:
    """验证 docstring 中的参数是否与函数注解匹配。

    Args:
        arg_descriptions: 在 docstring 中描述的参数。
        annotations: 来自函数签名的类型注解。

    Raises:
        ValueError: 如果 docstring 中的参数在函数签名中未找到。
    """
    for docstring_arg in arg_descriptions:
        if docstring_arg not in annotations:
            msg = f"Arg {docstring_arg} in docstring not found in function signature."
            raise ValueError(msg)


def _infer_arg_descriptions(
    fn: Callable[..., Any],
    *,
    parse_docstring: bool = False,
    error_on_invalid_docstring: bool = False,
) -> tuple[str, dict[str, str]]:
    """根据函数 docstring 和注解推断参数描述。

    Args:
        fn: 用于推断描述的函数。
        parse_docstring: 是否解析 docstring 以获取描述。
        error_on_invalid_docstring: 在 docstring 无效时是否抛出错误。

    Returns:
        包含函数描述和参数描述的元组。
    """
    annotations = get_type_hints(fn, include_extras=True)
    if parse_docstring:
        description, arg_descriptions = _parse_python_function_docstring(
            fn, annotations, error_on_invalid_docstring=error_on_invalid_docstring
        )
    else:
        description = inspect.getdoc(fn) or ""
        arg_descriptions = {}
    if parse_docstring:
        _validate_docstring_args_against_annotations(arg_descriptions, annotations)
    for arg, arg_type in annotations.items():
        if arg in arg_descriptions:
            continue
        if desc := _get_annotation_description(arg_type):
            arg_descriptions[arg] = desc
    return description, arg_descriptions


def _is_pydantic_annotation(annotation: Any, pydantic_version: str = "v2") -> bool:
    """检查类型注解是否为 Pydantic 模型。

    Args:
        annotation: 要检查的类型注解。
        pydantic_version: 要检查的 Pydantic 版本（`'v1'` 或 `'v2'`）。

    Returns:
        `True` 表示注解是 Pydantic 模型，否则为 `False`。
    """
    base_model_class = BaseModelV1 if pydantic_version == "v1" else BaseModel
    try:
        return issubclass(annotation, base_model_class)
    except TypeError:
        return False


def _function_annotations_are_pydantic_v1(
    signature: inspect.Signature, func: Callable[..., Any]
) -> bool:
    """检查函数中所有 Pydantic 注解是否均来自 v1。

    Args:
        signature: 要检查的函数签名。
        func: 正在检查的函数。

    Returns:
        如果所有 Pydantic 注解均来自 v1 则返回 `True`，否则返回 `False`。

    Raises:
        NotImplementedError: 如果函数包含 v1 和 v2 注解混合的情况。
    """
    any_v1_annotations = any(
        _is_pydantic_annotation(parameter.annotation, pydantic_version="v1")
        for parameter in signature.parameters.values()
    )
    any_v2_annotations = any(
        _is_pydantic_annotation(parameter.annotation, pydantic_version="v2")
        for parameter in signature.parameters.values()
    )
    if any_v1_annotations and any_v2_annotations:
        msg = (
            f"Function {func} contains a mix of Pydantic v1 and v2 annotations. "
            "Only one version of Pydantic annotations per function is supported."
        )
        raise NotImplementedError(msg)
    return any_v1_annotations and not any_v2_annotations


class _SchemaConfig:
    """从函数签名生成的 Pydantic 模型的配置。"""

    extra: str = "forbid"
    """是否允许模型中存在额外字段。"""

    arbitrary_types_allowed: bool = True
    """是否允许模型中存在任意类型。"""


def create_schema_from_function(
    model_name: str,
    func: Callable[..., Any],
    *,
    filter_args: Sequence[str] | None = None,
    parse_docstring: bool = False,
    error_on_invalid_docstring: bool = False,
    include_injected: bool = True,
) -> TypeBaseModel:
    """从函数签名创建 Pydantic 模式。

    Args:
        model_name: Name to assign to the generated Pydantic schema.
        func: Function to generate the schema from.
        filter_args: Optional list of arguments to exclude from the schema.

            Defaults to `FILTERED_ARGS`.
        parse_docstring: Whether to parse the function's docstring for descriptions
            for each argument.
        error_on_invalid_docstring: If `parse_docstring` is provided, configure
            whether to raise `ValueError` on invalid Google Style docstrings.
        include_injected: Whether to include injected arguments in the schema.

            默认为 `True`，因为在 *验证* 工具输入时我们希望将其包含在模式中。

    Returns:
        与函数参数相同的 Pydantic 模型。
    """
    sig = inspect.signature(func)

    if _function_annotations_are_pydantic_v1(sig, func):
        validated = validate_arguments_v1(func, config=_SchemaConfig)  # type: ignore[call-overload]
    else:
                # https://docs.pydantic.dev/latest/usage/validation_decorator/
        with warnings.catch_warnings():
                        # 此处使用了已弃用的功能。
                        # 此代码应重写为直接使用 inspect.signature 和 create_model 构造 Pydantic
            # 模型
                        # 让我们忽略类和实例方法中的 `self` 和 `cls` 参数
            warnings.simplefilter("ignore", category=PydanticDeprecationWarning)
            validated = validate_arguments(func, config=_SchemaConfig)  # type: ignore[operator]

        # 如果限定名称包含 "."，则很可能属于类命名空间
        # 复制以避免修改调用者的序列
    in_class = bool(func.__qualname__ and "." in func.__qualname__)

    has_args = False
    has_kwargs = False

    for param in sig.parameters.values():
        if param.kind == param.VAR_POSITIONAL:
            has_args = True
        elif param.kind == param.VAR_KEYWORD:
            has_kwargs = True

    inferred_model = validated.model

    if filter_args:
                # 处理类方法和实例方法
        filter_args_ = list(filter_args)
    else:
                # 无论是否提供了 `filter_args`，均排除注入的参数。注入的参数（如
        # `InjectedToolArg`、`ToolRuntime`）在运行时提供，而非由模型生成，因此不应出现在生成的模式中。
        existing_params: list[str] = list(sig.parameters.keys())
        if existing_params and existing_params[0] in {"self", "cls"} and in_class:
            filter_args_ = [existing_params[0], *list(FILTERED_ARGS)]
        else:
            filter_args_ = list(FILTERED_ARGS)

        # Pydantic 添加了占位虚拟字段，需要移除
        # 内部 pydantic 字段
        # 对常见错误注解抛出错误
        # TODO: 使用 get_args / get_origin 并完全
    if not include_injected:
        for existing_param in sig.parameters:
            if (
                _is_injected_arg_type(sig.parameters[existing_param].annotation)
                and existing_param not in filter_args_
            ):
                filter_args_.append(existing_param)

    description, arg_descriptions = _infer_arg_descriptions(
        func,
        parse_docstring=parse_docstring,
        error_on_invalid_docstring=error_on_invalid_docstring,
    )
        # 指定有效的注解。
    valid_properties = []
    for field in get_fields(inferred_model):
        if not has_args and field == "args":
            continue
        if not has_kwargs and field == "kwargs":
            continue

        if field == "v__duplicate_kwargs":  # 适应正向引用未在模型构造期间解析的情况
            continue

        if field not in filter_args_:
            valid_properties.append(field)

    return _create_subset_model(
        model_name,
        inferred_model,
        list(valid_properties),
        descriptions=arg_descriptions,
        fn_description=description,
    )


class ToolException(Exception):  # noqa: N818
    """工具执行错误时抛出的异常。

    此异常允许工具在不中断智能体的情况下传递错误信息。

    错误根据工具的 `handle_tool_error` 设置进行处理，并作为观察结果返回给智能体。
    """


ArgsSchema = TypeBaseModel | dict[str, Any]
MessageContentBlock = str | dict[str, Any]
"""A single message content block: plain text or a structured block.

A dict block is only considered valid at runtime when its `type` key is one of
`TOOL_MESSAGE_BLOCK_TYPES` (see `_is_message_content_block`); the static type
intentionally stays broad because block payloads vary by provider format.
"""
ToolExceptionHandlerOutput = str | Sequence[MessageContentBlock]
"""Content returned by a `handle_tool_error` callable.

Error handlers may return plain text or a sequence of structured message
content blocks. When the original tool call includes a `tool_call_id`, this
content is normalized to the content of a `ToolMessage` with `status="error"`.
"""

_EMPTY_SET: frozenset[str] = frozenset()


_TOOL_CALL_SCHEMA_FIELDS = frozenset({"name", "description", "args_schema"})
"""Fields the memoized `tool_call_schema` is built from; reassignment clears it."""


def _patch_json_schema_cache(model_cls: type) -> None:
    """修补 `model_json_schema`（或 pydantic v1 的 `schema`）以实现缓存。

    Pydantic regenerates the full JSON-schema dict on every
    `model_json_schema()` call — there is no per-class cache.  When the
    model class is stable (memoized on a `BaseTool` instance), this patch
    caches the dict on the class so repeated calls return instantly.

    Only calls with all-default arguments are cached; any explicit arguments
    bypass the cache and delegate to the original method.
    """
    method_name = (
        "model_json_schema" if hasattr(model_cls, "model_json_schema") else "schema"
    )
    orig = getattr(model_cls, method_name)

    def _cached_json_schema(cls: type, *args: Any, **kwargs: Any) -> dict[str, Any]:
        if not args and not kwargs:
            cached = cls.__dict__.get("_json_schema_cache")
            if cached is not None:
                return cast("dict[str, Any]", cached)
        result = orig(*args, **kwargs)
        if not args and not kwargs:
            cls._json_schema_cache = result  # type: ignore[attr-defined]
        return cast("dict[str, Any]", result)

    setattr(model_cls, method_name, classmethod(_cached_json_schema))


class BaseTool(RunnableSerializable[str | dict[str, Any] | ToolCall, Any]):
    """所有 LangChain 工具的基类。

    此抽象类定义了所有 LangChain 工具必须实现的接口。

    工具是可由智能体调用以执行特定操作的组件。
    """

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """在子类创建时验证工具类定义。

        Args:
            **kwargs: 传递给父类的额外关键字参数。

        Raises:
            SchemaAnnotationError: 如果 `args_schema` 具有不正确的类型注解。
        """
        super().__init_subclass__(**kwargs)

        args_schema_type = cls.__annotations__.get("args_schema", None)

        if args_schema_type is not None and args_schema_type == BaseModel:
                        # 在检查时，若模型模式不完整则立即失败，以确保底层序列化模式不会将工具 JSON
            # 模式中的属性缩小为空字典
                        # 检查工具的 `_run`（对于仅异步的子类则回退到
            # `_arun`）以识别直接注入的参数，如 `ToolRuntime` 或标注为
            # `InjectedToolArg`
            # 的参数。这些参数在调用时提供，而非由模型生成，因此必须从模式中排除，并在执行时重新注入。`StructuredTool`
            # 会覆盖此行为，改而检查其封装的 `func`/`coroutine`。
                        # --- Runnable ---
            typehint_mandate = """
class ChildTool(BaseTool):
    ...
    args_schema: Type[BaseModel] = SchemaClass
    ..."""
            name = cls.__name__
            msg = (
                f"Tool definition for {name} must include valid type annotations"
                f" for argument 'args_schema' to behave as expected.\n"
                f"Expected annotation of 'Type[BaseModel]'"
                f" but got '{args_schema_type}'.\n"
                f"Expected class looks like:\n"
                f"{typehint_mandate}"
            )
            raise SchemaAnnotationError(msg)

    name: str
    """清晰传达工具用途的唯一名称。"""

    description: str
    """用于告知模型如何/何时/为何使用该工具。

    您可以在描述中提供少量示例。
    """

    args_schema: Annotated[ArgsSchema | None, SkipValidation()] = Field(
        default=None, description="The tool schema."
    )
    """用于验证和解析工具输入参数的 Pydantic 模型类。

    参数模式应为：

    - `pydantic.BaseModel` 的子类。
    - 若在 pydantic 2 中访问 v1 命名空间，则为 `pydantic.v1.BaseModel` 的子类
    - JSON 模式字典
    """

    return_direct: bool = False
    """是否直接返回工具的输出。

    设置为 `True` 表示工具调用后，`AgentExecutor` 将停止循环。
    """

    verbose: bool = False
    """是否记录工具的进度。"""

    callbacks: Callbacks = Field(default=None, exclude=True)
    """在工具执行期间调用的回调。"""

    tags: list[str] | None = None
    """与工具关联的可选标签列表。

    这些标签将与对该工具的每次调用关联，
    并作为参数传递给 `callbacks` 中定义的处理器。

    You can use these to, e.g., identify a specific instance of a tool with its use
    case.
    """

    metadata: dict[str, Any] | None = None
    """与工具关联的可选元数据。

    此元数据将与对该工具的每次调用关联，
    并作为参数传递给 `callbacks` 中定义的处理器。

    您可以使用它们来识别特定工具实例及其使用场景。
    """

    handle_tool_error: (
        bool | str | Callable[[ToolException], ToolExceptionHandlerOutput] | None
    ) = False
    """处理工具执行过程中引发的 `ToolException`。

    If `False`, the exception is re-raised. If `True`, the exception message is
    returned as tool output. If a string is passed, that string is returned
    as tool output. If a callable is passed, it receives the exception and
    its return value is used as the tool output.

    Callable handlers may return either a string or a list of message
    content blocks. If the tool was invoked with a `tool_call_id`, the handled
    content is wrapped in a `ToolMessage` with `status="error"`.
    """

    handle_validation_error: (
        bool | str | Callable[[ValidationError | ValidationErrorV1], str] | None
    ) = False
    """处理 `ValidationError` 抛出的内容。"""

    response_format: Literal["content", "content_and_artifact"] = "content"
    """工具响应格式。

    如果为 `'content'`，则工具的输出被解释为 `ToolMessage` 的内容。如果为
    `'content_and_artifact'`，则输出预期为一个二元组，对应 `ToolMessage` 的 `(content,
    artifact)`。
    """

    extras: dict[str, Any] | None = None
    """可选的、与提供方相关的额外字段，用于工具。

    用于传递不适用于标准工具字段的提供方特定配置。

    Example:
        Anthropic 特有的字段，如
        [`cache_control`](https://docs.langchain.com/oss/python/integrations/chat/anthropic#prompt-caching)、[`defer_loading`](https://docs.langchain.com/oss/python/integrations/chat/anthropic#tool-search)
        或 `input_examples`。

        ```python
        @tool(extras={"defer_loading": True, "cache_control": {"type": "ephemeral"}})
        def my_tool(x: str) -> str:
            return x
        ```
    """

    def __init__(self, **kwargs: Any) -> None:
        """初始化工具。

        Raises:
            TypeError: 如果 `args_schema` 不是 pydantic `BaseModel` 的子类或 `dict`。
        """
        if (
            "args_schema" in kwargs
            and kwargs["args_schema"] is not None
            and not is_basemodel_subclass(kwargs["args_schema"])
            and not isinstance(kwargs["args_schema"], dict)
        ):
            msg = (
                "args_schema must be a subclass of pydantic BaseModel or "
                f"a JSON schema dict. Got: {kwargs['args_schema']}."
            )
            raise TypeError(msg)
        super().__init__(**kwargs)

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )

    @property
    def is_single_input(self) -> bool:
        """检查工具是否仅接受单个输入参数。

        Returns:
            如果工具只有一个输入参数，则为 `True`，否则为 `False`。
        """
        keys = {k for k in self.args if k != "kwargs"}
        return len(keys) == 1

    @property
    def args(self) -> dict[str, Any]:
        """获取工具的输入参数模式。

        Returns:
            包含工具参数属性的 `dict`。
        """
        if isinstance(self.args_schema, dict):
            json_schema = self.args_schema
        else:
            input_schema = self.tool_call_schema
            if isinstance(input_schema, dict):
                json_schema = input_schema
            else:
                json_schema = model_json_schema(input_schema)
        return cast("dict[str, Any]", json_schema["properties"])

    _tool_call_schema_memo: ArgsSchema | None = PrivateAttr(default=None)
    """已缓存的 `tool_call_schema` 结果。

    Building the subset model is expensive, and pydantic does not cache
    `model_json_schema()` per class, so agent loops would otherwise pay full
    schema generation for every tool on every model call. The subset model
    class is memoized here and its `model_json_schema`/`schema` method is
    patched to cache the generated dict, so both costs are paid only once per
    tool instance.
    Cleared whenever `name`, `description`, or `args_schema` is reassigned (see
    `__setattr__` and `model_copy`).
    """

    @override
    def __setattr__(self, name: str, value: Any) -> None:
        """当工具的输入被重新赋值时，清除工具调用模式的缓存。"""
        super().__setattr__(name, value)
        if name in _TOOL_CALL_SCHEMA_FIELDS and self.__pydantic_private__ is not None:
            self._tool_call_schema_memo = None

    @override
    def model_copy(
        self, *, update: Mapping[str, Any] | None = None, deep: bool = False
    ) -> Self:
        """复制工具，如果 `update` 影响了模式，则清除模式缓存。

        `model_copy` 直接将 `update` 写入副本的 `__dict__`，而不经过
        `__setattr__`，且私有属性（包括缓存）会传递到副本，因此当 `update`
        触及模式构建所依赖的字段时，此处会清除缓存。
        """
        copied = super().model_copy(update=update, deep=deep)
        if update and not _TOOL_CALL_SCHEMA_FIELDS.isdisjoint(update):
            copied._tool_call_schema_memo = None  # noqa: SLF001
        return copied

    def __getstate__(self) -> dict[Any, Any]:
        """在序列化时丢弃工具调用模式的缓存。

        The memoized subset model is a dynamically created class that cannot be
        pickled by reference; it is rebuilt lazily on next access.
        """
        state = super().__getstate__()
        private = state.get("__pydantic_private__")
        if private and private.get("_tool_call_schema_memo") is not None:
            state = dict(state)
            state["__pydantic_private__"] = {
                **private,
                "_tool_call_schema_memo": None,
            }
        return state

    @property
    def tool_call_schema(self) -> ArgsSchema:
        """获取工具调用的模式，排除注入的参数。

        Returns:
            应用于语言模型工具调用的模式。

            The returned model class is memoized per tool instance (invalidated
            when `name`, `description`, or `args_schema` is reassigned) so
            repeated access does not regenerate the class. The class's
            `model_json_schema` method is also patched to cache the generated
            schema dict, since pydantic does not cache it per class.
        """
        if isinstance(self.args_schema, dict):
            if self.description:
                return {
                    **self.args_schema,
                    "description": self.description,
                }

            return self.args_schema

        if (memo := self._tool_call_schema_memo) is not None:
            return memo

        full_schema = self.get_input_schema()
        fields = []

                # --- Tool ---
                # 检查 args_schema 中是否存在 InjectedToolCallId
                # 检查 args_schema 中是否存在 InjectedToolCallId
                # 包含 tool_input 中的字段、通过 Pydantic
        # 别名提供的字段，以及具有显式默认值的字段。这会应用 Pydantic 默认值（如
        # Field(default=1)），同时排除 Pydantic 为 *args/**kwargs 创建的合成
        # "args"/"kwargs" 字段。
        if (
            is_pydantic_v2_subclass(full_schema)
            and not full_schema.__pydantic_complete__
        ):
            full_schema.model_rebuild()

        for name, type_ in get_all_basemodel_annotations(full_schema).items():
            if not _is_injected_arg_type(type_):
                fields.append(name)
        subset_model = _create_subset_model(
            self.name, full_schema, fields, fn_description=self.description
        )
        _patch_json_schema_cache(subset_model)
        self._tool_call_schema_memo = subset_model
        return subset_model

    @functools.cached_property
    def _injected_args_keys(self) -> frozenset[str]:
                # 字段已在输入中提供——包含它（已验证）
                # 字段通过 Pydantic 别名提供——包含它。
                # 检查字段在模式中是否有显式定义的默认值。
                # 排除
        # "args"/"kwargs"，因为它们是用于可变参数的合成字段，不应作为关键字参数传递。
                # Pydantic v2 使用 is_required() 方法，v1 使用 required 属性
                # 从常量中获取过滤后的参数开始
        for method in (self._run, self._arun):
            keys = _get_injected_args_keys_from_signature(method)
            if keys:
                return keys
        return _EMPTY_SET

        # 添加来自函数签名的注入参数（例如 ToolRuntime 参数）

    @override
    def get_input_schema(self, config: RunnableConfig | None = None) -> TypeBaseModel:
        """工具的输入模式。

        Args:
            config: 工具的配置。

        Returns:
            工具的输入模式。
        """
        if self.args_schema is not None:
            if isinstance(self.args_schema, dict):
                return super().get_input_schema(config)
            return self.args_schema
        return create_schema_from_function(self.name, self._run)

    @override
    def invoke(
        self,
        input: str | dict[str, Any] | ToolCall,
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> Any:
        tool_input, kwargs = _prep_run_args(input, config, **kwargs)
        return self.run(tool_input, **kwargs)

    @override
    async def ainvoke(
        self,
        input: str | dict[str, Any] | ToolCall,
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> Any:
        tool_input, kwargs = _prep_run_args(input, config, **kwargs)
        return await self.arun(tool_input, **kwargs)

        # 如果有 args_schema，使用它来识别注入的参数

    def _parse_input(
        self, tool_input: str | dict[str, Any], tool_call_id: str | None
    ) -> str | dict[str, Any]:
        """使用 args 模式解析并验证工具输入。

        Args:
            tool_input: 工具的原始输入。
            tool_call_id: 工具调用的 ID（如果可用）。

        Returns:
            解析并验证后的输入。

        Raises:
            ValueError: If `string` input is provided with JSON schema `args_schema`.
            ValueError: If `InjectedToolCallId` is required but `tool_call_id` is not
                provided.
            TypeError: If `args_schema` is not a Pydantic `BaseModel` or dict.
        """
        input_args = self.args_schema

        if isinstance(tool_input, str):
            if input_args is not None:
                if isinstance(input_args, dict):
                    msg = (
                        "String tool inputs are not allowed when "
                        "using tools with JSON schema args_schema."
                    )
                    raise ValueError(msg)
                key_ = next(iter(get_fields(input_args).keys()))
                if issubclass(input_args, BaseModel):
                    input_args.model_validate({key_: tool_input})
                elif issubclass(input_args, BaseModelV1):
                    input_args.parse_obj({key_: tool_input})
                else:
                    msg = f"args_schema must be a Pydantic BaseModel, got {input_args}"  # type: ignore[unreachable]
                    raise TypeError(msg)
            return tool_input

        if input_args is not None:
            if isinstance(input_args, dict):
                return tool_input
            result: BaseModel | BaseModelV1
            if issubclass(input_args, BaseModel):
                                # 如果 args_schema 是字典（JSON Schema），则跳过，因为它不是 Pydantic
                # 模型
                for k, v in get_all_basemodel_annotations(input_args).items():
                    if _is_injected_arg_type(v, injected_type=InjectedToolCallId):
                        if tool_call_id is None:
                            msg = (
                                "When tool includes an InjectedToolCallId "
                                "argument, tool must always be invoked with a full "
                                "model ToolCall of the form: {'args': {...}, "
                                "'name': '...', 'type': 'tool_call', "
                                "'tool_call_id': '...'}"
                            )
                            raise ValueError(msg)
                        tool_input[k] = tool_call_id
                result_v2 = input_args.model_validate(tool_input)
                result_dict = result_v2.model_dump()
                provided_fields = result_v2.model_fields_set
                result = result_v2
            elif issubclass(input_args, BaseModelV1):
                                # 如果无法获取注解，就直接使用 FILTERED_ARGS
                for k, v in get_all_basemodel_annotations(input_args).items():
                    if _is_injected_arg_type(v, injected_type=InjectedToolCallId):
                        if tool_call_id is None:
                            msg = (
                                "When tool includes an InjectedToolCallId "
                                "argument, tool must always be invoked with a full "
                                "model ToolCall of the form: {'args': {...}, "
                                "'name': '...', 'type': 'tool_call', "
                                "'tool_call_id': '...'}"
                            )
                            raise ValueError(msg)
                        tool_input[k] = tool_call_id
                result_v1 = input_args.parse_obj(tool_input)
                result_dict = result_v1.dict()
                provided_fields = result_v1.__fields_set__
                result = result_v1
            else:
                msg = (  # type: ignore[unreachable]
                    f"args_schema must be a Pydantic BaseModel, got {self.args_schema}"
                )
                raise NotImplementedError(msg)

                        # 从 tool_input 中过滤掉注入的键
                        # 无参数的 StructuredTool
                        # 为向后兼容，如果 run_input 是字符串，
                        # 则作为位置参数传递。
            field_info = get_fields(input_args)
            validated_input = {}
            for k in result_dict:
                if k in tool_input:
                                        # 对输入进行浅拷贝，以便下游代码可以修改输入的根级别而不影响原始输入。
                    validated_input[k] = getattr(result, k)
                elif k in provided_fields:
                                        # 此用于工具在运行时注入信息，例如回调管理器。
                    validated_input[k] = getattr(result, k)
                elif k in field_info and k not in {"args", "kwargs"}:
                                        # 此代码路径预期不可达。
                                        # 排除 "args"/"kwargs"，因为它们是可变参数的合成字段
                                        # 不应作为关键字参数传递的参数。
                    fi = field_info[k]
                                        # Pydantic v2 使用 is_required() 方法，v1 使用 required 属性
                    has_default = (
                        not fi.is_required()
                        if hasattr(fi, "is_required")
                        else not getattr(fi, "required", True)
                    )
                    if has_default:
                        validated_input[k] = getattr(result, k)

            for k in self._injected_args_keys:
                if k in tool_input:
                    validated_input[k] = tool_input[k]
                elif k == "tool_call_id":
                    if tool_call_id is None:
                        msg = (
                            "When tool includes an InjectedToolCallId "
                            "argument, tool must always be invoked with a full "
                            "model ToolCall of the form: {'args': {...}, "
                            "'name': '...', 'type': 'tool_call', "
                            "'tool_call_id': '...'}"
                        )
                        raise ValueError(msg)
                    validated_input[k] = tool_call_id

            return validated_input

        return tool_input

    @abstractmethod
    def _run(self, *args: Any, **kwargs: Any) -> Any:
        """使用该工具。

        Add `run_manager: CallbackManagerForToolRun | None = None` to child
        implementations to enable tracing.

        Returns:
            工具执行的结果。
        """

    async def _arun(self, *args: Any, **kwargs: Any) -> Any:
        """异步使用该工具。

        Add `run_manager: AsyncCallbackManagerForToolRun | None = None` to child
        implementations to enable tracing.

        Returns:
            工具执行的结果。
        """
        if kwargs.get("run_manager") and _signature_parameters(self._run).get(
            "run_manager"
        ):
            kwargs["run_manager"] = kwargs["run_manager"].get_sync()
        return await run_in_executor(None, self._run, *args, **kwargs)

    def _filter_injected_args(self, tool_input: dict[str, Any]) -> dict[str, Any]:
        """从输入字典中过滤掉注入的工具参数。

        Injected arguments are those annotated with `InjectedToolArg` or its
        subclasses, or arguments in `FILTERED_ARGS` like `run_manager` and callbacks.

        Args:
            tool_input: 需要过滤的工具输入字典。

        Returns:
            过滤后的字典，已移除注入的参数。
        """
                # 从常量中获取过滤后的参数开始
        filtered_keys = set[str](FILTERED_ARGS)

                # 从函数签名中注入参数（例如，ToolRuntime 参数）
        filtered_keys.update(self._injected_args_keys)

                # 如果有 args_schema，就用它来识别注入的参数
                # 如果 args_schema 是 dict（JSON Schema），则跳过，因为它不是 Pydantic 模型
        if self.args_schema is not None and not isinstance(self.args_schema, dict):
            try:
                annotations = get_all_basemodel_annotations(self.args_schema)
                for field_name, field_type in annotations.items():
                    if _is_injected_arg_type(field_type):
                        filtered_keys.add(field_name)
            except Exception:
                                # 如果无法获取注解，就使用 FILTERED_ARGS
                _logger.debug(
                    "Failed to get args_schema annotations for filtering.",
                    exc_info=True,
                )

                # 过滤掉注入的键（keys）从 tool_input 中
        return {k: v for k, v in tool_input.items() if k not in filtered_keys}

    def _to_args_and_kwargs(
        self, tool_input: str | dict[str, Any], tool_call_id: str | None
    ) -> tuple[tuple[str, ...], dict[str, Any]]:
        """将工具输入转换为位置参数和关键字参数。

        Args:
            tool_input: 工具的输入。
            tool_call_id: 工具调用的 ID（如果可用）。

        Returns:
            工具所需的位置参数和关键字参数组成的元组 `(positional_args, keyword_args)`。

        Raises:
            TypeError: 如果工具输入类型无效。
        """
        if (
            self.args_schema is not None
            and isinstance(self.args_schema, type)
            and is_basemodel_subclass(self.args_schema)
            and not get_fields(self.args_schema)
        ):
                        # 带无参数的 StructuredTool
            return (), {}
        tool_input = self._parse_input(tool_input, tool_call_id)
                # 为保持向后兼容性，如果 run_input 是字符串，则
                # 作为位置参数传递。
        if isinstance(tool_input, str):
            return (tool_input,), {}
        if isinstance(tool_input, dict):
                        # 创建输入的浅拷贝，以便下游代码使用
                        # 修改输入的根级别，而不影响
                        # 原始输入。
                        # 此用于工具注入运行时信息，例如
                        # 回调管理器。
            return (), tool_input.copy()
                # 此代码路径预期不可达。
        msg = f"Invalid tool input type: {type(tool_input)}"  # type: ignore[unreachable]
        raise TypeError(msg)

    def run(
        self,
        tool_input: str | dict[str, Any],
        verbose: bool | None = None,  # noqa: FBT001
        start_color: str | None = "green",
        color: str | None = "green",
        callbacks: Callbacks = None,
        *,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        run_name: str | None = None,
        run_id: uuid.UUID | None = None,
        config: RunnableConfig | None = None,
        tool_call_id: str | None = None,
        **kwargs: Any,
    ) -> Any:
        """运行该工具。

        Args:
            tool_input: 工具的输入。
            verbose: 是否记录工具的执行进度。
            start_color: 开始工具时使用的颜色。
            color: 结束工具时使用的颜色。
            callbacks: 执行过程中调用的回调。
            tags: 与工具关联的可选标签列表。
            metadata: 与工具关联的可选元数据。
            run_name: 运行的名称。
            run_id: 运行的 ID。
            config: 工具的配置。
            tool_call_id: 工具调用的 ID。
            **kwargs: 传递给工具回调（事件处理器）的关键字参数

        Returns:
            工具的输出。

        Raises:
            ToolException: 如果工具执行过程中发生错误。
        """
        callback_manager = CallbackManager.configure(
            callbacks,
            self.callbacks,
            self.verbose or bool(verbose),
            tags,
            self.tags,
            metadata,
            self.metadata,
        )

                # 过滤掉回调输入中的注入参数
        filtered_tool_input = (
            self._filter_injected_args(tool_input)
            if isinstance(tool_input, dict)
            else None
        )

                # 同样使用过滤后的输入作为 input_str 参数
        tool_input_str = (
            tool_input
            if isinstance(tool_input, str)
            else str(
                filtered_tool_input if filtered_tool_input is not None else tool_input
            )
        )

        run_manager = callback_manager.on_tool_start(
            {"name": self.name, "description": self.description},
            tool_input_str,
            color=start_color,
            name=run_name,
            run_id=run_id,
            inputs=filtered_tool_input,
            tool_call_id=tool_call_id,
            **kwargs,
        )

        content = None
        artifact = None
        status = "success"
        error_to_raise: Exception | KeyboardInterrupt | None = None
        try:
            child_config = patch_config(config, callbacks=run_manager.get_child())
            with set_config_context(child_config) as context:
                tool_args, tool_kwargs = self._to_args_and_kwargs(
                    tool_input, tool_call_id
                )
                if _signature_parameters(self._run).get("run_manager"):
                    tool_kwargs |= {"run_manager": run_manager}
                if config_param := _get_runnable_config_param(self._run):
                    tool_kwargs |= {config_param: config}
                response = context.run(self._run, *tool_args, **tool_kwargs)
            if self.response_format == "content_and_artifact":
                msg = (
                    "Since response_format='content_and_artifact' "
                    "a two-tuple of the message content and raw tool output is "
                    f"expected. Instead, generated response is of type: "
                    f"{type(response)}."
                )
                if not isinstance(response, tuple):
                    error_to_raise = ValueError(msg)
                else:
                    try:
                        content, artifact = response
                    except ValueError:
                        error_to_raise = ValueError(msg)
            else:
                content = response
        except (ValidationError, ValidationErrorV1) as e:
            if not self.handle_validation_error:
                error_to_raise = e
            else:
                content = _handle_validation_error(e, flag=self.handle_validation_error)
                status = "error"
        except ToolException as e:
            if not self.handle_tool_error:
                error_to_raise = e
            else:
                content = _handle_tool_error(e, flag=self.handle_tool_error)
                status = "error"
        except (Exception, KeyboardInterrupt) as e:
            error_to_raise = e

        if error_to_raise:
            run_manager.on_tool_error(error_to_raise, tool_call_id=tool_call_id)
            raise error_to_raise
        output = _format_output(content, artifact, tool_call_id, self.name, status)
        run_manager.on_tool_end(output, color=color, name=self.name, **kwargs)
        return output

    async def arun(
        self,
        tool_input: str | dict[str, Any],
        verbose: bool | None = None,  # noqa: FBT001
        start_color: str | None = "green",
        color: str | None = "green",
        callbacks: Callbacks = None,
        *,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        run_name: str | None = None,
        run_id: uuid.UUID | None = None,
        config: RunnableConfig | None = None,
        tool_call_id: str | None = None,
        **kwargs: Any,
    ) -> Any:
        """异步运行该工具。

        Args:
            tool_input: 工具的输入。
            verbose: 是否记录工具的执行进度。
            start_color: 开始工具时使用的颜色。
            color: 结束工具时使用的颜色。
            callbacks: 执行过程中调用的回调。
            tags: 与工具关联的可选标签列表。
            metadata: 与工具关联的可选元数据。
            run_name: 运行的名称。
            run_id: 运行的 ID。
            config: 工具的配置。
            tool_call_id: 工具调用的 ID。
            **kwargs: 传递给工具回调的关键字参数

        Returns:
            工具的输出。

        Raises:
            ToolException: 如果工具执行过程中发生错误。
        """
        callback_manager = AsyncCallbackManager.configure(
            callbacks,
            self.callbacks,
            self.verbose or bool(verbose),
            tags,
            self.tags,
            metadata,
            self.metadata,
        )

                # 过滤掉回调输入中的注入参数
        filtered_tool_input = (
            self._filter_injected_args(tool_input)
            if isinstance(tool_input, dict)
            else None
        )

                # 同样使用过滤后的输入作为 input_str 参数
        tool_input_str = (
            tool_input
            if isinstance(tool_input, str)
            else str(
                filtered_tool_input if filtered_tool_input is not None else tool_input
            )
        )

        run_manager = await callback_manager.on_tool_start(
            {"name": self.name, "description": self.description},
            tool_input_str,
            color=start_color,
            name=run_name,
            run_id=run_id,
            inputs=filtered_tool_input,
            tool_call_id=tool_call_id,
            **kwargs,
        )
        content = None
        artifact = None
        status = "success"
        error_to_raise: Exception | KeyboardInterrupt | None = None
        try:
            tool_args, tool_kwargs = self._to_args_and_kwargs(tool_input, tool_call_id)
            child_config = patch_config(config, callbacks=run_manager.get_child())
            with set_config_context(child_config) as context:
                func_to_check = (
                    self._run if self.__class__._arun is BaseTool._arun else self._arun  # noqa: SLF001
                )
                if _signature_parameters(func_to_check).get("run_manager"):
                    tool_kwargs["run_manager"] = run_manager
                if config_param := _get_runnable_config_param(func_to_check):
                    tool_kwargs[config_param] = config

                coro = self._arun(*tool_args, **tool_kwargs)
                response = await coro_with_context(coro, context)
            if self.response_format == "content_and_artifact":
                msg = (
                    "Since response_format='content_and_artifact' "
                    "a two-tuple of the message content and raw tool output is "
                    f"expected. Instead, generated response is of type: "
                    f"{type(response)}."
                )
                if not isinstance(response, tuple):
                    error_to_raise = ValueError(msg)
                else:
                    try:
                        content, artifact = response
                    except ValueError:
                        error_to_raise = ValueError(msg)
            else:
                content = response
        except (ValidationError, ValidationErrorV1) as e:
            if not self.handle_validation_error:
                error_to_raise = e
            else:
                content = _handle_validation_error(e, flag=self.handle_validation_error)
                status = "error"
        except ToolException as e:
            if not self.handle_tool_error:
                error_to_raise = e
            else:
                content = _handle_tool_error(e, flag=self.handle_tool_error)
                status = "error"
        except (Exception, KeyboardInterrupt) as e:
            error_to_raise = e

        if error_to_raise:
            await run_manager.on_tool_error(error_to_raise, tool_call_id=tool_call_id)
            raise error_to_raise

        output = _format_output(content, artifact, tool_call_id, self.name, status)
        await run_manager.on_tool_end(output, color=color, name=self.name, **kwargs)
        return output


def _is_tool_call(x: Any) -> bool:
    """检查输入是否为工具调用字典。

    Args:
        x: 要检查的输入。

    Returns:
        如果输入是工具调用，则返回 `True`，否则返回 `False`。
    """
    return isinstance(x, dict) and x.get("type") == "tool_call"


def _handle_validation_error(
    e: ValidationError | ValidationErrorV1,
    *,
    flag: Literal[True] | str | Callable[[ValidationError | ValidationErrorV1], str],
) -> str:
    """根据配置的标志处理验证错误。

    Args:
        e: 发生的验证错误。
        flag: 如何处理错误（`bool`、`str` 或 `Callable`）。

    Returns:
        返回的错误消息。

    Raises:
        ValueError: 如果标志类型意外。
    """
    if isinstance(flag, bool):
        content = "Tool input validation error"
    elif isinstance(flag, str):
        content = flag
    elif callable(flag):
        content = flag(e)
    else:
        msg = (  # type: ignore[unreachable]
            f"Got unexpected type of `handle_validation_error`. Expected bool, "
            f"str or callable. Received: {flag}"
        )
        raise ValueError(msg)  # noqa: TRY004
    return content


def _handle_tool_error(
    e: ToolException,
    *,
    flag: Literal[True]
    | str
    | Callable[[ToolException], ToolExceptionHandlerOutput]
    | None,
) -> ToolExceptionHandlerOutput:
    """将 `ToolException` 转换为已处理的工具输出内容。

    Args:
        e: The tool exception that occurred.
        flag: How to handle the error. `True` uses the exception message, a string
            replaces the message, and a callable computes replacement content from
            the exception.

    Returns:
        已处理的错误内容。这可能是纯文本或结构化消息内容块；调用者将其通过正常的工具输出格式化流程传递。

    Raises:
        ValueError: 如果标志类型意外。
    """
    if isinstance(flag, bool):
        content = e.args[0] if e.args else "Tool execution error"
    elif isinstance(flag, str):
        content = flag
    elif callable(flag):
        content = flag(e)
    else:
        msg = (
            f"Got unexpected type of `handle_tool_error`. Expected bool, str "
            f"or callable. Received: {flag}"
        )
        raise ValueError(msg)  # noqa: TRY004
    return content


def _prep_run_args(
    value: str | dict[str, Any] | ToolCall,
    config: RunnableConfig | None,
    **kwargs: Any,
) -> tuple[str | dict[str, Any], dict[str, Any]]:
    """准备工具执行的参数。

    Args:
        value: 输入值（`str`、`dict` 或 `ToolCall`）。
        config: Runnable 配置。
        **kwargs: 其他关键字参数。

    Returns:
        一个元组 `(tool_input, run_kwargs)`。
    """
    config = ensure_config(config)
    tool_input: str | dict[str, Any]
    if _is_tool_call(value):
        tool_call_id: str | None = cast("ToolCall", value)["id"]
        tool_input = cast("ToolCall", value)["args"].copy()
    else:
        tool_call_id = None
        tool_input = cast("str | dict[str, Any]", value)
    return (
        tool_input,
        dict(
            callbacks=config.get("callbacks"),
            tags=config.get("tags"),
            metadata=config.get("metadata"),
            run_name=config.get("run_name"),
            run_id=config.pop("run_id", None),
            config=config,
            tool_call_id=tool_call_id,
            **kwargs,
        ),
    )


def _format_output(
    content: Any,
    artifact: Any,
    tool_call_id: str | None,
    name: str,
    status: str,
) -> ToolOutputMixin | Any:
    """如果合适，将工具输出格式化为 `ToolMessage`。

    Args:
        content: 工具输出的主要内容。
        artifact: 工具产生的任何附加数据。
        tool_call_id: 工具调用的 ID。
        name: 工具名称。
        status: 执行状态。

    Returns:
        格式化后的输出，可以是 `ToolMessage`、原始内容，或未更改的 `ToolOutputMixin`
        实例列表。
    """
    if (
        isinstance(content, list)
        and content
        and all(isinstance(item, ToolOutputMixin) for item in content)
    ):
        return content
    if isinstance(content, ToolOutputMixin) or tool_call_id is None:
        return content
    normalized_content = _normalize_message_content(content)
    content = _stringify(content) if normalized_content is None else normalized_content
    return ToolMessage(
        content,
        artifact=artifact,
        tool_call_id=tool_call_id,
        name=name,
        status=status,
    )


def _normalize_message_content(obj: Any) -> str | list[MessageContentBlock] | None:
    """将有效的消息内容强制转换为 `ToolMessage` 所期望的格式。

    字符串保持不变；任何有效的内容块序列（例如列表或元组）将被实例化为 `list`。返回
    `None` 表示调用者（`_format_output`）该对象不是消息内容，应转为字符串处理。

    Args:
        obj: 待规范化的对象。

    Returns:
        规范化后的内容，或当 `obj` 不是有效消息内容时为 `None`。
    """
    if isinstance(obj, str):
        return obj
        # 延迟验证：在实例化前进行验证：`all` 在遇到第一个无效元素时会短路
        # 因此，大型非内容序列（例如 `range(10**12)`）将回退为字符串化处理，而无需分配内存。
        # `object.__new__`/`object.__init__`
    # 是槽包装器，不携带任何有用的注解，因此只有继承的 Python 构造函数才符合条件。
    if isinstance(obj, Sequence) and all(_is_message_content_block(e) for e in obj):
        return list(obj)
    return None


def _is_message_content_block(obj: Any) -> bool:
    """检查对象是否为有效消息内容块。

    验证 OpenAI 或 Anthropic 格式的内容块。

    Args:
        obj: 待检查的对象。

    Returns:
        如果对象是有效内容块则返回 `True`，否则返回 `False`。
    """
    if isinstance(obj, str):
        return True
    if isinstance(obj, dict):
        return obj.get("type", None) in TOOL_MESSAGE_BLOCK_TYPES
    return False


def _stringify(content: Any) -> str:
    """将内容转换为字符串，优先使用 JSON 格式。

    Args:
        content: 待转为字符串的内容。

    Returns:
        内容的字符串表示形式。
    """
    try:
        return json.dumps(content, ensure_ascii=False)
    except Exception:
        return str(content)


def _describe_callable(func: Any) -> str:
    """返回可调用对象的可读标识符，用于日志消息中。"""
    return getattr(func, "__qualname__", None) or repr(func)


def _get_type_hints(
    func: Callable[..., Any], *, include_extras: bool = False
) -> dict[str, type] | None:
    """从函数获取类型提示，处理部分函数的情况。

    Args:
        func: 获取类型提示的函数。 include_extras: 是否在解析后的提示中保留 `Annotated`
        元数据（例如用于检测 `Annotated[..., InjectedToolArg]`）。

    Returns:
        类型提示的 `dict`，或提取失败时为 `None`。
    """
    if isinstance(func, functools.partial):
        func = func.func
    try:
        return get_type_hints(func, include_extras=include_extras)
    except Exception:
        _logger.debug(
            "Failed to resolve type hints for %s.",
            _describe_callable(func),
            exc_info=True,
        )
        return None


def _get_class_signature_source(cls: type) -> Callable[..., Any] | None:
    """返回描述 `signature(cls)` 参数的可调用对象。

    A class's own annotations describe its attributes, not its constructor
    parameters, so a class is never its own annotation owner. Constructor
    precedence mirrors `inspect.signature`: a metaclass `__call__` override,
    then a `__new__`/`__init__` the class defines itself, then an inherited one.

    Args:
        cls: 被检查的类。

    Returns:
        The constructor whose annotations describe `signature(cls)`, or `None`
        when only slot wrappers inherited from `object` are available.
    """
    metaclass = type(cls)
    if metaclass is not type:
        metaclass_call = inspect.getattr_static(metaclass, "__call__", None)
        if inspect.isfunction(metaclass_call):
            return metaclass_call
    for attr in ("__new__", "__init__"):
        if attr in cls.__dict__:
            return cast("Callable[..., Any]", getattr(cls, attr))
    for attr in ("__new__", "__init__"):
                # 如果未指定注入类型，则
                # 检查该类型是否为直接注入的参数
        inherited = getattr(cls, attr, None)
        if inspect.isfunction(inherited):
            return cast("Callable[..., Any]", inherited)
    return None


def _get_type_hints_source(
    func: Callable[..., Any],
) -> Callable[..., Any] | None:
    """返回拥有有效签名注解的可调用对象。

    解包过程在可调用对象声明自身
    `__signature__`（或为绑定方法）时立即停止，因为该声明——而非其所包装对象的注解——定义了有效签名。

    Args:
        func: 被检查的可调用对象。

    Returns:
        描述 `signature(func)` 注解的函数或方法，或当显式 `__signature__`
        提供注解，或无法识别注解所有者时为 `None`。
    """
    func = inspect.unwrap(
        func,
        stop=lambda wrapped: (
            hasattr(wrapped, "__signature__") or inspect.ismethod(wrapped)
        ),
    )
    if inspect.ismethod(func):
        return _get_type_hints_source(func.__func__)
    if getattr(func, "__signature__", None) is not None:
        return None
    if isinstance(func, functools.partial):
        return _get_type_hints_source(func.func)
    if inspect.isclass(func):
        constructor = _get_class_signature_source(func)
        if constructor is None:
            return None
        return _get_type_hints_source(constructor)
    if not inspect.isroutine(func):
        callable_obj = cast("Any", func)
        return _get_type_hints_source(cast("Callable[..., Any]", callable_obj.__call__))
    return func


def _get_callable_globals(func: Callable[..., Any]) -> dict[str, Any]:
    """返回与可调用对象关联的全局命名空间。

    Explicit `__signature__` objects store annotations separately from a
    callable's `__annotations__`. String annotations on such signatures
    still need the callable's defining globals to be resolved.

    Args:
        func: 需要定位定义全局命名空间的可调用对象。

    Returns:
        可调用对象的全局命名空间，或当无法定位时为空的 `dict`（例如类、内置函数及其他 C
        实现的可调用对象），此时字符串注解将无法解析。
    """
    if isinstance(func, functools.partial):
        return _get_callable_globals(func.func)
    if inspect.ismethod(func):
        return _get_callable_globals(func.__func__)
    globalns = getattr(func, "__globals__", None)
    if isinstance(globalns, dict):
        return globalns
    if not inspect.isroutine(func) and not inspect.isclass(func):
        callable_obj = cast("Any", func)
        return _get_callable_globals(cast("Callable[..., Any]", callable_obj.__call__))
    return {}


def _get_injected_args_keys_from_signature(func: Callable[..., Any]) -> frozenset[str]:
    """识别可调用对象的注入参数。

    Resolve annotations with `get_type_hints` (rather than reading raw
    `signature` annotations) so postponed annotations -- e.g. from
    `from __future__ import annotations` or quoted forward references -- are
    recognized. `get_type_hints` resolves every annotation at once and raises
    on the first failure, so any parameter it leaves uncovered is retried on its
    own: an unrelated unresolvable forward reference must not disable injection
    for parameters whose annotations do resolve. Each distinct string annotation
    is resolved at most once per call, failures included.

    一个 *字符串*
    注解，其命名的类型仅存在于可调用对象定义的局部作用域中，无法在此处解析，因为仅可用全局命名空间。此类参数保持其原始字符串注解，而
    `_is_injected_arg_type`
    永远不会将其分类为已注入，因此它对模型保持可见，而非在调用时注入。非字符串注解不受影响，因为
    `signature` 返回的是实时对象。

    Args:
        func: 需要检查签名的函数（或绑定方法）。

    Returns:
        被标注为注入参数的参数名的 `frozenset`。
    """
    params = _signature_parameters(func)
    hint_source = _get_type_hints_source(func)
    hints = (
        _get_type_hints(hint_source, include_extras=True)
        if hint_source is not None
        else None
    ) or {}
    globalns: dict[str, Any] | None = None
    resolved_annotations: dict[str, Any] = {}
    keys = set()
    for name, param in params.items():
        annotation = hints.get(name, param.annotation)
        if isinstance(annotation, str):
            if globalns is None:
                globalns = _get_callable_globals(
                    func if hint_source is None else hint_source
                )
            if annotation not in resolved_annotations:
                resolved_annotations[annotation] = _resolve_forward_ref(
                    annotation, globalns
                )
            resolved = resolved_annotations[annotation]
            if resolved is None:
                _logger.debug(
                    "Could not resolve annotation %r for parameter %r of %s; it "
                    "will not be treated as an injected argument.",
                    annotation,
                    name,
                    _describe_callable(func),
                )
            else:
                annotation = resolved
        if _is_injected_arg_type(annotation):
            keys.add(name)
    return frozenset(keys)


def _resolve_forward_ref(annotation: str, globalns: dict[str, Any]) -> Any:
    """解析单个字符串注解，失败时返回 `None`。

    使用临时带注解的函数，使每个注解都能通过公开的 `get_type_hints` API 独立处理。

    Args:
        annotation: 需要解析的原始字符串注解。
        globalns: 用于解析名称的全局命名空间。

    Returns:
        解析后的类型，如果无法解析则返回 `None`。
    """

    def _annotation_holder() -> None:
        pass

    _annotation_holder.__annotations__ = {"value": annotation}
    try:
        return get_type_hints(
            _annotation_holder,
            globalns=globalns,
            include_extras=True,
        )["value"]
    except Exception:
        _logger.debug("Failed to resolve annotation %r.", annotation, exc_info=True)
        return None


def _get_runnable_config_param(func: Callable[..., Any]) -> str | None:
    """查找函数中 `RunnableConfig` 的参数名。

    Args:
        func: 需要检查的函数。

    Returns:
        `RunnableConfig` 的参数名，若未找到则返回 `None`。
    """
    type_hints = _get_type_hints(func)
    if not type_hints:
        return None
    for name, type_ in type_hints.items():
        if type_ is RunnableConfig:
            return name
    return None


class InjectedToolArg:
    """用于在运行时注入工具参数的注解。

    Tool arguments annotated with this class are not included in the tool
    schema sent to language models and are instead injected during execution.
    """


class _DirectlyInjectedToolArg:
    """用于在运行时注入工具参数的注解。

    通过直接类型注解注入，而非通过注解元数据。

    例如，`ToolRuntime` 是一个直接注入的参数。

    注意直接注解而非冗长的替代形式：
    `Annotated[ToolRuntime, InjectedRuntime]`

    ```python
    from langchain_core.tools import tool, ToolRuntime


    @tool
    def foo(x: int, runtime: ToolRuntime) -> str:
        # use runtime.state, runtime.context, runtime.store, etc.
        ...
    ```
    """


class InjectedToolCallId(InjectedToolArg):
    """用于注入工具调用 ID 的注解。

    此注解用于标记应在运行时接收工具调用 ID 的工具参数。

    ```python
    from typing import Annotated
    from langchain_core.messages import ToolMessage
    from langchain_core.tools import tool, InjectedToolCallId

    @tool
    def foo(
        x: int, tool_call_id: Annotated[str, InjectedToolCallId]
    ) -> ToolMessage:
        \"\"\"Return x.\"\"\"
        return ToolMessage(
            str(x),
            artifact=x,
            name="foo",
            tool_call_id=tool_call_id
        )
    ```
    """


def _is_directly_injected_arg_type(type_: Any) -> bool:
    """检查类型注解是否表示直接注入的参数。

    当前仅用于 `ToolRuntime`。

    Checks if either the annotation itself is a subclass of `_DirectlyInjectedToolArg`
    or the origin of the annotation is a subclass of `_DirectlyInjectedToolArg`.

    例如，`ToolRuntime` 或 `ToolRuntime[ContextT, StateT]` 均返回 `True`。
    """
    return (
        isinstance(type_, type) and issubclass(type_, _DirectlyInjectedToolArg)
    ) or (
        (origin := get_origin(type_)) is not None
        and isinstance(origin, type)
        and issubclass(origin, _DirectlyInjectedToolArg)
    )


def _is_injected_arg_type(
    type_: type | TypeVar, injected_type: type[InjectedToolArg] | None = None
) -> bool:
    """检查类型注解是否表示注入的参数。

    Args:
        type_: 需要检查的类型注解。
        injected_type: 需要检查的具体注入类型。

    Returns:
        如果类型是注入的参数，则为 `True`，否则为 `False`。
    """
    if injected_type is None:
                # 如果该类型是 Annotated 类型，则检查其注解元数据
                # 是否为注入类型的实例或子类
        if _is_directly_injected_arg_type(type_):
            return True
        injected_type = InjectedToolArg

        # cls 无下标：cls = FooBar
        # 排除 pydantic Config 添加的隐藏 init 参数。例如，若
    return any(
        isinstance(arg, injected_type)
        or (isinstance(arg, type) and issubclass(arg, injected_type))
        for arg in get_args(type_)[1:]
    )


def get_all_basemodel_annotations(
    cls: TypeBaseModel | Any, *, default_to_bound: bool = True
) -> dict[str, type | TypeVar]:
    """从 Pydantic `BaseModel` 及其父类中获取所有注解。

    Args:
        cls: Pydantic `BaseModel` 类。
        default_to_bound: 如果存在，是否默认使用 `TypeVar` 的绑定值。

    Returns:
        `dict`，字段名到其类型注解的映射。
    """
    orig_bases: tuple[type, ...]
        # BaseModel(extra="allow")，则 "extra_data" 将成为 init 签名的一部分。
    if isinstance(cls, type):
        fields = get_fields(cls)
        alias_map = {field.alias: name for name, field in fields.items() if field.alias}

        annotations: dict[str, type | TypeVar] = {}
        for name, param in inspect.signature(cls).parameters.items():
                        # cls 有下标：cls = FooBar[int]
                        # Pydantic v2 会自动解析继承的泛型，而 Pydantic v1 不会。
            if name not in fields and name not in alias_map:
                continue
            field_name = alias_map.get(name, name)
            annotations[field_name] = param.annotation
        orig_bases = getattr(cls, "__orig_bases__", ())
        # 若 cls = FooBar 继承自 Baz[str]，orig_bases 中将包含 Baz[str]
    else:
        annotations = get_all_basemodel_annotations(
            get_origin(cls), default_to_bound=False
        )
        orig_bases = (cls,)

        # 若 cls = FooBar 继承自 Baz，orig_bases 中将包含 Baz
    if not (isinstance(cls, type) and is_pydantic_v2_subclass(cls)):
                # 若 cls = FooBar[int]，orig_bases 中将包含 FooBar[int]
                # 若类 = FooBar 继承自 Baz，parent = Baz
                # 若类 = FooBar 继承自非 pydantic 类
        for parent in orig_bases:
                        # 若类 = FooBar 继承自 Baz[str]：
            if isinstance(parent, type) and is_pydantic_v1_subclass(parent):
                annotations.update(
                    get_all_basemodel_annotations(parent, default_to_bound=False)
                )
                continue

            parent_origin = get_origin(parent)

                        # parent = 类 Baz[str]，
            if not parent_origin:
                continue

                        # parent_origin = 类 Baz，
                        # generic_type_vars = (Baz 中的类型变量)
                        # generic_map = {Baz 中的类型变量: str}
                        # generic_type_vars = (Baz 中的类型变量)
                        # generic_map = {type var in Baz: str}
            generic_type_vars: tuple[TypeVar, ...] = getattr(
                parent_origin, "__parameters__", ()
            )
            generic_map = dict(zip(generic_type_vars, get_args(parent), strict=False))
            for field in getattr(parent_origin, "__annotations__", {}):
                annotations[field] = _replace_type_vars(
                    annotations[field], generic_map, default_to_bound=default_to_bound
                )

    return {
        k: _replace_type_vars(v, default_to_bound=default_to_bound)
        for k, v in annotations.items()
    }


def _replace_type_vars(
    type_: type | TypeVar,
    generic_map: dict[TypeVar, type] | None = None,
    *,
    default_to_bound: bool = True,
) -> type | TypeVar:
    """将类型注解中的 `TypeVar` 替换为具体类型。

    Args:
        type_: 需要处理的类型注解。
        generic_map: `TypeVar` 到具体类型的映射。
        default_to_bound: 是否使用 `TypeVar` 的边界作为默认值。

    Returns:
        替换 `TypeVar` 后的类型。
    """
    generic_map = generic_map or {}
    if isinstance(type_, TypeVar):
        if type_ in generic_map:
            return generic_map[type_]
        if default_to_bound:
            return type_.__bound__ if type_.__bound__ is not None else Any
        return type_
    if (origin := get_origin(type_)) and (args := get_args(type_)):
        new_args = tuple(
            _replace_type_vars(arg, generic_map, default_to_bound=default_to_bound)
            for arg in args
        )
        return cast("type", _py_38_safe_origin(origin)[new_args])  # type: ignore[index]
    return type_


class BaseToolkit(BaseModel, ABC):
    """包含相关工具的工具集的基类。

    A toolkit is a collection of related tools that can be used together to accomplish a
    specific task or work with a particular system.
    """

    @abstractmethod
    def get_tools(self) -> list[BaseTool]:
        """获取工具集中的所有工具。

        Returns:
            此工具集包含的工具列表。
        """

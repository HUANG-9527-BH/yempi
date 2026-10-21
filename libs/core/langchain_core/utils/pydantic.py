"""Pydantic 的实用工具。"""

from __future__ import annotations

import inspect
import textwrap
import warnings
from contextlib import nullcontext
from functools import lru_cache, wraps
from types import GenericAlias
from typing import (
    TYPE_CHECKING,
    Any,
    TypeGuard,
    TypeVar,
    cast,
    overload,
)

import pydantic
from packaging import version

# root_validator 已弃用，但我们需要它以保持 @pre_init 的向后兼容性
from pydantic import (  # type: ignore[deprecated]
    BaseModel,
    ConfigDict,
    Field,
    PydanticDeprecationWarning,
    RootModel,
    root_validator,
)
from pydantic import (
    create_model as _create_model_base,
)
from pydantic.fields import FieldInfo as FieldInfoV2
from pydantic.json_schema import (
    DEFAULT_REF_TEMPLATE,
    GenerateJsonSchema,
    JsonSchemaMode,
    JsonSchemaValue,
)
from pydantic.v1 import BaseModel as BaseModelV1
from pydantic.v1 import create_model as create_model_v1
from typing_extensions import deprecated, override

if TYPE_CHECKING:
    from collections.abc import Callable

    from pydantic.v1.fields import ModelField
    from pydantic_core import core_schema

PYDANTIC_VERSION = version.parse(pydantic.__version__)


@deprecated("Use PYDANTIC_VERSION.major instead.")
def get_pydantic_major_version() -> int:
    """已弃用 - 获取 Pydantic 的主版本号。

    请改用 `PYDANTIC_VERSION.major`。

    Returns:
        Pydantic 的主版本号。
    """
    return PYDANTIC_VERSION.major


PYDANTIC_MAJOR_VERSION = PYDANTIC_VERSION.major
PYDANTIC_MINOR_VERSION = PYDANTIC_VERSION.minor

IS_PYDANTIC_V1 = False
IS_PYDANTIC_V2 = True

PydanticBaseModel = BaseModel | BaseModelV1
TypeBaseModel = type[BaseModel] | type[BaseModelV1]

TBaseModel = TypeVar("TBaseModel", bound=PydanticBaseModel)


def is_pydantic_v1_subclass(cls: type) -> bool:
    """检查给定类是否为 Pydantic v1 风格。

    Returns:
        如果给定类是 Pydantic `BaseModel` 1.x 的子类，则返回 `True`。
    """
    return issubclass(cls, BaseModelV1)


def is_pydantic_v2_subclass(cls: type) -> TypeGuard[type[BaseModel]]:
    """检查给定类是否为 Pydantic v2 风格。

    Returns:
        如果给定类是 Pydantic `BaseModel` 2.x 的子类，则返回 `True`。
    """
    return issubclass(cls, BaseModel)


def is_basemodel_subclass(cls: type) -> bool:
    """检查给定类是否为 Pydantic `BaseModel` 的子类。

    检查给定类是否为以下任意一个的子类：

    * Pydantic 2.x 中的 `pydantic.BaseModel`
    * Pydantic 2.x 中的 `pydantic.v1.BaseModel`

    Returns:
        如果给定类是 Pydantic `BaseModel` 的子类，则返回 `True`。
    """
        # 在可以对 cls 使用 issubclass 之前，需要先检查它是否为类
    if not inspect.isclass(cls) or isinstance(cls, GenericAlias):
        return False

    return issubclass(cls, (BaseModel, BaseModelV1))


def is_basemodel_instance(obj: Any) -> bool:
    """检查给定类是否为 Pydantic `BaseModel` 的实例。

    检查给定类是否为以下任意一个的实例：

    * Pydantic 2.x 中的 `pydantic.BaseModel`
    * Pydantic 2.x 中的 `pydantic.v1.BaseModel`

    Returns:
        如果给定类是 Pydantic `BaseModel` 的实例，则返回 `True`。
    """
    return isinstance(obj, (BaseModel, BaseModelV1))


# 如何对此进行类型提示？
def pre_init(
    func: Callable[[Any, dict[str, Any]], Any],
) -> Callable[[Any, dict[str, Any]], Any]:
    """在模型初始化前运行函数的装饰器。

    Args:
        func: 在模型初始化前运行的函数。

    Returns:
        被装饰的函数。
    """
    with warnings.catch_warnings():
        warnings.filterwarnings(action="ignore", category=PydanticDeprecationWarning)

                # 理想情况下我们会使用 @model_validator(mode="before")，但这会改变验证器的顺序。
                # 参见 https://github.com/pydantic/pydantic/discussions/7434。
                # 因此我们保留 root_validator 以保持向后兼容性。
        @root_validator(pre=True)  # type: ignore[deprecated]
        @wraps(func)
        def wrapper(cls: type[BaseModel], values: dict[str, Any]) -> Any:
            """装饰器，在模型初始化前运行一个函数。

            Args:
                cls: 模型类。
                values: 用于初始化模型的值。

            Returns:
                用于初始化模型的值。
            """
                        # 插入默认值
            fields = cls.model_fields
            for name, field_info in fields.items():
                                # 检查是否启用了 allow_population_by_field_name
                                # 如果启用，则将字段名设置为别名
                if (
                    hasattr(cls, "Config")
                    and hasattr(cls.Config, "allow_population_by_field_name")
                    and cls.Config.allow_population_by_field_name
                    and field_info.alias in values
                ):
                    values[name] = values.pop(field_info.alias)
                if (
                    hasattr(cls, "model_config")
                    and cls.model_config.get("populate_by_name")
                    and field_info.alias in values
                ):
                    values[name] = values.pop(field_info.alias)

                if (
                    name not in values or values[name] is None
                ) and not field_info.is_required():
                    if field_info.default_factory is not None:
                        values[name] = field_info.default_factory()  # type: ignore[call-arg]
                    else:
                        values[name] = field_info.default

                        # 调用已装饰的函数
            return func(cls, values)

    return wrapper


class _IgnoreUnserializable(GenerateJsonSchema):
    """忽略未知类型的 JSON 模式生成器。

    https://docs.pydantic.dev/latest/concepts/json_schema/#customizing-the-json-schema-generation-process
    """

    @override
    def handle_invalid_for_json_schema(
        self, schema: core_schema.CoreSchema, error_info: str
    ) -> JsonSchemaValue:
        return {}


def _create_subset_model_v1(
    name: str,
    model: type[BaseModelV1],
    field_names: list[str],
    *,
    descriptions: dict[str, str] | None = None,
    fn_description: str | None = None,
) -> type[BaseModelV1]:
    """使用 Pydantic 模型的部分字段创建新模型。"""
    fields = {}

    for field_name in field_names:
                # 使用 pydantic v1，因此可以访问 __fields__ 作为字典。
        field = model.__fields__[field_name]
        t = (
                        # 这并非完美，但对大多数函数应能正常工作
            field.outer_type_
            if field.required and not field.allow_none
            else field.outer_type_ | None
        )
        if descriptions and field_name in descriptions:
            field.field_info.description = descriptions[field_name]
        fields[field_name] = (t, field.field_info)

    rtn = cast("type[BaseModelV1]", create_model_v1(name, **fields))  # type: ignore[call-overload]
    rtn.__doc__ = textwrap.dedent(fn_description or model.__doc__ or "")
    return rtn


def _create_subset_model_v2(
    name: str,
    model: type[BaseModel],
    field_names: list[str],
    *,
    descriptions: dict[str, str] | None = None,
    fn_description: str | None = None,
) -> type[BaseModel]:
    """使用模型字段的子集创建 Pydantic 模型。"""
    descriptions_ = descriptions or {}
    fields = {}
    for field_name in field_names:
        field = model.model_fields[field_name]
        description = descriptions_.get(field_name, field.description)
        field_kwargs: dict[str, Any] = {"description": description}
        if field.default_factory is not None:
            field_kwargs["default_factory"] = field.default_factory
        else:
            field_kwargs["default"] = field.default
        field_info = FieldInfoV2(**field_kwargs)
        if field.metadata:
            field_info.metadata = field.metadata
        fields[field_name] = (field.annotation, field_info)

    rtn = cast(
        "type[BaseModel]",
        _create_model_base(  # type: ignore[call-overload]
            name, **fields, __config__=ConfigDict(arbitrary_types_allowed=True)
        ),
    )

        # TODO(0.3): 确定是否有更符合 "pydantic" 的方式来保留注解。
        # 这是为了在使用 pydantic 2.x 时保留 __annotations__
        # 并使用 Annotated 类型与 TypedDict。
        # 注释掉以下行，以触发相应的测试用例。
    selected_annotations = [
        (name, annotation)
        for name, annotation in model.__annotations__.items()
        if name in field_names
    ]

    rtn.__annotations__ = dict(selected_annotations)
    rtn.__doc__ = textwrap.dedent(fn_description or model.__doc__ or "")
    return rtn


# 私有功能，用于创建一个在不同版本的 pydantic 间兼容的子模型。
# 处理 pydantic 版本 2.x，包括 2.x 中的 pydantic v1。
# 然而，无法找到为该函数添加类型提示的方法。
# 保留名称应捕获 BaseModel 内部使用的所有 `公开` 名称/方法。这将保持保留名称的最新状态。
def _create_subset_model(
    name: str,
    model: TypeBaseModel,
    field_names: list[str],
    *,
    descriptions: dict[str, str] | None = None,
    fn_description: str | None = None,
) -> TypeBaseModel:
    """使用与输入模型相同版本的 Pydantic 创建子集模型。

    Returns:
        创建的子集模型。
    """
    if issubclass(model, BaseModelV1):
        return _create_subset_model_v1(
            name,
            model,
            field_names,
            descriptions=descriptions,
            fn_description=fn_description,
        )
    return _create_subset_model_v2(
        name,
        model,
        field_names,
        descriptions=descriptions,
        fn_description=fn_description,
    )


@overload
def get_fields(model: type[BaseModel]) -> dict[str, FieldInfoV2]: ...


@overload
def get_fields(model: BaseModel) -> dict[str, FieldInfoV2]: ...


@overload
def get_fields(model: type[BaseModelV1]) -> dict[str, ModelField]: ...


@overload
def get_fields(model: BaseModelV1) -> dict[str, ModelField]: ...


def get_fields(
    model: type[BaseModel | BaseModelV1] | BaseModel | BaseModelV1,
) -> dict[str, FieldInfoV2] | dict[str, ModelField]:
    """返回 Pydantic 模型的字段名称。

    Args:
        model: Pydantic 模型或实例。

    Raises:
        TypeError: 如果模型不是 Pydantic 模型。
    """
    if not isinstance(model, type):
        model = type(model)
    if issubclass(model, BaseModel):
        return model.model_fields
    if issubclass(model, BaseModelV1):
        return model.__fields__
    msg = f"Expected a Pydantic model. Got {model}"  # type: ignore[unreachable]
    raise TypeError(msg)


def model_json_schema(model: TypeBaseModel) -> dict[str, Any]:
    """返回任意主版本的 Pydantic 模型类的 JSON 模式。

    Dispatches to the correct method for Pydantic v1 (`schema`) or v2
    (`model_json_schema`), so callers holding a `TypeBaseModel` don't have to
    branch on the model's version themselves.

    Args:
        model: Pydantic 模型类。

    Raises:
        TypeError: 如果模型不是 Pydantic 模型类。
    """
    if issubclass(model, BaseModel):
        return model.model_json_schema()
    if issubclass(model, BaseModelV1):
        return model.schema()
    msg = f"Expected a Pydantic model. Got {model}"  # type: ignore[unreachable]
    raise TypeError(msg)


def model_validate(model: TypeBaseModel, obj: Any) -> PydanticBaseModel:
    """使用任意主版本的 Pydantic 模型类验证 `obj`。

    Dispatches to the correct method for Pydantic v1 (`parse_obj`) or v2
    (`model_validate`), so callers holding a `TypeBaseModel` don't have to
    branch on the model's version themselves.

    Args:
        model: 用于验证的 Pydantic 模型类。
        obj: 待验证的对象。

    Raises:
        TypeError: 如果模型不是 Pydantic 模型类。
    """
    if issubclass(model, BaseModel):
        return model.model_validate(obj)
    if issubclass(model, BaseModelV1):
        return model.parse_obj(obj)
    msg = f"Expected a Pydantic model. Got {model}"  # type: ignore[unreachable]
    raise TypeError(msg)


_SchemaConfig = ConfigDict(
    arbitrary_types_allowed=True, frozen=True, protected_namespaces=()
)

NO_DEFAULT = object()


def _create_root_model(
    name: str,
    type_: Any,
    module_name: str | None = None,
    default_: object = NO_DEFAULT,
) -> type[BaseModel]:
    """创建基类。"""

    def schema(
        cls: type[BaseModelV1],
        by_alias: bool = True,  # noqa: FBT001,FBT002
        ref_template: str = DEFAULT_REF_TEMPLATE,
    ) -> dict[str, Any]:
        super_cls = cast("type[BaseModelV1]", super(cls, cls))
        schema_ = super_cls.schema(by_alias=by_alias, ref_template=ref_template)
        schema_["title"] = name
        return schema_

    def model_json_schema(
        cls: type[BaseModel],
        by_alias: bool = True,  # noqa: FBT001,FBT002
        ref_template: str = DEFAULT_REF_TEMPLATE,
        schema_generator: type[GenerateJsonSchema] = GenerateJsonSchema,
        mode: JsonSchemaMode = "validation",
    ) -> dict[str, Any]:
        super_cls = cast("type[BaseModel]", super(cls, cls))
        schema_ = super_cls.model_json_schema(
            by_alias=by_alias,
            ref_template=ref_template,
            schema_generator=schema_generator,
            mode=mode,
        )
        schema_["title"] = name
        return schema_

    base_class_attributes = {
        "__annotations__": {"root": type_},
        "model_config": ConfigDict(arbitrary_types_allowed=True),
        "schema": classmethod(schema),
        "model_json_schema": classmethod(model_json_schema),
        "__module__": module_name or "langchain_core.runnables.utils",
    }

    if default_ is not NO_DEFAULT:
        base_class_attributes["root"] = default_
    with warnings.catch_warnings():
        try:
            if isinstance(type_, type) and issubclass(type_, BaseModelV1):
                warnings.filterwarnings(
                    action="ignore", category=PydanticDeprecationWarning
                )
        except TypeError:
            pass
        custom_root_type = type(name, (RootModel,), base_class_attributes)
    return cast("type[BaseModel]", custom_root_type)


@lru_cache(maxsize=256)
def _create_root_model_cached(
    model_name: str,
    type_: Any,
    *,
    module_name: str | None = None,
    default_: object = NO_DEFAULT,
) -> type[BaseModel]:
    return _create_root_model(
        model_name, type_, default_=default_, module_name=module_name
    )


@lru_cache(maxsize=256)
def _create_model_cached(
    model_name: str,
    /,
    **field_definitions: Any,
) -> type[BaseModel]:
    return _create_model_base(
        model_name,
        __config__=_SchemaConfig,
        **_remap_field_definitions(field_definitions),
    )


def create_model(
    model_name: str,
    module_name: str | None = None,
    /,
    **field_definitions: Any,
) -> type[BaseModel]:
    """使用给定的字段定义创建 Pydantic 模型。

    请改用 `create_model_v2` 而非此函数。

    Args:
        model_name: 模型名称。
        module_name: 模型定义所在的模块名称。

            该名称用于 Pydantic 解析任何正向引用。
        **field_definitions: 模型的字段定义。

    Returns:
        创建的模型。
    """
    kwargs = {}
    if "__root__" in field_definitions:
        kwargs["root"] = field_definitions.pop("__root__")

    return create_model_v2(
        model_name,
        module_name=module_name,
        field_definitions=field_definitions,
        **kwargs,
    )


# 参考，保留名称包括：
# "construct", "copy", "dict", "from_orm", "json", "parse_file", "parse_obj",
# "parse_raw", "schema", "schema_json", "update_forward_refs", "validate",
# "model_computed_fields", "model_config", "model_construct", "model_copy",
# "model_dump", "model_dump_json", "model_extra", "model_fields",
# "model_fields_set", "model_json_schema", "model_parametrized_name",
# "model_post_init", "model_rebuild", "model_validate", "model_validate_json",
# "model_validate_strings"
# 让我们添加前缀以避免与内部 pydantic 字段发生冲突
# 某些参数传入 _create_root_model_cached 时不可哈希
_RESERVED_NAMES = {key for key in dir(BaseModel) if not key.startswith("_")}


def _remap_field_definitions(field_definitions: dict[str, Any]) -> dict[str, Any]:
    """此操作重新映射字段以避免与内部 pydantic 字段冲突。"""
    remapped = {}
    for key, value in field_definitions.items():
        if key.startswith("_") or key in _RESERVED_NAMES:
                        # 无根节点，仅包含字段定义
            if isinstance(value, FieldInfoV2):
                msg = (
                    f"Remapping for fields starting with '_' or fields with a name "
                    f"matching a reserved name {_RESERVED_NAMES} is not supported if "
                    f" the field is a pydantic Field instance. Got {key}."
                )
                raise NotImplementedError(msg)
            type_, default_ = value
            remapped[f"private_{key}"] = (
                type_,
                Field(
                    default=default_,
                    alias=key,
                    serialization_alias=key,
                    title=key.lstrip("_").replace("_", " ").title(),
                ),
            )
        else:
            remapped[key] = value
    return remapped


def create_model_v2(
    model_name: str,
    *,
    module_name: str | None = None,
    field_definitions: dict[str, Any] | None = None,
    root: Any | None = None,
) -> type[BaseModel]:
    """使用给定的字段定义创建 Pydantic 模型。

    !!! warning

        不要在 langchain 包以外使用。此 API 可能在任何时候更改。

    Args:
        model_name: 模型的名称。
        module_name: 模型定义所在的模块名称。

            这用于 Pydantic 解析任何向前引用。
        field_definitions: 模型的字段定义。
        root: 根模型的类型（`RootModel`）

    Returns:
        创建的模型。
    """
    field_definitions = field_definitions or {}

    if root:
        if field_definitions:
            msg = (
                "When specifying __root__ no other "
                f"fields should be provided. Got {field_definitions}"
            )
            raise NotImplementedError(msg)

        if isinstance(root, tuple):
            kwargs = {"type_": root[0], "default_": root[1]}
        else:
            kwargs = {"type_": root}

        try:
            named_root_model = _create_root_model_cached(
                model_name, module_name=module_name, **kwargs
            )
        except TypeError:
                        # 此外，如果使用了任何非保留名称（例如 model_id 或 model_name）
            named_root_model = _create_root_model(
                model_name,
                module_name=module_name,
                **kwargs,
            )
        return named_root_model

        # 字段定义中的某部分不可哈希
    names = set(field_definitions.keys())

    capture_warnings = False

    for name in names:
                # 另外，如果使用了任何非保留名称（例如 model_id 或 model_name）
        if name.startswith("model"):
            capture_warnings = True

    with warnings.catch_warnings() if capture_warnings else nullcontext():
        if capture_warnings:
            warnings.filterwarnings(action="ignore")
        try:
            return _create_model_cached(model_name, **field_definitions)
        except TypeError:
                        # 字段定义中的某些内容不可哈希
            return _create_model_base(
                model_name,
                __config__=_SchemaConfig,
                **_remap_field_definitions(field_definitions),
            )

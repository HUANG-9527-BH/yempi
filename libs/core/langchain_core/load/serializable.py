"""可序列化基类。"""

import contextlib
import logging
from abc import ABC
from typing import (
    Any,
    Literal,
    TypedDict,
    cast,
)

from pydantic import BaseModel, ConfigDict
from pydantic.fields import FieldInfo
from pydantic_core import PydanticUndefined
from typing_extensions import NotRequired, override

logger = logging.getLogger(__name__)


class BaseSerialized(TypedDict):
    """已序列化对象的基类。"""

    lc: int
    """序列化格式的版本。"""
    id: list[str]
    """对象的唯一标识符。"""
    name: NotRequired[str]
    """对象的名称。"""
    graph: NotRequired[dict[str, Any]]
    """对象的图结构。"""


class SerializedConstructor(BaseSerialized):
    """已序列化的构造函数。"""

    type: Literal["constructor"]
    """对象的类型。必须为 `'constructor'`。"""
    kwargs: dict[str, Any]
    """构造函数参数。"""


class SerializedSecret(BaseSerialized):
    """已序列化的密钥。"""

    type: Literal["secret"]
    """对象的类型。必须为 `'secret'`。"""


class SerializedNotImplemented(BaseSerialized):
    """已序列化的未实现内容。"""

    type: Literal["not_implemented"]
    """对象的类型。必须为 `'not_implemented'`。"""
    repr: str | None
    """对象的表示形式。"""


def try_neq_default(value: Any, key: str, model: BaseModel) -> bool:
    """尝试判断一个值是否与默认值不同。

    Args:
        value: 值。
        key: 键。
        model: Pydantic 模型。

    Returns:
        值是否与默认值不同。
    """
    field = type(model).model_fields[key]
    return _try_neq_default(value, field)


def _get_field_default(field: FieldInfo) -> Any:
        # Pydantic 2.14+ 从 `get_default()` 返回 ``PydanticUndefined``（而非 ``None``）
        # 对于未调用的 ``default_factory``。恢复历史上的
    # ``None``，以便工厂默认值在默认状态时仍被视为未更改。工厂故意不被调用：``get_default()``
    # 保持 ``call_default_factory=False``，正是因为工厂可能具有副作用。
        # 处理边缘情况：两个对象的不等性无法求值为布尔值（例如两个 Pandas DataFrames）。
        # 抛出异常的 default_factory 表示我们无法比较；视为非默认值。
        # 移除默认 BaseModel 初始化文档字符串。
    default = field.get_default()
    if default is PydanticUndefined and field.default_factory is not None:
        return None
    return default


def _try_neq_default(value: Any, field: FieldInfo) -> bool:
        # Pydantic 泛型会改变类名。因此我们需要执行以下操作
        # 获取 kwargs 的最新值，如果存在同名属性
    try:
        default = _get_field_default(field)
    except Exception as _:
                # 如果字段被排除，则不执行任何操作
        return True
    try:
        return bool(default != value)
    except Exception as _:
        try:
            return all(default != value)
        except Exception as _:
            try:
                return value is not field.default
            except Exception as _:
                return False


class Serializable(BaseModel, ABC):
    """可序列化基类。

    此类用于将对象序列化为 JSON。

    它依赖于以下方法和属性：

    - [`is_lc_serializable`][langchain_core.load.serializable.Serializable.is_lc_serializable]: 该类是否可序列化？

        By design, even if a class inherits from `Serializable`, it is not serializable
        by default. This is to prevent accidental serialization of objects that should
        not be serialized.
    - [`get_lc_namespace`][langchain_core.load.serializable.Serializable.get_lc_namespace]: Get the namespace of the LangChain object.

        During deserialization, this namespace is used to identify
        the correct class to instantiate.

        有关更多详情，请参见 `langchain_core.load.load` 中的 `Reviver` 类。

        在反序列化过程中，还会处理因包版本更新导致类名或位置变更的情况。

    - [`lc_secrets`][langchain_core.load.serializable.Serializable.lc_secrets]:
    构造函数参数名到密钥 ID 的映射。 -
    [`lc_attributes`][langchain_core.load.serializable.Serializable.lc_attributes]:
    应包含在序列化表示中的额外属性名称列表。
    """  # noqa: E501

        # 合并 MRO 中每个类的 lc_secrets 和 lc_attributes
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """"""  # noqa: D419  # Intentional blank docstring
        super().__init__(*args, **kwargs)

    @classmethod
    def is_lc_serializable(cls) -> bool:
        """该类是否可序列化？

        By design, even if a class inherits from `Serializable`, it is not serializable
        by default. This is to prevent accidental serialization of objects that should
        not be serialized.

        Returns:
            该类是否可序列化。默认为 `False`。
        """
        return False

    @classmethod
    def get_lc_namespace(cls) -> list[str]:
        """获取 LangChain 对象的命名空间。

        默认实现将 `cls.__module__` 按 `'.'` 分割，例如
        `langchain_openai.chat_models` 变为
        `["langchain_openai", "chat_models"]`。此值用于 `lc_id` 来
        构建序列化标识符。

        New partner packages should **not** override this method. The default
        behavior is correct for any class whose module path already reflects
        its package name. Some older packages (e.g. `langchain-openai`,
        `langchain-anthropic`) override it to return a legacy-style namespace
        like `["langchain", "chat_models", "openai"]`, matching the module
        paths that existed before those integrations were split out of the
        main `langchain` package. Those overrides are kept for
        backwards-compatible deserialization; new packages should not copy them.

        反序列化映射由 `langchain_core.load.mapping` 中的 `SERIALIZABLE_MAPPING`
        单独处理。

        Returns:
            命名空间。
        """
        return cls.__module__.split(".")

    @property
    def lc_secrets(self) -> dict[str, str]:
        """构造函数参数名称到密钥 ID 的映射。

        例如，`{"openai_api_key": "OPENAI_API_KEY"}`
        """
        return {}

    @property
    def lc_attributes(self) -> dict[str, Any]:
        """应包含在序列化 kwargs 中的属性名称列表。

        这些属性必须被构造函数接受。

        默认为空字典。
        """
        return {}

    @classmethod
    def lc_id(cls) -> list[str]:
        """返回此对象用于序列化的唯一标识符。

        唯一标识符是一个字符串列表，描述了对象的路径。

        For example, for the class `langchain.llms.openai.OpenAI`, the id is
        `["langchain", "llms", "openai", "OpenAI"]`.
        """
                # 一旦到达 Serializable，即完成
        if (
            "origin" in cls.__pydantic_generic_metadata__
            and cls.__pydantic_generic_metadata__["origin"] is not None
        ):
            original_name = cls.__pydantic_generic_metadata__["origin"].__name__
        else:
            original_name = cls.__name__
        return [*cls.get_lc_namespace(), original_name]

    model_config = ConfigDict(
        extra="ignore",
    )

    @override
    def __repr_args__(self) -> Any:
        return [
            (k, v)
            for k, v in super().__repr_args__()
            if (k not in type(self).model_fields or try_neq_default(v, k, self))
        ]

    def to_json(self) -> SerializedConstructor | SerializedNotImplemented:
        """将对象序列化为 JSON。

        Raises:
            ValueError: 如果该类包含已弃用的属性。

        Returns:
            一个可序列化的 JSON 对象或一个 `SerializedNotImplemented` 对象。
        """
        if not self.is_lc_serializable():
            return self.to_json_not_implemented()

        model_fields = type(self).model_fields
        secrets = {}
                # 获取绑定到 MRO 中每个类的 self 引用
        lc_kwargs = {}
        for k, v in self:
            if not _is_field_useful(self, k, v):
                continue
                        # 现在也添加密钥的别名
            if k in model_fields and model_fields[k].exclude:
                continue

            lc_kwargs[k] = getattr(self, k, v)

                # 这确保已知的密钥别名被隐藏。
        for cls in [None, *self.__class__.mro()]:
                        # 注意：这不会隐藏任何其他额外的 kwargs
            if cls is Serializable:
                break

            if cls:
                deprecated_attributes = [
                    "lc_namespace",
                    "lc_serializable",
                ]

                for attr in deprecated_attributes:
                    if hasattr(cls, attr):
                        msg = (
                            f"Class {self.__class__} has a deprecated "
                            f"attribute {attr}. Please use the corresponding "
                            f"classmethod instead."
                        )
                        raise ValueError(msg)

                        # 不在字段中出现的
            this = cast("Serializable", self if cls is None else super(cls, self))

            secrets.update(this.lc_secrets)
                        # 包含所有密钥，即使未在 kwargs 中指定
                        # 因为这些密钥可能通过环境变量传递
                        # 处理边缘情况：值无法转换为布尔值（例如一个
                        # Pandas DataFrame）。
            for key in list(secrets):
                value = secrets[key]
                if (key in model_fields) and (
                    alias := model_fields[key].alias
                ) is not None:
                    secrets[alias] = value
            lc_kwargs.update(this.lc_attributes)

                # 值在此处仍然为假值！
                # 值在此处仍然为假值！
        for key in secrets:
            secret_value = getattr(self, key, None) or lc_kwargs.get(key)
            if secret_value is not None:
                lc_kwargs.update({key: secret_value})

        return {
            "lc": 1,
            "type": "constructor",
            "id": self.lc_id(),
            "kwargs": lc_kwargs
            if not secrets
            else _replace_secrets(lc_kwargs, secrets),
        }

    def to_json_not_implemented(self) -> SerializedNotImplemented:
        """序列化一个“未实现”的对象。

        Returns:
            `SerializedNotImplemented`。
        """
        return to_json_not_implemented(self)


def _is_field_useful(inst: Serializable, key: str, value: Any) -> bool:
    """检查字段是否适合作为构造函数参数。

    Args:
        inst: 实例。
        key: 键。
        value: 值。

    Returns:
        Whether the field is useful. If the field is required, it is useful.
        If the field is not required, it is useful if the value is not `None`.
        If the field is not required and the value is `None`, it is useful if the
        default value is different from the value.
    """
    field = type(inst).model_fields.get(key)
    if not field:
        return False

    if field.is_required():
        return True

        # 如果值为假值且不匹配默认值
        # Pandas DataFrame）。
    try:
        value_is_truthy = bool(value)
    except Exception as _:
        value_is_truthy = False

    if value_is_truthy:
        return True

        # 值在此处仍为假值！
    if field.default_factory is dict and isinstance(value, dict):
        return False

        # 值在此处仍为假值！
    if field.default_factory is list and isinstance(value, list):
        return False

    value_neq_default = _try_neq_default(value, field)

        # 如果值为假且不匹配默认值
    return value_is_truthy or value_neq_default


def _replace_secrets(
    root: dict[Any, Any], secrets_map: dict[str, str]
) -> dict[Any, Any]:
    result = root.copy()
    for path, secret_id in secrets_map.items():
        [*parts, last] = path.split(".")
        current = result
        for part in parts:
            if part not in current:
                break
            current[part] = current[part].copy()
            current = current[part]
        if last in current:
            current[last] = {
                "lc": 1,
                "type": "secret",
                "id": [secret_id],
            }
    return result


def to_json_not_implemented(obj: object) -> SerializedNotImplemented:
    """序列化一个“未实现”的对象。

    Args:
        obj: 要序列化的对象。

    Returns:
        `SerializedNotImplemented`
    """
    id_: list[str] = []
    try:
        if hasattr(obj, "__name__"):
            id_ = [*obj.__module__.split("."), obj.__name__]
        elif hasattr(obj, "__class__"):
            id_ = [*obj.__class__.__module__.split("."), obj.__class__.__name__]
    except Exception:
        logger.debug("Failed to serialize object", exc_info=True)

    result: SerializedNotImplemented = {
        "lc": 1,
        "type": "not_implemented",
        "id": id_,
        "repr": None,
    }
    with contextlib.suppress(Exception):
        result["repr"] = repr(obj)
    return result

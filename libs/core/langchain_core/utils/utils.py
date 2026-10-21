"""通用实用函数。"""

import contextlib
import datetime
import functools
import importlib
import os
import warnings
from collections.abc import Callable, Iterator, Sequence
from importlib.metadata import version
from typing import Any, overload
from uuid import uuid4

from packaging.version import parse
from pydantic import SecretStr
from requests import HTTPError, Response
from typing_extensions import override

from langchain_core.utils.pydantic import (
    is_pydantic_v1_subclass,
)


def xor_args(*arg_groups: tuple[str, ...]) -> Callable[..., Any]:
    """验证指定的关键字参数互斥。

    Args:
        *arg_groups: 互斥关键字参数组。

    Returns:
        装饰器，用于验证指定的关键字参数互斥。
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            """验证每组中恰好有一个参数不为 None。"""
            counts = [
                sum(1 for arg in arg_group if kwargs.get(arg) is not None)
                for arg_group in arg_groups
            ]
            invalid_groups = [i for i, count in enumerate(counts) if count != 1]
            if invalid_groups:
                invalid_group_names = [", ".join(arg_groups[i]) for i in invalid_groups]
                msg = (
                    "Exactly one argument in each of the following"
                    " groups must be defined:"
                    f" {', '.join(invalid_group_names)}"
                )
                raise ValueError(msg)
            return func(*args, **kwargs)

        return wrapper

    return decorator


def raise_for_status_with_text(response: Response) -> None:
    """若响应包含错误状态码，则抛出异常并附带响应文本。

    Args:
        response: 要检查错误的响应。

    Raises:
        ValueError: 若响应具有错误状态码。
    """
    try:
        response.raise_for_status()
    except HTTPError as e:
        raise ValueError(response.text) from e


@contextlib.contextmanager
def mock_now(dt_value: datetime.datetime) -> Iterator[type]:
    """用于在单元测试中模拟 datetime.now() 的上下文管理器。

    Args:
        dt_value: 用于 datetime.now() 的模拟时间值。

    Yields:
        模拟的时间类。

    Example:
        ```python
        with mock_now(datetime.datetime(2011, 2, 3, 10, 11)):
            assert datetime.datetime.now() == datetime.datetime(2011, 2, 3, 10, 11)
        ```
    """

    class MockDateTime(datetime.datetime):
        """用固定的时间戳模拟 `datetime.datetime.now()`。"""

        @classmethod
        @override
        def now(cls, tz: datetime.tzinfo | None = None) -> "MockDateTime":
                        # 创建 `dt_value` 的副本。
            return MockDateTime(
                dt_value.year,
                dt_value.month,
                dt_value.day,
                dt_value.hour,
                dt_value.minute,
                dt_value.second,
                dt_value.microsecond,
                dt_value.tzinfo,
            )

    real_datetime = datetime.datetime
    datetime.datetime = MockDateTime  # type: ignore[misc]
    try:
        yield datetime.datetime
    finally:
        datetime.datetime = real_datetime  # type: ignore[misc]


def guard_import(
    module_name: str, *, pip_name: str | None = None, package: str | None = None
) -> Any:
    """动态导入一个模块。

    若模块未安装则抛出异常。

    Args:
        module_name: 要导入的模块名称。
        pip_name: 使用 pip 安装该模块时的名称。
        package: 从哪个包中导入该模块。

    Returns:
        导入的模块。

    Raises:
        ImportError: 若模块未安装。
    """
    try:
        module = importlib.import_module(module_name, package)
    except (ImportError, ModuleNotFoundError) as e:
        pip_name = pip_name or module_name.split(".", maxsplit=1)[0].replace("_", "-")
        msg = (
            f"Could not import {module_name} python package. "
            f"Please install it with `pip install {pip_name}`."
        )
        raise ImportError(msg) from e
    return module


def check_package_version(
    package: str,
    lt_version: str | None = None,
    lte_version: str | None = None,
    gt_version: str | None = None,
    gte_version: str | None = None,
) -> None:
    """检查包的版本。

    Args:
        package: 包名称。
        lt_version: 版本必须小于该值。
        lte_version: 版本必须小于或等于该值。
        gt_version: 版本必须大于该值。
        gte_version: 版本必须大于或等于该值。


    Raises:
        ValueError: 若包版本不符合要求。
    """
    imported_version = parse(version(package))
    if lt_version is not None and imported_version >= parse(lt_version):
        msg = (
            f"Expected {package} version to be < {lt_version}. Received "
            f"{imported_version}."
        )
        raise ValueError(msg)
    if lte_version is not None and imported_version > parse(lte_version):
        msg = (
            f"Expected {package} version to be <= {lte_version}. Received "
            f"{imported_version}."
        )
        raise ValueError(msg)
    if gt_version is not None and imported_version <= parse(gt_version):
        msg = (
            f"Expected {package} version to be > {gt_version}. Received "
            f"{imported_version}."
        )
        raise ValueError(msg)
    if gte_version is not None and imported_version < parse(gte_version):
        msg = (
            f"Expected {package} version to be >= {gte_version}. Received "
            f"{imported_version}."
        )
        raise ValueError(msg)


def get_pydantic_field_names(pydantic_cls: Any) -> set[str]:
    """获取 Pydantic 类的字段名，包括别名。

    Args:
        pydantic_cls: Pydantic 类。

    Returns:
        字段名列表。
    """
    all_required_field_names = set()
    if is_pydantic_v1_subclass(pydantic_cls):
        for field in pydantic_cls.__fields__.values():
            all_required_field_names.add(field.name)
            if field.has_alias:
                all_required_field_names.add(field.alias)
    else:  # 假设当前使用 pydantic 2
        for name, field in pydantic_cls.model_fields.items():
            all_required_field_names.add(name)
            if field.alias:
                all_required_field_names.add(field.alias)
    return all_required_field_names


def _build_model_kwargs(
    values: dict[str, Any],
    all_required_field_names: set[str],
) -> dict[str, Any]:
    """从 Pydantic 构造函数的值构建 `model_kwargs` 参数。

    Args:
        values: 用户传入的所有初始化参数。
        all_required_field_names: Pydantic 类的所有必填字段名。

    Returns:
        额外的关键字参数。

    Raises:
        ValueError: 若某个字段同时出现在 `values` 和 `extra_kwargs` 中。
        ValueError: 若某个字段出现在 `model_kwargs` 中。
    """
    extra_kwargs = values.get("model_kwargs", {})
    for field_name in list(values):
        if field_name in extra_kwargs:
            msg = f"Found {field_name} supplied twice."
            raise ValueError(msg)
        if field_name not in all_required_field_names:
            warnings.warn(
                f"""WARNING! {field_name} is not default parameter.
                {field_name} was transferred to model_kwargs.
                Please confirm that {field_name} is what you intended.""",
                stacklevel=7,
            )
            extra_kwargs[field_name] = values.pop(field_name)

    invalid_model_kwargs = all_required_field_names.intersection(extra_kwargs.keys())
    if invalid_model_kwargs:
        warnings.warn(
            f"Parameters {invalid_model_kwargs} should be specified explicitly. "
            f"Instead they were passed in as part of `model_kwargs` parameter.",
            stacklevel=7,
        )
        for k in invalid_model_kwargs:
            values[k] = extra_kwargs.pop(k)

    values["model_kwargs"] = extra_kwargs
    return values


# 不要使用！保留以保证向后兼容性，但本不应公开。
def build_extra_kwargs(
    extra_kwargs: dict[str, Any],
    values: dict[str, Any],
    all_required_field_names: set[str],
) -> dict[str, Any]:
    """从 values 和 extra_kwargs 构建额外的关键字参数。

    !!! danger "不要使用"

        Kept for backwards-compatibility but should never have been public. Use the
        internal `_build_model_kwargs` function instead.

    Args:
        extra_kwargs: 用户传入的额外关键字参数。
        values: 用户传入的值。
        all_required_field_names: Pydantic 类的所有必填字段名。

    Returns:
        额外的关键字参数。

    Raises:
        ValueError: 若某个字段同时出现在 `values` 和 `extra_kwargs` 中。
        ValueError: 若某个字段出现在 `model_kwargs` 中。
    """
        # 不要使用！保留以保证向后兼容性，但本不应公开。
    for field_name in list(values):
        if field_name in extra_kwargs:
            msg = f"Found {field_name} supplied twice."
            raise ValueError(msg)
        if field_name not in all_required_field_names:
            warnings.warn(
                f"""WARNING! {field_name} is not default parameter.
                {field_name} was transferred to model_kwargs.
                Please confirm that {field_name} is what you intended.""",
                stacklevel=7,
            )
            extra_kwargs[field_name] = values.pop(field_name)

        # 不要使用！保留以保证向后兼容性，但本不应公开。
    invalid_model_kwargs = all_required_field_names.intersection(extra_kwargs.keys())
    if invalid_model_kwargs:
        msg = (
            f"Parameters {invalid_model_kwargs} should be specified explicitly. "
            f"Instead they were passed in as part of `model_kwargs` parameter."
        )
        raise ValueError(msg)

        # 不要使用！保留以保证向后兼容性，但本不应公开。
    return extra_kwargs


def convert_to_secret_str(value: SecretStr | str) -> SecretStr:
    """若需要，将字符串转换为 `SecretStr`。

    Args:
        value: 要转换的值。

    Returns:
        `SecretStr` 类型的值。
    """
    if isinstance(value, SecretStr):
        return value
    return SecretStr(value)


class _NoDefaultType:
    """表示未提供默认值的类型。"""


_NoDefault = _NoDefaultType()


@overload
def from_env(key: str, /) -> Callable[[], str]: ...


@overload
def from_env(key: str, /, *, default: str) -> Callable[[], str]: ...


@overload
def from_env(key: Sequence[str], /, *, default: str) -> Callable[[], str]: ...


@overload
def from_env(key: str, /, *, error_message: str) -> Callable[[], str]: ...


@overload
def from_env(
    key: str | Sequence[str], /, *, default: str, error_message: str | None
) -> Callable[[], str]: ...


@overload
def from_env(
    key: str, /, *, default: None, error_message: str | None
) -> Callable[[], str | None]: ...


@overload
def from_env(
    key: str | Sequence[str], /, *, default: None
) -> Callable[[], str | None]: ...


def from_env(
    key: str | Sequence[str],
    /,
    *,
    default: str | _NoDefaultType | None = _NoDefault,
    error_message: str | None = None,
) -> Callable[[], str] | Callable[[], str | None]:
    """创建一个从环境变量获取值的工厂方法。

    Args:
        key: 要查找的环境变量。

            If a list of keys is provided, the first key found in the environment will
            be used. If no key is found, the default value will be used if set,
            otherwise an error will be raised.
        default: The default value to return if the environment variable is not set.
        error_message: The error message which will be raised if the key is not found
            and no default value is provided.

            这将作为 `ValueError` 被抛出。

    Returns:
        用于从环境变量中查找值的工厂方法。
    """

    def get_from_env_fn() -> str | None:
        """从环境变量中获取一个值。

        Raises:
            ValueError: 如果环境变量未设置且未提供默认值。

        Returns:
            来自环境的值。
        """
        if isinstance(key, (list, tuple)):
            for k in key:
                if k in os.environ:
                    return os.environ[k]
        if isinstance(key, str) and key in os.environ:
            return os.environ[key]

        if isinstance(default, (str, type(None))):
            return default
        if error_message:
            raise ValueError(error_message)
        msg = (
            f"Did not find {key}, please add an environment variable"
            f" `{key}` which contains it, or pass"
            f" `{key}` as a named parameter."
        )
        raise ValueError(msg)

    return get_from_env_fn


@overload
def secret_from_env(key: str | Sequence[str], /) -> Callable[[], SecretStr]: ...


@overload
def secret_from_env(key: str, /, *, default: str) -> Callable[[], SecretStr]: ...


@overload
def secret_from_env(
    key: str | Sequence[str], /, *, default: None
) -> Callable[[], SecretStr | None]: ...


@overload
def secret_from_env(key: str, /, *, error_message: str) -> Callable[[], SecretStr]: ...


def secret_from_env(
    key: str | Sequence[str],
    /,
    *,
    default: str | _NoDefaultType | None = _NoDefault,
    error_message: str | None = None,
) -> Callable[[], SecretStr | None] | Callable[[], SecretStr]:
    """环境中的密钥。

    Args:
        key: The environment variable to look up.
        default: The default value to return if the environment variable is not set.
        error_message: The error message which will be raised if the key is not found
            and no default value is provided.

            这将作为 `ValueError` 被抛出。

    Returns:
        用于从环境变量中查找密钥的工厂方法。
    """

    def get_secret_from_env() -> SecretStr | None:
        """从环境变量中获取一个值。

        Raises:
            ValueError: 如果环境变量未设置且未提供默认值。

        Returns:
            来自环境的密钥。
        """
        if isinstance(key, (list, tuple)):
            for k in key:
                if k in os.environ:
                    return SecretStr(os.environ[k])
        if isinstance(key, str) and key in os.environ:
            return SecretStr(os.environ[key])
        if isinstance(default, str):
            return SecretStr(default)
        if default is None:
            return None
        if error_message:
            raise ValueError(error_message)
        msg = (
            f"Did not find {key}, please add an environment variable"
            f" `{key}` which contains it, or pass"
            f" `{key}` as a named parameter."
        )
        raise ValueError(msg)

    return get_secret_from_env


LC_AUTO_PREFIX = "lc_"
"""LangChain auto-generated ID prefix for messages and content blocks."""

LC_ID_PREFIX = "lc_run-"
"""Internal tracing/callback system identifier.

Used for:

- Tracing. Every LangChain operation (LLM call, chain execution, tool use, etc.)
    gets a unique run_id (UUID)
- Enables tracking parent-child relationships between operations
"""


def ensure_id(id_val: str | None) -> str:
    """确保 ID 是一个有效的字符串，若未提供则生成一个新的 UUID。

    自动生成的 UUID 以 `'lc_'` 为前缀，以表明其由 LangChain 生成。

    Args:
        id_val: 可选的字符串 ID 值，用于验证。

    Returns:
        一个字符串 ID，可以是验证后的原始值，也可以是新生成的 UUID4。
    """
    return id_val or f"{LC_AUTO_PREFIX}{uuid4()}"

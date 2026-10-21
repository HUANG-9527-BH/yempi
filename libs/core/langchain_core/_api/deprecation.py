"""用于弃用 LangChain API 部分内容的辅助函数。

This module was adapted from matplotlib's [`_api/deprecation.py`](https://github.com/matplotlib/matplotlib/blob/main/lib/matplotlib/_api/deprecation.py)
module.

!!! warning

    This module is for internal use only. Do not use it in your own code. We may change
    the API at any time with no warning.
"""

import contextlib
import functools
import inspect
import sys
import warnings
from collections.abc import Callable, Generator
from contextvars import ContextVar
from typing import (
    TYPE_CHECKING,
    Any,
    ParamSpec,
    TypeGuard,
    TypeVar,
    cast,
)

from pydantic.fields import FieldInfo

from langchain_core._api.internal import is_caller_internal

if TYPE_CHECKING:
    from pydantic.v1.fields import FieldInfo as FieldInfoV1


def _is_pydantic_v1_field_info(obj: Any) -> TypeGuard["FieldInfoV1"]:
    """检查 `obj` 是否为 `pydantic.v1.fields.FieldInfo`，而无需强制导入。

    Avoids importing `pydantic.v1`, which emits a `UserWarning` on Python 3.14+
    with Pydantic 2.12. A v1 `FieldInfo` can only exist once a caller has imported
    `pydantic.v1.fields`, so this check consults it only if it is already loaded.
    """
    mod = sys.modules.get("pydantic.v1.fields")
    if mod is None:
        return False
    return isinstance(obj, mod.FieldInfo)


def _build_deprecation_message(
    *,
    alternative: str = "",
    alternative_import: str = "",
) -> str:
    """为 `__deprecated__` 属性构建一个简单的弃用消息。

    Args:
        alternative: 替代 API 名称。
        alternative_import: 替代 API 的完整导入路径。

    Returns:
        用于 IDE/类型检查器显示的弃用消息字符串。
    """
    if alternative_import:
        return f"Use {alternative_import} instead."
    if alternative:
        return f"Use {alternative} instead."
    return "Deprecated."


class LangChainDeprecationWarning(DeprecationWarning):
    """用于向 LangChain 用户发出弃用警告的类。"""


class LangChainPendingDeprecationWarning(PendingDeprecationWarning):
    """用于向 LangChain 用户发出弃用警告的类。"""


# 跟踪调用者是否有意静默 LangChain 弃用警告。
# 被抑制的警告不应消耗已弃用可调用对象的一次性警告状态；否则内部兼容路径可能阻止首次用户可见的调用发出警告。
# PUBLIC API
# Bound 为 `Any`（而非 `FieldInfoV1`），因为模块作用域导入 `pydantic.v1` 会在 Python
# 3.14+ 上引发 `UserWarning`；v1 `FieldInfo` 支持在运行时通过
# `_is_pydantic_v1_field_info` 处理。
_SUPPRESSING_LANGCHAIN_DEPRECATION_WARNING = ContextVar(
    "_SUPPRESSING_LANGCHAIN_DEPRECATION_WARNING", default=False
)


# 仅当警告未被 `suppress_langchain_deprecation_warning()` 明确抑制时，才标记为已发出。


# 仅当警告未被 `suppress_langchain_deprecation_warning()` 明确抑制时，才标记为已发出。
# 无法在某些扩展对象上设置 new_doc。
# 仅当警告未被 `suppress_langchain_deprecation_warning()` 明确抑制时，才标记为已发出。
T = TypeVar("T", bound=type | Callable[..., Any] | Any)


def _validate_deprecation_params(
    removal: str,
    alternative: str,
    alternative_import: str,
    *,
    pending: bool,
) -> None:
    """验证弃用参数。"""
    if pending and removal:
        msg = "A pending deprecation cannot have a scheduled removal"
        raise ValueError(msg)
    if alternative and alternative_import:
        msg = "Cannot specify both alternative and alternative_import"
        raise ValueError(msg)

    if alternative_import and "." not in alternative_import:
        msg = (
            "alternative_import must be a fully qualified module path. Got "
            f" {alternative_import}"
        )
        raise ValueError(msg)


def deprecated(
    since: str,
    *,
    message: str = "",
    name: str = "",
    alternative: str = "",
    alternative_import: str = "",
    pending: bool = False,
    obj_type: str = "",
    addendum: str = "",
    removal: str = "",
    package: str = "",
) -> Callable[[T], T]:
    """用于将函数、类或属性标记为弃用的装饰器。

    当弃用类方法、静态方法或属性时，`@deprecated` 装饰器应位于 `@classmethod` 和
    `@staticmethod` 之下（即 `deprecated` 应直接装饰底层可调用对象），但位于
    `@property` 之上。

    When deprecating a class `C` intended to be used as a base class in a multiple
    inheritance hierarchy, `C` *must* define an `__init__` method (if `C` instead
    inherited its `__init__` from its own base class, then `@deprecated` would mess up
    `__init__` inheritance when installing its own (deprecation-emitting) `C.__init__`).

    Parameters are the same as for `warn_deprecated`, except that *obj_type* defaults to
    'class' if decorating a class, 'attribute' if decorating a property, and 'function'
    otherwise.

    Args:
        since: 此 API 开始被弃用的版本。
        message: 覆盖默认的弃用消息。

            `%(since)s`、`%(name)s`、`%(alternative)s`、`%(obj_type)s`、 `%(addendum)s`
            和 `%(removal)s` 格式说明符将被替换为传入此函数的相应参数值。 name:
            被弃用对象的名称。 alternative: 用户可用来替代被弃用 API 的替代 API。

            The deprecation warning will tell the user about this alternative if
            provided.
        alternative_import: An alternative import that the user may use instead.
        pending: If `True`, uses a `PendingDeprecationWarning` instead of a
            `DeprecationWarning`.

            Cannot be used together with removal.
        obj_type: The object type being deprecated.
        addendum: Additional text appended directly to the final message.
        removal: The expected removal version.

            使用默认值（空字符串）时，警告消息中不会显示移除版本。

            不能与 pending 一起使用。
        package: 被弃用对象的包名。

    Returns:
        用于标记函数或类为已弃用的装饰器。

    Example:
        ```python
        @deprecated("1.4.0")
        def the_function_to_deprecate():
            pass
        ```
    """
    _validate_deprecation_params(
        removal, alternative, alternative_import, pending=pending
    )

    def deprecate(
        obj: T,
        *,
        _obj_type: str = obj_type,
        _name: str = name,
        _message: str = message,
        _alternative: str = alternative,
        _alternative_import: str = alternative_import,
        _pending: bool = pending,
        _addendum: str = addendum,
        _package: str = package,
    ) -> T:
        """`deprecated` 返回的装饰器的实际实现。"""

        def emit_warning() -> None:
            """发出警告。"""
            warn_deprecated(
                since,
                message=_message,
                name=_name,
                alternative=_alternative,
                alternative_import=_alternative_import,
                pending=_pending,
                obj_type=_obj_type,
                addendum=_addendum,
                removal=removal,
                package=_package,
            )

        warned = False

        def warning_emitting_wrapper(*args: Any, **kwargs: Any) -> Any:
            """包装原始可调用对象并发出警告的包装器。

            Args:
                *args: 函数的位置参数。
                **kwargs: 函数的关键字参数。

            Returns:
                被包装函数的返回值。
            """
            nonlocal warned
            if not warned and not is_caller_internal():
                emit_warning()
                                # 为 PEP 702（IDE/类型检查器支持）设置 __deprecated__
                                # 为 PEP 702（IDE/类型检查器支持）设置 __deprecated__
                warned = not _SUPPRESSING_LANGCHAIN_DEPRECATION_WARNING.get()
            return wrapped(*args, **kwargs)

        async def awarning_emitting_wrapper(*args: Any, **kwargs: Any) -> Any:
            """与 warning_emitting_wrapper 相同，但适用于异步函数。"""
            nonlocal warned
            if not warned and not is_caller_internal():
                emit_warning()
                                # 特殊情况：当函数位于另一个函数内
                                # 在测试中，这会将其识别为 "方法" 而非 "函数"
                warned = not _SUPPRESSING_LANGCHAIN_DEPRECATION_WARNING.get()
            return await wrapped(*args, **kwargs)

        _package = _package or obj.__module__.split(".")[0].replace("_", "-")

        if isinstance(obj, type):
            if not _obj_type:
                _obj_type = "class"
            wrapped = obj.__init__  # type: ignore[misc]
            _name = _name or obj.__qualname__
            old_doc = obj.__doc__

            def finalize(_: Callable[..., Any], new_doc: str, /) -> T:
                """完成类的弃用处理。"""
                                # 为 PEP 702（IDE/类型检查器支持）设置 __deprecated__
                with contextlib.suppress(AttributeError):
                    obj.__doc__ = new_doc

                def warn_if_direct_instance(
                    self: Any, *args: Any, **kwargs: Any
                ) -> Any:
                    """警告该类处于测试版状态。"""
                    nonlocal warned
                    if not warned and type(self) is obj and not is_caller_internal():
                        emit_warning()
                                                # old_doc 可能为 None
                                                # 修改 docstring 以包含弃用通知。
                        warned = not _SUPPRESSING_LANGCHAIN_DEPRECATION_WARNING.get()
                    return wrapped(self, *args, **kwargs)

                obj.__init__ = functools.wraps(obj.__init__)(  # type: ignore[misc]
                    warn_if_direct_instance
                )
                                # 为 PEP 702（IDE/类型检查器支持）设置 __deprecated__
                obj.__deprecated__ = _build_deprecation_message(  # type: ignore[attr-defined]
                    alternative=alternative,
                    alternative_import=alternative_import,
                )
                return obj

        elif _is_pydantic_v1_field_info(obj):
            wrapped = None
            if not _obj_type:
                _obj_type = "attribute"
            if not _name:
                msg = f"Field {obj} must have a name to be deprecated."
                raise ValueError(msg)
            old_doc = obj.description

            def finalize(_: Callable[..., Any], new_doc: str, /) -> T:
                from pydantic.v1.fields import FieldInfo as FieldInfoV1  # noqa: PLC0415

                return cast(
                    "T",
                    FieldInfoV1(
                        default=obj.default,
                        default_factory=obj.default_factory,
                        description=new_doc,
                        alias=obj.alias,
                        exclude=obj.exclude,
                    ),
                )

        elif isinstance(obj, FieldInfo):
            wrapped = None
            if not _obj_type:
                _obj_type = "attribute"
            if not _name:
                msg = f"Field {obj} must have a name to be deprecated."
                raise ValueError(msg)
            old_doc = obj.description

            def finalize(_: Callable[..., Any], new_doc: str, /) -> T:
                return cast(
                    "T",
                    FieldInfo(
                        default=obj.default,
                        default_factory=obj.default_factory,
                        description=new_doc,
                        alias=obj.alias,
                        exclude=obj.exclude,
                    ),
                )

        elif isinstance(obj, property):
            if not _obj_type:
                _obj_type = "attribute"
            wrapped = None
            _name = _name or cast("type", obj.fget).__qualname__
            old_doc = obj.__doc__

            class _DeprecatedProperty(property):
                """一个已弃用的属性。"""

                def __init__(
                    self,
                    fget: Callable[[Any], Any] | None = None,
                    fset: Callable[[Any, Any], None] | None = None,
                    fdel: Callable[[Any], None] | None = None,
                    doc: str | None = None,
                ) -> None:
                    super().__init__(fget, fset, fdel, doc)
                    self.__orig_fget = fget
                    self.__orig_fset = fset
                    self.__orig_fdel = fdel

                def __get__(self, instance: Any, owner: type | None = None) -> Any:
                    if instance is not None or owner is not None:
                        emit_warning()
                    if self.fget is None:
                        return None
                    return self.fget(instance)

                def __set__(self, instance: Any, value: Any) -> None:
                    if instance is not None:
                        emit_warning()
                    if self.fset is not None:
                        self.fset(instance, value)

                def __delete__(self, instance: Any) -> None:
                    if instance is not None:
                        emit_warning()
                    if self.fdel is not None:
                        self.fdel(instance)

                def __set_name__(self, owner: type | None, set_name: str) -> None:
                    nonlocal _name
                    if _name == "<lambda>":
                        _name = set_name

            def finalize(_: Callable[..., Any], new_doc: str, /) -> T:
                """完成属性的处理。"""
                prop = _DeprecatedProperty(
                    fget=obj.fget, fset=obj.fset, fdel=obj.fdel, doc=new_doc
                )
                                # 为 PEP 702（IDE/类型检查器支持）设置 __deprecated__
                prop.__deprecated__ = _build_deprecation_message(  # type: ignore[attr-defined]
                    alternative=alternative,
                    alternative_import=alternative_import,
                )
                return cast("T", prop)

        else:
            _name = _name or cast("type", obj).__qualname__
            if not _obj_type:
                                # 边缘情况：当一个函数嵌套在另一个函数中时
                                # 在测试中，这将把它称为“方法”而非“函数”
                _obj_type = "function" if "." not in _name else "method"
            wrapped = obj
            old_doc = wrapped.__doc__

            def finalize(wrapper: Callable[..., Any], new_doc: str, /) -> T:
                """使用包装器包装被包装函数，并更新文档字符串。

                Args:
                    wrapper: 包装函数。
                    new_doc: 新的文档字符串。

                Returns:
                    被包装的函数。
                """
                wrapper = functools.wraps(wrapped)(wrapper)
                wrapper.__doc__ = new_doc
                                # 为 PEP 702（IDE/类型检查器支持）设置 __deprecated__
                wrapper.__deprecated__ = _build_deprecation_message(  # type: ignore[attr-defined]
                    alternative=alternative,
                    alternative_import=alternative_import,
                )
                return cast("T", wrapper)

        old_doc = inspect.cleandoc(old_doc or "").strip("\n")

                # old_doc 可以为 None
        if not old_doc:
            old_doc = ""

                # 修改 docstring 以包含弃用通知。
        if (
            _alternative
            and _alternative.rsplit(".", maxsplit=1)[-1].lower()
            == _alternative.rsplit(".", maxsplit=1)[-1]
        ) or _alternative:
            _alternative = f"`{_alternative}`"

        if (
            _alternative_import
            and _alternative_import.rsplit(".", maxsplit=1)[-1].lower()
            == _alternative_import.rsplit(".", maxsplit=1)[-1]
        ) or _alternative_import:
            _alternative_import = f"`{_alternative_import}`"

        components = [
            _message,
            f"Use {_alternative} instead." if _alternative else "",
            f"Use {_alternative_import} instead." if _alternative_import else "",
            _addendum,
        ]
        details = " ".join([component.strip() for component in components if component])
        package = _package or (
            _name.split(".")[0].replace("_", "-") if "." in _name else None
        )
        if removal:
            if removal.startswith("1.") and package and package.startswith("langchain"):
                removal_str = f"It will not be removed until {package}=={removal}."
            else:
                removal_str = f"It will be removed in {package}=={removal}."
        else:
            removal_str = ""
        new_doc = f"""\
!!! deprecated "{since} {details} {removal_str}"

{old_doc}\
"""

        if inspect.iscoroutinefunction(obj):
            return finalize(awarning_emitting_wrapper, new_doc)
        return finalize(warning_emitting_wrapper, new_doc)

    return deprecate


@contextlib.contextmanager
def suppress_langchain_deprecation_warning() -> Generator[None, None, None]:
    """上下文管理器，用于抑制 `LangChainDeprecationWarning`。"""
    token = _SUPPRESSING_LANGCHAIN_DEPRECATION_WARNING.set(True)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", LangChainDeprecationWarning)
            warnings.simplefilter("ignore", LangChainPendingDeprecationWarning)
            yield
    finally:
        _SUPPRESSING_LANGCHAIN_DEPRECATION_WARNING.reset(token)


def warn_deprecated(
    since: str,
    *,
    message: str = "",
    name: str = "",
    alternative: str = "",
    alternative_import: str = "",
    pending: bool = False,
    obj_type: str = "",
    addendum: str = "",
    removal: str = "",
    package: str = "",
) -> None:
    """显示标准化的弃用信息。

    Args:
        since: 此 API 开始被弃用的版本。
        message: 覆盖默认的弃用消息。

            `%(since)s`, `%(name)s`, `%(alternative)s`, `%(obj_type)s`, `%(addendum)s`,
            和 `%(removal)s` 格式化占位符将被相应参数的值替换。 name:
            被弃用对象的名称。 alternative: 用户可替代被弃用 API 使用的其他 API。

            The deprecation warning will tell the user about this alternative if
            provided.
        alternative_import: An alternative import that the user may use instead.
        pending: If `True`, uses a `PendingDeprecationWarning` instead of a
            `DeprecationWarning`.

            不能与 removal 一起使用。
        obj_type: 被弃用的对象类型。
        addendum: 直接附加到最终消息末尾的额外文本。
        removal: 预期的移除版本。

            使用默认值（空字符串）时，警告消息中不会显示移除版本。

            不能与 pending 一起使用。
        package: 被弃用对象的包名。
    """
    if not pending and removal:
        removal = f"in {removal}"

    if not message:
        message = ""
        package_ = (
            package or name.split(".", maxsplit=1)[0].replace("_", "-")
            if "." in name
            else "LangChain"
        )

        if obj_type:
            message += f"The {obj_type} `{name}`"
        else:
            message += f"`{name}`"

        if pending:
            message += " will be deprecated in a future version"
        else:
            message += f" was deprecated in {package_} {since}"

            if removal:
                message += f" and will be removed {removal}"

        if alternative_import:
            alt_package = alternative_import.split(".", maxsplit=1)[0].replace("_", "-")
            if alt_package == package_:
                message += f". Use {alternative_import} instead."
            else:
                alt_module, alt_name = alternative_import.rsplit(".", 1)
                message += (
                    f". An updated version of the {obj_type} exists in the "
                    f"{alt_package} package and should be used instead. To use it run "
                    f"`pip install -U {alt_package}` and import as "
                    f"`from {alt_module} import {alt_name}`."
                )
        elif alternative:
            message += f". Use {alternative} instead."

        if addendum:
            message += f" {addendum}"

    warning_cls = (
        LangChainPendingDeprecationWarning if pending else LangChainDeprecationWarning
    )
    warning = warning_cls(message)
    warnings.warn(warning, category=LangChainDeprecationWarning, stacklevel=4)


def surface_langchain_deprecation_warnings() -> None:
    """恢复 LangChain 弃用警告的显示。"""
    warnings.filterwarnings(
        "default",
        category=LangChainPendingDeprecationWarning,
    )

    warnings.filterwarnings(
        "default",
        category=LangChainDeprecationWarning,
    )


_P = ParamSpec("_P")
_R = TypeVar("_R")


def rename_parameter(
    *,
    since: str,
    removal: str,
    old: str,
    new: str,
) -> Callable[[Callable[_P, _R]], Callable[_P, _R]]:
    """标记 *func* 的参数 *old* 已重命名为 *new* 的装饰器。

    The actual implementation of *func* should use *new*, not *old*. If *old* is passed
    to *func*, a `DeprecationWarning` is emitted, and its value is used, even if *new*
    is also passed by keyword.

    Args:
        since: 参数重命名的版本。
        removal: 旧参数将被移除的版本。
        old: 旧参数名称。
        new: 新参数名称。

    Returns:
        标记参数已被重命名的装饰器。

    Example:
        ```python
        @_api.rename_parameter("3.1", "bad_name", "good_name")
        def func(good_name): ...
        ```
    """

    def decorator(f: Callable[_P, _R]) -> Callable[_P, _R]:
        @functools.wraps(f)
        def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _R:
            if new in kwargs and old in kwargs:
                msg = f"{f.__name__}() got multiple values for argument {new!r}"
                raise TypeError(msg)
            if old in kwargs:
                warn_deprecated(
                    since,
                    removal=removal,
                    message=f"The parameter `{old}` of `{f.__name__}` was "
                    f"deprecated in {since} and will be removed "
                    f"in {removal} Use `{new}` instead.",
                )
                kwargs[new] = kwargs.pop(old)
            return f(*args, **kwargs)

        return wrapper

    return decorator

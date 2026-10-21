"""用于 `Runnable` 对象的工具代码。"""

from __future__ import annotations

import ast
import asyncio
import inspect
import sys
import textwrap

# Cannot move to TYPE_CHECKING as Mapping and Sequence are needed at runtime by
# RunnableConfigurableFields.
from collections.abc import Mapping, Sequence  # noqa: TC003
from functools import lru_cache
from itertools import groupby
from typing import (
    TYPE_CHECKING,
    Any,
    NamedTuple,
    Protocol,
    TypeGuard,
    TypeVar,
)

from typing_extensions import override

from langchain_core.utils._signature import _signature_parameters

# Re-export create-model for backwards compatibility
from langchain_core.utils.pydantic import create_model  # noqa: F401

if TYPE_CHECKING:
    from collections.abc import (
        AsyncIterable,
        AsyncIterator,
        Awaitable,
        Callable,
        Coroutine,
        Iterable,
    )
    from contextvars import Context

    from langchain_core.runnables.schema import StreamEvent

Input = TypeVar("Input", contravariant=True)  # noqa: PLC0105
# Output type should implement __concat__, as eg str, list, dict do
Output = TypeVar("Output", covariant=True)  # noqa: PLC0105


async def gated_coro(
    semaphore: asyncio.Semaphore, coro: Coroutine[Any, Any, Any]
) -> Any:
    """使用信号量运行协程。

    Args:
        semaphore: 要使用的信号量。
        coro: 要运行的协程。

    Returns:
        协程的结果。
    """
    async with semaphore:
        return await coro


async def gather_with_concurrency(
    n: int | None, *coros: Coroutine[Any, Any, Any]
) -> list[Any]:
    """在限制并发数量的情况下收集协程。

    Args:
        n: 要并发运行的协程数量。
        *coros: 要运行的协程。

    Returns:
        协程的结果。
    """
    if n is None:
        return await asyncio.gather(*coros)

    semaphore = asyncio.Semaphore(n)

    return await asyncio.gather(*(gated_coro(semaphore, c) for c in coros))


def accepts_run_manager(callable: Callable[..., Any]) -> bool:  # noqa: A002
    """检查可调用对象是否接受 run_manager 参数。

    Args:
        callable: 要检查的可调用对象。

    Returns:
        如果可调用对象接受 run_manager 参数则为 `True`，否则为 `False`。
    """
    try:
        return _signature_parameters(callable).get("run_manager") is not None
    except ValueError:
        return False


def accepts_config(callable: Callable[..., Any]) -> bool:  # noqa: A002
    """检查可调用对象是否接受 config 参数。

    Args:
        callable: 要检查的可调用对象。

    Returns:
        如果可调用对象接受 config 参数则为 `True`，否则为 `False`。
    """
    try:
        return _signature_parameters(callable).get("config") is not None
    except ValueError:
        return False


def accepts_context(callable: Callable[..., Any]) -> bool:  # noqa: A002
    """检查可调用对象是否接受 context 参数。

    Args:
        callable: 要检查的可调用对象。

    Returns:
        如果可调用对象接受 context 参数则为 `True`，否则为 `False`。
    """
    try:
        return _signature_parameters(callable).get("context") is not None
    except ValueError:
        return False


def asyncio_accepts_context() -> bool:
    """检查 asyncio.create_task 是否接受 `context` 参数。

    Returns:
        如果 `asyncio.create_task` 接受 context 参数则为 `True`，否则为 `False`。
    """
    return sys.version_info >= (3, 11)


_T = TypeVar("_T")


def coro_with_context(
    coro: Awaitable[_T], context: Context, *, create_task: bool = False
) -> Awaitable[_T]:
    """使用上下文等待协程。

    Args:
        coro: 要等待的协程。
        context: 要使用的上下文。
        create_task: 为兼容性保留；此辅助函数始终创建任务。

    Returns:
        带有上下文的协程。
    """
    if asyncio_accepts_context():
        return asyncio.create_task(coro, context=context)  # type: ignore[arg-type,call-arg,unused-ignore]
    del create_task
    return context.run(asyncio.create_task, coro)  # type: ignore[arg-type]


class IsLocalDict(ast.NodeVisitor):
    """检查名称是否为本地字典。"""

    def __init__(self, name: str, keys: set[str]) -> None:
        """初始化访问器。

        Args:
            name: 要检查的名称。
            keys: 要填充的键。
        """
        self.name = name
        self.keys = keys

    @override
    def visit_Subscript(self, node: ast.Subscript) -> None:
        """访问下标节点。

        Args:
            node: 要访问的节点。
        """
        if (
            isinstance(node.ctx, ast.Load)
            and isinstance(node.value, ast.Name)
            and node.value.id == self.name
            and isinstance(node.slice, ast.Constant)
            and isinstance(node.slice.value, str)
        ):
                        # we've found a subscript access on the name we're looking for
            self.keys.add(node.slice.value)

    @override
    def visit_Call(self, node: ast.Call) -> None:
        """访问调用节点。

        Args:
            node: 要访问的节点。
        """
        if (
            isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == self.name
            and node.func.attr == "get"
            and len(node.args) in {1, 2}
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
        ):
                        # we've found a .get() call on the name we're looking for
            self.keys.add(node.args[0].value)


class IsFunctionArgDict(ast.NodeVisitor):
    """检查函数的第一个参数是否为字典。"""

    def __init__(self) -> None:
        """创建一个 IsFunctionArgDict 访问器。"""
        self.keys: set[str] = set()

    @override
    def visit_Lambda(self, node: ast.Lambda) -> None:
        """访问 lambda 函数。

        Args:
            node: 要访问的节点。
        """
        if not node.args.args:
            return
        input_arg_name = node.args.args[0].arg
        IsLocalDict(input_arg_name, self.keys).visit(node.body)

    @override
    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """访问函数定义。

        Args:
            node: 要访问的节点。
        """
        if not node.args.args:
            return
        input_arg_name = node.args.args[0].arg
        IsLocalDict(input_arg_name, self.keys).visit(node)

    @override
    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        """访问异步函数定义。

        Args:
            node: 要访问的节点。
        """
        if not node.args.args:
            return
        input_arg_name = node.args.args[0].arg
        IsLocalDict(input_arg_name, self.keys).visit(node)


class NonLocals(ast.NodeVisitor):
    """获取被访问的非局部变量。"""

    def __init__(self) -> None:
        """创建一个 NonLocals 访问器。"""
        self.loads: set[str] = set()
        self.stores: set[str] = set()

    @override
    def visit_Name(self, node: ast.Name) -> None:
        """访问名称节点。

        Args:
            node: 要访问的节点。
        """
        if isinstance(node.ctx, ast.Load):
            self.loads.add(node.id)
        elif isinstance(node.ctx, ast.Store):
            self.stores.add(node.id)

    @override
    def visit_Attribute(self, node: ast.Attribute) -> None:
        """访问属性节点。

        Args:
            node: 要访问的节点。
        """
        if isinstance(node.ctx, ast.Load):
            parent = node.value
            attr_expr = node.attr
            while isinstance(parent, ast.Attribute):
                attr_expr = parent.attr + "." + attr_expr
                parent = parent.value
            if isinstance(parent, ast.Name):
                self.loads.add(parent.id + "." + attr_expr)
                self.loads.discard(parent.id)
            elif isinstance(parent, ast.Call):
                if isinstance(parent.func, ast.Name):
                    self.loads.add(parent.func.id)
                else:
                    parent = parent.func
                    attr_expr = ""
                    while isinstance(parent, ast.Attribute):
                        if attr_expr:
                            attr_expr = parent.attr + "." + attr_expr
                        else:
                            attr_expr = parent.attr
                        parent = parent.value
                    if isinstance(parent, ast.Name):
                        self.loads.add(parent.id + "." + attr_expr)


class FunctionNonLocals(ast.NodeVisitor):
    """获取函数被访问的非局部变量。"""

    def __init__(self) -> None:
        """创建一个 FunctionNonLocals 访问器。"""
        self.nonlocals: set[str] = set()

    @override
    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """访问函数定义。

        Args:
            node: 要访问的节点。
        """
        visitor = NonLocals()
        visitor.visit(node)
        self.nonlocals.update(visitor.loads - visitor.stores)

    @override
    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        """访问异步函数定义。

        Args:
            node: 要访问的节点。
        """
        visitor = NonLocals()
        visitor.visit(node)
        self.nonlocals.update(visitor.loads - visitor.stores)

    @override
    def visit_Lambda(self, node: ast.Lambda) -> None:
        """访问 lambda 函数。

        Args:
            node: 要访问的节点。
        """
        visitor = NonLocals()
        visitor.visit(node)
        self.nonlocals.update(visitor.loads - visitor.stores)


class GetLambdaSource(ast.NodeVisitor):
    """获取 lambda 函数的源代码。"""

    def __init__(self) -> None:
        """初始化访问器。"""
        self.source: str | None = None
        self.count = 0

    @override
    def visit_Lambda(self, node: ast.Lambda) -> None:
        """访问 lambda 函数。

        Args:
            node: 要访问的节点。
        """
        self.count += 1
        if hasattr(ast, "unparse"):
            self.source = ast.unparse(node)


def get_function_first_arg_dict_keys(func: Callable[..., Any]) -> list[str] | None:
    """获取函数第一个参数的键（如果它是字典）。

    Args:
        func: 要检查的函数。

    Returns:
        如果第一个参数是字典则返回其键，否则返回 None。
    """
    try:
        code = inspect.getsource(func)
        tree = ast.parse(textwrap.dedent(code))
        visitor = IsFunctionArgDict()
        visitor.visit(tree)
        return sorted(visitor.keys) if visitor.keys else None
    except (SyntaxError, TypeError, OSError, SystemError):
        return None


def get_lambda_source(func: Callable[..., Any]) -> str | None:
    """获取 lambda 函数的源代码。

    Args:
        func: 可以是 lambda 函数的可调用对象。

    Returns:
        lambda 函数的源代码。
    """
    try:
        name = func.__name__ if func.__name__ != "<lambda>" else None
    except AttributeError:
        name = None
    try:
        code = inspect.getsource(func)
        tree = ast.parse(textwrap.dedent(code))
        visitor = GetLambdaSource()
        visitor.visit(tree)
    except (SyntaxError, TypeError, OSError, SystemError):
        return name
    return visitor.source if visitor.count == 1 else name


@lru_cache(maxsize=256)
def get_function_nonlocals(func: Callable[..., Any]) -> list[Any]:
    """获取函数访问的非局部变量。

    Args:
        func: 要检查的函数。

    Returns:
        函数访问的非局部变量。
    """
    try:
        code = inspect.getsource(func)
        tree = ast.parse(textwrap.dedent(code))
        visitor = FunctionNonLocals()
        visitor.visit(tree)
        values: list[Any] = []
        closure = (
            inspect.getclosurevars(func.__wrapped__)
            if hasattr(func, "__wrapped__") and callable(func.__wrapped__)
            else inspect.getclosurevars(func)
        )
        candidates = {**closure.globals, **closure.nonlocals}
        for k, v in candidates.items():
            if k in visitor.nonlocals:
                values.append(v)
            for kk in visitor.nonlocals:
                if "." in kk and kk.startswith(k):
                    vv = v
                    for part in kk.split(".")[1:]:
                        if vv is None:
                            break
                        try:
                            vv = getattr(vv, part)
                        except AttributeError:
                            break
                    else:
                        values.append(vv)
    except (SyntaxError, TypeError, OSError, SystemError):
        return []

    return values


def indent_lines_after_first(text: str, prefix: str) -> str:
    """缩进文本中除第一行外的所有行。

    Args:
        text: 要缩进的文本。
        prefix: 用于确定缩进空格数。

    Returns:
        缩进后的文本。
    """
    n_spaces = len(prefix)
    spaces = " " * n_spaces
    lines = text.splitlines()
    return "\n".join([lines[0]] + [spaces + line for line in lines[1:]])


class AddableDict(dict[str, Any]):
    """可与其他字典相加的字典。"""

    def __add__(self, other: AddableDict) -> AddableDict:
        """将字典添加到此字典中。

        Args:
            other: 要添加的另一个字典。

        Returns:
            两个字典相加后的结果字典。

        Raises:
            TypeError: 如果共享键的值类型不兼容。
        """
        chunk = AddableDict(self)
        for key in other:
            if key not in chunk or chunk[key] is None:
                chunk[key] = other[key]
            elif other[key] is not None:
                try:
                    added = chunk[key] + other[key]
                except TypeError as exc:
                    msg = (
                        f"Cannot add incompatible types for key {key!r}: "
                        f"{type(chunk[key]).__name__!r} and "
                        f"{type(other[key]).__name__!r}."
                    )
                    raise TypeError(msg) from exc
                chunk[key] = added
        return chunk

    def __radd__(self, other: AddableDict) -> AddableDict:
        """将此字典添加到另一个字典中。

        Args:
            other: 要被添加的另一个字典。

        Returns:
            两个字典相加后的结果字典。

        Raises:
            TypeError: 如果共享键的值类型不兼容。
        """
        chunk = AddableDict(other)
        for key in self:
            if key not in chunk or chunk[key] is None:
                chunk[key] = self[key]
            elif self[key] is not None:
                try:
                    added = chunk[key] + self[key]
                except TypeError as exc:
                    msg = (
                        f"Cannot add incompatible types for key {key!r}: "
                        f"{type(chunk[key]).__name__!r} and "
                        f"{type(self[key]).__name__!r}."
                    )
                    raise TypeError(msg) from exc
                chunk[key] = added
        return chunk


_T_co = TypeVar("_T_co", covariant=True)
_T_contra = TypeVar("_T_contra", contravariant=True)


class SupportsAdd(Protocol[_T_contra, _T_co]):
    """支持加法的对象协议。"""

    def __add__(self, x: _T_contra, /) -> _T_co:
        """将对象添加到另一个对象中。"""


Addable = TypeVar("Addable", bound=SupportsAdd[Any, Any])


def add(addables: Iterable[Addable]) -> Addable | None:
    """将一系列可加对象相加。

    Args:
        addables: 要相加的可加对象。

    Returns:
        可加对象相加的结果。
    """
    final: Addable | None = None
    for chunk in addables:
        final = chunk if final is None else final + chunk
    return final


async def aadd(addables: AsyncIterable[Addable]) -> Addable | None:
    """异步地将一系列可加对象相加。

    Args:
        addables: 要相加的可加对象。

    Returns:
        可加对象相加的结果。
    """
    final: Addable | None = None
    async for chunk in addables:
        final = chunk if final is None else final + chunk
    return final


class ConfigurableField(NamedTuple):
    """用户可配置的字段。"""

    id: str
    """字段的唯一标识符。"""

    name: str | None = None
    """字段的名称。"""

    description: str | None = None
    """字段的描述。"""

    annotation: Any | None = None
    """字段的注解。"""

    is_shared: bool = False
    """字段是否共享。"""

    @override
    def __hash__(self) -> int:
        return hash((self.id, self.annotation))


class ConfigurableFieldSingleOption(NamedTuple):
    """用户可配置的字段，带有默认值。"""

    id: str
    """字段的唯一标识符。"""

    options: Mapping[str, Any]
    """字段的选项。"""

    default: str
    """字段的默认值。"""

    name: str | None = None
    """字段的名称。"""

    description: str | None = None
    """字段的描述。"""

    is_shared: bool = False
    """字段是否共享。"""

    @override
    def __hash__(self) -> int:
        return hash((self.id, tuple(self.options.keys()), self.default))


class ConfigurableFieldMultiOption(NamedTuple):
    """用户可配置的字段，带有多个默认值。"""

    id: str
    """字段的唯一标识符。"""

    options: Mapping[str, Any]
    """字段的选项。"""

    default: Sequence[str]
    """字段的默认值。"""

    name: str | None = None
    """字段的名称。"""

    description: str | None = None
    """字段的描述。"""

    is_shared: bool = False
    """字段是否共享。"""

    @override
    def __hash__(self) -> int:
        return hash((self.id, tuple(self.options.keys()), tuple(self.default)))


AnyConfigurableField = (
    ConfigurableField | ConfigurableFieldSingleOption | ConfigurableFieldMultiOption
)


class ConfigurableFieldSpec(NamedTuple):
    """用户可配置的字段。它是字段的规范。"""

    id: str
    """字段的唯一标识符。"""

    annotation: Any
    """字段的注释。"""

    name: str | None = None
    """字段的名称。"""

    description: str | None = None
    """字段的描述。"""

    default: Any = None
    """字段的默认值。"""

    is_shared: bool = False
    """字段是否共享。"""

    dependencies: list[str] | None = None
    """字段的依赖项。"""


def get_unique_config_specs(
    specs: Iterable[ConfigurableFieldSpec],
) -> list[ConfigurableFieldSpec]:
    """从一组配置规范中获取唯一的配置规范。

    Args:
        specs: 配置规范。

    Returns:
        唯一的配置规范。

    Raises:
        ValueError: 如果 Runnable 序列包含冲突的配置规范。
    """
    grouped = groupby(
        sorted(specs, key=lambda s: (s.id, *(s.dependencies or []))), lambda s: s.id
    )
    unique: list[ConfigurableFieldSpec] = []
    for spec_id, dupes in grouped:
        first = next(dupes)
        others = list(dupes)
        if len(others) == 0 or all(o == first for o in others):
            unique.append(first)
        else:
            msg = (
                "RunnableSequence contains conflicting config specs"
                f"for {spec_id}: {[first, *others]}"
            )
            raise ValueError(msg)
    return unique


class _RootEventFilter:
    def __init__(
        self,
        *,
        include_names: Sequence[str] | None = None,
        include_types: Sequence[str] | None = None,
        include_tags: Sequence[str] | None = None,
        exclude_names: Sequence[str] | None = None,
        exclude_types: Sequence[str] | None = None,
        exclude_tags: Sequence[str] | None = None,
    ) -> None:
        """用于在 astream_events 实现中过滤根事件的工具。

        这只是将参数绑定到命名空间，以减少在 astream_events 实现中的输入量。
        """
        self.include_names = include_names
        self.include_types = include_types
        self.include_tags = include_tags
        self.exclude_names = exclude_names
        self.exclude_types = exclude_types
        self.exclude_tags = exclude_tags

    def include_event(self, event: StreamEvent, root_type: str) -> bool:
        """确定是否包含事件。"""
        if (
            self.include_names is None
            and self.include_types is None
            and self.include_tags is None
        ):
            include = True
        else:
            include = False

        event_tags = event.get("tags") or []

        if self.include_names is not None:
            include = include or event["name"] in self.include_names
        if self.include_types is not None:
            include = include or root_type in self.include_types
        if self.include_tags is not None:
            include = include or any(tag in self.include_tags for tag in event_tags)

        if self.exclude_names is not None:
            include = include and event["name"] not in self.exclude_names
        if self.exclude_types is not None:
            include = include and root_type not in self.exclude_types
        if self.exclude_tags is not None:
            include = include and all(
                tag not in self.exclude_tags for tag in event_tags
            )

        return include


def is_async_generator(
    func: Any,
) -> TypeGuard[Callable[..., AsyncIterator[Any]]]:
    """检查函数是否为异步生成器。

    Args:
        func: 要检查的函数。

    Returns:
        `True` 表示函数是异步生成器，否则为 `False`。
    """
    return inspect.isasyncgenfunction(func) or (
        hasattr(func, "__call__")  # noqa: B004
        and inspect.isasyncgenfunction(func.__call__)
    )


def is_async_callable(
    func: Any,
) -> TypeGuard[Callable[..., Awaitable[Any]]]:
    """检查函数是否为异步函数。

    Args:
        func: 要检查的函数。

    Returns:
        `True` 表示函数是异步函数，否则为 `False`。
    """
    return inspect.iscoroutinefunction(func) or (
        hasattr(func, "__call__")  # noqa: B004
        and inspect.iscoroutinefunction(func.__call__)
    )

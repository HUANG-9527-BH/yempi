"""Runnable 对象的配置工具。"""

from __future__ import annotations

import asyncio

# Cannot move uuid to TYPE_CHECKING as RunnableConfig is used in Pydantic models
import uuid  # noqa: TC003
import warnings
from collections.abc import (
    Awaitable,
    Callable,
    Generator,
    Iterable,
    Iterator,
    Mapping,
    Sequence,
)
from concurrent.futures import Executor, Future, ThreadPoolExecutor
from contextlib import contextmanager
from contextvars import Context, ContextVar, Token, copy_context
from functools import partial
from typing import (
    TYPE_CHECKING,
    Any,
    ParamSpec,
    TypeVar,
    cast,
)

from typing_extensions import TypedDict

from langchain_core.callbacks.manager import AsyncCallbackManager, CallbackManager
from langchain_core.runnables.utils import (
    Input,
    Output,
    accepts_config,
    accepts_run_manager,
)

if TYPE_CHECKING:
    from langchain_core.callbacks.base import BaseCallbackManager, Callbacks
    from langchain_core.callbacks.manager import (
        AsyncCallbackManagerForChainRun,
        CallbackManagerForChainRun,
    )
else:
        # Pydantic validates through typed dicts, but
        # the callbacks need forward refs updated
    Callbacks = list | Any | None


class EmptyDict(TypedDict, total=False):
    """空字典类型。"""


class RunnableConfig(TypedDict, total=False):
    """Runnable 的配置。

    !!! note 自定义值

        TypedDict 设置了 total=False，目的是：

        - Allow partial configs to be created and merged together via `merge_configs`
        - Support config propagation from parent to child runnables via
            `var_child_runnable_config` (a `ContextVar` that automatically passes
            config down the call stack without explicit parameter passing), where
            configs are merged rather than replaced

        !!! example

            ```python
            # Parent sets tags
            chain.invoke(input, config={"tags": ["parent"]})
            # Child automatically inherits and can add:
            # ensure_config({"tags": ["child"]}) -> {"tags": ["parent", "child"]}
            ```
    """

    tags: list[str]
    """此调用及任何子调用的标签（例如，链调用 LLM）。

    你可以使用这些标签来过滤调用。
    """

    metadata: dict[str, Any]
    """此调用及任何子调用的元数据（例如，链调用 LLM）。

    键应为字符串，值应为 JSON 可序列化的。
    """

    callbacks: Callbacks
    """此调用及任何子调用（例如链调用大语言模型）的回调。

    标签会传递给所有回调，元数据仅传递给 handle*Start 回调。
    """

    run_name: str
    """此调用的追踪运行名称。

    默认为类名。"""

    max_concurrency: int | None
    """最多并行调用次数。

    若未提供，默认使用 `ThreadPoolExecutor` 的默认值。
    """

    recursion_limit: int
    """调用递归的最大次数。

    若未提供，默认为 `25`。
    """

    configurable: dict[str, Any]
    """Runtime values for attributes previously made configurable on this `Runnable`,
    or sub-`Runnable` objects, through `configurable_fields` or
    `configurable_alternatives`.

    Check `output_schema` for a description of the attributes that have been made
    configurable.
    """

    run_id: uuid.UUID | None
    """此调用的追踪运行的唯一标识符。

    如果未提供，将生成一个新的 UUID。
    """


CONFIG_KEYS = [
    "tags",
    "metadata",
    "callbacks",
    "run_name",
    "max_concurrency",
    "recursion_limit",
    "configurable",
    "run_id",
]

COPIABLE_KEYS = [
    "tags",
    "metadata",
    "callbacks",
    "configurable",
]


# Users are expected to use the `context` API with a context object
# (which does not get traced)
CONFIGURABLE_TO_TRACING_METADATA_EXCLUDED_KEYS = frozenset(("api_key",))


def _get_langsmith_inheritable_metadata_from_config(
    config: RunnableConfig,
) -> dict[str, Any] | None:
    """获取 LangSmith 专用的可继承元数据默认值，源自配置。"""
    configurable = config.get("configurable") or {}
    metadata = {
        key: value
        for key, value in configurable.items()
        if not key.startswith("__")
        and isinstance(value, (str, int, float, bool))
        and key not in config.get("metadata", {})
        and key not in CONFIGURABLE_TO_TRACING_METADATA_EXCLUDED_KEYS
    }
    return metadata or None


DEFAULT_RECURSION_LIMIT = 25


var_child_runnable_config: ContextVar[RunnableConfig | None] = ContextVar(
    "child_runnable_config", default=None
)


# This is imported and used in langgraph, so don't break.
def _set_config_context(
    config: RunnableConfig,
) -> tuple[Token[RunnableConfig | None], dict[str, Any] | None]:
    """设置子 Runnable 的配置和追踪上下文。

    Args:
        config: 要设置的配置。

    Returns:
        重置配置和先前追踪上下文的 token。
    """
        # Deferred to avoid importing langsmith at module level (~132ms).
    from langsmith.run_helpers import (  # noqa: PLC0415
        _set_tracing_context,
        get_tracing_context,
    )

    from langchain_core.tracers.langchain import LangChainTracer  # noqa: PLC0415

    config_token = var_child_runnable_config.set(config)
    current_context = None
    if (
        (callbacks := config.get("callbacks"))
        and (
            parent_run_id := getattr(callbacks, "parent_run_id", None)
        )  # Is callback manager
        and (
            tracer := next(
                (
                    handler
                    for handler in getattr(callbacks, "handlers", [])
                    if isinstance(handler, LangChainTracer)
                ),
                None,
            )
        )
        and (run := tracer.run_map.get(str(parent_run_id)))
    ):
        current_context = get_tracing_context()
        _set_tracing_context({"parent": run})
    return config_token, current_context


@contextmanager
def set_config_context(config: RunnableConfig) -> Generator[Context, None, None]:
    """设置子 Runnable 的配置和追踪上下文。

    Args:
        config: 要设置的配置。

    Yields:
        配置上下文。
    """
        # Deferred to avoid importing langsmith at module level (~132ms).
    from langsmith.run_helpers import _set_tracing_context  # noqa: PLC0415

    ctx = copy_context()
    config_token, _ = ctx.run(_set_config_context, config)
    try:
        yield ctx
    finally:
        ctx.run(var_child_runnable_config.reset, config_token)
        ctx.run(
            _set_tracing_context,
            {
                "parent": None,
                "project_name": None,
                "tags": None,
                "metadata": None,
                "enabled": None,
                "client": None,
            },
        )


def ensure_config(config: RunnableConfig | None = None) -> RunnableConfig:
    """确保配置是一个包含所有键的字典。

    Args:
        config: 要确保的配置。

    Returns:
        确保后的配置。
    """
    empty = RunnableConfig(
        tags=[],
        metadata={},
        callbacks=None,
        recursion_limit=DEFAULT_RECURSION_LIMIT,
        configurable={},
    )
    if var_config := var_child_runnable_config.get():
        empty.update(
            cast(
                "RunnableConfig",
                {
                    k: v.copy() if k in COPIABLE_KEYS else v  # type: ignore[attr-defined]
                    for k, v in var_config.items()
                    if v is not None
                },
            )
        )
    if config is not None:
        empty.update(
            cast(
                "RunnableConfig",
                {
                    k: v.copy() if k in COPIABLE_KEYS else v  # type: ignore[attr-defined]
                    for k, v in config.items()
                    if v is not None and k in CONFIG_KEYS
                },
            )
        )
    if config is not None:
        for k, v in config.items():
            if k not in CONFIG_KEYS and v is not None:
                empty["configurable"][k] = v
    for configurable_key in ("model", "checkpoint_ns"):
        if (
            isinstance(
                configurable_value := empty.get("configurable", {}).get(
                    configurable_key
                ),
                str,
            )
            and configurable_key not in empty["metadata"]
        ):
            empty["metadata"][configurable_key] = configurable_value
    return empty


def get_config_list(
    config: RunnableConfig | Sequence[RunnableConfig] | None, length: int
) -> list[RunnableConfig]:
    """从单个配置或配置列表中获取配置列表。

     对于重写 batch() 或 abatch() 的子类很有用。

    Args:
        config: 配置或配置列表。
        length: 列表长度。

    Returns:
        配置列表。

    Raises:
        ValueError: 如果列表长度不等于输入长度。

    """
    if length < 0:
        msg = f"length must be >= 0, but got {length}"
        raise ValueError(msg)
    if isinstance(config, Sequence) and len(config) != length:
        msg = (
            f"config must be a list of the same length as inputs, "
            f"but got {len(config)} configs for {length} inputs"
        )
        raise ValueError(msg)

    if isinstance(config, Sequence):
        return list(map(ensure_config, config))
    if length > 1 and isinstance(config, dict) and config.get("run_id") is not None:
        warnings.warn(
            "Provided run_id be used only for the first element of the batch.",
            category=RuntimeWarning,
            stacklevel=3,
        )
        subsequent = cast(
            "RunnableConfig", {k: v for k, v in config.items() if k != "run_id"}
        )
        return [
            ensure_config(subsequent) if i else ensure_config(config)
            for i in range(length)
        ]
    return [ensure_config(config) for i in range(length)]


def patch_config(
    config: RunnableConfig | None,
    *,
    callbacks: BaseCallbackManager | None = None,
    recursion_limit: int | None = None,
    max_concurrency: int | None = None,
    run_name: str | None = None,
    configurable: dict[str, Any] | None = None,
) -> RunnableConfig:
    """使用新值修补配置。

    Args:
        config: 要修补的配置。
        callbacks: 要设置的回调。
        recursion_limit: 要设置的递归限制。
        max_concurrency: 要设置的最大并发数。
        run_name: 要设置的运行名称。
        configurable: 要设置的可配置项。

    Returns:
        修补后的配置。
    """
    config = ensure_config(config)
    if callbacks is not None:
                # If we're replacing callbacks, we need to unset run_name
                # As that should apply only to the same run as the original callbacks
        config["callbacks"] = callbacks
        if "run_name" in config:
            del config["run_name"]
        if "run_id" in config:
            del config["run_id"]
    if recursion_limit is not None:
        config["recursion_limit"] = recursion_limit
    if max_concurrency is not None:
        config["max_concurrency"] = max_concurrency
    if run_name is not None:
        config["run_name"] = run_name
    if configurable is not None:
        config["configurable"] = {**config.get("configurable", {}), **configurable}
    return config


def _merge_metadata_dicts(
    base: Mapping[str, Any], incoming: Mapping[str, Any]
) -> dict[str, Any]:
    """合并元数据字典，仅累积 `lc_versions` 嵌套映射。

    Metadata generally uses last-writer-wins semantics. The LangChain-owned
    `lc_versions` key is the only nested mapping that accumulates across merges
    so package versions from core, `langchain`, and partner packages coexist.

    Args:
        base: 基础元数据字典。

            此处的值除非被 `incoming` 覆盖，否则将被保留。
        incoming: 要合并到顶层的元数据字典。

            冲突时其值具有优先权。

    Returns:
        一个新的合并后的字典。
    """
    merged = {**base, **incoming}
    base_versions = base.get("lc_versions")
    incoming_versions = incoming.get("lc_versions")
    if isinstance(base_versions, Mapping) and isinstance(incoming_versions, Mapping):
        merged["lc_versions"] = {**base_versions, **incoming_versions}
    elif isinstance(incoming_versions, Mapping):
        merged["lc_versions"] = {**incoming_versions}
    elif "lc_versions" not in incoming and isinstance(base_versions, Mapping):
        merged["lc_versions"] = {**base_versions}
    return merged


def merge_configs(*configs: RunnableConfig | None) -> RunnableConfig:
    """将多个配置合并为一个。

    Args:
        *configs: 要合并的配置。

    Returns:
        合并后的配置。
    """
    base: RunnableConfig = {}
        # 尽管键不是字面量，但这样写是正确的
        # 因为两个字典都是同一种类型
    for config in (ensure_config(c) for c in configs if c is not None):
        for key in config:
            if key == "metadata":
                base["metadata"] = _merge_metadata_dicts(
                    base.get("metadata", {}),
                    config.get("metadata") or {},
                )
            elif key == "tags":
                base["tags"] = sorted(
                    set(base.get("tags", []) + (config.get("tags") or [])),
                )
            elif key == "configurable":
                base["configurable"] = {
                    **base.get("configurable", {}),
                    **(config.get("configurable") or {}),
                }
            elif key == "callbacks":
                base_callbacks = base.get("callbacks")
                these_callbacks = config["callbacks"]
                                # 回调可以是 None、list[handler] 或 manager
                                # 因此合并两个回调值有 6 种情况
                if isinstance(these_callbacks, list):
                    if base_callbacks is None:
                        base["callbacks"] = these_callbacks.copy()
                    elif isinstance(base_callbacks, list):
                        base["callbacks"] = base_callbacks + these_callbacks
                    else:
                                                # base_callbacks 是一个 manager
                        mngr = base_callbacks.copy()
                        for callback in these_callbacks:
                            mngr.add_handler(callback, inherit=True)
                        base["callbacks"] = mngr
                elif these_callbacks is not None:
                                        # these_callbacks 是一个 manager
                    if base_callbacks is None:
                        base["callbacks"] = these_callbacks.copy()
                    elif isinstance(base_callbacks, list):
                        mngr = these_callbacks.copy()
                        for callback in base_callbacks:
                            mngr.add_handler(callback, inherit=True)
                        base["callbacks"] = mngr
                    else:
                                                # base_callbacks 也是一个 manager
                        base["callbacks"] = base_callbacks.merge(these_callbacks)
            elif key == "recursion_limit":
                if config["recursion_limit"] != DEFAULT_RECURSION_LIMIT:
                    base["recursion_limit"] = config["recursion_limit"]
            elif key in COPIABLE_KEYS and config[key] is not None:  # type: ignore[literal-required]
                base[key] = config[key].copy()  # type: ignore[literal-required]
            else:
                base[key] = config[key] or base.get(key)  # type: ignore[literal-required]
    return base


def call_func_with_variable_args(
    func: Callable[[Input], Output]
    | Callable[[Input, RunnableConfig], Output]
    | Callable[[Input, CallbackManagerForChainRun], Output]
    | Callable[[Input, CallbackManagerForChainRun, RunnableConfig], Output],
    input: Input,
    config: RunnableConfig,
    run_manager: CallbackManagerForChainRun | None = None,
    **kwargs: Any,
) -> Output:
    """调用一个可能可选地接受 run_manager 和/或 config 的函数。

    Args:
        func: 要调用的函数。
        input: 函数的输入。
        config: 要传递给函数的配置。
        run_manager: 要传递给函数的运行管理器。
        **kwargs: 要传递给函数的关键字参数。

    Returns:
        函数的输出。
    """
    if accepts_config(func):
        if run_manager is not None:
            kwargs["config"] = patch_config(config, callbacks=run_manager.get_child())
        else:
            kwargs["config"] = config
    if run_manager is not None and accepts_run_manager(func):
        kwargs["run_manager"] = run_manager
    return func(input, **kwargs)  # type: ignore[call-arg]


def acall_func_with_variable_args(
    func: Callable[[Input], Awaitable[Output]]
    | Callable[[Input, RunnableConfig], Awaitable[Output]]
    | Callable[[Input, AsyncCallbackManagerForChainRun], Awaitable[Output]]
    | Callable[
        [Input, AsyncCallbackManagerForChainRun, RunnableConfig], Awaitable[Output]
    ],
    input: Input,
    config: RunnableConfig,
    run_manager: AsyncCallbackManagerForChainRun | None = None,
    **kwargs: Any,
) -> Awaitable[Output]:
    """异步调用一个可能可选地接受 run_manager 和/或 config 的函数。

    Args:
        func: 要调用的函数。
        input: 函数的输入。
        config: 要传递给函数的配置。
        run_manager: 要传递给函数的运行管理器。
        **kwargs: 要传递给函数的关键字参数。

    Returns:
        函数的输出。
    """
    if accepts_config(func):
        if run_manager is not None:
            kwargs["config"] = patch_config(config, callbacks=run_manager.get_child())
        else:
            kwargs["config"] = config
    if run_manager is not None and accepts_run_manager(func):
        kwargs["run_manager"] = run_manager
    return func(input, **kwargs)  # type: ignore[call-arg]


def get_callback_manager_for_config(config: RunnableConfig) -> CallbackManager:
    """获取配置对应的回调管理器。

    Args:
        config: 配置。

    Returns:
        回调管理器。
    """
    return CallbackManager.configure(
        inheritable_callbacks=config.get("callbacks"),
        inheritable_tags=config.get("tags"),
        inheritable_metadata=config.get("metadata"),
        langsmith_inheritable_metadata=_get_langsmith_inheritable_metadata_from_config(
            config
        ),
    )


def get_async_callback_manager_for_config(
    config: RunnableConfig,
) -> AsyncCallbackManager:
    """获取配置对应的异步回调管理器。

    Args:
        config: 配置。

    Returns:
        异步回调管理器。
    """
    return AsyncCallbackManager.configure(
        inheritable_callbacks=config.get("callbacks"),
        inheritable_tags=config.get("tags"),
        inheritable_metadata=config.get("metadata"),
        langsmith_inheritable_metadata=_get_langsmith_inheritable_metadata_from_config(
            config
        ),
    )


P = ParamSpec("P")
T = TypeVar("T")


class ContextThreadPoolExecutor(ThreadPoolExecutor):
    """复制上下文到子线程的 ThreadPoolExecutor。"""

    def submit(  # type: ignore[override]
        self,
        func: Callable[P, T],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> Future[T]:
        """向执行器提交一个函数。

        Args:
            func: 要提交的函数。
            *args: 函数的位置参数。
            **kwargs: 函数的关键字参数。

        Returns:
            函数的未来对象。
        """
        return super().submit(
            cast("Callable[..., T]", partial(copy_context().run, func, *args, **kwargs))
        )

    def map(
        self,
        fn: Callable[..., T],
        *iterables: Iterable[Any],
        **kwargs: Any,
    ) -> Iterator[T]:
        """将函数映射到多个可迭代对象上。

        Args:
            fn: 要映射的函数。
            *iterables: 要遍历的可迭代对象。
            timeout: 映射的超时时间。
            chunksize: 映射的分块大小。

        Returns:
            映射函数的迭代器。
        """
        contexts = [copy_context() for _ in range(len(iterables[0]))]  # type: ignore[arg-type]

        def _wrapped_fn(*args: Any) -> T:
            return contexts.pop().run(fn, *args)

        return super().map(
            _wrapped_fn,
            *iterables,
            **kwargs,
        )


@contextmanager
def get_executor_for_config(
    config: RunnableConfig | None,
) -> Generator[Executor, None, None]:
    """获取配置对应的执行器。

    Args:
        config: 配置。

    Yields:
        执行器。
    """
    config = config or {}
    with ContextThreadPoolExecutor(
        max_workers=config.get("max_concurrency")
    ) as executor:
        yield executor


async def run_in_executor(
    executor_or_config: Executor | RunnableConfig | None,
    func: Callable[P, T],
    *args: P.args,
    **kwargs: P.kwargs,
) -> T:
    """在执行器中运行一个函数。

    Args:
        executor_or_config: 要运行的执行器或配置。
        func: 要运行的函数。
        *args: 函数的位置参数。
        **kwargs: 函数的关键字参数。

    Returns:
        函数的输出。
    """

    def wrapper() -> T:
        try:
            return func(*args, **kwargs)
        except StopIteration as exc:
                        # StopIteration 不能设置在 asyncio.Future 上
                        # 它会引发 TypeError 并使 Future 永久挂起
                        # 所以我们需要将其转换为 RuntimeError
            raise RuntimeError from exc

    if executor_or_config is None or isinstance(executor_or_config, dict):
                # 使用默认执行器，并复制当前上下文的上下文
        return await asyncio.get_running_loop().run_in_executor(
            None,
            cast("Callable[..., T]", partial(copy_context().run, wrapper)),
        )

    return await asyncio.get_running_loop().run_in_executor(executor_or_config, wrapper)

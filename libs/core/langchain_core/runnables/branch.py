"""根据条件选择要运行的分支的 Runnable。"""

from collections.abc import (
    AsyncIterator,
    Awaitable,
    Callable,
    Iterator,
    Mapping,
    Sequence,
)
from typing import (
    Any,
    cast,
)

from pydantic import ConfigDict
from typing_extensions import override

from langchain_core.runnables.base import (
    Runnable,
    RunnableLike,
    RunnableSerializable,
    coerce_to_runnable,
)
from langchain_core.runnables.config import (
    RunnableConfig,
    ensure_config,
    get_async_callback_manager_for_config,
    get_callback_manager_for_config,
    patch_config,
)
from langchain_core.runnables.utils import (
    ConfigurableFieldSpec,
    Input,
    Output,
    get_unique_config_specs,
)
from langchain_core.utils.pydantic import TypeBaseModel

_MIN_BRANCHES = 2


class RunnableBranch(RunnableSerializable[Input, Output]):
    """根据条件选择要运行的分支的 Runnable。

    The `Runnable` is initialized with a list of `(condition, Runnable)` pairs and
    a default branch.

    处理输入时，第一个求值为 True 的条件会被选中，并在其上运行对应的 `Runnable`。

    如果没有条件求值为 `True`，则在输入上运行默认分支。

    Examples:
        ```python
        from langchain_core.runnables import RunnableBranch

        branch = RunnableBranch(
            (lambda x: isinstance(x, str), lambda x: x.upper()),
            (lambda x: isinstance(x, int), lambda x: x + 1),
            (lambda x: isinstance(x, float), lambda x: x * 2),
            lambda x: "goodbye",
        )

        branch.invoke("hello")  # "HELLO"
        branch.invoke(None)  # "goodbye"
        ```
    """

    branches: Sequence[tuple[Runnable[Input, bool], Runnable[Input, Output]]]
    """一个 `(condition, Runnable)` 对的列表。"""
    default: Runnable[Input, Output]
    """当没有条件满足时运行的 `Runnable`。"""

    def __init__(
        self,
        *branches: tuple[
            Runnable[Input, bool]
            | Callable[[Input], bool]
            | Callable[[Input], Awaitable[bool]],
            RunnableLike[Input, Output],
        ]
        | RunnableLike[Input, Output],
    ) -> None:
        """一个根据条件运行两个分支之一的 `Runnable`。

        Args:
            *branches: 一个 `(condition, Runnable)` 对的列表。
                默认的 `Runnable`，用于在无条件满足时运行。

        Raises:
            ValueError: 如果分支数量少于 `2`。
            TypeError: 如果默认分支不是 `Runnable`、`Callable` 或 `Mapping`。
            TypeError: 如果某个分支不是 `tuple` 或 `list`。
            ValueError: 如果某个分支长度不是 `2`。
        """
        if len(branches) < _MIN_BRANCHES:
            msg = "RunnableBranch requires at least two branches"
            raise ValueError(msg)

        default = branches[-1]

        if not isinstance(
            default,
            (Runnable, Callable, Mapping),  # type: ignore[arg-type]
        ):
            msg = "RunnableBranch default must be Runnable, callable or mapping."
            raise TypeError(msg)

        default_ = coerce_to_runnable(cast("Runnable[Input, Output]", default))

        branches_ = []

        for branch in branches[:-1]:
            if not isinstance(branch, (tuple, list)):
                msg = (
                    f"RunnableBranch branches must be "
                    f"tuples or lists, not {type(branch)}"
                )
                raise TypeError(msg)

            if len(branch) != _MIN_BRANCHES:
                msg = (
                    f"RunnableBranch branches must be "
                    f"tuples or lists of length 2, not {len(branch)}"
                )
                raise ValueError(msg)
            condition, runnable = branch
            condition = cast("Runnable[Input, bool]", coerce_to_runnable(condition))
            runnable = coerce_to_runnable(cast("Runnable[Input, Output]", runnable))
            branches_.append((condition, runnable))

        super().__init__(
            branches=branches_,
            default=default_,
        )

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )

    @classmethod
    def is_lc_serializable(cls) -> bool:
        """返回 `True`，因为此类可序列化。"""
        return True

    @classmethod
    @override
    def get_lc_namespace(cls) -> list[str]:
        """获取 LangChain 对象的命名空间。

        Returns:
            `["langchain", "schema", "runnable"]`
        """
        return ["langchain", "schema", "runnable"]

    @override
    def get_input_schema(self, config: RunnableConfig | None = None) -> TypeBaseModel:
        runnables = (
            [self.default]
            + [r for _, r in self.branches]
            + [r for r, _ in self.branches]
        )

        for runnable in runnables:
            if runnable.get_input_jsonschema(config).get("type") is not None:
                return runnable.get_input_schema(config)

        return super().get_input_schema(config)

    @property
    @override
    def config_specs(self) -> list[ConfigurableFieldSpec]:
        return get_unique_config_specs(
            spec
            for step in (
                [self.default]
                + [r for _, r in self.branches]
                + [r for r, _ in self.branches]
            )
            for spec in step.config_specs
        )

    @override
    def invoke(
        self, input: Input, config: RunnableConfig | None = None, **kwargs: Any
    ) -> Output:
        """先求值条件，然后委托给 `True` 或 `False` 分支。

        Args:
            input: 传递给 `Runnable` 的输入。
            config: 传递给 `Runnable` 的配置。
            **kwargs: 传递给 `Runnable` 的额外关键字参数。

        Returns:
            运行的分支的输出。
        """
        config = ensure_config(config)
        callback_manager = get_callback_manager_for_config(config)
        run_manager = callback_manager.on_chain_start(
            None,
            input,
            name=config.get("run_name") or self.get_name(),
            run_id=config.pop("run_id", None),
        )

        try:
            for idx, branch in enumerate(self.branches):
                condition, runnable = branch

                expression_value = condition.invoke(
                    input,
                    config=patch_config(
                        config,
                        callbacks=run_manager.get_child(tag=f"condition:{idx + 1}"),
                    ),
                )

                if expression_value:
                    output = runnable.invoke(
                        input,
                        config=patch_config(
                            config,
                            callbacks=run_manager.get_child(tag=f"branch:{idx + 1}"),
                        ),
                        **kwargs,
                    )
                    break
            else:
                output = self.default.invoke(
                    input,
                    config=patch_config(
                        config, callbacks=run_manager.get_child(tag="branch:default")
                    ),
                    **kwargs,
                )
        except BaseException as e:
            run_manager.on_chain_error(e)
            raise
        run_manager.on_chain_end(output)
        return output

    @override
    async def ainvoke(
        self, input: Input, config: RunnableConfig | None = None, **kwargs: Any
    ) -> Output:
        config = ensure_config(config)
        callback_manager = get_async_callback_manager_for_config(config)
        run_manager = await callback_manager.on_chain_start(
            None,
            input,
            name=config.get("run_name") or self.get_name(),
            run_id=config.pop("run_id", None),
        )
        try:
            for idx, branch in enumerate(self.branches):
                condition, runnable = branch

                expression_value = await condition.ainvoke(
                    input,
                    config=patch_config(
                        config,
                        callbacks=run_manager.get_child(tag=f"condition:{idx + 1}"),
                    ),
                )

                if expression_value:
                    output = await runnable.ainvoke(
                        input,
                        config=patch_config(
                            config,
                            callbacks=run_manager.get_child(tag=f"branch:{idx + 1}"),
                        ),
                        **kwargs,
                    )
                    break
            else:
                output = await self.default.ainvoke(
                    input,
                    config=patch_config(
                        config, callbacks=run_manager.get_child(tag="branch:default")
                    ),
                    **kwargs,
                )
        except BaseException as e:
            await run_manager.on_chain_error(e)
            raise
        await run_manager.on_chain_end(output)
        return output

    @override
    def stream(
        self,
        input: Input,
        config: RunnableConfig | None = None,
        **kwargs: Any | None,
    ) -> Iterator[Output]:
        """先求值条件，然后委托给 `True` 或 `False` 分支。

        Args:
            input: 传递给 `Runnable` 的输入。
            config: 传递给 `Runnable` 的配置。
            **kwargs: 传递给 `Runnable` 的额外关键字参数。

        Yields:
            运行的分支的输出。
        """
        config = ensure_config(config)
        callback_manager = get_callback_manager_for_config(config)
        run_manager = callback_manager.on_chain_start(
            None,
            input,
            name=config.get("run_name") or self.get_name(),
            run_id=config.pop("run_id", None),
        )
        final_output: Output | None = None
        final_output_supported = True

        try:
            for idx, branch in enumerate(self.branches):
                condition, runnable = branch

                expression_value = condition.invoke(
                    input,
                    config=patch_config(
                        config,
                        callbacks=run_manager.get_child(tag=f"condition:{idx + 1}"),
                    ),
                )

                if expression_value:
                    for chunk in runnable.stream(
                        input,
                        config=patch_config(
                            config,
                            callbacks=run_manager.get_child(tag=f"branch:{idx + 1}"),
                        ),
                        **kwargs,
                    ):
                        yield chunk
                        if final_output_supported:
                            if final_output is None:
                                final_output = chunk
                            else:
                                try:
                                    final_output = final_output + chunk  # type: ignore[operator]
                                except TypeError:
                                    final_output = None
                                    final_output_supported = False
                    break
            else:
                for chunk in self.default.stream(
                    input,
                    config=patch_config(
                        config,
                        callbacks=run_manager.get_child(tag="branch:default"),
                    ),
                    **kwargs,
                ):
                    yield chunk
                    if final_output_supported:
                        if final_output is None:
                            final_output = chunk
                        else:
                            try:
                                final_output = final_output + chunk  # type: ignore[operator]
                            except TypeError:
                                final_output = None
                                final_output_supported = False
        except BaseException as e:
            run_manager.on_chain_error(e)
            raise
        run_manager.on_chain_end(final_output)

    @override
    async def astream(
        self,
        input: Input,
        config: RunnableConfig | None = None,
        **kwargs: Any | None,
    ) -> AsyncIterator[Output]:
        """先求值条件，然后委托给 `True` 或 `False` 分支。

        Args:
            input: 传递给 `Runnable` 的输入。
            config: 传递给 `Runnable` 的配置。
            **kwargs: 传递给 `Runnable` 的额外关键字参数。

        Yields:
            运行的分支的输出。
        """
        config = ensure_config(config)
        callback_manager = get_async_callback_manager_for_config(config)
        run_manager = await callback_manager.on_chain_start(
            None,
            input,
            name=config.get("run_name") or self.get_name(),
            run_id=config.pop("run_id", None),
        )
        final_output: Output | None = None
        final_output_supported = True

        try:
            for idx, branch in enumerate(self.branches):
                condition, runnable = branch

                expression_value = await condition.ainvoke(
                    input,
                    config=patch_config(
                        config,
                        callbacks=run_manager.get_child(tag=f"condition:{idx + 1}"),
                    ),
                )

                if expression_value:
                    async for chunk in runnable.astream(
                        input,
                        config=patch_config(
                            config,
                            callbacks=run_manager.get_child(tag=f"branch:{idx + 1}"),
                        ),
                        **kwargs,
                    ):
                        yield chunk
                        if final_output_supported:
                            if final_output is None:
                                final_output = chunk
                            else:
                                try:
                                    final_output = final_output + chunk  # type: ignore[operator]
                                except TypeError:
                                    final_output = None
                                    final_output_supported = False
                    break
            else:
                async for chunk in self.default.astream(
                    input,
                    config=patch_config(
                        config,
                        callbacks=run_manager.get_child(tag="branch:default"),
                    ),
                    **kwargs,
                ):
                    yield chunk
                    if final_output_supported:
                        if final_output is None:
                            final_output = chunk
                        else:
                            try:
                                final_output = final_output + chunk  # type: ignore[operator]
                            except TypeError:
                                final_output = None
                                final_output_supported = False
        except BaseException as e:
            await run_manager.on_chain_error(e)
            raise
        await run_manager.on_chain_end(final_output)

"""``RunnablePassthrough`` 的实现。"""

from __future__ import annotations

import asyncio
import inspect
import threading
from collections.abc import Awaitable, Callable
from typing import (
    TYPE_CHECKING,
    Any,
)

from pydantic import RootModel
from typing_extensions import override

from langchain_core.runnables.base import (
    Other,
    Runnable,
    RunnableParallel,
    RunnableSerializable,
    _get_schema_field_definition,
)
from langchain_core.runnables.config import (
    RunnableConfig,
    acall_func_with_variable_args,
    call_func_with_variable_args,
    ensure_config,
    get_executor_for_config,
    patch_config,
)
from langchain_core.runnables.utils import (
    AddableDict,
    ConfigurableFieldSpec,
)
from langchain_core.utils.aiter import atee
from langchain_core.utils.iter import safetee
from langchain_core.utils.pydantic import TypeBaseModel, create_model_v2, get_fields

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Iterator, Mapping

    from langchain_core.callbacks.manager import (
        AsyncCallbackManagerForChainRun,
        CallbackManagerForChainRun,
    )
    from langchain_core.runnables.graph import Graph


def identity(x: Other) -> Other:
    """恒等函数。

    Args:
        x: 输入。

    Returns:
        输出。
    """
    return x


async def aidentity(x: Other) -> Other:
    """异步恒等函数。

    Args:
        x: 输入。

    Returns:
        输出。
    """
    return x


class RunnablePassthrough(RunnableSerializable[Other, Other]):
    """用于原样传递输入或添加额外键的 Runnable。

    This `Runnable` behaves almost like the identity function, except that it
    can be configured to add additional keys to the output, if the input is a
    dict.

    下面的例子展示了如何使用一些简单的链来演示此 `Runnable`
    的工作方式。这些链依赖于简单的 lambdas 以使示例易于执行和实验。

    Examples:
        ```python
        from langchain_core.runnables import (
            RunnableLambda,
            RunnableParallel,
            RunnablePassthrough,
        )

        runnable = RunnableParallel(
            origin=RunnablePassthrough(), modified=lambda x: x + 1
        )

        runnable.invoke(1)  # {'origin': 1, 'modified': 2}


        def fake_llm(prompt: str) -> str:  # Fake LLM for the example
            return "completion"


        chain = RunnableLambda(fake_llm) | {
            "original": RunnablePassthrough(),  # Original LLM output
            "parsed": lambda text: text[::-1],  # Parsing logic
        }

        chain.invoke("hello")  # {'original': 'completion', 'parsed': 'noitelpmoc'}
        ```

    In some cases, it may be useful to pass the input through while adding some
    keys to the output. In this case, you can use the `assign` method:

        ```python
        from langchain_core.runnables import RunnablePassthrough


        def fake_llm(prompt: str) -> str:  # Fake LLM for the example
            return "completion"


        runnable = {
            "llm1": fake_llm,
            "llm2": fake_llm,
        } | RunnablePassthrough.assign(
            total_chars=lambda inputs: len(inputs["llm1"] + inputs["llm2"])
        )

        runnable.invoke("hello")
        # {'llm1': 'completion', 'llm2': 'completion', 'total_chars': 20}
        ```
    """

    input_type: type[Other] | None = None

    func: Callable[[Other], None] | Callable[[Other, RunnableConfig], None] | None = (
        None
    )

    afunc: (
        Callable[[Other], Awaitable[None]]
        | Callable[[Other, RunnableConfig], Awaitable[None]]
        | None
    ) = None

    @override
    def __repr_args__(self) -> Any:
                # 没有此语句，repr(self) 会引发 RecursionError
                # 参见 https://github.com/pydantic/pydantic/issues/7327
        return []

    def __init__(
        self,
        func: Callable[[Other], None]
        | Callable[[Other, RunnableConfig], None]
        | Callable[[Other], Awaitable[None]]
        | Callable[[Other, RunnableConfig], Awaitable[None]]
        | None = None,
        afunc: Callable[[Other], Awaitable[None]]
        | Callable[[Other, RunnableConfig], Awaitable[None]]
        | None = None,
        *,
        input_type: type[Other] | None = None,
        **kwargs: Any,
    ) -> None:
        """创建一个 `RunnablePassthrough`。

        Args:
            func: 用输入调用的函数。
            afunc: 用输入调用的异步函数。
            input_type: 输入的类型。
        """
        if inspect.iscoroutinefunction(func):
            afunc = func
            func = None

        super().__init__(func=func, afunc=afunc, input_type=input_type, **kwargs)

    @classmethod
    @override
    def is_lc_serializable(cls) -> bool:
        """返回 `True`，因为此类可序列化。"""
        return True

    @classmethod
    def get_lc_namespace(cls) -> list[str]:
        """获取 LangChain 对象的命名空间。

        Returns:
            `["langchain", "schema", "runnable"]`
        """
        return ["langchain", "schema", "runnable"]

    @property
    @override
    def InputType(self) -> Any:
        return self.input_type or Any

    @property
    @override
    def OutputType(self) -> Any:
        return self.input_type or Any

    @classmethod
    @override
    def assign(
        cls,
        **kwargs: Runnable[dict[str, Any], Any]
        | Callable[[dict[str, Any]], Any]
        | Mapping[str, Runnable[dict[str, Any], Any] | Callable[[dict[str, Any]], Any]],
    ) -> RunnableAssign:
        """将字典输入与映射参数产生的输出合并。

        Args:
            **kwargs: `Runnable`, `Callable` or a `Mapping` from keys to `Runnable`
                objects or `Callable`s.

        Returns:
            A `Runnable` that merges the `dict` input with the output produced by the
            mapping argument.
        """
        return RunnableAssign(RunnableParallel[dict[str, Any]](kwargs))

    @override
    def invoke(
        self, input: Other, config: RunnableConfig | None = None, **kwargs: Any
    ) -> Other:
        if self.func is not None:
            call_func_with_variable_args(
                self.func, input, ensure_config(config), **kwargs
            )
        return self._call_with_config(identity, input, config)

    @override
    async def ainvoke(
        self,
        input: Other,
        config: RunnableConfig | None = None,
        **kwargs: Any | None,
    ) -> Other:
        if self.afunc is not None:
            await acall_func_with_variable_args(
                self.afunc, input, ensure_config(config), **kwargs
            )
        elif self.func is not None:
            call_func_with_variable_args(
                self.func, input, ensure_config(config), **kwargs
            )
        return await self._acall_with_config(aidentity, input, config)

    @override
    def transform(
        self,
        input: Iterator[Other],
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> Iterator[Other]:
        if self.func is None:
            for chunk in self._transform_stream_with_config(input, identity, config):
                yield chunk
        else:
            final: Other
            got_first_chunk = False

            for chunk in self._transform_stream_with_config(input, identity, config):
                yield chunk

                if not got_first_chunk:
                    final = chunk
                    got_first_chunk = True
                else:
                    try:
                        final = final + chunk  # type: ignore[operator]
                    except TypeError:
                        final = chunk

            if got_first_chunk:
                call_func_with_variable_args(
                    self.func, final, ensure_config(config), **kwargs
                )

    @override
    async def atransform(
        self,
        input: AsyncIterator[Other],
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[Other]:
        if self.afunc is None and self.func is None:
            async for chunk in self._atransform_stream_with_config(
                input, identity, config
            ):
                yield chunk
        else:
            got_first_chunk = False

            async for chunk in self._atransform_stream_with_config(
                input, identity, config
            ):
                yield chunk

                                # 根据定义，函数将在聚合后的输入上操作。
                                # 因此我们将聚合输入，直到到达最后一个块。
                                # 如果输入不可加，则假设我们可以
                                # 如果输入不可相加，则我们假设可以
                                # 仅对最后一个块进行操作。
                if not got_first_chunk:
                    final = chunk
                    got_first_chunk = True
                else:
                    try:
                        final = final + chunk  # type: ignore[operator]
                    except TypeError:
                        final = chunk

            if got_first_chunk:
                config = ensure_config(config)
                if self.afunc is not None:
                    await acall_func_with_variable_args(
                        self.afunc, final, config, **kwargs
                    )
                elif self.func is not None:
                    call_func_with_variable_args(self.func, final, config, **kwargs)

    @override
    def stream(
        self,
        input: Other,
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> Iterator[Other]:
        return self.transform(iter([input]), config, **kwargs)

    @override
    async def astream(
        self,
        input: Other,
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[Other]:
        async def input_aiter() -> AsyncIterator[Other]:
            yield input

        async for chunk in self.atransform(input_aiter(), config, **kwargs):
            yield chunk


_graph_passthrough = RunnablePassthrough[Any]()


class RunnableAssign(RunnableSerializable[dict[str, Any], dict[str, Any]]):
    """可将键值对分配给 `dict[str, Any]` 输入的 Runnable。

    The `RunnableAssign` class takes input dictionaries and, through a
    `RunnableParallel` instance, applies transformations, then combines
    these with the original data, introducing new key-value pairs based
    on the mapper's logic.

    Examples:
        ```python
        # This is a RunnableAssign
        from langchain_core.runnables.passthrough import (
            RunnableAssign,
            RunnableParallel,
        )
        from langchain_core.runnables.base import RunnableLambda


        def add_ten(x: dict[str, int]) -> dict[str, int]:
            return {"added": x["input"] + 10}


        mapper = RunnableParallel(
            {
                "add_step": RunnableLambda(add_ten),
            }
        )

        runnable_assign = RunnableAssign(mapper)

        # Synchronous example
        runnable_assign.invoke({"input": 5})
        # returns {'input': 5, 'add_step': {'added': 15}}

        # Asynchronous example
        await runnable_assign.ainvoke({"input": 5})
        # returns {'input': 5, 'add_step': {'added': 15}}
        ```
    """

        # 理想情况下，我们会将 mapper 的类型定义为 RunnableParallel[dict[str, Any]]
        # 但此写法在 Pydantic <2.10 时会验证失败
    mapper: RunnableParallel  # type: ignore[type-arg]

    def __init__(self, mapper: RunnableParallel[dict[str, Any]], **kwargs: Any) -> None:
        """创建一个 `RunnableAssign`。

        Args:
            mapper: 用于转换输入字典的 `RunnableParallel` 实例。
        """
        super().__init__(mapper=mapper, **kwargs)

    @classmethod
    @override
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
    def get_name(self, suffix: str | None = None, *, name: str | None = None) -> str:
        name = (
            name
            or self.name
            or f"RunnableAssign<{','.join(self.mapper.steps__.keys())}>"
        )
        return super().get_name(suffix, name=name)

    @override
    def get_input_schema(self, config: RunnableConfig | None = None) -> TypeBaseModel:
        map_input_schema = self.mapper.get_input_schema(config)
        if not issubclass(map_input_schema, RootModel):
                        # 即它是一个字典
            return map_input_schema

        return super().get_input_schema(config)

    @override
    def get_output_schema(self, config: RunnableConfig | None = None) -> TypeBaseModel:
                # 返回类型保持为 TypeBaseModel（而非缩小为
                # `type[BaseModel]`，如 `RunnableParallel.get_output_schema` 所做），因为
                # 回退分支返回 mapper 的输出模式或委托给
                # `super().get_output_schema()`，两者都可能是 Pydantic v1 模型。
        map_input_schema = self.mapper.get_input_schema(config)
        map_output_schema = self.mapper.get_output_schema(config)
        if not issubclass(map_input_schema, RootModel) and not issubclass(
            map_output_schema, RootModel
        ):
            fields = {}

            for name, field_info in get_fields(map_input_schema).items():
                fields[name] = _get_schema_field_definition(field_info)

            for name, field_info in get_fields(map_output_schema).items():
                fields[name] = _get_schema_field_definition(field_info)

            return create_model_v2("RunnableAssignOutput", field_definitions=fields)
        if not issubclass(map_output_schema, RootModel):
                        # 即仅映射输出是字典
                        # 即输入类型未知或推断错误
            return map_output_schema

        return super().get_output_schema(config)

    @property
    @override
    def config_specs(self) -> list[ConfigurableFieldSpec]:
        return self.mapper.config_specs

    @override
    def get_graph(self, config: RunnableConfig | None = None) -> Graph:
                # 从 mapper 获取图结构
        graph = self.mapper.get_graph(config)
                # 添加透传节点和边
        input_node = graph.first_node()
        output_node = graph.last_node()
        if input_node is not None and output_node is not None:
            passthrough_node = graph.add_node(_graph_passthrough)
            graph.add_edge(input_node, passthrough_node)
            graph.add_edge(passthrough_node, output_node)
        return graph

    def _invoke(
        self,
        value: dict[str, Any],
        run_manager: CallbackManagerForChainRun,
        config: RunnableConfig,
        **kwargs: Any,
    ) -> dict[str, Any]:
        if not isinstance(value, dict):
            msg = "The input to RunnablePassthrough.assign() must be a dict."  # type: ignore[unreachable]
            raise ValueError(msg)  # noqa: TRY004

        return {
            **value,
            **self.mapper.invoke(
                value,
                patch_config(config, callbacks=run_manager.get_child()),
                **kwargs,
            ),
        }

    @override
    def invoke(
        self,
        input: dict[str, Any],
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        return self._call_with_config(self._invoke, input, config, **kwargs)

    async def _ainvoke(
        self,
        value: dict[str, Any],
        run_manager: AsyncCallbackManagerForChainRun,
        config: RunnableConfig,
        **kwargs: Any,
    ) -> dict[str, Any]:
        if not isinstance(value, dict):
            msg = "The input to RunnablePassthrough.assign() must be a dict."  # type: ignore[unreachable]
            raise ValueError(msg)  # noqa: TRY004

        return {
            **value,
            **await self.mapper.ainvoke(
                value,
                patch_config(config, callbacks=run_manager.get_child()),
                **kwargs,
            ),
        }

    @override
    async def ainvoke(
        self,
        input: dict[str, Any],
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        return await self._acall_with_config(self._ainvoke, input, config, **kwargs)

    def _transform(
        self,
        values: Iterator[dict[str, Any]],
        run_manager: CallbackManagerForChainRun,
        config: RunnableConfig,
        **kwargs: Any,
    ) -> Iterator[dict[str, Any]]:
                # 收集 mapper 的键
        mapper_keys = set(self.mapper.steps__.keys())
                # 创建两个流，一个用于映射，一个用于透传
        for_passthrough, for_map = safetee(values, 2, lock=threading.Lock())

                # 创建映射输出流
        map_output = self.mapper.transform(
            for_map,
            patch_config(
                config,
                callbacks=run_manager.get_child(),
            ),
            **kwargs,
        )

                # 获取执行器以在后台启动映射输出流
        with get_executor_for_config(config) as executor:
                        # 启动映射输出流
            first_map_chunk_future = executor.submit(
                next,
                map_output,
                None,
            )
                        # 消费透传流
            for chunk in for_passthrough:
                if not isinstance(chunk, dict):
                    msg = "The input to RunnablePassthrough.assign() must be a dict."  # type: ignore[unreachable]
                    raise ValueError(msg)  # noqa: TRY004
                                # 从透传块中移除 mapper 键，以便被映射结果覆盖
                filtered = AddableDict(
                    {k: v for k, v in chunk.items() if k not in mapper_keys}
                )
                if filtered:
                    yield filtered
                        # 生成映射输出
            first_chunk = first_map_chunk_future.result()
            if first_chunk is not None:
                yield first_chunk
                for chunk in map_output:
                    yield chunk

    @override
    def transform(
        self,
        input: Iterator[dict[str, Any]],
        config: RunnableConfig | None = None,
        **kwargs: Any | None,
    ) -> Iterator[dict[str, Any]]:
        yield from self._transform_stream_with_config(
            input, self._transform, config, **kwargs
        )

    async def _atransform(
        self,
        values: AsyncIterator[dict[str, Any]],
        run_manager: AsyncCallbackManagerForChainRun,
        config: RunnableConfig,
        **kwargs: Any,
    ) -> AsyncIterator[dict[str, Any]]:
                # 收集 mapper 的键
        mapper_keys = set(self.mapper.steps__.keys())
                # 创建两个流，一个用于映射，一个用于透传
        for_passthrough, for_map = atee(values, 2, lock=asyncio.Lock())
                # 创建映射输出流
        map_output = self.mapper.atransform(
            for_map,
            patch_config(
                config,
                callbacks=run_manager.get_child(),
            ),
            **kwargs,
        )
                # 启动映射输出流
        first_map_chunk_task = asyncio.create_task(
            anext(map_output, None),
        )
                # 消费透传流
        async for chunk in for_passthrough:
            if not isinstance(chunk, dict):
                msg = "The input to RunnablePassthrough.assign() must be a dict."  # type: ignore[unreachable]
                raise ValueError(msg)  # noqa: TRY004

                        # 从透传块中移除 mapper 键，以便被映射输出覆盖
            filtered = AddableDict(
                {k: v for k, v in chunk.items() if k not in mapper_keys}
            )
            if filtered:
                yield filtered
                # 生成映射输出
        first_chunk = await first_map_chunk_task
        if first_chunk is not None:
            yield first_chunk
            async for chunk in map_output:
                yield chunk

    @override
    async def atransform(
        self,
        input: AsyncIterator[dict[str, Any]],
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[dict[str, Any]]:
        async for chunk in self._atransform_stream_with_config(
            input, self._atransform, config, **kwargs
        ):
            yield chunk

    @override
    def stream(
        self,
        input: dict[str, Any],
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> Iterator[dict[str, Any]]:
        return self.transform(iter([input]), config, **kwargs)

    @override
    async def astream(
        self,
        input: dict[str, Any],
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[dict[str, Any]]:
        async def input_aiter() -> AsyncIterator[dict[str, Any]]:
            yield input

        async for chunk in self.atransform(input_aiter(), config, **kwargs):
            yield chunk


class RunnablePick(RunnableSerializable[dict[str, Any], Any]):
    """`Runnable`，用于从 `dict[str, Any]` 输入中选取键。

    `RunnablePick` class represents a `Runnable` that selectively picks keys from a
    dictionary input. It allows you to specify one or more keys to extract
    from the input dictionary.

    !!! note "返回类型行为"
        返回类型取决于 `keys` 参数：

        - When `keys` is a `str`: Returns the single value associated with that key
        - When `keys` is a `list`: Returns a dictionary containing only the selected
            keys

    Example:
        ```python
        from langchain_core.runnables.passthrough import RunnablePick

        input_data = {
            "name": "John",
            "age": 30,
            "city": "New York",
            "country": "USA",
        }

        # Single key - returns the value directly
        runnable_single = RunnablePick(keys="name")
        result_single = runnable_single.invoke(input_data)
        print(result_single)  # Output: "John"

        # Multiple keys - returns a dictionary
        runnable_multiple = RunnablePick(keys=["name", "age"])
        result_multiple = runnable_multiple.invoke(input_data)
        print(result_multiple)  # Output: {'name': 'John', 'age': 30}
        ```
    """

    keys: str | list[str]

    def __init__(self, keys: str | list[str], **kwargs: Any) -> None:
        """创建一个 `RunnablePick`。

        Args:
            keys: 要从输入字典中选取的单个键或键列表。
        """
        super().__init__(keys=keys, **kwargs)

    @classmethod
    @override
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
    def get_name(self, suffix: str | None = None, *, name: str | None = None) -> str:
        name = (
            name
            or self.name
            or "RunnablePick"
            f"<{','.join([self.keys] if isinstance(self.keys, str) else self.keys)}>"
        )
        return super().get_name(suffix, name=name)

    def _pick(self, value: dict[str, Any]) -> Any:
        if not isinstance(value, dict):
            msg = "The input to RunnablePassthrough.assign() must be a dict."  # type: ignore[unreachable]
            raise ValueError(msg)  # noqa: TRY004

        if isinstance(self.keys, str):
            return value.get(self.keys)
        picked = {k: value.get(k) for k in self.keys if k in value}
        if picked:
            return AddableDict(picked)
        return None

    @override
    def invoke(
        self,
        input: dict[str, Any],
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> Any:
        return self._call_with_config(self._pick, input, config, **kwargs)

    async def _ainvoke(
        self,
        value: dict[str, Any],
    ) -> Any:
        return self._pick(value)

    @override
    async def ainvoke(
        self,
        input: dict[str, Any],
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> Any:
        return await self._acall_with_config(self._ainvoke, input, config, **kwargs)

    def _transform(
        self,
        chunks: Iterator[dict[str, Any]],
    ) -> Iterator[Any]:
        for chunk in chunks:
            picked = self._pick(chunk)
            if picked is not None:
                yield picked

    @override
    def transform(
        self,
        input: Iterator[dict[str, Any]],
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> Iterator[Any]:
        yield from self._transform_stream_with_config(
            input, self._transform, config, **kwargs
        )

    async def _atransform(
        self,
        chunks: AsyncIterator[dict[str, Any]],
    ) -> AsyncIterator[Any]:
        async for chunk in chunks:
            picked = self._pick(chunk)
            if picked is not None:
                yield picked

    @override
    async def atransform(
        self,
        input: AsyncIterator[dict[str, Any]],
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[Any]:
        async for chunk in self._atransform_stream_with_config(
            input, self._atransform, config, **kwargs
        ):
            yield chunk

    @override
    def stream(
        self,
        input: dict[str, Any],
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> Iterator[Any]:
        return self.transform(iter([input]), config, **kwargs)

    @override
    async def astream(
        self,
        input: dict[str, Any],
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[Any]:
        async def input_aiter() -> AsyncIterator[dict[str, Any]]:
            yield input

        async for chunk in self.atransform(input_aiter(), config, **kwargs):
            yield chunk

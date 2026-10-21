"""直接接收函数或协程的工具。"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import (
    TYPE_CHECKING,
    Any,
)

from typing_extensions import override

# Cannot move to TYPE_CHECKING as _run/_arun parameter annotations are needed at runtime
from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,  # noqa: TC001
    CallbackManagerForToolRun,  # noqa: TC001
)
from langchain_core.runnables import RunnableConfig, run_in_executor
from langchain_core.tools.base import (
    ArgsSchema,
    BaseTool,
    ToolException,
    _get_runnable_config_param,
)
from langchain_core.utils._signature import _signature_parameters

if TYPE_CHECKING:
    from langchain_core.messages import ToolCall


class Tool(BaseTool):
    """直接接收函数或协程的工具。"""

    description: str = ""

    func: Callable[..., str] | None
    """调用工具时要运行的函数。"""

    coroutine: Callable[..., Awaitable[str]] | None = None
    """函数的异步版本。"""

        # --- Runnable ---

    @override
    async def ainvoke(
        self,
        input: str | dict[str, Any] | ToolCall,
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> Any:
        if not self.coroutine:
                        # 如果工具未实现异步，则回退到默认实现
            return await run_in_executor(config, self.invoke, input, config, **kwargs)

        return await super().ainvoke(input, config, **kwargs)

        # --- Tool ---

    @property
    def args(self) -> dict[str, Any]:
        """工具的输入参数。

        Returns:
            工具的输入参数。
        """
        if self.args_schema is not None:
            return super().args
                # 为了向后兼容，如果函数签名不明确，
                # 假设它接受单个字符串输入。
        return {"tool_input": {"type": "string"}}

    def _to_args_and_kwargs(
        self, tool_input: str | dict[str, Any], tool_call_id: str | None
    ) -> tuple[tuple[str, ...], dict[str, Any]]:
        """将工具输入转换为 Pydantic 模型。

        Args:
            tool_input: 工具的输入。
            tool_call_id: 工具调用的 ID。

        Raises:
            ToolException: 如果工具输入无效。

        Returns:
            Pydantic 模型的 args 和 kwargs。
        """
        args, kwargs = super()._to_args_and_kwargs(tool_input, tool_call_id)
                # 为了向后兼容。工具必须以单个输入运行
        all_args = list(args) + list(kwargs.values())
        if len(all_args) != 1:
            msg = (
                f"""Too many arguments to single-input tool {self.name}.
                Consider using StructuredTool instead."""
                f" Args: {all_args}"
            )
            raise ToolException(msg)
        return tuple(all_args), {}

    def _run(
        self,
        *args: Any,
        config: RunnableConfig,
        run_manager: CallbackManagerForToolRun | None = None,
        **kwargs: Any,
    ) -> Any:
        """使用工具。

        Args:
            *args: 传递给工具的位置参数
            config: 运行的配置
            run_manager: 可选的回调管理器，用于运行
            **kwargs: 传递给工具的关键字参数

        Returns:
            工具执行的结果
        """
        if self.func:
            if run_manager and _signature_parameters(self.func).get("callbacks"):
                kwargs["callbacks"] = run_manager.get_child()
            if config_param := _get_runnable_config_param(self.func):
                kwargs[config_param] = config
            return self.func(*args, **kwargs)
        msg = "Tool does not support sync invocation."
        raise NotImplementedError(msg)

    async def _arun(
        self,
        *args: Any,
        config: RunnableConfig,
        run_manager: AsyncCallbackManagerForToolRun | None = None,
        **kwargs: Any,
    ) -> Any:
        """异步使用工具。

        Args:
            *args: 传递给工具的位置参数
            config: 运行的配置
            run_manager: 可选的回调管理器，用于运行
            **kwargs: 传递给工具的关键字参数

        Returns:
            工具执行的结果
        """
        if self.coroutine:
            if run_manager and _signature_parameters(self.coroutine).get("callbacks"):
                kwargs["callbacks"] = run_manager.get_child()
            if config_param := _get_runnable_config_param(self.coroutine):
                kwargs[config_param] = config
            return await self.coroutine(*args, **kwargs)

                # NOTE: this code is unreachable since _arun is only called if coroutine is not
                # None.
        return await super()._arun(
            *args, config=config, run_manager=run_manager, **kwargs
        )

        # TODO: this is for backwards compatibility, remove in future
    def __init__(
        self,
        name: str,
        func: Callable[..., Any] | None,
        description: str,
        **kwargs: Any,
    ) -> None:
        """初始化工具。"""
        super().__init__(name=name, func=func, description=description, **kwargs)

    @classmethod
    def from_function(
        cls,
        func: Callable[..., Any] | None,
        name: str,  # We keep these required to support backwards compatibility
        description: str,
        return_direct: bool = False,  # noqa: FBT001,FBT002
        args_schema: ArgsSchema | None = None,
        coroutine: Callable[..., Awaitable[Any]]
        | None = None,  # This is last for compatibility, but should be after func
        **kwargs: Any,
    ) -> Tool:
        """从函数初始化工具。

        Args:
            func: 用于创建工具的函数。
            name: 工具的名称。
            description: 工具的描述。
            return_direct: 是否直接返回输出。
            args_schema: 工具输入参数的模式。
            coroutine: 函数的异步版本。
            **kwargs: 额外传递给工具的参数。

        Returns:
            工具。

        Raises:
            ValueError: 如果未提供函数。
        """
        if func is None and coroutine is None:
            msg = "Function and/or coroutine must be provided"
            raise ValueError(msg)
        return cls(
            name=name,
            func=func,
            coroutine=coroutine,
            description=description,
            return_direct=return_direct,
            args_schema=args_schema,
            **kwargs,
        )

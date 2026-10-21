"""智能体的工具错误中间件。"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from langchain_core.messages import ToolMessage
from langgraph.errors import GraphBubbleUp

from langchain.agents.middleware.types import (
    AgentMiddleware,
    AgentState,
    ContextT,
    ResponseT,
    TracePolicy,
    omit_payload,
)

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from langchain_core.messages import ContentBlock
    from langgraph.types import Command

    from langchain.agents.middleware.types import ToolCallRequest
    from langchain.tools import BaseTool

    OnError = Callable[[Exception, ToolCallRequest], str | list[ContentBlock] | None]
    """Sync handler: return content to surface the error as a `ToolMessage`; return
    `None` (or nothing) to let the exception propagate."""

    AOnError = Callable[[Exception, ToolCallRequest], Awaitable[str | list[ContentBlock] | None]]
    """Async handler: return content to surface the error as a `ToolMessage`; return
    `None` (or nothing) to let the exception propagate."""


class ToolErrorMiddleware(AgentMiddleware[AgentState[ResponseT], ContextT, ResponseT]):
    """将选定的工具执行异常返回给模型作为错误 `ToolMessage`。

    `on_error` is called for each exception raised by tool execution. Return content
    (a `str` or a list of content blocks) to convert the exception into a
    `ToolMessage(status="error")`; return `None` — or simply don't return — to let the
    exception propagate (halting the run). Handling is therefore opt-in — exceptions you
    do not return content for propagate unchanged, so arbitrary internal exceptions are
    never serialized to the model or end user unless you choose to surface them.

    Langgraph 控制流信号（中断、父级命令）始终会传播，且永远不会到达 `on_error`。

    Prefer returning content that names the exception type over the raw exception message,
    which may carry sensitive or internal detail.

    Provide at least one of `on_error` or `aon_error`. `aon_error` handles errors on the
    async execution path (falling back to `on_error` when omitted); the sync path only
    ever calls `on_error`. For async-only usage, pass `aon_error` alone — running such a
    middleware on the sync path raises, since the async handler cannot be awaited there.

    This middleware does not retry. For retries, compose with `ToolRetryMiddleware`
    placed *inner* and configured with `on_failure="error"` so exceptions reach this
    middleware.

    This middleware only sees exceptions raised by tool *execution*. Argument-binding
    and validation errors are handled upstream by `ToolNode` (converted to an error
    `ToolMessage` before the tool runs), so they do not reach `on_error`.

    Example:
        ```python
        from langchain.agents import create_agent
        from langchain.agents.middleware import ToolErrorMiddleware


        def on_error(exc: Exception, request: ToolCallRequest) -> str | None:
            if isinstance(exc, ValueError):
                return f"`{request.tool_call['name']}` failed; fix the input and retry."
            return None  # propagate everything else


        agent = create_agent(model, tools=[...], middleware=[ToolErrorMiddleware(on_error)])
        ```
    """

    trace_policy = TracePolicy(process_inputs=omit_payload)

    def __init__(
        self,
        on_error: OnError | None = None,
        *,
        aon_error: AOnError | None = None,
        tools: list[BaseTool | str] | None = None,
    ) -> None:
        """初始化 `ToolErrorMiddleware`。

        Args:
            on_error: Handler called for each exception raised by tool execution. Return
                content (`str` or list of content blocks) to convert the exception into an
                error `ToolMessage`. Return `None` — or simply don't return — to let the
                exception propagate. Falling through without a return therefore re-raises,
                so handle only the exceptions you mean to. Receives the exception and the
                tool call request (tool name, args, call id). Used on the sync path and,
                unless `aon_error` is given, on the async path.
            aon_error: Optional async handler, used on the async execution path. Falls back
                to `on_error` when not provided.
            tools: Optional list of tools or tool names to apply handling to. If `None`,
                applies to all tools.

        Raises:
            ValueError: 如果未提供 `on_error` 和 `aon_error` 中的任何一个，则抛出 `ValueError`。
        """
        super().__init__()

        if on_error is None and aon_error is None:
            msg = "ToolErrorMiddleware requires `on_error` and/or `aon_error`."
            raise ValueError(msg)

        self.on_error = on_error
        self.aon_error = aon_error

                # 从 BaseTool 实例或字符串中提取工具名称
        self._tool_filter: list[str] | None
        if tools is not None:
            self._tool_filter = [tool.name if not isinstance(tool, str) else tool for tool in tools]
        else:
            self._tool_filter = None

        self.tools = []  # 此中间件未注册额外的工具

    def _should_handle_tool(self, tool_name: str) -> bool:
        """检查错误处理是否适用于此工具。"""
        if self._tool_filter is None:
            return True
        return tool_name in self._tool_filter

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command[Any]],
    ) -> ToolMessage | Command[Any]:
        """拦截工具执行并将已处理的异常转换为错误消息。

        Args:
            request: 包含调用字典、`BaseTool`、状态和运行时的工具调用请求。
            handler: 要执行的工具调用函数。

        Returns:
            `ToolMessage` 或 `Command`（最终结果）。
        """
        tool_name = request.tool.name if request.tool else request.tool_call["name"]

        if not self._should_handle_tool(tool_name):
            return handler(request)

        try:
            return handler(request)
        except GraphBubbleUp:
                        # 控制流信号（中断、父命令）必须传播
            raise
        except Exception as exc:
            if self.on_error is None:
                                # 仅异步配置（aon_error）无法在同步路径上等待
                msg = (
                    "ToolErrorMiddleware has no sync `on_error`; run async "
                    "(ainvoke/astream) or provide `on_error`."
                )
                raise RuntimeError(msg) from exc
            content = self.on_error(exc, request)
            if content is None:
                raise
            return ToolMessage(
                content=cast("str | list[str | dict[Any, Any]]", content),
                tool_call_id=request.tool_call["id"],
                name=tool_name,
                status="error",
            )

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command[Any]]],
    ) -> ToolMessage | Command[Any]:
        """`wrap_tool_call` 的异步版本。

        Uses `aon_error` if provided, otherwise the sync `on_error`. The sync path never
        awaits.
        """
        tool_name = request.tool.name if request.tool else request.tool_call["name"]

        if not self._should_handle_tool(tool_name):
            return await handler(request)

        try:
            return await handler(request)
        except GraphBubbleUp:
                        # 控制流信号（中断、父命令）必须传播
            raise
        except Exception as exc:
            if self.aon_error is not None:
                content = await self.aon_error(exc, request)
            elif self.on_error is not None:
                content = self.on_error(exc, request)
            else:  # pragma: no cover - __init__ guarantees at least one handler
                raise
            if content is None:
                raise
            return ToolMessage(
                content=cast("str | list[str | dict[Any, Any]]", content),
                tool_call_id=request.tool_call["id"],
                name=tool_name,
                status="error",
            )

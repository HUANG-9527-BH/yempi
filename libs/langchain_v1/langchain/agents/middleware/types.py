"""中间件和智能体的类型。"""

from __future__ import annotations

import warnings
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field, replace
from inspect import iscoroutinefunction
from typing import (
    TYPE_CHECKING,
    Annotated,
    Any,
    Generic,
    Literal,
    cast,
    overload,
)

# Needed as top level import for Pydantic schema generation on AgentState
from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    BaseMessage,
    SystemMessage,
    ToolMessage,
)
from langgraph.channels.ephemeral_value import EphemeralValue
from langgraph.graph.message import add_messages
from langgraph.prebuilt.tool_node import ToolCallRequest, ToolCallWrapper
from langgraph.runtime import Runtime
from langgraph.types import Command, TracePolicy, omit_payload
from langgraph.typing import ContextT
from typing_extensions import NotRequired, Required, TypedDict, TypeVar, Unpack

if TYPE_CHECKING:
    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.tools import BaseTool
    from langgraph.stream._mux import TransformerFactory

    from langchain.agents.structured_output import ResponseFormat

__all__ = [
    "AgentMiddleware",
    "AgentState",
    "ContextT",
    "ExtendedModelResponse",
    "InputAgentState",
    "ModelCallResult",
    "ModelRequest",
    "ModelResponse",
    "OmitFromSchema",
    "OutputAgentState",
    "ResponseT",
    "StateT_co",
    "ToolCallRequest",
    "ToolCallWrapper",
    "TracePolicy",
    "after_agent",
    "after_model",
    "before_agent",
    "before_model",
    "dynamic_prompt",
    "hook_config",
    "omit_payload",
    "wrap_tool_call",
]

JumpTo = Literal["tools", "model", "end"]
"""Destination to jump to when a middleware node returns."""

ResponseT = TypeVar("ResponseT", default=Any)


class _ModelRequestOverrides(TypedDict, total=False):
    """`ModelRequest.override()` 方法可能的覆盖项。"""

    model: BaseChatModel
    system_message: SystemMessage | None
    messages: list[AnyMessage]
    tool_choice: Any | None
    tools: list[BaseTool | dict[str, Any]]
    response_format: ResponseFormat[Any] | None
    model_settings: dict[str, Any]
    state: AgentState[Any]


@dataclass(init=False)
class ModelRequest(Generic[ContextT]):
    """智能体使用的模型请求信息。

    类型参数:
        ContextT: 运行时上下文的类型。

            如果未指定，默认为 `None`。
    """

    model: BaseChatModel
    messages: list[AnyMessage]  # excluding system message
    system_message: SystemMessage | None
    tool_choice: Any | None
    tools: list[BaseTool | dict[str, Any]]
    response_format: ResponseFormat[Any] | None
    state: AgentState[Any]
    runtime: Runtime[ContextT]
    model_settings: dict[str, Any] = field(default_factory=dict)

    def __init__(
        self,
        *,
        model: BaseChatModel,
        messages: list[AnyMessage],
        system_message: SystemMessage | None = None,
        system_prompt: str | None = None,
        tool_choice: Any | None = None,
        tools: list[BaseTool | dict[str, Any]] | None = None,
        response_format: ResponseFormat[Any] | None = None,
        state: AgentState[Any] | None = None,
        runtime: Runtime[ContextT] | None = None,
        model_settings: dict[str, Any] | None = None,
    ) -> None:
        """如果存在系统消息则返回其内容，否则为 `None`。

        Args:
            设置属性并发出弃用警告。

        Raises:
            ValueError: 直接在 `ModelRequest` 上进行属性赋值已弃用。请改用 `override()` 方法来创建具有修改后属性的新请求。
        """
                # Handle system_prompt/system_message conversion and validation
        if system_prompt is not None and system_message is not None:
            msg = "Cannot specify both system_prompt and system_message"
            raise ValueError(msg)

        if system_prompt is not None:
            system_message = SystemMessage(content=system_prompt)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=DeprecationWarning)
            self.model = model
            self.messages = messages
            self.system_message = system_message
            self.tool_choice = tool_choice
            self.tools = tools if tools is not None else []
            self.response_format = response_format
            self.state = state if state is not None else {"messages": []}
            self.runtime = runtime  # type: ignore[assignment]
            self.model_settings = model_settings if model_settings is not None else {}

    @property
    def system_prompt(self) -> str | None:
        """Get system prompt text from system_message.

        Returns:
            如果存在系统消息，则为系统消息的内容，否则为 `None`。
        """
        if self.system_message is None:
            return None
        return self.system_message.text

    def __setattr__(self, name: str, value: Any) -> None:
        """设置属性并发出弃用警告。

        Direct attribute assignment on `ModelRequest` is deprecated. Use the
        `override()` method instead to create a new request with modified attributes.

        Args:
            name: 属性名称。
            value: 属性值。
        """
                # Special handling for system_prompt - convert to system_message
        if name == "system_prompt":
            warnings.warn(
                "Direct attribute assignment to ModelRequest.system_prompt is deprecated. "
                "Use request.override(system_message=SystemMessage(...)) instead to create "
                "a new request with the modified system message.",
                DeprecationWarning,
                stacklevel=2,
            )
            if value is None:
                object.__setattr__(self, "system_message", None)
            else:
                object.__setattr__(self, "system_message", SystemMessage(content=value))
            return

        warnings.warn(
            f"Direct attribute assignment to ModelRequest.{name} is deprecated. "
            f"Use request.override({name}=...) instead to create a new request "
            f"with the modified attribute.",
            DeprecationWarning,
            stacklevel=2,
        )
        object.__setattr__(self, name, value)

    def override(self, **overrides: Unpack[_ModelRequestOverrides]) -> ModelRequest[ContextT]:
        """用给定的覆盖项替换请求。

        返回一个具有指定属性替换的新 `ModelRequest` 实例。

        此操作遵循不可变模式，原始请求保持不变。

        Args:
            **overrides: 用于覆盖属性的关键字参数。

                支持的键：

                - `model`: `BaseChatModel` 实例
                - `system_prompt`: 已弃用，请改用 `system_message`
                - `system_message`: `SystemMessage` 实例
                - `messages`: 消息列表
                - `tool_choice`: 工具选择配置
                - `tools`: 可用工具列表
                - `response_format`: 响应格式规范
                - `model_settings`: 额外的模型设置
                - `state`: 智能体状态字典

        Returns:
            具有指定覆盖项应用的新 `ModelRequest` 实例。

        Examples:
            !!! example "创建使用不同模型的新请求"

                ```python
                new_request = request.override(model=different_model)
                ```

            !!! example "覆盖系统消息（推荐）"

                ```python
                from langchain_core.messages import SystemMessage

                new_request = request.override(
                    system_message=SystemMessage(content="New instructions")
                )
                ```

            !!! example "覆盖多个属性"

                ```python
                new_request = request.override(
                    model=ChatOpenAI(model="gpt-5.5"),
                    system_message=SystemMessage(content="New instructions"),
                )
                ```

        Raises:
            ValueError: 如果同时提供了 `system_prompt` 和 `system_message`。
        """
                # Handle system_prompt/system_message conversion
        if "system_prompt" in overrides and "system_message" in overrides:
            msg = "Cannot specify both system_prompt and system_message"
            raise ValueError(msg)

        if "system_prompt" in overrides:
            system_prompt = cast("str | None", overrides.pop("system_prompt"))  # type: ignore[typeddict-item]
            if system_prompt is None:
                overrides["system_message"] = None
            else:
                overrides["system_message"] = SystemMessage(content=system_prompt)

        return replace(self, **overrides)


@dataclass
class ModelResponse(Generic[ResponseT]):
    """模型执行的响应，包含消息和可选的结构化输出。

    结果通常包含一个 `AIMessage`，但如果模型使用工具生成了结构化输出，也可能包含额外的
    `ToolMessage`。

    类型参数:
        ResponseT: 结构化响应的类型。如果未指定，默认为 `Any`。
    """

    result: list[BaseMessage]
    """模型执行返回的消息列表。"""

    structured_response: ResponseT | None = None
    """如果指定了 `response_format`，则为解析后的结构化输出，否则为 `None`。"""


@dataclass
class ExtendedModelResponse(Generic[ResponseT]):
    """带有可选 'Command' 的模型响应，来自 'wrap_model_call' 中间件。

    Use this to return a 'Command' alongside the model response from a
    'wrap_model_call' handler. The command is applied as an additional state
    update after the model node completes, using the graph's reducers (e.g.
    'add_messages' for the 'messages' key).

    由于每个 'Command' 都通过 reducer 应用，因此命令中的消息会 **与**
    模型响应消息一并添加，而不是替换它们。对于非 reducer
    状态字段，后续命令会覆盖先前的命令（最外层中间件胜过内层）。

    类型参数:
        ResponseT: 结构化响应的类型。如果未指定，默认为 'Any'。
    """

    model_response: ModelResponse[ResponseT]
    """底层模型响应。"""

    command: Command[Any] | None = None
    """可选的命令，用于作为额外的状态更新。"""


ModelCallResult = ModelResponse[ResponseT] | AIMessage | ExtendedModelResponse[ResponseT]
"""Return type for model call handlers.

Middleware can return either:

- `ModelResponse`: Full response with messages and optional structured output
- `AIMessage`: Simplified return for simple use cases
- `ExtendedModelResponse`: Response with an optional `Command` for additional state updates
    `goto`, `resume`, and `graph` are not yet supported on these commands.
    A `NotImplementedError` will be raised if you try to use them.
"""


@dataclass
class OmitFromSchema:
    """用于标记状态属性从输入或输出模式中省略的注解。"""

    input: bool = True
    """是否从输入模式中省略该属性。"""

    output: bool = True
    """是否从输出模式中省略该属性。"""


OmitFromInput = OmitFromSchema(input=True, output=False)
"""Annotation used to mark state attributes as omitted from input schema."""

OmitFromOutput = OmitFromSchema(input=False, output=True)
"""Annotation used to mark state attributes as omitted from output schema."""

PrivateStateAttr = OmitFromSchema(input=True, output=True)
"""Annotation used to mark state attributes as purely internal for a given middleware."""


class AgentState(TypedDict, Generic[ResponseT]):
    """智能体的状态模式。"""

    messages: Required[Annotated[list[AnyMessage], add_messages]]
    jump_to: NotRequired[Annotated[JumpTo | None, EphemeralValue, PrivateStateAttr]]
    structured_response: NotRequired[Annotated[ResponseT, OmitFromInput]]


class InputAgentState(TypedDict):
    """智能体的输入状态模式。"""

    messages: Required[Annotated[list[AnyMessage | dict[str, Any]], add_messages]]


class OutputAgentState(TypedDict, Generic[ResponseT]):
    """智能体的输出状态模式。"""

    messages: Required[Annotated[list[AnyMessage], add_messages]]
    structured_response: NotRequired[ResponseT]


# Deprecated aliases kept for backwards compatibility with external consumers that
# imported the previously private names. Remove in a future release.
_InputAgentState = InputAgentState
_OutputAgentState = OutputAgentState


StateT = TypeVar("StateT", bound=AgentState[Any], default=AgentState[Any])
StateT_co = TypeVar("StateT_co", bound=AgentState[Any], default=AgentState[Any], covariant=True)
StateT_contra = TypeVar("StateT_contra", bound=AgentState[Any], contravariant=True)


class _DefaultAgentState(AgentState[Any]):
    """AgentMiddleware 默认状态。"""


class AgentMiddleware(Generic[StateT, ContextT, ResponseT]):
    """智能体的基础中间件类。

    继承此类并实现任意定义的方法，以自定义主智能体循环中各步骤之间的行为。

    类型参数:
        StateT: 智能体状态的类型。默认为 `AgentState[Any]`。
        ContextT: 运行时上下文的类型。默认为 `None`。
        ResponseT: 结构化响应的类型。默认为 `Any`。
    """

    state_schema: type[StateT] = cast("type[StateT]", _DefaultAgentState)
    """传递给中间件节点的状态模式。"""

    tools: Sequence[BaseTool]
    """中间件注册的附加工具。"""

    trace_policy: TracePolicy | None = None
    """Optional trace policy for this middleware's hook spans (`wrap_model_call`/
    `wrap_tool_call` and the `before_*`/`after_*` node hooks).

    By default (`None`), hook spans are traced normally. Set a `TracePolicy` to shape
    what they record -- e.g. `TracePolicy(process_inputs=omit_payload)` to drop the
    conversation `messages`/`state` payload while keeping the span and its timing.
    Messages are still captured on the inner model-call span.
    """

    transformers: Sequence[TransformerFactory] = ()
    """中间件注册的流式转换器工厂。

    Each entry is a scope-aware factory invoked as `factory(scope)` so every
    invocation receives a fresh instance. Factories are merged with the
    `transformers` argument of [`create_agent`][langchain.agents.create_agent]
    at graph compile time, after the `ToolCallTransformer` and before any
    user-supplied entries.
    """

    @property
    def name(self) -> str:
        """中间件实例的名称。

        默认为类名，但可覆盖以自定义命名。
        """
        return self.__class__.__name__

    def before_agent(self, state: StateT, runtime: Runtime[ContextT]) -> dict[str, Any] | None:
        """在智能体执行开始前运行的逻辑。

        Args:
            state: 当前智能体状态。
            runtime: 运行时上下文。

        Returns:
            在智能体执行前要应用的智能体状态更新。
        """

    async def abefore_agent(
        self, state: StateT, runtime: Runtime[ContextT]
    ) -> dict[str, Any] | None:
        """在智能体执行开始前运行的异步逻辑。

        Args:
            state: 当前智能体状态。
            runtime: 运行时上下文。

        Returns:
            在智能体执行前要应用的智能体状态更新。
        """

    def before_model(self, state: StateT, runtime: Runtime[ContextT]) -> dict[str, Any] | None:
        """在调用模型前运行的逻辑。

        Args:
            state: 当前智能体状态。
            runtime: 运行时上下文。

        Returns:
            在模型调用前要应用的智能体状态更新。
        """

    async def abefore_model(
        self, state: StateT, runtime: Runtime[ContextT]
    ) -> dict[str, Any] | None:
        """在调用模型前运行的异步逻辑。

        Args:
            state: 智能体状态。
            runtime: 运行时上下文。

        Returns:
            在模型调用前要应用的智能体状态更新。
        """

    def after_model(self, state: StateT, runtime: Runtime[ContextT]) -> dict[str, Any] | None:
        """在模型调用后运行的逻辑。

        Args:
            state: 当前智能体状态。
            runtime: 运行时上下文。

        Returns:
            在模型调用后要应用的智能体状态更新。
        """

    async def aafter_model(
        self, state: StateT, runtime: Runtime[ContextT]
    ) -> dict[str, Any] | None:
        """在模型调用后运行的异步逻辑。

        Args:
            state: 当前智能体状态。
            runtime: 运行时上下文。

        Returns:
            在模型调用后要应用的智能体状态更新。
        """

    def wrap_model_call(
        self,
        request: ModelRequest[ContextT],
        handler: Callable[[ModelRequest[ContextT]], ModelResponse[ResponseT]],
    ) -> ModelResponse[ResponseT] | AIMessage | ExtendedModelResponse[ResponseT]:
        """通过处理器回调拦截并控制模型执行。

        异步版本为 `awrap_model_call`

        The handler callback executes the model request and returns a `ModelResponse`.
        Middleware can call the handler multiple times for retry logic, skip calling
        it to short-circuit, or modify the request/response. Multiple middleware
        compose with first in list as outermost layer.

        Args:
            request: 要执行的模型请求（包含状态和运行时）。 handler: 执行模型请求并返回
            `ModelResponse` 的回调。

                调用此方法以执行模型。

                可多次调用以实现重试逻辑。

                可跳过调用以进行短路处理。

        Returns:
            模型调用的结果。

        Examples:
            !!! example "错误时重试"

                ```python
                def wrap_model_call(self, request, handler):
                    for attempt in range(3):
                        try:
                            return handler(request)
                        except Exception:
                            if attempt == 2:
                                raise
                ```

            !!! example "重写响应"

                ```python
                def wrap_model_call(self, request, handler):
                    response = handler(request)
                    ai_msg = response.result[0]
                    return ModelResponse(
                        result=[AIMessage(content=f"[{ai_msg.content}]")],
                        structured_response=response.structured_response,
                    )
                ```

            !!! example "错误转为回退"

                ```python
                def wrap_model_call(self, request, handler):
                    try:
                        return handler(request)
                    except Exception:
                        return ModelResponse(result=[AIMessage(content="Service unavailable")])
                ```

            !!! example "缓存/短路处理"

                ```python
                def wrap_model_call(self, request, handler):
                    if cached := get_cache(request):
                        return cached  # Short-circuit with cached result
                    response = handler(request)
                    save_cache(request, response)
                    return response
                ```

            !!! example "简单 `AIMessage` 返回（自动转换）"

                ```python
                def wrap_model_call(self, request, handler):
                    response = handler(request)
                    # Can return AIMessage directly for simple cases
                    return AIMessage(content="Simplified response")
                ```
        """
        msg = (
            "Synchronous implementation of wrap_model_call is not available. "
            "You are likely encountering this error because you defined only the async version "
            "(awrap_model_call) and invoked your agent in a synchronous context "
            "(e.g., using `stream()` or `invoke()`). "
            "To resolve this, either: "
            "(1) subclass AgentMiddleware and implement the synchronous wrap_model_call method, "
            "(2) use the @wrap_model_call decorator on a standalone sync function, or "
            "(3) invoke your agent asynchronously using `astream()` or `ainvoke()`."
        )
        raise NotImplementedError(msg)

    async def awrap_model_call(
        self,
        request: ModelRequest[ContextT],
        handler: Callable[[ModelRequest[ContextT]], Awaitable[ModelResponse[ResponseT]]],
    ) -> ModelResponse[ResponseT] | AIMessage | ExtendedModelResponse[ResponseT]:
        """通过回调处理器拦截并控制异步模型执行。

        处理器回调执行模型请求并返回 `ModelResponse`。

        Middleware can call the handler multiple times for retry logic, skip calling
        it to short-circuit, or modify the request/response. Multiple middleware
        compose with first in list as outermost layer.

        Args:
            request: 要执行的模型请求（包含状态和运行时）。 handler:
            异步回调，执行模型请求并返回 `ModelResponse`。

                调用此方法以执行模型。

                可多次调用以实现重试逻辑。

                可跳过调用以进行短路处理。

        Returns:
            模型调用的结果。

        Examples:
            !!! example "错误时重试"

                ```python
                async def awrap_model_call(self, request, handler):
                    for attempt in range(3):
                        try:
                            return await handler(request)
                        except Exception:
                            if attempt == 2:
                                raise
                ```
        """
        msg = (
            "Asynchronous implementation of awrap_model_call is not available. "
            "You are likely encountering this error because you defined only the sync version "
            "(wrap_model_call) and invoked your agent in an asynchronous context "
            "(e.g., using `astream()` or `ainvoke()`). "
            "To resolve this, either: "
            "(1) subclass AgentMiddleware and implement the asynchronous awrap_model_call method, "
            "(2) use the @wrap_model_call decorator on a standalone async function, or "
            "(3) invoke your agent synchronously using `stream()` or `invoke()`."
        )
        raise NotImplementedError(msg)

    def after_agent(self, state: StateT, runtime: Runtime[ContextT]) -> dict[str, Any] | None:
        """代理执行完成后要运行的逻辑。

        Args:
            state: 当前代理状态。
            runtime: 运行时上下文。

        Returns:
            代理执行后要应用的代理状态更新。
        """

    async def aafter_agent(
        self, state: StateT, runtime: Runtime[ContextT]
    ) -> dict[str, Any] | None:
        """异步逻辑，代理执行完成后运行。

        Args:
            state: 当前代理状态。
            runtime: 运行时上下文。

        Returns:
            代理执行后要应用的代理状态更新。
        """

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command[Any]],
    ) -> ToolMessage | Command[Any]:
        """拦截工具执行，用于重试、监控或修改。

        异步版本为 `awrap_tool_call`

        多个中间件会自动组合（先定义的为最外层）。

        异常会传播，除非在 `ToolNode` 上配置了 `handle_tool_errors`。

        Args:
            request: 包含调用 `dict`、`BaseTool`、状态和运行时的工具调用请求。

                通过 `request.state` 访问状态，通过 `request.runtime` 访问运行时。
            handler: 执行工具的 `Callable`（可多次调用）。

        Returns:
            `ToolMessage` 或 `Command`（最终结果）。

        处理器 `Callable` 可多次调用以实现重试逻辑。

        每次对处理器的调用都是独立且无状态的。

        Examples:
            !!! example "执行前修改请求"

                ```python
                def wrap_tool_call(self, request, handler):
                    modified_call = {
                        **request.tool_call,
                        "args": {
                            **request.tool_call["args"],
                            "value": request.tool_call["args"]["value"] * 2,
                        },
                    }
                    request = request.override(tool_call=modified_call)
                    return handler(request)
                ```

            !!! example "错误时重试（多次调用处理器）"

                ```python
                def wrap_tool_call(self, request, handler):
                    for attempt in range(3):
                        try:
                            result = handler(request)
                            if is_valid(result):
                                return result
                        except Exception:
                            if attempt == 2:
                                raise
                    return result
                ```

            !!! example "基于响应的条件性重试"

                ```python
                def wrap_tool_call(self, request, handler):
                    for attempt in range(3):
                        result = handler(request)
                        if isinstance(result, ToolMessage) and result.status != "error":
                            return result
                        if attempt < 2:
                            continue
                        return result
                ```
        """
        msg = (
            "Synchronous implementation of wrap_tool_call is not available. "
            "You are likely encountering this error because you defined only the async version "
            "(awrap_tool_call) and invoked your agent in a synchronous context "
            "(e.g., using `stream()` or `invoke()`). "
            "To resolve this, either: "
            "(1) subclass AgentMiddleware and implement the synchronous wrap_tool_call method, "
            "(2) use the @wrap_tool_call decorator on a standalone sync function, or "
            "(3) invoke your agent asynchronously using `astream()` or `ainvoke()`."
        )
        raise NotImplementedError(msg)

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command[Any]]],
    ) -> ToolMessage | Command[Any]:
        """通过回调处理器拦截并控制异步工具执行。

        The handler callback executes the tool call and returns a `ToolMessage` or
        `Command`. Middleware can call the handler multiple times for retry logic, skip
        calling it to short-circuit, or modify the request/response. Multiple middleware
        compose with first in list as outermost layer.

        Args:
            request: 包含调用 `dict`、`BaseTool`、状态和运行时的工具调用请求。

                通过 `request.state` 访问状态，通过 `request.runtime` 访问运行时。
            handler: 异步可调用对象，用于执行工具并返回 `ToolMessage` 或
                `Command`。

                调用此函数以执行工具。

                可多次调用以实现重试逻辑。

                可跳过调用以短路执行。

        Returns:
            `ToolMessage` 或 `Command`（最终结果）。

        处理程序 `Callable` 可被多次调用以实现重试逻辑。

        每次对处理程序的调用都是独立且无状态的。

        Examples:
            !!! example "Async retry on error"

                ```python
                async def awrap_tool_call(self, request, handler):
                    for attempt in range(3):
                        try:
                            result = await handler(request)
                            if is_valid(result):
                                return result
                        except Exception:
                            if attempt == 2:
                                raise
                    return result
                ```

                ```python
                async def awrap_tool_call(self, request, handler):
                    if cached := await get_cache_async(request):
                        return ToolMessage(content=cached, tool_call_id=request.tool_call["id"])
                    result = await handler(request)
                    await save_cache_async(request, result)
                    return result
                ```
        """
        msg = (
            "Asynchronous implementation of awrap_tool_call is not available. "
            "You are likely encountering this error because you defined only the sync version "
            "(wrap_tool_call) and invoked your agent in an asynchronous context "
            "(e.g., using `astream()` or `ainvoke()`). "
            "To resolve this, either: "
            "(1) subclass AgentMiddleware and implement the asynchronous awrap_tool_call method, "
            "(2) use the @wrap_tool_call decorator on a standalone async function, or "
            "(3) invoke your agent synchronously using `stream()` or `invoke()`."
        )
        raise NotImplementedError(msg)


_SyncCallableWithStateAndRuntime = Callable[
    [StateT_contra, Runtime[ContextT]], dict[str, Any] | Command[Any] | None
]
_AsyncCallableWithStateAndRuntime = Callable[
    [StateT_contra, Runtime[ContextT]], Awaitable[dict[str, Any] | Command[Any] | None]
]
_CallableWithStateAndRuntime = (
    _SyncCallableWithStateAndRuntime[StateT_contra, ContextT]
    | _AsyncCallableWithStateAndRuntime[StateT_contra, ContextT]
)

_SyncCallableReturningSystemMessage = Callable[[ModelRequest[ContextT]], str | SystemMessage]
_AsyncCallableReturningSystemMessage = Callable[
    [ModelRequest[ContextT]], Awaitable[str | SystemMessage]
]
_CallableReturningSystemMessage = (
    _SyncCallableReturningSystemMessage[ContextT] | _AsyncCallableReturningSystemMessage[ContextT]
)

# Sync/async signatures for `wrap_model_call` interception; see `@wrap_model_call`.
_SyncCallableReturningModelResponse = Callable[
    [ModelRequest[ContextT], Callable[[ModelRequest[ContextT]], ModelResponse[ResponseT]]],
    ModelCallResult,
]
_AsyncCallableReturningModelResponse = Callable[
    [
        ModelRequest[ContextT],
        Callable[[ModelRequest[ContextT]], Awaitable[ModelResponse[ResponseT]]],
    ],
    Awaitable[ModelCallResult],
]
_CallableReturningModelResponse = (
    _SyncCallableReturningModelResponse[ContextT, ResponseT]
    | _AsyncCallableReturningModelResponse[ContextT, ResponseT]
)

# Sync/async signatures for `wrap_tool_call` interception; see `@wrap_tool_call`.
_SyncCallableReturningToolResponse = Callable[
    [ToolCallRequest, Callable[[ToolCallRequest], ToolMessage | Command[Any]]],
    ToolMessage | Command[Any],
]
_AsyncCallableReturningToolResponse = Callable[
    [ToolCallRequest, Callable[[ToolCallRequest], Awaitable[ToolMessage | Command[Any]]]],
    Awaitable[ToolMessage | Command[Any]],
]
_CallableReturningToolResponse = (
    _SyncCallableReturningToolResponse | _AsyncCallableReturningToolResponse
)


CallableT = TypeVar("CallableT", bound=Callable[..., Any])


def hook_config(
    *,
    can_jump_to: list[JumpTo] | None = None,
) -> Callable[[CallableT], CallableT]:
    """用于在中间件方法中配置钩子行为的装饰器。

    将此装饰器应用于中间件类中的 `before_model` 或 `after_model`
    方法，以配置其行为。目前支持指定它们可以跳转的目的地，这建立了智能体图中的条件边。

    Args:
        can_jump_to: 可选的合法跳转目的地列表。

            可以是：

            - `'tools'`: 跳转到工具节点
            - `'model'`: 返回模型节点
            - `'end'`: 跳转到图的末尾

    Returns:
        标记方法并添加配置元数据的装饰器函数。

    Examples:
        !!! example "在类方法上使用装饰器"

            ```python
            class MyMiddleware(AgentMiddleware):
                @hook_config(can_jump_to=["end", "model"])
                def before_model(self, state: AgentState) -> dict[str, Any] | None:
                    if some_condition(state):
                        return {"jump_to": "end"}
                    return None
            ```

        替代方式：在 `before_model`/`after_model` 装饰器中使用 `can_jump_to` 参数：

        ```python
        @before_model(can_jump_to=["end"])
        def conditional_middleware(state: AgentState) -> dict[str, Any] | None:
            if should_exit(state):
                return {"jump_to": "end"}
            return None
        ```
    """

    def decorator(func: CallableT) -> CallableT:
        if can_jump_to is not None:
            func.__can_jump_to__ = can_jump_to  # type: ignore[attr-defined]
        return func

    return decorator


@overload
def before_model(
    func: _CallableWithStateAndRuntime[StateT, ContextT],
) -> AgentMiddleware[StateT, ContextT]: ...


@overload
def before_model(
    func: None = None,
    *,
    state_schema: type[StateT] | None = None,
    tools: list[BaseTool] | None = None,
    can_jump_to: list[JumpTo] | None = None,
    name: str | None = None,
) -> Callable[
    [_CallableWithStateAndRuntime[StateT, ContextT]], AgentMiddleware[StateT, ContextT]
]: ...


def before_model(
    func: _CallableWithStateAndRuntime[StateT, ContextT] | None = None,
    *,
    state_schema: type[StateT] | None = None,
    tools: list[BaseTool] | None = None,
    can_jump_to: list[JumpTo] | None = None,
    name: str | None = None,
) -> (
    Callable[[_CallableWithStateAndRuntime[StateT, ContextT]], AgentMiddleware[StateT, ContextT]]
    | AgentMiddleware[StateT, ContextT]
):
    """用于动态创建带有 `before_model` 钩子的中间件的装饰器。

    Args:
        func: 要装饰的函数。

            必须接受：`state: StateT, runtime: Runtime[ContextT]` - 状态和运行时上下文
            state_schema: 可选的自定义状态模式类型。

            If not provided, uses the default `AgentState` schema.
        tools: Optional list of additional tools to register with this middleware.
        can_jump_to: Optional list of valid jump destinations for conditional edges.

            Valid values are: `'tools'`, `'model'`, `'end'`
        name: Optional name for the generated middleware class.

            返回值为： - `AgentMiddleware` 实例（如果直接提供了 func） -
            或者是一个可应用于函数的装饰器函数，该函数会包装目标函数。

    Returns:
        被装饰的函数应返回：

    - `dict[str, Any]` - 要合并到智能体状态中的状态更新 - `Command` -
    用于控制流的命令（例如，跳转到不同节点） - `None` - 无状态更新或流控

    !!! example "基本用法"

    Examples:
        !!! example "带条件跳转的用法"

            ```python
            @before_model
            def log_before_model(state: AgentState, runtime: Runtime) -> None:
                print(f"About to call model with {len(state['messages'])} messages")
            ```

        !!! example "带自定义状态模式的用法"

            ```python
            @before_model(can_jump_to=["end"])
            def conditional_before_model(
                state: AgentState, runtime: Runtime
            ) -> dict[str, Any] | None:
                if some_condition(state):
                    return {"jump_to": "end"}
                return None
            ```

        当您的中间件需要读取或写入默认智能体状态之外的额外状态字段时，使用自定义状态模式。

        !!! example "在模型调用前流式输出自定义事件"

            ```python
            @before_model(state_schema=MyCustomState)
            def custom_before_model(state: MyCustomState, runtime: Runtime) -> dict[str, Any]:
                return {"custom_field": "updated_value"}
            ```

        使用 `runtime.stream_writer` 在每次模型调用前发出自定义事件。当以 `stream_mode="custom"` 流式处理时，可接收这些事件。

            Use `runtime.stream_writer` to emit custom events before each model invocation.
            Events are received when streaming with `stream_mode="custom"`.

            ```python
            @before_model
            async def notify_model_call(state: AgentState, runtime: Runtime) -> None:
                '''Notify user before model is called.'''
                runtime.stream_writer(
                    {
                        "type": "status",
                        "message": "Thinking...",
                    }
                )
            ```
    """

    def decorator(
        func: _CallableWithStateAndRuntime[StateT, ContextT],
    ) -> AgentMiddleware[StateT, ContextT]:
        is_async = iscoroutinefunction(func)

        func_can_jump_to = (
            can_jump_to if can_jump_to is not None else getattr(func, "__can_jump_to__", [])
        )

        if is_async:

            async def async_wrapped(
                _self: AgentMiddleware[StateT, ContextT],
                state: StateT,
                runtime: Runtime[ContextT],
            ) -> dict[str, Any] | Command[Any] | None:
                                # `iscoroutinefunction` narrows `func` at runtime, but type checkers
                                # cannot narrow this sync-or-async callable union.
                return await cast("_AsyncCallableWithStateAndRuntime[StateT, ContextT]", func)(
                    state, runtime
                )

                        # Preserve can_jump_to metadata on the wrapped function
            if func_can_jump_to:
                async_wrapped.__can_jump_to__ = func_can_jump_to  # type: ignore[attr-defined]

            middleware_name = name or cast(
                "str", getattr(func, "__name__", "BeforeModelMiddleware")
            )

                        # `type(...)` builds the correct middleware subclass at runtime, but
                        # type checkers cannot infer its generic `AgentMiddleware` parameters.
            return cast(
                "AgentMiddleware[StateT, ContextT]",
                type(
                    middleware_name,
                    (AgentMiddleware,),
                    {
                        "state_schema": state_schema or AgentState,
                        "tools": tools or [],
                        "abefore_model": async_wrapped,
                    },
                )(),
            )

        def wrapped(
            _self: AgentMiddleware[StateT, ContextT],
            state: StateT,
            runtime: Runtime[ContextT],
        ) -> dict[str, Any] | Command[Any] | None:
                        # `iscoroutinefunction` narrows `func` at runtime, but type checkers
                        # cannot narrow this sync-or-async callable union.
            return cast("_SyncCallableWithStateAndRuntime[StateT, ContextT]", func)(state, runtime)

                # Preserve can_jump_to metadata on the wrapped function
        if func_can_jump_to:
            wrapped.__can_jump_to__ = func_can_jump_to  # type: ignore[attr-defined]

                # Use function name as default if no name provided
        middleware_name = name or cast("str", getattr(func, "__name__", "BeforeModelMiddleware"))

                # `type(...)` builds the correct middleware subclass at runtime, but
                # type checkers cannot infer its generic `AgentMiddleware` parameters.
        return cast(
            "AgentMiddleware[StateT, ContextT]",
            type(
                middleware_name,
                (AgentMiddleware,),
                {
                    "state_schema": state_schema or AgentState,
                    "tools": tools or [],
                    "before_model": wrapped,
                },
            )(),
        )

    if func is not None:
        return decorator(func)
    return decorator


@overload
def after_model(
    func: _CallableWithStateAndRuntime[StateT, ContextT],
) -> AgentMiddleware[StateT, ContextT]: ...


@overload
def after_model(
    func: None = None,
    *,
    state_schema: type[StateT] | None = None,
    tools: list[BaseTool] | None = None,
    can_jump_to: list[JumpTo] | None = None,
    name: str | None = None,
) -> Callable[
    [_CallableWithStateAndRuntime[StateT, ContextT]], AgentMiddleware[StateT, ContextT]
]: ...


def after_model(
    func: _CallableWithStateAndRuntime[StateT, ContextT] | None = None,
    *,
    state_schema: type[StateT] | None = None,
    tools: list[BaseTool] | None = None,
    can_jump_to: list[JumpTo] | None = None,
    name: str | None = None,
) -> (
    Callable[[_CallableWithStateAndRuntime[StateT, ContextT]], AgentMiddleware[StateT, ContextT]]
    | AgentMiddleware[StateT, ContextT]
):
    """func: 要装饰的函数。

    Args:
        func: 要装饰的函数。

            必须接受：`state: StateT, runtime: Runtime[ContextT]` - 状态和运行时上下文
            state_schema: 可选的自定义状态模式类型。

            若未提供，则使用默认的 `AgentState` 模式。
        tools: 可选的额外工具列表，用于注册到此中间件。
        can_jump_to: 可选的条件边有效跳转目标列表。

            Valid values are: 有效值为：`'tools'`, `'model'`, `'end'`
        name: 生成的中间件类的可选名称。

            若未提供，则使用被装饰函数的名称。

    Returns:
        一个 `AgentMiddleware` 实例（若提供了 func）或可应用于函数的装饰器函数。

    被装饰的函数应返回：

    - `dict[str, Any]` - 要合并到智能体状态中的状态更新
    - `Command` - 控制流的命令（例如跳转到不同节点）
    - `None` - 无状态更新或流控

    Examples:
        !!! example "记录模型响应的基本用法"

            ```python
            @after_model
            def log_latest_message(state: AgentState, runtime: Runtime) -> None:
                print(state["messages"][-1].content)
            ```

        !!! example "使用自定义状态模式"

        当你的中间件需要读取或写入默认智能体状态之外的额外状态字段时，使用自定义状态模式。

            ```python
            @after_model(state_schema=MyCustomState, name="MyAfterModelMiddleware")
            def custom_after_model(state: MyCustomState, runtime: Runtime) -> dict[str, Any]:
                return {"custom_field": "updated_after_model"}
            ```

        !!! example "在模型调用后流式输出自定义事件"

            使用 `runtime.stream_writer` 在模型响应后发出自定义事件。
            当以 `stream_mode="custom"` 流式处理时可接收这些事件。

            ```python
            @after_model
            async def notify_model_response(state: AgentState, runtime: Runtime) -> None:
                '''Notify user after model has responded.'''
                last_message = state["messages"][-1]
                has_tool_calls = hasattr(last_message, "tool_calls") and last_message.tool_calls
                runtime.stream_writer(
                    {
                        "type": "status",
                        "message": "Using tools..." if has_tool_calls else "Response ready!",
                    }
                )
            ```
    """

    def decorator(
        func: _CallableWithStateAndRuntime[StateT, ContextT],
    ) -> AgentMiddleware[StateT, ContextT]:
        is_async = iscoroutinefunction(func)
                # Extract can_jump_to from decorator parameter or from function metadata
        func_can_jump_to = (
            can_jump_to if can_jump_to is not None else getattr(func, "__can_jump_to__", [])
        )

        if is_async:

            async def async_wrapped(
                _self: AgentMiddleware[StateT, ContextT],
                state: StateT,
                runtime: Runtime[ContextT],
            ) -> dict[str, Any] | Command[Any] | None:
                                # `iscoroutinefunction` narrows `func` at runtime, but type checkers
                                # cannot narrow this sync-or-async callable union.
                return await cast("_AsyncCallableWithStateAndRuntime[StateT, ContextT]", func)(
                    state, runtime
                )

                        # Preserve can_jump_to metadata on the wrapped function
            if func_can_jump_to:
                async_wrapped.__can_jump_to__ = func_can_jump_to  # type: ignore[attr-defined]

            middleware_name = name or cast("str", getattr(func, "__name__", "AfterModelMiddleware"))

                        # `type(...)` builds the correct middleware subclass at runtime, but
                        # type checkers cannot infer its generic `AgentMiddleware` parameters.
            return cast(
                "AgentMiddleware[StateT, ContextT]",
                type(
                    middleware_name,
                    (AgentMiddleware,),
                    {
                        "state_schema": state_schema or AgentState,
                        "tools": tools or [],
                        "aafter_model": async_wrapped,
                    },
                )(),
            )

        def wrapped(
            _self: AgentMiddleware[StateT, ContextT],
            state: StateT,
            runtime: Runtime[ContextT],
        ) -> dict[str, Any] | Command[Any] | None:
                        # `iscoroutinefunction` narrows `func` at runtime, but type checkers
                        # cannot narrow this sync-or-async callable union.
            return cast("_SyncCallableWithStateAndRuntime[StateT, ContextT]", func)(state, runtime)

                # Preserve can_jump_to metadata on the wrapped function
        if func_can_jump_to:
            wrapped.__can_jump_to__ = func_can_jump_to  # type: ignore[attr-defined]

                # Use function name as default if no name provided
        middleware_name = name or cast("str", getattr(func, "__name__", "AfterModelMiddleware"))

                # `type(...)` builds the correct middleware subclass at runtime, but
                # type checkers cannot infer its generic `AgentMiddleware` parameters.
        return cast(
            "AgentMiddleware[StateT, ContextT]",
            type(
                middleware_name,
                (AgentMiddleware,),
                {
                    "state_schema": state_schema or AgentState,
                    "tools": tools or [],
                    "after_model": wrapped,
                },
            )(),
        )

    if func is not None:
        return decorator(func)
    return decorator


@overload
def before_agent(
    func: _CallableWithStateAndRuntime[StateT, ContextT],
) -> AgentMiddleware[StateT, ContextT]: ...


@overload
def before_agent(
    func: None = None,
    *,
    state_schema: type[StateT] | None = None,
    tools: list[BaseTool] | None = None,
    can_jump_to: list[JumpTo] | None = None,
    name: str | None = None,
) -> Callable[
    [_CallableWithStateAndRuntime[StateT, ContextT]], AgentMiddleware[StateT, ContextT]
]: ...


def before_agent(
    func: _CallableWithStateAndRuntime[StateT, ContextT] | None = None,
    *,
    state_schema: type[StateT] | None = None,
    tools: list[BaseTool] | None = None,
    can_jump_to: list[JumpTo] | None = None,
    name: str | None = None,
) -> (
    Callable[[_CallableWithStateAndRuntime[StateT, ContextT]], AgentMiddleware[StateT, ContextT]]
    | AgentMiddleware[StateT, ContextT]
):
    """用于动态创建带有 `before_agent` 钩子的中间件的装饰器。

    Args:
        func: 被装饰的函数。

            必须接受：`state: StateT, runtime: Runtime[ContextT]` - 状态和运行时上下文
            state_schema: 可选的自定义状态模式类型。

            若未提供，则使用默认的 `AgentState` 模式。
        tools: 可选的额外工具列表，用于注册到此中间件。
        can_jump_to: 可选的条件边有效跳转目标列表。

            Valid values are: 有效值为：`'tools'`, `'model'`, `'end'`
        name: 生成的中间件类的可选名称。

            若未提供，则使用被装饰函数的名称。

    Returns:
        一个 `AgentMiddleware` 实例（若直接提供了 func）或可应用于函数的装饰器函数。

    被装饰的函数应返回：

    - `dict[str, Any]` - 要合并到智能体状态中的状态更新
    - `Command` - 控制流的命令（例如跳转到不同节点）
    - `None` - 无状态更新或流控

    Examples:
        !!! example "基本用法"

            ```python
            @before_agent
            def log_before_agent(state: AgentState, runtime: Runtime) -> None:
                print(f"Starting agent with {len(state['messages'])} messages")
            ```

        !!! example "带条件跳转的用法"

            ```python
            @before_agent(can_jump_to=["end"])
            def conditional_before_agent(
                state: AgentState, runtime: Runtime
            ) -> dict[str, Any] | None:
                if some_condition(state):
                    return {"jump_to": "end"}
                return None
            ```

        !!! example "使用自定义状态模式"

        当你的中间件需要读取或写入默认智能体状态之外的额外状态字段时，使用自定义状态模式。

            ```python
            @before_agent(state_schema=MyCustomState)
            def custom_before_agent(state: MyCustomState, runtime: Runtime) -> dict[str, Any]:
                return {"custom_field": "initialized_value"}
            ```

        !!! example "流式输出自定义事件"

            使用 `runtime.stream_writer` 在智能体执行期间发出自定义事件。
            当以 `stream_mode="custom"` 流式传输时接收事件。

            ```python
            from langchain.agents import create_agent
            from langchain.agents.middleware import before_agent, AgentState
            from langchain.messages import HumanMessage
            from langgraph.runtime import Runtime


            @before_agent
            async def notify_start(state: AgentState, runtime: Runtime) -> None:
                '''Notify user that agent is starting.'''
                runtime.stream_writer(
                    {
                        "type": "status",
                        "message": "Initializing agent session...",
                    }
                )
                # Perform prerequisite tasks here
                runtime.stream_writer({"type": "status", "message": "Agent ready!"})


            agent = create_agent(
                model="openai:gpt-5.2",
                tools=[...],
                middleware=[notify_start],
            )

            # Consume with stream_mode="custom" to receive events
            async for mode, event in agent.astream(
                {"messages": [HumanMessage("Hello")]},
                stream_mode=["updates", "custom"],
            ):
                if mode == "custom":
                    print(f"Status: {event}")
            ```
    """

    def decorator(
        func: _CallableWithStateAndRuntime[StateT, ContextT],
    ) -> AgentMiddleware[StateT, ContextT]:
        is_async = iscoroutinefunction(func)

        func_can_jump_to = (
            can_jump_to if can_jump_to is not None else getattr(func, "__can_jump_to__", [])
        )

        if is_async:

            async def async_wrapped(
                _self: AgentMiddleware[StateT, ContextT],
                state: StateT,
                runtime: Runtime[ContextT],
            ) -> dict[str, Any] | Command[Any] | None:
                                # `iscoroutinefunction` 在运行时缩小 `func` 的类型，但类型检查器
                                # 无法缩小此同步或异步可调用联合类型。
                return await cast("_AsyncCallableWithStateAndRuntime[StateT, ContextT]", func)(
                    state, runtime
                )

                        # 在包装函数上保留 `can_jump_to` 元数据
            if func_can_jump_to:
                async_wrapped.__can_jump_to__ = func_can_jump_to  # type: ignore[attr-defined]

            middleware_name = name or cast(
                "str", getattr(func, "__name__", "BeforeAgentMiddleware")
            )

                        # `type(...)` 在运行时构建正确的中间件子类，但
                        # 类型检查器无法推断其泛型 `AgentMiddleware` 参数。
            return cast(
                "AgentMiddleware[StateT, ContextT]",
                type(
                    middleware_name,
                    (AgentMiddleware,),
                    {
                        "state_schema": state_schema or AgentState,
                        "tools": tools or [],
                        "abefore_agent": async_wrapped,
                    },
                )(),
            )

        def wrapped(
            _self: AgentMiddleware[StateT, ContextT],
            state: StateT,
            runtime: Runtime[ContextT],
        ) -> dict[str, Any] | Command[Any] | None:
                        # `iscoroutinefunction` 在运行时缩小 `func` 的类型，但类型检查器
                        # 无法缩小此同步或异步可调用联合类型。
            return cast("_SyncCallableWithStateAndRuntime[StateT, ContextT]", func)(state, runtime)

                # 在包装函数上保留 `can_jump_to` 元数据
        if func_can_jump_to:
            wrapped.__can_jump_to__ = func_can_jump_to  # type: ignore[attr-defined]

                # 如果未提供名称，则使用函数名作为默认值
        middleware_name = name or cast("str", getattr(func, "__name__", "BeforeAgentMiddleware"))

                # `type(...)` 在运行时构建正确的中间件子类，但
                # 类型检查器无法推断其泛型 `AgentMiddleware` 参数。
        return cast(
            "AgentMiddleware[StateT, ContextT]",
            type(
                middleware_name,
                (AgentMiddleware,),
                {
                    "state_schema": state_schema or AgentState,
                    "tools": tools or [],
                    "before_agent": wrapped,
                },
            )(),
        )

    if func is not None:
        return decorator(func)
    return decorator


@overload
def after_agent(
    func: _CallableWithStateAndRuntime[StateT, ContextT],
) -> AgentMiddleware[StateT, ContextT]: ...


@overload
def after_agent(
    func: None = None,
    *,
    state_schema: type[StateT] | None = None,
    tools: list[BaseTool] | None = None,
    can_jump_to: list[JumpTo] | None = None,
    name: str | None = None,
) -> Callable[
    [_CallableWithStateAndRuntime[StateT, ContextT]], AgentMiddleware[StateT, ContextT]
]: ...


def after_agent(
    func: _CallableWithStateAndRuntime[StateT, ContextT] | None = None,
    *,
    state_schema: type[StateT] | None = None,
    tools: list[BaseTool] | None = None,
    can_jump_to: list[JumpTo] | None = None,
    name: str | None = None,
) -> (
    Callable[[_CallableWithStateAndRuntime[StateT, ContextT]], AgentMiddleware[StateT, ContextT]]
    | AgentMiddleware[StateT, ContextT]
):
    """用于动态创建带有 `after_agent` 钩子的中间件的装饰器。

    异步版本为 `aafter_agent`。

    Args:
        func: 要装饰的函数。

            Must accept: 必须接受: `state: StateT, runtime: Runtime[ContextT]` - 状态和运行时
            上下文
        state_schema: 可选的自定义状态模式类型。

            如果未提供，则使用默认的 `AgentState` 模式。
        tools: 可选的额外工具列表，用于注册此中间件。
        can_jump_to: 可选的条件边有效跳转目标列表。

            Valid values are: 有效值为: `'tools'`, `'model'`, `'end'`
        name: 生成的中间件类的可选名称。

            如果未提供，则使用被装饰函数的名称。

    Returns:
        一个 `AgentMiddleware` 实例（如果提供了 func）或可应用于函数的装饰器
            函数。

    被装饰的函数应返回：

    - `dict[str, Any]` - 要合并到智能体状态中的状态更新
    - `Command` - 控制流的命令（例如，跳转到不同节点）
    - `None` - 无状态更新或流控

    Examples:
        !!! example "记录智能体完成的基本用法"

            ```python
            @after_agent
            def log_completion(state: AgentState, runtime: Runtime) -> None:
                print(f"Agent completed with {len(state['messages'])} messages")
            ```

        !!! example "使用自定义状态模式"

        当你的中间件需要读取或写入默认智能体状态之外的额外状态字段时，使用自定义状态模式。

            ```python
            @after_agent(state_schema=MyCustomState, name="MyAfterAgentMiddleware")
            def custom_after_agent(state: MyCustomState, runtime: Runtime) -> dict[str, Any]:
                return {"custom_field": "finalized_value"}
            ```

        !!! example "在完成时流式输出自定义事件"

            使用 `runtime.stream_writer` 在智能体完成时发出自定义事件。
            当以 `stream_mode="custom"` 流式传输时接收事件。

            ```python
            @after_agent
            async def notify_completion(state: AgentState, runtime: Runtime) -> None:
                '''Notify user that agent has completed.'''
                runtime.stream_writer(
                    {
                        "type": "status",
                        "message": "Agent execution complete!",
                        "total_messages": len(state["messages"]),
                    }
                )
            ```
    """

    def decorator(
        func: _CallableWithStateAndRuntime[StateT, ContextT],
    ) -> AgentMiddleware[StateT, ContextT]:
        is_async = iscoroutinefunction(func)
                # 从装饰器参数或函数元数据中提取 `can_jump_to`
        func_can_jump_to = (
            can_jump_to if can_jump_to is not None else getattr(func, "__can_jump_to__", [])
        )

        if is_async:

            async def async_wrapped(
                _self: AgentMiddleware[StateT, ContextT],
                state: StateT,
                runtime: Runtime[ContextT],
            ) -> dict[str, Any] | Command[Any] | None:
                                # `iscoroutinefunction` 在运行时缩小 `func` 的类型，但类型检查器
                                # 无法缩小此同步或异步可调用联合类型。
                return await cast("_AsyncCallableWithStateAndRuntime[StateT, ContextT]", func)(
                    state, runtime
                )

                        # 在包装函数上保留 `can_jump_to` 元数据
            if func_can_jump_to:
                async_wrapped.__can_jump_to__ = func_can_jump_to  # type: ignore[attr-defined]

            middleware_name = name or cast("str", getattr(func, "__name__", "AfterAgentMiddleware"))

                        # `type(...)` 在运行时构建正确的中间件子类，但
                        # 类型检查器无法推断其泛型 `AgentMiddleware` 参数。
            return cast(
                "AgentMiddleware[StateT, ContextT]",
                type(
                    middleware_name,
                    (AgentMiddleware,),
                    {
                        "state_schema": state_schema or AgentState,
                        "tools": tools or [],
                        "aafter_agent": async_wrapped,
                    },
                )(),
            )

        def wrapped(
            _self: AgentMiddleware[StateT, ContextT],
            state: StateT,
            runtime: Runtime[ContextT],
        ) -> dict[str, Any] | Command[Any] | None:
                        # `iscoroutinefunction` 在运行时缩小 `func` 的类型，但类型检查器
                        # 无法缩小此同步或异步可调用联合类型。
            return cast("_SyncCallableWithStateAndRuntime[StateT, ContextT]", func)(state, runtime)

                # 在包装函数上保留 `can_jump_to` 元数据
        if func_can_jump_to:
            wrapped.__can_jump_to__ = func_can_jump_to  # type: ignore[attr-defined]

                # 如果未提供名称，则使用函数名作为默认值
        middleware_name = name or cast("str", getattr(func, "__name__", "AfterAgentMiddleware"))

                # `type(...)` 在运行时构建正确的中间件子类，但
                # 类型检查器无法推断其泛型 `AgentMiddleware` 参数。
        return cast(
            "AgentMiddleware[StateT, ContextT]",
            type(
                middleware_name,
                (AgentMiddleware,),
                {
                    "state_schema": state_schema or AgentState,
                    "tools": tools or [],
                    "after_agent": wrapped,
                },
            )(),
        )

    if func is not None:
        return decorator(func)
    return decorator


@overload
def dynamic_prompt(
    func: _CallableReturningSystemMessage[ContextT],
) -> AgentMiddleware[StateT, ContextT]: ...


@overload
def dynamic_prompt(
    func: None = None,
) -> Callable[
    [_CallableReturningSystemMessage[ContextT]],
    AgentMiddleware[StateT, ContextT],
]: ...


def dynamic_prompt(
    func: _CallableReturningSystemMessage[ContextT] | None = None,
) -> (
    Callable[
        [_CallableReturningSystemMessage[ContextT]],
        AgentMiddleware[StateT, ContextT],
    ]
    | AgentMiddleware[StateT, ContextT]
):
    """用于动态生成模型系统提示的装饰器。

    This is a convenience decorator that creates middleware using `wrap_model_call`
    specifically for dynamic prompt generation. The decorated function should return
    a string that will be set as the system prompt for the model request.

    Args:
        func: 要装饰的函数。

            Must accept: 必须接受: `request: ModelRequest` - 模型请求（包含状态和
            运行时）

    Returns:
        一个 `AgentMiddleware` 实例（如果提供了 func）或可应用于函数的装饰器
            函数。

    被装饰的函数应返回：
        - `str` – 用于模型请求的系统提示字符串
        - `SystemMessage` – 用于模型请求的完整系统消息

    Examples:
        带有动态内容的基本用法：

        ```python
        @dynamic_prompt
        def my_prompt(request: ModelRequest) -> str:
            user_name = request.runtime.context.get("user_name", "User")
            return f"You are a helpful assistant helping {user_name}."
        ```

        使用状态自定义提示：

        ```python
        @dynamic_prompt
        def context_aware_prompt(request: ModelRequest) -> str:
            msg_count = len(request.state["messages"])
            if msg_count > 10:
                return "You are in a long conversation. Be concise."
            return "You are a helpful assistant."
        ```

        与智能体一起使用：

        ```python
        agent = create_agent(model, middleware=[my_prompt])
        ```
    """

    def decorator(
        func: _CallableReturningSystemMessage[ContextT],
    ) -> AgentMiddleware[StateT, ContextT]:
        is_async = iscoroutinefunction(func)

        if is_async:

            async def async_wrapped(
                _self: AgentMiddleware[StateT, ContextT],
                request: ModelRequest[ContextT],
                handler: Callable[[ModelRequest[ContextT]], Awaitable[ModelResponse[Any]]],
            ) -> ModelResponse[Any] | AIMessage:
                prompt = await cast("_AsyncCallableReturningSystemMessage[ContextT]", func)(request)
                if isinstance(prompt, SystemMessage):
                    request = request.override(system_message=prompt)
                else:
                    request = request.override(system_message=SystemMessage(content=prompt))
                return await handler(request)

            middleware_name = cast("str", getattr(func, "__name__", "DynamicPromptMiddleware"))

                        # `type(...)` 在运行时构建正确的中间件子类，但
                        # 类型检查器无法推断其泛型 `AgentMiddleware` 参数。
            return cast(
                "AgentMiddleware[StateT, ContextT]",
                type(
                    middleware_name,
                    (AgentMiddleware,),
                    {
                        "state_schema": AgentState,
                        "tools": [],
                        "awrap_model_call": async_wrapped,
                    },
                )(),
            )

        def wrapped(
            _self: AgentMiddleware[StateT, ContextT],
            request: ModelRequest[ContextT],
            handler: Callable[[ModelRequest[ContextT]], ModelResponse[Any]],
        ) -> ModelResponse[Any] | AIMessage:
            prompt = cast("_SyncCallableReturningSystemMessage[ContextT]", func)(request)
            if isinstance(prompt, SystemMessage):
                request = request.override(system_message=prompt)
            else:
                request = request.override(system_message=SystemMessage(content=prompt))
            return handler(request)

        async def async_wrapped_from_sync(
            _self: AgentMiddleware[StateT, ContextT],
            request: ModelRequest[ContextT],
            handler: Callable[[ModelRequest[ContextT]], Awaitable[ModelResponse[Any]]],
        ) -> ModelResponse[Any] | AIMessage:
                        # 委托给同步函数
            prompt = cast("_SyncCallableReturningSystemMessage[ContextT]", func)(request)
            if isinstance(prompt, SystemMessage):
                request = request.override(system_message=prompt)
            else:
                request = request.override(system_message=SystemMessage(content=prompt))
            return await handler(request)

        middleware_name = cast("str", getattr(func, "__name__", "DynamicPromptMiddleware"))

                # `type(...)` 在运行时构建正确的中间件子类，但
                # 类型检查器无法推断其泛型 `AgentMiddleware` 参数。
        return cast(
            "AgentMiddleware[StateT, ContextT]",
            type(
                middleware_name,
                (AgentMiddleware,),
                {
                    "state_schema": AgentState,
                    "tools": [],
                    "wrap_model_call": wrapped,
                    "awrap_model_call": async_wrapped_from_sync,
                },
            )(),
        )

    if func is not None:
        return decorator(func)
    return decorator


@overload
def wrap_model_call(
    func: _CallableReturningModelResponse[ContextT, ResponseT],
) -> AgentMiddleware[StateT, ContextT]: ...


@overload
def wrap_model_call(
    func: None = None,
    *,
    state_schema: type[StateT] | None = None,
    tools: list[BaseTool] | None = None,
    name: str | None = None,
) -> Callable[
    [_CallableReturningModelResponse[ContextT, ResponseT]],
    AgentMiddleware[StateT, ContextT],
]: ...


def wrap_model_call(
    func: _CallableReturningModelResponse[ContextT, ResponseT] | None = None,
    *,
    state_schema: type[StateT] | None = None,
    tools: list[BaseTool] | None = None,
    name: str | None = None,
) -> (
    Callable[
        [_CallableReturningModelResponse[ContextT, ResponseT]],
        AgentMiddleware[StateT, ContextT],
    ]
    | AgentMiddleware[StateT, ContextT]
):
    """使用 `wrap_model_call` 钩子从函数创建中间件。

    Converts a function with handler callback into middleware that can intercept model
    calls, implement retry logic, handle errors, and rewrite responses.

    Args:
        func: 接受 (request, handler) 的函数，调用 handler(request) 执行模型并返回
        `ModelResponse` 或 `AIMessage`。 request 包含状态和运行时。 state_schema:
        自定义状态模式。

            默认为 `AgentState`。 tools: 额外要注册到此中间件的工具。 name: 中间件类名。

            Defaults to `AgentState`.
        tools: Additional tools to register with this middleware.
        name: Middleware class name.

            !!! example "基本重试逻辑"

    Returns:
        !!! example "模型回退"

    Examples:
        !!! example "重写响应内容（完整 `ModelResponse`）"

            ```python
            @wrap_model_call
            def retry_on_error(request, handler):
                max_retries = 3
                for attempt in range(max_retries):
                    try:
                        return handler(request)
                    except Exception:
                        if attempt == max_retries - 1:
                            raise
            ```

        !!! example "简单 `AIMessage` 返回（自动转换）"

            ```python
            @wrap_model_call
            def fallback_model(request, handler):
                # Try primary model
                try:
                    return handler(request)
                except Exception:
                    pass

                # Try fallback model
                request = request.override(model=fallback_model_instance)
                return handler(request)
            ```

        创建带有 `wrap_tool_call` 钩子的中间件，由函数生成。

            ```python
            @wrap_model_call
            def uppercase_responses(request, handler):
                response = handler(request)
                ai_msg = response.result[0]
                return ModelResponse(
                    result=[AIMessage(content=ai_msg.content.upper())],
                    structured_response=response.structured_response,
                )
            ```

        异步版本为 `awrap_tool_call`。

            ```python
            @wrap_model_call
            def simple_response(request, handler):
                # AIMessage is automatically converted to ModelResponse
                return AIMessage(content="Simple response")
            ```
    """

    def decorator(
        func: _CallableReturningModelResponse[ContextT, ResponseT],
    ) -> AgentMiddleware[StateT, ContextT]:
        is_async = iscoroutinefunction(func)

        if is_async:

            async def async_wrapped(
                _self: AgentMiddleware[StateT, ContextT],
                request: ModelRequest[ContextT],
                handler: Callable[[ModelRequest[ContextT]], Awaitable[ModelResponse[ResponseT]]],
            ) -> ModelCallResult[ResponseT]:
                return await cast(
                    "_AsyncCallableReturningModelResponse[ContextT, ResponseT]", func
                )(request, handler)

            middleware_name = name or cast(
                "str", getattr(func, "__name__", "WrapModelCallMiddleware")
            )

                        # `type(...)` 在运行时构建正确的中间件子类，但
                        # 类型检查器无法推断其泛型 `AgentMiddleware` 参数。
            return cast(
                "AgentMiddleware[StateT, ContextT]",
                type(
                    middleware_name,
                    (AgentMiddleware,),
                    {
                        "state_schema": state_schema or AgentState,
                        "tools": tools or [],
                        "awrap_model_call": async_wrapped,
                    },
                )(),
            )

        def wrapped(
            _self: AgentMiddleware[StateT, ContextT],
            request: ModelRequest[ContextT],
            handler: Callable[[ModelRequest[ContextT]], ModelResponse[ResponseT]],
        ) -> ModelCallResult[ResponseT]:
            return cast("_SyncCallableReturningModelResponse[ContextT, ResponseT]", func)(
                request, handler
            )

        middleware_name = name or cast("str", getattr(func, "__name__", "WrapModelCallMiddleware"))

                # `type(...)` 在运行时构建正确的中间件子类，但
                # 类型检查器无法推断其泛型 `AgentMiddleware` 参数。
        return cast(
            "AgentMiddleware[StateT, ContextT]",
            type(
                middleware_name,
                (AgentMiddleware,),
                {
                    "state_schema": state_schema or AgentState,
                    "tools": tools or [],
                    "wrap_model_call": wrapped,
                },
            )(),
        )

    if func is not None:
        return decorator(func)
    return decorator


@overload
def wrap_tool_call(
    func: _CallableReturningToolResponse,
) -> AgentMiddleware[StateT, ContextT]: ...


@overload
def wrap_tool_call(
    func: None = None,
    *,
    state_schema: type[StateT] | None = None,
    tools: list[BaseTool] | None = None,
    name: str | None = None,
) -> Callable[
    [_CallableReturningToolResponse],
    AgentMiddleware[StateT, ContextT],
]: ...


def wrap_tool_call(
    func: _CallableReturningToolResponse | None = None,
    *,
    state_schema: type[StateT] | None = None,
    tools: list[BaseTool] | None = None,
    name: str | None = None,
) -> (
    Callable[
        [_CallableReturningToolResponse],
        AgentMiddleware[StateT, ContextT],
    ]
    | AgentMiddleware[StateT, ContextT]
):
    """将一个带有 handler 回调的函数转换为中间件，可拦截工具调用、实现重试逻辑、监控执行过程并修改响应。

    func: 接受 (request, handler) 的函数，调用 handler(request) 执行工具并返回最终的
    `ToolMessage` 或 `Command`。 state_schema: 可选的自定义状态模式类型。

    Converts a function with handler callback into middleware that can intercept
    tool calls, implement retry logic, monitor execution, and modify responses.

    Args:
        默认为函数名。 `AgentMiddleware` 实例（如果提供了 func），否则为装饰器。

            !!! example "重试逻辑"

            If not provided, uses the default `AgentState` schema.
        tools: Additional tools to register with this middleware.
        name: Middleware class name.

            !!! example "修改请求"

    Returns:
        !!! example "使用缓存结果短路"

    Examples:
        !!! example "使用自定义状态模式"

            ```python
            @wrap_tool_call
            def retry_on_error(request, handler):
                max_retries = 3
                for attempt in range(max_retries):
                    try:
                        return handler(request)
                    except Exception:
                        if attempt == max_retries - 1:
                            raise
            ```

        使用自定义状态模式时，当中间件需要读取或写入默认代理状态之外的额外状态字段。

            ```python
            @wrap_tool_call
            async def async_retry(request, handler):
                for attempt in range(3):
                    try:
                        return await handler(request)
                    except Exception:
                        if attempt == 2:
                            raise
            ```

        初始化 `ModelRequest`，保留对 `system_prompt` 的向后兼容性。

            ```python
            @wrap_tool_call
            def modify_args(request, handler):
                modified_call = {
                    **request.tool_call,
                    "args": {
                        **request.tool_call["args"],
                        "value": request.tool_call["args"]["value"] * 2,
                    },
                }
                request = request.override(tool_call=modified_call)
                return handler(request)
            ```

        !!! example "Short-circuit with cached result"

            ```python
            @wrap_tool_call
            def with_cache(request, handler):
                if cached := get_cache(request):
                    return ToolMessage(content=cached, tool_call_id=request.tool_call["id"])
                result = handler(request)
                save_cache(request, result)
                return result
            ```

        ValueError: 如果同时提供了 `system_prompt` 和 `system_message`。

        获取来自 system_message 的系统提示文本。

            ```python
            @wrap_tool_call(state_schema=MyCustomState)
            def custom_wrap_tool_call(request, handler):
                return handler(request)
            ```
    """

    def decorator(
        func: _CallableReturningToolResponse,
    ) -> AgentMiddleware[StateT, ContextT]:
        is_async = iscoroutinefunction(func)

        if is_async:

            async def async_wrapped(
                _self: AgentMiddleware[StateT, ContextT],
                request: ToolCallRequest,
                handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command[Any]]],
            ) -> ToolMessage | Command[Any]:
                return await cast("_AsyncCallableReturningToolResponse", func)(request, handler)

            middleware_name = name or cast(
                "str", getattr(func, "__name__", "WrapToolCallMiddleware")
            )

                        # `type(...)` 在运行时构建正确的中间件子类，但
                        # 类型检查器无法推断其泛型 `AgentMiddleware` 参数。
            return cast(
                "AgentMiddleware[StateT, ContextT]",
                type(
                    middleware_name,
                    (AgentMiddleware,),
                    {
                        "state_schema": state_schema or AgentState,
                        "tools": tools or [],
                        "awrap_tool_call": async_wrapped,
                    },
                )(),
            )

        def wrapped(
            _self: AgentMiddleware[StateT, ContextT],
            request: ToolCallRequest,
            handler: Callable[[ToolCallRequest], ToolMessage | Command[Any]],
        ) -> ToolMessage | Command[Any]:
            return cast("_SyncCallableReturningToolResponse", func)(request, handler)

        middleware_name = name or cast("str", getattr(func, "__name__", "WrapToolCallMiddleware"))

                # `type(...)` 在运行时构建正确的中间件子类，但
                # 类型检查器无法推断其泛型 `AgentMiddleware` 参数。
        return cast(
            "AgentMiddleware[StateT, ContextT]",
            type(
                middleware_name,
                (AgentMiddleware,),
                {
                    "state_schema": state_schema or AgentState,
                    "tools": tools or [],
                    "wrap_tool_call": wrapped,
                },
            )(),
        )

    if func is not None:
        return decorator(func)
    return decorator

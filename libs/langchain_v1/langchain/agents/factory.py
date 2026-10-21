"""支持中间件的智能体工厂，用于创建智能体。"""

from __future__ import annotations

import functools
import importlib
import itertools
import re
from dataclasses import dataclass, field, fields
from typing import (
    TYPE_CHECKING,
    Annotated,
    Any,
    Generic,
    cast,
    get_args,
    get_origin,
    get_type_hints,
)

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    RemoveMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import BaseTool
from langgraph._internal._runnable import RunnableCallable
from langgraph.constants import END, START
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.graph.state import StateGraph
from langgraph.prebuilt import ToolCallTransformer
from langgraph.prebuilt.tool_node import ToolNode
from langgraph.types import Command, Send
from langsmith import traceable
from typing_extensions import NotRequired, Required, TypedDict, overload

from langchain.agents._subagent_transformer import SubagentTransformer
from langchain.agents.middleware._trace_policy import (
    _node_trace_policy,
    _resolved_transform,
)
from langchain.agents.middleware.types import (
    AgentMiddleware,
    AgentState,
    ContextT,
    ExtendedModelResponse,
    InputAgentState,
    JumpTo,
    ModelRequest,
    ModelResponse,
    OmitFromSchema,
    OutputAgentState,
    ResponseT,
    StateT_co,
    ToolCallRequest,
)
from langchain.agents.structured_output import (
    AutoStrategy,
    MultipleStructuredOutputsError,
    OutputToolBinding,
    ProviderStrategy,
    ProviderStrategyBinding,
    ResponseFormat,
    StructuredOutputError,
    StructuredOutputValidationError,
    ToolStrategy,
)
from langchain.chat_models import init_chat_model


@dataclass
class _ComposedExtendedModelResponse(Generic[ResponseT]):
    """由组合的 `wrap_model_call` 中间件生成的内部结果。

    与 `ExtendedModelResponse`（面向用户的单命令）不同，此结构在组合过程中
    保存了所有中间件层累积的完整命令列表。
    """

    model_response: ModelResponse[ResponseT]
    """底层模型响应。"""

    commands: list[Command[Any]] = field(default_factory=list)
    """从所有中间件层累积的命令（内层优先，然后是外层）。"""


if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Iterable, Sequence

    from langchain_core.messages import InvalidToolCall
    from langchain_core.runnables import Runnable, RunnableConfig
    from langgraph.cache.base import BaseCache
    from langgraph.graph.state import CompiledStateGraph
    from langgraph.runtime import Runtime
    from langgraph.store.base import BaseStore
    from langgraph.stream._mux import TransformerFactory
    from langgraph.types import Checkpointer

    from langchain.agents.middleware.types import ToolCallWrapper

    _ModelCallHandler = Callable[
        [ModelRequest[ContextT], Callable[[ModelRequest[ContextT]], ModelResponse]],
        ModelResponse | AIMessage | ExtendedModelResponse,
    ]

    _ComposedModelCallHandler = Callable[
        [ModelRequest[ContextT], Callable[[ModelRequest[ContextT]], ModelResponse]],
        _ComposedExtendedModelResponse,
    ]

    _AsyncModelCallHandler = Callable[
        [ModelRequest[ContextT], Callable[[ModelRequest[ContextT]], Awaitable[ModelResponse]]],
        Awaitable[ModelResponse | AIMessage | ExtendedModelResponse],
    ]

    _ComposedAsyncModelCallHandler = Callable[
        [ModelRequest[ContextT], Callable[[ModelRequest[ContextT]], Awaitable[ModelResponse]]],
        Awaitable[_ComposedExtendedModelResponse],
    ]


STRUCTURED_OUTPUT_ERROR_TEMPLATE = "Error: {error}\n Please fix your mistakes."

DYNAMIC_TOOL_ERROR_TEMPLATE = """
Middleware added tools that the agent doesn't know how to execute.

Unknown tools: {unknown_tool_names}
Registered tools: {available_tool_names}

This happens when middleware modifies `request.tools` in `wrap_model_call` to include
tools that weren't passed to `create_agent()`.

How to fix this:

Option 1: Register tools at agent creation (recommended for most cases)
    Pass the tools to `create_agent(tools=[...])` or set them on `middleware.tools`.
    This makes tools available for every agent invocation.

Option 2: Handle dynamic tools in middleware (for tools created at runtime)
    Implement `wrap_tool_call` to execute tools that are added dynamically:

    class MyMiddleware(AgentMiddleware):
        def wrap_tool_call(self, request, handler):
            if request.tool_call["name"] == "dynamic_tool":
                # Execute the dynamic tool yourself or override with tool instance
                return handler(request.override(tool=my_dynamic_tool))
            return handler(request)
""".strip()


def _scrub_inputs(inputs: dict[str, Any]) -> dict[str, Any]:
    """在将追踪数据发送至 LangSmith 前，移除 trace 输入中的 `runtime` 和 `handler`。"""
    filtered = inputs.copy()
    filtered.pop("handler", None)
    req = filtered.get("request")
    if isinstance(req, (ModelRequest, ToolCallRequest)):
        filtered["request"] = {
            f.name: getattr(req, f.name) for f in fields(req) if f.name != "runtime"
        }
    return filtered


def _wrap_trace_kwargs(middleware: AgentMiddleware[Any, Any]) -> dict[str, Any]:
    """`traceable` 关键字参数，用于中间件的 `wrap_*` 钩子跨度。

    `_scrub_inputs` 基线（剥离不可序列化的 `handler`/`runtime`）始终最先执行； 有效
    `TracePolicy`（中间件自身的，否则为全局默认值）在其上组合。
    有效策略在调用时解析，因此在 `create_agent` 之后调用 `configure_trace_policy`
    仍会生效。
    """
    process_inputs = _resolved_transform(middleware.trace_policy, "process_inputs")
    process_outputs = _resolved_transform(middleware.trace_policy, "process_outputs")
    return {
        "process_inputs": lambda inputs: process_inputs(_scrub_inputs(inputs)),
        "process_outputs": process_outputs,
    }


FALLBACK_MODELS_WITH_STRUCTURED_OUTPUT = [
        # 如果模型配置数据不可用，匹配这些模式的模型名
        # 被认为支持提供方原生的结构化输出。这些是正则表达式
        # 因此匹配范围限定在模型名片段，而非任意子字符串。
    r"(^|[/:.])gpt-4\.1($|[-/:])",
    r"(^|[/:.])gpt-4o($|[-/:])",
    r"(^|[/:.])gpt-5($|[-/:])",
    r"(^|[/:.])gpt-5\.1($|[-/:])",
    r"(^|[/:.])gpt-5\.2(-\d{4}-\d{2}-\d{2})?($|[/:])",
    r"(^|[/:.])gpt-5\.2-(chat|codex)($|[-/:])",
    r"(^|[/:.])gpt-5\.3($|[-/:])",
    r"(^|[/:.])gpt-5\.4(-\d{4}-\d{2}-\d{2})?($|[/:])",
    r"(^|[/:.])gpt-5\.4-(mini|nano)($|[-/:])",
    r"(^|[/:.])gpt-5\.5($|[-/:])",
    r"(^|[/:.])gpt-6-(astra|luna|sol)($|[-/:])",
    r"(^|[/:.])claude-(fable|mythos)-5(?:-\d{8})?(?:-v\d(?::\d)?)?($|[/:])",
    r"(^|[/:.])claude-haiku-4-5(?:-\d{8})?(?:-v\d(?::\d)?)?($|[/:])",
    r"(^|[/:.])claude-opus-4-(5|6|7|8)(?:-\d{8})?(?:-v\d(?::\d)?)?($|[/:])",
    r"(^|[/:.])claude-sonnet-4-(5|6)(?:-\d{8})?(?:-v\d(?::\d)?)?($|[/:])",
    r"(^|[/:.])grok-4($|[-.:/])",
    r"(^|[/:.])grok-build($|[-/:])",
]


def _normalize_to_model_response(
    result: ModelResponse | AIMessage | ExtendedModelResponse,
) -> ModelResponse:
    """将中间件返回值归一化为 ModelResponse。

    At inner composition boundaries, `ExtendedModelResponse` is unwrapped to its
    underlying `ModelResponse` so that inner middleware always sees `ModelResponse`
    from the handler.
    """
    if isinstance(result, AIMessage):
        return ModelResponse(result=[result], structured_response=None)
    if isinstance(result, ExtendedModelResponse):
        return result.model_response
    return result


def _build_commands(
    model_response: ModelResponse,
    middleware_commands: list[Command[Any]] | None = None,
    *,
    has_structured_output: bool = False,
    repaired_messages: list[AnyMessage] | None = None,
) -> list[Command[Any]]:
    """从模型响应和中间件命令构建命令列表。

    第一个命令包含模型响应状态（消息和可选的 structured_response）。
    中间件命令按原样追加。

    Args:
        model_response: The model response containing messages and optional
            structured output.
        middleware_commands: Commands accumulated from middleware layers during
            composition (inner-first ordering).
        has_structured_output: Whether the agent was configured with a
            `response_format`. When `True` and no structured response was
            produced, `structured_response` is explicitly cleared to avoid a
            stale value from a previous checkpointed turn.
        repaired_messages: Complete message history after repairing invalid tool calls.

    Returns:
        准备从模型节点返回的 `Command` 对象列表。
    """
    messages = model_response.result
    if repaired_messages is not None:
        messages = [
            RemoveMessage(id=REMOVE_ALL_MESSAGES),
            *repaired_messages,
            *model_response.result,
        ]
    state: dict[str, Any] = {"messages": messages}

    if model_response.structured_response is not None:
        state["structured_response"] = model_response.structured_response
    elif has_structured_output:
        state["structured_response"] = None

    for cmd in middleware_commands or []:
        if cmd.goto:
            msg = (
                "Command goto is not yet supported in wrap_model_call middleware. "
                "Use the jump_to state field with before_model/after_model hooks instead."
            )
            raise NotImplementedError(msg)
        if cmd.resume:
            msg = "Command resume is not yet supported in wrap_model_call middleware."
            raise NotImplementedError(msg)
        if cmd.graph:
            msg = "Command graph is not yet supported in wrap_model_call middleware."
            raise NotImplementedError(msg)

    commands: list[Command[Any]] = [Command(update=state)]
    commands.extend(middleware_commands or [])
    return commands


def _chain_model_call_handlers(
    handlers: Sequence[_ModelCallHandler[ContextT]],
) -> _ComposedModelCallHandler[ContextT] | None:
    """将多个 `wrap_model_call` 处理器组合成单一中间件栈。

    Composes handlers so first in list becomes outermost layer. Each handler receives a
    handler callback to execute inner layers. Commands from each layer are accumulated
    into a list (inner-first, then outer) without merging.

    Args:
        handlers: 处理器列表。

            第一个处理器包装其余所有处理器。

    Returns:
        组合后的处理器，返回 `_ComposedExtendedModelResponse`，
        或在处理器为空时返回 `None`。
    """
    if not handlers:
        return None

    def _to_composed_result(
        result: ModelResponse | AIMessage | ExtendedModelResponse | _ComposedExtendedModelResponse,
        extra_commands: list[Command[Any]] | None = None,
    ) -> _ComposedExtendedModelResponse:
        """将任何处理器结果归一化为 _ComposedExtendedModelResponse。"""
        commands: list[Command[Any]] = list(extra_commands or [])
        if isinstance(result, _ComposedExtendedModelResponse):
            commands.extend(result.commands)
            model_response = result.model_response
        elif isinstance(result, ExtendedModelResponse):
            model_response = result.model_response
            if result.command is not None:
                commands.append(result.command)
        else:
            model_response = _normalize_to_model_response(result)

        return _ComposedExtendedModelResponse(model_response=model_response, commands=commands)

    if len(handlers) == 1:
        single_handler = handlers[0]

        def normalized_single(
            request: ModelRequest[ContextT],
            handler: Callable[[ModelRequest[ContextT]], ModelResponse],
        ) -> _ComposedExtendedModelResponse:
            return _to_composed_result(single_handler(request, handler))

        return normalized_single

    def compose_two(
        outer: _ModelCallHandler[ContextT] | _ComposedModelCallHandler[ContextT],
        inner: _ModelCallHandler[ContextT] | _ComposedModelCallHandler[ContextT],
    ) -> _ComposedModelCallHandler[ContextT]:
        """组合两个处理器，外层包裹内层。"""

        def composed(
            request: ModelRequest[ContextT],
            handler: Callable[[ModelRequest[ContextT]], ModelResponse],
        ) -> _ComposedExtendedModelResponse:
                        # 闭包变量，用于捕获内层命令，以便在归一化前使用
            accumulated_commands: list[Command[Any]] = []

            def inner_handler(req: ModelRequest[ContextT]) -> ModelResponse:
                                # 每次调用时清空以确保重试安全
                accumulated_commands.clear()
                inner_result = inner(req, handler)
                if isinstance(inner_result, _ComposedExtendedModelResponse):
                    accumulated_commands.extend(inner_result.commands)
                    return inner_result.model_response
                if isinstance(inner_result, ExtendedModelResponse):
                    if inner_result.command is not None:
                        accumulated_commands.append(inner_result.command)
                    return inner_result.model_response
                return _normalize_to_model_response(inner_result)

            outer_result = outer(request, inner_handler)
            return _to_composed_result(
                outer_result,
                extra_commands=accumulated_commands or None,
            )

        return composed

        # 从右到左组合：outer(inner(innermost(handler)))
    composed_handler = compose_two(handlers[-2], handlers[-1])
    for h in reversed(handlers[:-2]):
        composed_handler = compose_two(h, composed_handler)

    return composed_handler


def _chain_async_model_call_handlers(
    handlers: Sequence[_AsyncModelCallHandler[ContextT]],
) -> _ComposedAsyncModelCallHandler[ContextT] | None:
    """将多个异步 `wrap_model_call` 处理器组合成单一中间件栈。

    Commands from each layer are accumulated into a list (inner-first, then outer)
    without merging.

    Args:
        handlers: 异步处理器列表。

            第一个处理器包装其余所有处理器。

    Returns:
        组合后的异步处理器，返回 `_ComposedExtendedModelResponse`，
        或在处理器为空时返回 `None`。
    """
    if not handlers:
        return None

    def _to_composed_result(
        result: ModelResponse | AIMessage | ExtendedModelResponse | _ComposedExtendedModelResponse,
        extra_commands: list[Command[Any]] | None = None,
    ) -> _ComposedExtendedModelResponse:
        """将任何处理器结果归一化为 _ComposedExtendedModelResponse。"""
        commands: list[Command[Any]] = list(extra_commands or [])
        if isinstance(result, _ComposedExtendedModelResponse):
            commands.extend(result.commands)
            model_response = result.model_response
        elif isinstance(result, ExtendedModelResponse):
            model_response = result.model_response
            if result.command is not None:
                commands.append(result.command)
        else:
            model_response = _normalize_to_model_response(result)

        return _ComposedExtendedModelResponse(model_response=model_response, commands=commands)

    if len(handlers) == 1:
        single_handler = handlers[0]

        async def normalized_single(
            request: ModelRequest[ContextT],
            handler: Callable[[ModelRequest[ContextT]], Awaitable[ModelResponse]],
        ) -> _ComposedExtendedModelResponse:
            return _to_composed_result(await single_handler(request, handler))

        return normalized_single

    def compose_two(
        outer: _AsyncModelCallHandler[ContextT] | _ComposedAsyncModelCallHandler[ContextT],
        inner: _AsyncModelCallHandler[ContextT] | _ComposedAsyncModelCallHandler[ContextT],
    ) -> _ComposedAsyncModelCallHandler[ContextT]:
        """组合两个异步处理器，外层包裹内层。"""

        async def composed(
            request: ModelRequest[ContextT],
            handler: Callable[[ModelRequest[ContextT]], Awaitable[ModelResponse]],
        ) -> _ComposedExtendedModelResponse:
                        # 闭包变量，用于捕获内层命令，以便在归一化前使用
            accumulated_commands: list[Command[Any]] = []

            async def inner_handler(req: ModelRequest[ContextT]) -> ModelResponse:
                                # 每次调用时清空以确保重试安全
                accumulated_commands.clear()
                inner_result = await inner(req, handler)
                if isinstance(inner_result, _ComposedExtendedModelResponse):
                    accumulated_commands.extend(inner_result.commands)
                    return inner_result.model_response
                if isinstance(inner_result, ExtendedModelResponse):
                    if inner_result.command is not None:
                        accumulated_commands.append(inner_result.command)
                    return inner_result.model_response
                return _normalize_to_model_response(inner_result)

            outer_result = await outer(request, inner_handler)
            return _to_composed_result(
                outer_result,
                extra_commands=accumulated_commands or None,
            )

        return composed

        # 从右到左组合：outer(inner(innermost(handler)))
    composed_handler = compose_two(handlers[-2], handlers[-1])
    for h in reversed(handlers[:-2]):
        composed_handler = compose_two(h, composed_handler)

    return composed_handler


@functools.lru_cache(maxsize=100)
def _get_schema_type_hints(schema: type) -> dict[str, Any]:
    """返回给定模式的缓存类型提示。"""
    return get_type_hints(schema, include_extras=True)


def _resolve_schemas(schemas: list[type]) -> tuple[type, type, type]:
    """解析给定模式的状态、输入和输出模式。

    模式按列表顺序合并；当多个模式声明同一字段时，后面的条目会覆盖前面的。重复项无害——一个类型多次出现时，仅在其最后一次位置被处理。
    """
    schema_hints: dict[type, dict[str, Any]] = {}
    for schema in schemas:
                # 重新插入重复项，使字典迭代反映其最终位置。
        schema_hints.pop(schema, None)
        schema_hints[schema] = _get_schema_type_hints(schema)
    return (
        _resolve_schema(schema_hints, "StateSchema", None),
        _resolve_schema(schema_hints, "InputSchema", "input"),
        _resolve_schema(schema_hints, "OutputSchema", "output"),
    )


def _resolve_schema(
    schema_hints: dict[type, dict[str, Any]],
    schema_name: str,
    omit_flag: str | None = None,
) -> type:
    """通过合并模式并可选地尊重 `OmitFromSchema` 注解来解析模式。

    Args:
        schema_hints: 要合并的已解析模式注解
        schema_name: 生成的 `TypedDict` 的名称
        omit_flag: 若指定，将跳过该标志设置的字段（`'input'` 或
            `'output'`）

    Returns:
        合并后的模式作为 `TypedDict`
    """
    all_annotations = {}

    for hints in schema_hints.values():
        for field_name, field_type in hints.items():
            should_omit = False

            if omit_flag:
                metadata = _extract_metadata(field_type)
                for meta in metadata:
                    if isinstance(meta, OmitFromSchema) and getattr(meta, omit_flag) is True:
                        should_omit = True
                        break

            if not should_omit:
                all_annotations[field_name] = field_type

        # `TypedDict` 动态创建一个类，但类型检查器无法推断
        # 运行时结果满足此函数的 `type` 返回契约。
    return cast("type", TypedDict(schema_name, all_annotations))  # type: ignore[operator]


def _extract_metadata(type_: type) -> list[Any]:
    """从字段类型中提取元数据，处理 `Required`/`NotRequired` 和 `Annotated` 包装器。"""  # noqa: E501
        # 处理 Required[Annotated[...]] 或 NotRequired[Annotated[...]]
    if get_origin(type_) in {Required, NotRequired}:
        inner_type = get_args(type_)[0]
        if get_origin(inner_type) is Annotated:
            return list(get_args(inner_type)[1:])

        # 处理直接的 Annotated[...]
    elif get_origin(type_) is Annotated:
        return list(get_args(type_)[1:])

    return []


def _get_can_jump_to(middleware: AgentMiddleware[Any, Any], hook_name: str) -> list[JumpTo]:
    """从同步或异步钩子方法中获取 `can_jump_to` 列表。

    Args:
        middleware: 要检查的中间件实例。
        hook_name: 钩子名称（`'before_model'` 或 `'after_model'`）。

    Returns:
        跳转目标列表，若未配置则为空列表。
    """
        # 获取基类方法用于比较
    base_sync_method = getattr(AgentMiddleware, hook_name, None)
    base_async_method = getattr(AgentMiddleware, f"a{hook_name}", None)

        # 优先尝试同步方法——仅当它覆盖了基类方法时
    sync_method = getattr(middleware.__class__, hook_name, None)
    if (
        sync_method
        and sync_method is not base_sync_method
        and hasattr(sync_method, "__can_jump_to__")
    ):
                # `hasattr` 证明元数据在运行时存在，但不保证其值类型。
        return cast("list[JumpTo]", sync_method.__can_jump_to__)

        # 尝试异步方法 - 仅当它从基类重写时
    async_method = getattr(middleware.__class__, f"a{hook_name}", None)
    if (
        async_method
        and async_method is not base_async_method
        and hasattr(async_method, "__can_jump_to__")
    ):
                # `hasattr` 证明元数据在运行时存在，但不保证其值类型。
        return cast("list[JumpTo]", async_method.__can_jump_to__)

    return []


def _supports_provider_strategy(
    model: str | BaseChatModel, tools: list[BaseTool | dict[str, Any]] | None = None
) -> bool:
    """检查模型是否支持提供方特定的结构化输出。

    Args:
        model: 模型名称字符串或 `BaseChatModel` 实例。
        tools: 可选的工具列表，提供给智能体。

            因为某些模型在与工具调用一起使用时不支持结构化输出。

    Returns:
        `True` 表示模型支持提供方特定的结构化输出，否则为 `False`。
    """
    model_name: str | None = None
    if isinstance(model, str):
        model_name = model
    elif isinstance(model, BaseChatModel):
        model_name = (
            getattr(model, "model_name", None)
            or getattr(model, "model", None)
            or getattr(model, "model_id", "")
        )
        model_profile = model.profile
        if (
            model_profile is not None
            and model_profile.get("structured_output")
                        # 我们对 Gemini 3 系列以下的模型例外处理，这些模型目前不支持
                        # 同时使用工具与结构化输出；3 系列模型支持。
            and not (
                tools
                and isinstance(model_name, str)
                and "gemini" in model_name.lower()
                and "gemini-3" not in model_name.lower()
            )
        ):
            return True

    return (
        any(
            re.search(pattern, model_name.lower())
            for pattern in FALLBACK_MODELS_WITH_STRUCTURED_OUTPUT
        )
        if model_name
        else False
    )


def _is_openai_compatible_model(model: BaseChatModel) -> bool:
    """检查模型是否继承自 `BaseChatOpenAI`。

    Used to redundantly set `strict=True` on tools when `response_format` is
    provided, as older versions of `langchain-openai` do not auto-set it.
    Covers `ChatOpenAI`, `ChatDeepSeek`, `ChatXAI`, etc.

    Args:
        model: 要检查的聊天模型。

    Returns:
        `True` 表示模型继承自 `BaseChatOpenAI`，否则为 `False`。
    """
    try:
        base_chat_openai = importlib.import_module("langchain_openai.chat_models.base")
    except ImportError:
        return False
    return isinstance(model, base_chat_openai.BaseChatOpenAI)


def _handle_structured_output_error(
    exception: Exception,
    response_format: ResponseFormat[Any],
) -> tuple[bool, str]:
    """处理结构化输出错误。

    返回 `(should_retry, retry_tool_message)`。
    """
    if not isinstance(response_format, ToolStrategy):
        return False, ""

    handle_errors = response_format.handle_errors

    if handle_errors is False:
        return False, ""
    if handle_errors is True:
        return True, STRUCTURED_OUTPUT_ERROR_TEMPLATE.format(error=str(exception))
    if isinstance(handle_errors, str):
        return True, handle_errors
    if isinstance(handle_errors, type):
        if issubclass(handle_errors, Exception) and isinstance(exception, handle_errors):
            return True, STRUCTURED_OUTPUT_ERROR_TEMPLATE.format(error=str(exception))
        return False, ""
    if isinstance(handle_errors, tuple):
        if any(isinstance(exception, exc_type) for exc_type in handle_errors):
            return True, STRUCTURED_OUTPUT_ERROR_TEMPLATE.format(error=str(exception))
        return False, ""
    return True, handle_errors(exception)


def _invalid_tool_call_message(tool_call: InvalidToolCall) -> ToolMessage | None:
    tool_call_id = tool_call.get("id")
    if tool_call_id is None:
        return None
    name = tool_call.get("name") or "unknown"
    return ToolMessage(
        content=(
            f"Tool call {name} with id {tool_call_id} could not be executed - "
            "arguments were malformed or truncated."
        ),
        name=name,
        tool_call_id=tool_call_id,
        status="error",
    )


def _patch_invalid_tool_calls(messages: Sequence[AnyMessage]) -> list[AnyMessage]:
    answered_ids = {
        message.tool_call_id for message in messages if isinstance(message, ToolMessage)
    }
    patched_messages: list[AnyMessage] = []
    for message in messages:
        patched_messages.append(message)
        if not isinstance(message, AIMessage):
            continue
        for tool_call in message.invalid_tool_calls:
            if tool_call.get("id") in answered_ids:
                continue
            if tool_message := _invalid_tool_call_message(tool_call):
                patched_messages.append(tool_message)
                answered_ids.add(tool_message.tool_call_id)
    return patched_messages


def _chain_tool_call_wrappers(
    wrappers: Sequence[ToolCallWrapper],
) -> ToolCallWrapper | None:
    """将包装器组合成中间件栈（第一个 = 最外层）。

    Args:
        wrappers: 中间件顺序的包装器。

    Returns:
        组合后的包装器，若为空则为 `None`。

    Example:
        ```python
        wrapper = _chain_tool_call_wrappers([auth, cache, retry])
        # Request flows: auth -> cache -> retry -> tool
        # Response flows: tool -> retry -> cache -> auth
        ```
    """
    if not wrappers:
        return None

    if len(wrappers) == 1:
        return wrappers[0]

    def compose_two(outer: ToolCallWrapper, inner: ToolCallWrapper) -> ToolCallWrapper:
        """组合两个包装器，外层包裹内层。"""

        def composed(
            request: ToolCallRequest,
            execute: Callable[[ToolCallRequest], ToolMessage | Command[Any]],
        ) -> ToolMessage | Command[Any]:
                        # 创建一个可调用对象，该对象使用原始 execute 调用内部函数
            def call_inner(req: ToolCallRequest) -> ToolMessage | Command[Any]:
                return inner(req, execute)

                        # 外部对象可以多次调用 call_inner
            return outer(request, call_inner)

        return composed

        # 将所有包装器串联：first -> second -> ... -> last
    result = wrappers[-1]
    for wrapper in reversed(wrappers[:-1]):
        result = compose_two(wrapper, result)

    return result


def _chain_async_tool_call_wrappers(
    wrappers: Sequence[
        Callable[
            [ToolCallRequest, Callable[[ToolCallRequest], Awaitable[ToolMessage | Command[Any]]]],
            Awaitable[ToolMessage | Command[Any]],
        ]
    ],
) -> (
    Callable[
        [ToolCallRequest, Callable[[ToolCallRequest], Awaitable[ToolMessage | Command[Any]]]],
        Awaitable[ToolMessage | Command[Any]],
    ]
    | None
):
    """将异步包装器组合成中间件栈（第一个 = 最外层）。

    Args:
        wrappers: 中间件顺序的异步包装器。

    Returns:
        组合后的异步包装器，若为空则为 `None`。
    """
    if not wrappers:
        return None

    if len(wrappers) == 1:
        return wrappers[0]

    def compose_two(
        outer: Callable[
            [ToolCallRequest, Callable[[ToolCallRequest], Awaitable[ToolMessage | Command[Any]]]],
            Awaitable[ToolMessage | Command[Any]],
        ],
        inner: Callable[
            [ToolCallRequest, Callable[[ToolCallRequest], Awaitable[ToolMessage | Command[Any]]]],
            Awaitable[ToolMessage | Command[Any]],
        ],
    ) -> Callable[
        [ToolCallRequest, Callable[[ToolCallRequest], Awaitable[ToolMessage | Command[Any]]]],
        Awaitable[ToolMessage | Command[Any]],
    ]:
        """组合两个异步包装器，外层包裹内层。"""

        async def composed(
            request: ToolCallRequest,
            execute: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command[Any]]],
        ) -> ToolMessage | Command[Any]:
                        # 创建一个异步可调用对象，该对象使用原始 execute 调用内部函数
            async def call_inner(req: ToolCallRequest) -> ToolMessage | Command[Any]:
                return await inner(req, execute)

                        # 外部对象可以多次调用 call_inner
            return await outer(request, call_inner)

        return composed

        # 将所有包装器串联：first -> second -> ... -> last
    result = wrappers[-1]
    for wrapper in reversed(wrappers[:-1]):
        result = compose_two(wrapper, result)

    return result


# 无 `response_format`：无结构化输出，因此 `ResponseT` 解析为 `Any`。
@overload
def create_agent(
    model: str | BaseChatModel,
    tools: Sequence[BaseTool | Callable[..., Any] | dict[str, Any]] | None = None,
    *,
    system_prompt: str | SystemMessage | None = None,
    middleware: Sequence[AgentMiddleware[StateT_co, ContextT]] = (),
    response_format: None = None,
    state_schema: None = None,
    context_schema: type[ContextT] | None = None,
    checkpointer: Checkpointer | None = None,
    store: BaseStore | None = None,
    interrupt_before: list[str] | None = None,
    interrupt_after: list[str] | None = None,
    debug: bool = False,
    name: str | None = None,
    cache: BaseCache[Any] | None = None,
    transformers: Sequence[TransformerFactory] | None = None,
) -> CompiledStateGraph[AgentState[Any], ContextT, InputAgentState, OutputAgentState[Any]]: ...


# 原始字典 `response_format`：结构化输出为无类型的 `dict[str, Any]`。
@overload
def create_agent(
    model: str | BaseChatModel,
    tools: Sequence[BaseTool | Callable[..., Any] | dict[str, Any]] | None = None,
    *,
    system_prompt: str | SystemMessage | None = None,
    middleware: Sequence[AgentMiddleware[StateT_co, ContextT]] = (),
    response_format: dict[str, Any],
    state_schema: type[AgentState[dict[str, Any]]] | None = None,
    context_schema: type[ContextT] | None = None,
    checkpointer: Checkpointer | None = None,
    store: BaseStore | None = None,
    interrupt_before: list[str] | None = None,
    interrupt_after: list[str] | None = None,
    debug: bool = False,
    name: str | None = None,
    cache: BaseCache[Any] | None = None,
    transformers: Sequence[TransformerFactory] | None = None,
) -> CompiledStateGraph[
    AgentState[dict[str, Any]], ContextT, InputAgentState, OutputAgentState[dict[str, Any]]
]: ...


# 模式类型 `response_format`：`ResponseT` 从模式/类型推断。
@overload
def create_agent(
    model: str | BaseChatModel,
    tools: Sequence[BaseTool | Callable[..., Any] | dict[str, Any]] | None = None,
    *,
    system_prompt: str | SystemMessage | None = None,
    middleware: Sequence[AgentMiddleware[StateT_co, ContextT]] = (),
    response_format: ResponseFormat[ResponseT] | type[ResponseT] | None = None,
    state_schema: type[AgentState[ResponseT]] | None = None,
    context_schema: type[ContextT] | None = None,
    checkpointer: Checkpointer | None = None,
    store: BaseStore | None = None,
    interrupt_before: list[str] | None = None,
    interrupt_after: list[str] | None = None,
    debug: bool = False,
    name: str | None = None,
    cache: BaseCache[Any] | None = None,
    transformers: Sequence[TransformerFactory] | None = None,
) -> CompiledStateGraph[
    AgentState[ResponseT], ContextT, InputAgentState, OutputAgentState[ResponseT]
]: ...


def create_agent(
    model: str | BaseChatModel,
    tools: Sequence[BaseTool | Callable[..., Any] | dict[str, Any]] | None = None,
    *,
    system_prompt: str | SystemMessage | None = None,
    middleware: Sequence[AgentMiddleware[StateT_co, ContextT]] = (),
    response_format: ResponseFormat[ResponseT] | type[ResponseT] | dict[str, Any] | None = None,
    state_schema: type[AgentState[ResponseT]] | None = None,
    context_schema: type[ContextT] | None = None,
    checkpointer: Checkpointer | None = None,
    store: BaseStore | None = None,
    interrupt_before: list[str] | None = None,
    interrupt_after: list[str] | None = None,
    debug: bool = False,
    name: str | None = None,
    cache: BaseCache[Any] | None = None,
    transformers: Sequence[TransformerFactory] | None = None,
) -> CompiledStateGraph[
    AgentState[ResponseT], ContextT, InputAgentState, OutputAgentState[ResponseT]
]:
    """创建一个智能体图，循环调用工具直到满足停止条件。

    关于使用 `create_agent` 的更多详情，
    请参阅 [智能体](https://docs.langchain.com/oss/python/langchain/agents) 文档。

    Args:
        model: 智能体使用的语言模型。

            可以是字符串标识符（例如 `"openai:gpt-5.5"`）或直接的聊天模型
            实例（例如 [`ChatOpenAI`][langchain_openai.ChatOpenAI] 或其他
            [LangChain 聊天模型](https://docs.langchain.com/oss/python/integrations/chat))。

            支持的模型字符串完整列表，请参见
            [`init_chat_model`][langchain.chat_models.init_chat_model(model_provider)]。

            !!! tip ""

                See the [Models](https://docs.langchain.com/oss/python/langchain/models)
                docs for more information.
        tools: A list of tools, `dict`, or `Callable`.

            如果为 `None` 或空列表，智能体将仅由一个模型节点组成，不包含工具调用循环。


            !!! tip ""

                See the [Tools](https://docs.langchain.com/oss/python/langchain/tools)
                docs for more information.
        system_prompt: An optional system prompt for the LLM.

            可以是 `str`（将被转换为 `SystemMessage`）或直接传入 `SystemMessage`
            实例。系统消息会在调用模型时添加到消息列表的开头。 middleware:
            要应用于智能体的一系列中间件实例。

            中间件可在不同阶段拦截并修改智能体行为。

            !!! tip ""

                See the [Middleware](https://docs.langchain.com/oss/python/langchain/middleware)
                docs for more information.
        response_format: An optional configuration for structured responses.

            可以是 `ToolStrategy`、`ProviderStrategy` 或 Pydantic 模型类。
            state_schema: 可选的 `TypedDict` 模式，用于扩展 `AgentState`。

            如果提供，该模式将作为合并中间件状态模式的基础，取代
            `AgentState`。这允许用户添加自定义状态字段，而无需创建自定义中间件。
            通常建议通过中间件使用 `state_schema` 扩展，以确保相关扩展与对应钩子 /
            工具的作用范围保持一致。 context_schema: 可选的运行时上下文模式。
            checkpointer: 可选的检查点保存对象。

            用于持久化图的状态（例如聊天记忆）以供单个线程（例如单次对话）使用。
            store: 可选的数据存储对象。

            用于跨多个线程（例如多个对话 / 用户）持久化数据。 interrupt_before:
            可选的在执行前中断的节点名称列表。

                See the [Structured output](https://docs.langchain.com/oss/python/langchain/structured-output)
                docs for more information.
        state_schema: An optional `TypedDict` schema that extends `AgentState`.

            When provided, this schema is used instead of `AgentState` as the base
            schema for merging with middleware state schemas. This allows users to
            add custom state fields without needing to create custom middleware.

            Generally, it's recommended to use `state_schema` extensions via middleware
            to keep relevant extensions scoped to corresponding hooks / tools.
        context_schema: An optional schema for runtime context.
        checkpointer: An optional checkpoint saver object.

            Used for persisting the state of the graph (e.g., as chat memory) for a
            single thread (e.g., a single conversation).
        store: An optional store object.

            Used for persisting data across multiple threads (e.g., multiple
            conversations / users).
        interrupt_before: An optional list of node names to interrupt before.

            Useful if you want to add a user confirmation or other interrupt
            before taking an action.
        interrupt_after: An optional list of node names to interrupt after.

            Useful if you want to return directly or run additional processing
            on an output.
        debug: Whether to enable verbose logging for graph execution.

            When enabled, prints detailed information about each node execution, state
            updates, and transitions during agent runtime. Useful for debugging
            middleware behavior and understanding agent execution flow.
        name: An optional name for the `CompiledStateGraph`.

            This name will be automatically used when adding the agent graph to
            another graph as a subgraph node - particularly useful for building
            multi-agent systems.
        cache: An optional `BaseCache` instance to enable caching of graph execution.
        transformers: Optional sequence of scope-aware `StreamTransformer`
            factories to register on the compiled graph in addition to
            the agent defaults. Each factory is invoked as `factory(scope)`
            so every invocation receives a fresh instance. The final order
            on the compiled graph is: `ToolCallTransformer`, then any
            factories declared by middleware via
            `AgentMiddleware.transformers`, then any factories supplied here.

    Returns:
        可用于聊天交互的已编译 `StateGraph`。

    Raises:
        AssertionError: 如果提供了重复的中间件实例。

    The agent node calls the language model with the messages list (after applying
    the system prompt). If the resulting [`AIMessage`][langchain.messages.AIMessage]
    contains `tool_calls`, the graph will then call the tools. The tools node executes
    the tools and adds the responses to the messages list as
    [`ToolMessage`][langchain.messages.ToolMessage] objects. The agent node then calls
    the language model again. The process repeats until no more `tool_calls` are present
    in the response. The agent then returns the full list of messages.

    Example:
        ```python
        from langchain.agents import create_agent


        def check_weather(location: str) -> str:
            '''Return the weather forecast for the specified location.'''
            return f"It's always sunny in {location}"


        graph = create_agent(
            model="anthropic:claude-sonnet-4-5-20250929",
            tools=[check_weather],
            system_prompt="You are a helpful assistant",
        )
        inputs = {"messages": [{"role": "user", "content": "what is the weather in sf"}]}
        for chunk in graph.stream(inputs, stream_mode="updates"):
            print(chunk)
        ```
    """
        # 初始化聊天模型
    if isinstance(model, str):
        model = init_chat_model(model)

        # 如需，将 system_prompt 转换为 SystemMessage
    system_message: SystemMessage | None = None
    if system_prompt is not None:
        if isinstance(system_prompt, SystemMessage):
            system_message = system_prompt
        else:
            system_message = SystemMessage(content=system_prompt)

        # 处理 tools 为 None 或空的情况
    if tools is None:
        tools = []

        # 转换 response format 并设置结构化输出工具
        # 原始模式被包裹在 AutoStrategy 中，以保留自动检测意图。
        # AutoStrategy 提前转换为 ToolStrategy，以便在智能体创建时计算工具，
        # 但之后可能根据模型能力替换为 ProviderStrategy。
    initial_response_format: ToolStrategy[Any] | ProviderStrategy[Any] | AutoStrategy[Any] | None
    if response_format is None:
        initial_response_format = None
    elif isinstance(response_format, (ToolStrategy, ProviderStrategy, AutoStrategy)):
                # 显式 Tool/Provider 策略，或用于后期能力检测的 AutoStrategy
        initial_response_format = response_format
    else:
                # 原始模式 - 包裹在 AutoStrategy 中以启用自动检测
        initial_response_format = AutoStrategy(schema=response_format)

        # 对于 AutoStrategy，转换为 ToolStrategy 以提前设置工具
        # (可能根据模型能力 later 替换为 ProviderStrategy)
    tool_strategy_for_setup: ToolStrategy[Any] | None = None
    if isinstance(initial_response_format, AutoStrategy):
        tool_strategy_for_setup = ToolStrategy(schema=initial_response_format.schema)
    elif isinstance(initial_response_format, ToolStrategy):
        tool_strategy_for_setup = initial_response_format

    structured_output_tools: dict[str, OutputToolBinding[Any]] = {}
    if tool_strategy_for_setup:
        for response_schema in tool_strategy_for_setup.schema_specs:
            structured_tool_info = OutputToolBinding.from_schema_spec(response_schema)
            structured_output_tools[structured_tool_info.tool.name] = structured_tool_info
    middleware_tools = [t for m in middleware for t in getattr(m, "tools", [])]

        # 收集具有 wrap_tool_call 或 awrap_tool_call 钩子的中间件
        # 包含任一实现的中间件，以确保在中间件不支持执行路径时抛出 NotImplementedError
        # 当中间件不支持执行路径时抛出 NotImplementedError
    middleware_w_wrap_tool_call = [
        m
        for m in middleware
        if m.__class__.wrap_tool_call is not AgentMiddleware.wrap_tool_call
        or m.__class__.awrap_tool_call is not AgentMiddleware.awrap_tool_call
    ]

        # 将所有 wrap_tool_call 处理器串联成单一组合处理器
    wrap_tool_call_wrapper = None
    if middleware_w_wrap_tool_call:
        wrappers = [
            traceable(name=f"{m.name}.wrap_tool_call", **_wrap_trace_kwargs(m))(m.wrap_tool_call)
            for m in middleware_w_wrap_tool_call
        ]
        wrap_tool_call_wrapper = _chain_tool_call_wrappers(wrappers)

        # 收集具有 awrap_tool_call 或 wrap_tool_call 钩子的中间件
        # 包含任一实现的中间件，以确保在中间件不支持执行路径时抛出 NotImplementedError
        # 当中间件不支持执行路径时抛出 NotImplementedError
    middleware_w_awrap_tool_call = [
        m
        for m in middleware
        if m.__class__.awrap_tool_call is not AgentMiddleware.awrap_tool_call
        or m.__class__.wrap_tool_call is not AgentMiddleware.wrap_tool_call
    ]

        # 将所有 awrap_tool_call 处理器串联成单一组合异步处理器
    awrap_tool_call_wrapper = None
    if middleware_w_awrap_tool_call:
        async_wrappers = [
            traceable(name=f"{m.name}.awrap_tool_call", **_wrap_trace_kwargs(m))(m.awrap_tool_call)
            for m in middleware_w_awrap_tool_call
        ]
        awrap_tool_call_wrapper = _chain_async_tool_call_wrappers(async_wrappers)

        # 设置工具
    tool_node: ToolNode | None = None
        # 提取内置提供者工具（字典格式）和常规工具（BaseTool/可调用对象）
    built_in_tools = [t for t in tools if isinstance(t, dict)]
    regular_tools = [t for t in tools if not isinstance(t, dict)]

        # 需要客户端执行的工具（必须位于 ToolNode 中）
    available_tools = middleware_tools + regular_tools

        # 如果存在客户端工具，或中间件定义了 wrap_tool_call，则创建 ToolNode
        # (可能处理动态注册的工具)
    tool_node = (
        ToolNode(
            tools=available_tools,
            wrap_tool_call=wrap_tool_call_wrapper,
            awrap_tool_call=awrap_tool_call_wrapper,
        )
        if available_tools or wrap_tool_call_wrapper or awrap_tool_call_wrapper
        else None
    )

        # ModelRequest 初始化的默认工具
        # 使用 ToolNode 转换的 BaseTool 实例（而非原始可调用对象）
        # 包含内置工具和转换后的工具（可通过中间件动态更改）
        # 结构化工具不包含在内 - 它们基于 response_format 动态添加
    if tool_node:
        default_tools = list(tool_node.tools_by_name.values()) + built_in_tools
    else:
        default_tools = list(built_in_tools)

        # 验证中间件
    if len({m.name for m in middleware}) != len(middleware):
        msg = "Please remove duplicate middleware instances."
        raise AssertionError(msg)
    middleware_w_before_agent = [
        m
        for m in middleware
        if m.__class__.before_agent is not AgentMiddleware.before_agent
        or m.__class__.abefore_agent is not AgentMiddleware.abefore_agent
    ]
    middleware_w_before_model = [
        m
        for m in middleware
        if m.__class__.before_model is not AgentMiddleware.before_model
        or m.__class__.abefore_model is not AgentMiddleware.abefore_model
    ]
    middleware_w_after_model = [
        m
        for m in middleware
        if m.__class__.after_model is not AgentMiddleware.after_model
        or m.__class__.aafter_model is not AgentMiddleware.aafter_model
    ]
    middleware_w_after_agent = [
        m
        for m in middleware
        if m.__class__.after_agent is not AgentMiddleware.after_agent
        or m.__class__.aafter_agent is not AgentMiddleware.aafter_agent
    ]
        # 收集具有 wrap_model_call 或 awrap_model_call 钩子的中间件
        # 包含任一实现的中间件，以确保在中间件不支持执行路径时抛出 NotImplementedError
        # 当中间件不支持执行路径时抛出 NotImplementedError
    middleware_w_wrap_model_call = [
        m
        for m in middleware
        if m.__class__.wrap_model_call is not AgentMiddleware.wrap_model_call
        or m.__class__.awrap_model_call is not AgentMiddleware.awrap_model_call
    ]
        # 收集具有 awrap_model_call 或 wrap_model_call 钩子的中间件
        # 包含中间件的任一实现，以确保当中间件不支持执行路径时抛出 NotImplementedError
        # 当中间件不支持执行路径时，确保抛出 NotImplementedError
    middleware_w_awrap_model_call = [
        m
        for m in middleware
        if m.__class__.awrap_model_call is not AgentMiddleware.awrap_model_call
        or m.__class__.wrap_model_call is not AgentMiddleware.wrap_model_call
    ]

        # 将 wrap_model_call 处理函数组合成一个单一的中间件堆栈（同步）
    wrap_model_call_handler = None
    if middleware_w_wrap_model_call:
        sync_handlers = [
            traceable(name=f"{m.name}.wrap_model_call", **_wrap_trace_kwargs(m))(m.wrap_model_call)
            for m in middleware_w_wrap_model_call
        ]
        wrap_model_call_handler = _chain_model_call_handlers(sync_handlers)

        # 将 awrap_model_call 处理函数组合成一个单一的中间件堆栈（异步）
    awrap_model_call_handler = None
    if middleware_w_awrap_model_call:
        async_handlers = [
            traceable(name=f"{m.name}.awrap_model_call", **_wrap_trace_kwargs(m))(
                m.awrap_model_call
            )
            for m in middleware_w_awrap_model_call
        ]
        awrap_model_call_handler = _chain_async_model_call_handlers(async_handlers)

    base_state = state_schema if state_schema is not None else AgentState
        # 构建有序列表：首先为中间件模式（按注册顺序），最后为
    # base_state，以便在字段冲突时获胜。这使得调用方的显式 state_schema
    # 覆盖中间件注解——例如，传递一个 DeltaChannel 注解的模式会覆盖 AgentState 中的
    # BinaryOperatorAggregate，而无需编译后修补。
        # 构建有序列表：首先为中间件模式（按注册顺序），最后为
    # base_state，以便在字段冲突时获胜。这使得调用方的显式 state_schema
    # 覆盖中间件注解——例如，传递一个 DeltaChannel 注解的模式会覆盖 AgentState 中的
    # BinaryOperatorAggregate，而无需编译后修补。
        # 使得调用方的显式 state_schema 覆盖中间件注解——例如，传递一个 DeltaChannel
    # 注解的模式会覆盖 AgentState 中的 BinaryOperatorAggregate，而无需编译后修补。
        # 例如，传递一个 DeltaChannel 注解的模式会覆盖 AgentState 中的
    # BinaryOperatorAggregate，而无需编译后修补。
        # 例如，传递一个 DeltaChannel 注解的模式会覆盖 AgentState 中的
    # BinaryOperatorAggregate，而无需编译后修补。
    state_schemas: list[type] = [*(m.state_schema for m in middleware), base_state]

    resolved_state_schema, input_schema, output_schema = _resolve_schemas(state_schemas)

        # 创建图并添加节点
    graph: StateGraph[
        AgentState[ResponseT], ContextT, InputAgentState, OutputAgentState[ResponseT]
    ] = StateGraph(
        state_schema=resolved_state_schema,
        input_schema=input_schema,
        output_schema=output_schema,
        context_schema=context_schema,
    )

    def _handle_model_output(
        output: AIMessage, effective_response_format: ResponseFormat[Any] | None
    ) -> dict[str, Any]:
        """处理模型输出，包括结构化响应。

        Args:
            output: The AI message output from the model.
            effective_response_format: The actual strategy used (may differ from initial
                if auto-detected).
        """
                # 使用提供方策略处理结构化输出
        if isinstance(effective_response_format, ProviderStrategy):
            if not output.tool_calls:
                provider_strategy_binding = ProviderStrategyBinding.from_schema_spec(
                    effective_response_format.schema_spec
                )
                try:
                    structured_response = provider_strategy_binding.parse(output)
                except Exception as exc:
                    schema_name = getattr(
                        effective_response_format.schema_spec.schema, "__name__", "response_format"
                    )
                    validation_error = StructuredOutputValidationError(schema_name, exc, output)
                    raise validation_error from exc
                else:
                    return {"messages": [output], "structured_response": structured_response}
            return {"messages": [output]}

                # 使用工具策略处理结构化输出
        if (
            isinstance(effective_response_format, ToolStrategy)
            and isinstance(output, AIMessage)
            and output.tool_calls
        ):
            structured_tool_calls = [
                tc for tc in output.tool_calls if tc["name"] in structured_output_tools
            ]

            if structured_tool_calls:
                exception: StructuredOutputError | None = None
                if len(structured_tool_calls) > 1:
                                        # 处理多个结构化输出错误
                    tool_names = [tc["name"] for tc in structured_tool_calls]
                    exception = MultipleStructuredOutputsError(tool_names, output)
                    should_retry, error_message = _handle_structured_output_error(
                        exception, effective_response_format
                    )
                    if not should_retry:
                        raise exception

                                        # 添加错误消息和重试机制
                    tool_messages = [
                        ToolMessage(
                            content=error_message,
                            tool_call_id=tc["id"],
                            name=tc["name"],
                        )
                        for tc in structured_tool_calls
                    ]
                    return {"messages": [output, *tool_messages]}

                                # 处理单个结构化输出
                tool_call = structured_tool_calls[0]
                try:
                    structured_tool_binding = structured_output_tools[tool_call["name"]]
                    structured_response = structured_tool_binding.parse(tool_call["args"])

                    tool_message_content = (
                        effective_response_format.tool_message_content
                        or f"Returning structured response: {structured_response}"
                    )

                    return {
                        "messages": [
                            output,
                            ToolMessage(
                                content=tool_message_content,
                                tool_call_id=tool_call["id"],
                                name=tool_call["name"],
                            ),
                        ],
                        "structured_response": structured_response,
                    }
                except Exception as exc:
                    exception = StructuredOutputValidationError(tool_call["name"], exc, output)
                    should_retry, error_message = _handle_structured_output_error(
                        exception, effective_response_format
                    )
                    if not should_retry:
                        raise exception from exc

                    return {
                        "messages": [
                            output,
                            ToolMessage(
                                content=error_message,
                                tool_call_id=tool_call["id"],
                                name=tool_call["name"],
                            ),
                        ],
                    }

        return {"messages": [output]}

    def _get_bound_model(
        request: ModelRequest[ContextT],
    ) -> tuple[Runnable[Any, Any], ResponseFormat[Any] | None]:
        """获取带有适当工具绑定的模型。

        根据模型能力，必要时执行自动检测策略。

        Args:
            request: 包含模型、工具和响应格式的模型请求。

        Returns:
            Tuple of `(bound_model, effective_response_format)` where
            `effective_response_format` is the actual strategy used (may differ from
            initial if auto-detected).

        Raises:
            ValueError: 若中间件返回了未知的客户端工具名称。
            ValueError: 若 `ToolStrategy` 指定了未提前声明的工具。
        """
                # 仅验证需要存在于 tool_node 中的客户端侧工具
                # 当定义了 wrap_tool_call 时跳过验证，因为中间件可能处理通过 wrap_model_call
        # 在运行时动态添加的工具
                # 当定义了 wrap_tool_call 时跳过验证，因为中间件可能处理通过 wrap_model_call
        # 在运行时动态添加的工具
        has_wrap_tool_call = wrap_tool_call_wrapper or awrap_tool_call_wrapper

                # 从 ToolNode 构建可用客户端侧工具的映射
                # （已将可调用对象转换完毕）
        available_tools_by_name = {}
        if tool_node:
            available_tools_by_name = tool_node.tools_by_name.copy()

                # 检查是否有请求的工具是未知的客户端侧工具
                # 仅在未定义 wrap_tool_call 时进行验证（无动态工具处理）
        if not has_wrap_tool_call:
            unknown_tool_names = []
            for t in request.tools:
                                # 仅验证 BaseTool 实例（跳过内置字典工具）
                if isinstance(t, dict):
                    continue
                if isinstance(t, BaseTool) and t.name not in available_tools_by_name:
                    unknown_tool_names.append(t.name)

            if unknown_tool_names:
                available_tool_names = sorted(available_tools_by_name.keys())
                msg = DYNAMIC_TOOL_ERROR_TEMPLATE.format(
                    unknown_tool_names=unknown_tool_names,
                    available_tool_names=available_tool_names,
                )
                raise ValueError(msg)

                # 将原始模式规范化为 AutoStrategy
                # （处理中间件通过原始 Pydantic 类覆盖的情况）
        response_format: ResponseFormat[Any] | Any | None = request.response_format
        if response_format is not None and not isinstance(
            response_format, (AutoStrategy, ToolStrategy, ProviderStrategy)
        ):
            response_format = AutoStrategy(schema=response_format)

                # 确定有效响应格式（如需则自动检测）
        effective_response_format: ResponseFormat[Any] | None
        if isinstance(response_format, AutoStrategy):
                        # 用户通过 AutoStrategy 提供原始模式——根据模型自动检测最佳策略
            if _supports_provider_strategy(request.model, tools=request.tools):
                                # 模型支持提供方策略——使用该策略
                effective_response_format = ProviderStrategy(schema=response_format.schema)
            elif response_format is initial_response_format and tool_strategy_for_setup is not None:
                                # 模型不支持提供方策略——使用 ToolStrategy
                                # 尽可能复用设置阶段的策略以保留工具名称
                effective_response_format = tool_strategy_for_setup
            else:
                effective_response_format = ToolStrategy(schema=response_format.schema)
        else:
                        # 用户明确指定了策略——保留该策略
            effective_response_format = response_format

                # 构建最终工具列表，包括结构化输出工具
                # request.tools 现在仅包含 BaseTool 实例（由可调用对象转换而来）
                # 以及字典（内置工具）
        final_tools = list(request.tools)
        if isinstance(effective_response_format, ToolStrategy):
                        # 将结构化输出工具添加到最终工具列表中，仅限于（可能被中间件缩小的）响应格式中存在的模式。
                        # 将结构化输出工具添加到最终工具列表中，仅限于（可能被中间件缩小的）响应格式中存在的模式。
                        # 这确保了通过 `request.override()` 将联合 `ToolStrategy`
            # 缩小为子集的中间件，实际上限制了模型可选择的结构化输出工具范围。
                        # 这确保了通过 `request.override()` 将联合 `ToolStrategy`
            # 缩小为子集的中间件，实际上限制了模型可选择的结构化输出工具范围。
                        # 这确保了通过 `request.override()` 将联合 `ToolStrategy`
            # 缩小为子集的中间件，实际上限制了模型可选择的结构化输出工具范围。
            narrowed_tool_names = {spec.name for spec in effective_response_format.schema_specs}
            structured_tools = [
                info.tool
                for name, info in structured_output_tools.items()
                if name in narrowed_tool_names
            ]
            final_tools.extend(structured_tools)

                # 基于有效响应格式绑定模型
        if isinstance(effective_response_format, ProviderStrategy):
                        # （向后兼容）使用 OpenAI 格式的结构化输出
                        # 对 OpenAI 兼容模型冗余设置 strict=True，因为旧版本的 langchain-openai
            # 不会在 bind_tools 中自动设置它。
                        # 对 OpenAI 兼容模型冗余设置 strict=True，因为旧版本的 langchain-openai
            # 不会在 bind_tools 中自动设置它。
            kwargs = effective_response_format.to_model_kwargs()
            bind_kwargs: dict[str, Any] = {**kwargs, **request.model_settings}
            if _is_openai_compatible_model(request.model) and not getattr(
                request.model, "use_responses_api", False
            ):
                bind_kwargs["strict"] = True
            return (
                request.model.bind_tools(final_tools, **bind_kwargs),
                effective_response_format,
            )

        if isinstance(effective_response_format, ToolStrategy):
                        # 当前实现要求用于结构化输出的工具必须在创建智能体时作为响应格式的一部分提前声明。中间件允许在使用
            # ToolStrategy
            # 时将响应格式更改为其原始结构化工具的子集，但不允许添加未提前声明的新结构化工具。
                        # 当前实现要求用于结构化输出的工具必须在创建智能体时作为响应格式的一部分提前声明。中间件允许在使用
            # ToolStrategy
            # 时将响应格式更改为其原始结构化工具的子集，但不允许添加未提前声明的新结构化工具。
                        # 当前实现要求用于结构化输出的工具必须在创建智能体时作为响应格式的一部分提前声明。中间件允许在使用
            # ToolStrategy
            # 时将响应格式更改为其原始结构化工具的子集，但不允许添加未提前声明的新结构化工具。
                        # 当前实现要求用于结构化输出的工具必须在创建智能体时作为响应格式的一部分提前声明。中间件允许在使用
            # ToolStrategy
            # 时将响应格式更改为其原始结构化工具的子集，但不允许添加未提前声明的新结构化工具。
                        # 当前实现要求用于结构化输出的工具必须在创建智能体时作为响应格式的一部分提前声明。中间件允许在使用
            # ToolStrategy
            # 时将响应格式更改为其原始结构化工具的子集，但不允许添加未提前声明的新结构化工具。
                        # 计算输出绑定
            for tc in effective_response_format.schema_specs:
                if tc.name not in structured_output_tools:
                    msg = (
                        f"ToolStrategy specifies tool '{tc.name}' "
                        "which wasn't declared in the original "
                        "response format when creating the agent."
                    )
                    raise ValueError(msg)

                        # 如果有结构化输出工具，则强制使用工具
            tool_choice = "any" if structured_output_tools else request.tool_choice
            return (
                request.model.bind_tools(
                    final_tools, tool_choice=tool_choice, **request.model_settings
                ),
                effective_response_format,
            )

                # 无结构化输出——标准模型绑定
        if final_tools:
            return (
                request.model.bind_tools(
                    final_tools, tool_choice=request.tool_choice, **request.model_settings
                ),
                None,
            )
        return request.model.bind(**request.model_settings), None

    def _execute_model_sync(request: ModelRequest[ContextT]) -> ModelResponse:
        """执行模型并返回响应。

        这是被 `wrap_model_call` 处理器封装的核心模型执行逻辑。

        在模型调用过程中发生异常时抛出。
        """
                # 获取绑定后的模型（如需则自动检测）
        model_, effective_response_format = _get_bound_model(request)
        messages = request.messages
        if request.system_message:
            messages = [request.system_message, *messages]

        output = model_.invoke(messages)
        if name:
            output.name = name

                # 处理模型输出以获取消息和结构化响应
        handled_output = _handle_model_output(output, effective_response_format)
        messages_list = handled_output["messages"]
        structured_response = handled_output.get("structured_response")

        return ModelResponse(
            result=messages_list,
            structured_response=structured_response,
        )

    def model_node(state: AgentState[Any], runtime: Runtime[ContextT]) -> list[Command[Any]]:
        """同步模型请求处理器，支持顺序中间件处理。"""
        messages = _patch_invalid_tool_calls(state["messages"])
        request = ModelRequest(
            model=model,
            tools=default_tools,
            system_message=system_message,
            response_format=initial_response_format,
            messages=messages,
            tool_choice=None,
            state=state,
            runtime=runtime,
        )

        has_structured_output = initial_response_format is not None
        if wrap_model_call_handler is None:
            model_response = _execute_model_sync(request)
            return _build_commands(
                model_response,
                has_structured_output=has_structured_output,
                repaired_messages=messages if messages != state["messages"] else None,
            )

        result = wrap_model_call_handler(request, _execute_model_sync)
        return _build_commands(
            result.model_response,
            result.commands,
            has_structured_output=has_structured_output,
            repaired_messages=messages if messages != state["messages"] else None,
        )

    async def _execute_model_async(request: ModelRequest[ContextT]) -> ModelResponse:
        """异步执行模型并返回响应。

        This is the core async model execution logic wrapped by `wrap_model_call`
        handlers.

        在模型调用过程中发生异常时抛出。
        """
                # 获取绑定的模型（如需则自动检测）
        model_, effective_response_format = _get_bound_model(request)
        messages = request.messages
        if request.system_message:
            messages = [request.system_message, *messages]

        output = await model_.ainvoke(messages)
        if name:
            output.name = name

                # 处理模型输出以获取消息和结构化响应
        handled_output = _handle_model_output(output, effective_response_format)
        messages_list = handled_output["messages"]
        structured_response = handled_output.get("structured_response")

        return ModelResponse(
            result=messages_list,
            structured_response=structured_response,
        )

    async def amodel_node(state: AgentState[Any], runtime: Runtime[ContextT]) -> list[Command[Any]]:
        """支持顺序中间件处理的异步模型请求处理器。"""
        messages = _patch_invalid_tool_calls(state["messages"])
        request = ModelRequest(
            model=model,
            tools=default_tools,
            system_message=system_message,
            response_format=initial_response_format,
            messages=messages,
            tool_choice=None,
            state=state,
            runtime=runtime,
        )

        has_structured_output = initial_response_format is not None
        if awrap_model_call_handler is None:
            model_response = await _execute_model_async(request)
            return _build_commands(
                model_response,
                has_structured_output=has_structured_output,
                repaired_messages=messages if messages != state["messages"] else None,
            )

        result = await awrap_model_call_handler(request, _execute_model_async)
        return _build_commands(
            result.model_response,
            result.commands,
            has_structured_output=has_structured_output,
            repaired_messages=messages if messages != state["messages"] else None,
        )

        # 根据模型能力选择同步或异步
    graph.add_node("model", RunnableCallable(model_node, amodel_node, trace=False))

        # 仅在存在工具时添加工具节点
    if tool_node is not None:
        graph.add_node("tools", tool_node)

        # 添加中间件节点
    for m in middleware:
        if (
            m.__class__.before_agent is not AgentMiddleware.before_agent
            or m.__class__.abefore_agent is not AgentMiddleware.abefore_agent
        ):
                        # 使用 RunnableCallable 以同时支持同步和异步
                        # 若未重写，同步模式传入 None 以避免签名冲突
            sync_before_agent = (
                m.before_agent
                if m.__class__.before_agent is not AgentMiddleware.before_agent
                else None
            )
            async_before_agent = (
                m.abefore_agent
                if m.__class__.abefore_agent is not AgentMiddleware.abefore_agent
                else None
            )
            before_agent_node = RunnableCallable(sync_before_agent, async_before_agent, trace=False)
            graph.add_node(
                f"{m.name}.before_agent",
                before_agent_node,
                input_schema=resolved_state_schema,
                trace_policy=_node_trace_policy(m.trace_policy),
            )

        if (
            m.__class__.before_model is not AgentMiddleware.before_model
            or m.__class__.abefore_model is not AgentMiddleware.abefore_model
        ):
                        # 使用 RunnableCallable 以同时支持同步和异步
                        # 若未重写，同步模式传入 None 以避免签名冲突
            sync_before = (
                m.before_model
                if m.__class__.before_model is not AgentMiddleware.before_model
                else None
            )
            async_before = (
                m.abefore_model
                if m.__class__.abefore_model is not AgentMiddleware.abefore_model
                else None
            )
            before_node = RunnableCallable(sync_before, async_before, trace=False)
            graph.add_node(
                f"{m.name}.before_model",
                before_node,
                input_schema=resolved_state_schema,
                trace_policy=_node_trace_policy(m.trace_policy),
            )

        if (
            m.__class__.after_model is not AgentMiddleware.after_model
            or m.__class__.aafter_model is not AgentMiddleware.aafter_model
        ):
                        # 使用 RunnableCallable 以同时支持同步和异步
                        # 若未重写，同步模式传入 None 以避免签名冲突
            sync_after = (
                m.after_model
                if m.__class__.after_model is not AgentMiddleware.after_model
                else None
            )
            async_after = (
                m.aafter_model
                if m.__class__.aafter_model is not AgentMiddleware.aafter_model
                else None
            )
            after_node = RunnableCallable(sync_after, async_after, trace=False)
            graph.add_node(
                f"{m.name}.after_model",
                after_node,
                input_schema=resolved_state_schema,
                trace_policy=_node_trace_policy(m.trace_policy),
            )

        if (
            m.__class__.after_agent is not AgentMiddleware.after_agent
            or m.__class__.aafter_agent is not AgentMiddleware.aafter_agent
        ):
                        # 使用 RunnableCallable 以同时支持同步和异步
                        # 若未重写，同步模式传入 None 以避免签名冲突
            sync_after_agent = (
                m.after_agent
                if m.__class__.after_agent is not AgentMiddleware.after_agent
                else None
            )
            async_after_agent = (
                m.aafter_agent
                if m.__class__.aafter_agent is not AgentMiddleware.aafter_agent
                else None
            )
            after_agent_node = RunnableCallable(sync_after_agent, async_after_agent, trace=False)
            graph.add_node(
                f"{m.name}.after_agent",
                after_agent_node,
                input_schema=resolved_state_schema,
                trace_policy=_node_trace_policy(m.trace_policy),
            )

        # 确定入口节点（启动时运行一次）：before_agent -> before_model -> model
    if middleware_w_before_agent:
        entry_node = f"{middleware_w_before_agent[0].name}.before_agent"
    elif middleware_w_before_model:
        entry_node = f"{middleware_w_before_model[0].name}.before_model"
    else:
        entry_node = "model"

        # 确定循环入口节点（智能体循环开始处，不包含 before_agent）
        # 这是工具将在下一轮迭代中返回的位置
    if middleware_w_before_model:
        loop_entry_node = f"{middleware_w_before_model[0].name}.before_model"
    else:
        loop_entry_node = "model"

        # 确定循环退出节点（每次迭代结束处，可运行多次）
        # 这是 after_model 或 model，但不是 after_agent
    if middleware_w_after_model:
        loop_exit_node = f"{middleware_w_after_model[0].name}.after_model"
    else:
        loop_exit_node = "model"

        # 确定退出节点（结束时运行一次）：after_agent 或 END
    if middleware_w_after_agent:
        exit_node = f"{middleware_w_after_agent[-1].name}.after_agent"
    else:
        exit_node = END

    graph.add_edge(START, entry_node)
        # 仅在存在工具时添加条件边
    if tool_node is not None:
                # 仅当任一工具的 return_direct=True 时，才将 exit_node 包含在目标中
                # 或当存在结构化输出工具时
        tools_to_model_destinations = [loop_entry_node]
        if (
            any(tool.return_direct for tool in tool_node.tools_by_name.values())
            or structured_output_tools
        ):
            tools_to_model_destinations.append(exit_node)

        graph.add_conditional_edges(
            "tools",
            RunnableCallable(
                _make_tools_to_model_edge(
                    tool_node=tool_node,
                    model_destination=loop_entry_node,
                    structured_output_tools=structured_output_tools,
                    end_destination=exit_node,
                ),
                trace=False,
            ),
            tools_to_model_destinations,
        )

                # 当中间件可注入合成工具时，包含 loop_entry_node
                # messages，或当结构化输出或 after-model 钩子可重新路由至该节点时
        model_to_tools_destinations = ["tools", exit_node]
        if response_format or loop_exit_node != "model" or middleware_w_wrap_model_call:
            model_to_tools_destinations.append(loop_entry_node)

        graph.add_conditional_edges(
            loop_exit_node,
            RunnableCallable(
                _make_model_to_tools_edge(
                    model_destination=loop_entry_node,
                    structured_output_tools=structured_output_tools,
                    end_destination=exit_node,
                ),
                trace=False,
            ),
            model_to_tools_destinations,
        )
    elif len(structured_output_tools) > 0:
        graph.add_conditional_edges(
            loop_exit_node,
            RunnableCallable(
                _make_model_to_model_edge(
                    model_destination=loop_entry_node,
                    end_destination=exit_node,
                ),
                trace=False,
            ),
            [loop_entry_node, exit_node],
        )
    elif loop_exit_node == "model":
                # 若无工具且无 after_model，则直接跳转至 exit_node
        graph.add_edge(loop_exit_node, exit_node)
        # 无工具但存在 after_model - 将 after_model 连接到 exit_node
    else:
        _add_middleware_edge(
            graph,
            name=f"{middleware_w_after_model[0].name}.after_model",
            default_destination=exit_node,
            model_destination=loop_entry_node,
            end_destination=exit_node,
            can_jump_to=_get_can_jump_to(middleware_w_after_model[0], "after_model"),
        )

        # 添加 before_agent 中间件边
    if middleware_w_before_agent:
        for m1, m2 in itertools.pairwise(middleware_w_before_agent):
            _add_middleware_edge(
                graph,
                name=f"{m1.name}.before_agent",
                default_destination=f"{m2.name}.before_agent",
                model_destination=loop_entry_node,
                end_destination=exit_node,
                can_jump_to=_get_can_jump_to(m1, "before_agent"),
            )
                # 将最后一个 before_agent 连接到 loop_entry_node（before_model 或 model）
        _add_middleware_edge(
            graph,
            name=f"{middleware_w_before_agent[-1].name}.before_agent",
            default_destination=loop_entry_node,
            model_destination=loop_entry_node,
            end_destination=exit_node,
            can_jump_to=_get_can_jump_to(middleware_w_before_agent[-1], "before_agent"),
        )

        # 添加 before_model 中间件边
    if middleware_w_before_model:
        for m1, m2 in itertools.pairwise(middleware_w_before_model):
            _add_middleware_edge(
                graph,
                name=f"{m1.name}.before_model",
                default_destination=f"{m2.name}.before_model",
                model_destination=loop_entry_node,
                end_destination=exit_node,
                can_jump_to=_get_can_jump_to(m1, "before_model"),
            )
                # 在最后一个 before_model 后直接进入 model
        _add_middleware_edge(
            graph,
            name=f"{middleware_w_before_model[-1].name}.before_model",
            default_destination="model",
            model_destination=loop_entry_node,
            end_destination=exit_node,
            can_jump_to=_get_can_jump_to(middleware_w_before_model[-1], "before_model"),
        )

        # 添加 after_model 中间件边
    if middleware_w_after_model:
        graph.add_edge("model", f"{middleware_w_after_model[-1].name}.after_model")
        for idx in range(len(middleware_w_after_model) - 1, 0, -1):
            m1 = middleware_w_after_model[idx]
            m2 = middleware_w_after_model[idx - 1]
            _add_middleware_edge(
                graph,
                name=f"{m1.name}.after_model",
                default_destination=f"{m2.name}.after_model",
                model_destination=loop_entry_node,
                end_destination=exit_node,
                can_jump_to=_get_can_jump_to(m1, "after_model"),
            )
                # 注意：从 after_model 到 after_agent/END 的连接已在上方的条件边部分处理
                # 在条件边部分

        # 添加 after_agent 中间件边
    if middleware_w_after_agent:
                # 链式连接 after_agent 中间件（在最后运行一次，位于 END 之前）
        for idx in range(len(middleware_w_after_agent) - 1, 0, -1):
            m1 = middleware_w_after_agent[idx]
            m2 = middleware_w_after_agent[idx - 1]
            _add_middleware_edge(
                graph,
                name=f"{m1.name}.after_agent",
                default_destination=f"{m2.name}.after_agent",
                model_destination=loop_entry_node,
                end_destination=exit_node,
                can_jump_to=_get_can_jump_to(m1, "after_agent"),
            )

                # 将最后一个 after_agent 连接到 END
        _add_middleware_edge(
            graph,
            name=f"{middleware_w_after_agent[0].name}.after_agent",
            default_destination=END,
            model_destination=loop_entry_node,
            end_destination=exit_node,
            can_jump_to=_get_can_jump_to(middleware_w_after_agent[0], "after_agent"),
        )

        # 设置递归限制为 9_999
        # https://github.com/langchain-ai/langgraph/issues/7313
    config: RunnableConfig = {"recursion_limit": 9_999}
    config["metadata"] = {"ls_integration": "langchain_create_agent"}
    if name:
        config["metadata"]["lc_agent_name"] = name

        # 执行内部模型调用的中间件（例如 `SummarizationMiddleware`）
        # 每个中间件在其自身的 `transformers` 元组中声明
    # `InternalCallTransformer`，以便仅在实际使用时注册
        # 1. 若状态中存在显式的 jump_to，使用它
    middleware_transformers = [t for m in middleware for t in getattr(m, "transformers", ())]

    return graph.compile(
        checkpointer=checkpointer,
        store=store,
        interrupt_before=interrupt_before,
        interrupt_after=interrupt_after,
        debug=debug,
        name=name,
        cache=cache,
        transformers=_dedupe_transformers(
            [
                ToolCallTransformer,
                SubagentTransformer,
                *middleware_transformers,
                *(transformers or ()),
            ]
        ),
    ).with_config(config)


def _dedupe_transformers(
    factories: Iterable[TransformerFactory],
) -> list[TransformerFactory]:
    """按身份保持顺序去重的 transformer 工厂。

    `AgentMiddleware.transformers` accepts any scope-aware callable, and
    callables aren't required to be hashable (e.g. a dataclass-based factory
    with `__hash__ = None`), so this can't use a `set`/`dict` keyed on the
    factories themselves. Combining several middleware that each declare the
    same shared transformer class (e.g. `InternalCallTransformer`) would
    otherwise register it once per middleware.

    Args:
        factories: 需要去重的 transformer 工厂，按注册顺序排列。

    Returns:
        移除完全重复项（通过 `is` 判断）后的相同工厂，保持原有顺序。
    """
    seen_ids: set[int] = set()
    deduped: list[TransformerFactory] = []
    for factory in factories:
        if id(factory) not in seen_ids:
            seen_ids.add(id(factory))
            deduped.append(factory)
    return deduped


def _resolve_jump(
    jump_to: JumpTo | None,
    *,
    model_destination: str,
    end_destination: str,
) -> str | None:
    if jump_to == "model":
        return model_destination
    if jump_to == "end":
        return end_destination
    if jump_to == "tools":
        return "tools"
    return None


def _fetch_last_ai_and_tool_messages(
    messages: list[AnyMessage],
) -> tuple[AIMessage | None, list[ToolMessage]]:
    """返回最后一条 AI 消息及后续的所有工具消息。

    Args:
        messages: 需要搜索的消息列表。

    Returns:
        一个元组 (last_ai_message, tool_messages)。如果没有找到 AIMessage，
        返回 (None, [])。调用方必须适当地处理 None 情况。
    """
    for i in range(len(messages) - 1, -1, -1):
        if isinstance(messages[i], AIMessage):
            last_ai_message = cast("AIMessage", messages[i])
            tool_messages = [m for m in messages[i + 1 :] if isinstance(m, ToolMessage)]
            return last_ai_message, tool_messages

    return None, []


def _make_model_to_tools_edge(
    *,
    model_destination: str,
    structured_output_tools: dict[str, OutputToolBinding[Any]],
    end_destination: str,
) -> Callable[[dict[str, Any]], str | list[Send] | None]:
    def model_to_tools(
        state: dict[str, Any],
    ) -> str | list[Send] | None:
                # 2. 若不存在 AIMessage（例如消息被清除），退出循环
        if jump_to := state.get("jump_to"):
            return _resolve_jump(
                jump_to,
                model_destination=model_destination,
                end_destination=end_destination,
            )

        last_ai_message, tool_messages = _fetch_last_ai_and_tool_messages(state["messages"])

                # 3. 若模型未调用任何工具，退出循环
        if last_ai_message is None:
            return end_destination

        tool_message_ids = [m.tool_call_id for m in tool_messages]

                # 这是智能体循环的经典退出条件
                # 4. 若存在待处理的工具调用，跳转至工具节点
        if len(last_ai_message.tool_calls) == 0:
            return end_destination

        pending_tool_calls = [
            c
            for c in last_ai_message.tool_calls
            if c["id"] not in tool_message_ids and c["name"] not in structured_output_tools
        ]

                # 工具节点通过在执行时从 channels 读取 CONFIG_KEY_READ 来填充
        # ToolRuntime.state，因此我们不再将完整状态内联到每个 Send 中（此前会导致 TASKS
        # 写入复杂度达到 O(N^2)）
                # 5. 若本次调用生成了新的结构化响应，退出循环
                # 6. AIMessage
        # 包含工具调用，但无待处理工具调用，表明人工工具消息已被注入。跳转至 model 节点
                # 将完整状态写入每个 Send（此前在 TASKS 写入中为 O(N^2)）。
        if pending_tool_calls:
            return [Send("tools", [tool_call]) for tool_call in pending_tool_calls]

                # 5. 如果本次调用生成了新的结构化响应，则退出循环
        if state.get("structured_response") is not None:
            return end_destination

                # 6. AIMessage 包含工具调用，但没有待处理的工具调用，这表明
                # 人工工具消息的注入。跳转到模型节点
        return model_destination

    return model_to_tools


def _make_model_to_model_edge(
    *,
    model_destination: str,
    end_destination: str,
) -> Callable[[dict[str, Any]], str | list[Send] | None]:
    def model_to_model(
        state: dict[str, Any],
    ) -> str | list[Send] | None:
                # 1. 优先级：检查中间件是否发出明确的 jump_to 指令
        if jump_to := state.get("jump_to"):
            return _resolve_jump(
                jump_to,
                model_destination=model_destination,
                end_destination=end_destination,
            )

                # 2. 退出条件：本次调用生成了新的结构化输出
        if state.get("structured_response") is not None:
            return end_destination

                # 3. 默认情况：继续循环，可能在结构化输出生成时出现问题
                # 因此需要重试
        return model_destination

    return model_to_model


def _make_tools_to_model_edge(
    *,
    tool_node: ToolNode,
    model_destination: str,
    structured_output_tools: dict[str, OutputToolBinding[Any]],
    end_destination: str,
) -> Callable[[dict[str, Any]], str | None]:
    def tools_to_model(state: dict[str, Any]) -> str | None:
        last_ai_message, tool_messages = _fetch_last_ai_and_tool_messages(state["messages"])

                # 1. 若不存在 AIMessage（例如消息已被清除），则路由到模型
        if last_ai_message is None:
            return model_destination

                # 2. 退出条件：所有已执行的工具均返回 return_direct=True
                # 仅保留客户端工具（提供方工具不在 tool_node 中）
                # 优先使用 ToolMessage 中的工具名称，因为存在重定向（例如来自 HITL）
        executed_by_id = {
            t.tool_call_id: t.name for t in tool_messages if t.name in tool_node.tools_by_name
        }
        executed_names = [
            name
            for c in last_ai_message.tool_calls
            if (name := executed_by_id.get(c["id"] or "", c["name"])) in tool_node.tools_by_name
        ]
        if executed_names and all(
            tool_node.tools_by_name[name].return_direct for name in executed_names
        ):
            return end_destination

                # 3. 退出条件：已执行结构化输出工具
        if any(t.name in structured_output_tools and t.status != "error" for t in tool_messages):
            return end_destination

                # 4. 默认情况：继续循环
                #    工具执行成功完成，路由回模型
                #    以便其处理工具结果并决定下一步操作
        return model_destination

    return tools_to_model


def _add_middleware_edge(
    graph: StateGraph[
        AgentState[ResponseT], ContextT, InputAgentState, OutputAgentState[ResponseT]
    ],
    *,
    name: str,
    default_destination: str,
    model_destination: str,
    end_destination: str,
    can_jump_to: list[JumpTo] | None,
) -> None:
    """向图中添加一条中间件节点的边。

    Args:
        graph: 要添加边的图。
        name: 中间件节点的名称。
        default_destination: 边的默认目标。
        model_destination: 边到模型的目标。
        end_destination: 边到末端的目标。
        can_jump_to: 条件跳转的目标。
    """
    if can_jump_to:

        def jump_edge(state: dict[str, Any]) -> str:
            return (
                _resolve_jump(
                    state.get("jump_to"),
                    model_destination=model_destination,
                    end_destination=end_destination,
                )
                or default_destination
            )

        destinations = [default_destination]

        if "end" in can_jump_to:
            destinations.append(end_destination)
        if "tools" in can_jump_to:
            destinations.append("tools")
        if "model" in can_jump_to and name != model_destination:
            destinations.append(model_destination)

        graph.add_conditional_edges(name, RunnableCallable(jump_edge, trace=False), destinations)

    else:
        graph.add_edge(name, default_destination)


__all__ = [
    "create_agent",
]

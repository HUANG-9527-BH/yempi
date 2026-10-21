"""使用 LangGraph 中断来回答 MCP 采集请求。

A server needing input mid-call returns an `InputRequiredResult` and expects the
`tools/call` to be retried with answers. This module drives that loop, sourcing
each answer from `interrupt()` so the human reviewing the agent answers too.

We drive the loop here rather than via the SDK's `run_input_required_driver`
because that driver answers from callbacks run concurrently in a task group:
LangGraph matches resume values to `interrupt()` calls by order, so concurrent
firing scrambles the matching, and FastMCP would swallow the `GraphInterrupt` as
an MCP error. Calling `interrupt()` from this frame keeps one per round.

仅回答采集请求。采样、根节点和继续轮次的请求被拒绝，而非部分响应。
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any, Final, Literal, TypeAlias, TypedDict, TypeVar

from fastmcp.client.group import ClientGroup
from langgraph.types import interrupt
from mcp.types import (
    CallToolResult,
    ElicitRequest,
    ElicitRequestFormParams,
    ElicitResult,
    InputRequiredResult,
    InputResponses,
)
from mcp.types.version import LATEST_MODERN_VERSION, is_version_at_least
from typing_extensions import NotRequired

if TYPE_CHECKING:
    from collections.abc import Coroutine

    from fastmcp.client import Client
    from mcp.client.session import ClientSession
    from mcp.types import InputRequest


_ResultT = TypeVar("_ResultT")


ELICITATION_INTERRUPT_TYPE: Final = "mcp_elicitation"
"""Discriminator on the interrupt payload, so a handler can recognize it."""

_ACTIONS: Final = ("accept", "decline", "cancel")
"""Actions the wire accepts, for validating a resumed answer."""

MCPFormContent: TypeAlias = dict[str, str | int | float | bool | list[str] | None]
"""Values a form answer may carry, as narrow as `mcp.types.ElicitResult.content`."""


class MCPElicitationFormRequest(TypedDict):
    """一个匹配模式的数据请求。

    Attributes:
        key: The server's identifier for this request. An answer must be
            returned under the same key.
        message: The server's prompt, intended for a human to read.
        mode: Always `'form'`.
        requested_schema: JSON schema the answer's `content` must satisfy.
    """

    key: str
    message: str
    mode: Literal["form"]
    requested_schema: dict[str, Any]


class MCPElicitationUrlRequest(TypedDict):
    """一个请求人类访问某个地址的请求。

    Attributes:
        key: The server's identifier for this request. An answer must be
            returned under the same key.
        message: The server's prompt, intended for a human to read.
        mode: Always `'url'`.
        url: The address the human should visit.
    """

    key: str
    message: str
    mode: Literal["url"]
    url: str


MCPElicitationRequest: TypeAlias = MCPElicitationFormRequest | MCPElicitationUrlRequest
"""One question an MCP server asks before finishing a tool call. Narrow on `mode`."""


class MCPElicitationInterrupt(TypedDict):
    """在 MCP 工具调用等待输入时引发的中断负载。

    Attributes:
        type: 始终为 `'mcp_elicitation'`。
        tool_name: 正在等待调用的 MCP 工具名称。
        requests: 本轮所有问题，按应提问的顺序排列。
    """

    type: Literal["mcp_elicitation"]
    tool_name: str
    requests: list[MCPElicitationRequest]


class MCPElicitationAccept(TypedDict):
    """对 `MCPElicitationRequest` 的回答。

    Attributes:
        action: 始终为 `'accept'`。
        content: 回答本身，需匹配请求的 `requested_schema`。
            对 `'form'` 请求预期存在；对 `'url'` 请求则不包含。
    """

    action: Literal["accept"]
    content: NotRequired[MCPFormContent | None]


class MCPElicitationDecline(TypedDict):
    """拒绝回答 `MCPElicitationRequest`，使调用继续进行。

    Attributes:
        action: 始终为 `'decline'`。
    """

    action: Literal["decline"]


class MCPElicitationCancel(TypedDict):
    """拒绝，放弃工具调用，而不仅仅是拒绝问题。

    Attributes:
        action: 始终为 `'cancel'`。
    """

    action: Literal["cancel"]


MCPElicitationResponse: TypeAlias = (
    MCPElicitationAccept | MCPElicitationDecline | MCPElicitationCancel
)
"""One answer to an `MCPElicitationRequest`. Narrow on `action`; only accept carries content."""


class MCPElicitationResume(TypedDict):
    """用于恢复因 MCP 采集而中断的运行的值。

    Attributes:
        responses: 每个请求一个答案，按请求的 `key` 键入。
    """

    responses: dict[str, MCPElicitationResponse]


def _describe_request(key: str, request: ElicitRequest) -> MCPElicitationRequest:
    """将一个服务器请求渲染为人类可操作的负载。"""
    params = request.params
    if isinstance(params, ElicitRequestFormParams):
        return MCPElicitationFormRequest(
            key=key,
            message=params.message,
            mode="form",
            requested_schema=params.requested_schema,
        )
    return MCPElicitationUrlRequest(
        key=key,
        message=params.message,
        mode="url",
        url=str(params.url),
    )


def _elicit_requests(
    requests: dict[str, InputRequest],
    tool_name: str,
) -> dict[str, ElicitRequest]:
    """将一轮请求缩小至采集请求，对采样或根节点请求抛出异常。"""
    unsupported = sorted(
        f"{key} ({request.method})"
        for key, request in requests.items()
        if not isinstance(request, ElicitRequest)
    )
    if unsupported:
        msg = (
            f"The MCP tool {tool_name!r} asked for input that interrupt-based "
            f"elicitation cannot answer: {', '.join(unsupported)}. Only "
            "elicitation requests are supported."
        )
        raise NotImplementedError(msg)
    return {key: request for key, request in requests.items() if isinstance(request, ElicitRequest)}


def _build_responses(
    requests: dict[str, ElicitRequest],
    answers: dict[str, MCPElicitationResponse],
    tool_name: str,
) -> InputResponses:
    """将恢复的答案转换为服务器的响应，对任何缺失或格式错误的响应抛出异常。"""
    missing = sorted(set(requests) - set(answers))
    if missing:
        msg = (
            f"Resuming the MCP tool {tool_name!r} needs an answer for every "
            f"elicitation request, but these had none: {', '.join(missing)}."
        )
        raise ValueError(msg)

    responses: InputResponses = {}
    for key in requests:
        answer = answers[key]
        action = answer.get("action")
        if action not in _ACTIONS:
            msg = (
                f"Elicitation answer for {key!r} has action {action!r}; expected "
                "'accept', 'decline', or 'cancel'."
            )
            raise ValueError(msg)
        content = answer.get("content") if action == "accept" else None
        responses[key] = ElicitResult(action=action, content=content)
    return responses


_ARMED_MARKER: Final = "_langchain_mcp_interrupt_armed"
"""Attribute stamped on a client the adapter armed for interrupt elicitation."""


async def _declare_elicitation_capability(*_: object) -> dict[str, Any]:
    """仅用于声明能力的采集处理器。

    FastMCP advertises `elicitation` only when a handler is set, so a client
    that should drive the interrupt loop needs one. The loop answers from the
    tool call, not this handler, so it never runs on a modern server. It only
    fires when a legacy server initiates elicitation the old way — which the
    interrupt loop cannot answer, hence the error.
    """
    msg = (
        "This MCP server asked for input over the legacy server-initiated "
        "elicitation path, which interrupt-based elicitation cannot answer. It "
        "needs a server that returns its requests as an `InputRequiredResult`, "
        "or a pre-built client with your own `elicitation_handler`."
    )
    raise NotImplementedError(msg)


def _arm_for_interrupts(client: Client[Any]) -> None:
    """设置哨兵处理器，并将客户端标记为适配器已启用。"""
    client.set_elicitation_callback(_declare_elicitation_capability)
    setattr(client, _ARMED_MARKER, True)


def _drives_interrupts(client: Client[Any]) -> bool:
    """报告通过 `client` 的调用是否应驱动中断循环。

    True only for a client the adapter armed (not one carrying a caller's own
    handler) that is connected to a modern-era server. The `InputRequiredResult`
    the loop answers is a modern feature (SEP-2322); a legacy server never
    returns one, so the call falls back to a plain `call_tool` there.
    """
    if not getattr(client, _ARMED_MARKER, False):
        return False
    version = getattr(client, "protocol_version", None)
    return version is not None and is_version_at_least(version, LATEST_MODERN_VERSION)


async def _await_monitored(client: Client[Any], coro: Coroutine[Any, Any, _ResultT]) -> _ResultT:
    """将会话请求与会话任务并行竞争，以防止错误导致其挂起。

    On HTTP transports a server error surfaces in the background session task,
    not in the awaiting coroutine. FastMCP's guard for this is private, so warn
    and fall back if it ever moves.
    """
    monitored = getattr(client, "_await_with_session_monitoring", None)
    if monitored is None:
        warnings.warn(
            "This version of FastMCP does not expose "
            "`Client._await_with_session_monitoring`, so an MCP elicitation "
            "round cannot be raced against the session task. A transport "
            "failure mid-elicitation may hang instead of raising.",
            RuntimeWarning,
            stacklevel=2,
        )
        return await coro
    result: _ResultT = await monitored(coro)
    return result


async def _resolve_session(
    client: Client[Any] | ClientGroup, tool_name: str
) -> tuple[Client[Any], ClientSession, str]:
    """解析工具所属的成员客户端、其会话以及服务器名称。

    对于组而言，`list_tools` 将目录命名空间化为
    `{server}_{tool}`，但提供服务的客户端仅知晓工具的自身名称；`resolve_tool`
    将命名空间名称映射回该成员及其上游名称。对于单个客户端，工具已通过其自身名称直接指定。

    Args:
        client: 提供工具服务的武装客户端或组。
        tool_name: 此适配器发布时工具的名称。

    Returns:
        成员客户端、其连接的会话，以及成员服务器所知的工具名称。
    """
    if isinstance(client, ClientGroup):
        route = await client.resolve_tool(tool_name)
        return route.client, route.client.session, route.upstream_name
    return client, client.session, tool_name


async def _call_tool_with_interrupts(
    client: Client[Any] | ClientGroup,
    tool_name: str,
    arguments: dict[str, Any],
) -> CallToolResult:
    """调用 MCP 工具，对每轮请求的输入以中断方式响应。

    `interrupt()` unwinds the whole call, so on resume it is re-issued from the
    first round. A server that asks before doing work repeats nothing; one that
    works first repeats that work once per round.

    Args:
        client: 一个已武装并可宣传获取能力的连接客户端。
        tool_name: 要调用的 MCP 工具。
        arguments: 工具的参数。

    Returns:
        工具的最终结果。

    Raises:
        GraphInterrupt: 每个未回答的轮次。这是实现机制。
        NotImplementedError: 在采样/根请求或继续轮次时。
        ValueError: 如果恢复的答案缺失或格式错误。
    """
        # FastMCP 的公开 `call_tool` 不再转发 `allow_input_required`；它
        # 自身通过并发驱动程序驱动中断循环，这正是本模块用每轮一个 `interrupt()`
    # 替代的行为。因此
        # 应针对成员会话驱动它，其 `call_tool` 仍会返回我们需回应的
        # `InputRequiredResult`。
        # 一个仅包含 `request_state`
    # 的轮次是继续：服务器仍在工作，没有问题需要中断。如同采样一样被拒绝。
    member, session, upstream_name = await _resolve_session(client, tool_name)

    result = await _await_monitored(
        member, session.call_tool(upstream_name, arguments, allow_input_required=True)
    )

    while isinstance(result, InputRequiredResult):
        if not result.input_requests:
                        # 不透明；原样回显
                        # 仍在运行，没有问题需要中断。拒绝行为类似采样。
            msg = (
                f"The MCP tool {tool_name!r} returned a continuation round with "
                "no input requests. Interrupt-based elicitation only answers "
                "elicitation requests, not long-running work that resumes over "
                "several round trips."
            )
            raise NotImplementedError(msg)

        requests = _elicit_requests(result.input_requests, tool_name)
        request: MCPElicitationInterrupt = {
            "type": ELICITATION_INTERRUPT_TYPE,
            "tool_name": tool_name,
            "requests": [_describe_request(key, req) for key, req in requests.items()],
        }
        resume: MCPElicitationResume = interrupt(request)
        responses = _build_responses(requests, resume["responses"], tool_name)

        result = await _await_monitored(
            member,
            session.call_tool(
                upstream_name,
                arguments,
                input_responses=responses,
                request_state=result.request_state,  # 不透明；原样回显
                allow_input_required=True,
            ),
        )

    return result


__all__ = [
    "ELICITATION_INTERRUPT_TYPE",
    "MCPElicitationAccept",
    "MCPElicitationCancel",
    "MCPElicitationDecline",
    "MCPElicitationFormRequest",
    "MCPElicitationInterrupt",
    "MCPElicitationRequest",
    "MCPElicitationResponse",
    "MCPElicitationResume",
    "MCPElicitationUrlRequest",
    "MCPFormContent",
]

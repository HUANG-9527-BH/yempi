"""将 MCP 工具和工具结果转换为 LangChain 原生值。

Tool results follow `langchain-mcp-adapters`, so a call made through
`langchain.mcp` reaches a model in the same shape as one loaded by that
package. Tool *metadata* is richer here: it is grouped under a single `mcp`
namespace with the tool's annotations and `_meta` under `mcp.tool` and the
serving server's identity under `mcp.server`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, TypedDict

from fastmcp.client.group import ClientGroup
from langchain_core.messages.content import (
    FileContentBlock,
    ImageContentBlock,
    TextContentBlock,
    create_file_block,
    create_image_block,
    create_text_block,
)
from langchain_core.tools import BaseTool, StructuredTool, ToolException
from mcp.types import (
    AudioContent,
    BlobResourceContents,
    ContentBlock,
    EmbeddedResource,
    ImageContent,
    ResourceLink,
    TextContent,
    TextResourceContents,
)

from langchain.mcp.elicitation import _call_tool_with_interrupts, _drives_interrupts

if TYPE_CHECKING:
    from fastmcp.client import Client
    from mcp.types import Tool


class _ToolCallResult(Protocol):
    """此模块读取的结果字段。

    FastMCP's `CallToolResult` dataclass and the MCP SDK's own pydantic
    `CallToolResult` both satisfy this, and elicitation-driven calls return the
    latter.
    """

    content: list[ContentBlock]
    structured_content: dict[str, Any] | None
    is_error: bool


ToolMessageContentBlock = TextContentBlock | ImageContentBlock | FileContentBlock
"""LangChain content blocks an MCP tool result can convert into."""


class MCPToolArtifact(TypedDict):
    """附加到 MCP 工具调用生成的 `ToolMessage` 的工件。

    Wrapping the structured content in a `TypedDict` leaves room for further
    MCP result fields without changing the artifact's shape.

    Attributes:
        structured_content: MCP 工具结果的 `structuredContent`。
    """

    structured_content: Any


def _summarize_tool_error(tool_content: list[ToolMessageContentBlock]) -> str:
    """从失败工具调用的内容块构建可读消息。

    Image and file blocks are summarized by count rather than interpolated, so a
    base64 payload never lands in an exception message.
    """
    error_parts = [block["text"] for block in tool_content if block["type"] == "text"]
    if error_parts:
        return "\n".join(error_parts)
    if tool_content:
        return (
            "The MCP tool reported an error with no text content "
            f"({len(tool_content)} non-text content block(s))."
        )
    return "The MCP tool reported an error with empty content."


class _MCPToolExecutionError(ToolException):
    """一个已运行并报告失败的 MCP 工具，其 `isError=True`。

    携带转换后的内容块，以便 `_handle_mcp_tool_error`
    可将服务器自身的错误详情传递给模型，而不是终止运行。

    Deliberately narrow: only `isError=True` results raise this. Transport and
    conversion failures are not `ToolException`s, so they bypass error handling.
    """

    def __init__(self, tool_content: list[ToolMessageContentBlock]) -> None:
        super().__init__(_summarize_tool_error(tool_content))
        self.tool_content = tool_content


def _handle_mcp_tool_error(error: ToolException) -> list[ToolMessageContentBlock]:
    """将 MCP 执行错误暴露给模型，作为失败的工具输出。

    Wired as `handle_tool_error`, so what this returns becomes a `ToolMessage`
    with `status="error"` carrying the server's own detail. Any other
    `ToolException` is re-raised.
    """
    if isinstance(error, _MCPToolExecutionError):
        if error.tool_content:
            return error.tool_content
                # 空的 `ToolMessage` 对某些提供方来说是脆弱的结构，因此
                # 应使用占位符而非完全不传内容。
        return [create_text_block(text=str(error))]
    raise error


def _convert_content_block(content: ContentBlock) -> ToolMessageContentBlock:
    """将一个 MCP 内容块转换为其 LangChain 对应项。"""
    if isinstance(content, TextContent):
        return create_text_block(text=content.text)

    if isinstance(content, ImageContent):
        return create_image_block(base64=content.data, mime_type=content.mime_type)

    if isinstance(content, AudioContent):
        msg = (
            "Converting MCP audio content to a LangChain content block is not yet "
            f"supported. Received audio with mime type: {content.mime_type}"
        )
        raise NotImplementedError(msg)

    if isinstance(content, ResourceLink):
        mime_type = content.mime_type or None
        if mime_type and mime_type.startswith("image/"):
            return create_image_block(url=content.uri, mime_type=mime_type)
        return create_file_block(url=content.uri, mime_type=mime_type)

    if isinstance(content, EmbeddedResource):
        resource = content.resource
        if isinstance(resource, TextResourceContents):
            return create_text_block(text=resource.text)
        if isinstance(resource, BlobResourceContents):
            mime_type = resource.mime_type or None
            if mime_type and mime_type.startswith("image/"):
                return create_image_block(base64=resource.blob, mime_type=mime_type)
            return create_file_block(base64=resource.blob, mime_type=mime_type)
                # 在 SDK 的资源联合体持有期间不可达；参见下方注释。
        msg = f"Unknown embedded resource type: {type(resource).__name__}"  # type: ignore[unreachable]
        raise ValueError(msg)

        # 两个联合体在类型检查时均已关闭，因此 mypy 可证明这两行代码
        # 不可达——而 `unreachable` 忽略项会在 SDK
    # 增加成员的瞬间变成未使用的忽略错误，这正是保持完整性的原因。
        # 运行时保护依然有其价值：联合体的封闭性取决于安装的 `mcp`，而较新 SDK
    # 的调用者更应受益于命名类型，而非 `assert_never` 所引发的裸 `AssertionError`。
    #
        # snake_case（无 `by_alias`）在 Python 中读起来更自然，并与
        # `langchain-mcp-adapters` 生成的内容一致。
        # `_meta` 是 MCP
    # 的网络字段名；保持原样，以确保键与协议匹配，且服务器的负载保留在同一嵌套位置。
    msg = (  # type: ignore[unreachable]
        f"Unknown MCP content type: {type(content).__name__}. This usually means "
        "the installed `mcp` is newer than the version this adapter supports."
    )
    raise ValueError(msg)


def _convert_call_tool_result(
    result: _ToolCallResult,
) -> tuple[list[ToolMessageContentBlock], MCPToolArtifact | None]:
    """将一个 MCP 工具结果拆分为模型可见的内容和一个工件。"""
    tool_content = [_convert_content_block(block) for block in result.content]

    if result.is_error:
        raise _MCPToolExecutionError(tool_content)

    artifact: MCPToolArtifact | None = None
    if result.structured_content is not None:
        artifact = MCPToolArtifact(structured_content=result.structured_content)
    return tool_content, artifact


def _tool_metadata(tool: Tool, client: Client[Any] | None) -> dict[str, Any] | None:
    """收集值得保留的 MCP 工具级和服务器级元数据。

    Everything lives under a single `mcp` namespace so a consumer can tell an
    MCP tool's provenance from any other metadata on the LangChain tool, and so
    tool-level fields (`annotations`, `_meta`) stay distinct from the identity
    of the server that served the tool.
    """
    tool_meta: dict[str, Any] = {}
    if tool.annotations is not None:
                # 服务器身份来自连接，而非工具——`Tool`
        # 不携带服务器字段。它在客户端连接时填充，这正是转换时的情况。
                # 保留 MCP 错误结果，用于转换为失败的工具消息。
        tool_meta["annotations"] = tool.annotations.model_dump(exclude_none=True)
    if tool.meta is not None:
                # _meta 是 MCP 通信字段名称；保持原样以确保键匹配
                # 协议和服务器的负载保持在同一个嵌套位置。
        tool_meta["_meta"] = tool.meta

    mcp: dict[str, Any] = {}
    if tool_meta:
        mcp["tool"] = tool_meta
        # 服务器身份信息来自连接，而非工具——一个 `Tool` 携带
        # 无 server 字段。该字段在客户端连接时填充，此时
        # 转换时的情况。
    if client is not None and client.server_info is not None:
        mcp["server"] = client.server_info.model_dump(exclude_none=True)
    return {"mcp": mcp} if mcp else None


def _normalize_mcp_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """在提供方模式转换期间保持开放对象参数的开放性。"""
    normalized = dict(schema)
    properties = schema.get("properties")
    if not isinstance(properties, dict):
        return normalized
    normalized["properties"] = dict(properties)
    for name, value in properties.items():
        if not isinstance(value, dict):
            continue
        types = value.get("type")
        is_object = types == "object" or (isinstance(types, list) and "object" in types)
        if (
            is_object
            and not value.get("properties")
            and "additionalProperties" not in value
            and "unevaluatedProperties" not in value
        ):
            normalized["properties"][name] = {**value, "additionalProperties": True}
    return normalized


async def as_langchain_tool(
    tool: Tool,
    client: Client[Any] | ClientGroup,
) -> BaseTool:
    """将一个 MCP 工具转换为 LangChain 工具。

    返回的工具在每次调用时通过 `client` 调用 MCP 工具。FastMCP
    客户端是可重入的，因此无论其他位置是否持有连接，该工具都可以自行打开客户端。

    A server that needs input mid-call is answered with a LangGraph
    `interrupt()` when `client` carries the interrupt-driving sentinel handler
    that `MCPAdapter` installs — so a human answers and the call resumes, see
    `langchain.mcp.elicitation`. A client that carries a different handler (a
    caller's own) uses that handler instead, and one with no handler simply
    never gets asked. Which path a call takes is read off the client, so the
    behavior matches whatever it was armed with.

    An MCP tool that runs and reports failure reaches the model as a
    `ToolMessage` with `status="error"`, carrying the server's own error
    content, so an agent can correct itself and retry. Transport failures and
    unconvertible content propagate instead, since a model cannot act on them.

    Args:
        tool: 由 `fastmcp.Client.list_tools` 返回的 MCP 工具。
        client: 通过该工具调用的 FastMCP 客户端。

    Returns:
        一个异步调用 MCP 工具的 LangChain 工具。

    Example:
        ```python
        from fastmcp import Client

        from langchain.mcp import as_langchain_tool

        client = Client("https://example.com/mcp")
        async with client:
            mcp_tools = await client.list_tools()
        tools = [await as_langchain_tool(t, client) for t in mcp_tools]
        ```
    """
    if isinstance(client, ClientGroup):
        tool_route = await client.resolve_tool(tool.name)
        requesting_client = tool_route.client
    else:
        requesting_client = client
    drives_interrupts = _drives_interrupts(requesting_client)

    async def call_tool(
        **arguments: Any,
    ) -> tuple[list[ToolMessageContentBlock], MCPToolArtifact | None]:
        """调用捕获的 MCP 工具并转换其结果。"""
        result: _ToolCallResult
        async with client:
            if drives_interrupts:
                result = await _call_tool_with_interrupts(client, tool.name, arguments)
            else:
                                # 保留 MCP 错误结果，用于转换为失败的工具消息。
                result = await client.call_tool(tool.name, arguments, raise_on_error=False)
        return _convert_call_tool_result(result)

    return StructuredTool(
        name=tool.name,
        description=tool.description or "",
        args_schema=_normalize_mcp_schema(tool.input_schema),
        coroutine=call_tool,
        response_format="content_and_artifact",
        metadata=_tool_metadata(tool, requesting_client),
        handle_tool_error=_handle_mcp_tool_error,
    )


__all__ = ["MCPToolArtifact", "as_langchain_tool"]

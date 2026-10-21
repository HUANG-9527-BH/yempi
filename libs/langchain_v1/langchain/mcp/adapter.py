"""将 MCP 工具适配为适用于 `create_agent` 的 LangChain 工具。"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, TypeAlias

from pydantic import AnyUrl, TypeAdapter, ValidationError
from typing_extensions import Self

from langchain.mcp.elicitation import _arm_for_interrupts
from langchain.mcp.tools import as_langchain_tool

try:
    from fastmcp.client import Client as FastMCPClient
    from fastmcp.client.group import ClientGroup
    from fastmcp.client.transports import ClientTransport
    from fastmcp.mcp_config import MCPConfig
    from mcp.server.mcpserver import MCPServer
except ImportError as _import_error:
    msg = "Please install FastMCP to use `MCPAdapter` — `pip install fastmcp`."
    raise ImportError(msg) from _import_error


if TYPE_CHECKING:
    from types import TracebackType

    from fastmcp import FastMCP
    from langchain_core.tools import BaseTool

        # `CacheMode` 是 MCP SDK 自身的类型（SEP-2549）；FastMCP 重新导入它
        # 而非定义自己的版本，因此这是标准来源。
    from mcp.client.caching import CacheMode
else:
    try:
        from fastmcp import FastMCP
    except ImportError:
                # 进程内 FastMCP 服务器，以及多服务器 `MCPConfig` 目标，位于 FastMCP
        # 的服务器端。
                # `mcp` 可选依赖项安装它，但通过仅客户端安装到达此模块的调用者
                # （`fastmcp-slim[client]`）仍可使用所有其他目标类型——因此注解降级而非导入失败，
        # 与 FastMCP 在 `fastmcp.client.transports.inference` 中的降级方式相同。
                # `AnyUrl` 单独并非边界。它会将 Windows 路径如
                # `C:\server.py` 解析为方案 `c` 并接受它——且在 Windows 上该路径确实可解析，
        # 因此 FastMCP 将启动它。FastMCP 无法推断出任何传输方式的方案（如
        # `file:`、`ws:`） 也在此处被拒绝，以提供明确指出问题的错误，而非
        # "无法推断传输方式"。
                # FastMCP 在 `fastmcp.client.transports.inference` 中会降低性能。
        FastMCP = Any


MCPAdapterTarget: TypeAlias = (
    FastMCPClient[Any]
    | ClientGroup
    | ClientTransport
    | FastMCP
    | MCPServer
    | AnyUrl
    | Path
    | MCPConfig
    | dict[str, Any]
    | str
)
"""Anything `MCPAdapter` accepts as its `target`.

Every transport `fastmcp.Client` accepts, plus a pre-built `fastmcp.Client`.

A `str` target is the one member narrowed relative to `fastmcp.Client`: it must
be an `http`/`https` URL. Local servers are reached through `Path`, an explicit
transport, or `MCPConfig` — see `MCPAdapter`.

`ClientGroup` is the one member `fastmcp.Client` does not accept at all, since a
group is a peer of `Client` rather than a transport it could wrap.
"""


_URL_SCHEMES: Final = frozenset({"http", "https"})
"""Schemes a `str` target may carry.

The only schemes `fastmcp.Client` can reach a server over from a string, so
nothing that would have connected is forbidden here. It selects that branch on
a bare `str.startswith("http")` though, which also admits strings that are not
URLs at all — those fail on connect there, and on construction here.
"""

_ANY_URL: Final = TypeAdapter(AnyUrl)


def _validate_url_target(target: str) -> None:
    """拒绝不是 `http`/`https` URL 的 `str` 类型目标。

    `fastmcp.Client` infers a transport from a string by testing it as a
    filesystem path *before* testing it as a URL, so a string naming an
    existing `.py` or `.js` file launches that file as a subprocess. A string
    that reaches an application from configuration, a request body, or a model
    is expected to address a server over the network; letting it silently
    select local execution instead makes the dangerous reading the implicit
    one.

    Args:
        target: 传递给适配器的目标字符串。

    Raises:
        ValueError: 如果 `target` 不是 `http` 或 `https` URL。
    """
    hint = (
        "To run a local MCP server over stdio, ask for it explicitly with "
        "`Path('server.py')`, a `fastmcp` transport, or an `MCPConfig` — a bare "
        "string is read as a URL so that it cannot silently become a subprocess."
    )
    try:
        url = _ANY_URL.validate_python(target)
    except ValidationError as validation_error:
        msg = f"MCP target {target!r} is not a valid URL. {hint}"
        raise ValueError(msg) from validation_error

        # `AnyUrl` 单独使用并不是边界。它会读取如
        # `C:\server.py` 作为模式 `c` 并接受它——在 Windows 上就是这样
        # path *确实* 会解析，因此 FastMCP 会启动它。FastMCP 无法处理的协议
        # 从任何传输方式推断（如 `file:`、`ws:`）在此也被拒绝，因为
        # 一个明确指出问题所在而非 "无法推断传输方式" 的错误。
    if url.scheme not in _URL_SCHEMES:
        msg = (
            f"MCP target {target!r} has scheme {url.scheme!r}, but a string target must "
            f"be an http or https URL. {hint}"
        )
        raise ValueError(msg)


class MCPAdapter:
    """将 MCP 目标适配为 LangChain 工具。

    `MCPAdapter` uses FastMCP for protocol negotiation and connection management,
    then converts discovered MCP tools into asynchronous LangChain tools. The
    resulting tools can be passed directly to `create_agent`.

    传输推断由 `fastmcp.Client` 负责，因此目标可以是 URL、本地脚本路径（通过 stdio
    启动）、进程内服务器，或已构建的客户端。

    !!! warning "字符串目标必须是 http(s) URL"

        `fastmcp.Client` resolves a string by testing it as a filesystem path
        before testing it as a URL, so a string naming an existing `.py` or `.js`
        file launches that file as a subprocess. Because strings are the form a
        target most often arrives in from configuration or from a model,
        `MCPAdapter` rejects one that is not an `http` or `https` URL rather than
        let it select local execution. Reach a local server through `Path`, a
        `fastmcp` transport, or an `MCPConfig`, all of which say so explicitly.

    A server that needs input mid-call is answered with a LangGraph
    `interrupt()`, so a human answers and the run resumes — see
    `langchain.mcp.elicitation`. This is the default: the adapter arms every
    client it builds to advertise the elicitation capability and drives the
    interrupt loop on each call. A server that never asks for input is
    unaffected, since the loop only runs when the server returns a request.

    A pre-built client (or `ClientGroup`) that already carries its own
    elicitation handler is honored rather than overridden: its handler keeps
    answering, and the adapter leaves the client as the caller built it. Only a
    client with no handler is armed, and it is cloned first so the caller's own
    object is never mutated.

    Args:
        target: `fastmcp.Client` 接受的 MCP 目标，包括现有的 FastMCP 客户端。`str`
        必须是 `http`/`https` URL。

    Example:
        ```python
        from langchain.agents import create_agent
        from langchain.mcp import MCPAdapter

        async with MCPAdapter("https://example.com/mcp") as adapter:
            agent = create_agent("anthropic:claude-sonnet-5", await adapter.list_tools())
            result = await agent.ainvoke({"messages": [{"role": "user", "content": "..."}]})
        ```
    """

    def __init__(self, target: MCPAdapterTarget) -> None:
        """围绕 FastMCP 目标、客户端或组初始化适配器。

        Each underlying client is armed to answer elicitation with an interrupt,
        unless it already carries the caller's own handler, which is honored
        instead. A caller's client is cloned rather than mutated. See the class
        docstring.
        """

        def armed(client: FastMCPClient[Any]) -> FastMCPClient[Any]:
            if getattr(client, "_elicitation_callback", None) is not None:
                return client
            clone = client.new()
            _arm_for_interrupts(clone)
            return clone

        if isinstance(target, ClientGroup):
            members = {name: armed(client) for name, client in target.clients.items()}
            self._client: FastMCPClient[Any] | ClientGroup = ClientGroup(members)
        elif isinstance(target, FastMCPClient):
            self._client = armed(target)
        else:
            if isinstance(target, str):
                _validate_url_target(target)
            client = FastMCPClient(target)
            _arm_for_interrupts(client)
            self._client = client

    @property
    def client(self) -> FastMCPClient[Any] | ClientGroup:
        """返回底层的 FastMCP 客户端或客户端组，用于高级 MCP 操作。"""
        return self._client

    async def __aenter__(self) -> Self:
        """在上下文期间连接底层客户端或客户端组。"""
        await self._client.__aenter__()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """释放此上下文对底层客户端或客户端组的持有。"""
        await self._client.__aexit__(exc_type, exc_value, traceback)

    async def list_tools(self, *, cache_mode: CacheMode = "use") -> list[BaseTool]:
        """发现并适配 MCP 工具以供 LangChain 使用。

        Args:
            cache_mode: How discovery interacts with the client-side response
                cache (SEP-2549). `use` serves a cached tool list when one is
                present and still within the server's TTL hint, `refresh` calls
                the server and repopulates the cache, and `bypass` skips the
                cache entirely. The cache and its per-principal isolation are
                configured on the client itself (`Client(cache=...)`); this only
                selects how discovery reads it. Defaults to `use` so a
                configured cache is honored — note this differs from a bare
                `ClientGroup.list_tools()`, whose own default is `refresh`.

        Returns:
            用于异步调用相应 MCP 工具的 LangChain 工具。
            每个工具都持有适配器的客户端，因此在适配器上下文退出后仍可调用。
        """
        async with self:
            remote_tools = await self._client.list_tools(cache_mode=cache_mode)
            return [await as_langchain_tool(tool, self._client) for tool in remote_tools]


__all__ = ["MCPAdapter", "MCPAdapterTarget"]

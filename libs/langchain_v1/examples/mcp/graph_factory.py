"""一个位于 `langgraph dev` 图形工厂后的按用户划分的 MCP 舰队。"""

from __future__ import annotations

from typing import Any

import httpx2
from _servers import token_for
from fastmcp.client import Client
from fastmcp.client.auth import BearerAuth
from fastmcp.client.group import ClientGroup
from fastmcp.client.transports import StreamableHttpTransport

# 运行时导入（不包含 TYPE_CHECKING）：`langgraph dev` 将工厂分类
# 通过 `get_type_hints(make_graph)`，必须解析每个注解。
from langgraph.graph.state import CompiledStateGraph  # noqa: TC002
from langgraph_sdk.runtime import ServerRuntime  # noqa: TC002
from mcp.client.caching import CacheConfig, InMemoryResponseCacheStore

from langchain.agents import create_agent
from langchain.mcp import MCPAdapter

SYSTEM_PROMPT = "You answer questions using the tools available to you."
SERVERS = {
    "calendar": "http://localhost:8001/mcp",
    "docs": "http://localhost:8002/mcp",
}

_POOL = httpx2.AsyncHTTPTransport()  # 一个池，由所有人共享
_CACHE = InMemoryResponseCacheStore()  # 一个缓存，按用户分区


class _SharedPool(httpx2.AsyncBaseTransport):
        # 借出 `_POOL`，但不允许借用者关闭它。
    handle_async_request = _POOL.handle_async_request

    async def aclose(self) -> None: ...


def _client_factory(**kwargs: Any) -> httpx2.AsyncClient:
    return httpx2.AsyncClient(transport=_SharedPool(), **kwargs)


async def make_graph(runtime: ServerRuntime) -> CompiledStateGraph:
    """在调用用户的 MCP 舰队上构建智能体（每次运行时调用）。"""
    user = runtime.user.identity if runtime.user is not None else "anonymous"
    auth = BearerAuth(token_for(user))
    group = ClientGroup(
        {
            name: Client(
                StreamableHttpTransport(url, auth=auth, httpx_client_factory=_client_factory),
                cache=CacheConfig(store=_CACHE, target_id=user, partition=user),
            )
            for name, url in SERVERS.items()
        }
    )
        # `cache_mode="use"` 读取用户专属缓存；组的默认值会刷新它。
    tools = await MCPAdapter(group).list_tools(cache_mode="use")
    return create_agent("anthropic:claude-sonnet-5", tools, system_prompt=SYSTEM_PROMPT)

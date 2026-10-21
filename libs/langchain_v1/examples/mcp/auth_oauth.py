"""一个遵循完整 OAuth 2.1 流程的服务器，客户端在初始阶段无需提供凭证。

此进程扮演两个通常分离的角色：

- Resource Server: serves `/mcp` and rejects unauthenticated calls with a 401
  pointing at its protected-resource metadata.
- Authorization Server: serves discovery, `/register`, `/authorize`, `/token`.

In production those split — the resource server is yours, the authorization
server is Auth0 / WorkOS / Okta. FastMCP ships providers for those, and the flow
the client runs is identical, which is the point of the spec.

动态客户端注册使得整个过程如同即插即用：客户端在运行时自行注册，而非由您预先配置客户端
ID。 `InMemoryOAuthProvider`
会自动批准，因此浏览器标签页打开后会直接重定向回；真实服务器则会在那里显示登录和授权页面。

    uv run examples/mcp/auth_oauth.py

打开一个浏览器窗口以完成流程。
"""

from __future__ import annotations

import asyncio

from fastmcp import FastMCP
from fastmcp.client import Client
from fastmcp.server.auth.providers.in_memory import InMemoryOAuthProvider
from fastmcp.server.dependencies import get_access_token
from fastmcp.utilities.tests import run_server_in_process
from mcp.server.auth.settings import ClientRegistrationOptions

from langchain.mcp import MCPAdapter


def run_oauth_server(host: str, port: int) -> None:
    """提供一个日历服务器，该服务器自行生成令牌。"""
        # `base_url` 必须与客户端实际访问的 URL 一致：它会在发现文档中被通告，并作为 OAuth
    # 的 `resource` 使用。
        # 发现文档中会通告 `base_url`，并在 OAuth 请求中作为 `resource` 参数使用。
    auth = InMemoryOAuthProvider(
        base_url=f"http://127.0.0.1:{port}",
        client_registration_options=ClientRegistrationOptions(
            enabled=True,
            valid_scopes=["calendar:read"],
            default_scopes=["calendar:read"],
        ),
        required_scopes=["calendar:read"],
    )
    mcp: FastMCP[None] = FastMCP("calendar", auth=auth)

    @mcp.tool
    def whoami() -> str:
        """报告服务器从访问令牌中推导出的身份信息。"""
        token = get_access_token()
        if token is None:
            return "unauthenticated"
        return f"client_id={token.client_id} scopes={token.scopes}"

    mcp.run(transport="http", host=host, port=port, show_banner=False, log_level="warning")


async def main() -> None:
    """注册、授权并调用受保护的工具。"""
    with run_server_in_process(run_oauth_server) as url:
                # "oauth" 执行发现、动态注册、浏览器重定向以及令牌交换。
        # 令牌存储在内存中，因此每次运行都会重复浏览器步骤；若要持久化令牌，请传入
        # `OAuth(..., token_storage=...)`。
                # 令牌存储在内存中，因此每次运行都会重复浏览器步骤；若要持久化令牌，请传入
        # `OAuth(..., token_storage=...)`。
                # 令牌存储在内存中，因此每次运行都会重复浏览器步骤；若要持久化令牌，请传入
        # `OAuth(..., token_storage=...)`。
                # 令牌存储在内存中，因此每次运行都会重复浏览器步骤；若要持久化令牌，请传入
        # `OAuth(..., token_storage=...)`。
        async with MCPAdapter(Client(f"{url}/mcp", auth="oauth")) as adapter:
            [whoami] = await adapter.list_tools()
            [block] = await whoami.ainvoke({})
            print("authenticated as:", block["text"])


if __name__ == "__main__":
    asyncio.run(main())

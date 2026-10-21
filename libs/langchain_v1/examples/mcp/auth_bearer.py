"""一个位于静态持有者令牌后的服务器。

MCP 认证的简单端点：服务器从不颁发凭证，它仅验证在 `Authorization: Bearer <token>`
中到达的令牌。无发现、无浏览器、无刷新——您已通过带外方式配置了令牌，客户端只需呈现该令牌即可。

    uv run examples/mcp/auth_bearer.py
"""

from __future__ import annotations

import asyncio

from fastmcp import FastMCP
from fastmcp.client import Client
from fastmcp.server.auth.providers.jwt import StaticTokenVerifier
from fastmcp.utilities.tests import run_server_in_process

from langchain.mcp import MCPAdapter

TOKEN = "demo-weather-token"  # noqa: S105


def run_guarded_server(host: str, port: int) -> None:
    """提供一个天气服务器，该服务器仅响应携带 `TOKEN` 的调用方。"""
        # 仅用于演示——`StaticTokenVerifier` 将令牌以明文形式存储在内存中。实际部署应使用
    # `JWTVerifier` 配合 IdP 的 JWKS，或 OAuth（参见 `auth_oauth.py`）。
        # 部署应使用 `JWTVerifier` 配合 IdP 的 JWKS，或 OAuth（参见 `auth_oauth.py`）。
        # `auth` 接受一个持有者令牌字符串、字面量 "oauth"，或任何
    # `httpx2.Auth`。每个服务器的配置字典使用相同的键。
    mcp: FastMCP[None] = FastMCP(
        "weather",
        auth=StaticTokenVerifier(tokens={TOKEN: {"client_id": "demo", "scopes": ["read"]}}),
    )

    @mcp.tool
    def get_forecast(city: str) -> str:
        """报告某城市的天气预报。"""
        return f"{city}: 18C and clear."

    mcp.run(transport="http", host=host, port=port, show_banner=False, log_level="warning")


async def main() -> None:
    """使用和不使用令牌调用受保护的服务器。"""
    with run_server_in_process(run_guarded_server) as url:
                # `auth` 接受一个 bearer token 字符串、字面量 "oauth"，或任意
                # `httpx2.Auth`. 配置字典中每个服务器使用相同的键。
        async with MCPAdapter(Client(f"{url}/mcp", auth=TOKEN)) as adapter:
            [forecast] = await adapter.list_tools()
            [block] = await forecast.ainvoke({"city": "Oslo"})
            print("with token:   ", block["text"])

        try:
            async with MCPAdapter(f"{url}/mcp") as adapter:
                await adapter.list_tools()
        except Exception as exc:
            print("without token:", type(exc).__name__)


if __name__ == "__main__":
    asyncio.run(main())

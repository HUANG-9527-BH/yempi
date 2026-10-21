"""来自两个 MCP 协议时代的服务器，集成在一个智能体中。

MCP changed how a client and server agree on what each supports: the 2025-11-25
era negotiates with an `initialize` handshake, the 2026-07-28 era with
`server/discover` instead. FastMCP negotiates per connection, so tools from both
can end up in the same agent and nothing on this side has to know which is
which.

Note the shape: one adapter per server, not one `MCPConfig` fleet naming both.
A fleet is composed behind a single client, and that composite negotiates one
era for everything it holds. A fleet of modern servers does negotiate the modern
era — but add a backend that only speaks the handshake era and the whole fleet
drops to it. Separate adapters are what let each connection keep the best era
its own server supports.

    uv run examples/mcp/protocol_eras.py
"""

from __future__ import annotations

import asyncio
from typing import Any

from _servers import calculator_server, weather_server
from fastmcp.client import Client

from langchain.agents import create_agent
from langchain.mcp import MCPAdapter


async def main() -> None:
    """分别在每个时代发现工具，然后将所有工具交由一个智能体使用。"""
        # `mode` 选择协议时代。"legacy" 固定为握手时代；"auto"
    # 则协商服务器理解的最新版本。每个客户端独立进行协商。
        # 仅握手时代会填充 `initialize_result`，因此它也作为实际运行的协商方式的证明。
    legacy: Client[Any] = Client(weather_server(), mode="legacy")
    modern: Client[Any] = Client(calculator_server(), mode="auto")

    async with MCPAdapter(legacy) as legacy_adapter, MCPAdapter(modern) as modern_adapter:
        tools = await legacy_adapter.list_tools() + await modern_adapter.list_tools()

                # 只有握手阶段会填充 `initialize_result`，因此它也充当
                # 证明实际运行的协商过程。
        print(
            f"legacy server  -> {legacy.protocol_version} (handshake ran: "
            f"{legacy.initialize_result is not None})"
        )
        print(
            f"modern server  -> {modern.protocol_version} (handshake ran: "
            f"{modern.initialize_result is not None})"
        )

    print("tools:", [tool.name for tool in tools])

    agent = create_agent("anthropic:claude-sonnet-5", tools)
    result = await agent.ainvoke(
        {"messages": [{"role": "user", "content": "Forecast for Oslo? And what is 84 / 4?"}]}
    )
    print("\nagent:", result["messages"][-1].text)


if __name__ == "__main__":
    asyncio.run(main())

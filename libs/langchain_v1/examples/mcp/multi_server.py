"""多个 MCP 服务器通过一个适配器连接。

An `MCPConfig` dict names each backend. FastMCP connects to all of them and
prefixes every tool with its config key, so two servers exposing the same tool
name stay distinguishable in the list handed to a model.

每个后端独立访问，因此整个集群可以混合使用不同传输方式——这两个是 stdio，但任一都可为
URL。

    uv run examples/mcp/multi_server.py
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from langchain.agents import create_agent
from langchain.mcp import MCPAdapter

_STDIO_SERVER = str(Path(__file__).parent / "_stdio_server.py")

CONFIG = {
    "mcpServers": {
        "weather": {"command": sys.executable, "args": [_STDIO_SERVER, "weather"]},
        "calc": {"command": sys.executable, "args": [_STDIO_SERVER, "calculator"]},
    }
}


async def main() -> None:
    """让一个智能体同时拥有两个服务器的工具。"""
    async with MCPAdapter(CONFIG) as adapter:
        tools = await adapter.list_tools()

    print("tools:", [tool.name for tool in tools])

    agent = create_agent("anthropic:claude-sonnet-5", tools)
    result = await agent.ainvoke(
        {"messages": [{"role": "user", "content": "Forecast for Oslo? Also, what is 84 / 4?"}]}
    )
    print("\nagent:", result["messages"][-1].text)


if __name__ == "__main__":
    asyncio.run(main())

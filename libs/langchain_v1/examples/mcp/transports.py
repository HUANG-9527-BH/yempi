"""一个适配器，三种传输方式。

`MCPAdapter` infers the transport from whatever you hand it, so the only thing
that changes between an in-process server, a local script over stdio, and a
remote URL is the target itself.

    uv run examples/mcp/transports.py
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING

from _servers import run_weather_http, weather_server
from fastmcp.utilities.tests import run_server_in_process

from langchain.mcp import MCPAdapter

if TYPE_CHECKING:
    from langchain.mcp.adapter import MCPAdapterTarget

_STDIO_SERVER = Path(__file__).parent / "_stdio_server.py"


async def show(label: str, target: MCPAdapterTarget) -> None:
    """适配一个目标并调用它暴露的工具。"""
    async with MCPAdapter(target) as adapter:
        [forecast] = await adapter.list_tools()
                # MCP 结果以 LangChain 内容块的形式返回，而非原始字符串。
        [block] = await forecast.ainvoke({"city": "Oslo"})
        print(f"{label:12} {forecast.name} -> {block['text']}")


async def main() -> None:
    """依次通过每种传输方式连接同一服务器。"""
        # 进程内：无子进程，无套接字。适用于测试。
    await show("in-memory", weather_server())

        # 脚本路径通过 stdio 启动，每个适配器对应一个子进程。
    await show("stdio", _STDIO_SERVER)

        # URL 通过可流式传输的 HTTP 访问。FastMCP
    # 自身的测试辅助函数会在子进程中运行服务器，并返回其 URL。
        # 服务器在子进程中运行，并返回其 URL。
    with run_server_in_process(run_weather_http) as url:
        await show("http", f"{url}/mcp")


if __name__ == "__main__":
    asyncio.run(main())

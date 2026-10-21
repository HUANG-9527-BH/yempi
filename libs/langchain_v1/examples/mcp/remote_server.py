"""将适配器指向公共互联网上的真实MCP服务器。

DeepWiki answers questions about public GitHub repositories and serves MCP over
streamable HTTP with no auth. Nothing here is special-cased for it: the URL is
the entire configuration.

    uv run examples/mcp/remote_server.py
"""

from __future__ import annotations

import asyncio

from langchain.agents import create_agent
from langchain.mcp import MCPAdapter

DEEPWIKI = "https://mcp.deepwiki.com/mcp"


async def main() -> None:
    """让智能体通过远程MCP服务器研究一个仓库。"""
    async with MCPAdapter(DEEPWIKI) as adapter:
        tools = await adapter.list_tools()

    print("tools:", [tool.name for tool in tools])

    agent = create_agent(
        "anthropic:claude-sonnet-5",
        tools,
        system_prompt="Answer only from the deepwiki tools. Never answer from memory.",
    )
    question = "What transports does the client in the jlowin/fastmcp repo support?"
    result = await agent.ainvoke({"messages": [{"role": "user", "content": question}]})

        # 证明是远程服务器完成了工作，而非模型自行回忆。
    for message in result["messages"]:
        if message.type == "tool":
            print(f"\ncalled {message.name} -> {len(message.text)} chars from DeepWiki")

    print("\nagent:", result["messages"][-1].text)


if __name__ == "__main__":
    asyncio.run(main())

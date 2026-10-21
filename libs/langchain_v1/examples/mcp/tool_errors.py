"""一个失败的 MCP 工具会传递给模型，而不是结束运行。

When a server reports `isError=True`, the adapter turns it into a `ToolMessage`
with `status="error"` carrying the server's own message. The agent reads that
and can correct itself. Transport failures still raise, because a model cannot
act on those.

    uv run examples/mcp/tool_errors.py
"""

from __future__ import annotations

import asyncio

from _servers import calculator_server

from langchain.agents import create_agent
from langchain.mcp import MCPAdapter


async def main() -> None:
    """请求执行除以零操作，并观察智能体如何恢复。"""
    async with MCPAdapter(calculator_server()) as adapter:
        tools = await adapter.list_tools()

    agent = create_agent(
        "anthropic:claude-sonnet-5",
        tools,
                # 没有此设置，模型只会根据已知的算术知识直接回答，
                # 导致错误路径从未执行。
        system_prompt="Always use the `divide` tool for arithmetic. Never compute it yourself.",
    )
    result = await agent.ainvoke(
        {"messages": [{"role": "user", "content": "What is 10 divided by 0? Then try 10 / 4."}]}
    )

    for message in result["messages"]:
        if message.type == "tool":
            print(f"tool call -> status={message.status}: {message.text.strip()[:70]}")

    print("\nagent:", result["messages"][-1].text)


if __name__ == "__main__":
    asyncio.run(main())

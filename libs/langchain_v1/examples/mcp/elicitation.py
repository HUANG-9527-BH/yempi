"""一个在调用过程中需要输入的服务器，由人类回答。

某些 MCP 工具无法在不提问的情况下完成。`MCPAdapter` 将问题作为 LangGraph 的
`interrupt()` 抛出，这样正在审查智能体工作的人员可以回答问题，然后运行继续。

The adapter arms every client it builds to advertise the elicitation capability,
so this works without any opt-in — a server only asks a client that made the
promise on the wire, and the adapter makes it for you.

    uv run examples/mcp/elicitation.py
"""

from __future__ import annotations

import asyncio
from typing import Any

from _servers import booking_server
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from langchain.agents import create_agent
from langchain.mcp import MCPAdapter


async def main() -> None:
    """预订一张桌子，从这里回答服务器的问题。"""
    async with MCPAdapter(booking_server()) as adapter:
        tools = await adapter.list_tools()

        # 恢复需要持久化，因此中断的运行有地方等待。
    agent = create_agent("anthropic:claude-sonnet-5", tools, checkpointer=InMemorySaver())
    config: Any = {"configurable": {"thread_id": "booking-1"}}

    paused = await agent.ainvoke(
        {"messages": [{"role": "user", "content": "Book a table for 4."}]}, config
    )

    [interrupt] = paused["__interrupt__"]
    [question] = interrupt.value["requests"]
    print(f"server asks ({question['mode']}): {question['message']}")

        # 答案按服务器自身的请求键进行索引，因此无需在暂停期间跟踪任何内容。`decline` 或
    # `cancel` 会拒绝。
        # 则会拒绝。
    answer = {"action": "accept", "content": {"date": "2026-09-14"}}
    resumed = await agent.ainvoke(Command(resume={"responses": {question["key"]: answer}}), config)

    print("agent:", resumed["messages"][-1].text)


if __name__ == "__main__":
    asyncio.run(main())

"""用于测试的工具模拟中间件。"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any, Generic

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, ToolMessage

from langchain.agents.middleware.internal_call_transformer import (
    InternalCallTransformer,
    internal_call_metadata,
)
from langchain.agents.middleware.types import AgentMiddleware, AgentState, ContextT
from langchain.chat_models.base import init_chat_model

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from langgraph.types import Command

    from langchain.agents.middleware.types import ToolCallRequest
    from langchain.tools import BaseTool

_DEFAULT_EMULATOR_MODEL = "anthropic:claude-sonnet-4-5-20250929"


class LLMToolEmulator(AgentMiddleware[AgentState[Any], ContextT], Generic[ContextT]):
    """使用 LLM 模拟指定工具，而非实际执行它们。

    此中间件允许选择性地模拟工具以用于测试目的。

    By default (when `tools=None`), all tools are emulated. You can specify which
    tools to emulate by passing a list of tool names or `BaseTool` instances.

    Examples:
        !!! example "模拟所有工具（默认行为）"

            ```python
            from langchain.agents.middleware import LLMToolEmulator

            middleware = LLMToolEmulator()

            agent = create_agent(
                model="openai:gpt-5.5",
                tools=[get_weather, get_user_location, calculator],
                middleware=[middleware],
            )
            ```

        !!! example "通过名称模拟特定工具"

            ```python
            middleware = LLMToolEmulator(tools=["get_weather", "get_user_location"])
            ```

        !!! example "使用自定义模型进行模拟"

            ```python
            middleware = LLMToolEmulator(
                tools=["get_weather"], model="anthropic:claude-sonnet-4-5-20250929"
            )
            ```

        !!! example "通过传递工具实例模拟特定工具"

            ```python
            middleware = LLMToolEmulator(tools=[get_weather, get_user_location])
            ```
    """

    transformers = (InternalCallTransformer,)
    """将模拟模型调用的 token 从 `run.messages` 中排除。

    Registered only when this middleware is used — see
    `InternalCallTransformer` for why the call needs tagging and filtering.
    """

    def __init__(
        self,
        *,
        tools: list[str | BaseTool] | None = None,
        model: str | BaseChatModel | None = None,
    ) -> None:
        """初始化工具模拟器。

        Args:
            tools: 要模拟的工具名称（`str`）列表或 `BaseTool` 实例。

                如果为 `None`，则模拟所有工具。

                如果为空列表，则不模拟任何工具。
            model: 用于模拟的模型。

                默认值为 `'anthropic:claude-sonnet-4-5-20250929'`，需要安装
                `langchain-anthropic`。

                !!! warning "已弃用"
                依赖隐式默认值已被弃用，并将在未来版本中移除，因为它会使此中间件即使未明确指定也依赖于
                `langchain-anthropic`。请显式传入 `model`。

                可以是模型标识符字符串或 `BaseChatModel` 实例。
        """
        super().__init__()

                # 从工具中提取工具名称
                # None 表示模拟所有工具
        self.emulate_all = tools is None
        self.tools_to_emulate: set[str] = set()

        if tools is not None:
            for tool in tools:
                if isinstance(tool, str):
                    self.tools_to_emulate.add(tool)
                else:
                                        # 假设为具有 .name 属性的 BaseTool
                    self.tools_to_emulate.add(tool.name)

                # 初始化模拟模型
        if model is None:
            warnings.warn(
                "LLMToolEmulator's default model "
                f"({_DEFAULT_EMULATOR_MODEL!r}) is deprecated and will be removed "
                "in a future release. Pass `model` explicitly instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            try:
                self.model = init_chat_model(_DEFAULT_EMULATOR_MODEL, temperature=1)
            except ImportError as e:
                msg = (
                    "LLMToolEmulator's default model requires `langchain-anthropic` "
                    "to be installed. Install it with `pip install langchain-anthropic`, "
                    "or pass `model=...` explicitly to use a different provider."
                )
                raise ImportError(msg) from e
        elif isinstance(model, BaseChatModel):
            self.model = model
        else:
            self.model = init_chat_model(model, temperature=1)

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command[Any]],
    ) -> ToolMessage | Command[Any]:
        """使用 LLM 模拟工具执行（如果该工具应被模拟）。

        Args:
            request: 可能需要模拟的工具调用请求。
            handler: 执行工具的回调（可多次调用）。

        Returns:
            ToolMessage with emulated response if tool should be emulated,
                otherwise calls handler for normal execution.
        """
        tool_name = request.tool_call["name"]

                # 检查该工具是否应被模拟
        should_emulate = self.emulate_all or tool_name in self.tools_to_emulate

        if not should_emulate:
                        # 通过调用 handler 让其正常执行
            return handler(request)

                # 提取用于模拟的工具信息
        tool_args = request.tool_call["args"]
        tool_description = request.tool.description if request.tool else "No description available"

                # 构建模拟 LLM 的提示词
        prompt = (
            f"You are emulating a tool call for testing purposes.\n\n"
            f"Tool: {tool_name}\n"
            f"Description: {tool_description}\n"
            f"Arguments: {tool_args}\n\n"
            f"Generate a realistic response that this tool would return "
            f"given these arguments.\n"
            f"Return ONLY the tool's output, no explanation or preamble. "
            f"Introduce variation into your responses."
        )

                # 从 LLM 获取模拟响应
        response = self.model.invoke(
            [HumanMessage(prompt)],
            config={"metadata": {"lc_source": "tool_emulation", **internal_call_metadata()}},
        )

                # 短路：返回模拟结果而不执行真实工具
        return ToolMessage(
            content=response.content,
            tool_call_id=request.tool_call["id"],
            name=tool_name,
        )

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command[Any]]],
    ) -> ToolMessage | Command[Any]:
        """`wrap_tool_call` 的异步版本。

        使用 LLM 模拟工具执行（如果该工具应被模拟）。

        Args:
            request: 可能需要模拟的工具调用请求。
            handler: 异步回调以执行工具（可多次调用）。

        Returns:
            ToolMessage with emulated response if tool should be emulated,
                otherwise calls handler for normal execution.
        """
        tool_name = request.tool_call["name"]

                # 检查该工具是否应被模拟
        should_emulate = self.emulate_all or tool_name in self.tools_to_emulate

        if not should_emulate:
                        # 通过调用 handler 让其正常执行
            return await handler(request)

                # 提取用于模拟的工具信息
        tool_args = request.tool_call["args"]
        tool_description = request.tool.description if request.tool else "No description available"

                # 构建模拟 LLM 的提示词
        prompt = (
            f"You are emulating a tool call for testing purposes.\n\n"
            f"Tool: {tool_name}\n"
            f"Description: {tool_description}\n"
            f"Arguments: {tool_args}\n\n"
            f"Generate a realistic response that this tool would return "
            f"given these arguments.\n"
            f"Return ONLY the tool's output, no explanation or preamble. "
            f"Introduce variation into your responses."
        )

                # 从 LLM 获取模拟响应（使用异步调用）
        response = await self.model.ainvoke(
            [HumanMessage(prompt)],
            config={"metadata": {"lc_source": "tool_emulation", **internal_call_metadata()}},
        )

                # 短路：返回模拟结果而不执行真实工具
        return ToolMessage(
            content=response.content,
            tool_call_id=request.tool_call["id"],
            name=tool_name,
        )

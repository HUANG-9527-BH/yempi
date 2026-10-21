"""智能体的工具调用次数限制中间件。"""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Any, Literal

from langchain_core.messages import AIMessage, ToolCall, ToolMessage
from langgraph.channels.untracked_value import UntrackedValue
from langgraph.typing import ContextT
from typing_extensions import NotRequired, override

from langchain.agents.middleware.types import (
    AgentMiddleware,
    AgentState,
    PrivateStateAttr,
    ResponseT,
    hook_config,
)

if TYPE_CHECKING:
    from langgraph.runtime import Runtime

ExitBehavior = Literal["continue", "error", "end"]
"""How to handle execution when tool call limits are exceeded.

- `'continue'`: Block exceeded tools with error messages, let other tools continue
    (default)
- `'error'`: Raise a `ToolCallLimitExceededError` exception
- `'end'`: Stop execution immediately, injecting a `ToolMessage` for each tool call
    that exceeded the limit and an `AIMessage` explaining why. Any other pending tool
    call on the same `AIMessage` (e.g. from parallel tool calling) also gets an
    explanatory `ToolMessage` instead of being executed.
"""


class ToolCallLimitState(AgentState[ResponseT]):
    """`ToolCallLimitMiddleware` 的状态模式。

    继承 `AgentState` 并添加工具调用追踪字段。

    The count fields are dictionaries mapping tool names to execution counts. This
    allows multiple middleware instances to track different tools independently. The
    special key `'__all__'` is used for tracking all tool calls globally.

    类型参数：
        ResponseT: 结构化响应的类型。默认为 `Any`。
    """

    thread_tool_call_count: NotRequired[Annotated[dict[str, int], PrivateStateAttr]]
    run_tool_call_count: NotRequired[Annotated[dict[str, int], UntrackedValue, PrivateStateAttr]]


def _build_tool_message_content(tool_name: str | None) -> str:
    """当调用限制超出时，为 `ToolMessage` 构建错误消息内容。

    此消息会发送给模型，因此不应引用模型无概念的线程/运行概念。

    Args:
        tool_name: 被限制的工具名称（如果是特定工具），或 `None` 表示所有工具。

    Returns:
        一条简洁的消息，指示模型不要再调用该工具。
    """
        # 始终指示模型不要再调用，无论触发了哪个限制
    if tool_name:
        return f"Tool call limit exceeded. Do not call '{tool_name}' again."
    return "Tool call limit exceeded. Do not make additional tool calls."


def _build_final_ai_message_content(
    thread_count: int,
    run_count: int,
    thread_limit: int | None,
    run_limit: int | None,
    tool_name: str | None,
) -> str:
    """为 `'end'` 行为构建最终的 AI 消息内容。

    此消息会显示给用户，因此应包含关于哪些限制被超出的详细信息。

    Args:
        thread_count: 当前线程的工具调用次数。
        run_count: 当前运行的工具调用次数。
        thread_limit: 线程工具调用限制（如果设置）。
        run_limit: 运行工具调用限制（如果设置）。
        tool_name: 被限制的工具名称（如果是特定工具），或 `None` 表示所有工具。

    Returns:
        一条格式化的消息，描述了哪些限制被超出。
    """
    tool_desc = f"'{tool_name}' tool" if tool_name else "Tool"
    exceeded_limits = []

    if thread_limit is not None and thread_count > thread_limit:
        exceeded_limits.append(f"thread limit exceeded ({thread_count}/{thread_limit} calls)")
    if run_limit is not None and run_count > run_limit:
        exceeded_limits.append(f"run limit exceeded ({run_count}/{run_limit} calls)")

    limits_text = " and ".join(exceeded_limits)
    return f"{tool_desc} call limit reached: {limits_text}."


class ToolCallLimitExceededError(Exception):
    """当工具调用限制超出时抛出的异常。

    This exception is raised when the configured exit behavior is `'error'` and either
    the thread or run tool call limit has been exceeded.
    """

    def __init__(
        self,
        thread_count: int,
        run_count: int,
        thread_limit: int | None,
        run_limit: int | None,
        tool_name: str | None = None,
    ) -> None:
        """使用调用次数信息初始化异常。

        Args:
            thread_count: 当前线程的工具调用次数。
            run_count: 当前运行的工具调用次数。
            thread_limit: 线程工具调用限制（如果设置）。
            run_limit: 运行工具调用限制（如果设置）。
            tool_name: 被限制的工具名称（如果是特定工具），或 None 表示所有工具。
        """
        self.thread_count = thread_count
        self.run_count = run_count
        self.thread_limit = thread_limit
        self.run_limit = run_limit
        self.tool_name = tool_name

        msg = _build_final_ai_message_content(
            thread_count, run_count, thread_limit, run_limit, tool_name
        )
        super().__init__(msg)


class ToolCallLimitMiddleware(AgentMiddleware[ToolCallLimitState[ResponseT], ContextT, ResponseT]):
    """追踪工具调用次数，并在智能体执行期间强制实施限制。

    此中间件监控工具调用次数，并在限制超出时终止或限制执行。它支持线程级别（跨运行持久）和运行级别（每次调用独立）的调用计数。

    Configuration:
        - `exit_behavior`: How to handle when limits are exceeded
            - `'continue'`: Block exceeded tools, let execution continue (default)
            - `'error'`: Raise an exception
            - `'end'`: Stop immediately with a `ToolMessage` for each exceeded tool
                call, a `ToolMessage` explaining any other pending calls were skipped,
                and an AI message explaining why execution stopped

    Examples:
        !!! example "继续执行但阻止超出限制的工具调用（默认）"

            ```python
            from langchain.agents.middleware.tool_call_limit import ToolCallLimitMiddleware
            from langchain.agents import create_agent

            # Block exceeded tools but let other tools and model continue
            limiter = ToolCallLimitMiddleware(
                thread_limit=20,
                run_limit=10,
                exit_behavior="continue",  # default
            )

            agent = create_agent("openai:gpt-5.5", middleware=[limiter])
            ```

        !!! example "限制超出时立即停止"

            ```python
            # End execution immediately with an AI message
            limiter = ToolCallLimitMiddleware(run_limit=5, exit_behavior="end")

            agent = create_agent("openai:gpt-5.5", middleware=[limiter])
            ```

        !!! example "限制超出时抛出异常"

            ```python
            # Strict limit with exception handling
            limiter = ToolCallLimitMiddleware(
                tool_name="search", thread_limit=5, exit_behavior="error"
            )

            agent = create_agent("openai:gpt-5.5", middleware=[limiter])

            try:
                result = await agent.invoke({"messages": [HumanMessage("Task")]})
            except ToolCallLimitExceededError as e:
                print(f"Search limit exceeded: {e}")
            ```

    """

    state_schema = ToolCallLimitState  # type: ignore[assignment]

    def __init__(
        self,
        *,
        tool_name: str | None = None,
        thread_limit: int | None = None,
        run_limit: int | None = None,
        exit_behavior: ExitBehavior = "continue",
    ) -> None:
        """初始化工具调用次数限制中间件。

        Args:
            tool_name: 要限制的特定工具名称。如果为 `None`，则限制适用于所有工具。
            thread_limit: 每个线程允许的最大工具调用次数。`None` 表示无限制。
            run_limit: 每次运行允许的最大工具调用次数。`None` 表示无限制。
            exit_behavior: 限制超出时的处理方式。

                - `'continue'`: Block exceeded tools with error messages, let other
                    tools continue. Model decides when to end.
                - `'error'`: Raise a `ToolCallLimitExceededError` exception
                - `'end'`: Stop execution immediately with a `ToolMessage` for each
                    tool call that exceeded the limit and an AI message explaining
                    why. Any other pending tool call on the same `AIMessage` (e.g.
                    from parallel tool calling) is given an explanatory
                    `ToolMessage` instead of being executed.

        Raises:
            ValueError: If both limits are `None`, if `exit_behavior` is invalid,
                or if `run_limit` exceeds `thread_limit`.
        """
        super().__init__()

        if thread_limit is None and run_limit is None:
            msg = "At least one limit must be specified (thread_limit or run_limit)"
            raise ValueError(msg)

        valid_behaviors = ("continue", "error", "end")
        if exit_behavior not in valid_behaviors:
            msg = f"Invalid exit_behavior: {exit_behavior!r}. Must be one of {valid_behaviors}"
            raise ValueError(msg)

        if thread_limit is not None and run_limit is not None and run_limit > thread_limit:
            msg = (
                f"run_limit ({run_limit}) cannot exceed thread_limit ({thread_limit}). "
                "The run limit should be less than or equal to the thread limit."
            )
            raise ValueError(msg)

        self.tool_name = tool_name
        self.thread_limit = thread_limit
        self.run_limit = run_limit
        self.exit_behavior = exit_behavior

    @property
    def name(self) -> str:
        """中间件实例的名称。

        Includes the tool name if specified to allow multiple instances
        of this middleware with different tool names.
        """
        base_name = self.__class__.__name__
        if self.tool_name:
            return f"{base_name}[{self.tool_name}]"
        return base_name

    def _would_exceed_limit(self, thread_count: int, run_count: int) -> bool:
        """检查增加计数是否会超过任何已配置的限制。

        Args:
            thread_count: 当前线程调用计数。
            run_count: 当前运行调用计数。

        Returns:
            如果任一限制在多一次调用后会被超过，则返回 True。
        """
        return (self.thread_limit is not None and thread_count + 1 > self.thread_limit) or (
            self.run_limit is not None and run_count + 1 > self.run_limit
        )

    def _matches_tool_filter(self, tool_call: ToolCall) -> bool:
        """检查工具调用是否匹配此中间件的工具过滤器。

        Args:
            tool_call: 要检查的工具调用。

        Returns:
            如果此中间件应跟踪该工具调用，则返回 True。
        """
        return self.tool_name is None or tool_call["name"] == self.tool_name

    def _separate_tool_calls(
        self, tool_calls: list[ToolCall], thread_count: int, run_count: int
    ) -> tuple[list[ToolCall], list[ToolCall], int, int]:
        """根据限制将工具调用分为允许和阻塞两类。

        Args:
            tool_calls: 待评估的工具调用列表。
            thread_count: 当前线程调用计数。
            run_count: 当前运行调用计数。

        Returns:
            返回元组 `(allowed_calls, blocked_calls, final_thread_count, final_run_count)`。
        """
        allowed_calls: list[ToolCall] = []
        blocked_calls: list[ToolCall] = []
        temp_thread_count = thread_count
        temp_run_count = run_count

        for tool_call in tool_calls:
            if not self._matches_tool_filter(tool_call):
                continue

            if self._would_exceed_limit(temp_thread_count, temp_run_count):
                blocked_calls.append(tool_call)
            else:
                allowed_calls.append(tool_call)
                temp_thread_count += 1
                temp_run_count += 1

        return allowed_calls, blocked_calls, temp_thread_count, temp_run_count

    @hook_config(can_jump_to=["end"])
    @override
    def after_model(
        self,
        state: ToolCallLimitState[ResponseT],
        runtime: Runtime[ContextT],
    ) -> dict[str, Any] | None:
        """在模型调用后增加工具调用计数并检查限制。

        Args:
            state: 当前智能体状态。
            runtime: langgraph 运行时。

        Returns:
            State updates with incremented tool call counts. If limits are exceeded
                and exit_behavior is `'end'`, also includes a jump to end with a
                `ToolMessage` for each exceeded tool call, an explanatory
                `ToolMessage` for any other pending tool calls, and an AI message.

        Raises:
            ToolCallLimitExceededError: 如果限制被超过且 `exit_behavior` 为 `'error'`
            时抛出。
        """
                # 获取最后一条 AIMessage 以检查工具调用
        messages = state.get("messages", [])
        if not messages:
            return None

                # 查找最后一条 AIMessage
        last_ai_message = None
        for message in reversed(messages):
            if isinstance(message, AIMessage):
                last_ai_message = message
                break

        if not last_ai_message or not last_ai_message.tool_calls:
            return None

                # 获取此中间件实例的计数键
        count_key = self.tool_name or "__all__"

                # 获取当前计数
        thread_counts = state.get("thread_tool_call_count", {}).copy()
        run_counts = state.get("run_tool_call_count", {}).copy()
        current_thread_count = thread_counts.get(count_key, 0)
        current_run_count = run_counts.get(count_key, 0)

                # 将工具调用分为允许和被阻止的两类
        allowed_calls, blocked_calls, new_thread_count, new_run_count = self._separate_tool_calls(
            last_ai_message.tool_calls, current_thread_count, current_run_count
        )

                # 更新计数，仅包含允许的调用以用于线程计数
                # (被阻止的调用不计入线程级跟踪)
                # 但运行计数包含被阻止的调用，因为它们在本次运行中已被尝试
        thread_counts[count_key] = new_thread_count
        run_counts[count_key] = new_run_count + len(blocked_calls)

                # 如果没有工具调用被阻止，直接更新计数
        if not blocked_calls:
            if allowed_calls:
                return {
                    "thread_tool_call_count": thread_counts,
                    "run_tool_call_count": run_counts,
                }
            return None

                # 获取最终计数以构建消息
        final_thread_count = thread_counts[count_key]
        final_run_count = run_counts[count_key]

                # 处理不同的 exit_behavior
        if self.exit_behavior == "error":
                        # 使用假设的线程计数来显示哪个限制被超出
            hypothetical_thread_count = final_thread_count + len(blocked_calls)
            raise ToolCallLimitExceededError(
                thread_count=hypothetical_thread_count,
                run_count=final_run_count,
                thread_limit=self.thread_limit,
                run_limit=self.run_limit,
                tool_name=self.tool_name,
            )

                # 构建工具消息内容（发送给模型 - 不包含线程/运行细节）
        tool_msg_content = _build_tool_message_content(self.tool_name)

                # 为被阻止的工具调用注入人工错误的 ToolMessages
        artificial_messages: list[ToolMessage | AIMessage] = [
            ToolMessage(
                content=tool_msg_content,
                tool_call_id=tool_call["id"],
                name=tool_call.get("name"),
                status="error",
            )
            for tool_call in blocked_calls
        ]

        if self.exit_behavior == "end":
                        # 由于跳转到 `"end"` 会跳过工具执行，因此为所有待处理调用添加 `ToolMessage`
                        # 以确保没有调用被执行，且消息历史保持有效。
            blocked_ids = {tool_call["id"] for tool_call in blocked_calls}
            pending_tool_calls = [
                tc for tc in last_ai_message.tool_calls if tc["id"] not in blocked_ids
            ]
            artificial_messages.extend(
                ToolMessage(
                    content=(
                        "Execution stopped before this tool call could run because "
                        "another tool call in the same batch exceeded the limit."
                    ),
                    tool_call_id=tool_call["id"],
                    name=tool_call.get("name"),
                    status="error",
                )
                for tool_call in pending_tool_calls
            )

                        # 构建最终 AI 消息内容（显示给用户 - 包含线程/运行细节）
                        # 使用假设的线程计数（如果调用未被阻止，它本应是多少）
                        # 来显示实际被超出的限制
            hypothetical_thread_count = final_thread_count + len(blocked_calls)
            final_msg_content = _build_final_ai_message_content(
                hypothetical_thread_count,
                final_run_count,
                self.thread_limit,
                self.run_limit,
                self.tool_name,
            )
            artificial_messages.append(AIMessage(content=final_msg_content))

                        # 由于跳转到 `"end"` 后不再执行任何调用，因此不将此批调用计入
                        # 线程级计数。只有运行级计数记录已尝试的调用。
            thread_counts[count_key] = current_thread_count

            return {
                "thread_tool_call_count": thread_counts,
                "run_tool_call_count": run_counts,
                "jump_to": "end",
                "messages": artificial_messages,
            }

                # 对于 exit_behavior="continue"，返回错误消息以阻止超出限制的工具
        return {
            "thread_tool_call_count": thread_counts,
            "run_tool_call_count": run_counts,
            "messages": artificial_messages,
        }

    @hook_config(can_jump_to=["end"])
    async def aafter_model(
        self,
        state: ToolCallLimitState[ResponseT],
        runtime: Runtime[ContextT],
    ) -> dict[str, Any] | None:
        """异步在模型调用后增加工具调用计数并检查限制。

        Args:
            state: 当前智能体状态。
            runtime: langgraph 运行时。

        Returns:
            State updates with incremented tool call counts. If limits are exceeded
                and exit_behavior is `'end'`, also includes a jump to end with a
                `ToolMessage` for each exceeded tool call, an explanatory
                `ToolMessage` for any other pending tool calls, and an AI message.

        Raises:
            ToolCallLimitExceededError：如果超出限制且 `exit_behavior` 为 `'error'`
            时抛出。
        """
        return self.after_model(state, runtime)

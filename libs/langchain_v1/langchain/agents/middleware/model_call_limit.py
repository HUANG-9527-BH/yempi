"""智能体的调用追踪中间件。"""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Any, Literal

from langchain_core.messages import AIMessage
from langgraph.channels.untracked_value import UntrackedValue
from typing_extensions import NotRequired, override

from langchain.agents.middleware.types import (
    AgentMiddleware,
    AgentState,
    ContextT,
    PrivateStateAttr,
    ResponseT,
    hook_config,
)

if TYPE_CHECKING:
    from langgraph.runtime import Runtime


class ModelCallLimitState(AgentState[ResponseT]):
    """ModelCallLimitMiddleware 的状态模式。

    在 `AgentState` 上扩展模型调用追踪字段。

    类型参数:
        ResponseT: 结构化响应的类型。默认为 `Any`。
    """

    thread_model_call_count: NotRequired[Annotated[int, PrivateStateAttr]]
    run_model_call_count: NotRequired[Annotated[int, UntrackedValue, PrivateStateAttr]]


def _build_limit_exceeded_message(
    thread_count: int,
    run_count: int,
    thread_limit: int | None,
    run_limit: int | None,
) -> str:
    """构建一条指示哪些限制被超过的消息。

    Args:
        thread_count: 当前线程模型调用次数。
        run_count: 当前运行模型调用次数。
        thread_limit: 线程模型调用限制（如果已设置）。
        run_limit: 运行模型调用限制（如果已设置）。

    Returns:
        一条格式化的消息，描述了哪些限制被超过。
    """
    exceeded_limits = []
    if thread_limit is not None and thread_count >= thread_limit:
        exceeded_limits.append(f"thread limit ({thread_count}/{thread_limit})")
    if run_limit is not None and run_count >= run_limit:
        exceeded_limits.append(f"run limit ({run_count}/{run_limit})")

    return f"Model call limits exceeded: {', '.join(exceeded_limits)}"


class ModelCallLimitExceededError(Exception):
    """当模型调用限制被超过时抛出的异常。

    This exception is raised when the configured exit behavior is `'error'` and either
    the thread or run model call limit has been exceeded.
    """

    def __init__(
        self,
        thread_count: int,
        run_count: int,
        thread_limit: int | None,
        run_limit: int | None,
    ) -> None:
        """使用调用次数信息初始化异常。

        Args:
            thread_count: 当前线程模型调用次数。
            run_count: 当前运行模型调用次数。
            thread_limit: 线程模型调用限制（如果已设置）。
            run_limit: 运行模型调用限制（如果已设置）。
        """
        self.thread_count = thread_count
        self.run_count = run_count
        self.thread_limit = thread_limit
        self.run_limit = run_limit

        msg = _build_limit_exceeded_message(thread_count, run_count, thread_limit, run_limit)
        super().__init__(msg)


class ModelCallLimitMiddleware(
    AgentMiddleware[ModelCallLimitState[ResponseT], ContextT, ResponseT]
):
    """追踪模型调用次数并强制执行限制。

    This middleware monitors the number of model calls made during agent execution
    and can terminate the agent when specified limits are reached. It supports
    both thread-level and run-level call counting with configurable exit behaviors.

    线程级：中间件跟踪模型调用次数，并在智能体的多次运行（调用）之间持久化调用计数。

    运行级：中间件跟踪单次智能体运行（调用）期间发起的模型调用次数。

    Example:
        ```python
        from langchain.agents.middleware import ModelCallLimitMiddleware
        from langchain.agents import create_agent

        # Create middleware with limits
        call_tracker = ModelCallLimitMiddleware(thread_limit=10, run_limit=5, exit_behavior="end")

        agent = create_agent("openai:gpt-5.5", middleware=[call_tracker])

        # Agent will automatically jump to end when limits are exceeded
        result = await agent.invoke({"messages": [HumanMessage("Help me with a task")]})
        ```
    """

    state_schema = ModelCallLimitState  # type: ignore[assignment]

    def __init__(
        self,
        *,
        thread_limit: int | None = None,
        run_limit: int | None = None,
        exit_behavior: Literal["end", "error"] = "end",
    ) -> None:
        """初始化调用追踪中间件。

        Args:
            thread_limit: 每个线程允许的最大模型调用次数。

                `None` 表示无限制。
            run_limit: 每次运行允许的最大模型调用次数。

                `None` 表示无限制。
            exit_behavior: 限制被超过时的处理方式。

                - `'end'`: 跳转到智能体执行末尾，并 注入一条人工 AI
                消息，表明限制已被超过。 - `'error'`: 抛出一个
                `ModelCallLimitExceededError`

        Raises:
            ValueError: 如果两个限制均为 `None`，或 `exit_behavior` 无效。
        """
        super().__init__()

        if thread_limit is None and run_limit is None:
            msg = "At least one limit must be specified (thread_limit or run_limit)"
            raise ValueError(msg)

        if exit_behavior not in {"end", "error"}:
            msg = f"Invalid exit_behavior: {exit_behavior}. Must be 'end' or 'error'"
            raise ValueError(msg)

        self.thread_limit = thread_limit
        self.run_limit = run_limit
        self.exit_behavior = exit_behavior

    @hook_config(can_jump_to=["end"])
    @override
    def before_model(
        self, state: ModelCallLimitState[ResponseT], runtime: Runtime[ContextT]
    ) -> dict[str, Any] | None:
        """在发起模型调用前检查模型调用限制。

        Args:
            state: 包含调用次数的当前智能体状态。
            runtime: langgraph 运行时。

        Returns:
            If limits are exceeded and exit_behavior is `'end'`, returns
                a `Command` to jump to the end with a limit exceeded message. Otherwise
                returns `None`.

        Raises:
            ModelCallLimitExceededError: 如果限制被超过且 `exit_behavior`
                为 `'error'`。
        """
        thread_count = state.get("thread_model_call_count", 0)
        run_count = state.get("run_model_call_count", 0)

                # 检查下一次调用是否会超出任何限制
        thread_limit_exceeded = self.thread_limit is not None and thread_count >= self.thread_limit
        run_limit_exceeded = self.run_limit is not None and run_count >= self.run_limit

        if thread_limit_exceeded or run_limit_exceeded:
            if self.exit_behavior == "error":
                raise ModelCallLimitExceededError(
                    thread_count=thread_count,
                    run_count=run_count,
                    thread_limit=self.thread_limit,
                    run_limit=self.run_limit,
                )
            if self.exit_behavior == "end":
                                # 创建一条表示限制已超出的消息
                limit_message = _build_limit_exceeded_message(
                    thread_count, run_count, self.thread_limit, self.run_limit
                )
                limit_ai_message = AIMessage(content=limit_message)

                return {"jump_to": "end", "messages": [limit_ai_message]}

        return None

    @hook_config(can_jump_to=["end"])
    async def abefore_model(
        self,
        state: ModelCallLimitState[ResponseT],
        runtime: Runtime[ContextT],
    ) -> dict[str, Any] | None:
        """异步检查模型调用限制，再发起模型调用。

        Args:
            state: 包含调用次数的当前智能体状态。
            runtime: langgraph 运行时。

        Returns:
            If limits are exceeded and exit_behavior is `'end'`, returns
                a `Command` to jump to the end with a limit exceeded message. Otherwise
                returns `None`.

        Raises:
            ModelCallLimitExceededError: 如果超出限制且 `exit_behavior` 为 `'error'`。
        """
        return self.before_model(state, runtime)

    @override
    def after_model(
        self, state: ModelCallLimitState[ResponseT], runtime: Runtime[ContextT]
    ) -> dict[str, Any] | None:
        """模型调用后增加模型调用计数。

        Args:
            state: 当前智能体状态。
            runtime: langgraph 运行时。

        Returns:
            状态更新，包含递增后的调用计数。
        """
        return {
            "thread_model_call_count": state.get("thread_model_call_count", 0) + 1,
            "run_model_call_count": state.get("run_model_call_count", 0) + 1,
        }

    async def aafter_model(
        self,
        state: ModelCallLimitState[ResponseT],
        runtime: Runtime[ContextT],
    ) -> dict[str, Any] | None:
        """异步在模型调用后增加模型调用计数。

        Args:
            state: 当前智能体状态。
            runtime: langgraph 运行时。

        Returns:
            状态更新，包含递增后的调用计数。
        """
        return self.after_model(state, runtime)

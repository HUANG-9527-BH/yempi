"""智能体的任务规划与管理中间件。"""

from collections.abc import Awaitable, Callable
from typing import Annotated, Any, Literal, cast

from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from langchain_core.tools import InjectedToolCallId, StructuredTool, tool
from langgraph.runtime import Runtime
from langgraph.types import Command
from pydantic import BaseModel
from typing_extensions import NotRequired, TypedDict, override

from langchain.agents.middleware.types import (
    AgentMiddleware,
    AgentState,
    ContextT,
    ModelRequest,
    ModelResponse,
    OmitFromInput,
    ResponseT,
)
from langchain.tools import ToolRuntime


class Todo(TypedDict):
    """一个包含内容和状态的待办事项。"""

    content: str
    """待办事项的内容/描述。"""

    status: Literal["pending", "in_progress", "completed"]
    """待办事项的当前状态。"""


class PlanningState(AgentState[ResponseT]):
    """待办事项中间件的状态模式。

    类型参数:
        ResponseT: 结构化响应的类型。默认为 `Any`。
    """

    todos: Annotated[NotRequired[list[Todo]], OmitFromInput]
    """用于跟踪任务进度的待办事项列表。"""


class WriteTodosInput(BaseModel):
    """`write_todos` 工具的输入模式。"""

    todos: list[Todo]


WRITE_TODOS_TOOL_DESCRIPTION = """Use this tool to create and manage a structured task list for your current work session. This helps you track progress and organize complex tasks.

Only use this tool if you think it will be helpful in staying organized. If the user's request is trivial and takes less than 3 steps, it is better to NOT use this tool and just do the task directly.

## When to Use This Tool

Use this tool in these scenarios:

1. Complex multi-step tasks - When a task requires 3 or more distinct steps or actions
2. Non-trivial and complex tasks - Tasks that require careful planning or multiple operations
3. User explicitly requests todo list - When the user directly asks you to use the todo list
4. User provides multiple tasks - When users provide a list of things to be done (numbered or comma-separated)
5. The plan may need future revisions or updates based on results from the first few steps

## How to Use This Tool

1. When you start working on a task - Mark it as in_progress BEFORE beginning work.
2. After completing a task - Mark it as completed and add any new follow-up tasks discovered during implementation.
3. You can also update future tasks, such as deleting them if they are no longer necessary, or adding new tasks that are necessary. Don't change previously completed tasks.
4. You can make several updates to the todo list at once. For example, when you complete a task, you can mark the next task you need to start as in_progress.

## When NOT to Use This Tool

It is important to skip using this tool when:
1. There is only a single, straightforward task
2. The task is trivial and tracking it provides no benefit
3. The task can be completed in less than 3 trivial steps
4. The task is purely conversational or informational

## Task States and Management

1. **Task States**: Use these states to track progress:
    - pending: Task not yet started
    - in_progress: Currently working on (you can have multiple tasks in_progress at a time if they are not related to each other and can be run in parallel)
    - completed: Task finished successfully

2. **Task Management**:
    - Update task status in real-time as you work
    - Mark tasks complete IMMEDIATELY after finishing (don't batch completions)
    - Complete current tasks before starting new ones
    - Remove tasks that are no longer relevant from the list entirely
    - IMPORTANT: When you write this todo list, you should mark your first task (or tasks) as in_progress immediately!.
    - IMPORTANT: Unless all tasks are completed, you should always have at least one task in_progress.

3. **Task Completion Requirements**:
    - ONLY mark a task as completed when you have FULLY accomplished it
    - If you encounter errors, blockers, or cannot finish, keep the task as in_progress
    - When blocked, create a new task describing what needs to be resolved
    - Never mark a task as completed if:
        - There are unresolved issues or errors
        - Work is partial or incomplete
        - You encountered blockers that prevent completion
        - You couldn't find necessary resources or dependencies
        - Quality standards haven't been met

4. **Task Breakdown**:
    - Create specific, actionable items
    - Break complex tasks into smaller, manageable steps
    - Use clear, descriptive task names

Being proactive with task management ensures you complete all requirements successfully
Remember: If you only need to make a few tool calls to complete a task, and it is clear what you need to do, it is better to just do the task directly and NOT call this tool at all.

## When You Finish

`write_todos` tracks your work; it does not deliver the answer. Whatever the user asked for — computations, summaries, comparisons, data — must appear as text content in a message after your final `write_todos` call. Marking the last todo complete is not itself an answer to the user."""  # noqa: E501

WRITE_TODOS_SYSTEM_PROMPT = """## `write_todos`

You have access to the `write_todos` tool to help you manage and plan complex objectives.
Use this tool for complex objectives to ensure that you are tracking each necessary step.
This tool is very helpful for planning complex objectives, and for breaking down these larger complex objectives into smaller steps.

It is critical that you mark todos as completed as soon as you are done with a step. Do not batch up multiple steps before marking them as completed.
For simple objectives that only require a few steps, it is better to just complete the objective directly and NOT use this tool.
Writing todos takes time and tokens, use it when it is helpful for managing complex many-step problems! But not for simple few-step requests.

## Important To-Do List Usage Notes to Remember

- The `write_todos` tool should never be called multiple times in parallel.
- Don't be afraid to revise the To-Do list as you go. New information may reveal new tasks that need to be done, or old tasks that are irrelevant.

## Finishing a task

When you finish all work, write your final answer in the message AFTER your last `write_todos` call — not in the same turn as that call. Start the final message with the substantive content the user asked for — the data, computation, summary, or analysis. The user wants the result, not confirmation that the work is done."""  # noqa: E501


@tool(description=WRITE_TODOS_TOOL_DESCRIPTION)
def write_todos(
    todos: list[Todo], tool_call_id: Annotated[str, InjectedToolCallId]
) -> Command[Any]:
    """为当前工作会话创建并管理结构化的任务列表。"""
    return Command(
        update={
            "todos": todos,
            "messages": [ToolMessage(f"Updated todo list to {todos}", tool_call_id=tool_call_id)],
        }
    )


# 动态创建带有自定义描述的 write_todos 工具
def _write_todos(
    runtime: ToolRuntime[ContextT, PlanningState[ResponseT]], todos: list[Todo]
) -> Command[Any]:
    """为当前工作会话创建并管理结构化的任务列表。"""
    return Command(
        update={
            "todos": todos,
            "messages": [
                ToolMessage(f"Updated todo list to {todos}", tool_call_id=runtime.tool_call_id)
            ],
        }
    )


async def _awrite_todos(
    runtime: ToolRuntime[ContextT, PlanningState[ResponseT]], todos: list[Todo]
) -> Command[Any]:
    """为当前工作会话创建并管理结构化的任务列表。"""
    return _write_todos(runtime, todos)


class TodoListMiddleware(AgentMiddleware[PlanningState[ResponseT], ContextT, ResponseT]):
    """为智能体提供待办事项列表管理功能的中间件。

    This middleware adds a `write_todos` tool that allows agents to create and manage
    structured task lists for complex multi-step operations. It's designed to help
    agents track progress, organize complex tasks, and provide users with visibility
    into task completion status.

    中间件会自动注入系统提示，指导智能体如何有效使用待办事项功能。同时强制要求每个模型回合中最多调用一次
    `write_todos`
    工具，因为该工具会替换整个待办事项列表，多次并行调用会导致优先级不明确。

    Example:
        ```python
        from langchain.agents.middleware import TodoListMiddleware
        from langchain.agents import create_agent

        agent = create_agent("openai:gpt-5.5", middleware=[TodoListMiddleware()])

        # Agent now has access to write_todos tool and todo state tracking
        result = await agent.invoke({"messages": [HumanMessage("Help me refactor my codebase")]})

        print(result["todos"])  # Array of todo items with status tracking
        ```
    """

    state_schema = PlanningState  # type: ignore[assignment]

    def __init__(
        self,
        *,
        system_prompt: str = WRITE_TODOS_SYSTEM_PROMPT,
        tool_description: str = WRITE_TODOS_TOOL_DESCRIPTION,
    ) -> None:
        """使用可选的自定义提示初始化 `TodoListMiddleware`。

        Args:
            system_prompt: Custom system prompt to guide the agent on using the todo
                tool.
            tool_description: Custom description for the `write_todos` tool.
        """
        super().__init__()
        self.system_prompt = system_prompt
        self.tool_description = tool_description

        self.tools = [
            StructuredTool.from_function(
                name="write_todos",
                description=tool_description,
                func=_write_todos,
                coroutine=_awrite_todos,
                args_schema=WriteTodosInput,
                infer_schema=False,
            )
        ]

    def wrap_model_call(
        self,
        request: ModelRequest[ContextT],
        handler: Callable[[ModelRequest[ContextT]], ModelResponse[ResponseT]],
    ) -> ModelResponse[ResponseT] | AIMessage:
        """更新系统消息以包含待办事项系统提示。

        Args:
            request: 要执行的模型请求（包含状态和运行时）。
            handler: 异步回调，执行模型请求并返回
                `ModelResponse`。

        Returns:
            模型调用结果。
        """
        if request.system_message is not None:
            new_system_content = [
                *request.system_message.content_blocks,
                {"type": "text", "text": f"\n\n{self.system_prompt}"},
            ]
        else:
            new_system_content = [{"type": "text", "text": self.system_prompt}]
        new_system_message = SystemMessage(
            content=cast("list[str | dict[str, str]]", new_system_content)
        )
        return handler(request.override(system_message=new_system_message))

    async def awrap_model_call(
        self,
        request: ModelRequest[ContextT],
        handler: Callable[[ModelRequest[ContextT]], Awaitable[ModelResponse[ResponseT]]],
    ) -> ModelResponse[ResponseT] | AIMessage:
        """更新系统消息以包含待办事项系统提示。

        Args:
            request: 要执行的模型请求（包含状态和运行时）。
            handler: 异步回调，执行模型请求并返回
                `ModelResponse`。

        Returns:
            模型调用结果。
        """
        if request.system_message is not None:
            new_system_content = [
                *request.system_message.content_blocks,
                {"type": "text", "text": f"\n\n{self.system_prompt}"},
            ]
        else:
            new_system_content = [{"type": "text", "text": self.system_prompt}]
        new_system_message = SystemMessage(
            content=cast("list[str | dict[str, str]]", new_system_content)
        )
        return await handler(request.override(system_message=new_system_message))

    @override
    def after_model(
        self, state: PlanningState[ResponseT], runtime: Runtime[ContextT]
    ) -> dict[str, Any] | None:
        """检查是否存在并行的 write_todos 工具调用，若检测到则返回错误。

        The todo list is designed to be updated at most once per model turn. Since
        the `write_todos` tool replaces the entire todo list with each call, making
        multiple parallel calls would create ambiguity about which update should take
        precedence. This method prevents such conflicts by rejecting any response that
        contains multiple write_todos tool calls.

        Args:
            state: 包含消息的当前智能体状态。
            runtime: LangGraph 运行时实例。

        Returns:
            A dict containing error ToolMessages for each write_todos call if multiple
            parallel calls are detected, otherwise None to allow normal execution.
        """
        messages = state["messages"]
        if not messages:
            return None

        last_ai_msg = next((msg for msg in reversed(messages) if isinstance(msg, AIMessage)), None)
        if not last_ai_msg or not last_ai_msg.tool_calls:
            return None

                # 统计 write_todos 工具调用次数
        write_todos_calls = [tc for tc in last_ai_msg.tool_calls if tc["name"] == "write_todos"]

        if len(write_todos_calls) > 1:
                        # 为所有 write_todos 调用创建错误工具消息
            error_messages = [
                ToolMessage(
                    content=(
                        "Error: The `write_todos` tool should never be called multiple times "
                        "in parallel. Please call it only once per model invocation to update "
                        "the todo list."
                    ),
                    tool_call_id=tc["id"],
                    status="error",
                )
                for tc in write_todos_calls
            ]

                        # 保留 AI 消息中的工具调用，但返回错误消息
                        # 此模式与 HumanInTheLoopMiddleware 保持一致
            return {"messages": error_messages}

        return None

    @override
    async def aafter_model(
        self, state: PlanningState[ResponseT], runtime: Runtime[ContextT]
    ) -> dict[str, Any] | None:
        """检查是否存在并行的 write_todos 工具调用，若检测到则返回错误。

        Async version of `after_model`. The todo list is designed to be updated at
        most once per model turn. Since the `write_todos` tool replaces the entire
        todo list with each call, making multiple parallel calls would create ambiguity
        about which update should take precedence. This method prevents such conflicts
        by rejecting any response that contains multiple write_todos tool calls.

        Args:
            state: 包含消息的当前智能体状态。
            runtime: LangGraph 运行时实例。

        Returns:
            A dict containing error ToolMessages for each write_todos call if multiple
            parallel calls are detected, otherwise None to allow normal execution.
        """
        return self.after_model(state, runtime)

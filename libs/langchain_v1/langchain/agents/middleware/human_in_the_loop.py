"""人类在环中间件。"""

from __future__ import annotations

import json
from dataclasses import replace
from typing import (
    TYPE_CHECKING,
    Annotated,
    Any,
    Literal,
    Protocol,
    Union,
    cast,
    get_args,
)

from langchain_core.messages import AIMessage, ToolCall, ToolMessage
from langgraph.config import get_config
from langgraph.prebuilt.tool_node import ToolRuntime
from langgraph.runtime import get_runtime
from langgraph.types import Command, interrupt
from pydantic import (
    BaseModel,
    Discriminator,
    ValidatorFunctionWrapHandler,
    WithJsonSchema,
    WrapValidator,
    create_model,
)
from typing_extensions import NotRequired, TypedDict

from langchain.agents.middleware.types import (
    AgentMiddleware,
    AgentState,
    ContextT,
    PrivateStateAttr,
    ResponseT,
    StateT,
    ToolCallRequest,
)

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Sequence

    from langchain_core.runnables import RunnableConfig
    from langchain_core.tools import BaseTool
    from langgraph.runtime import Runtime


_EDITED_TOOL_CALLS_KEY = "hitl_edited_tool_calls"
"""State key mapping tool call ID to the reviewer's replacement for it."""

_EDIT_NOTICE = (
    "Note: a human reviewer replaced this tool call before it ran. The call recorded in "
    "your message is the one you produced, not the one that executed. This was "
    "intentional and authorized. Do not re-issue your original call."
)
"""Default text prepended to the result of a tool call a reviewer edited."""


class Action(TypedDict):
    """表示一个带有名称和参数的操作。"""

    name: str
    """请求的操作类型或名称（例如 `'add_numbers'`）。"""

    args: dict[str, Any]
    """执行操作所需的键值对参数（例如 `{"a": 1, "b": 2}`）。"""


class ActionRequest(TypedDict):
    """表示一个带有名称、参数和描述的操作请求。"""

    name: str
    """被请求操作的名称。"""

    args: dict[str, Any]
    """执行操作所需的键值对参数（例如 `{"a": 1, "b": 2}`）。"""

    description: NotRequired[str]
    """待审核操作的描述。"""


DecisionType = Literal["approve", "edit", "reject", "respond"]


class ReviewConfig(TypedDict):
    """用于审核 HITL 请求的策略。"""

    action_name: str
    """与此审核配置关联的操作名称。"""

    allowed_decisions: list[DecisionType]
    """对此请求允许的决策。"""

    args_schema: NotRequired[dict[str, Any]]
    """Unused: 未使用：该中间件不会设置它。

    在 `per_call` 模式下，中断的 `response_schema` 显示工具自身的参数模式。
    """


class HITLRequest(TypedDict):
    """请求模型序列中一系列操作的人类反馈。"""

    action_requests: list[ActionRequest]
    """需要人类审核的一组智能体操作。"""

    review_configs: list[ReviewConfig]
    """所有可能操作的审核配置。"""


class ApproveDecision(TypedDict):
    """人类批准操作时的响应。"""

    type: Literal["approve"]
    """人类批准操作时的响应类型。"""


class EditDecision(TypedDict):
    """人类编辑操作时的响应。"""

    type: Literal["edit"]
    """人类编辑操作时的响应类型。"""

    edited_action: Action
    """供智能体执行的已编辑操作。

    Ex: 例如，对于工具调用，人类审核者可以编辑工具名称和参数。
    """


class RejectDecision(TypedDict):
    """人类拒绝操作时的响应。"""

    type: Literal["reject"]
    """人类拒绝操作时的响应类型。"""

    message: NotRequired[str]
    """人类拒绝操作提供的理由。

    The reason is framed as a user rejection when sent to the model. If omitted,
    the model is told that the tool was not executed and should not retry the same
    tool call unless the user asks for it.
    """


class RespondDecision(TypedDict):
    """人类代表工具作出回应，跳过执行。

    Used for "ask user" style tools whose real implementation is the human's
    response. The tool is not executed; instead, a synthetic `ToolMessage` with
    `status="success"` and the provided `message` is returned to the model.
    """

    type: Literal["respond"]
    """人类代表工具作出回应时的响应类型。"""

    message: str
    """返回给模型的合成 `ToolMessage` 内容。"""


Decision = ApproveDecision | EditDecision | RejectDecision | RespondDecision


InterruptMode = Literal["batched", "per_call"]
"""How HITL pauses: one interrupt per model turn (`batched`) or per gated tool call."""


class ToolApprovalRequest(TypedDict):
    """在 `per_call` 模式下，每个受保护的工具调用触发一次中断时抛出的中断值。

    Takes the place of `HITLRequest`. Answer it with a single `Decision`, keyed by the
    interrupt's ID, not a `HITLResponse`. The interrupt's `response_schema` lists the
    decisions allowed for the tool and, for edits, the tool's argument schema.
    """

    type: Literal["tool_approval"]
    """始终为 `"tool_approval"`；告诉客户端如何读取此中断。"""

    tool_call_id: str
    """此审批所涉及的模型工具调用的 ID。"""

    name: str
    """模型请求的工具名称。"""

    args: dict[str, Any]
    """模型请求的工具参数。"""

    description: str
    """显示给审核者的文本。"""


def _edit_args(tool: BaseTool | None) -> object:
    """编辑的 `args` 必须符合的格式：如果工具有，则为工具的 Pydantic 模式。

    That's the schema the model sees (`tool_call_schema`): it checks arg types and
    required args. Args it doesn't declare pass through, and the tool ignores them as
    it would from the model.
    Validator methods on the tool's `args_schema` aren't part of it; they run when the
    tool does.
    A JSON-schema tool's schema is shown but not enforced, and with no tool any object
    is accepted.
    """
    schema = tool.tool_call_schema if tool else None
    if not (isinstance(schema, type) and issubclass(schema, BaseModel)):
        shown = schema if isinstance(schema, dict) else {"type": "object"}
        return Annotated[dict[str, Any], WithJsonSchema(shown)]
    return schema


def _as_sent(answer: object, check: ValidatorFunctionWrapHandler) -> object:
    """检查 `answer`，然后将其原样传递，以便工具在参数验证时仅执行一次。"""
    check(answer)
    return answer


def _edit_decision(name: str, tool: BaseTool | None) -> object:
    """`EditDecision` 用于单个工具：`name` 固定在其上，`args` 来自 `_edit_args`。"""
    edited_action = create_model(
        "EditedAction",
        name=(Literal[name], ...),
        args=(_edit_args(tool), ...),
    )
    decision = create_model(
        "EditDecision",
        type=(Literal["edit"], ...),
        edited_action=(edited_action, ...),
    )
    return Annotated[decision, WrapValidator(_as_sent)]


def _decision_schema(
    allowed: Sequence[DecisionType], name: str, tool: BaseTool | None = None
) -> type[Decision]:
    """每次调用的 `response_schema`：今日的决策类型，限于 `allowed` 范围内。

    The edit branch is built for this tool: its name is pinned, so an edit can't switch
    tools, and its args follow the tool's own schema. An answer is checked only against
    the branch its `type` names, so a bad one gets a single error about what's wrong.
    A one-decision tool gets that decision's plain object schema.

    返回类型而非 JSON 模式，因为 LangGraph 仅根据 Python 类型检查答案；它会将类型的
    JSON 模式发布给客户端。
    """
    by_type: dict[DecisionType, object] = {
        "approve": ApproveDecision,
        "reject": RejectDecision,
        "respond": RespondDecision,
    }
    if "edit" in allowed:
        by_type["edit"] = _edit_decision(name, tool)
        # 去除重复项，保持顺序：一种类型的并集不能包含 `Discriminator`。
    members = tuple(by_type[d] for d in dict.fromkeys(allowed))
        # 类型检查器无法追踪在运行时构建的类型。在运行时，`interrupt()` 会检查每条答案
        # 是否符合该类型，因此其返回值是这些决策之一。
    if len(members) == 1:
        return cast("type[Decision]", members[0])
        # 仅检查答案的 `type` 所命名的分支，以便错误答案获得一个精确的
        # 错误，而不是每个分支都报错一次。
    return cast("type[Decision]", Annotated[Union[members], Discriminator("type")])  # noqa: UP007


def _answer_message(tool_call: ToolCall, decision: RejectDecision | RespondDecision) -> ToolMessage:
    """模型在工具结果位置收到的消息。"""
    if decision["type"] == "respond":
                # 跳过工具执行；由人类代为作出决策。
        content = decision["message"]
    elif reason := decision.get("message"):
        content = f"User rejected the tool call for `{tool_call['name']}` with reason: {reason}"
    else:
        content = (
            f"User rejected the tool call for `{tool_call['name']}` with id "
            f"{tool_call['id']}. The tool was not executed. Do not retry this tool "
            "call unless the user explicitly requests it."
        )
    return ToolMessage(
        content=content,
        name=tool_call["name"],
        tool_call_id=tool_call["id"],
        status="success" if decision["type"] == "respond" else "error",
    )


class HITLResponse(TypedDict):
    """HITLRequest 的响应负载。"""

    decisions: list[Decision]
    """人类做出的决策。"""


class _DescriptionFactory(Protocol):
    """生成工具调用描述的可调用对象。"""

    def __call__(
        self, tool_call: ToolCall, state: AgentState[Any], runtime: Runtime[ContextT]
    ) -> str:
        """生成工具调用的描述。"""
        ...


class InterruptOnConfig(TypedDict):
    """需要人工介入的操作的配置。

    这是 `HumanInTheLoopMiddleware.__init__` 方法中使用的配置格式。
    """

    allowed_decisions: list[DecisionType]
    """此操作允许的决策。"""

    description: NotRequired[str | _DescriptionFactory]
    """附加到人工输入请求的描述。

    可以是：

    - 描述审批请求的静态字符串 -
    根据代理状态、运行时和工具调用信息动态生成描述的可调用对象

    Example:
        ```python
        # Static string description
        config = InterruptOnConfig(
            allowed_decisions=["approve", "reject"],
            description="Please review this tool execution"
        )

        # Dynamic callable description
        def format_tool_description(
            tool_call: ToolCall,
            state: AgentState,
            runtime: Runtime[ContextT]
        ) -> str:
            import json
            return (
                f"Tool: {tool_call['name']}\\n"
                f"Arguments:\\n{json.dumps(tool_call['args'], indent=2)}"
            )

        config = InterruptOnConfig(
            allowed_decisions=["approve", "edit", "reject"],
            description=format_tool_description
        )
        ```
    """
    args_schema: NotRequired[dict[str, Any]]
    """Unused: 未使用：中间件在任一模式下都不会将其发送给评审者。

    在 `per_call` 模式下，中断的 `response_schema` 显示工具自身的参数模式。
    """

    when: NotRequired[Callable[[ToolCallRequest], bool]]
    """可选谓词，控制是否针对特定工具调用中断。

    接收一个 `ToolCallRequest`，返回 `True` 以中断或 `False` 以自动批准。 该谓词在
    `after_model` 期间调用，在工具调用添加到批处理的人工介入请求之前执行。

    The request is constructed with `tool=None` and a new `ToolRuntime`. The
    `ToolRuntime` copies `context`, `store`, `stream_writer`, `execution_info`,
    and `server_info` from the node-level `Runtime`, while `tool_call_id` is
    populated from the current tool call. The `tools` argument is not supplied,
    so it uses its default empty list.

    In `per_call` mode the predicate runs in `wrap_tool_call` and receives the real
    `ToolCallRequest`, with `tool` and the tool's `ToolRuntime` set.

    In both modes the predicate runs again when the run resumes, so it must return the
    same answer for the same call. If it returns `False` on resume, the reviewer's
    answer is skipped and the tool runs, even if they rejected it. In `batched` mode,
    if it flips for only some of the reviewed calls, the decisions no longer line up
    with the calls and resuming raises a `ValueError`.

    Example:
        ```python
        # Only interrupt delete_file calls targeting /etc
        config = InterruptOnConfig(
            allowed_decisions=["approve", "reject"],
            when=lambda req: req.tool_call["args"].get("path", "").startswith("/etc"),
        )
        ```
    """


class _HumanInTheLoopState(AgentState[ResponseT]):
    """`HumanInTheLoopMiddleware` 的状态模式。"""

    hitl_edited_tool_calls: NotRequired[Annotated[dict[str, Action], PrivateStateAttr]]
    """跟踪来自 `after_model` 的工具调用编辑，以便 `wrap_tool_call` 使用。"""


class HumanInTheLoopMiddleware(AgentMiddleware[StateT, ContextT, ResponseT]):
    """暂停智能体，以便人员可在工具运行前审查工具调用。

    Each tool in `interrupt_on` needs a decision before it runs: approve, edit, reject,
    or respond. The pause is a LangGraph `interrupt`, so the agent needs a checkpointer,
    and the run continues with `Command(resume=...)`.

    `interrupt_mode` 设置智能体暂停的方式：

    - `"batched"` (default): one interrupt per model turn. Its value is a `HITLRequest`
        listing every gated call, answered with a `HITLResponse` whose decisions match
        the calls by position.
    - `"per_call"`: one interrupt per gated tool call, as the call starts. Its value is
        a `ToolApprovalRequest` and its `response_schema` lists the decisions that tool
        allows. Each is answered with a single `Decision` keyed by interrupt ID, and
        LangGraph checks the answer against the schema before saving it. `"per_call"`
        is planned to become the default in the next major release.

    Examples:
        !!! example "批处理：每轮模型调用一次中断"

            ```python
            from langchain.agents import create_agent
            from langchain.agents.middleware import HumanInTheLoopMiddleware
            from langgraph.checkpoint.memory import InMemorySaver
            from langgraph.types import Command

            agent = create_agent(
                model,
                tools=[send_email, delete_file],
                middleware=[
                    HumanInTheLoopMiddleware(interrupt_on={"send_email": True, "delete_file": True})
                ],
                checkpointer=InMemorySaver(),
            )
            config = {"configurable": {"thread_id": "1"}}
            result = agent.invoke(
                {"messages": [{"role": "user", "content": "Email Alice, then delete the report."}]},
                config,
                version="v2",
            )
            # One interrupt; `value["action_requests"]` lists send_email, then delete_file.
            agent.invoke(
                Command(
                    resume={
                        "decisions": [
                            {"type": "approve"},
                            {"type": "reject", "message": "Keep the report."},
                        ]
                    }
                ),
                config,
                version="v2",
            )
            ```

        !!! example "按调用：每个受控工具调用一次中断"

            ```python
            agent = create_agent(
                model,
                tools=[send_email, delete_file],
                middleware=[
                    HumanInTheLoopMiddleware(
                        interrupt_on={
                            "send_email": True,
                            "delete_file": {"allowed_decisions": ["approve", "reject"]},
                        },
                        interrupt_mode="per_call",
                    )
                ],
                checkpointer=InMemorySaver(),
            )
            config = {"configurable": {"thread_id": "2"}}
            result = agent.invoke(
                {"messages": [{"role": "user", "content": "Email Alice, then delete the report."}]},
                config,
                version="v2",
            )
            # Two interrupts, each with `value["type"] == "tool_approval"` and its own
            # `response_schema`. Answer each by its ID.
            answers = {}
            for interrupt in result.interrupts:
                if interrupt.value["name"] == "delete_file":
                    answers[interrupt.id] = {"type": "reject", "message": "Keep the report."}
                else:
                    answers[interrupt.id] = {"type": "approve"}
            agent.invoke(Command(resume=answers), config, version="v2")
            ```
    """

    state_schema = _HumanInTheLoopState  # type: ignore[assignment]

    def __init__(
        self,
        interrupt_on: dict[str, bool | InterruptOnConfig],
        *,
        description_prefix: str = "Tool execution requires approval",
        edit_notice: str | None = _EDIT_NOTICE,
        interrupt_mode: InterruptMode = "batched",
    ) -> None:
        """初始化人机协同中间件。

        Args:
            interrupt_on: 工具名称到允许操作的映射。

                如果某个工具没有条目，则默认自动批准。

                * `True` indicates all decisions are allowed: approve, edit, reject,
                    and respond.
                * `False` indicates that the tool is auto-approved.
                * `InterruptOnConfig` indicates the specific decisions allowed for this
                    tool.

                    The `InterruptOnConfig` can include a `description` field (`str` or
                    `Callable`) for custom formatting of the interrupt description.

                    description_prefix: 构造操作请求时使用的前缀。

                用于提供关于工具调用及所请求操作的上下文信息。
                如果工具在其 `InterruptOnConfig` 中有 `description`，则不使用此字段。

                Not used if a tool has a `description` in its `InterruptOnConfig`.
            edit_notice: Text prepended to the result of a tool call a reviewer replaced
                via an `edit` decision. Pass `None` to add nothing.
            interrupt_mode: How the agent pauses for review.

                - `"batched"` (default): one interrupt per model turn for all gated
                    tool calls, answered with a `HITLResponse`.
                - `"per_call"`: one `ToolApprovalRequest` interrupt per gated tool
                    call, with a typed `response_schema`, answered with a single
                    `Decision` keyed by interrupt ID.

                在 `per_call` 模式下，LangGraph 会在保存前检查每个答案：

                - 编辑不能切换工具，且参数必须与工具的参数类型匹配。
                工具未声明的参数会被忽略，这与模型发送时的行为一致。 工具的
                `args_schema` 上的验证方法会在工具执行时运行。 对于参数为 JSON 模式而非
                Pydantic 模型的工具，编辑会显示在 `response_schema` 中但不会被检查。 -
                无效的回答会引发 `pydantic.ValidationError`
                并不保存，因此相同的中断可以再次回答。

                !!! warning "同时回答多个调用"

                    If one of several answers sent together is invalid, the others'
                    tools may already have run even if they still show as pending,
                    and answering them again would run them twice. Check answers
                    against `response_schema` before sending them together, or send
                    one at a time.

                !!! warning "`per_call` 模式中的中间件顺序"

                    The review happens as the tool call starts, inside the tool-call
                    middleware chain. List this middleware before tool retry or
                    error-handling middleware so it wraps them, and don't enable it
                    on subclasses that raise their own interrupts in `after_model`.
                    On resume, each paused tool call runs the middleware listed
                    before this one again up to the pause, so that middleware must
                    not have side effects before calling `handler`. The tool itself
                    runs only after the answer.

                与 `batched` 模式类似，异步运行智能体（`ainvoke`, `astream`）需要
                Python 3.11 或更高版本。

        Raises:
            ValueError: If a tool's `InterruptOnConfig` does not have a non-empty
                `allowed_decisions` list (e.g. a misspelled key or an empty list).
                An interrupt config without decisions would otherwise be silently
                dropped, disabling the approval gate for that tool.
                Also if `interrupt_mode` is invalid.
        """
        super().__init__()
        if interrupt_mode not in (modes := get_args(InterruptMode)):
            allowed = " or ".join(repr(mode) for mode in modes)
            msg = f"`interrupt_mode` must be {allowed}, got {interrupt_mode!r}."
            raise ValueError(msg)
        self.interrupt_mode = interrupt_mode
        self.edit_notice = edit_notice
        resolved_configs: dict[str, InterruptOnConfig] = {}
        for tool_name, tool_config in interrupt_on.items():
            if isinstance(tool_config, bool):
                if tool_config is True:
                    resolved_configs[tool_name] = InterruptOnConfig(
                        allowed_decisions=["approve", "edit", "reject", "respond"]
                    )
            elif tool_config.get("allowed_decisions"):
                resolved_configs[tool_name] = tool_config
            else:
                msg = (
                    f"Invalid `interrupt_on` config for tool '{tool_name}': "
                    "`allowed_decisions` must be a non-empty list of decision types "
                    "(e.g. ['approve', 'reject']). Got config with keys "
                    f"{sorted(tool_config.keys())} and "
                    f"allowed_decisions={tool_config.get('allowed_decisions')!r}."
                )
                raise ValueError(msg)
        self.interrupt_on = resolved_configs
        self.description_prefix = description_prefix

    def _create_action_and_config(
        self,
        tool_call: ToolCall,
        config: InterruptOnConfig,
        state: AgentState[Any],
        runtime: Runtime[ContextT],
    ) -> tuple[ActionRequest, ReviewConfig]:
        """为工具调用创建一个 ActionRequest 和 ReviewConfig。"""
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]

                # 使用 description 字段（str 或可调用对象）生成描述。
        description_value = config.get("description")
        if callable(description_value):
            description = description_value(tool_call, state, runtime)
        elif description_value is not None:
            description = description_value
        else:
            description = f"{self.description_prefix}\n\nTool: {tool_name}\nArgs: {tool_args}"

                # 使用描述创建 ActionRequest。
        action_request = ActionRequest(
            name=tool_name,
            args=tool_args,
            description=description,
        )

                # 创建 ReviewConfig。
        review_config = ReviewConfig(
            action_name=tool_name,
            allowed_decisions=config["allowed_decisions"],
        )

        return action_request, review_config

    @staticmethod
    def _process_decision(
        decision: Decision,
        tool_call: ToolCall,
        config: InterruptOnConfig,
    ) -> tuple[ToolCall | None, ToolMessage | None]:
        """处理单个决策并返回修订后的工具调用和可选的工具消息。"""
        allowed_decisions = config["allowed_decisions"]

        if decision["type"] == "approve" and "approve" in allowed_decisions:
            return tool_call, None
        if decision["type"] == "edit" and "edit" in allowed_decisions:
                        # 保留消息中的模型自身调用；`wrap_tool_call` 在执行时替换为
                        # 审查者的调用。
            return tool_call, None
        if decision["type"] in allowed_decisions and (
            decision["type"] == "reject" or decision["type"] == "respond"
        ):
            return tool_call, _answer_message(tool_call, decision)
        msg = (
            f"Unexpected human decision: {decision}. "
            f"Decision type '{decision.get('type')}' "
            f"is not allowed for tool '{tool_call['name']}'. "
            f"Expected one of {allowed_decisions} based on the tool's configuration."
        )
        raise ValueError(msg)

    def _should_interrupt(
        self,
        tool_call: ToolCall,
        config: InterruptOnConfig,
        state: AgentState[Any],
        runtime: Runtime[ContextT],
    ) -> bool:
        """如果 `when` 谓词拒绝此工具调用，则返回 False，否则返回 True。"""
        when = config.get("when")
        if when is None:
            return True
        runnable_config: RunnableConfig
        try:
            runnable_config = get_config()
        except RuntimeError:
            runnable_config = {}
        tool_runtime = ToolRuntime(
            state=state,
            context=runtime.context,
            config=runnable_config,
            stream_writer=runtime.stream_writer,
            tool_call_id=tool_call["id"],
            store=runtime.store,
            execution_info=runtime.execution_info,
            server_info=runtime.server_info,
        )
        req = ToolCallRequest(
            tool_call=tool_call,
            tool=None,
            state=state,
            runtime=tool_runtime,  # type: ignore[arg-type]
        )
        return when(req)

    def _tool_approval_request(
        self, request: ToolCallRequest, config: InterruptOnConfig
    ) -> ToolApprovalRequest:
        """为一个受控工具调用构建 per-call 中断值。

        Raises:
            ValueError: 如果工具调用没有 ID。它无法在没有 ID
            的情况下运行，因此不会向审查者请求批准。
        """
        tool_call = request.tool_call
        tool_call_id = tool_call["id"]
                # 仅 `None`：空 ID 仍会运行，因此在批处理模式下 `per_call` 仍可正常工作。
        if tool_call_id is None:
            msg = (
                f"Tool call `{tool_call['name']}` has no ID, so its result can't be matched "
                "to it. Make sure the chat model returns tool call IDs."
            )
            raise ValueError(msg)
                # 描述工厂接收图的 `Runtime`，如批处理模式中所示。
        runtime: Runtime[ContextT] = get_runtime()
        action_request, _ = self._create_action_and_config(
            tool_call, config, request.state, runtime
        )
        return ToolApprovalRequest(
            type="tool_approval",
            tool_call_id=tool_call_id,
            name=tool_call["name"],
            args=tool_call["args"],
            description=action_request.get("description", ""),
        )

    def _resolve_per_call(
        self, request: ToolCallRequest
    ) -> tuple[ToolCallRequest, Action | None] | ToolMessage:
        """如果工具调用受控，向审查者询问该工具调用。

        Returns the request to run, with the reviewer's call if they edited it, or the
        message to return instead of running the tool.
        """
        tool_call = request.tool_call
        config = self.interrupt_on.get(tool_call["name"])
        if config is None:
            return request, None
        when = config.get("when")
        if when is not None and not when(request):
            return request, None

        value = self._tool_approval_request(request, config)
        response_schema = _decision_schema(
            config["allowed_decisions"], tool_call["name"], request.tool
        )
                # LangGraph 在保存前会根据 `response_schema` 解析答案，因此它
                # 返回该工具允许的决策之一。
        decision = interrupt(value, response_schema=response_schema)
        if decision["type"] == "approve":
            return request, None
        if decision["type"] == "edit":
                        # 模式固定了工具名称，因此始终是同一个工具。
            executed = decision["edited_action"]
            return self._apply_edit(request, executed), executed
        return _answer_message(tool_call, decision)

    def after_model(
        self, state: AgentState[Any], runtime: Runtime[ContextT]
    ) -> dict[str, Any] | None:
        """在收到 `AIMessage` 后，触发相关工具调用的中断流程。

        Args:
            state: 当前智能体状态。
            runtime: 运行时上下文。

        Returns:
            使用修订后的工具调用更新消息。

        Raises:
            ValueError: 如果人类决策数量与中断的工具调用数量不匹配。
        """
        if self.interrupt_mode == "per_call":
                        # 中断在每个工具调用时触发，位于 `wrap_tool_call` 中。
            return None
        messages = state["messages"]
        if not messages:
            return None

        last_ai_msg = next((msg for msg in reversed(messages) if isinstance(msg, AIMessage)), None)
        if not last_ai_msg or not last_ai_msg.tool_calls:
            return None

                # 为需要审批的工具创建操作请求和审查配置
        action_requests: list[ActionRequest] = []
        review_configs: list[ReviewConfig] = []
        interrupt_indices: list[int] = []

        for idx, tool_call in enumerate(last_ai_msg.tool_calls):
            if (config := self.interrupt_on.get(tool_call["name"])) is not None:
                if not self._should_interrupt(tool_call, config, state, runtime):
                    continue
                action_request, review_config = self._create_action_and_config(
                    tool_call, config, state, runtime
                )
                action_requests.append(action_request)
                review_configs.append(review_config)
                interrupt_indices.append(idx)

                # 如果没有中断需求，提前返回，丢弃之前轮次的任何编辑，以防止它们被应用到本轮的工具调用中。
                # 创建包含所有操作和配置的单一 HITLRequest
        if not action_requests:
            if state.get(_EDITED_TOOL_CALLS_KEY):
                return {_EDITED_TOOL_CALLS_KEY: {}}
            return None

                # 发送中断并获取响应
        hitl_request = HITLRequest(
            action_requests=action_requests,
            review_configs=review_configs,
        )

                # 验证决策数量与中断工具调用数量一致
        decisions = interrupt(hitl_request)["decisions"]

                # 处理决策并按原始顺序重建工具调用
        if (decisions_len := len(decisions)) != (interrupt_count := len(interrupt_indices)):
            msg = (
                f"Number of human decisions ({decisions_len}) does not match "
                f"number of hanging tool calls ({interrupt_count})."
            )
            raise ValueError(msg)

                # 这是一个中断工具调用 - 处理该决策
        revised_tool_calls: list[ToolCall] = []
        artificial_tool_messages: list[ToolMessage] = []
        edited_tool_calls: dict[str, Action] = {}
        decision_idx = 0

        for idx, tool_call in enumerate(last_ai_msg.tool_calls):
            if idx in interrupt_indices:
                                # 这是自动批准 - 保留原始内容
                config = self.interrupt_on[tool_call["name"]]
                decision = decisions[decision_idx]
                decision_idx += 1

                revised_tool_call, tool_message = self._process_decision(
                    decision, tool_call, config
                )
                if revised_tool_call is not None:
                    revised_tool_calls.append(revised_tool_call)
                    if decision["type"] == "edit" and (edited_id := revised_tool_call.get("id")):
                        edited_tool_calls[edited_id] = decision["edited_action"]
                if tool_message:
                    artificial_tool_messages.append(tool_message)
            else:
                                # 更新 AI 消息，仅包含已批准的工具调用
                revised_tool_calls.append(tool_call)

                # `wrap_tool_call`
        # 会读取此内容以替换并注释调用。始终写入，因此之前轮次的编辑无法延续到本轮。
        last_ai_msg.tool_calls = revised_tool_calls

                # `tool_call["name"]` 和 `tool` 必须保持一致。
                # 匹配周围块的结构；提供方可能拒绝混合列表。
        return {
            "messages": [last_ai_msg, *artificial_tool_messages],
            _EDITED_TOOL_CALLS_KEY: edited_tool_calls,
        }

    async def aafter_model(
        self, state: AgentState[Any], runtime: Runtime[ContextT]
    ) -> dict[str, Any] | None:
        """在 `AIMessage` 之后异步触发相关工具调用的中断流。

        Args:
            state: 当前智能体状态。
            runtime: 运行时上下文。

        Returns:
            使用修订后的工具调用更新消息。
        """
        return self.after_model(state, runtime)

    def _reviewer_edit(self, request: ToolCallRequest) -> Action | None:
        """如果 `edit` 决策替换了该调用，则为审查者的替换内容。"""
        tool_call_id = request.tool_call.get("id")
        if not tool_call_id:
            return None
        edited = request.state.get(_EDITED_TOOL_CALLS_KEY) or {}
        if tool_call_id in edited:
            executed: Action = edited[tool_call_id]
            return executed
        return None

    def _apply_edit(self, request: ToolCallRequest, executed: Action) -> ToolCallRequest:
        """将请求指向审查者的调用，解决重定向的工具调用。

        Raises:
            ValueError: 如果审查者命名了智能体不存在的工具。
        """
        tool_call: ToolCall = {
            **request.tool_call,
            "name": executed["name"],
            "args": executed["args"],
        }
        if executed["name"] == request.tool_call["name"]:
            return request.override(tool_call=tool_call)

                # 一个 `Command` 在其状态更新中携带 `ToolMessage`。
        available = request.runtime.tools
        tool = next((t for t in available if t.name == executed["name"]), None)
        if tool is None:
            names = ", ".join(sorted(t.name for t in available))
            msg = (
                f"Reviewer edited tool call {request.tool_call['id']!r} to "
                f"{executed['name']!r}, which is not an available tool. "
                f"Available tools: {names}."
            )
            raise ValueError(msg)
        return request.override(tool_call=tool_call, tool=tool)

    def _notice(self, executed: Action, *, has_content: bool) -> str:
        """通知文本，说明实际运行的调用。"""
        notice = (
            f"{self.edit_notice} Executed instead: {executed['name']} with arguments "
            f"{json.dumps(executed['args'], default=str)}."
        )
        return f"{notice}\n\nTool response:" if has_content else notice

    def _prepend_notice(self, message: ToolMessage, executed: Action) -> ToolMessage:
        """返回 `message`，并将审查者编辑通知前置到其内容中。"""
        if not self.edit_notice:
            return message
        edit_notice = self._notice(executed, has_content=bool(message.content))

        content: str | list[str | dict[Any, Any]]
        if isinstance(message.content, str):
            if edit_notice in message.content:
                return message
            separator = "\n" if message.content else ""
            content = f"{edit_notice}{separator}{message.content}"
        else:
            if any(edit_notice in str(block) for block in message.content):
                return message
                        # 匹配周围块的形状；提供方可能拒绝混合列表。
            notice: str | dict[Any, Any] = (
                edit_notice
                if message.content and all(isinstance(b, str) for b in message.content)
                else {"type": "text", "text": edit_notice}
            )
            content = [notice, *message.content]

        return message.model_copy(update={"content": content})

    def _annotate_edited_result(
        self,
        result: ToolMessage | Command[Any],
        request: ToolCallRequest,
        executed: Action | None,
    ) -> ToolMessage | Command[Any]:
        """告知模型审查者已替换该调用，以及替换为何物。"""
        if not self.edit_notice or executed is None:
            return result

        if isinstance(result, ToolMessage):
            return self._prepend_notice(result, executed)

                # 一个 `Command` 在其状态更新中携带 `ToolMessage`。
        if not isinstance(result, Command) or not isinstance(result.update, dict):
            return result
        messages = result.update.get("messages")
        if not isinstance(messages, list):
            return result
        tool_call_id = request.tool_call.get("id")
        return replace(
            result,
            update={
                **result.update,
                "messages": [
                    self._prepend_notice(message, executed)
                    if isinstance(message, ToolMessage) and message.tool_call_id == tool_call_id
                    else message
                    for message in messages
                ],
            },
        )

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command[Any]],
    ) -> ToolMessage | Command[Any]:
        """在编辑后的工具调用结果前添加审查者编辑指引。

        在 `per_call` 模式下，这也是中断被抛出的位置。

        Args:
            request: 正在执行的工具调用请求。
            handler: 执行工具的可调用对象。

        Returns:
            工具结果，若审查者编辑了调用则在前面添加备注。

        Raises:
            ValidationError: In `per_call` mode, if the reviewer's answer doesn't match
                the tool call's `response_schema`. Nothing is saved, so the call can be
                answered again.
            ValueError: In `per_call` mode, if a gated tool call has no ID.
        """
        if self.interrupt_mode == "per_call":
            resolved = self._resolve_per_call(request)
            if isinstance(resolved, ToolMessage):
                return resolved
            to_run, executed = resolved
            return self._annotate_edited_result(handler(to_run), to_run, executed)
        executed = self._reviewer_edit(request)
        if executed is not None:
            request = self._apply_edit(request, executed)
        return self._annotate_edited_result(handler(request), request, executed)

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command[Any]]],
    ) -> ToolMessage | Command[Any]:
        """`wrap_tool_call` 的异步变体。

        在 `per_call` 模式下，这也是中断被抛出的位置。

        Args:
            request: 正在执行的工具调用请求。
            handler: 可等待的可调用对象，用于执行工具。

        Returns:
            工具结果，若审查者编辑了调用则在前面添加备注。

        Raises:
            ValidationError: In `per_call` mode, if the reviewer's answer doesn't match
                the tool call's `response_schema`. Nothing is saved, so the call can be
                answered again.
            ValueError: In `per_call` mode, if a gated tool call has no ID.
        """
        if self.interrupt_mode == "per_call":
            resolved = self._resolve_per_call(request)
            if isinstance(resolved, ToolMessage):
                return resolved
            to_run, executed = resolved
            return self._annotate_edited_result(await handler(to_run), to_run, executed)
        executed = self._reviewer_edit(request)
        if executed is not None:
            request = self._apply_edit(request, executed)
        return self._annotate_edited_result(await handler(request), request, executed)

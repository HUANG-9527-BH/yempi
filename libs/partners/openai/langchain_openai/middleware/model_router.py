"""由 OpenAI Decisions API 驱动的实验性模型路由中间件。"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

from langchain.agents.middleware.types import (
    AgentMiddleware,
    AgentState,
    ContextT,
    ModelRequest,
    ModelResponse,
    ResponseT,
    TracePolicy,
    omit_payload,
)
from langchain.chat_models import init_chat_model
from langchain_core._api import beta
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, convert_to_openai_messages
from typing_extensions import NotRequired, TypedDict, override

from langchain_openai.decisions import (
    Choice,
    DecisionRequest,
    DecisionResponse,
    OpenAIDecisions,
)
from langchain_openai.middleware._fallback import (
    adecide_with_text_fallback,
    decide_with_text_fallback,
)

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from langgraph.runtime import Runtime

_QUESTION_NAME = "model_route"


@dataclass(frozen=True)
class ModelChoice:
    """路由可用的模型以及选择它的标准。

    Args:
        model: LangChain 模型实例或 `init_chat_model` 接受的模型字符串。
        criteria: 适合该模型的任务描述。
    """

    model: str | BaseChatModel
    criteria: str


class ModelRoute(TypedDict):
    """路由结果存储在智能体状态的 `model_route` 下。

    一个普通映射而非 `ChoiceAnswer`，以便检查点可以持久化它
    而无需注册自定义类型。
    """

    choice: str
    """所选路由的名称。"""

    probabilities: dict[str, float]
    """分配给每个路由的概率。"""

    confidence: float
    """从分布中得出的标量确定性，取值范围为 `0` 到 `1`。"""


class _ModelRouterState(AgentState):
    """用于持久化路由结果的智能体状态。"""

    model_route: NotRequired[ModelRoute | None]


@beta()
class OpenAIModelRouterMiddleware(AgentMiddleware[_ModelRouterState]):
    """使用 OpenAI Decisions `Choice` 问题选择智能体的模型。

    Classifies the latest human message once before each agent run, stores the
    answer as a `ModelRoute` in agent state under `model_route`, and uses the selected
    route for every model call in the run. Keeping the complete answer makes
    probabilities and confidence available in state and traces.

    The message text and base64 images are classified. Hosted image URLs, files,
    and audio are replaced with placeholders such as `[image omitted]`. If a request
    with images is rejected (HTTP 400 or 413), routing is retried once with every
    image replaced.

    当未选择任何路由时（例如模型拒绝或状态中没有用户消息），`model_route` 将设置为
    `None`，模型调用将使用智能体自身的模型。API 错误将传播并终止运行。

    !!! warning

        此中间件为实验性功能。其 API 可能随时更改而无通知。

    Args:
        choices: Named model choices, each containing a LangChain model or model
            string and the criterion for selecting it.
        instructions: Routing question sent to the Decisions API.
        model: Decisions model name, or a configured `OpenAIDecisions` instance.

    ??? example "按任务路由智能体调用"

        ```python
        from langchain.agents import create_agent
        from langchain_openai.middleware import (
            ModelChoice,
            OpenAIModelRouterMiddleware,
        )

        router = OpenAIModelRouterMiddleware(
            choices={
                "fast": ModelChoice(
                    model="openai:gpt-6-luna",
                    criteria="Simple, well-scoped tasks.",
                ),
                "powerful": ModelChoice(
                    model="openai:gpt-6-sol",
                    criteria="Complex tasks requiring deeper reasoning.",
                ),
            },
            instructions="Choose the least costly model suited to the task.",
            model="gpt-6-luna",
        )
        agent = create_agent("openai:gpt-6-luna", middleware=[router])
        ```
    """

    state_schema = _ModelRouterState
    trace_policy = TracePolicy(process_inputs=omit_payload)
    """从中间件追踪中排除用户消息。"""

    def __init__(
        self,
        *,
        choices: Mapping[str, ModelChoice],
        instructions: str,
        model: str | OpenAIDecisions,
    ) -> None:
        """初始化模型路由器。

        Args:
            choices: 命名的模型选项及每个选项的选择标准。
            instructions: 发送给 Decisions API 的路由问题。
            model: Decisions 模型名称，或配置好的 `OpenAIDecisions` 实例。

        Raises:
            ValueError: 如果 `choices` 为空或 `instructions` 为空白。
        """
        if not choices:
            msg = "`choices` must contain at least one model choice."
            raise ValueError(msg)
        if not instructions.strip():
            msg = "`instructions` must not be empty."
            raise ValueError(msg)
        self.question = Choice(
            instructions=instructions,
            choices={route: choice.criteria for route, choice in choices.items()},
        )
        self.models = {
            route: init_chat_model(choice.model)
            if isinstance(choice.model, str)
            else choice.model
            for route, choice in choices.items()
        }
        self.decisions = (
            OpenAIDecisions(model=model) if isinstance(model, str) else model
        )

    @override
    def before_agent(
        self, state: _ModelRouterState, runtime: Runtime[ContextT]
    ) -> dict[str, Any]:
        """对最新任务进行分类并存储路由结果。"""
        requests = self._decision_requests(state)
        if not requests:
            return {"model_route": None}
        response = decide_with_text_fallback(self.decisions, requests)
        return {"model_route": _route(response)}

    @override
    async def abefore_agent(
        self, state: _ModelRouterState, runtime: Runtime[ContextT]
    ) -> dict[str, Any]:
        """异步对最新任务进行分类并存储路由结果。"""
        requests = self._decision_requests(state)
        if not requests:
            return {"model_route": None}
        response = await adecide_with_text_fallback(self.decisions, requests)
        return {"model_route": _route(response)}

    @override
    def wrap_model_call(
        self,
        request: ModelRequest[ContextT],
        handler: Callable[[ModelRequest[ContextT]], ModelResponse[ResponseT]],
    ) -> ModelResponse[ResponseT]:
        """将同步模型调用路由到选定的模型。"""
        return handler(self._routed(request))

    @override
    async def awrap_model_call(
        self,
        request: ModelRequest[ContextT],
        handler: Callable[
            [ModelRequest[ContextT]], Awaitable[ModelResponse[ResponseT]]
        ],
    ) -> ModelResponse[ResponseT]:
        """将异步模型调用路由到选定的模型。"""
        return await handler(self._routed(request))

    def _decision_requests(self, state: _ModelRouterState) -> list[DecisionRequest]:
        """Return requests to try in order: 返回尝试请求的顺序：先带图像（如有），再纯文本。"""
        message = next(
            (m for m in reversed(state["messages"]) if isinstance(m, HumanMessage)),
            None,
        )
        if message is None:
            return []
        text_only = _routable(message, keep_images=False)
        with_images = _routable(message, keep_images=True)
        inputs = (
            [text_only]
            if with_images.content == text_only.content
            else [with_images, text_only]
        )
        return [
            {"input": m, "questions": {_QUESTION_NAME: self.question}} for m in inputs
        ]

    def _routed(self, request: ModelRequest[ContextT]) -> ModelRequest[ContextT]:
        route = cast("ModelRoute | None", request.state.get("model_route"))
        if route is None:
            return request
        return request.override(model=self.models[route["choice"]])


def _routable(message: HumanMessage, *, keep_images: bool) -> HumanMessage:
    """保留文本（以及如请求则保留 base64 图像）；替换其他块。"""
    if isinstance(message.content, str):
        return message
    content = [
        block
        if isinstance(block, str)
        or block.get("type") == "text"
        or (keep_images and _is_base64_image(block))
        else {"type": "text", "text": f"[{block.get('type', 'content')} omitted]"}
        for block in message.content
    ]
    return message.model_copy(update={"content": content})


def _is_base64_image(block: dict[str, Any]) -> bool:
    if block.get("type") not in {"image", "image_url"}:
        return False
    try:
        part = convert_to_openai_messages(HumanMessage(content=[block]))["content"][0]
    except (ValueError, KeyError, TypeError):
        return False
    url = part.get("image_url", {}).get("url", "") if isinstance(part, dict) else ""
    return url.startswith("data:")


def _route(response: DecisionResponse) -> ModelRoute | None:
    answer = response.choices.get(_QUESTION_NAME)
    if answer is None:
        return None
    return {
        "choice": str(answer.choice),
        "probabilities": {str(k): v for k, v in answer.probabilities.items()},
        "confidence": answer.confidence,
    }


__all__ = ["ModelChoice", "ModelRoute", "OpenAIModelRouterMiddleware"]

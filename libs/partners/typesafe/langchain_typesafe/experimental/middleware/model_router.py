"""由 TypeSafe 提供支持的实验性模型路由中间件。"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass

from langchain.agents.middleware import Runtime
from langchain.agents.middleware.types import ContextT

try:
    from langchain.agents.middleware.types import (
        AgentMiddleware,
        AgentState,
        ModelRequest,
        ModelResponse,
        ResponseT,
        TracePolicy,
        omit_payload,
    )
    from langchain.chat_models import init_chat_model
except ImportError as error:
    msg = (
        "ModelRouterMiddleware requires the LangChain agent framework. "
        "Install it with `pip install 'langchain-typesafe[experimental]'`."
    )
    raise ImportError(msg) from error

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field, JsonValue
from typing_extensions import NotRequired, override

from langchain_typesafe.classifier import TypeSafeClassifier
from langchain_typesafe.types import Choice, ChoiceAnswer, Question

_QUESTION_ID = "model_route"
_QuestionContent = str | dict[str, JsonValue] | list[JsonValue]


@dataclass(frozen=True)
class ModelChoice:
    """路由器可用的模型，以及选择该模型的标准。

    Args:
        model: LangChain 模型实例或 `init_chat_model` 接受的模型字符串。
        criteria: 适合该模型的任务描述。
    """

    model: str | BaseChatModel
    criteria: JsonValue


class _ModelRouterConfig(BaseModel):
    """已验证的模型路由配置。"""

    choices: dict[str, ModelChoice] = Field(min_length=1)
    instructions: _QuestionContent


def _routing_questions(config: _ModelRouterConfig) -> dict[str, Question]:
    """根据已验证的中间件配置构建路由问题。"""
    return {
        _QUESTION_ID: Choice(
            instructions=config.instructions,
            criteria={
                route: choice.criteria for route, choice in config.choices.items()
            },
        )
    }


class _ModelRouterState(AgentState):
    """用于持久化 TypeSafe 路由答案的智能体状态。"""

    model_route: NotRequired[ChoiceAnswer]


class ModelRouterMiddleware(AgentMiddleware[_ModelRouterState]):
    """使用 TypeSafe `Choice` 分类选择智能体的模型。

    The middleware classifies the latest human message once before an agent run,
    stores the complete `ChoiceAnswer` in agent state, and uses its selected label
    for every model call in the run. Keeping the complete answer makes probabilities
    and confidence available in state and traces. Classifier failures propagate and
    terminate the run rather than silently selecting a different model.

    !!! warning

        此中间件为实验性功能。其 API 可能随时更改而无需通知。

    安装实验性额外包以使用此类：

    ```bash
    pip install "langchain-typesafe[experimental]"
    ```

    Args:
        choices: Named model choices, each containing a LangChain model or model
            string and the criterion for selecting it.
        instructions: Additional instructions TypeSafe should follow when selecting a
            route.

    Raises:
        pydantic.ValidationError: 如果未提供任何模型选项。

    ??? example "按任务路由智能体调用"

        ```python
        from langchain.agents import create_agent
        from langchain_typesafe.experimental.middleware import (
            ModelChoice,
            ModelRouterMiddleware,
        )

        router = ModelRouterMiddleware(
            choices={
                "fast": ModelChoice(
                    model="openai:gpt-5-mini",
                    criteria="Simple, well-scoped tasks.",
                ),
                "powerful": ModelChoice(
                    model=powerful_model,
                    criteria="Complex tasks requiring deeper reasoning.",
                ),
            },
            instructions="Choose the least costly model suited to the task.",
        )
        agent = create_agent("openai:gpt-5-mini", middleware=[router])
        ```
    """

    state_schema = _ModelRouterState
    trace_policy = TracePolicy(process_inputs=omit_payload)

    def __init__(
        self,
        *,
        choices: Mapping[str, ModelChoice],
        instructions: _QuestionContent,
    ) -> None:
        """初始化模型路由器。"""
        self.config = _ModelRouterConfig.model_validate(
            {"choices": choices, "instructions": instructions}
        )
        self.models = {
            route: init_chat_model(choice.model)
            if isinstance(choice.model, str)
            else choice.model
            for route, choice in self.config.choices.items()
        }
        self.classifier = TypeSafeClassifier()

    @staticmethod
    def _latest_human_message(state: _ModelRouterState) -> HumanMessage:
        """从智能体状态返回最新的用户消息。"""
        return next(
            message
            for message in reversed(state["messages"])
            if isinstance(message, HumanMessage)
        )

    @override
    def before_agent(
        self, state: _ModelRouterState, runtime: Runtime[ContextT]
    ) -> dict[str, ChoiceAnswer]:
        """对最新任务进行分类并存储完整的路由答案。"""
        response = self.classifier.invoke(
            {
                "state": self._latest_human_message(state),
                "questions": _routing_questions(self.config),
            }
        )
        return {"model_route": response.choices[_QUESTION_ID]}

    @override
    async def abefore_agent(
        self, state: _ModelRouterState, runtime: Runtime[ContextT]
    ) -> dict[str, ChoiceAnswer]:
        """异步对最新任务进行分类并存储路由答案。"""
        response = await self.classifier.ainvoke(
            {
                "state": self._latest_human_message(state),
                "questions": _routing_questions(self.config),
            }
        )
        return {"model_route": response.choices[_QUESTION_ID]}

    @override
    def wrap_model_call(
        self,
        request: ModelRequest[ContextT],
        handler: Callable[[ModelRequest[ContextT]], ModelResponse[ResponseT]],
    ) -> ModelResponse[ResponseT]:
        """将同步模型调用路由至选定的模型。"""
        answer: ChoiceAnswer = request.state["model_route"]  # type: ignore[typeddict-item]
        return handler(request.override(model=self.models[answer.choice]))

    @override
    async def awrap_model_call(
        self,
        request: ModelRequest[ContextT],
        handler: Callable[
            [ModelRequest[ContextT]], Awaitable[ModelResponse[ResponseT]]
        ],
    ) -> ModelResponse[ResponseT]:
        """将异步模型调用路由至选定的模型。"""
        answer: ChoiceAnswer = request.state["model_route"]  # type: ignore[typeddict-item]
        return await handler(request.override(model=self.models[answer.choice]))


__all__ = ["ModelChoice", "ModelRouterMiddleware"]

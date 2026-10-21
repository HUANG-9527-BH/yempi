"""基于 LLM 的工具选择中间件。"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Annotated, Any, Literal, TypeGuard, Union

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from pydantic import Field, TypeAdapter
from typing_extensions import TypedDict

from langchain.agents.middleware.internal_call_transformer import (
    InternalCallTransformer,
    internal_call_metadata,
)
from langchain.agents.middleware.types import (
    AgentMiddleware,
    AgentState,
    ContextT,
    ModelRequest,
    ModelResponse,
    ResponseT,
)
from langchain.chat_models.base import init_chat_model

if TYPE_CHECKING:
    from collections.abc import Awaitable

    from langchain_core.runnables import RunnableConfig

    from langchain.tools import BaseTool

logger = logging.getLogger(__name__)

DEFAULT_SYSTEM_PROMPT = (
    "Your goal is to select the most relevant tools for answering the user's query."
)

OnParsingFailure = Literal["error", "none", "all"] | list[str] | Callable[[Any], list[str]]
"""Behavior when the selection model keeps returning a malformed response.

Can be either:
- `'error'`: Raise a `ValueError` (the default).
- `'none'`: Select no tools.
- `'all'`: Select every available tool.
- A `list[str]` of tool names to fall back to.
- A callable that takes the last (malformed) response and returns tool names to use.
"""


@dataclass
class _SelectionRequest:
    """用于工具选择的预处理输入。"""

    available_tools: list[BaseTool]
    system_message: str
    last_user_message: HumanMessage
    model: BaseChatModel
    valid_tool_names: list[str]


def _create_tool_selection_response(tools: list[BaseTool]) -> TypeAdapter[Any]:
    """为工具选择创建结构化输出模式。

    Args:
        tools: 可用工具，包含在模式中。

    Returns:
        `TypeAdapter` for a schema where each tool name is a `Literal` with its
            description.

    Raises:
        AssertionError: 如果 `tools` 为空。
    """
    if not tools:
        msg = "Invalid usage: tools must be non-empty"
        raise AssertionError(msg)

        # 为每个工具名称创建带描述的 Annotated Literal 类型的联合类型
        # 例如：Union[Annotated[Literal["tool1"], Field(description="...")], ...]
    literals = [
        Annotated[Literal[tool.name], Field(description=tool.description)] for tool in tools
    ]
    selected_tool_type = Union[tuple(literals)]  # type: ignore[valid-type]  # noqa: UP007

    description = "Tools to use. Place the most relevant tools first."

    class ToolSelectionResponse(TypedDict):
        """用于选择相关工具。"""

        tools: Annotated[list[selected_tool_type], Field(description=description)]  # type: ignore[valid-type]

    return TypeAdapter(ToolSelectionResponse)


def _is_valid_selection_response(response: Any) -> TypeGuard[dict[str, Any]]:
    """检查结构化输出响应是否为格式正确的工具选择。

    Args:
        response: 来自结构化输出模型调用的原始响应。

    Returns:
        如果 `response` 是包含 `tools` 列表的字典，则返回 `True`，否则返回 `False`。
    """
    return isinstance(response, dict) and isinstance(response.get("tools"), list)


def _render_tool_list(tools: list[BaseTool]) -> str:
    """将工具格式化为 Markdown 列表。

    Args:
        tools: 要格式化的工具。

    Returns:
        每个工具占一行的 Markdown 字符串。
    """
    return "\n".join(f"- {tool.name}: {tool.description}" for tool in tools)


class LLMToolSelectorMiddleware(AgentMiddleware[AgentState[ResponseT], ContextT, ResponseT]):
    """使用 LLM 在调用主模型前选择相关工具。

    When an agent has many tools available, this middleware filters them down
    to only the most relevant ones for the user's query. This reduces token usage
    and helps the main model focus on the right tools.

    Examples:
        !!! example "限制为 3 个工具"

            ```python
            from langchain.agents.middleware import LLMToolSelectorMiddleware

            middleware = LLMToolSelectorMiddleware(max_tools=3)

            agent = create_agent(
                model="openai:gpt-5.5",
                tools=[tool1, tool2, tool3, tool4, tool5],
                middleware=[middleware],
            )
            ```

        !!! example "使用更小的模型进行选择"

            ```python
            middleware = LLMToolSelectorMiddleware(model="openai:gpt-5.4-mini", max_tools=2)
            ```
    """

    transformers = (InternalCallTransformer,)
    """将工具选择模型调用的 token 保留在 `run.messages` 之外。

    Registered only when this middleware is used — see
    `InternalCallTransformer` for why the call needs tagging and filtering.
    """

    def __init__(
        self,
        *,
        model: str | BaseChatModel | None = None,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        max_tools: int | None = None,
        always_include: list[str] | None = None,
        max_retries: int = 1,
        on_parsing_failure: OnParsingFailure = "error",
    ) -> None:
        """初始化工具选择器。

        Args:
            model: 用于选择的模型。

                如果未提供，则使用智能体的主模型。

                Can be a model identifier string or `BaseChatModel` instance.
            system_prompt: Instructions for the selection model.
            max_tools: Maximum number of tools to select.

                system_prompt: 用于选择模型的指令。

                max_tools: 最多选择的工具数量。

                These do not count against the `max_tools` limit.
            max_retries: Maximum number of retry attempts after the initial call if
                the selection model returns a malformed response (not a dict with a
                `tools` list).

                如果未指定，则无限制。

                always_include: 无论选择结果如何都始终包含的工具名称。

                这些工具不计入 `max_tools` 限制。

                max_retries: 在初始调用后，若选择模型返回格式错误的响应（非包含 `tools`
                列表的字典），最多重试次数。

        Raises:
            ValueError: 必须为 `>= 0`。
        """
        super().__init__()
        if max_retries < 0:
            msg = "max_retries must be >= 0"
            raise ValueError(msg)
        self.system_prompt = system_prompt
        self.max_tools = max_tools
        self.always_include = always_include or []
        self.max_retries = max_retries
        self.on_parsing_failure = on_parsing_failure

        if isinstance(model, (BaseChatModel, type(None))):
            self.model: BaseChatModel | None = model
        else:
            self.model = init_chat_model(model)

    def _prepare_selection_request(
        self, request: ModelRequest[ContextT]
    ) -> _SelectionRequest | None:
        """on_parsing_failure: 当 `max_retries` 用尽且响应仍格式错误时的行为。

        Args:
            request: 选项：

        Returns:
            准备好的 `SelectionRequest`，如果没有选择需求则返回 `None`。

        Raises:
            ValueError: 如果 `always_include` 中的工具在请求中找不到。
            AssertionError: 如果请求消息中没有找到用户消息。
        """
                # 如果没有可用工具，返回 None
        if not request.tools or len(request.tools) == 0:
            return None

                # 仅保留 BaseTool 实例（排除特定提供方的工具字典）
        base_tools = [tool for tool in request.tools if not isinstance(tool, dict)]

                # 验证 `always_include` 工具是否存在
        if self.always_include:
            available_tool_names = {tool.name for tool in base_tools}
            missing_tools = [
                name for name in self.always_include if name not in available_tool_names
            ]
            if missing_tools:
                msg = (
                    f"Tools in always_include not found in request: {missing_tools}. "
                    f"Available tools: {sorted(available_tool_names)}"
                )
                raise ValueError(msg)

                # 将始终包含的工具与可供选择的工具分离开
        available_tools = [tool for tool in base_tools if tool.name not in self.always_include]

                # 如果无可选工具，返回 None
        if not available_tools:
            return None

        system_message = self.system_prompt
                # 如果有 `max_tools` 限制，向系统提示词中添加说明
        if self.max_tools is not None:
            system_message += (
                f"\nIMPORTANT: List the tool names in order of relevance, "
                f"with the most relevant first. "
                f"If you exceed the maximum number of tools, "
                f"only the first {self.max_tools} will be used."
            )

                # 从对话历史中获取最后一个用户消息
        last_user_message: HumanMessage
        for message in reversed(request.messages):
            if isinstance(message, HumanMessage):
                last_user_message = message
                break
        else:
            msg = "No user message found in request messages"
            raise AssertionError(msg)

        model = self.model or request.model
        valid_tool_names = [tool.name for tool in available_tools]

        return _SelectionRequest(
            available_tools=available_tools,
            system_message=system_message,
            last_user_message=last_user_message,
            model=model,
            valid_tool_names=valid_tool_names,
        )

    def _process_selection_response(
        self,
        response: dict[str, Any],
        available_tools: list[BaseTool],
        valid_tool_names: list[str],
        request: ModelRequest[ContextT],
        *,
        apply_max_tools: bool = True,
    ) -> ModelRequest[ContextT]:
        """处理选择响应并返回过滤后的 `ModelRequest`。

        Args:
            response: 选择响应，预期包含一个 `tools` 列表。
            available_tools: 可供选择的工具。
            valid_tool_names: `available_tools` 的名称列表。
            request: 需要覆盖的原始模型请求。
            apply_max_tools: 是否将选择数量限制在 `max_tools` 内。

                Set to `False` for an already-deliberate fallback selection (e.g.
                `on_parsing_failure="all"`), where truncating would contradict the
                fallback's own semantics.
        """
        selected_tool_names: list[str] = []
        invalid_tool_selections = []
        max_tools = self.max_tools if apply_max_tools else None

        for tool_name in response.get("tools", []):
            if tool_name not in valid_tool_names:
                invalid_tool_selections.append(tool_name)
                continue

                        # 仅在未被选中且未超过 `max_tools` 限制时添加
            if tool_name not in selected_tool_names and (
                max_tools is None or len(selected_tool_names) < max_tools
            ):
                selected_tool_names.append(tool_name)

        if invalid_tool_selections:
            msg = f"Model selected invalid tools: {invalid_tool_selections}"
            raise ValueError(msg)

                # 根据选择结果过滤工具，并追加始终包含的工具
        selected_tools: list[BaseTool] = [
            tool for tool in available_tools if tool.name in selected_tool_names
        ]
        always_included_tools: list[BaseTool] = [
            tool
            for tool in request.tools
            if not isinstance(tool, dict) and tool.name in self.always_include
        ]
        selected_tools.extend(always_included_tools)

                # 同时保留原始请求中的任何提供方特定工具字典
        provider_tools = [tool for tool in request.tools if isinstance(tool, dict)]

        return request.override(tools=[*selected_tools, *provider_tools])

    def _resolve_parsing_failure(self, response: Any, valid_tool_names: list[str]) -> list[str]:
        """在 `max_retries` 用尽后确定应使用的工具名称。

        Args:
            response: 选择模型最后一次（仍格式错误）的响应。
            valid_tool_names: 可供选择的工具名称列表。

        Returns:
            根据 `on_parsing_failure` 确定要选择的工具名称。

        Raises:
            ValueError: 如果 `on_parsing_failure == "error"`（默认情况）。
        """
        if self.on_parsing_failure == "error":
            msg = (
                "LLMToolSelectorMiddleware: selection model returned a malformed "
                f"response after {self.max_retries} retries (expected a dict with a "
                f"'tools' list): {response!r}"
            )
            raise ValueError(msg)
        if self.on_parsing_failure == "none":
            return []
        if self.on_parsing_failure == "all":
            return valid_tool_names
        if callable(self.on_parsing_failure):
            return self.on_parsing_failure(response)
        return list(self.on_parsing_failure)

    def wrap_model_call(
        self,
        request: ModelRequest[ContextT],
        handler: Callable[[ModelRequest[ContextT]], ModelResponse[ResponseT]],
    ) -> ModelResponse[ResponseT] | AIMessage:
        """在通过处理器调用模型前，基于 LLM 的选择结果过滤工具。

        Args:
            request: 要执行的模型请求（包含状态和运行时）。
            handler: 异步回调，用于执行模型请求并返回 `ModelResponse`。

        Returns:
            模型调用的结果。

        Raises:
            ValueError: If `on_parsing_failure == "error"` (the default) and the
                selection model returns a malformed response after `max_retries`
                retries.
        """
        selection_request = self._prepare_selection_request(request)
        if selection_request is None:
            return handler(request)

                # 创建具有可用工具名称 Literal 枚举的动态响应模型
        type_adapter = _create_tool_selection_response(selection_request.available_tools)
        schema = type_adapter.json_schema()
        structured_model = selection_request.model.with_structured_output(schema)

        messages: list[BaseMessage | dict[str, Any]] = [
            {"role": "system", "content": selection_request.system_message},
            selection_request.last_user_message,
        ]
        config: RunnableConfig = {
            "metadata": {"lc_source": "tool_selection", **internal_call_metadata()}
        }

        response = structured_model.invoke(messages, config=config)
        attempts = 0
        while not _is_valid_selection_response(response) and attempts < self.max_retries:
                        # 结构化输出格式错误通常是暂时的，在放弃前应重试，而不是静默降级为未选择或全部工具被选中。
                        # 放弃前应重试，而不是静默降级为未选择或全部工具被选中。
            response = structured_model.invoke(messages, config=config)
            attempts += 1

        if _is_valid_selection_response(response):
            modified_request = self._process_selection_response(
                response,
                selection_request.available_tools,
                selection_request.valid_tool_names,
                request,
            )
        else:
            fallback_tool_names = self._resolve_parsing_failure(
                response, selection_request.valid_tool_names
            )
                        # 绕过 max_tools：回退机制已经是有意的选择
                        # （例如 `on_parsing_failure="all"`），而非需要限制的原始模型输出。
                        # 绕过 max_tools：回退机制已经是有意的选择
            modified_request = self._process_selection_response(
                {"tools": fallback_tool_names},
                selection_request.available_tools,
                selection_request.valid_tool_names,
                request,
                apply_max_tools=False,
            )
        return handler(modified_request)

    async def awrap_model_call(
        self,
        request: ModelRequest[ContextT],
        handler: Callable[[ModelRequest[ContextT]], Awaitable[ModelResponse[ResponseT]]],
    ) -> ModelResponse[ResponseT] | AIMessage:
        """在通过处理器调用模型前，基于 LLM 的选择结果过滤工具。

        Args:
            request: 要执行的模型请求（包含状态和运行时）。
            handler: 异步回调，用于执行模型请求并返回 `ModelResponse`。

        Returns:
            模型调用的结果。

        Raises:
            ValueError: If `on_parsing_failure == "error"` (the default) and the
                selection model returns a malformed response after `max_retries`
                retries.
        """
        selection_request = self._prepare_selection_request(request)
        if selection_request is None:
            return await handler(request)

                # 创建动态响应模型，使用可用工具名称的 Literal 枚举
        type_adapter = _create_tool_selection_response(selection_request.available_tools)
        schema = type_adapter.json_schema()
        structured_model = selection_request.model.with_structured_output(schema)

        messages: list[BaseMessage | dict[str, Any]] = [
            {"role": "system", "content": selection_request.system_message},
            selection_request.last_user_message,
        ]
        config: RunnableConfig = {
            "metadata": {"lc_source": "tool_selection", **internal_call_metadata()}
        }

        response = await structured_model.ainvoke(messages, config=config)
        attempts = 0
        while not _is_valid_selection_response(response) and attempts < self.max_retries:
                        # 结构化输出格式错误通常是暂时的，在放弃前应重试，而不是静默降级为未选择或全部工具被选中。
                        # 放弃前应重试，而不是静默降级为未选择或全部工具被选中。
            response = await structured_model.ainvoke(messages, config=config)
            attempts += 1

        if _is_valid_selection_response(response):
            modified_request = self._process_selection_response(
                response,
                selection_request.available_tools,
                selection_request.valid_tool_names,
                request,
            )
        else:
            fallback_tool_names = self._resolve_parsing_failure(
                response, selection_request.valid_tool_names
            )
                        # 绕过 max_tools：回退机制已经是有意的选择
                        # （例如 `on_parsing_failure="all"`），而非需要限制的原始模型输出。
                        # 绕过 max_tools：回退机制已经是有意的选择
            modified_request = self._process_selection_response(
                {"tools": fallback_tool_names},
                selection_request.available_tools,
                selection_request.valid_tool_names,
                request,
                apply_max_tools=False,
            )
        return await handler(modified_request)

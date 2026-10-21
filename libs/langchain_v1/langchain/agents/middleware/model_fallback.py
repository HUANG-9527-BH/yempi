"""智能体的模型回退中间件。

When an outer caching middleware modifies a request, those changes happen before
the fallback loop runs. Provider-specific cache settings can cause API errors if
the loop reuses them with a different provider. This middleware therefore strips
unsupported Anthropic and Fireworks cache settings from each fallback attempt,
while preserving settings accepted by the selected fallback.

此提供方知识存放于此，因为外部缓存中间件在回退过程中永远不会重新运行，因此无法自行清理。
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from langchain_core.tools import BaseTool
from langgraph.errors import GraphBubbleUp

from langchain.agents.middleware.types import (
    AgentMiddleware,
    AgentState,
    ContextT,
    ModelRequest,
    ModelResponse,
    ResponseT,
)
from langchain.chat_models import init_chat_model

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.messages import AIMessage, AnyMessage, SystemMessage

logger = logging.getLogger(__name__)

_FIREWORKS_LLM_TYPE = "fireworks-chat"
_FIREWORKS_SESSION_AFFINITY_HEADER = "x-session-affinity"


def _sanitize_content_blocks(
    content: str | list[str | dict[str, Any]],
) -> str | list[str | dict[str, Any]]:
    """从消息内容块中移除 Anthropic 缓存标记。"""
    if not isinstance(content, list):
        return content

    sanitized_content: list[str | dict[str, Any]] = []
    changed = False

    for block in content:
        if not isinstance(block, dict):
            sanitized_content.append(block)
            continue

        sanitized_block, block_changed = _without_cache_control_from_content_block(block)
        changed = changed or block_changed
        sanitized_content.append(sanitized_block)

    return sanitized_content if changed else content


def _sanitize_system_message(
    system_message: SystemMessage | None,
) -> SystemMessage | None:
    """从系统消息中移除 Anthropic 缓存标记。"""
    if system_message is None:
        return None

    sanitized_content = _sanitize_content_blocks(system_message.content)
    if sanitized_content is system_message.content:
        return system_message

    return system_message.model_copy(update={"content": sanitized_content})


def _sanitize_messages(messages: list[AnyMessage]) -> list[AnyMessage]:
    """从请求消息中移除 Anthropic 缓存标记。"""
    sanitized_messages: list[AnyMessage] = []
    changed = False

    for message in messages:
        sanitized_message, message_changed = _sanitize_message(message)
        changed = changed or message_changed
        sanitized_messages.append(sanitized_message)

    return sanitized_messages if changed else messages


def _sanitize_tools(
    tools: list[BaseTool | dict[str, Any]],
) -> list[BaseTool | dict[str, Any]]:
    """从工具负载中移除 Anthropic 缓存标记。"""
    sanitized_tools: list[BaseTool | dict[str, Any]] = []
    changed = False

    for tool in tools:
        sanitized_tool: BaseTool | dict[str, Any]
        if isinstance(tool, BaseTool):
            sanitized_tool, tool_changed = _sanitize_base_tool(tool)
        else:
            sanitized_tool, tool_changed = _sanitize_dict_tool(tool)

        changed = changed or tool_changed
        sanitized_tools.append(sanitized_tool)

    return sanitized_tools if changed else tools


def _sanitize_request_for_fallback(
    request: ModelRequest[ContextT],
    fallback_model: BaseChatModel | None = None,
) -> ModelRequest[ContextT]:
    """在回退尝试前，清理提供方特定的缓存设置。"""
    overrides: dict[str, Any] = {}
    supports_anthropic_cache = _supports_anthropic_cache_control(fallback_model)

    model_settings = request.model_settings
    model_settings_changed = False

    if not supports_anthropic_cache:
        model_settings, cache_control_changed = _without_cache_control(model_settings)
        model_settings_changed = model_settings_changed or cache_control_changed

    if not _supports_fireworks_prompt_cache(fallback_model):
        model_settings, fireworks_cache_changed = _without_fireworks_session_affinity(
            model_settings
        )
        model_settings_changed = model_settings_changed or fireworks_cache_changed

    if not _supports_prompt_cache_key(fallback_model) and "prompt_cache_key" in model_settings:
        model_settings = {
            key: value for key, value in model_settings.items() if key != "prompt_cache_key"
        }
        model_settings_changed = True

    if model_settings_changed:
        overrides["model_settings"] = model_settings

    if not supports_anthropic_cache:
        system_message = _sanitize_system_message(request.system_message)
        if system_message is not request.system_message:
            overrides["system_message"] = system_message

        messages = _sanitize_messages(request.messages)
        if messages is not request.messages:
            overrides["messages"] = messages

        tools = _sanitize_tools(request.tools)
        if tools is not request.tools:
            overrides["tools"] = tools

    if not overrides:
        return request

        # 仅记录变更的字段名称，从不记录请求内容（可能包含提示词数据或个人身份信息）。
        # 仅记录变更的字段名称，从不记录请求内容（可能包含提示词数据或个人身份信息）。
    logger.debug(
        "Stripped provider-specific cache settings from %s before fallback attempt",
        sorted(overrides),
    )

    return request.override(**overrides)


def _sanitize_message(message: AnyMessage) -> tuple[AnyMessage, bool]:
    """从单个消息中移除 Anthropic 缓存标记。

    Returns:
        已清理的消息（未更改时为原始实例）以及是否移除了任何标记。
    """
    sanitized_content = _sanitize_content_blocks(message.content)
    if sanitized_content is message.content:
        return message, False

    return message.model_copy(update={"content": sanitized_content}), True


def _sanitize_base_tool(tool: BaseTool) -> tuple[BaseTool, bool]:
    """从 `BaseTool` 负载中移除 Anthropic 缓存标记。

    Returns:
        已清理的工具（未更改时为原始实例）以及是否移除了任何标记。

            空的 `extras` 会恢复为 `None`。
    """
    if not tool.extras:
        return tool, False

    sanitized_extras, changed = _without_cache_control(tool.extras)
    if not changed:
        return tool, False

    return tool.model_copy(update={"extras": sanitized_extras or None}), True


def _sanitize_dict_tool(tool: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    """从字典类型的工具负载中移除 Anthropic 缓存标记。

    Returns:
        已清理的工具（未更改时为原始实例）以及是否移除了任何标记。

            空的 `extras` 会恢复为 `None`。
    """
    sanitized_tool, changed = _without_cache_control(tool)

    extras = sanitized_tool.get("extras")
    if not isinstance(extras, dict):
        return sanitized_tool, changed

    sanitized_extras, extras_changed = _without_cache_control(extras)
    if not extras_changed:
        return sanitized_tool, changed

    return {**sanitized_tool, "extras": sanitized_extras or None}, True


def _without_cache_control(payload: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    """返回不含 `cache_control` 的负载，以及是否发生更改。"""
    if "cache_control" not in payload:
        return payload, False

    return (
        {key: value for key, value in payload.items() if key != "cache_control"},
        True,
    )


def _without_fireworks_session_affinity(
    model_settings: dict[str, Any],
) -> tuple[dict[str, Any], bool]:
    """返回不含 Fireworks 特定亲和性头的模型设置。"""
    extra_headers = model_settings.get("extra_headers")
    if not isinstance(extra_headers, Mapping):
        return model_settings, False

    sanitized_headers = {
        key: value
        for key, value in extra_headers.items()
        if not (isinstance(key, str) and key.lower() == _FIREWORKS_SESSION_AFFINITY_HEADER)
    }
    if len(sanitized_headers) == len(extra_headers):
        return model_settings, False

    sanitized_settings = dict(model_settings)
    if sanitized_headers:
        sanitized_settings["extra_headers"] = sanitized_headers
    else:
        sanitized_settings.pop("extra_headers")
    return sanitized_settings, True


def _without_cache_control_from_content_block(
    block: dict[str, Any],
) -> tuple[dict[str, Any], bool]:
    """返回不含 Anthropic 缓存标记的内容块。

    Strips `cache_control` from the block itself and from its nested `extras` and
    `metadata` payloads.

    Returns:
        已清理的块（未更改时为原始实例）以及是否移除了任何标记。
    """
    sanitized_block, changed = _without_cache_control(block)

    for nested_key in ("extras", "metadata"):
        nested_payload = sanitized_block.get(nested_key)
        if not isinstance(nested_payload, dict):
            continue

        sanitized_payload, nested_changed = _without_cache_control(nested_payload)
        if not nested_changed:
            continue

        if sanitized_block is block:
            sanitized_block = dict(block)
        sanitized_block[nested_key] = sanitized_payload
        changed = True

    return sanitized_block, changed


# `_llm_type` 值中表示模型支持 Anthropic 兼容 API 并接受 `cache_control` 标记的类型
# 直接的 Anthropic 模型（`ChatAnthropic`）报告 `"anthropic-chat"`；Bedrock 托管的
# Claude（`ChatAnthropicBedrock`，`langchain-aws` 中的 `ChatAnthropic` 子类）报告
# `"anthropic-bedrock-chat"` 并将顶层关键字参数转换为继承的
# `ChatAnthropic._get_request_payload` 内部的块级断点，而内容块和工具的 `cache_control`
# 标记保持不变。
# Vertex 托管的 Claude（`langchain-google` 中的 `ChatAnthropicVertex`）报告
# `"anthropic-chat-vertexai"` 并通过其自身请求构建器嵌套相同的标记结构——而非共享的
# `ChatAnthropic` 方法。三者均在回退时保持提示词缓存完整。
# 请保持此集合与这些类的 `_llm_type` 值同步，这些值位于独立的仓库中。若某值偏离或新的
# Anthropic
# 传输方式发布，故障模式为静默丢失提示词缓存（支持该功能的模型上标记被剥离），而非硬错误——因此此处的
# CI 不会捕获该问题。
# 初始化所有备用模型
# 首先尝试主模型
# 尝试备用模型——清理选定备用模型无法接受的提供方特定缓存设置。
# 请求在 try 外推导，以便 sanitizer 或 `_llm_type`
# 错误能直接暴露，而非被掩盖为模型失败。
# 首先尝试主模型
#
# 尝试备用模型——清理选定备用模型无法接受的提供方特定缓存设置。
# 请求在 try 外推导，以便 sanitizer 或 `_llm_type`
# 错误能直接暴露，而非被掩盖为模型失败。
# 失败模式是提示词缓存的静默丢失（标记被从
# 支持它们的模型），而不是硬性错误——因此这里的 CI 无法捕获它。
_ANTHROPIC_LLM_TYPES: frozenset[str] = frozenset(
    {
        "anthropic-chat",
        "anthropic-bedrock-chat",
        "anthropic-chat-vertexai",
    }
)


def _supports_anthropic_cache_control(model: BaseChatModel | None) -> bool:
    """返回 `model` 是否接受 Anthropic `cache_control` 标记。

    Checked via `_llm_type` so the decision is provider-based rather than
    model-name-based: any Anthropic-compatible model (including future model IDs
    we have not seen) keeps its cache markers on fallback, while OpenAI, Gemini,
    and other non-Anthropic providers get a sanitized request.
    """
    llm_type = getattr(model, "_llm_type", None)
    return isinstance(llm_type, str) and llm_type in _ANTHROPIC_LLM_TYPES


def _supports_fireworks_prompt_cache(model: BaseChatModel | None) -> bool:
    """返回 `model` 是否接受 Fireworks 会话亲和性头。"""
    return getattr(model, "_llm_type", None) == _FIREWORKS_LLM_TYPE


def _supports_prompt_cache_key(model: BaseChatModel | None) -> bool:
    """返回 `model` 是否接受共享的 `prompt_cache_key` 参数。"""
    llm_type = getattr(model, "_llm_type", None)
    return isinstance(llm_type, str) and llm_type in {
        _FIREWORKS_LLM_TYPE,
        "openai-chat",
        "azure-openai-chat",
    }


class ModelFallbackMiddleware(AgentMiddleware[AgentState[ResponseT], ContextT, ResponseT]):
    """在出错时自动回退到替代模型。

    Retries failed model calls with alternative models in sequence until
    success or all models exhausted. Primary model specified in `create_agent`.

    Example:
        ```python
        from langchain.agents.middleware import ModelFallbackMiddleware
        from langchain.agents import create_agent

        fallback = ModelFallbackMiddleware(
            "openai:gpt-5.5",  # Try first on error
            "anthropic:claude-sonnet-4-5-20250929",  # Then this
        )

        agent = create_agent(
            model="openai:gpt-5.5",  # Primary model
            middleware=[fallback],
        )

        # If primary fails: tries gpt-5.5, then claude-sonnet-4-5-20250929
        result = await agent.invoke({"messages": [HumanMessage("Hello")]})
        ```
    """

    def __init__(
        self,
        first_model: str | BaseChatModel,
        *additional_models: str | BaseChatModel,
    ) -> None:
        """初始化模型回退中间件。

        Args:
            first_model: 首个回退模型（字符串名称或实例）。
            *additional_models: 按顺序的额外回退模型。
        """
        super().__init__()

                # 初始化所有回退模型
        all_models = (first_model, *additional_models)
        self.models: list[BaseChatModel] = []
        for model in all_models:
            if isinstance(model, str):
                self.models.append(init_chat_model(model))
            else:
                self.models.append(model)

    def wrap_model_call(
        self,
        request: ModelRequest[ContextT],
        handler: Callable[[ModelRequest[ContextT]], ModelResponse[ResponseT]],
    ) -> ModelResponse[ResponseT] | AIMessage:
        """按顺序尝试回退模型以应对错误。

        Args:
            request: 初始模型请求。
            handler: 执行模型的回调。

        Returns:
            成功模型调用返回的 AIMessage。

        Raises:
            Exception: 异常：如果所有模型均失败，则重新抛出最后一个异常。
        """
                # 尝试先使用主模型
        last_exception: Exception
        try:
            return handler(request)
        except GraphBubbleUp:
            raise
        except Exception as e:
            last_exception = e

                # 尝试备用模型 — 清理提供方特定的缓存设置
                # 所选的备用方案无法接受。
                # 请求在 try 之外生成，因此可能需要清理器或 `_llm_type`
                # 错误会直接暴露，而不是被掩盖为模型失败。
        for fallback_model in self.models:
            fallback_request = _sanitize_request_for_fallback(request, fallback_model)
            try:
                return handler(fallback_request.override(model=fallback_model))
            except GraphBubbleUp:
                raise
            except Exception as e:
                last_exception = e
                continue

        raise last_exception

    async def awrap_model_call(
        self,
        request: ModelRequest[ContextT],
        handler: Callable[[ModelRequest[ContextT]], Awaitable[ModelResponse[ResponseT]]],
    ) -> ModelResponse[ResponseT] | AIMessage:
        """按顺序尝试备用模型（异步版本）。

        Args:
            request: 初始模型请求。
            handler: 异步回调，用于执行模型。

        Returns:
            成功模型调用返回的 AIMessage。

        Raises:
            Exception: 异常：如果所有模型均失败，则重新抛出最后一个异常。
        """
                # 尝试先使用主模型
        last_exception: Exception
        try:
            return await handler(request)
        except GraphBubbleUp:
            raise
        except Exception as e:
            last_exception = e

                # 尝试备用模型 — 清理提供方特定的缓存设置
                # 所选的回退无法接受。
                # 请求在 try 之外生成，因此需要外部的清理器或 `_llm_type`
                # 错误会直接暴露，而不是被掩盖为模型失败。
        for fallback_model in self.models:
            fallback_request = _sanitize_request_for_fallback(request, fallback_model)
            try:
                return await handler(fallback_request.override(model=fallback_model))
            except GraphBubbleUp:
                raise
            except Exception as e:
                last_exception = e
                continue

        raise last_exception

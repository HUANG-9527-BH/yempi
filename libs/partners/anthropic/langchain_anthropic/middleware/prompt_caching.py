"""Anthropic 提示词缓存中间件。

需要：
    - `langchain`：用于智能体中间件框架
    - `langchain-anthropic`：用于 `ChatAnthropic` 模型（已是依赖项）
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any, Literal
from warnings import warn

from langchain_core.messages import SystemMessage
from langchain_core.tools import BaseTool

from langchain_anthropic.chat_models import ChatAnthropic

try:
    from langchain.agents.middleware.types import (
        AgentMiddleware,
        ModelCallResult,
        ModelRequest,
        ModelResponse,
    )
except ImportError as e:
    msg = (
        "AnthropicPromptCachingMiddleware requires 'langchain' to be installed. "
        "This middleware is designed for use with LangChain agents. "
        "Install it with: pip install langchain"
    )
    raise ImportError(msg) from e


def _get_trace_policy() -> Any:
    """在安装的 LangChain 支持时，使用追踪优化。"""
    try:
        from langchain.agents.middleware.types import TracePolicy, omit_payload
    except ImportError:
        return None
    return TracePolicy(process_inputs=omit_payload)


_TRACE_POLICY = _get_trace_policy()


class AnthropicPromptCachingMiddleware(AgentMiddleware):
    """提示词缓存中间件。

    通过缓存 Anthropic 模型的对话前缀来优化 API 使用。

    需要安装 `langchain` 和 `langchain-anthropic` 包。

    该中间件标记稳定智能体内容，并通过 `model_settings` 传递 `cache_control`：

    - **System message**: Tags the last content block of the system message
        with `cache_control` so static system prompt content is cached.
    - **Tools**: Tags the last tool definition with `cache_control`. Because
        tool definitions are sent as one contiguous block, a single trailing
        breakpoint caches the entire tool set across turns.
    - **`model_settings`**: Passes `cache_control` to the chat model. The chat
        model/provider then applies the correct message-tail and
        provider-specific behavior at request time.

    了解更多关于 Anthropic 提示词缓存的信息
    [这里](https://platform.claude.com/docs/en/build-with-claude/prompt-caching)。
    """

    trace_policy = _TRACE_POLICY

    def __init__(
        self,
        type: Literal["ephemeral"] = "ephemeral",  # noqa: A002
        ttl: Literal["5m", "1h"] = "5m",
        min_messages_to_cache: int = 0,
        unsupported_model_behavior: Literal["ignore", "warn", "raise"] = "warn",
    ) -> None:
        """使用缓存控制设置初始化中间件。

        Args:
            type: The type of cache to use, only `'ephemeral'` is supported.
            ttl: The time to live for the cache, only `'5m'` and `'1h'` are
                supported.
            min_messages_to_cache: The minimum number of messages until the
                cache is used.
            unsupported_model_behavior: The behavior to take when an
                unsupported model is used.

                `'ignore'` 将忽略不支持的模型并继续运行，不进行缓存。

                `'warn'` 将警告用户并继续运行，不进行缓存。

                `'raise'` 将引发错误并停止智能体。
        """
        self.type = type
        self.ttl = ttl
        self.min_messages_to_cache = min_messages_to_cache
        self.unsupported_model_behavior = unsupported_model_behavior

    @property
    def _cache_control(self) -> dict[str, str]:
        return {"type": self.type, "ttl": self.ttl}

    def _should_apply_caching(self, request: ModelRequest) -> bool:
        """检查是否应对请求应用缓存。

        Args:
            request: 要检查的模型请求。

        Returns:
            如果应应用缓存，则为 `True`，否则为 `False`。

        Raises:
            ValueError: 如果模型不受支持且行为设置为 `'raise'`。
        """
        if not isinstance(request.model, ChatAnthropic):
            msg = (
                "AnthropicPromptCachingMiddleware caching middleware only supports "
                f"Anthropic models, not instances of {type(request.model)}"
            )
            if self.unsupported_model_behavior == "raise":
                raise ValueError(msg)
            if self.unsupported_model_behavior == "warn":
                warn(msg, stacklevel=3)
            return False

        messages_count = (
            len(request.messages) + 1
            if request.system_message
            else len(request.messages)
        )
        return messages_count >= self.min_messages_to_cache

    def _apply_caching(self, request: ModelRequest) -> ModelRequest:
        """对系统消息、工具和模型设置应用缓存控制。

        Args:
            request: 要修改的模型请求。

        Returns:
            应用缓存控制后的新请求。
        """
        overrides: dict[str, Any] = {}
        cache_control = self._cache_control

                # 始终在模型设置的顶层设置 `cache_control`。Anthropic
                # 聊天模型会将该参数转换为正确传输格式，
                # 直接 API 以原样接收，而 Bedrock 则由 `_get_request_payload`
        # 在块级别展开为断点。
                # 直接 API 以原样接收，而 Bedrock 则由 `_get_request_payload`
        # 在块级别展开为断点。
        overrides["model_settings"] = {
            **request.model_settings,
            "cache_control": cache_control,
        }

        system_message = _tag_system_message(request.system_message, cache_control)
        if system_message is not request.system_message:
            overrides["system_message"] = system_message

        tools = _tag_tools(request.tools, cache_control)
        if tools is not request.tools:
            overrides["tools"] = tools

        return request.override(**overrides)

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelCallResult:
        """修改模型请求以添加缓存控制块。

        Args:
            request: 可能需要修改的模型请求。
            handler: 执行模型请求的处理器。

        Returns:
            处理器返回的模型响应。
        """
        if not self._should_apply_caching(request):
            return handler(request)

        return handler(self._apply_caching(request))

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelCallResult:
        """修改模型请求以添加缓存控制块（异步版本）。

        Args:
            request: 可能需要修改的模型请求。
            handler: 异步执行模型请求的处理器。

        Returns:
            处理器返回的模型响应。
        """
        if not self._should_apply_caching(request):
            return await handler(request)

        return await handler(self._apply_caching(request))


def _tag_system_message(
    system_message: Any,
    cache_control: dict[str, str],
) -> Any:
    """将系统消息的最后一个内容块标记为 cache_control。

    Returns the original system_message unchanged if there are no blocks
    to tag.

    Args:
        system_message: 要标记的系统消息。
        cache_control: 要应用的缓存控制字典。

    Returns:
        A new SystemMessage with cache_control on the last block, or the
        original if no modification was needed.
    """
    if system_message is None:
        return system_message

    content = system_message.content
    if isinstance(content, str):
        if not content:
            return system_message
        new_content: list[str | dict[str, Any]] = [
            {"type": "text", "text": content, "cache_control": cache_control}
        ]
    elif isinstance(content, list):
        if not content:
            return system_message
        new_content = list(content)
        last = new_content[-1]
        base = last if isinstance(last, dict) else {}
        new_content[-1] = {**base, "cache_control": cache_control}
    else:
        return system_message

    return SystemMessage(content=new_content)


def _tag_tools(
    tools: list[Any] | None,
    cache_control: dict[str, str],
) -> list[Any] | None:
    """通过 extras 字典将 cache_control 标记到最后一个工具上。

    仅标记最后一个工具以最小化显式缓存断点数量（Anthropic 限制每请求最多 4
    个）。由于工具定义作为连续块发送，对最后一个工具的单个断点即可缓存整个工具集。

    Creates a copy of the last tool with cache_control added to extras,
    without mutating the original.

    Args:
        tools: 要标记的工具列表。
        cache_control: 要应用的缓存控制字典。

    Returns:
        A new list with cache_control on the last tool's extras, or the
        original if no tools are present.
    """
    if not tools:
        return tools

    last = tools[-1]
    if not isinstance(last, BaseTool):
        return tools

    new_extras = {**(last.extras or {}), "cache_control": cache_control}
    return [*tools[:-1], last.model_copy(update={"extras": new_extras})]

"""上下文编辑中间件。

通过在对话超过可配置的 token 阈值后清除较旧的工具结果，模拟 Anthropic 的上下文编辑功能。

The implementation is intentionally model-agnostic so it can be used with any LangChain
chat model.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Iterable, Sequence
from copy import deepcopy
from dataclasses import dataclass
from typing import Literal

from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    BaseMessage,
    ToolMessage,
)
from langchain_core.messages.utils import count_tokens_approximately
from typing_extensions import Protocol

from langchain.agents.middleware.types import (
    AgentMiddleware,
    AgentState,
    ContextT,
    ModelRequest,
    ModelResponse,
    ResponseT,
)

DEFAULT_TOOL_PLACEHOLDER = "[cleared]"


TokenCounter = Callable[
    [Sequence[BaseMessage]],
    int,
]


class ContextEdit(Protocol):
    """描述上下文编辑策略的协议。"""

    def apply(
        self,
        messages: list[AnyMessage],
        *,
        count_tokens: TokenCounter,
    ) -> None:
        """就地对消息列表应用一次编辑。"""
        ...


@dataclass(slots=True)
class ClearToolUsesEdit(ContextEdit):
    """在超出 token 限制时清除工具输出的配置。"""

    trigger: int = 100_000
    """触发编辑的 token 数量。"""

    clear_at_least: int = 0
    """执行编辑时必须回收的最少 token 数量。"""

    keep: int = 3
    """必须保留的最近工具结果数量。"""

    clear_tool_inputs: bool = False
    """是否清除 AI 消息中原始工具调用的参数。"""

    exclude_tools: Sequence[str] = ()
    """排除在清除之外的工具名称列表。"""

    placeholder: str = DEFAULT_TOOL_PLACEHOLDER
    """用于替换被清除工具输出的占位文本。"""

    def apply(
        self,
        messages: list[AnyMessage],
        *,
        count_tokens: TokenCounter,
    ) -> None:
        """应用清除工具使用情况的策略。"""
        tokens = count_tokens(messages)

        if tokens <= self.trigger:
            return

        candidates = [
            (idx, msg) for idx, msg in enumerate(messages) if isinstance(msg, ToolMessage)
        ]

        if self.keep >= len(candidates):
            candidates = []
        elif self.keep:
            candidates = candidates[: -self.keep]

        cleared_tokens = 0
        excluded_tools = set(self.exclude_tools)

        for idx, tool_message in candidates:
            if tool_message.response_metadata.get("context_editing", {}).get("cleared"):
                continue

            ai_message = next(
                (m for m in reversed(messages[:idx]) if isinstance(m, AIMessage)), None
            )

            if ai_message is None:
                continue

            tool_call = next(
                (
                    call
                    for call in ai_message.tool_calls
                    if call.get("id") == tool_message.tool_call_id
                ),
                None,
            )

            if tool_call is None:
                continue

            if (tool_message.name or tool_call["name"]) in excluded_tools:
                continue

            messages[idx] = tool_message.model_copy(
                update={
                    "artifact": None,
                    "content": self.placeholder,
                    "response_metadata": {
                        **tool_message.response_metadata,
                        "context_editing": {
                            "cleared": True,
                            "strategy": "clear_tool_uses",
                        },
                    },
                }
            )

            if self.clear_tool_inputs:
                messages[messages.index(ai_message)] = self._build_cleared_tool_input_message(
                    ai_message,
                    tool_message.tool_call_id,
                )

            if self.clear_at_least > 0:
                new_token_count = count_tokens(messages)
                cleared_tokens = max(0, tokens - new_token_count)
                if cleared_tokens >= self.clear_at_least:
                    break

        return

    @staticmethod
    def _build_cleared_tool_input_message(
        message: AIMessage,
        tool_call_id: str,
    ) -> AIMessage:
        updated_tool_calls = []
        cleared_any = False
        for tool_call in message.tool_calls:
            updated_call = dict(tool_call)
            if updated_call.get("id") == tool_call_id:
                updated_call["args"] = {}
                cleared_any = True
            updated_tool_calls.append(updated_call)

        metadata = dict(getattr(message, "response_metadata", {}))
        context_entry = dict(metadata.get("context_editing", {}))
        if cleared_any:
            cleared_ids = set(context_entry.get("cleared_tool_inputs", []))
            cleared_ids.add(tool_call_id)
            context_entry["cleared_tool_inputs"] = sorted(cleared_ids)
            metadata["context_editing"] = context_entry

        return message.model_copy(
            update={
                "tool_calls": updated_tool_calls,
                "response_metadata": metadata,
            }
        )


class ContextEditingMiddleware(AgentMiddleware[AgentState[ResponseT], ContextT, ResponseT]):
    """自动修剪工具结果以管理上下文大小。

    当总输入 token 数量超过配置阈值时，中间件会应用一系列编辑操作。

    Currently the `ClearToolUsesEdit` strategy is supported, aligning with Anthropic's
    `clear_tool_uses_20250919` behavior [(read more)](https://platform.claude.com/docs/en/agents-and-tools/tool-use/memory-tool).
    """

    edits: list[ContextEdit]
    token_count_method: Literal["approximate", "model"]
    token_counter: TokenCounter | None

    def __init__(
        self,
        *,
        edits: Iterable[ContextEdit] | None = None,
        token_count_method: Literal["approximate", "model"] = "approximate",  # noqa: S107
        token_counter: TokenCounter | None = None,
    ) -> None:
        """初始化一个上下文编辑中间件实例。

        Args:
            edits: 要应用的编辑策略序列。

                默认为单个 `ClearToolUsesEdit`，与 Anthropic 默认行为一致。
            token_count_method: 是否使用近似 token 计数
                （更快，准确性较低）或由聊天模型实现的精确计数
                （可能更慢，但更准确）。

                Ignored when `token_counter` is provided.
            token_counter: Optional custom function counting tokens for a
                sequence of messages. Takes precedence over
                `token_count_method` when provided, mirroring
                `SummarizationMiddleware`. Useful when the built-in
                approximation is inaccurate for a workload (e.g. CJK-heavy
                conversations) or when a provider-specific tokenizer should
                be used without implementing
                `get_num_tokens_from_messages`.
        """
        super().__init__()
        self.edits = list(edits or (ClearToolUsesEdit(),))
        self.token_count_method = token_count_method
        self.token_counter = token_counter

    def _resolve_token_counter(self, request: ModelRequest[ContextT]) -> TokenCounter:
        """选择用于评估编辑触发条件的 token 计数器。"""
        if self.token_counter is not None:
            return self.token_counter

        if self.token_count_method == "approximate":  # noqa: S105
            return count_tokens_approximately

        system_msg = [request.system_message] if request.system_message else []

        def count_tokens(messages: Sequence[BaseMessage]) -> int:
            return request.model.get_num_tokens_from_messages(
                system_msg + list(messages), request.tools
            )

        return count_tokens

    def wrap_model_call(
        self,
        request: ModelRequest[ContextT],
        handler: Callable[[ModelRequest[ContextT]], ModelResponse[ResponseT]],
    ) -> ModelResponse[ResponseT] | AIMessage:
        """在通过处理器调用模型之前应用上下文编辑。

        Args:
            request: 要执行的模型请求（包含状态和运行时）。 handler:
            异步回调，执行模型请求并返回 `ModelResponse`。

        Returns:
            使用可能已编辑的消息调用处理器后的结果。
        """
        if not request.messages:
            return handler(request)

        count_tokens = self._resolve_token_counter(request)

        edited_messages = deepcopy(list(request.messages))
        for edit in self.edits:
            edit.apply(edited_messages, count_tokens=count_tokens)

        return handler(request.override(messages=edited_messages))

    async def awrap_model_call(
        self,
        request: ModelRequest[ContextT],
        handler: Callable[[ModelRequest[ContextT]], Awaitable[ModelResponse[ResponseT]]],
    ) -> ModelResponse[ResponseT] | AIMessage:
        """在通过处理器调用模型之前应用上下文编辑。

        Args:
            request: 要执行的模型请求（包含状态和运行时）。 handler:
            异步回调，执行模型请求并返回 `ModelResponse`。

        Returns:
            使用可能已编辑的消息调用处理器后的结果。
        """
        if not request.messages:
            return await handler(request)

        count_tokens = self._resolve_token_counter(request)

        edited_messages = deepcopy(list(request.messages))
        for edit in self.edits:
            edit.apply(edited_messages, count_tokens=count_tokens)

        return await handler(request.override(messages=edited_messages))


__all__ = [
    "ClearToolUsesEdit",
    "ContextEditingMiddleware",
]

"""提供方侧工具搜索中间件。"""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeAlias

from langchain_core.tools import BaseTool
from typing_extensions import NotRequired, TypedDict

from langchain.chat_models.base import _attempt_infer_model_provider

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.messages import AIMessage

from langchain.agents.middleware.types import (
    AgentMiddleware,
    AgentState,
    ContextT,
    ModelRequest,
    ModelResponse,
    ResponseT,
)

ToolIdentifier: TypeAlias = str | BaseTool
"""Tool name or tool instance that can be deferred behind provider tool search."""


class _ServerToolSearchSpec(TypedDict):
    """作为工具发送给模型的提供方原生工具搜索工具描述符。"""

    type: str
    name: NotRequired[str]


# 按规范化提供方名称（见
# `_normalize_provider`）索引的原生提供方工具搜索描述符。此映射是唯一真实来源，用于判断哪些提供方支持服务端工具搜索。
# 提供方原生工具搜索描述符按规范化提供方名称（见
# `_normalize_provider`）索引。此映射是唯一真实来源，用于判断哪些提供方支持服务端工具搜索。
# 此映射是唯一真实来源，用于判断哪些提供方支持服务端工具搜索。
#
# 以下标识符由提供方版本标记，可能过期；更新时请重新对照提供方文档验证：
# 以下标识符由提供方版本标记，可能过期；更新时请重新对照提供方文档验证：
# - Anthropic:
# https://docs.langchain.com/oss/python/integrations/chat/anthropic#tool-search
# - OpenAI: 服务端 `tool_search` 工具。
_SERVER_TOOL_SEARCH_TOOLS: dict[str, _ServerToolSearchSpec] = {
    "anthropic": {
        "type": "tool_search_tool_bm25_20251119",
        "name": "tool_search_tool_bm25",
    },
    "openai": {"type": "tool_search"},
}


class ProviderToolSearchMiddleware(AgentMiddleware[AgentState[ResponseT], ContextT, ResponseT]):
    """将选定的工具延迟至提供方原生工具搜索之后。

    不是每次调用都发送每个工具的模式，此中间件通过 `extras["defer_loading"]`
    标记选定的工具为延迟加载，并注入提供方的服务器端工具搜索工具。当模型需要时，提供方才检索延迟工具的完整模式，从而在绑定多个工具时保持请求负载较小。

    当工具名称（或实例）传入 `searchable_tools` 时，或工具已携带
    `extras["defer_loading"] is True` 时，该工具会被延迟。

    Only providers with server-side tool search are supported (currently
    Anthropic and OpenAI). The provider is inferred from the bound model.

    !!! warning

        此功能依赖于提供方原生工具搜索，且仅对受支持的提供方生效。如果工具被延迟但模型的提供方无法识别或不支持工具搜索，则模型调用会抛出
        `ValueError`。当没有工具被延迟时，无论提供方如何，中间件均原样传递请求。

    Example:
        ```python
        from langchain.agents import create_agent
        from langchain.agents.middleware import ProviderToolSearchMiddleware

        agent = create_agent(
            "anthropic:claude-opus-4-8",
            tools=[get_weather, send_email, lookup_order],
            middleware=[ProviderToolSearchMiddleware(searchable_tools=["lookup_order"])],
        )
        ```
    """

    def __init__(self, *, searchable_tools: list[ToolIdentifier] | None = None) -> None:
        """初始化提供方侧工具搜索。

        Args:
            searchable_tools: 要延迟至提供方原生工具搜索的工具或工具名称。
        """
        super().__init__()
        self.searchable_tool_names = _to_tool_names(searchable_tools)

    def _prepare_request(self, request: ModelRequest[ContextT]) -> ModelRequest[ContextT]:
        """使用延迟工具和提供方搜索准备模型请求。

        验证 `searchable_tools`
        中的每个名称均已绑定到模型，然后（仅当至少有一个工具被延迟时）解析模型的提供方并注入提供方原生工具搜索工具。无延迟工具的请求则原样通过。

        Args:
            request: 要准备的模型请求。

        Returns:
            The original request when nothing is deferred, otherwise a new
            request with deferred tools and the provider search tool appended.

        Raises:
            ValueError: If `searchable_tools` references a tool not bound to the
                model, or if a tool is deferred but the model's provider cannot
                be identified or does not support server-side tool search.
        """
        tools = request.tools
        if self.searchable_tool_names:
            available = {tool.name for tool in tools if isinstance(tool, BaseTool)}
            unknown = sorted(self.searchable_tool_names - available)
            if unknown:
                msg = (
                    "ProviderToolSearchMiddleware: searchable_tools references "
                    f"tool(s) not bound to the model: {', '.join(unknown)}"
                )
                raise ValueError(msg)

        if not any(_is_deferred_tool(tool, self.searchable_tool_names) for tool in tools):
            return request

        provider = _get_model_provider(request.model, request.runtime)
        if provider is None:
            msg = (
                "ProviderToolSearchMiddleware could not determine the provider for "
                f"model {request.model.__class__.__name__!r}; server-side tool search "
                f"supports: {', '.join(sorted(_SERVER_TOOL_SEARCH_TOOLS))}"
            )
            raise ValueError(msg)
        if provider not in _SERVER_TOOL_SEARCH_TOOLS:
            msg = (
                "ProviderToolSearchMiddleware requires a provider with server-side "
                f"tool search, but got {provider!r}; supported providers: "
                f"{', '.join(sorted(_SERVER_TOOL_SEARCH_TOOLS))}"
            )
            raise ValueError(msg)

        bound_tools = [_defer_tool_if_needed(tool, self.searchable_tool_names) for tool in tools]
        return request.override(tools=[*bound_tools, dict(_SERVER_TOOL_SEARCH_TOOLS[provider])])

    def wrap_model_call(
        self,
        request: ModelRequest[ContextT],
        handler: Callable[[ModelRequest[ContextT]], ModelResponse[ResponseT]],
    ) -> ModelResponse[ResponseT] | AIMessage:
        """在调用模型前延迟工具。

        Args:
            request: 要执行的模型请求。
            handler: 执行模型请求的回调。

        Returns:
            模型调用结果。

        Raises:
            ValueError: If `searchable_tools` references a tool not bound to the
                model, or if a tool is deferred but the model's provider cannot
                be identified or does not support server-side tool search.
        """
        return handler(self._prepare_request(request))

    async def awrap_model_call(
        self,
        request: ModelRequest[ContextT],
        handler: Callable[[ModelRequest[ContextT]], Awaitable[ModelResponse[ResponseT]]],
    ) -> ModelResponse[ResponseT] | AIMessage:
        """在异步调用模型前延迟工具。

        Args:
            request: 要执行的模型请求。
            handler: 执行模型请求的回调。

        Returns:
            模型调用结果。

        Raises:
            ValueError: If `searchable_tools` references a tool not bound to the
                model, or if a tool is deferred but the model's provider cannot
                be identified or does not support server-side tool search.
        """
        return await handler(self._prepare_request(request))


def _to_tool_names(tools: list[ToolIdentifier] | None) -> set[str]:
    """将工具标识符转换为名称。"""
    if tools is None:
        return set()
    return {tool if isinstance(tool, str) else tool.name for tool in tools}


def _is_deferred_tool(tool: BaseTool | dict[str, Any], tool_names: set[str]) -> bool:
    """返回是否应延迟某个工具。

    只有 `BaseTool` 实例可以被延迟；字典形式的工具（如提供方工具规范）没有 `extras`
    或名称可供匹配，因此从不被延迟。
    """
    if not isinstance(tool, BaseTool):
        return False
    extras = tool.extras if isinstance(tool.extras, dict) else {}
    return extras.get("defer_loading") is True or tool.name in tool_names


def _defer_tool_if_needed(
    tool: BaseTool | dict[str, Any], tool_names: set[str]
) -> BaseTool | dict[str, Any]:
    """返回设置 `defer_loading` 的工具，否则返回未修改的工具。

    当工具不应被延迟或不是 `BaseTool` 时，返回输入不变（只有 `BaseTool`
    实例携带用于标记延迟的 `extras`）。
    """
    if not _is_deferred_tool(tool, tool_names):
        return tool
    if not isinstance(tool, BaseTool):
        return tool
    extras = {**(tool.extras or {}), "defer_loading": True}
    return tool.model_copy(update={"extras": extras})


def _get_model_provider(model: BaseChatModel, runtime: Any) -> str | None:
    """推断用于服务器端工具搜索的标准化提供方名称。

    Returns `None` when no provider can be identified, so callers can
    distinguish a detection failure from a provider that is simply unsupported.
    """
    default_config = getattr(model, "_default_config", None)
    model_params_fn = getattr(model, "_model_params", None)
    if callable(model_params_fn):
        config = getattr(runtime, "config", None)
                # `_model_params` 需要一个配置映射（或 None）；将格式错误的非映射配置强制转换为
        # None，使其被视为 "无配置" 而非在可配置模型内部抛出异常。
                # 将格式错误的非映射配置强制转换为 None，使其被视为 "无配置"
        # 而非在可配置模型内部抛出异常。
                # 将格式错误的非映射配置强制转换为 None，使其被视为 "无配置"
        # 而非在可配置模型内部抛出异常。
        if config is not None and not isinstance(config, Mapping):
            config = None
        model_params = model_params_fn(config)
        if isinstance(model_params, dict):
            params = (
                {**default_config, **model_params}
                if isinstance(default_config, dict)
                else model_params
            )
            if provider := _provider_from_params(params):
                return provider

    if isinstance(default_config, dict) and (provider := _provider_from_params(default_config)):
        return provider

    get_ls_params = getattr(model, "_get_ls_params", None)
    if callable(get_ls_params):
        ls_params = get_ls_params()
        if isinstance(ls_params, dict) and isinstance(ls_params.get("ls_provider"), str):
            return _normalize_provider(ls_params["ls_provider"])

    return _provider_from_class_name(model.__class__.__name__)


def _provider_from_params(params: dict[str, Any]) -> str | None:
    """从模型参数推断提供方，若不存在则返回 `None`。"""
    provider = params.get("model_provider")
    if isinstance(provider, str):
        return _normalize_provider(provider)
    model_name = params.get("model")
    if isinstance(model_name, str):
        return _provider_from_model_name(model_name)
    return None


def _provider_from_model_name(model_name: str) -> str | None:
    """从模型名称推断提供方，若无法识别则返回 `None`。"""
    provider, _, rest = model_name.partition(":")
    if rest:
        return _normalize_provider(provider)
        # 推断出的提供方仅用于注册表查找，因此抑制 `_attempt_infer_model_provider` 发出的
    # `model_provider` 推断弃用警告（例如 `gemini*` 等名称）；其指导意义与路由无关。
        # 推断出的提供方仅用于注册表查找，因此抑制 `_attempt_infer_model_provider` 发出的
    # `model_provider` 推断弃用警告（例如 `gemini*` 等名称）；其指导意义与路由无关。
        # 推断出的提供方仅用于注册表查找，因此抑制 `_attempt_infer_model_provider` 发出的
    # `model_provider` 推断弃用警告（例如 `gemini*` 等名称）；其指导意义与路由无关。
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        inferred = _attempt_infer_model_provider(model_name)
    return _normalize_provider(inferred) if inferred else None


def _provider_from_class_name(class_name: str) -> str | None:
    """从模型类名推断提供方，若无法识别则返回 `None`。"""
    if class_name in {"ChatAnthropic", "AnthropicChat"}:
        return "anthropic"
    if class_name in {"ChatOpenAI", "OpenAIChat"}:
        return "openai"
    return None


def _normalize_provider(provider: str) -> str:
    """通过小写化并将 `-` 替换为 `_` 来规范化提供方标识符。"""
    return provider.replace("-", "_").lower()

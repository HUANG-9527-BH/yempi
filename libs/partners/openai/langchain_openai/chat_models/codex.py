"""`_ChatOpenAICodex`: 实验性的基于 OAuth 认证的聊天模型。

Wraps `ChatOpenAI` to target the ChatGPT codex backend
(`https://chatgpt.com/backend-api/codex`) and supplies refresh-aware
`Authorization` and `ChatGPT-Account-Id` headers from a
`_ChatGPTOAuthTokenProvider`.

标准的 `ChatOpenAI`（API 密钥）流程保持不变。

!!! warning "实验性和非官方"

    `_ChatOpenAICodex` is not an official OpenAI API integration. Use it only
    where your OpenAI account, workspace, plan, and applicable OpenAI terms
    permit ChatGPT-authenticated Codex access. You are responsible for ensuring
    your implementation complies with OpenAI's terms, usage policies, account
    restrictions, rate limits, and safeguards.
"""

from __future__ import annotations

import logging
import os
import warnings
from typing import TYPE_CHECKING, Any

from langchain_core.language_models.chat_models import LangSmithParams
from langchain_core.messages import BaseMessage, ChatMessage, SystemMessage
from pydantic import Field, model_validator

from langchain_openai.chat_models.base import ChatOpenAI
from langchain_openai.chatgpt_oauth import (
    _ChatGPTOAuthTokenProvider,
    _FileChatGPTOAuthTokenProvider,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from langchain_core.callbacks import AsyncCallbackManagerForLLMRun
    from langchain_core.language_models import LanguageModelInput, ModelProfile
    from langchain_core.outputs import ChatGenerationChunk, ChatResult


logger = logging.getLogger(__name__)


CHATGPT_CODEX_BASE_URL = "https://chatgpt.com/backend-api/codex"
ORIGINATOR_HEADER = "originator"
ORIGINATOR_VALUE = "langchain"
"""Built-in default for the `originator` header value.

Identifies requests as coming from `langchain-openai`. Override per-instance
via the `originator` field or globally via the `LANGCHAIN_CODEX_ORIGINATOR`
env var.
"""
ORIGINATOR_ENV_VAR = "LANGCHAIN_CODEX_ORIGINATOR"
ACCOUNT_ID_HEADER = "ChatGPT-Account-Id"
_CODEX_HEADERS_KWARG = "_codex_headers"
"""Private kwarg used to hand pre-built Codex headers to `_get_request_payload`.

The async `_agenerate`/`_astream` paths build the headers from a token fetched
off the event loop (via `aget_token`) and pass them through this kwarg so the
sync payload builder doesn't fall back to `_codex_headers_sync` — which would
acquire a thread + cross-process file lock on the loop. Leading underscore keeps
it out of the public surface; it is popped before the payload reaches the SDK.
"""
EXPERIMENTAL_UNOFFICIAL_WARNING = (
    "`_ChatOpenAICodex` is experimental and unofficial. It uses ChatGPT "
    "subscription OAuth against Codex endpoints and must only be used where "
    "permitted by your OpenAI account, workspace, plan, and applicable OpenAI "
    "terms and policies. You are responsible for implementing and operating "
    "it responsibly, including respecting OpenAI's usage policies, rate "
    "limits, and safeguards."
)
_experimental_warning_emitted = False
_INSTRUCTION_ROLES = frozenset({"system", "developer"})


def _default_originator() -> str:
    """解析 `originator` 头部默认值，尊重环境变量覆盖。"""
    return os.environ.get(ORIGINATOR_ENV_VAR) or ORIGINATOR_VALUE


def _warn_experimental_unofficial() -> None:
    """仅警告一次 `_ChatOpenAICodex` 为实验性和非官方。"""
    global _experimental_warning_emitted
    if _experimental_warning_emitted:
        return
    _experimental_warning_emitted = True
    warnings.warn(EXPERIMENTAL_UNOFFICIAL_WARNING, UserWarning, stacklevel=5)


def _maybe_has_system_messages(input_: Any) -> bool:
    """如果 `input_` *可能* 包含 system-role 消息，则返回 `True`。

    Cheap structural probe used to skip the full `_convert_input` pipeline
    when there is no chance the lift logic will fire. False positives only
    cost an extra conversion; false negatives would silently skip the lift,
    so the probe is biased toward `True` for unknown shapes.
    """
    if isinstance(input_, str):
        return False
    if isinstance(input_, BaseMessage):
        return _is_instruction_message(input_)
    if isinstance(input_, (list, tuple)):
        for item in input_:
            if isinstance(item, BaseMessage) and _is_instruction_message(item):
                return True
            if isinstance(item, dict) and item.get("role") in _INSTRUCTION_ROLES:
                return True
            if (
                isinstance(item, tuple)
                and item
                and isinstance(item[0], str)
                and item[0] in _INSTRUCTION_ROLES
            ):
                return True
        return False
        # `PromptValue` 或任何未来形状——为安全起见，运行慢路径。
    return True


def _is_instruction_message(message: BaseMessage) -> bool:
    return isinstance(message, SystemMessage) or (
        isinstance(message, ChatMessage) and message.role in _INSTRUCTION_ROLES
    )


def _flatten_system_message_content(system_messages: list[BaseMessage]) -> str:
    """将 system/developer 消息内容合并为单一的 `instructions` 字符串。

    Codex rejects system-role entries in the input list, so their content
    is lifted into the top-level `instructions` field. Content that uses
    list-of-content-blocks form is accepted only when every block is
    `{"type": "text", ...}`; anything else cannot be flattened into the
    string-typed `instructions` field.

    Raises:
        ValueError: system/developer 消息携带了非文本内容块。
    """
    parts: list[str] = []
    for index, message in enumerate(system_messages):
        message_name = type(message).__name__
        content = message.content
        if isinstance(content, str):
            parts.append(content)
            continue
        if not isinstance(content, list):
            msg = (
                f"`{message_name}` at index {index} has unsupported content "
                f"type {type(content).__name__!r}; only `str` and "
                "list-of-text-blocks are accepted by `_ChatOpenAICodex`."
            )
            raise ValueError(msg)
        text_parts: list[str] = []
        for block_index, block in enumerate(content):
            if not isinstance(block, dict) or block.get("type") != "text":
                msg = (
                    f"`{message_name}` at index {index} contains a "
                    f"non-text content block at position {block_index} "
                    "(Codex `instructions` is a string field — only "
                    '`{"type": "text", "text": "..."}` blocks can be '
                    "lifted into it). Move the non-text content to a "
                    "`HumanMessage`, or pass plain instructions via the "
                    "constructor or `instructions=` kwarg."
                )
                raise ValueError(msg)
            text_value = block.get("text", "")
            if not isinstance(text_value, str):
                msg = (
                    f"`{message_name}` at index {index} has a text block "
                    f"at position {block_index} whose `text` is not a "
                    "string."
                )
                raise ValueError(msg)
            text_parts.append(text_value)
        parts.append("".join(text_parts))
    return "\n\n".join(parts)


DEFAULT_INSTRUCTIONS = "You are ChatGPT, a large language model trained by OpenAI."
"""Generic fallback for the Responses-API `instructions` field.

The Codex backend rejects any request missing a top-level `instructions`
value (400 `Instructions are required`), so this constant keeps zero-config
construction working. **Most callers should override it** with their own
prompt — see `_ChatOpenAICodex.instructions` for the resolution rules.
"""
_FORCED_VALUES: dict[str, Any] = {
    "use_responses_api": True,
    "store": False,
    "streaming": True,
}
"""Values forced onto every `_ChatOpenAICodex` instance.

These are the wire-level constraints the Codex backend imposes:

- `use_responses_api=True`: Codex is only reachable through the Responses
    API surface.
- `store=False`: the backend rejects `store=true`
    (`400 'Store must be set to false'`).
- `streaming=True`: the backend rejects non-streaming requests
    (`400 'Stream must be set to true'`). Pinning this routes `invoke`
    through `_stream` so a streaming request is always sent and chunks
    are aggregated back into a single message for the caller.

`output_version` is intentionally **not** forced — it is a client-side
`AIMessage` projection (see `ChatOpenAI.output_version`) that never
appears in the request payload, so callers can pick `"v0"`, `"v1"`, or
`"responses/v1"` freely.

`base_url` (and its `openai_api_base` alias) is also pinned — to
`CHATGPT_CODEX_BASE_URL` — under the same raise-don't-rewrite contract.
It is enforced separately in the validator rather than listed here
because a caller-controlled endpoint combined with the OAuth bearer
token would be a token-exfiltration vector; see the validator for the
rationale.
"""


class _ChatOpenAICodex(ChatOpenAI):
    """实验性的 `ChatOpenAI` 变体，通过 ChatGPT OAuth 认证。

    This integration is unofficial and should only be used where your OpenAI
    account, workspace, plan, and applicable OpenAI terms permit
    ChatGPT-authenticated Codex access. Users are responsible for implementing
    and operating it in compliance with OpenAI's terms, usage policies, account
    restrictions, rate limits, and safeguards.

    Routes requests to `https://chatgpt.com/backend-api/codex` and forces
    the wire-level fields the Codex backend requires
    (`use_responses_api=True`, `store=False`, `streaming=True`). These
    values are forced — passing a conflicting value to the constructor
    raises. `output_version` (a client-side `AIMessage` projection) is
    not forced; pick whichever projection you want. Authorization and
    `ChatGPT-Account-Id` headers are taken from `token_provider` on every
    request so a freshly-refreshed access token is always used.

    Example:
        ```python
        from langchain_openai.chat_models.codex import _ChatOpenAICodex
        from langchain_openai.chatgpt_oauth import login_chatgpt

        # One-time setup. The returned provider writes to the default store
        # at `~/.langchain/chatgpt-auth.json`, which `_ChatOpenAICodex` also
        # reads from by default — so subsequent constructions need no
        # explicit `token_provider`.
        login_chatgpt()
        model = _ChatOpenAICodex(
            model="gpt-5.5",
            instructions="You are a senior Python reviewer. Be terse.",
        )
        response = model.invoke("hello")
        ```

    !!! tip "覆盖 `instructions`"

        The Codex backend requires a top-level `instructions` value on every
        request. A generic default keeps zero-config use working, but most
        callers should override it via the constructor (above) or per call
        (`model.invoke(..., instructions=...)`). See the field's docstring
        for the full resolution rules.

    !!! note

        Token storage is handled by `_FileChatGPTOAuthTokenProvider`, which
        defaults to `~/.langchain/chatgpt-auth.json` so it does not collide
        with the Codex CLI / VS Code session at `~/.codex/auth.json`.

    !!! note "始终流式传输"

        The Codex backend only accepts streaming requests, so `streaming=True`
        is forced. `invoke` still returns a single aggregated `AIMessage` —
        chunks are collected internally — but the underlying HTTP request is
        a stream either way. Expect every call to show up as a streamed
        request in network logs and LangSmith traces.
    """

    token_provider: Any = Field(default=None, exclude=True)
    """支持刷新的 ChatGPT OAuth token 提供者。

    必须实现 `_ChatGPTOAuthTokenProvider` 协议。若为
    `None`，则会基于默认存储路径构造一个 `_FileChatGPTOAuthTokenProvider`。
    """

    originator: str | None = Field(default_factory=_default_originator)
    """在 `originator` 请求头中发送的值，或 `None` 以省略该头。

    用于标识发起请求的客户端。默认为 `"langchain"`，以便 OpenAI
    将此包的调用归入遥测数据。下游使用者（例如基于 `_ChatOpenAICodex`
    构建的框架）可将其覆盖为自身标识，或设置 `None` 以禁用该头。

    解析顺序（首个匹配项优先）：

    1. Per-call `extra_headers={"originator": "..."}` (always trumps the
        field; pass an explicit value to override on a single call).
    2. Constructor / kwarg value (`_ChatOpenAICodex(originator="my-app")`).
    3. The `LANGCHAIN_CODEX_ORIGINATOR` env var, if set and non-empty.
    4. `ORIGINATOR_VALUE` (`"langchain"`).

    Setting `originator=None` disables the header entirely; the constructor
    default never resolves to `None`.
    """

    instructions: str = Field(default=DEFAULT_INSTRUCTIONS)
    """在 Responses-API 的 `instructions` 字段中发送的系统提示词。

    `instructions` 是 Responses API 请求的 *顶层* 字段 —— 它不是聊天消息。Codex
    后端会拒绝任何缺少或为空的此字段的请求（400 `Instructions are
    required`）**且**拒绝输入列表中的任何 `SystemMessage` 条目（400 `System messages
    are not allowed`）。为透明地弥合这些限制，`_ChatOpenAICodex`
    在每次调用时按以下优先级解析 `instructions`（最高优先级胜出）：

    1. Explicit `instructions=` kwarg on `invoke` / `stream`.
    2. Concatenated content of any `SystemMessage` entries in the input
        list — joined with `"\\n\\n"` and stripped from the input before
        sending. Set the explicit kwarg in (1) to override.
    3. This constructor field (defaults to a generic ChatGPT prompt).

    The Codex backend is stateless for this client (`store=False` is
    forced), so `instructions` is sent on every request and can be changed
    between calls — useful for switching persona / tooling mid-conversation:

    ```python
    model = _ChatOpenAICodex(
        model="gpt-5.5",
        instructions="You are a senior Python reviewer. Be terse.",
    )
    model.invoke("review this diff…")
    model.invoke(
        "now translate the review to French",
        instructions="You are a translator.",
    )
    ```

    `SystemMessage` content that uses list-of-content-blocks form is
    accepted only if every block is `{"type": "text", ...}`; any other
    block type raises `ValueError` since it cannot be flattened into the
    string-typed `instructions` field.
    """

    @model_validator(mode="before")
    @classmethod
    def _apply_codex_defaults(cls, values: dict[str, Any]) -> dict[str, Any]:
        """在父验证器运行前应用 Codex 特有的默认值。"""
        _warn_experimental_unofficial()
        if not isinstance(values, dict):
            return values
        for key, forced in _FORCED_VALUES.items():
            supplied = values.get(key)
            if supplied is not None and supplied != forced:
                msg = (
                    f"`_ChatOpenAICodex` requires `{key}={forced!r}`; "
                    f"got `{key}={supplied!r}`. Use `ChatOpenAI` if you "
                    "need to customize this."
                )
                raise ValueError(msg)
            values[key] = forced
                # 将 `base_url`（及其旧版别名 `openai_api_base`）固定为 Codex
                # 端点。OAuth bearer token 以下方方式作为 `api_key` 绑定，因此调用者控制的
        # `base_url` 会将 token
        # 泄露到攻击者选择的主机上。拒绝任何不匹配的覆盖，而非静默重写，与
        # `_FORCED_VALUES` 约定保持一致。
                # 端点。OAuth bearer token 以下方方式作为 `api_key` 绑定，因此调用者控制的
        # `base_url` 会将 token
        # 泄露到攻击者选择的主机上。拒绝任何不匹配的覆盖，而非静默重写，与
        # `_FORCED_VALUES` 约定保持一致。
                # 端点。OAuth bearer token 以下方方式作为 `api_key` 绑定，因此调用者控制的
        # `base_url` 会将 token
        # 泄露到攻击者选择的主机上。拒绝任何不匹配的覆盖，而非静默重写，与
        # `_FORCED_VALUES` 约定保持一致。
                # 端点。OAuth bearer token 以下方方式作为 `api_key` 绑定，因此调用者控制的
        # `base_url` 会将 token
        # 泄露到攻击者选择的主机上。拒绝任何不匹配的覆盖，而非静默重写，与
        # `_FORCED_VALUES` 约定保持一致。
        for key in ("base_url", "openai_api_base"):
            supplied = values.get(key)
            if supplied is not None and supplied != CHATGPT_CODEX_BASE_URL:
                msg = (
                    f"`_ChatOpenAICodex` requires `{key}={CHATGPT_CODEX_BASE_URL!r}`; "
                    f"got `{key}={supplied!r}`. Use `ChatOpenAI` if you need to "
                    "target a different endpoint."
                )
                raise ValueError(msg)
            values[key] = CHATGPT_CODEX_BASE_URL

        provider = values.get("token_provider")
        if provider is None:
            provider = _FileChatGPTOAuthTokenProvider.from_default_store()
            values["token_provider"] = provider
        if not isinstance(provider, _ChatGPTOAuthTokenProvider):
            msg = (
                "`token_provider` must implement the "
                "`_ChatGPTOAuthTokenProvider` protocol."
            )
            raise TypeError(msg)

                # OAuth `token_provider` 是唯一的认证来源：其访问令牌以下方方式作为 `api_key`
        # 绑定到 OpenAI SDK。调用者提供的 `api_key`（或其 `openai_api_key`
        # 别名）会无声地压倒 OAuth bearer
        # token，导致模型处于冲突状态——因此应拒绝它（抛出异常而非重写，与上方
        # `base_url` 处理方式一致）。
                # OAuth `token_provider` 是唯一的认证来源：其访问令牌以下方方式作为 `api_key`
        # 绑定到 OpenAI SDK。调用者提供的 `api_key`（或其 `openai_api_key`
        # 别名）会无声地压倒 OAuth bearer
        # token，导致模型处于冲突状态——因此应拒绝它（抛出异常而非重写，与上方
        # `base_url` 处理方式一致）。
                # OAuth `token_provider` 是唯一的认证来源：其访问令牌以下方方式作为 `api_key`
        # 绑定到 OpenAI SDK。调用者提供的 `api_key`（或其 `openai_api_key`
        # 别名）会无声地压倒 OAuth bearer
        # token，导致模型处于冲突状态——因此应拒绝它（抛出异常而非重写，与上方
        # `base_url` 处理方式一致）。
                # OAuth `token_provider` 是唯一的认证来源：其访问令牌以下方方式作为 `api_key`
        # 绑定到 OpenAI SDK。调用者提供的 `api_key`（或其 `openai_api_key`
        # 别名）会无声地压倒 OAuth bearer
        # token，导致模型处于冲突状态——因此应拒绝它（抛出异常而非重写，与上方
        # `base_url` 处理方式一致）。
                # OAuth `token_provider` 是唯一的认证来源：其访问令牌以下方方式作为 `api_key`
        # 绑定到 OpenAI SDK。调用者提供的 `api_key`（或其 `openai_api_key`
        # 别名）会无声地压倒 OAuth bearer
        # token，导致模型处于冲突状态——因此应拒绝它（抛出异常而非重写，与上方
        # `base_url` 处理方式一致）。
                # 不会检查 `OPENAI_API_KEY` 环境变量：因为 `api_key`
        # 总是此处设置，所以该字段的默认工厂从不执行。
                # 调用者提供的 `extra_headers` 会覆盖我们的 Codex
        # 默认值，以便用户可以覆盖（例如，发送不同的 `originator`）。
        for key in ("api_key", "openai_api_key"):
            if values.get(key) is not None:
                msg = (
                    f"`_ChatOpenAICodex` manages authentication via "
                    f"`token_provider`; drop the explicit `{key}=`. Use "
                    "`ChatOpenAI` if you want API-key authentication."
                )
                raise ValueError(msg)
        values["api_key"] = _SyncTokenCallable(provider)
        return values

    def _codex_headers_sync(self) -> dict[str, str]:
        token = self.token_provider.get_token()
        return self._build_headers(token.account_id)

    def _build_headers(self, account_id: str | None) -> dict[str, str]:
        headers: dict[str, str] = {}
        if account_id:
            headers[ACCOUNT_ID_HEADER] = account_id
        if self.originator is not None:
            headers[ORIGINATOR_HEADER] = self.originator
        return headers

    def _merge_codex_headers(
        self, payload: dict[str, Any], headers: dict[str, str]
    ) -> dict[str, Any]:
                # Codex 后端拒绝没有 `instructions`
        # 的请求——如果调用者未提供，则填充该字段的值。调用者显式传入的空字符串会被保留（后端会拒绝它，但静默重写会隐藏编程错误）。
                # 异步调用者可能已提前在事件循环外构建了头部并传递给
        # `_codex_headers`。应原样尊重它们——`is not None`
        # 检查（非真值性）是故意的：显式空字典表示“无头部，已异步决定”，绝不能触发重新生成
        if not headers:
            return payload
        merged = {**headers, **(payload.get("extra_headers") or {})}
        payload["extra_headers"] = merged
        return payload

    def _get_request_payload(
        self,
        input_: LanguageModelInput,
        *,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> dict:
        """构建请求负载并附加 Codex 认证头。

        Lifts any `SystemMessage` content out of the input list into the
        top-level `instructions` field, since Codex rejects `SystemMessage`
        chat turns. See the `instructions` field docstring for the
        precedence rules.

        快速路径：当输入无法携带 `SystemMessage` 时，跳过本地转换，直接将 `input_`
        委托给父类——这样 `_convert_input`
        只运行一次（在父类内部），而不是在这里运行一次，再在父类中运行一次。
        """
        codex_headers = kwargs.pop(_CODEX_HEADERS_KWARG, None)
        payload_input: LanguageModelInput = input_
        if _maybe_has_system_messages(input_):
            messages = self._convert_input(input_).to_messages()
            system_messages = [m for m in messages if _is_instruction_message(m)]
            if system_messages:
                non_system = [m for m in messages if not _is_instruction_message(m)]
                lifted = _flatten_system_message_content(system_messages)
                explicit = kwargs.get("instructions")
                if explicit is not None:
                    logger.warning(
                        "Both `instructions=` and a `SystemMessage` were "
                        "provided; the explicit `instructions=` kwarg wins "
                        "and the `SystemMessage` content is discarded for "
                        "this call. Discarded length: %d.",
                        len(lifted),
                    )
                else:
                    kwargs["instructions"] = lifted
                payload_input = non_system

        payload = super()._get_request_payload(payload_input, stop=stop, **kwargs)
                # Codex 后端会拒绝没有 `instructions` 的请求 —— 请填充
                # 如果调用方未提供该字段的值。显式空
                # 调用方传入的字符串会被保留（后端会拒绝它，但
                # 静默覆盖会隐藏编程错误。
        if payload.get("instructions") is None:
            payload["instructions"] = self.instructions
                # 异步调用方可能已经从事件循环中构建了请求头
                # 并将其通过 `_codex_headers` 传递。原样保留它们——
                # 显式检查 `is not None`（而非真值）是故意的：明确的空值
                # dict 表示 "无标题，已确定异步"，且绝不能触发
                # 同步
        # `get_token()`（在文件锁上阻塞循环）。仅当关键字参数不存在时，纯同步路径在此处读取
        # token。
                # 纯同步路径，其中关键字参数缺失，此处读取 token。
        headers = (
            codex_headers if codex_headers is not None else self._codex_headers_sync()
        )
        return self._merge_codex_headers(payload, headers)

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
                # 从事件循环中获取 token 并在此构建请求头，然后通过 `_codex_headers`
        # 将其传递给同步负载构建器。
                # 这可防止 `'_get_request_payload'`（在 `super()._agenerate`
        # 内部的事件循环中运行）回退到同步的
        # `get_token()`，后者会在事件循环上获取线程和跨进程文件锁。
                # 从事件循环中获取 token 构建请求头（参见 `_agenerate`
        # 了解原因），并将它们传递给同步负载构建器。
                # 系统内容被提升至字符串字段
        # `instructions`，因此它既不能保留在原位，也不能携带工具定义。
                # 线程 + 跨进程文件锁的循环。
        token = await self.token_provider.aget_token()
        kwargs[_CODEX_HEADERS_KWARG] = self._build_headers(token.account_id)
        return await super()._agenerate(
            messages, stop=stop, run_manager=run_manager, **kwargs
        )

    async def _astream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
                # 从事件循环中获取的 token 构建请求头（参见
                # `_agenerate` 用于原因）并将其传递给同步负载构建器。
        token = await self.token_provider.aget_token()
        kwargs[_CODEX_HEADERS_KWARG] = self._build_headers(token.account_id)
        async for chunk in super()._astream(
            messages, stop=stop, run_manager=run_manager, **kwargs
        ):
            yield chunk

    def _get_ls_params(
        self, stop: list[str] | None = None, **kwargs: Any
    ) -> LangSmithParams:
        params = super()._get_ls_params(stop=stop, **kwargs)
        params["ls_provider"] = "openai-codex"
        return params

    def _resolve_model_profile(self) -> ModelProfile | None:
        profile = super()._resolve_model_profile()
        if profile is not None:
                        # 系统内容被提升到字符串 `instructions` 字段中，因此
                        # 它既不能保持原位，也不能携带工具定义。
            profile["mid_conversation_system_messages"] = False
            profile["mid_conversation_tools"] = False
        return profile

    @property
    def _llm_type(self) -> str:
        return "openai-codex-chat"

    @classmethod
    def is_lc_serializable(cls) -> bool:
        """`_ChatOpenAICodex` 不可序列化（持有活跃的 token 提供者）。"""
        return False


class _SyncTokenCallable:
    """OpenAI SDK 的 token 提供者同步调用包装器。

    The OpenAI Python SDK accepts a callable returning a string for `api_key`.
    Wrapping the provider lets the SDK fetch a freshly-refreshed access token
    on every request without exposing the provider's other methods.
    """

    __slots__ = ("_provider",)

    def __init__(self, provider: _ChatGPTOAuthTokenProvider) -> None:
        self._provider = provider

    def __call__(self) -> str:
        return self._provider.get_access_token()


__all__: list[str] = []

"""用于对话式人工智能的聊天模型。"""

from __future__ import annotations

import asyncio
import builtins  # noqa: TC003  # runtime-evaluated; subclass `dict()` shadows the builtin
import contextlib
import json
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Callable, Iterator, Sequence
from functools import cached_property
from operator import itemgetter
from typing import TYPE_CHECKING, Any, Literal, cast, overload

from langchain_protocol.protocol import MessageFinishData
from pydantic import BaseModel, ConfigDict, Field, model_validator
from typing_extensions import Self, override

from langchain_core._api import beta, deprecated, suppress_langchain_deprecation_warning
from langchain_core.caches import BaseCache
from langchain_core.callbacks import (
    AsyncCallbackManager,
    AsyncCallbackManagerForLLMRun,
    CallbackManager,
    CallbackManagerForLLMRun,
    Callbacks,
)
from langchain_core.globals import get_llm_cache
from langchain_core.language_models._compat_bridge import (
    achunks_to_events,
    amessage_to_events,
    chunks_to_events,
    message_to_events,
)
from langchain_core.language_models._utils import (
    _filter_invocation_params_for_tracing,
    _normalize_messages,
    _update_message_content_to_blocks,
)
from langchain_core.language_models.base import (
    BaseLanguageModel,
    LangSmithParams,
    LanguageModelInput,
)
from langchain_core.language_models.chat_model_stream import (
    AsyncChatModelStream,
    ChatModelStream,
)
from langchain_core.language_models.model_profile import (
    ModelProfile,
    _warn_unknown_profile_keys,
)
from langchain_core.load import dumpd, dumps
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    AnyMessage,
    BaseMessage,
    convert_to_messages,
    is_data_content_block,
    message_chunk_to_message,
)
from langchain_core.messages import content as types
from langchain_core.messages.block_translators.openai import (
    convert_to_openai_image_block,
)
from langchain_core.output_parsers.openai_tools import (
    JsonOutputKeyToolsParser,
    JsonOutputToolsParser,
    PydanticToolsParser,
)
from langchain_core.outputs import (
    ChatGeneration,
    ChatGenerationChunk,
    ChatResult,
    Generation,
    LLMResult,
    RunInfo,
)
from langchain_core.outputs.chat_generation import merge_chat_generation_chunks
from langchain_core.prompt_values import ChatPromptValue, PromptValue, StringPromptValue
from langchain_core.rate_limiters import BaseRateLimiter
from langchain_core.runnables import RunnableBinding, RunnableMap, RunnablePassthrough
from langchain_core.runnables.config import ensure_config, run_in_executor
from langchain_core.tracers._streaming import (
    _StreamingCallbackHandler,
    _V2StreamingCallbackHandler,
)
from langchain_core.utils._gateway import (
    GATEWAY_METADATA_RESPONSE_KEY,
    _parse_gateway_metadata,
)
from langchain_core.utils._signature import _signature_parameters
from langchain_core.utils.function_calling import (
    convert_to_json_schema,
    convert_to_openai_tool,
)
from langchain_core.utils.pydantic import is_basemodel_subclass
from langchain_core.utils.utils import LC_ID_PREFIX, from_env

if TYPE_CHECKING:
    import uuid
    from collections.abc import Awaitable

    from langchain_protocol.protocol import MessagesData

    from langchain_core.runnables import Runnable, RunnableConfig
    from langchain_core.runnables.schema import StreamEvent
    from langchain_core.tools import BaseTool


def _generate_response_from_error(error: BaseException) -> list[ChatGeneration]:
    if hasattr(error, "response"):
        response = error.response
        metadata: dict[str, Any] = {}
        generation_info: dict[str, Any] = {}
        if hasattr(response, "json"):
            try:
                metadata["body"] = response.json()
            except Exception:
                try:
                    metadata["body"] = getattr(response, "text", None)
                except Exception:
                    metadata["body"] = None
        if hasattr(response, "headers"):
            try:
                headers = response.headers
                metadata["headers"] = dict(headers)
                gateway_metadata = _parse_gateway_metadata(headers)
                if gateway_metadata is not None:
                    generation_info[GATEWAY_METADATA_RESPONSE_KEY] = gateway_metadata
            except Exception:
                metadata["headers"] = None
        if hasattr(response, "status_code"):
            metadata["status_code"] = response.status_code
        if hasattr(error, "request_id"):
            metadata["request_id"] = error.request_id
        generations = [
            ChatGeneration(
                message=AIMessage(content="", response_metadata=metadata),
                generation_info=generation_info or None,
            )
        ]
    else:
        generations = []

    return generations


def _format_for_tracing(messages: list[BaseMessage]) -> list[BaseMessage]:
    """为 `on_chat_model_start` 中的追踪格式化消息。

    - 将图像内容块更新为 OpenAI 聊天补全格式（向后兼容）。 - 为仅包含单一键的内容块添加
    `type` 键。

    Args:
        messages: 需要格式化的消息列表。

    Returns:
        格式化后的消息列表，用于追踪。

    """
    messages_to_trace = []
    for message in messages:
        message_to_trace = message
        if isinstance(message.content, list):
            for idx, block in enumerate(message.content):
                if isinstance(block, dict):
                                        # 更新图像内容块以符合 OpenAI # Chat Completions 格式。
                    if (
                        block.get("type") == "image"
                        and is_data_content_block(block)
                        and not ("file_id" in block or block.get("source_type") == "id")
                    ):
                        if message_to_trace is message:
                                                        # 浅拷贝
                            message_to_trace = message.model_copy()
                            message_to_trace.content = list(message_to_trace.content)

                        message_to_trace.content[idx] = (  # type: ignore[index]  # mypy confused by .model_copy
                            convert_to_openai_image_block(block)
                        )
                    elif (
                        block.get("type") == "file"
                        and is_data_content_block(block)  # v0（图像/音频/文件）或 v1
                        and "base64" in block
                                                # 向后兼容：将 v1 base64 块转换为 v0
                    ):
                        if message_to_trace is message:
                                                        # 浅拷贝
                            message_to_trace = message.model_copy()
                            message_to_trace.content = list(message_to_trace.content)

                        message_to_trace.content[idx] = {  # type: ignore[index]
                            **{k: v for k, v in block.items() if k != "base64"},
                            "data": block["base64"],
                            "source_type": "base64",
                        }
                    elif len(block) == 1 and "type" not in block:
                                                # 追踪假设所有内容块都有 "type" 键。此处
                                                # 我们在此添加该键（如果缺失），并且当块中只有一个键时，类型选择是明确的。
                                                # 浅拷贝
                        if message_to_trace is message:
                                                        # TODO: 考虑在 BaseChatModel 上添加 `_model_identifier` 属性
                            message_to_trace = message.model_copy()
                            message_to_trace.content = list(message_to_trace.content)
                        key = next(iter(block))
                        message_to_trace.content[idx] = {  # type: ignore[index]
                            "type": key,
                            key: block[key],
                        }
        messages_to_trace.append(message_to_trace)

    return messages_to_trace


def generate_from_stream(stream: Iterator[ChatGenerationChunk]) -> ChatResult:
    """从流中生成。

    Args:
        stream: `ChatGenerationChunk` 的迭代器。

    Raises:
        ValueError: 如果在流中未找到任何生成结果。

    Returns:
        聊天结果。

    """
    generation = next(stream, None)
    if generation:
        generation += list(stream)
    if generation is None:
        msg = "No generations found in stream."
        raise ValueError(msg)
    return ChatResult(
        generations=[
            ChatGeneration(
                message=message_chunk_to_message(generation.message),
                generation_info=generation.generation_info,
            )
        ]
    )


async def agenerate_from_stream(
    stream: AsyncIterator[ChatGenerationChunk],
) -> ChatResult:
    """异步从流中生成。

    Args:
        stream: `ChatGenerationChunk` 的异步迭代器。

    Returns:
        聊天结果。

    """
    chunks = [chunk async for chunk in stream]
    return await run_in_executor(None, generate_from_stream, iter(chunks))


def _format_ls_structured_output(
    ls_structured_output_format: dict[str, Any] | None,
) -> dict[str, Any]:
    if ls_structured_output_format:
        try:
            ls_structured_output_format_dict = {
                "ls_structured_output_format": {
                    "kwargs": ls_structured_output_format.get("kwargs", {}),
                    "schema": convert_to_json_schema(
                        ls_structured_output_format["schema"]
                    ),
                }
            }
        except ValueError:
            ls_structured_output_format_dict = {}
    else:
        ls_structured_output_format_dict = {}

    return ls_structured_output_format_dict


class BaseChatModel(BaseLanguageModel[AIMessage], ABC):
    r"""聊天模型的基类。

    主要的强制性方法：
        实际调用底层模型的方法。

        本表格简要概述了主要的强制性方法。请参阅基础 `Runnable` 参考文档以获取完整说明。

        | 方法                 | 输入                                                        | 输出                                                     | 描述                                                                      |
        | ---------------------- | ------------------------------------------------------------ | ---------------------------------------------------------- | -------------------------------------------------------------------------------- |
        | `invoke`               | `str` \| `list[dict | tuple | BaseMessage]` \| `PromptValue` | `BaseMessage`                                              | 单次聊天模型调用。                                                        |
        | `ainvoke`              | `'''`                                                        | `BaseMessage`                                              | 默认在异步执行器中运行 `invoke`。                               |
        | `stream`               | `'''`                                                        | `Iterator[BaseMessageChunk]`                               | 默认产出 `invoke` 的输出。                                         |
        | `astream`              | `'''`                                                        | `AsyncIterator[BaseMessageChunk]`                          | 默认产出 `ainvoke` 的输出。                                        |
        | `astream_events`       | `'''`                                                        | `AsyncIterator[StreamEvent]`                               | 事件类型：`on_chat_model_start`，`on_chat_model_stream`，`on_chat_model_end`。 |
        | `batch`                | `list[''']`                                                  | `list[BaseMessage]`                                        | 默认在并发线程中运行 `invoke`。                              |
        | `abatch`               | `list[''']`                                                  | `list[BaseMessage]`                                        | 默认在并发线程中运行 `ainvoke`。                             |
        | `batch_as_completed`   | `list[''']`                                                  | `Iterator[tuple[int, Union[BaseMessage, Exception]]]`      | 默认在并发线程中运行 `invoke`。                              |
        | `abatch_as_completed`  | `list[''']`                                                  | `AsyncIterator[tuple[int, Union[BaseMessage, Exception]]]` | 默认在并发线程中运行 `ainvoke`。                             |

    关键声明式方法：
        使用聊天模型创建另一个 `Runnable` 的方法。

        本表简要概述了主要的声明式方法。请参阅每个方法的参考文档以获取完整说明。

        | 方法                       | 描述                                                                                |
        | ---------------------------- | ------------------------------------------------------------------------------------------ |
        | `bind_tools`                 | 创建可调用工具的聊天模型。                                                                     |
        | `with_structured_output`     | 创建使用 schema 结构化模型输出的包装器。                                                       |
        | `with_retry`                 | 创建在失败时重试模型调用的包装器。                                                             |
        | `with_fallbacks`             | 创建在失败时回退到其他模型的包装器。                                                           |
        | `configurable_fields`        | 指定可在运行时通过 `RunnableConfig` 配置的模型初始化参数。                                     |
        | `configurable_alternatives`  | 指定可在运行时通过 `RunnableConfig` 替换的替代模型。                                           |

    Creating custom chat model:
        Custom chat model implementations should inherit from this class.
        Please reference the table below for information about which
        methods and properties are required or optional for implementations.

        | 方法/属性                  | 描述                                                        | 必需          |
        | -------------------------------- | ------------------------------------------------------------------ | ----------------- |
        | `_generate`                      | 用于从提示词生成聊天结果                        | 必需          |
        | `_llm_type` (属性)           | 用于唯一标识模型类型。用于日志记录。 | 必需          |
        | `_identifying_params` (属性) | 用于追踪目的的模型参数化表示。             | 可选          |
        | `_stream`                        | 用于实现流式处理                                         | 可选          |
        | `_agenerate`                     | 用于实现原生异步方法                             | 可选          |
        | `_astream`                       | 用于实现 `_stream` 的异步版本                        | 可选          |

    """  # noqa: E501

    rate_limiter: BaseRateLimiter | None = Field(default=None, exclude=True)
    "可选的速率限制器，用于限制请求次数。"

    disable_streaming: bool | Literal["tool_calling"] = False
    """是否禁用此模型的流式处理。

    如果跳过流式处理，则 `stream`/`astream`/`astream_events` 将
    委托给 `invoke`/`ainvoke`。

    - If `True`, will always bypass streaming case.
    - If `'tool_calling'`, will bypass streaming case only when the model is called
        with a `tools` keyword argument. In other words, LangChain will automatically
        switch to non-streaming behavior (`invoke`) only when the tools argument is
        provided. This offers the best of both worlds.
    - If `False` (Default), will always use streaming case if available.

    The main reason for this flag is that code might be written using `stream` and
    a user may want to swap out a given model for another model whose implementation
    does not properly support streaming.
    """

    output_version: str | None = Field(
        default_factory=from_env("LC_OUTPUT_VERSION", default=None)
    )
    """用于在消息内容中存储 `AIMessage` 输出格式的版本。

    `AIMessage.content_blocks` will lazily parse the contents of `content` into a
    standard format. This flag can be used to additionally store the standard format
    in message content, e.g., for serialization purposes.

    支持的值：

    - `'v0'`: provider-specific format in content (can lazily-parse with
        `content_blocks`)
    - `'v1'`: standardized format in content (consistent with `content_blocks`)

    合作包（例如
    [`langchain-openai`](https://pypi.org/project/langchain-openai)）也可使用此字段以向后兼容的方式推出新的内容格式。

    !!! version-added "在 `langchain-core` 1.0.0 中新增"

    """

    profile: ModelProfile | None = Field(default=None, exclude=True)
    """描述模型能力的配置文件。

    !!! warning "测试版功能"

        这是一个测试版功能。模型配置文件的格式可能会发生变化。

    If not specified, automatically loaded from the provider package on initialization
    if data is available.

    Example profile data includes context window sizes, supported modalities, or support
    for tool calling, structured output, and other features.

    !!! version-added "在 `langchain-core` 1.1.0 中新增"
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )

    def _resolve_model_profile(self) -> ModelProfile | None:
        """返回默认模型配置文件，若不可用则返回 `None`。

        在子类中重写此方法，而不是
        `_set_model_profile`。基类验证器会自动调用它并处理赋值。这避免了合作伙伴代码与
        Pydantic 验证机制耦合。

        Each partner needs its own override because things can vary per-partner,
        such as the attribute that identifies the model (e.g., `model`,
        `model_name`, `model_id`, `deployment_name`) and the partner-local
        `_get_default_model_profile` function that reads from each partner's own
        profile data.
        """
                # 以标准化合作伙伴识别其模型的方式，这可能允许在此处使用默认实现，调用共享的
                # 配置文件加载机制。
                # 抑制来自合作伙伴覆盖的错误（例如，缺少配置文件、损坏的导入），以确保模型构造不会因可选字段而失败。
                # NOTE: `_check_profile_keys` 必须在 `_set_model_profile` 之后定义。
        return None

    @model_validator(mode="after")
    def _set_model_profile(self) -> Self:
        """如果未提供 `profile`，则从 `_resolve_model_profile` 填充。

        Partners should override `_resolve_model_profile` rather than this
        validator. Overriding this with a new `@model_validator` replaces the
        base validator (Pydantic v2 behavior), bypassing the standard resolution
        path. A plain method override does not prevent the base validator from
        running.
        """
        if self.profile is None:
                        # Pydantic v2 在定义顺序中运行 mode="after" 验证器。
                        # isinstance 保护：ModelProfile 是一个
            # TypedDict（始终为字典），但可防止来自合作伙伴覆盖的意外类型。
                        # self 在此情况下始终是 Serializable 对象，因此结果保证为字典，因为 dumpd
            # 使用默认回调，该回调使用 obj.to_json，后者始终返回 TypedDict 子类
            with contextlib.suppress(Exception):
                self.profile = self._resolve_model_profile()
        return self

        # --- Runnable 方法 ---
        # 我们假设所有模型中的工具都通过 "tools" 关键字参数传入。
    @model_validator(mode="after")
    def _check_profile_keys(self) -> Self:
        """对未识别的 profile 键发出警告。"""
                # 检查是否实现了流式处理。
                # 注意，由于异步会回退到同步，因此此处需同时检查。
        if self.profile and isinstance(self.profile, dict):
            _warn_unknown_profile_keys(self.profile)
        return self

    @cached_property
    def _serialized(self) -> builtins.dict[str, Any]:
                # 肯定：显式 `stream=<truthy>` 关键字参数。
                # 肯定：实例级 `streaming=True` 属性。
                # 肯定：已附加 v1 流式回调处理器。
        return cast("builtins.dict[str, Any]", dumpd(self))

        # 可选：仅当附加了 v2 处理器时才通过 v2 路由。

    @property
    @override
    def OutputType(self) -> Any:
        """获取此 `Runnable` 的输出类型。"""
        return AnyMessage

    def _convert_input(self, model_input: LanguageModelInput) -> PromptValue:
        if isinstance(model_input, PromptValue):
            return model_input
        if isinstance(model_input, str):
            return StringPromptValue(text=model_input)
        if isinstance(model_input, Sequence):
            return ChatPromptValue(messages=convert_to_messages(model_input))
        msg = (  # type: ignore[unreachable]
            f"Invalid input type {type(model_input)}. "
            "Must be a PromptValue, str, or list of BaseMessages."
        )
        raise ValueError(msg)

    @override
    def invoke(
        self,
        input: LanguageModelInput,
        config: RunnableConfig | None = None,
        *,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> AIMessage:
        config = ensure_config(config)
        return cast(
            "AIMessage",
            cast(
                "ChatGeneration",
                self.generate_prompt(
                    [self._convert_input(input)],
                    stop=stop,
                    callbacks=config.get("callbacks"),
                    tags=config.get("tags"),
                    metadata=config.get("metadata"),
                    run_name=config.get("run_name"),
                    run_id=config.pop("run_id", None),
                    **kwargs,
                ).generations[0][0],
            ).message,
        )

    @override
    async def ainvoke(
        self,
        input: LanguageModelInput,
        config: RunnableConfig | None = None,
        *,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> AIMessage:
        config = ensure_config(config)
        llm_result = await self.agenerate_prompt(
            [self._convert_input(input)],
            stop=stop,
            callbacks=config.get("callbacks"),
            tags=config.get("tags"),
            metadata=config.get("metadata"),
            run_name=config.get("run_name"),
            run_id=config.pop("run_id", None),
            **kwargs,
        )
        return cast(
            "AIMessage", cast("ChatGeneration", llm_result.generations[0][0]).message
        )

    def _streaming_disabled(self, **kwargs: Any) -> bool:
        """返回此次调用是否硬性禁用流式传输。

        Shared opt-outs honored by both `_should_stream` and
        `_should_use_protocol_streaming` — these override any affirmative trigger
        (attached handler, `stream=True`, etc.):

        - `self.disable_streaming is True`
        - `self.disable_streaming == "tool_calling"` 且传入了 `tools`
        - 在调用参数中 `stream=<falsy>`
        - 实例上的 `self.streaming is False`
        """
        if self.disable_streaming is True:
            return True
                # 需要请求风味的 v2 事件源。原生
        if self.disable_streaming == "tool_calling" and kwargs.get("tools"):
            return True
        if "stream" in kwargs and not kwargs["stream"]:
            return True
        return (
            "streaming" in self.model_fields_set
            and getattr(self, "streaming", None) is False
        )

    def _should_stream(
        self,
        *,
        async_api: bool,
        run_manager: CallbackManagerForLLMRun
        | AsyncCallbackManagerForLLMRun
        | None = None,
        **kwargs: Any,
    ) -> bool:
        """确定给定模型调用是否应命中流式 API。"""
        sync_not_implemented = type(self)._stream == BaseChatModel._stream  # noqa: SLF001
        async_not_implemented = type(self)._astream == BaseChatModel._astream  # noqa: SLF001

                # `_(a)stream_chat_model_events` 钩子绕过桥接；
        if (not async_api) and sync_not_implemented:
            return False
                # 否则桥接包装 `_stream` / `_astream`。异步可以
        if async_api and async_not_implemented and sync_not_implemented:
            return False

        if self._streaming_disabled(**kwargs):
            return False

                # 肯定：显式 `stream=<truthy>` 关键字参数。
        if kwargs.get("stream"):
            return True

                # 肯定：实例级别的 `streaming=True` 属性。
        if (
            "streaming" in self.model_fields_set
            and getattr(self, "streaming", None) is True
        ):
            return True

                # 确认：已附加 v1 流式回调处理器。
        handlers = run_manager.handlers if run_manager else []
        return any(isinstance(h, _StreamingCallbackHandler) for h in handlers)

    def _should_use_protocol_streaming(
        self,
        *,
        async_api: bool,
        run_manager: CallbackManagerForLLMRun
        | AsyncCallbackManagerForLLMRun
        | None = None,
        **kwargs: Any,
    ) -> bool:
        """确定一次调用是否应通过 v2 事件路径路由。

        在 `_generate_with_cache` / `_agenerate_with_cache` 内部与 `_should_stream`
        并行运行——在运行管理器打开后——当处理器声明自己为 `_V2StreamingCallbackHandler`
        时，会胜过 v1 流式分支。与 `_should_stream` 平行而非委托——v1 和 v2
        有独立的肯定触发条件。

        Args:
            async_api: Whether the caller is on the async path.
            run_manager: The active LLM run manager.
            **kwargs: Call kwargs; inspected for `disable_streaming`
                semantics and an explicit `stream=False` override.

        Returns:
            `True` if any attached handler inherits
            `_V2StreamingCallbackHandler` and the model can drive the v2
            event generator (natively or via the `_stream` compat
            bridge).
        """
                # 可选：仅当附加了 v2 处理程序时才通过 v2 路由。
        handlers = run_manager.handlers if run_manager else []
        if not any(isinstance(h, _V2StreamingCallbackHandler) for h in handlers):
            return False

                # 需要一个针对指定口味的 v2 事件源。原生
                # `_(a)stream_chat_model_events` hook 跳过桥接；
                # 否则桥接器会包装 `_stream` / `_astream`。异步情况下
                # 回退到同步模式。
        #
                # `cls._stream is not BaseChatModel._stream` 是一种身份判断
                # 检查是否“子类重写了 `_stream`”——与以下模式相同
                # `_should_stream`。
        cls = type(self)
        has_native_sync = getattr(cls, "_stream_chat_model_events", None) is not None
        has_native_async = getattr(cls, "_astream_chat_model_events", None) is not None
        overrides_sync = cls._stream is not BaseChatModel._stream
        overrides_async = cls._astream is not BaseChatModel._astream
        has_sync_source = has_native_sync or overrides_sync
        has_async_source = has_native_async or overrides_async
        has_source = (
            (has_sync_source or has_async_source) if async_api else has_sync_source
        )
        if not has_source:
            return False

        return not self._streaming_disabled(**kwargs)

    def _iter_v2_events(
        self,
        messages: list[BaseMessage],
        *,
        run_manager: CallbackManagerForLLMRun,
        stream: ChatModelStream,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> Iterator[MessagesData]:
        """通过逐事件分发驱动 v2 事件生成器。

        Shared between the `stream_events(version="v3")` pump and the
        invoke-time v2 branch in `_generate_with_cache`. Picks the native
        `_stream_chat_model_events` hook when the subclass provides one,
        else bridges `_stream` chunks via `chunks_to_events`. Each event
        is dispatched into `stream` and fired as `on_stream_event` on
        the run manager. Run-lifecycle callbacks
        (`on_chat_model_start` / `on_llm_end` / `on_llm_error`) and
        rate-limiter acquisition are the caller's responsibility.

        Args:
            messages: 标准化输入消息。 run_manager: 活跃的 LLM
            运行管理器；接收每个事件的 `on_stream_event`。 stream:
            由调用方拥有的累加器；通过 `stream.dispatch` 接收每个事件。 stop:
            可选停止序列。 **kwargs: 转发给事件生产者。

        Yields:
            模型生成的每个协议事件。
        """
        native = cast(
            "Callable[..., Iterator[MessagesData]] | None",
            getattr(self, "_stream_chat_model_events", None),
        )
        if native is not None:
            event_iter: Iterator[MessagesData] = native(
                messages, stop=stop, run_manager=run_manager, **kwargs
            )
        else:
            event_iter = chunks_to_events(
                self._stream(messages, stop=stop, run_manager=run_manager, **kwargs),
                message_id=stream.message_id,
            )
        for event in event_iter:
            stream.dispatch(event)
            run_manager.on_stream_event(event)
            yield event

    async def _aiter_v2_events(
        self,
        messages: list[BaseMessage],
        *,
        run_manager: AsyncCallbackManagerForLLMRun,
        stream: AsyncChatModelStream,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[MessagesData]:
        """`_iter_v2_events` 的异步对应实现。

        参见 `_iter_v2_events` 以了解共享契约。
        """
        native = cast(
            "Callable[..., AsyncIterator[MessagesData]] | None",
            getattr(self, "_astream_chat_model_events", None),
        )
        if native is not None:
            event_iter: AsyncIterator[MessagesData] = native(
                messages, stop=stop, run_manager=run_manager, **kwargs
            )
        else:
            event_iter = achunks_to_events(
                self._astream(messages, stop=stop, run_manager=run_manager, **kwargs),
                message_id=stream.message_id,
            )
        async for event in event_iter:
            stream.dispatch(event)
            await run_manager.on_stream_event(event)
            yield event

    @override
    def stream(
        self,
        input: LanguageModelInput,
        config: RunnableConfig | None = None,
        *,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> Iterator[AIMessageChunk]:
        if not self._should_stream(async_api=False, **{**kwargs, "stream": True}):
                        # 模型不支持流式输出，因此使用默认实现
            yield cast(
                "AIMessageChunk",
                self.invoke(input, config=config, stop=stop, **kwargs),
            )
        else:
            config = ensure_config(config)
            messages = self._convert_input(input).to_messages()
            ls_structured_output_format = kwargs.pop(
                "ls_structured_output_format", None
            ) or kwargs.pop("structured_output_format", None)
            ls_structured_output_format_dict = _format_ls_structured_output(
                ls_structured_output_format
            )

            params = self._get_invocation_params(stop=stop, **kwargs)
            options = {
                "stop": stop,
                **{key: params[key] for key in kwargs if key in params},
                **ls_structured_output_format_dict,
            }
            inheritable_metadata = {
                **(config.get("metadata") or {}),
                **self._get_ls_params_with_defaults(stop=stop, **kwargs),
            }
            callback_manager = CallbackManager.configure(
                config.get("callbacks"),
                self.callbacks,
                self.verbose,
                config.get("tags"),
                self.tags,
                inheritable_metadata,
                self.metadata,
                langsmith_inheritable_metadata=_filter_invocation_params_for_tracing(
                    params
                ),
            )
            (run_manager,) = callback_manager.on_chat_model_start(
                self._serialized,
                [_format_for_tracing(messages)],
                invocation_params=params,
                options=options,
                name=config.get("run_name"),
                run_id=config.pop("run_id", None),
                batch_size=1,
            )

            chunks: list[ChatGenerationChunk] = []

            if self.rate_limiter:
                self.rate_limiter.acquire(blocking=True)

            try:
                input_messages = _normalize_messages(messages)
                run_id = "-".join((LC_ID_PREFIX, str(run_manager.run_id)))
                yielded = False
                index = -1
                index_type = ""
                for chunk in self._stream(input_messages, stop=stop, **kwargs):
                    if chunk.message.id is None:
                        chunk.message.id = run_id
                    chunk.message.response_metadata = _gen_info_and_msg_metadata(chunk)
                    if self.output_version == "v1":
                                                # 用 .content_blocks 重写 .content
                        chunk.message = _update_message_content_to_blocks(
                            chunk.message, "v1"
                        )
                        for block in cast(
                            "list[types.ContentBlock]", chunk.message.content
                        ):
                            if block["type"] != index_type:
                                index_type = block["type"]
                                index += 1
                            if "index" not in block:
                                block["index"] = index
                    run_manager.on_llm_new_token(chunk.message.content, chunk=chunk)
                    chunks.append(chunk)
                    yield cast("AIMessageChunk", chunk.message)
                    yielded = True

                                # 如果尚未生成，则以 chunk_position="last" 生成一个最终的空块
                                # 已生成
                if (
                    yielded
                    and isinstance(chunk.message, AIMessageChunk)
                    and not chunk.message.chunk_position
                ):
                    empty_content: str | list[str | dict[str, Any]] = (
                        "" if isinstance(chunk.message.content, str) else []
                    )
                    msg_chunk = AIMessageChunk(
                        content=empty_content, chunk_position="last", id=run_id
                    )
                    run_manager.on_llm_new_token(
                        "", chunk=ChatGenerationChunk(message=msg_chunk)
                    )
                    yield msg_chunk
            except BaseException as e:
                generations_with_error_metadata = _generate_response_from_error(e)
                chat_generation_chunk = merge_chat_generation_chunks(chunks)
                if chat_generation_chunk:
                    generations = [
                        [chat_generation_chunk],
                        generations_with_error_metadata,
                    ]
                else:
                    generations = [generations_with_error_metadata]
                run_manager.on_llm_error(
                    e,
                    response=LLMResult(generations=generations),
                )
                raise

            generation = merge_chat_generation_chunks(chunks)
            if generation is None:
                err = ValueError("No generation chunks were returned")
                run_manager.on_llm_error(err, response=LLMResult(generations=[]))
                raise err

            run_manager.on_llm_end(LLMResult(generations=[[generation]]))

    @override
    async def astream(
        self,
        input: LanguageModelInput,
        config: RunnableConfig | None = None,
        *,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[AIMessageChunk]:
        if not self._should_stream(async_api=True, **{**kwargs, "stream": True}):
                        # 未实现异步或同步流式输出，因此回退到 ainvoke
            yield cast(
                "AIMessageChunk",
                await self.ainvoke(input, config=config, stop=stop, **kwargs),
            )
            return

        config = ensure_config(config)
        messages = self._convert_input(input).to_messages()

        ls_structured_output_format = kwargs.pop(
            "ls_structured_output_format", None
        ) or kwargs.pop("structured_output_format", None)
        ls_structured_output_format_dict = _format_ls_structured_output(
            ls_structured_output_format
        )

        params = self._get_invocation_params(stop=stop, **kwargs)
        options = {
            "stop": stop,
            **{key: params[key] for key in kwargs if key in params},
            **ls_structured_output_format_dict,
        }
        inheritable_metadata = {
            **(config.get("metadata") or {}),
            **self._get_ls_params_with_defaults(stop=stop, **kwargs),
        }
        callback_manager = AsyncCallbackManager.configure(
            config.get("callbacks"),
            self.callbacks,
            self.verbose,
            config.get("tags"),
            self.tags,
            inheritable_metadata,
            self.metadata,
            langsmith_inheritable_metadata=_filter_invocation_params_for_tracing(
                params
            ),
        )
        (run_manager,) = await callback_manager.on_chat_model_start(
            self._serialized,
            [_format_for_tracing(messages)],
            invocation_params=params,
            options=options,
            name=config.get("run_name"),
            run_id=config.pop("run_id", None),
            batch_size=1,
        )

        if self.rate_limiter:
            await self.rate_limiter.aacquire(blocking=True)

        chunks: list[ChatGenerationChunk] = []

        try:
            input_messages = _normalize_messages(messages)
            run_id = "-".join((LC_ID_PREFIX, str(run_manager.run_id)))
            yielded = False
            index = -1
            index_type = ""
            async for chunk in self._astream(
                input_messages,
                stop=stop,
                **kwargs,
            ):
                if chunk.message.id is None:
                    chunk.message.id = run_id
                chunk.message.response_metadata = _gen_info_and_msg_metadata(chunk)
                if self.output_version == "v1":
                                        # 用 .content_blocks 重写 .content
                    chunk.message = _update_message_content_to_blocks(
                        chunk.message, "v1"
                    )
                    for block in cast(
                        "list[types.ContentBlock]", chunk.message.content
                    ):
                        if block["type"] != index_type:
                            index_type = block["type"]
                            index += 1
                        if "index" not in block:
                            block["index"] = index
                await run_manager.on_llm_new_token(chunk.message.content, chunk=chunk)
                chunks.append(chunk)
                yield cast("AIMessageChunk", chunk.message)
                yielded = True

                        # 如果尚未生成，则以 chunk_position="last" 生成一个最终的空块
            if (
                yielded
                and isinstance(chunk.message, AIMessageChunk)
                and not chunk.message.chunk_position
            ):
                empty_content: str | list[str | dict[str, Any]] = (
                    "" if isinstance(chunk.message.content, str) else []
                )
                msg_chunk = AIMessageChunk(
                    content=empty_content, chunk_position="last", id=run_id
                )
                await run_manager.on_llm_new_token(
                    "", chunk=ChatGenerationChunk(message=msg_chunk)
                )
                yield msg_chunk
        except BaseException as e:
            generations_with_error_metadata = _generate_response_from_error(e)
            chat_generation_chunk = merge_chat_generation_chunks(chunks)
            if chat_generation_chunk:
                generations = [[chat_generation_chunk], generations_with_error_metadata]
            else:
                generations = [generations_with_error_metadata]
            await run_manager.on_llm_error(
                e,
                response=LLMResult(generations=generations),
            )
            raise

        generation = merge_chat_generation_chunks(chunks)
        if not generation:
            err = ValueError("No generation chunks were returned")
            await run_manager.on_llm_error(err, response=LLMResult(generations=[]))
            raise err

        await run_manager.on_llm_end(
            LLMResult(generations=[[generation]]),
        )

        # --- stream_events v3 ---

    @beta()
    def _chat_model_stream_v3(
        self,
        input: LanguageModelInput,
        config: RunnableConfig | None = None,
        *,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> ChatModelStream:
        """内部 v3 同步流式实现。

        Public entry point: 公共入口点：`stream_events(version='v3')`。
        """
        config = ensure_config(config)
        messages = self._convert_input(input).to_messages()
        input_messages = _normalize_messages(messages)

                # 在转发给 `_stream` 之前移除仅用于追踪的参数——与
                # `stream()` / `astream()` 保持一致。提供方客户端会拒绝未知参数，
                # 因此 `.with_structured_output().stream_events(version="v3", ...)`
                # 以及任何携带 `ls_structured_output_format`
                # / `structured_output_format` 的绑定都会因缺少此 pop 而报错。
        ls_structured_output_format = kwargs.pop(
            "ls_structured_output_format", None
        ) or kwargs.pop("structured_output_format", None)
        ls_structured_output_format_dict = _format_ls_structured_output(
            ls_structured_output_format
        )

        params = self._get_invocation_params(stop=stop, **kwargs)
        options = {
            "stop": stop,
            **{key: params[key] for key in kwargs if key in params},
            **ls_structured_output_format_dict,
        }
        inheritable_metadata = {
            **(config.get("metadata") or {}),
            **self._get_ls_params_with_defaults(stop=stop, **kwargs),
        }
        callback_manager = CallbackManager.configure(
            config.get("callbacks"),
            self.callbacks,
            self.verbose,
            config.get("tags"),
            self.tags,
            inheritable_metadata,
            self.metadata,
            langsmith_inheritable_metadata=_filter_invocation_params_for_tracing(
                params
            ),
        )
        stream = ChatModelStream()
        run_manager: CallbackManagerForLLMRun | None = None
        event_iter_ref: Iterator[MessagesData] | None = None
        rate_limiter_acquired = self.rate_limiter is None
        run_name = config.get("run_name")
        run_id = config.pop("run_id", None)

        def ensure_started() -> None:
            nonlocal event_iter_ref, run_manager
            if event_iter_ref is not None:
                return

            (run_manager,) = callback_manager.on_chat_model_start(
                self._serialized,
                [_format_for_tracing(messages)],
                invocation_params=params,
                options=options,
                name=run_name,
                run_id=run_id,
                batch_size=1,
            )
            stream.set_message_id("-".join((LC_ID_PREFIX, str(run_manager.run_id))))
            event_iter_ref = iter(
                self._iter_v2_events(
                    input_messages,
                    run_manager=run_manager,
                    stream=stream,
                    stop=stop,
                    **kwargs,
                )
            )

        def pump_one() -> bool:
            nonlocal rate_limiter_acquired
            ensure_started()
            if not rate_limiter_acquired:
                assert self.rate_limiter is not None  # noqa: S101
                self.rate_limiter.acquire(blocking=True)
                rate_limiter_acquired = True
            assert event_iter_ref is not None  # noqa: S101
            assert run_manager is not None  # noqa: S101
            try:
                next(event_iter_ref)
            except StopIteration:
                if not stream.done:
                    if stream.has_events:
                                                # 原生事件生产者可能省略终端
                                                # `message-finish`。在此处关闭生命周期，以便
                                                # `on_llm_end` 仍能观察到组装后的
                                                # 消息。完全为空的流仍视为错误
                                                # 以保持与 `stream()` 的一致性。
                        stream.dispatch(MessageFinishData(event="message-finish"))
                    else:
                        err = ValueError("No generation chunks were returned")
                        stream.fail(err)
                        run_manager.on_llm_error(
                            err,
                            response=LLMResult(generations=[]),
                        )
                        return False
                if stream.done and stream.output_message is not None:
                    run_manager.on_llm_end(
                        LLMResult(
                            generations=[
                                [ChatGeneration(message=stream.output_message)],
                            ],
                        ),
                    )
                return False
            except BaseException as exc:
                stream.fail(exc)
                run_manager.on_llm_error(
                    exc,
                    response=LLMResult(generations=[]),
                )
                return False
            if stream.done and stream.output_message is not None:
                run_manager.on_llm_end(
                    LLMResult(
                        generations=[
                            [ChatGeneration(message=stream.output_message)],
                        ],
                    ),
                )
            return True

        stream.set_start(ensure_started)
        stream.bind_pump(pump_one)
        return stream

    @beta()
    async def _achat_model_stream_v3(
        self,
        input: LanguageModelInput,
        config: RunnableConfig | None = None,
        *,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> AsyncChatModelStream:
        """内部 v3 异步流式实现。

        Public entry point: 公共入口点：`astream_events(version='v3')`。
        """
        config = ensure_config(config)
        messages = self._convert_input(input).to_messages()
        input_messages = _normalize_messages(messages)

                # 在转发前移除仅用于追踪的参数——参见同步 v3
                # 实现中的完整理由。
        ls_structured_output_format = kwargs.pop(
            "ls_structured_output_format", None
        ) or kwargs.pop("structured_output_format", None)
        ls_structured_output_format_dict = _format_ls_structured_output(
            ls_structured_output_format
        )

        params = self._get_invocation_params(stop=stop, **kwargs)
        options = {
            "stop": stop,
            **{key: params[key] for key in kwargs if key in params},
            **ls_structured_output_format_dict,
        }
        inheritable_metadata = {
            **(config.get("metadata") or {}),
            **self._get_ls_params_with_defaults(stop=stop, **kwargs),
        }
        callback_manager = AsyncCallbackManager.configure(
            config.get("callbacks"),
            self.callbacks,
            self.verbose,
            config.get("tags"),
            self.tags,
            inheritable_metadata,
            self.metadata,
            langsmith_inheritable_metadata=_filter_invocation_params_for_tracing(
                params
            ),
        )
        stream = AsyncChatModelStream()
        run_manager: AsyncCallbackManagerForLLMRun | None = None
        run_name = config.get("run_name")
        run_id = config.pop("run_id", None)
        start_lock = asyncio.Lock()

        async def _produce() -> None:
            assert run_manager is not None  # noqa: S101
            try:
                if self.rate_limiter:
                    await self.rate_limiter.aacquire(blocking=True)

                async for _event in self._aiter_v2_events(
                    input_messages,
                    run_manager=run_manager,
                    stream=stream,
                    stop=stop,
                    **kwargs,
                ):
                    pass
                if not stream.done:
                    if stream.has_events:
                                                # 原生事件生产者可能省略终端
                                                # `message-finish`。在此处关闭生命周期，以便
                                                # `on_llm_end` 能看到最终的消息。一个
                                                # 真正为空的流仍被视为错误，以保持与
                                                # `astream()` 的一致性。
                        stream.dispatch(MessageFinishData(event="message-finish"))
                    else:
                        err = ValueError("No generation chunks were returned")
                        stream.fail(err)
                        await run_manager.on_llm_error(
                            err,
                            response=LLMResult(generations=[]),
                        )
                        return
                if stream.done and stream.output_message is not None:
                    await run_manager.on_llm_end(
                        LLMResult(
                            generations=[
                                [ChatGeneration(message=stream.output_message)],
                            ],
                        ),
                    )
            except asyncio.CancelledError as exc:
                stream.fail(exc)
                                # 关闭回调生命周期，使追踪能够观察到与之前的 `on_chat_model_start`
                # 匹配的结束事件。
                                # `on_llm_error` 被 @shielded 装饰，因此即使此处的 `await`
                # 重新抛出取消异常，回调也会在后台完成执行。
                                # 仅在 `stream.aclose()` 被调用且生产者在 `_produce`
                # 执行前被取消时触发——因此来自 `CancelledError` 处理器的 `on_llm_error`
                # 从未触发。由回调管理器屏蔽；即使调用方被取消，也仍会运行至完成。
                                # v1/v2：转发至 Runnable.astream_events（异步生成器）。
                                # --- 自定义方法 ---
                with contextlib.suppress(Exception):
                    await run_manager.on_llm_error(
                        exc,
                        response=LLMResult(generations=[]),
                    )
                raise
            except BaseException as exc:
                stream.fail(exc)
                await run_manager.on_llm_error(
                    exc,
                    response=LLMResult(generations=[]),
                )

        async def ensure_started() -> None:
            nonlocal run_manager
            if stream._producer_task is not None:  # noqa: SLF001
                return

            async with start_lock:
                if stream._producer_task is not None:  # noqa: SLF001
                    return  # type: ignore[unreachable]

                (run_manager,) = await callback_manager.on_chat_model_start(
                    self._serialized,
                    [_format_for_tracing(messages)],
                    invocation_params=params,
                    options=options,
                    name=run_name,
                    run_id=run_id,
                    batch_size=1,
                )
                stream.set_message_id("-".join((LC_ID_PREFIX, str(run_manager.run_id))))
                stream._producer_task = asyncio.get_running_loop().create_task(  # noqa: SLF001
                    _produce()
                )

        async def _on_aclose_fail(exc: BaseException) -> None:
            assert run_manager is not None  # noqa: S101
                        # 通过从文本内容创建 AIMessage 将 Generation 转换为 ChatGeneration
                        # 已经是 ChatGeneration 或其他预期类型
                        # 我们在缓存命中时将成本归零
                        # 从类名获取默认提供方
                        # model
            await run_manager.on_llm_error(
                exc,
                response=LLMResult(generations=[]),
            )

        stream.set_start(ensure_started)
        stream._on_aclose_fail = _on_aclose_fail  # noqa: SLF001
        return stream

    @overload  # type: ignore[override]
    def stream_events(
        self,
        input: LanguageModelInput,
        config: RunnableConfig | None = None,
        *,
        version: Literal["v1", "v2"] = "v2",
        **kwargs: Any,
    ) -> Iterator[StreamEvent]: ...

    @overload
    def stream_events(
        self,
        input: LanguageModelInput,
        config: RunnableConfig | None = None,
        *,
        version: Literal["v3"],
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> ChatModelStream: ...

    def stream_events(
        self,
        input: LanguageModelInput,
        config: RunnableConfig | None = None,
        *,
        version: Literal["v1", "v2", "v3"] = "v2",
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> Iterator[StreamEvent] | ChatModelStream:
        """从该聊天模型中流式输出事件。

        For `version="v1"` / `"v2"`, yields `StreamEvent` dicts (see
        `Runnable.stream_events`). For `version="v3"`, returns a
        `ChatModelStream` exposing typed projections (`.text`,
        `.reasoning`, `.tool_calls`, `.output`).

        !!! warning "测试版"

            `version="v3"` is in beta. The protocol shape, return type,
            and surface area may change in future releases. Calling it
            emits a `LangChainBetaWarning` at runtime.

        !!! note "v3 始终生成 v1 格式的内容"

            `ChatModelStream.output.content` is always a list of v1
            content blocks (text / reasoning / tool_call / image / …),
            regardless of the model's `output_version` attribute. The
            setting only affects the legacy `stream()` / `astream()` /
            `invoke()` paths. If you're mixing
            `stream_events(version="v3")` with those paths in the same
            pipeline and need a consistent output shape across them,
            set `output_version="v1"` on the model.

        Args:
            input: The model input.
            config: Optional runnable config.
            version: Streaming-event schema version. `"v3"` selects the
                content-block-centric streaming protocol.
            stop: Optional stop sequences. Only used for `version="v3"`;
                ignored otherwise.
            **kwargs: Additional keyword arguments. For `version="v3"`,
                forwarded to the model.

        Returns:
            For `version="v3"`, a `ChatModelStream` with typed
            projections. Otherwise an `Iterator[StreamEvent]`.
        """
        if version == "v3":
            return self._chat_model_stream_v3(input, config, stop=stop, **kwargs)
        return super().stream_events(
            input, config, version=version, stop=stop, **kwargs
        )

    @overload
    def astream_events(
        self,
        input: LanguageModelInput,
        config: RunnableConfig | None = None,
        *,
        version: Literal["v1", "v2"] = "v2",
        **kwargs: Any,
    ) -> AsyncIterator[StreamEvent]: ...

    @overload
    def astream_events(
        self,
        input: LanguageModelInput,
        config: RunnableConfig | None = None,
        *,
        version: Literal["v3"],
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> Awaitable[AsyncChatModelStream]: ...

    def astream_events(
        self,
        input: LanguageModelInput,
        config: RunnableConfig | None = None,
        *,
        version: Literal["v1", "v2", "v3"] = "v2",
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[StreamEvent] | Awaitable[AsyncChatModelStream]:
        """`stream_events` 的异步变体。参见 `stream_events` 获取完整文档。"""
        if version == "v3":
            return self._achat_model_stream_v3(input, config, stop=stop, **kwargs)
                # temperature
        return super().astream_events(
            input, config, version=version, stop=stop, **kwargs
        )

        # max_tokens

    def _combine_llm_outputs(
        self, _llm_outputs: list[builtins.dict[str, Any] | None], /
    ) -> builtins.dict[str, Any]:
        return {}

    def _convert_cached_generations(
        self, cache_val: list[Generation]
    ) -> list[ChatGeneration]:
        """将缓存的 Generation 对象转换为 ChatGeneration 对象。

        处理缓存中包含 Generation 对象而非 ChatGeneration
        对象的情况。这可能是由于序列化/反序列化问题或旧版缓存数据导致（参见 #22389）。

        Args:
            cache_val: 缓存的生成对象列表。

        Returns:
            ChatGeneration 对象列表。

        """
        converted_generations = []
        for gen in cache_val:
            if isinstance(gen, Generation) and not isinstance(gen, ChatGeneration):
                                # 此代码效率不高，因为它在 json 和 dict 之间来回转换。
                                # 除非显式设置为 False，否则我们应该检查缓存
                chat_gen = ChatGeneration(
                    message=AIMessage(content=gen.text),
                    generation_info=gen.generation_info,
                )
                converted_generations.append(chat_gen)
            else:
                                # None 缓存表示应使用已配置的默认全局缓存
                if hasattr(gen, "message") and isinstance(gen.message, AIMessage):
                                        # 在检查缓存后应用速率限制器，因为通常我们不希望对缓存查询进行限速，但
                    gen.message = gen.message.model_copy(
                        update={
                            "usage_metadata": {
                                **(gen.message.usage_metadata or {}),
                                "total_cost": 0,
                            }
                        }
                    )
                converted_generations.append(gen)
        return converted_generations

    def _replay_v2_events_for_cache_hit(
        self,
        generations: list[ChatGeneration],
        *,
        run_manager: CallbackManagerForLLMRun | None,
        **kwargs: Any,
    ) -> None:
        """当附加了 v2 处理器时，以 v2 事件的形式重放缓存的消息。

        热缓存必须产生与冷调用相同的 `on_stream_event` 流，以确保 LangGraph
        风格的消费者不会观察到依赖于缓存状态的行为差异。由
        `_should_use_protocol_streaming` 控制，因此抑制冷调用中 v2 的
        `disable_streaming` 配置也会在此处抑制它。
        """
        if run_manager is None or not self._should_use_protocol_streaming(
            async_api=False, run_manager=run_manager, **kwargs
        ):
            return
        message_id = f"{LC_ID_PREFIX}-{run_manager.run_id}"
        for gen in generations:
            msg = getattr(gen, "message", None)
            if not isinstance(msg, AIMessage):
                continue
            for event in message_to_events(msg, message_id=message_id):
                run_manager.on_stream_event(event)

    async def _areplay_v2_events_for_cache_hit(
        self,
        generations: list[ChatGeneration],
        *,
        run_manager: AsyncCallbackManagerForLLMRun | None,
        **kwargs: Any,
    ) -> None:
        """与 `_replay_v2_events_for_cache_hit` 对应的异步版本。"""
        if run_manager is None or not self._should_use_protocol_streaming(
            async_api=True, run_manager=run_manager, **kwargs
        ):
            return
        message_id = f"{LC_ID_PREFIX}-{run_manager.run_id}"
        for gen in generations:
            msg = getattr(gen, "message", None)
            if not isinstance(msg, AIMessage):
                continue
            async for event in amessage_to_events(msg, message_id=message_id):
                await run_manager.on_stream_event(event)

    def _get_invocation_params(
        self,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> builtins.dict[str, Any]:
        params = self._dict_for_compat()
        params["stop"] = stop
        return {**params, **kwargs}

    def _get_ls_params(
        self,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> LangSmithParams:
        """获取 LangSmith 追踪的标准参数。

        Subclasses **should override** this method to populate `ls_provider`
        and `ls_model_name` from provider-specific attributes (e.g. `self.model`,
        `self.model_name`, `self.model_id`) and to honor per-call overrides
        passed via `kwargs["model"]` so that runtime `bind`/`invoke` model
        changes are reflected in traces.

        The implementation here is a best-effort fallback for subclasses that
        do not override it. It is not part of a stable contract and the
        derivation rules may change:

        - `ls_provider` 通过去除类名的前导或尾随 `"Chat"` 并将剩余部分转为小写来推导。这对多词提供方会产生不美观的值（例如 `ChatGoogleGenerativeAI` 会变成 `"googlegenerativeai"`）。
            应覆盖以设置稳定且常规的值，如 `"google_genai"`。
            - `ls_model_name` 依次从 `kwargs["model"]`、
            `self.model`、`self.model_name` 解析。

            Override to set a stable, conventional value
            such as `"google_genai"`.
        - `ls_model_name` is resolved from `kwargs["model"]`, then
            `self.model`, then `self.model_name`.

            Subclasses whose model attribute has a different name
            (`model_id`, `deployment_name`, ...) must override.
        """
                # 我们确实希望对 API 请求进行限速。
        default_provider = self.__class__.__name__
        if default_provider.startswith("Chat"):
            default_provider = default_provider[4:].lower()
        elif default_provider.endswith("Chat"):
            default_provider = default_provider[:-4]
        default_provider = default_provider.lower()

        ls_params = LangSmithParams(ls_provider=default_provider, ls_model_type="chat")
        if stop:
            ls_params["ls_stop"] = stop

                # v2 流式：当任何附加处理器通过 `_V2StreamingCallbackHandler`
        # 选择启用时，优先于 v1。驱动协议事件生成器
        if "model" in kwargs and isinstance(kwargs["model"], str):
            ls_params["ls_model_name"] = kwargs["model"]
        elif hasattr(self, "model") and isinstance(self.model, str):
            ls_params["ls_model_name"] = self.model
        elif hasattr(self, "model_name") and isinstance(self.model_name, str):
            ls_params["ls_model_name"] = self.model_name

                # (原生或 `_stream` 兼容桥接) 通过共享辅助函数，使 `on_stream_event`
        # 按每个事件触发，然后返回一个正常的 `ChatResult`
        if "temperature" in kwargs and isinstance(kwargs["temperature"], (int, float)):
            ls_params["ls_temperature"] = kwargs["temperature"]
        elif hasattr(self, "temperature") and isinstance(
            self.temperature, (int, float)
        ):
            ls_params["ls_temperature"] = self.temperature

                # 以便缓存 / `on_llm_end` 保持在现有的 generate 路径上。
        if "max_tokens" in kwargs and isinstance(kwargs["max_tokens"], int):
            ls_params["ls_max_tokens"] = kwargs["max_tokens"]
        elif hasattr(self, "max_tokens") and isinstance(self.max_tokens, int):
            ls_params["ls_max_tokens"] = self.max_tokens

        return ls_params

    def _get_ls_params_with_defaults(
        self,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> LangSmithParams:
        """将一系列提示词传递给模型并返回模型生成结果。"""
        ls_params = self._get_ls_params(stop=stop, **kwargs)
        ls_params["ls_integration"] = "langchain_chat_model"
        return ls_params

    def _get_llm_string(self, stop: list[str] | None = None, **kwargs: Any) -> str:
        if self.is_lc_serializable():
            params = {**kwargs, "stop": stop}
            param_string = str(sorted(params.items()))
                        # 如果未明确设置流式，则检查是否由 astream_events() 或 astream_log()
            # 隐式请求。如果未实现 `_stream` 则退出
                        # 用 .content_blocks 重写 .content
            serialized_repr = self._serialized
            _cleanup_llm_representation(serialized_repr, 1)
            llm_string = json.dumps(serialized_repr, sort_keys=True)
            return llm_string + "---" + param_string
        params = self._get_invocation_params(stop=stop, **kwargs)
        params = {**params, **kwargs}
        return str(sorted(params.items()))

    def generate(
        self,
        messages: list[list[BaseMessage]],
        stop: list[str] | None = None,
        callbacks: Callbacks = None,
        *,
        tags: list[str] | None = None,
        metadata: builtins.dict[str, Any] | None = None,
        run_name: str | None = None,
        run_id: uuid.UUID | None = None,
        **kwargs: Any,
    ) -> LLMResult:
        """此方法应利用支持批处理的模型提供的批处理 API。

        This method should make use of batched calls for models that expose a batched
        API.

        1. 利用批处理调用， 2. 需要模型输出超过最顶部生成值的信息， 3.
        构建对底层语言模型类型无关的链（例如纯文本补全模型与聊天模型）。

        1. Take advantage of batched calls,
        2. Need more output from the model than just the top generated value,
        3. Are building chains that are agnostic to the underlying language model
            type (e.g., pure text completion models vs chat models).

        Args:
            messages: 模型输出在首次出现这些子串中的任意一个时被截断。
            stop: callbacks: 要传递的 `Callbacks`。

                模型输出在首次出现以下任意子字符串时被截断。 callbacks: 传递过去的
                `Callbacks`。

                Used for executing additional functionality, such as logging or
                streaming, throughout generation.
            tags: The tags to apply.
            metadata: The metadata to apply.
            run_name: The name of the run.
            run_id: The ID of the run.
            **kwargs: Arbitrary additional keyword arguments.

                通常传递给模型提供方的 API 调用。

        Returns:
            An `LLMResult`, which contains a list of candidate `Generations` for each
                input prompt and additional model provider-specific output.

        """
        ls_structured_output_format = kwargs.pop(
            "ls_structured_output_format", None
        ) or kwargs.pop("structured_output_format", None)
        ls_structured_output_format_dict = _format_ls_structured_output(
            ls_structured_output_format
        )

        params = self._get_invocation_params(stop=stop, **kwargs)
        options = {"stop": stop, **ls_structured_output_format_dict}
        inheritable_metadata = {
            **(metadata or {}),
            **self._get_ls_params_with_defaults(stop=stop, **kwargs),
        }

        callback_manager = CallbackManager.configure(
            callbacks,
            self.callbacks,
            self.verbose,
            tags,
            self.tags,
            inheritable_metadata,
            self.metadata,
            langsmith_inheritable_metadata=_filter_invocation_params_for_tracing(
                params
            ),
        )
        messages_to_trace = [
            _format_for_tracing(message_list) for message_list in messages
        ]
        run_managers = callback_manager.on_chat_model_start(
            self._serialized,
            messages_to_trace,
            invocation_params=params,
            options=options,
            name=run_name,
            run_id=run_id,
            batch_size=len(messages),
        )
        results = []
        input_messages = [
            _normalize_messages(message_list) for message_list in messages
        ]
        for i, m in enumerate(input_messages):
            try:
                results.append(
                    self._generate_with_cache(
                        m,
                        stop=stop,
                        run_manager=run_managers[i] if run_managers else None,
                        **kwargs,
                    )
                )
            except BaseException as e:
                if run_managers:
                    generations_with_error_metadata = _generate_response_from_error(e)
                    run_managers[i].on_llm_error(
                        e,
                        response=LLMResult(
                            generations=[generations_with_error_metadata]
                        ),
                    )
                raise
        flattened_outputs = [
            LLMResult(generations=[res.generations], llm_output=res.llm_output)
            for res in results
        ]
        llm_output = self._combine_llm_outputs([res.llm_output for res in results])
        generations = [res.generations for res in results]
        output = LLMResult(generations=generations, llm_output=llm_output)
        if run_managers:
            run_infos = []
            for manager, flattened_output in zip(
                run_managers, flattened_outputs, strict=False
            ):
                manager.on_llm_end(flattened_output)
                run_infos.append(RunInfo(run_id=manager.run_id))
            output.run = run_infos
        return output

    async def agenerate(
        self,
        messages: list[list[BaseMessage]],
        stop: list[str] | None = None,
        callbacks: Callbacks = None,
        *,
        tags: list[str] | None = None,
        metadata: builtins.dict[str, Any] | None = None,
        run_name: str | None = None,
        run_id: uuid.UUID | None = None,
        **kwargs: Any,
    ) -> LLMResult:
        """异步地将一系列提示词传递给模型并返回生成结果。

        This method should make use of batched calls for models that expose a batched
        API.

        使用此方法时，您希望：

        1. Take advantage of batched calls,
        2. Need more output from the model than just the top generated value,
        3. Are building chains that are agnostic to the underlying language model
            type (e.g., pure text completion models vs chat models).

        Args:
            messages: 消息列表的列表。
            stop: 生成时使用的停止词。

                模型输出将在首次出现这些子字符串中的任何一个时被截断。 callbacks:
                要传递的 `Callbacks`。

                Used for executing additional functionality, such as logging or
                streaming, throughout generation.
            tags: The tags to apply.
            metadata: The metadata to apply.
            run_name: The name of the run.
            run_id: The ID of the run.
            **kwargs: Arbitrary additional keyword arguments.

                通常传递给模型提供方的 API 调用。

        Returns:
            An `LLMResult`, which contains a list of candidate `Generations` for each
                input prompt and additional model provider-specific output.

        """
        ls_structured_output_format = kwargs.pop(
            "ls_structured_output_format", None
        ) or kwargs.pop("structured_output_format", None)
        ls_structured_output_format_dict = _format_ls_structured_output(
            ls_structured_output_format
        )

        params = self._get_invocation_params(stop=stop, **kwargs)
        options = {"stop": stop, **ls_structured_output_format_dict}
        inheritable_metadata = {
            **(metadata or {}),
            **self._get_ls_params_with_defaults(stop=stop, **kwargs),
        }

        callback_manager = AsyncCallbackManager.configure(
            callbacks,
            self.callbacks,
            self.verbose,
            tags,
            self.tags,
            inheritable_metadata,
            self.metadata,
            langsmith_inheritable_metadata=_filter_invocation_params_for_tracing(
                params
            ),
        )

        messages_to_trace = [
            _format_for_tracing(message_list) for message_list in messages
        ]
        run_managers = await callback_manager.on_chat_model_start(
            self._serialized,
            messages_to_trace,
            invocation_params=params,
            options=options,
            name=run_name,
            batch_size=len(messages),
            run_id=run_id,
        )

        input_messages = [
            _normalize_messages(message_list) for message_list in messages
        ]
        results = await asyncio.gather(
            *[
                self._agenerate_with_cache(
                    m,
                    stop=stop,
                    run_manager=run_managers[i] if run_managers else None,
                    **kwargs,
                )
                for i, m in enumerate(input_messages)
            ],
            return_exceptions=True,
        )
        exceptions = []
        for i, res in enumerate(results):
            if isinstance(res, BaseException):
                if run_managers:
                    generations_with_error_metadata = _generate_response_from_error(res)
                    await run_managers[i].on_llm_error(
                        res,
                        response=LLMResult(
                            generations=[generations_with_error_metadata]
                        ),
                    )
                exceptions.append(res)
        if exceptions:
            if run_managers:
                await asyncio.gather(
                    *[
                        run_manager.on_llm_end(
                            LLMResult(
                                generations=[res.generations],  # type: ignore[union-attr]
                                llm_output=res.llm_output,  # type: ignore[union-attr]
                            )
                        )
                        for run_manager, res in zip(run_managers, results, strict=False)
                        if not isinstance(res, Exception)
                    ]
                )
            raise exceptions[0]
        flattened_outputs = [
            LLMResult(generations=[res.generations], llm_output=res.llm_output)  # type: ignore[union-attr]
            for res in results
        ]
        llm_output = self._combine_llm_outputs([res.llm_output for res in results])  # type: ignore[union-attr]
        generations = [res.generations for res in results]  # type: ignore[union-attr]
        output = LLMResult(generations=generations, llm_output=llm_output)
        await asyncio.gather(
            *[
                run_manager.on_llm_end(flattened_output)
                for run_manager, flattened_output in zip(
                    run_managers, flattened_outputs, strict=False
                )
            ]
        )
        if run_managers:
            output.run = [
                RunInfo(run_id=run_manager.run_id) for run_manager in run_managers
            ]
        return output

    @override
    def generate_prompt(
        self,
        prompts: list[PromptValue],
        stop: list[str] | None = None,
        callbacks: Callbacks = None,
        **kwargs: Any,
    ) -> LLMResult:
        prompt_messages = [p.to_messages() for p in prompts]
        return self.generate(prompt_messages, stop=stop, callbacks=callbacks, **kwargs)

    @override
    async def agenerate_prompt(
        self,
        prompts: list[PromptValue],
        stop: list[str] | None = None,
        callbacks: Callbacks = None,
        **kwargs: Any,
    ) -> LLMResult:
        prompt_messages = [p.to_messages() for p in prompts]
        return await self.agenerate(
            prompt_messages, stop=stop, callbacks=callbacks, **kwargs
        )

    def _generate_with_cache(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        llm_cache = self.cache if isinstance(self.cache, BaseCache) else get_llm_cache()
                # 除非显式设置为 False，否则我们应该检查缓存
                # None 缓存表示我们应该使用默认的全局缓存
                # 如果已配置。
        check_cache = self.cache is not False
        if check_cache:
            if llm_cache is not None:
                llm_string = self._get_llm_string(stop=stop, **kwargs)
                normalized_messages = [
                    (
                        msg.model_copy(update={"id": None})
                        if getattr(msg, "id", None) is not None
                        else msg
                    )
                    for msg in messages
                ]
                prompt = dumps(normalized_messages)
                cache_val = llm_cache.lookup(prompt, llm_string)
                if isinstance(cache_val, list):
                    converted_generations = self._convert_cached_generations(cache_val)
                    self._replay_v2_events_for_cache_hit(
                        converted_generations,
                        run_manager=run_manager,
                        **kwargs,
                    )
                    return ChatResult(generations=converted_generations)
            elif self.cache is None:
                pass
            else:
                msg = "Asked to cache, but no cache found at `langchain.cache`."
                raise ValueError(msg)

                # 在检查缓存后应用速率限制，因为
                # 我们通常不希望对缓存查找进行速率限制，但
                # 我们确实需要对 API 请求进行速率限制。
        if self.rate_limiter:
            self.rate_limiter.acquire(blocking=True)

                # v2 流式：当任何附加的处理器通过以下方式选择启用时，优先于 v1
                # _V2StreamingCallbackHandler。驱动协议事件生成器
                # (原生或 `_stream` 兼容桥接) 通过共享的辅助函数实现
                # on_stream_event 按事件逐个触发，然后返回一个正常的 ChatResult
                # 因此缓存 / `on_llm_end` 保持在现有的 generate 路径上。
        if self._should_use_protocol_streaming(
            async_api=False,
            run_manager=run_manager,
            **kwargs,
        ):
            stream_accum = ChatModelStream(
                message_id=(
                    f"{LC_ID_PREFIX}-{run_manager.run_id}" if run_manager else None
                )
            )
            assert run_manager is not None  # noqa: S101
            for _event in self._iter_v2_events(
                messages,
                run_manager=run_manager,
                stream=stream_accum,
                stop=stop,
                **kwargs,
            ):
                pass
            if stream_accum.output_message is None:
                msg = "v2 stream finished without producing a message"
                raise RuntimeError(msg)
            result = ChatResult(
                generations=[ChatGeneration(message=stream_accum.output_message)]
            )
                # 如果未显式设置 stream，则检查是否隐式请求了
                # astream_events() 或 astream_log()。如果未实现 _stream，则退出
        elif self._should_stream(
            async_api=False,
            run_manager=run_manager,
            **kwargs,
        ):
            chunks: list[ChatGenerationChunk] = []
            run_id: str | None = (
                f"{LC_ID_PREFIX}-{run_manager.run_id}" if run_manager else None
            )
            yielded = False
            index = -1
            index_type = ""
            for chunk in self._stream(messages, stop=stop, **kwargs):
                chunk.message.response_metadata = _gen_info_and_msg_metadata(chunk)
                if self.output_version == "v1":
                                        # 用 .content_blocks 覆盖 .content
                    chunk.message = _update_message_content_to_blocks(
                        chunk.message, "v1"
                    )
                    for block in cast(
                        "list[types.ContentBlock]", chunk.message.content
                    ):
                        if block["type"] != index_type:
                            index_type = block["type"]
                            index += 1
                        if "index" not in block:
                            block["index"] = index
                if run_manager:
                    if chunk.message.id is None:
                        chunk.message.id = run_id
                    run_manager.on_llm_new_token(chunk.message.content, chunk=chunk)
                chunks.append(chunk)
                yielded = True

                        # 如果尚未生成，则输出一个 final empty chunk，其 chunk_position="last"
            if (
                yielded
                and isinstance(chunk.message, AIMessageChunk)
                and not chunk.message.chunk_position
            ):
                empty_content: str | list[str | dict[str, Any]] = (
                    "" if isinstance(chunk.message.content, str) else []
                )
                chunk = ChatGenerationChunk(
                    message=AIMessageChunk(
                        content=empty_content, chunk_position="last", id=run_id
                    )
                )
                if run_manager:
                    run_manager.on_llm_new_token("", chunk=chunk)
                chunks.append(chunk)
            result = generate_from_stream(iter(chunks))
        elif _signature_parameters(self._generate).get("run_manager"):
            result = self._generate(
                messages, stop=stop, run_manager=run_manager, **kwargs
            )
        else:
            result = self._generate(messages, stop=stop, **kwargs)

        if self.output_version == "v1":
                        # .content 用 .content_blocks 覆盖
            for generation in result.generations:
                generation.message = _update_message_content_to_blocks(
                    generation.message, "v1"
                )

                # 将响应元数据添加到每次生成中
        for idx, generation in enumerate(result.generations):
            if run_manager and generation.message.id is None:
                generation.message.id = f"{LC_ID_PREFIX}-{run_manager.run_id}-{idx}"
            generation.message.response_metadata = _gen_info_and_msg_metadata(
                generation
            )
        if len(result.generations) == 1 and result.llm_output is not None:
            result.generations[0].message.response_metadata = {
                **result.llm_output,
                **result.generations[0].message.response_metadata,
            }
        if check_cache and llm_cache is not None:
            llm_cache.update(prompt, llm_string, result.generations)
        return result

    async def _agenerate_with_cache(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        llm_cache = self.cache if isinstance(self.cache, BaseCache) else get_llm_cache()
                # 除非显式设置为 False，否则我们应该检查缓存
                # None 缓存表示应使用默认全局缓存
                # 如果已配置。
        check_cache = self.cache is not False
        if check_cache:
            if llm_cache is not None:
                llm_string = self._get_llm_string(stop=stop, **kwargs)
                normalized_messages = [
                    (
                        msg.model_copy(update={"id": None})
                        if getattr(msg, "id", None) is not None
                        else msg
                    )
                    for msg in messages
                ]
                prompt = dumps(normalized_messages)
                cache_val = await llm_cache.alookup(prompt, llm_string)
                if isinstance(cache_val, list):
                    converted_generations = self._convert_cached_generations(cache_val)
                    await self._areplay_v2_events_for_cache_hit(
                        converted_generations,
                        run_manager=run_manager,
                        **kwargs,
                    )
                    return ChatResult(generations=converted_generations)
            elif self.cache is None:
                pass
            else:
                msg = "Asked to cache, but no cache found at `langchain.cache`."
                raise ValueError(msg)

                # 在检查缓存后应用速率限制器，因为
                # 我们通常不希望对缓存查询进行速率限制，但
                # 我们确实希望对 API 请求进行速率限制。
        if self.rate_limiter:
            await self.rate_limiter.aacquire(blocking=True)

                # v2 流式：参见 `_generate_with_cache` 中的同步对应实现。
        if self._should_use_protocol_streaming(
            async_api=True,
            run_manager=run_manager,
            **kwargs,
        ):
            stream_accum = AsyncChatModelStream(
                message_id=(
                    f"{LC_ID_PREFIX}-{run_manager.run_id}" if run_manager else None
                )
            )
            assert run_manager is not None  # noqa: S101
            async for _event in self._aiter_v2_events(
                messages,
                run_manager=run_manager,
                stream=stream_accum,
                stop=stop,
                **kwargs,
            ):
                pass
            if stream_accum.output_message is None:
                msg = "v2 stream finished without producing a message"
                raise RuntimeError(msg)
            result = ChatResult(
                generations=[ChatGeneration(message=stream_accum.output_message)]
            )
                # 如果未显式设置 stream，则检查是否由
                # astream_events() 或 astream_log() 隐式请求。如果 _astream 未实现则退出
        elif self._should_stream(
            async_api=True,
            run_manager=run_manager,
            **kwargs,
        ):
            chunks: list[ChatGenerationChunk] = []
            run_id: str | None = (
                f"{LC_ID_PREFIX}-{run_manager.run_id}" if run_manager else None
            )
            yielded = False
            index = -1
            index_type = ""
            async for chunk in self._astream(messages, stop=stop, **kwargs):
                chunk.message.response_metadata = _gen_info_and_msg_metadata(chunk)
                if self.output_version == "v1":
                                        # .content 用 .content_blocks 覆盖
                    chunk.message = _update_message_content_to_blocks(
                        chunk.message, "v1"
                    )
                    for block in cast(
                        "list[types.ContentBlock]", chunk.message.content
                    ):
                        if block["type"] != index_type:
                            index_type = block["type"]
                            index += 1
                        if "index" not in block:
                            block["index"] = index
                if run_manager:
                    if chunk.message.id is None:
                        chunk.message.id = run_id
                    await run_manager.on_llm_new_token(
                        chunk.message.content, chunk=chunk
                    )
                chunks.append(chunk)
                yielded = True

                        # 如果尚未生成，则输出一个 final empty chunk，其 chunk_position="last"
            if (
                yielded
                and isinstance(chunk.message, AIMessageChunk)
                and not chunk.message.chunk_position
            ):
                empty_content: str | list[str | dict[str, Any]] = (
                    "" if isinstance(chunk.message.content, str) else []
                )
                chunk = ChatGenerationChunk(
                    message=AIMessageChunk(
                        content=empty_content, chunk_position="last", id=run_id
                    )
                )
                if run_manager:
                    await run_manager.on_llm_new_token("", chunk=chunk)
                chunks.append(chunk)
            result = generate_from_stream(iter(chunks))
        elif _signature_parameters(self._agenerate).get("run_manager"):
            result = await self._agenerate(
                messages, stop=stop, run_manager=run_manager, **kwargs
            )
        else:
            result = await self._agenerate(messages, stop=stop, **kwargs)

        if self.output_version == "v1":
                        # .content 用 .content_blocks 覆盖
            for generation in result.generations:
                generation.message = _update_message_content_to_blocks(
                    generation.message, "v1"
                )

                # 将响应元数据添加到每次生成中
        for idx, generation in enumerate(result.generations):
            if run_manager and generation.message.id is None:
                generation.message.id = f"{LC_ID_PREFIX}-{run_manager.run_id}-{idx}"
            generation.message.response_metadata = _gen_info_and_msg_metadata(
                generation
            )
        if len(result.generations) == 1 and result.llm_output is not None:
            result.generations[0].message.response_metadata = {
                **result.llm_output,
                **result.generations[0].message.response_metadata,
            }
        if check_cache and llm_cache is not None:
            await llm_cache.aupdate(prompt, llm_string, result.generations)
        return result

    @abstractmethod
    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        """生成结果。

        Args:
            messages: 用于生成的消息。
            stop: 可选的停止词列表，用于生成。
            run_manager: 可选的回调管理器，用于此次调用。
            **kwargs: 传递给模型的额外关键字参数。

        Returns:
            聊天结果。
        """

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        """生成结果。

        Args:
            messages: 用于生成的消息。
            stop: 可选的停止词列表，用于生成。
            run_manager: 可选的回调管理器，用于此次调用。
            **kwargs: 传递给模型的额外关键字参数。

        Returns:
            聊天结果。
        """
        return await run_in_executor(
            None,
            self._generate,
            messages,
            stop,
            run_manager.get_sync() if run_manager else None,
            **kwargs,
        )

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        """流式输出模型的输出。

        Args:
            messages: 用于生成的消息。
            stop: 可选的停止词列表，用于生成。
            run_manager: 可选的回调管理器，用于此次调用。
            **kwargs: 传递给模型的额外关键字参数。

        Yields:
            聊天生成的分块内容。
        """
        raise NotImplementedError

    async def _astream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        """流式输出模型的输出。

        Args:
            messages: 用于生成的对话消息。
            stop: 可选的停止词列表，用于生成时使用。
            run_manager: 可选的回调管理器，用于此次调用。
            **kwargs: 额外的关键字参数，将传递给模型。

        Yields:
            对话生成的分块内容。
        """
        iterator = await run_in_executor(
            None,
            self._stream,
            messages,
            stop,
            run_manager.get_sync() if run_manager else None,
            **kwargs,
        )
        done = object()
        while True:
            item = await run_in_executor(
                None,
                next,
                iterator,
                done,
            )
            if item is done:
                break
            yield item  # type: ignore[misc]

    async def _call_async(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        callbacks: Callbacks = None,
        **kwargs: Any,
    ) -> BaseMessage:
        result = await self.agenerate(
            [messages], stop=stop, callbacks=callbacks, **kwargs
        )
        generation = result.generations[0][0]
        if isinstance(generation, ChatGeneration):
            return generation.message
        msg = "Unexpected generation type"
        raise ValueError(msg)

    @property
    @abstractmethod
    def _llm_type(self) -> str:
        """对话模型的返回类型。"""

    @deprecated("1.4.2", alternative="asdict", removal="2.0.0")
    @override
    def dict(self, **_kwargs: Any) -> builtins.dict[str, Any]:
        """已弃用 - 请改用 `asdict()`。

        返回对话模型的字典表示形式。
        """
        return self.asdict()

    def asdict(self) -> builtins.dict[str, Any]:
        """返回对话模型的字典表示形式。"""
        starter_dict = dict(self._identifying_params)
        starter_dict["_type"] = self._llm_type
        return starter_dict

    def _dict_for_compat(self) -> builtins.dict[str, Any]:
        """返回对话模型的字典表示形式，同时保留已弃用的覆盖项。"""
        with suppress_langchain_deprecation_warning():
            return self.dict()

    @override
    def bind(self, **kwargs: Any) -> _ChatModelBinding:
        """将关键字参数绑定到此对话模型，返回一个类型的 `_ChatModelBinding`。

        重写 `Runnable.bind`，以确保结果保留对话模型特有的
        `stream_events` / `astream_events` 重载。若无此重写，
        `model.bind(...).stream_events(version="v3")` 的类型将为
        `Iterator[Any]`，而 `await model.bind(...).astream_events(version="v3")`
        的类型将为 `Any`，迫使调用者进行 `cast` 操作。
        """
        return _ChatModelBinding(bound=self, kwargs=kwargs, config={})

    def bind_tools(
        self,
        tools: Sequence[builtins.dict[str, Any] | type | Callable[..., Any] | BaseTool],
        *,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, AIMessage]:
        """将工具绑定到模型。

        Args:
            tools: 要绑定到模型的工具序列。
            tool_choice: 要使用的工具。若为 "any"，则可使用任意工具。

        Returns:
            一个返回消息的 Runnable。

        """
        raise NotImplementedError

    def with_structured_output(
        self,
        schema: builtins.dict[str, Any] | type,
        *,
        include_raw: bool = False,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, builtins.dict[str, Any] | BaseModel]:
        """将输出格式化为匹配给定模式的模型包装器。

        Args:
            schema: 输出模式。可作为以下形式传入：

                - OpenAI 函数/工具模式，
                - JSON Schema，
                - `TypedDict` 类，
                - 或 Pydantic 类。

                If `schema` is a Pydantic class then the model output will be a
                Pydantic instance of that class, and the model-generated fields will be
                validated by the Pydantic class. Otherwise the model output will be a
                dict and will not be validated.

                有关如何正确指定模式字段的类型和描述，请参阅
                `langchain_core.utils.function_calling.convert_to_openai_tool`。

            include_raw:
                若为 `False`，则仅返回解析后的结构化输出。

                若在模型输出解析过程中发生错误，将被抛出。

                若为 `True`，则同时返回原始模型响应（一个 `BaseMessage`）和
                解析后的模型响应。

                若在输出解析过程中发生错误，将被捕获并一同返回。

                最终输出始终是一个包含键 `'raw'`、`'parsed'` 和
                `'parsing_error'` 的 `dict`。

        Raises:
            ValueError: 如果存在任何不支持的 `kwargs`。 NotImplementedError:
            如果模型未实现 `with_structured_output()`。

        Returns:
            A `Runnable` that takes same inputs as a
                `langchain_core.language_models.chat.BaseChatModel`. If `include_raw` is
                `False` and `schema` is a Pydantic class, `Runnable` outputs an instance
                of `schema` (i.e., a Pydantic object). Otherwise, if `include_raw` is
                `False` then `Runnable` outputs a `dict`.

                如果 `include_raw` 为 `True`，则 `Runnable` 输出一个包含以下键的 `dict`：

                - `'raw'`: `BaseMessage`
                - `'parsed'`: `None` if there was a parsing error, otherwise the type
                    depends on the `schema` as described above.
                - `'parsing_error'`: `BaseException | None`

        ???+ example "Pydantic schema (`include_raw=False`)"

            ```python
            from pydantic import BaseModel


            class AnswerWithJustification(BaseModel):
                '''An answer to the user question along with justification for the answer.'''

                answer: str
                justification: str


            model = ChatModel(model="model-name", temperature=0)
            structured_model = model.with_structured_output(AnswerWithJustification)

            structured_model.invoke(
                "What weighs more a pound of bricks or a pound of feathers"
            )

            # -> AnswerWithJustification(
            #     answer='They weigh the same',
            #     justification='Both a pound of bricks and a pound of feathers weigh one pound. The weight is the same, but the volume or density of the objects may differ.'
            # )
            ```

        ??? example "Pydantic schema (`include_raw=True`)"

            ```python
            from pydantic import BaseModel


            class AnswerWithJustification(BaseModel):
                '''An answer to the user question along with justification for the answer.'''

                answer: str
                justification: str


            model = ChatModel(model="model-name", temperature=0)
            structured_model = model.with_structured_output(
                AnswerWithJustification, include_raw=True
            )

            structured_model.invoke(
                "What weighs more a pound of bricks or a pound of feathers"
            )
            # -> {
            #     'raw': AIMessage(content='', additional_kwargs={'tool_calls': [{'id': 'call_Ao02pnFYXD6GN1yzc0uXPsvF', 'function': {'arguments': '{"answer":"They weigh the same.","justification":"Both a pound of bricks and a pound of feathers weigh one pound. The weight is the same, but the volume or density of the objects may differ."}', 'name': 'AnswerWithJustification'}, 'type': 'function'}]}),
            #     'parsed': AnswerWithJustification(answer='They weigh the same.', justification='Both a pound of bricks and a pound of feathers weigh one pound. The weight is the same, but the volume or density of the objects may differ.'),
            #     'parsing_error': None
            # }
            ```

        ??? example "Dictionary schema (`include_raw=False`)"

            ```python
            from pydantic import BaseModel
            from langchain_core.utils.function_calling import convert_to_openai_tool


            class AnswerWithJustification(BaseModel):
                '''An answer to the user question along with justification for the answer.'''

                answer: str
                justification: str


            dict_schema = convert_to_openai_tool(AnswerWithJustification)
            model = ChatModel(model="model-name", temperature=0)
            structured_model = model.with_structured_output(dict_schema)

            structured_model.invoke(
                "What weighs more a pound of bricks or a pound of feathers"
            )
            # -> {
            #     'answer': 'They weigh the same',
            #     'justification': 'Both a pound of bricks and a pound of feathers weigh one pound. The weight is the same, but the volume and density of the two substances differ.'
            # }
            ```

        !!! warning "行为变更：`langchain-core` 0.2.26 版本中"

            增加了对 `TypedDict` 类的支持。

        """  # noqa: E501
        _ = kwargs.pop("method", None)
        _ = kwargs.pop("strict", None)
        if kwargs:
            msg = f"Received unsupported arguments {kwargs}"
            raise ValueError(msg)

        if type(self).bind_tools is BaseChatModel.bind_tools:
            msg = "with_structured_output is not implemented for this model."
            raise NotImplementedError(msg)

        llm = self.bind_tools(
            [schema],
            tool_choice="any",
            ls_structured_output_format={
                "kwargs": {"method": "function_calling"},
                "schema": schema,
            },
        )
        output_parser: JsonOutputToolsParser
        if isinstance(schema, type) and is_basemodel_subclass(schema):
            output_parser = PydanticToolsParser(tools=[schema], first_tool_only=True)
        else:
            key_name = convert_to_openai_tool(schema)["function"]["name"]
            output_parser = JsonOutputKeyToolsParser(
                key_name=key_name, first_tool_only=True
            )
        if include_raw:
            parser_assign = RunnablePassthrough.assign(
                parsed=itemgetter("raw") | output_parser, parsing_error=lambda _: None
            )
            parser_none = RunnablePassthrough.assign(parsed=lambda _: None)
            parser_with_fallback = parser_assign.with_fallbacks(
                [parser_none], exception_key="parsing_error"
            )
            return RunnableMap(raw=llm) | parser_with_fallback
        return llm | output_parser


class _ChatModelBinding(RunnableBinding[LanguageModelInput, AIMessage]):  # type: ignore[no-redef]
    """`RunnableBinding`，保留聊天模型类型的 v3 重载。

    Returned by `BaseChatModel.bind` so that callers of the bound runnable's
    `stream_events(version="v3")` / `astream_events(version="v3")` get the
    typed `ChatModelStream` / `AsyncChatModelStream` back without needing
    `cast`. At runtime this is a plain `RunnableBinding`; the subclass
    exists purely to give the type checker a more specific surface.

    由于 `RunnableBinding.bind` 通过 `self.__class__(...)`
    构造结果，因此聊天模型的类型缩小会保留在后续的 `bind` / `with_config` 调用中。
    """

    @classmethod
    @override
    def lc_id(cls) -> list[str]:
        """序列化为 `RunnableBinding`。

        At runtime this class is behaviorally identical to `RunnableBinding`;
        keeping the serialized id stable means existing snapshots and the
        load mapping continue to work without registering a new entry.
        """
        return [*cls.get_lc_namespace(), "RunnableBinding"]

    @overload  # type: ignore[override]
    def stream_events(
        self,
        input: LanguageModelInput,
        config: RunnableConfig | None = None,
        *,
        version: Literal["v1", "v2"] = "v2",
        **kwargs: Any,
    ) -> Iterator[StreamEvent]: ...

    @overload
    def stream_events(
        self,
        input: LanguageModelInput,
        config: RunnableConfig | None = None,
        *,
        version: Literal["v3"],
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> ChatModelStream: ...

    def stream_events(
        self,
        input: LanguageModelInput,
        config: RunnableConfig | None = None,
        *,
        version: Literal["v1", "v2", "v3"] = "v2",
        **kwargs: Any,
    ) -> Iterator[StreamEvent] | ChatModelStream:
        return super().stream_events(input, config, version=version, **kwargs)

    @overload
    def astream_events(
        self,
        input: LanguageModelInput,
        config: RunnableConfig | None = None,
        *,
        version: Literal["v1", "v2"] = "v2",
        **kwargs: Any,
    ) -> AsyncIterator[StreamEvent]: ...

    @overload
    def astream_events(
        self,
        input: LanguageModelInput,
        config: RunnableConfig | None = None,
        *,
        version: Literal["v3"],
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> Awaitable[AsyncChatModelStream]: ...

    def astream_events(
        self,
        input: LanguageModelInput,
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[StreamEvent] | Awaitable[AsyncChatModelStream]:
        return cast(
            "AsyncIterator[StreamEvent] | Awaitable[AsyncChatModelStream]",
            super().astream_events(input, config, **kwargs),
        )


class SimpleChatModel(BaseChatModel):
    """简化版的聊天模型实现，供继承使用。

    !!! note
        This implementation is primarily here for backwards compatibility. For new
        implementations, please use `BaseChatModel` directly.

    """

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        output_str = self._call(messages, stop=stop, run_manager=run_manager, **kwargs)
        message = AIMessage(content=output_str)
        generation = ChatGeneration(message=message)
        return ChatResult(generations=[generation])

    @abstractmethod
    def _call(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> str:
        """更简洁的接口。"""

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        return await run_in_executor(
            None,
            self._generate,
            messages,
            stop=stop,
            run_manager=run_manager.get_sync() if run_manager else None,
            **kwargs,
        )


def _gen_info_and_msg_metadata(
    generation: ChatGeneration | ChatGenerationChunk,
) -> dict[str, Any]:
    """提取响应元数据。

    从生成信息中移除 GATEWAY_METADATA。
    """
    generation_info = generation.generation_info or {}
    return {
        **{
            key: value
            for key, value in generation_info.items()
            if key != GATEWAY_METADATA_RESPONSE_KEY
        },
        **generation.message.response_metadata,
    }


_MAX_CLEANUP_DEPTH = 100


def _cleanup_llm_representation(serialized: Any, depth: int) -> None:
    """从序列化对象中移除不可序列化对象。"""
    if depth > _MAX_CLEANUP_DEPTH:  # 对于异常情况不进行协作
        return

    if not isinstance(serialized, dict):
        return

    if (
        "type" in serialized
        and serialized["type"] == "not_implemented"
        and "repr" in serialized
    ):
        del serialized["repr"]

    if "graph" in serialized:
        del serialized["graph"]

    if "kwargs" in serialized:
        kwargs = serialized["kwargs"]

        for value in kwargs.values():
            _cleanup_llm_representation(value, depth + 1)

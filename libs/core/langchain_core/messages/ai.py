"""AI 消息。"""

import itertools
import json
import logging
import operator
from collections.abc import Sequence
from typing import Any, Literal, cast, overload

from pydantic import Field, model_validator
from typing_extensions import NotRequired, Self, TypedDict, override

from langchain_core.messages import content as types
from langchain_core.messages.base import (
    BaseMessage,
    BaseMessageChunk,
    _extract_reasoning_from_additional_kwargs,
    merge_content,
)
from langchain_core.messages.content import InvalidToolCall
from langchain_core.messages.tool import (
    ToolCall,
    ToolCallChunk,
    default_tool_chunk_parser,
    default_tool_parser,
)
from langchain_core.messages.tool import invalid_tool_call as create_invalid_tool_call
from langchain_core.messages.tool import tool_call as create_tool_call
from langchain_core.messages.tool import tool_call_chunk as create_tool_call_chunk
from langchain_core.utils._merge import merge_dicts, merge_lists
from langchain_core.utils.json import parse_partial_json
from langchain_core.utils.usage import _dict_int_op
from langchain_core.utils.utils import LC_AUTO_PREFIX, LC_ID_PREFIX

logger = logging.getLogger(__name__)


class InputTokenDetails(TypedDict, total=False):
    """输入 token 数量的分解。

    不需要加总为完整的输入 token 数量。不需要包含所有键。

    Example:
        ```python
        {
            "audio": 10,
            "cache_creation": 200,
            "cache_read": 100,
        }
        ```

    也可能包含额外的提供方特定键。

    !!! version-added "在 `langchain-core` 0.3.9 中新增"
    """

    audio: int
    """音频输入 token。"""

    cache_creation: int
    """被缓存且发生缓存未命中时的输入 token。

    由于发生了缓存未命中，这些 token 用于创建缓存。
    """

    cache_read: int
    """被缓存且发生缓存命中的输入 token。

    Since there was a cache hit, the tokens were read from the cache. More precisely,
    the model state given these tokens was read from the cache.
    """


class OutputTokenDetails(TypedDict, total=False):
    """输出 token 数量的分解。

    不需要加总为完整的输出 token 数量。不需要包含所有键。

    Example:
        ```python
        {
            "audio": 10,
            "reasoning": 200,
        }
        ```

    也可能包含额外的提供方特定键。

    !!! version-added "在 `langchain-core` 0.3.9 中新增"

    """

    audio: int
    """音频输出 token。"""

    reasoning: int
    """推理输出 token。

    模型在思维链过程中生成但未作为模型输出返回的 token。
    """


class UsageMetadata(TypedDict):
    """消息的使用元数据，例如 token 数量。

    这是一种跨模型一致的标准 token 使用表示方式。

    Example:
        ```python
        {
            "input_tokens": 350,
            "output_tokens": 240,
            "total_tokens": 590,
            "input_token_details": {
                "audio": 10,
                "cache_creation": 200,
                "cache_read": 100,
            },
            "output_token_details": {
                "audio": 10,
                "reasoning": 200,
            },
        }
        ```

    !!! warning "行为变更在 `langchain-core` 0.3.9 中"

        新增了 `input_token_details` 和 `output_token_details`。

    !!! note "LangSmith SDK"

        LangSmith SDK 也拥有一个 `UsageMetadata` 类。虽然两者共享字段， 但 LangSmith 的
        `UsageMetadata` 包含额外字段，用于捕获 LangSmith 平台使用的成本信息。
    """

    input_tokens: int
    """输入（或提示词）token 数量。所有输入 token 类型的总和。"""

    output_tokens: int
    """输出（或完成内容）token 数量。所有输出 token 类型的总和。"""

    total_tokens: int
    """总 token 数量。`input_tokens` + `output_tokens` 的总和。"""

    input_token_details: NotRequired[InputTokenDetails]
    """输入 token 数量的分解。

    不需要加总为完整的输入 token 数量。不需要包含所有键。
    """

    output_token_details: NotRequired[OutputTokenDetails]
    """输出 token 数量的分解。

    不需要加总为完整的输出 token 数量。不需要包含所有键。
    """


class AIMessage(BaseMessage):
    """来自 AI 的消息。

    `AIMessage` 是聊天模型对提示词的响应结果。

    此消息代表模型的输出，包括模型返回的原始输出以及由 LangChain 框架添加的标准字段
    （例如，工具调用、使用元数据）。
    """

    tool_calls: list[ToolCall] = Field(default_factory=list)
    """如果存在，与该消息关联的工具调用。"""

    invalid_tool_calls: list[InvalidToolCall] = Field(default_factory=list)
    """如果存在，与该消息关联且解析出错的工具调用。"""

    usage_metadata: UsageMetadata | None = None
    """如果存在，消息的使用元数据，例如 token 数量。

    这是一种跨模型一致的标准 token 使用表示方式。
    """

    type: Literal["ai"] = "ai"
    """消息的类型（用于反序列化）。"""

    @overload
    def __init__(
        self,
        content: str | list[str | dict[Any, Any]],
        **kwargs: Any,
    ) -> None: ...

    @overload
    def __init__(
        self,
        content: str | list[str | dict[Any, Any]] | None = None,
        content_blocks: list[types.ContentBlock] | None = None,
        **kwargs: Any,
    ) -> None: ...

    def __init__(
        self,
        content: str | list[str | dict[Any, Any]] | None = None,
        content_blocks: list[types.ContentBlock] | None = None,
        **kwargs: Any,
    ) -> None:
        """初始化一个 `AIMessage`。

        将 `content` 作为位置参数指定，或使用 `content_blocks` 进行类型标注。

        Args:
            content: 消息的内容。
            content_blocks: 类型化的标准内容。
            **kwargs: 传递给父类的额外参数。
        """
        if content_blocks is not None:
                        # 如果 content_blocks 中存在工具调用，但 tool_calls 中没有，则添加它们
            content_tool_calls = [
                block for block in content_blocks if block.get("type") == "tool_call"
            ]
            if content_tool_calls and "tool_calls" not in kwargs:
                kwargs["tool_calls"] = content_tool_calls

            super().__init__(
                content=cast("list[str | dict[Any, Any]]", content_blocks),
                **kwargs,
            )
        else:
            super().__init__(content=content, **kwargs)

    @property
    def lc_attributes(self) -> dict[str, Any]:
        """需要序列化的属性。

        Includes all attributes, even if they are derived from other initialization
        arguments.
        """
        return {
            "tool_calls": self.tool_calls,
            "invalid_tool_calls": self.invalid_tool_calls,
        }

    @property
    def content_blocks(self) -> list[types.ContentBlock]:
        """从消息返回标准、类型的 `ContentBlock` 字典。

        如果消息具有已知的模型提供方，优先使用提供方特定的转换器，再回退到尽力解析。详情请参见
        `BaseMessage` 上的属性。
        """
        if self.response_metadata.get("output_version") == "v1" and isinstance(
            self.content, list
        ):
                        # 仅在 content 为列表时进行短路（假设 v1 版本下
                        # 已包含 ContentBlock 字典；该转换未经检查）。参见
                        # AIMessageChunk.content_blocks 以获取完整理由。
            return cast("list[types.ContentBlock]", self.content)

        model_provider = self.response_metadata.get("model_provider")
        if model_provider:
            from langchain_core.messages.block_translators import (  # noqa: PLC0415
                get_translator,
            )

            translator = get_translator(model_provider)
            if translator:
                try:
                    return translator["translate_content"](self)
                except NotImplementedError:
                    pass

                # 否则，使用最佳努力方式解析
        blocks = super().content_blocks

        if self.tool_calls:
                        # 如果 content 中缺少，从 tool_calls 中添加
            content_tool_call_ids = {
                block.get("id")
                for block in self.content
                if isinstance(block, dict) and block.get("type") == "tool_call"
            }
            for tool_call in self.tool_calls:
                if (id_ := tool_call.get("id")) and id_ not in content_tool_call_ids:
                    tool_call_block: types.ToolCall = {
                        "type": "tool_call",
                        "id": id_,
                        "name": tool_call["name"],
                        "args": tool_call["args"],
                    }
                    if "index" in tool_call:
                        tool_call_block["index"] = tool_call["index"]  # type: ignore[typeddict-item]
                    if "extras" in tool_call:
                        tool_call_block["extras"] = tool_call["extras"]  # type: ignore[typeddict-item]
                    blocks.append(tool_call_block)

                # 从 additional_kwargs 中提取最佳努力推理内容
                # 仅在尚未存在时添加推理内容
                # 插入到所有其他块之前，以确保推理位于开头
        has_reasoning = any(block.get("type") == "reasoning" for block in blocks)
        if not has_reasoning and (
            reasoning_block := _extract_reasoning_from_additional_kwargs(self)
        ):
            blocks.insert(0, reasoning_block)

        return blocks

        # TODO：如有可能，移除此逻辑，减少更改的破坏性
    @model_validator(mode="before")
    @classmethod
    def _backwards_compat_tool_calls(cls, values: dict[str, Any]) -> Any:
        check_additional_kwargs = not any(
            values.get(k)
            for k in ("tool_calls", "invalid_tool_calls", "tool_call_chunks")
        )
        if check_additional_kwargs and (
            raw_tool_calls := values.get("additional_kwargs", {}).get("tool_calls")
        ):
            try:
                if issubclass(cls, AIMessageChunk):
                    values["tool_call_chunks"] = default_tool_chunk_parser(
                        raw_tool_calls
                    )
                else:
                    parsed_tool_calls, parsed_invalid_tool_calls = default_tool_parser(
                        raw_tool_calls
                    )
                    values["tool_calls"] = parsed_tool_calls
                    values["invalid_tool_calls"] = parsed_invalid_tool_calls
            except Exception:
                logger.debug("Failed to parse tool calls", exc_info=True)

                # 确保所有类似工具调用的字典中 "type" 正确设置。
        if tool_calls := values.get("tool_calls"):
            values["tool_calls"] = [
                create_tool_call(
                    **{k: v for k, v in tc.items() if k not in {"type", "extras"}}
                )
                for tc in tool_calls
            ]
        if invalid_tool_calls := values.get("invalid_tool_calls"):
            values["invalid_tool_calls"] = [
                create_invalid_tool_call(**{k: v for k, v in tc.items() if k != "type"})
                for tc in invalid_tool_calls
            ]

        if tool_call_chunks := values.get("tool_call_chunks"):
            values["tool_call_chunks"] = [
                create_tool_call_chunk(**{k: v for k, v in tc.items() if k != "type"})
                for tc in tool_call_chunks
            ]

        return values

    @override
    def pretty_repr(self, html: bool = False) -> str:
        """返回用于显示的美化消息表示形式。

        Args:
            html: 是否返回 HTML 格式的字符串。

        Returns:
            消息的美化表示形式。

        Example:
            ```python
            from langchain_core.messages import AIMessage

            msg = AIMessage(
                content="Let me check the weather.",
                tool_calls=[
                    {"name": "get_weather", "args": {"city": "Paris"}, "id": "1"}
                ],
            )
            ```

            结果为：
            ```python
            >>> print(msg.pretty_repr())
            ================================== Ai Message ==================================

            Let me check the weather.
            Tool Calls:
              get_weather (1)
             Call ID: 1
              Args:
                city: Paris
            ```
        """  # noqa: E501
        base = super().pretty_repr(html=html)
        lines = []

        def _format_tool_args(tc: ToolCall | InvalidToolCall) -> list[str]:
            lines = [
                f"  {tc.get('name', 'Tool')} ({tc.get('id')})",
                f" Call ID: {tc.get('id')}",
            ]
            if tc.get("error"):
                lines.append(f"  Error: {tc.get('error')}")
            lines.append("  Args:")
            args = tc.get("args")
            if isinstance(args, str):
                lines.append(f"    {args}")
            elif isinstance(args, dict):
                for arg, value in args.items():
                    lines.append(f"    {arg}: {value}")
            return lines

        if self.tool_calls:
            lines.append("Tool Calls:")
            for tc in self.tool_calls:
                lines.extend(_format_tool_args(tc))
        if self.invalid_tool_calls:
            lines.append("Invalid Tool Calls:")
            for itc in self.invalid_tool_calls:
                lines.extend(_format_tool_args(itc))
        return (base.strip() + "\n" + "\n".join(lines)).strip()


class AIMessageChunk(AIMessage, BaseMessageChunk):
    """来自 AI 的消息块（流式传输时产生）。"""

        # 忽略 mypy 的重新赋值，因为我们正在覆盖该值
        # 以确保 chunk 变体可与非 chunk 变体区分。
        # 仅在 content 已经是 ContentBlock 字典列表时进行短路。
    type: Literal["AIMessageChunk"] = "AIMessageChunk"  # type: ignore[assignment]
    """消息的类型（用于反序列化）。"""

    tool_call_chunks: list[ToolCallChunk] = Field(default_factory=list)
    """如果提供，与该消息关联的工具调用块。"""

    chunk_position: Literal["last"] | None = None
    """可选的跨度，由聚合的 `AIMessageChunk` 表示。

    如果带有 `chunk_position="last"` 的块被聚合到流中，
    消息内容中的 `tool_call_chunks` 将被解析为 `tool_calls`。
    """

    @property
    @override
    def lc_attributes(self) -> dict[str, Any]:
        return {
            "tool_calls": self.tool_calls,
            "invalid_tool_calls": self.invalid_tool_calls,
        }

    @property
    def content_blocks(self) -> list[types.ContentBlock]:
        """从消息返回标准、类型的 `ContentBlock` 字典。"""
        if self.response_metadata.get("output_version") == "v1" and isinstance(
            self.content, list
        ):
                        # 某些流式实现即使设置了 output_version="v1"，也保持 content 为字符串
                        # （例如 OpenAI Chat
                        # Completions），因此必须通过模型提供方
                        # 翻译器，该翻译器从 tool_calls /
                        # tool_call_chunks 构建 ContentBlock 字典。若无此保护，字符串内容将被
                        # tool_call_chunks。没有这个保护，字符串内容将会
                        # 直接返回，静默丢弃工具调用。
            return cast("list[types.ContentBlock]", self.content)

        model_provider = self.response_metadata.get("model_provider")
        if model_provider:
            from langchain_core.messages.block_translators import (  # noqa: PLC0415
                get_translator,
            )

            translator = get_translator(model_provider)
            if translator:
                try:
                    return translator["translate_content_chunk"](self)
                except NotImplementedError:
                    pass

                # 否则，使用尽力而为的解析方式
        blocks = super().content_blocks

        if (
            self.tool_call_chunks
            and not self.content
            and self.chunk_position != "last"  # 如果已聚合，则保留 tool_calls
        ):
            blocks = [
                block
                for block in blocks
                if block["type"] not in {"tool_call", "invalid_tool_call"}
            ]
            for tool_call_chunk in self.tool_call_chunks:
                tc: types.ToolCallChunk = {
                    "type": "tool_call_chunk",
                    "id": tool_call_chunk.get("id"),
                    "name": tool_call_chunk.get("name"),
                    "args": tool_call_chunk.get("args"),
                }
                if (idx := tool_call_chunk.get("index")) is not None:
                    tc["index"] = idx
                blocks.append(tc)

                # 从 additional_kwargs 中尽力提取推理内容
                # 仅在尚未存在时添加推理内容
                # 插入到所有其他块之前，以保持推理位于开头
        has_reasoning = any(block.get("type") == "reasoning" for block in blocks)
        if not has_reasoning and (
            reasoning_block := _extract_reasoning_from_additional_kwargs(self)
        ):
            blocks.insert(0, reasoning_block)

        return blocks

    @model_validator(mode="after")
    def init_tool_calls(self) -> Self:
        """从工具调用块初始化工具调用。

        Returns:
            已初始化工具调用的值。

        Raises:
            ValueError: 如果工具调用块格式错误。
        """
        if not self.tool_call_chunks:
            if self.tool_calls:
                self.tool_call_chunks = [
                    create_tool_call_chunk(
                        name=tc["name"],
                        args=json.dumps(tc["args"]),
                        id=tc["id"],
                        index=None,
                    )
                    for tc in self.tool_calls
                ]
            if self.invalid_tool_calls:
                tool_call_chunks = self.tool_call_chunks
                tool_call_chunks.extend(
                    [
                        create_tool_call_chunk(
                            name=tc["name"], args=tc["args"], id=tc["id"], index=None
                        )
                        for tc in self.invalid_tool_calls
                    ]
                )
                self.tool_call_chunks = tool_call_chunks

            return self
        tool_calls = []
        invalid_tool_calls = []

        def add_chunk_to_invalid_tool_calls(chunk: ToolCallChunk) -> None:
            invalid_tool_calls.append(
                create_invalid_tool_call(
                    name=chunk["name"],
                    args=chunk["args"],
                    id=chunk["id"],
                    error=None,
                )
            )

        for chunk in self.tool_call_chunks:
            try:
                args_ = parse_partial_json(chunk["args"]) if chunk["args"] else {}
                if isinstance(args_, dict):
                    tool_calls.append(
                        create_tool_call(
                            name=chunk["name"] or "",
                            args=args_,
                            id=chunk["id"],
                        )
                    )
                else:
                    add_chunk_to_invalid_tool_calls(chunk)
            except Exception:
                add_chunk_to_invalid_tool_calls(chunk)
        self.tool_calls = tool_calls
        self.invalid_tool_calls = invalid_tool_calls

        if (
            self.chunk_position == "last"
            and self.tool_call_chunks
            and self.response_metadata.get("output_version") == "v1"
            and isinstance(self.content, list)
        ):
            id_to_tc: dict[str, types.ToolCall] = {
                cast("str", tc.get("id")): {
                    "type": "tool_call",
                    "name": tc["name"],
                    "args": tc["args"],
                    "id": tc.get("id"),
                }
                for tc in self.tool_calls
                if "id" in tc
            }
            for idx, block in enumerate(self.content):
                if (
                    isinstance(block, dict)
                    and block.get("type") == "tool_call_chunk"
                    and (call_id := block.get("id"))
                    and call_id in id_to_tc
                ):
                    self.content[idx] = cast("dict[str, Any]", id_to_tc[call_id])
                    if "extras" in block:
                                                # mypy 未考虑上述 dict 的实例检查
                        self.content[idx]["extras"] = block["extras"]  # type: ignore[index]

        return self

    @model_validator(mode="after")
    def init_server_tool_calls(self) -> Self:
        """初始化服务器工具调用。

        从 [`ServerToolCallChunk`][langchain.messages.ServerToolCallChunk] 对象解析
        `server_tool_call_chunks`。
        """
        if (
            self.chunk_position == "last"
            and self.response_metadata.get("output_version") == "v1"
            and isinstance(self.content, list)
        ):
            for idx, block in enumerate(self.content):
                if (
                    isinstance(block, dict)
                    and block.get("type")
                    in {"server_tool_call", "server_tool_call_chunk"}
                    and isinstance(args_str := block.get("args") or "{}", str)
                ):
                    try:
                        args = json.loads(args_str)
                        if isinstance(args, dict):
                            self.content[idx]["type"] = "server_tool_call"  # type: ignore[index]
                            self.content[idx]["args"] = args  # type: ignore[index]
                    except json.JSONDecodeError:
                        pass
        return self

    @overload  # type: ignore[override]  # summing BaseMessages gives ChatPromptTemplate
    def __add__(self, other: "AIMessageChunk") -> "AIMessageChunk": ...

    @overload
    def __add__(self, other: Sequence["AIMessageChunk"]) -> "AIMessageChunk": ...

    @overload
    def __add__(self, other: Any) -> BaseMessageChunk: ...

    @override
    def __add__(self, other: Any) -> BaseMessageChunk:
        if isinstance(other, AIMessageChunk):
            return add_ai_message_chunks(self, other)
        if isinstance(other, (list, tuple)) and all(
            isinstance(o, AIMessageChunk) for o in other
        ):
            return add_ai_message_chunks(self, *other)
        return super().__add__(other)


def add_ai_message_chunks(
    left: AIMessageChunk, *others: AIMessageChunk
) -> AIMessageChunk:
    """将多个 `AIMessageChunk` 相加。

    Args:
        left: 第一个 `AIMessageChunk`。
        *others: 其他要合并的 `AIMessageChunk`s。

    Returns:
        生成的 `AIMessageChunk`。

    """
    content = merge_content(left.content, *(o.content for o in others))
    additional_kwargs = merge_dicts(
        left.additional_kwargs, *(o.additional_kwargs for o in others)
    )
    response_metadata = merge_dicts(
        left.response_metadata, *(o.response_metadata for o in others)
    )

        # 合并工具调用片段
    if raw_tool_calls := merge_lists(
        left.tool_call_chunks, *(o.tool_call_chunks for o in others)
    ):
        tool_call_chunks = [
            create_tool_call_chunk(
                name=rtc.get("name"),
                args=rtc.get("args"),
                index=rtc.get("index"),
                id=rtc.get("id"),
            )
            for rtc in raw_tool_calls
        ]
    else:
        tool_call_chunks = []

        # token 使用量
    if left.usage_metadata or any(o.usage_metadata is not None for o in others):
        usage_metadata: UsageMetadata | None = left.usage_metadata
        for other in others:
            usage_metadata = add_usage(usage_metadata, other.usage_metadata)
    else:
        usage_metadata = None

        # 排名由偏好顺序定义。数值越高表示优先级越高：
        # 2. 提供商分配的 ID（非 lc_* 和非 lc_run-*）
        # 1. lc_run-* ID
        # 0. lc_* 及其他剩余 ID
    best_rank = -1
    chunk_id = None
    candidates = itertools.chain([left.id], (o.id for o in others))

    for id_ in candidates:
        if not id_:
            continue

        if not id_.startswith(LC_ID_PREFIX) and not id_.startswith(LC_AUTO_PREFIX):
            chunk_id = id_
                        # 最高排名，立即返回
            break

        rank = 1 if id_.startswith(LC_ID_PREFIX) else 0

        if rank > best_rank:
            best_rank = rank
            chunk_id = id_

    chunk_position: Literal["last"] | None = (
        "last" if any(x.chunk_position == "last" for x in [left, *others]) else None
    )

    return left.__class__(
        content=content,
        additional_kwargs=additional_kwargs,
        tool_call_chunks=tool_call_chunks,
        response_metadata=response_metadata,
        usage_metadata=usage_metadata,
        id=chunk_id,
        chunk_position=chunk_position,
    )


def add_usage(left: UsageMetadata | None, right: UsageMetadata | None) -> UsageMetadata:
    """递归地合并两个 UsageMetadata 对象。

    Example:
        ```python
        from langchain_core.messages.ai import add_usage

        left = UsageMetadata(
            input_tokens=5,
            output_tokens=0,
            total_tokens=5,
            input_token_details=InputTokenDetails(cache_read=3),
        )
        right = UsageMetadata(
            input_tokens=0,
            output_tokens=10,
            total_tokens=10,
            output_token_details=OutputTokenDetails(reasoning=4),
        )

        add_usage(left, right)
        ```

        结果为

        ```python
        UsageMetadata(
            input_tokens=5,
            output_tokens=10,
            total_tokens=15,
            input_token_details=InputTokenDetails(cache_read=3),
            output_token_details=OutputTokenDetails(reasoning=4),
        )
        ```
    Args:
        left: 第一个 `UsageMetadata` 对象。
        right: 第二个 `UsageMetadata` 对象。

    Returns:
        两个 `UsageMetadata` 对象的和。

    """
    if not (left or right):
        return UsageMetadata(input_tokens=0, output_tokens=0, total_tokens=0)
    if not (left and right):
        return cast("UsageMetadata", left or right)

    return UsageMetadata(
        **cast(
            "UsageMetadata",
            _dict_int_op(
                cast("dict[str, Any]", left),
                cast("dict[str, Any]", right),
                operator.add,
            ),
        )
    )


def subtract_usage(
    left: UsageMetadata | None, right: UsageMetadata | None
) -> UsageMetadata:
    """递归地从两个 `UsageMetadata` 对象中减去。

    token 数量不能为负数，因此实际操作是 `max(left - right, 0)`。

    Example:
        ```python
        from langchain_core.messages.ai import subtract_usage

        left = UsageMetadata(
            input_tokens=5,
            output_tokens=10,
            total_tokens=15,
            input_token_details=InputTokenDetails(cache_read=4),
        )
        right = UsageMetadata(
            input_tokens=3,
            output_tokens=8,
            total_tokens=11,
            output_token_details=OutputTokenDetails(reasoning=4),
        )

        subtract_usage(left, right)
        ```

        结果为

        ```python
        UsageMetadata(
            input_tokens=2,
            output_tokens=2,
            total_tokens=4,
            input_token_details=InputTokenDetails(cache_read=4),
            output_token_details=OutputTokenDetails(reasoning=0),
        )
        ```
    Args:
        left: 第一个 `UsageMetadata` 对象。
        right: 第二个 `UsageMetadata` 对象。

    Returns:
        减法后的结果 `UsageMetadata`。

    """
    if not (left or right):
        return UsageMetadata(input_tokens=0, output_tokens=0, total_tokens=0)
    if not (left and right):
        return cast("UsageMetadata", left or right)

    return UsageMetadata(
        **cast(
            "UsageMetadata",
            _dict_int_op(
                cast("dict[str, Any]", left),
                cast("dict[str, Any]", right),
                (lambda le, ri: max(le - ri, 0)),
            ),
        )
    )

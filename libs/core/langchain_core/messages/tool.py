"""工具使用的消息。"""

import json
from typing import Any, Literal, cast, overload
from uuid import UUID

from pydantic import Field, model_validator
from typing_extensions import NotRequired, TypedDict, override

from langchain_core.messages import content as types
from langchain_core.messages.base import BaseMessage, BaseMessageChunk, merge_content
from langchain_core.messages.content import InvalidToolCall
from langchain_core.utils._merge import merge_dicts, merge_obj


class ToolOutputMixin:
    """工具可直接返回对象的混合类。

    If a custom BaseTool is invoked with a `ToolCall` and the output of custom code is
    not an instance of `ToolOutputMixin`, the output will automatically be coerced to
    a string and wrapped in a `ToolMessage`.

    """


class ToolMessage(BaseMessage, ToolOutputMixin):
    """用于将工具执行结果传回模型的消息。

    `ToolMessage` 对象包含工具调用的结果。通常，结果编码在 `content` 字段中。

    `tool_call_id` is used to associate the tool call request with the tool call
    response. Useful in situations where a chat model is able to request multiple tool
    calls in parallel.

    Example:
        一个表示工具调用 ID 为 的结果 `42` 的 `ToolMessage`。

        ```python
        from langchain_core.messages import ToolMessage

        ToolMessage(content="42", tool_call_id="call_Jja7J89XsjrOLA5r!MEOW!SL")
        ```

    Example:
        一个 `ToolMessage`，其中只有部分工具输出发送给模型，而完整输出则传递给
        artifact。

        ```python
        from langchain_core.messages import ToolMessage

        tool_output = {
            "stdout": "From the graph we can see that the correlation between "
            "x and y is ...",
            "stderr": None,
            "artifacts": {"type": "image", "base64_data": "/9j/4gIcSU..."},
        }

        ToolMessage(
            content=tool_output["stdout"],
            artifact=tool_output,
            tool_call_id="call_Jja7J89XsjrOLA5r!MEOW!SL",
        )
        ```
    """

    tool_call_id: str
    """此消息所响应的工具调用。"""

    type: Literal["tool"] = "tool"
    """消息的类型（用于序列化）。"""

    artifact: Any = None
    """工具执行的 artifact，不打算发送给模型。

    Should only be specified if it is different from the message content, e.g. if only
    a subset of the full tool output is being passed as message content but the full
    output is needed in other parts of the code.
    """

    status: Literal["success", "error"] = "success"
    """工具调用的状态。"""

    additional_kwargs: dict[Any, Any] = Field(default_factory=dict, repr=False)
    """当前从 `BaseMessage` 继承，但未使用。"""

    response_metadata: dict[Any, Any] = Field(default_factory=dict, repr=False)
    """当前从 `BaseMessage` 继承，但未使用。"""

    @model_validator(mode="before")
    @classmethod
    def coerce_args(cls, values: dict[str, Any]) -> dict[str, Any]:
        """将模型参数强制转换为正确的类型。

        Args:
            values: 模型参数。

        """
        content = values["content"]
        if isinstance(content, tuple):
            content = list(content)

        if not isinstance(content, (str, list)):
            try:
                values["content"] = str(content)
            except ValueError as e:
                msg = (
                    "ToolMessage content should be a string or a list of string/dicts. "
                    f"Received:\n\n{content=}\n\n which could not be coerced into a "
                    "string."
                )
                raise ValueError(msg) from e
        elif isinstance(content, list):
            values["content"] = []
            for i, x in enumerate(content):
                if not isinstance(x, (str, dict)):
                    try:
                        values["content"].append(str(x))
                    except ValueError as e:
                        msg = (
                            "ToolMessage content should be a string or a list of "
                            "string/dicts. Received a list but "
                            f"element ToolMessage.content[{i}] is not a dict and could "
                            f"not be coerced to a string.:\n\n{x}"
                        )
                        raise ValueError(msg) from e
                else:
                    values["content"].append(x)

        tool_call_id = values["tool_call_id"]
        if isinstance(tool_call_id, (UUID, int, float)):
            values["tool_call_id"] = str(tool_call_id)
        return values

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
        """初始化一个 `ToolMessage`。

        指定 `content` 为位置参数，或使用 `content_blocks` 进行类型标注。

        Args:
            content: 消息的内容。
            content_blocks: 类型化的标准内容。
            **kwargs: 其他字段。
        """
        if content_blocks is not None:
            super().__init__(
                content=cast("list[str | dict[Any, Any]]", content_blocks),
                **kwargs,
            )
        else:
            super().__init__(content=content, **kwargs)


class ToolMessageChunk(ToolMessage, BaseMessageChunk):
    """工具消息块。"""

        # 忽略 mypy 重新赋值，因为我们正在覆盖该值
        # 以确保块变体可以与非块变体区分。
        # （注：此处为注释，不参与实际逻辑）
    type: Literal["ToolMessageChunk"] = "ToolMessageChunk"  # type: ignore[assignment]

    @override
    def __add__(self, other: Any) -> BaseMessageChunk:  # type: ignore[override]
        if isinstance(other, ToolMessageChunk):
            if self.tool_call_id != other.tool_call_id:
                msg = "Cannot concatenate ToolMessageChunks with different names."
                raise ValueError(msg)

            return self.__class__(
                tool_call_id=self.tool_call_id,
                content=merge_content(self.content, other.content),
                artifact=merge_obj(self.artifact, other.artifact),
                additional_kwargs=merge_dicts(
                    self.additional_kwargs, other.additional_kwargs
                ),
                response_metadata=merge_dicts(
                    self.response_metadata, other.response_metadata
                ),
                id=self.id,
                status=_merge_status(self.status, other.status),
            )

        return super().__add__(other)


class ToolCall(TypedDict):
    """表示 AI 请求调用工具。

    Example:
        ```python
        {"name": "foo", "args": {"a": 1}, "id": "123"}
        ```

        This represents a request to call the tool named `'foo'` with arguments
        `{"a": 1}` and an identifier of `'123'`.

    !!! note "工厂函数"

        `tool_call` 也可用作创建 `ToolCall` 的工厂。优势包括：

        * 必需参数在创建时严格验证
    """

    name: str
    """要调用的工具名称。"""

    args: dict[str, Any]
    """工具调用的参数，以字典形式。"""

    id: str | None
    """与工具调用相关联的标识符。

    An identifier is needed to associate a tool call request with a tool
    call result in events when multiple concurrent tool calls are made.
    """

    type: NotRequired[Literal["tool_call"]]
    """用于区分类型。"""


def tool_call(
    *,
    name: str,
    args: dict[str, Any],
    id: str | None,
) -> ToolCall:
    """创建一个工具调用。

    Args:
        name: 要调用的工具名称。
        args: 工具调用的参数，以字典形式。
        id: 与工具调用相关联的标识符。

    Returns:
        创建的工具调用。
    """
    return ToolCall(name=name, args=args, id=id, type="tool_call")


class ToolCallChunk(TypedDict):
    """工具调用块（流式传输时生成）。

    When merging `ToolCallChunk` objects (e.g., via `AIMessageChunk.__add__`), all
    string attributes are concatenated. Chunks are only merged if their values of
    `index` are equal and not `None`.

    Example:
    ```python
    left_chunks = [ToolCallChunk(name="foo", args='{"a":', index=0)]
    right_chunks = [ToolCallChunk(name=None, args="1}", index=0)]

    (
        AIMessageChunk(content="", tool_call_chunks=left_chunks)
        + AIMessageChunk(content="", tool_call_chunks=right_chunks)
    ).tool_call_chunks == [ToolCallChunk(name="foo", args='{"a":1}', index=0)]
    ```
    """

    name: str | None
    """要调用的工具名称。"""

    args: str | None
    """工具调用的参数，以 JSON 可解析的字符串形式。"""

    id: str | None
    """与工具调用相关联的标识符。

    An identifier is needed to associate a tool call request with a tool
    call result in events when multiple concurrent tool calls are made.
    """

    index: int | None
    """工具调用在序列中的索引。

    用于合并块。
    """

    type: NotRequired[Literal["tool_call_chunk"]]
    """用于区分类型。"""


def tool_call_chunk(
    *,
    name: str | None = None,
    args: str | None = None,
    id: str | None = None,
    index: int | None = None,
) -> ToolCallChunk:
    """创建一个工具调用块。

    Args:
        name: 要调用的工具名称。
        args: 作为 JSON 字符串的工具调用参数。
        id: 与工具调用相关联的标识符。
        index: 工具调用在序列中的索引。

    Returns:
        创建的工具调用块。
    """
    return ToolCallChunk(
        name=name, args=args, id=id, index=index, type="tool_call_chunk"
    )


def invalid_tool_call(
    *,
    name: str | None = None,
    args: str | None = None,
    id: str | None = None,
    error: str | None = None,
) -> InvalidToolCall:
    """创建一个无效的工具调用。

    Args:
        name: 要调用的工具名称。
        args: 作为 JSON 字符串的工具调用参数。
        id: 与工具调用相关联的标识符。
        error: 与工具调用相关联的错误消息。

    Returns:
        创建的无效工具调用。
    """
    return InvalidToolCall(
        name=name, args=args, id=id, error=error, type="invalid_tool_call"
    )


def default_tool_parser(
    raw_tool_calls: list[dict[str, Any]],
) -> tuple[list[ToolCall], list[InvalidToolCall]]:
    """尽力解析工具。

    Args:
        raw_tool_calls: 需要解析的原始工具调用字典列表。

    Returns:
        工具调用和无效工具调用的列表。
    """
    tool_calls = []
    invalid_tool_calls = []
    for raw_tool_call in raw_tool_calls:
        if "function" not in raw_tool_call:
            continue
        function_name = raw_tool_call["function"]["name"]
        try:
            function_args = json.loads(raw_tool_call["function"]["arguments"])
            parsed = tool_call(
                name=function_name or "",
                args=function_args or {},
                id=raw_tool_call.get("id"),
            )
            tool_calls.append(parsed)
        except json.JSONDecodeError:
            invalid_tool_calls.append(
                invalid_tool_call(
                    name=function_name,
                    args=raw_tool_call["function"]["arguments"],
                    id=raw_tool_call.get("id"),
                    error=None,
                )
            )
    return tool_calls, invalid_tool_calls


def default_tool_chunk_parser(
    raw_tool_calls: list[dict[str, Any]],
) -> list[ToolCallChunk]:
    """尽力解析工具块。

    Args:
        raw_tool_calls: 需要解析的原始工具调用字典列表。

    Returns:
        解析后的 ToolCallChunk 对象列表。
    """
    tool_call_chunks = []
    for tool_call in raw_tool_calls:
        if "function" not in tool_call:
            function_args = None
            function_name = None
        else:
            function_args = tool_call["function"]["arguments"]
            function_name = tool_call["function"]["name"]
        parsed = tool_call_chunk(
            name=function_name,
            args=function_args,
            id=tool_call.get("id"),
            index=tool_call.get("index"),
        )
        tool_call_chunks.append(parsed)
    return tool_call_chunks


def _merge_status(
    left: Literal["success", "error"], right: Literal["success", "error"]
) -> Literal["success", "error"]:
    return "error" if "error" in {left, right} else "success"

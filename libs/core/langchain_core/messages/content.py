"""大型语言模型 I/O 的标准多模态内容块。

This module provides standardized data structures for representing inputs to and outputs
from LLMs. The core abstraction is the **Content Block**, a `TypedDict`.

**设计理由**

Different LLM providers use distinct and incompatible API schemas. This module provides
a unified, provider-agnostic format to facilitate these interactions. A message to or
from a model is simply a list of content blocks, allowing for the natural interleaving
of text, images, and other content in a single ordered sequence.

An adapter for a specific provider is responsible for translating this standard list of
blocks into the format required by its API.

**可扩展性**

Data **not yet mapped** to a standard block may be represented using the
`NonStandardContentBlock`, which allows for provider-specific data to be included
without losing the benefits of type checking and validation.

Furthermore, provider-specific fields **within** a standard block are fully supported
by default in the `extras` field of each block. This allows for additional metadata
to be included without breaking the standard structure. For example, Google's thought
signature:

```python
AIMessage(
    content=[
        {
            "type": "text",
            "text": "J'adore la programmation.",
            "extras": {"signature": "EpoWCpc..."},  # Thought signature
        }
    ], ...
)
```


!!! note

    Following widespread adoption of [PEP 728](https://peps.python.org/pep-0728/), we
    intend to add `extra_items=Any` as a param to Content Blocks. This will signify to
    type checkers that additional provider-specific fields are allowed outside of the
    `extras` field, and that will become the new standard approach to adding
    provider-specific metadata.

    ??? note

        **示例：使用 PEP 728 的提供商特定字段**

        ```python
        # Content block definition
        # NOTE: `extra_items=Any`
        class TextContentBlock(TypedDict, extra_items=Any):
            type: Literal["text"]
            id: NotRequired[str]
            text: str
            annotations: NotRequired[list[Annotation]]
            index: NotRequired[int]
        ```

        ```python
        from langchain_core.messages.content import TextContentBlock

        # Create a text content block with provider-specific fields
        my_block: TextContentBlock = {
            # Add required fields
            "type": "text",
            "text": "Hello, world!",
            # Additional fields not specified in the TypedDict
            # These are valid with PEP 728 and are typed as Any
            "openai_metadata": {"model": "gpt-5.5", "temperature": 0.7},
            "anthropic_usage": {"input_tokens": 10, "output_tokens": 20},
            "custom_field": "any value",
        }

        # Mutating an existing block to add provider-specific fields
        openai_data = my_block["openai_metadata"]  # Type: Any
        ```

**使用示例**

```python
# Direct construction
from langchain_core.messages.content import TextContentBlock, ImageContentBlock

multimodal_message: AIMessage(
    content_blocks=[
        TextContentBlock(type="text", text="What is shown in this image?"),
        ImageContentBlock(
            type="image",
            url="https://www.langchain.com/images/brand/langchain_logo_text_w_white.png",
            mime_type="image/png",
        ),
    ]
)

# Using factories
from langchain_core.messages.content import create_text_block, create_image_block

multimodal_message: AIMessage(
    content=[
        create_text_block("What is shown in this image?"),
        create_image_block(
            url="https://www.langchain.com/images/brand/langchain_logo_text_w_white.png",
            mime_type="image/png",
        ),
    ]
)
```

工厂函数提供的优势包括：

- 未提供时自动生成 ID
- 无需手动指定 `type` 字段
"""

from typing import Any, Literal, get_args, get_type_hints

from typing_extensions import NotRequired, TypedDict

from langchain_core.utils.utils import ensure_id


class Citation(TypedDict):
    """用于引用文档数据的注解。

    !!! note

        `start`/`end` 索引指的是 **响应文本**，
        而非源文本。这意味着索引是相对于模型的响应，而不是原始文档（如 `url`
        中所指定）。

    !!! note "工厂函数"

        `create_citation` 也可用作创建 `Citation` 的工厂函数。
        优点包括：

        * 未提供时自动生成 ID
        * 必需参数在创建时严格验证
    """

    type: Literal["citation"]
    """内容块的类型。用于区分类型。"""

    id: NotRequired[str]
    """此内容块的唯一标识符。

    以下之一：

    - 由提供商生成
    - 由 LangChain 在创建时生成（以 `'lc_'` 前缀的 `UUID4`）
    """

    url: NotRequired[str]
    """文档源的 URL。"""

    title: NotRequired[str]
    """源文档标题。

    例如网页的页面标题或论文的标题。
    """

    start_index: NotRequired[int]
    """**响应文本**（`TextContentBlock.text`）的起始索引。"""

    end_index: NotRequired[int]
    """**响应文本**（`TextContentBlock.text`）的结束索引"""

    cited_text: NotRequired[str]
    """被引用的源文本摘录。"""

        # NOTE: 未包含原始文档文本的跨度（如 `text_start_index`
        # 和 `text_end_index`），因为目前任何提供商均不支持此功能。认为 `cited_text`
    # 对大多数用例已足够，且跨文件
        # 格式或编码方案可靠提取原始文档文本跨度存在困难。
        # TODO: 未来可考虑将其设为 NotRequired[str]。
        # TODO: 未来可考虑将字段设为 NotRequired[str]。

    extras: NotRequired[dict[str, Any]]
    """提供商特定的元数据。"""


class NonStandardAnnotation(TypedDict):
    """提供商特定的注解格式。"""

    type: Literal["non_standard_annotation"]
    """内容块的类型。用于区分类型。"""

    id: NotRequired[str]
    """此内容块的唯一标识符。

    以下之一：

    - 由提供方生成
    - 由 LangChain 在创建时生成（以 `UUID4` 为前缀，前面加上 `'lc_'`）
    """

    value: dict[str, Any]
    """提供方特定的注释数据。"""


Annotation = Citation | NonStandardAnnotation
"""A union of all defined `Annotation` types."""


class TextContentBlock(TypedDict):
    """大语言模型的文本输出。

    This typically represents the main text content of a message, such as the response
    from a language model or the text of a user message.

    !!! note "工厂函数"

        `create_text_block` 也可用作工厂函数来创建 `TextContentBlock`。优点包括：

        * 自动 ID 生成（当未提供时）
        * 创建时严格验证必需参数
    """

    type: Literal["text"]
    """内容块的类型。用于区分。"""

    id: NotRequired[str]
    """此内容块的唯一标识符。

    以下之一：

    - 由提供方生成
    - 由 LangChain 在创建时生成（以 `UUID4` 为前缀，前面加上 `'lc_'`）
    """

    text: str
    """块文本。"""

    annotations: NotRequired[list[Annotation]]
    """`Citation` 和其他注释。"""

    index: NotRequired[int | str]
    """在聚合响应中的块索引。用于流式传输。"""

    extras: NotRequired[dict[str, Any]]
    """提供方特定的元数据。"""


class ToolCall(TypedDict):
    """表示 AI 请求调用工具。

    Example:
        ```python
        {"name": "foo", "args": {"a": 1}, "id": "123"}
        ```

        This represents a request to call the tool named "foo" with arguments {"a": 1}
        and an identifier of "123".

    !!! note "工厂函数"

        `create_tool_call` 也可用作工厂函数来创建 `ToolCall`。优点包括：

        * 自动 ID 生成（当未提供时）
        * 创建时严格验证必需参数
    """

    type: Literal["tool_call"]
    """用于区分。"""

    id: str | None
    """与工具调用关联的标识符。

    An identifier is needed to associate a tool call request with a tool
    call result in events when multiple concurrent tool calls are made.
    """
        # TODO: 未来可考虑将其设为 NotRequired[str]。

    name: str
    """要调用的工具名称。"""

    args: dict[str, Any]
    """工具调用的参数。"""

    index: NotRequired[int | str]
    """在聚合响应中的块索引。用于流式传输。"""

    extras: NotRequired[dict[str, Any]]
    """提供方特定的元数据。"""


class ToolCallChunk(TypedDict):
    """工具调用的一个片段（在流式传输时产生）。

    When merging `ToolCallChunks` (e.g., via `AIMessageChunk.__add__`),
    all string attributes are concatenated. Chunks are only merged if their
    values of `index` are equal and not `None`.

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

        # TODO: 未来可考虑将字段设为 NotRequired[str]。

    type: Literal["tool_call_chunk"]
    """用于序列化。"""

    id: str | None
    """与工具调用关联的标识符。

    An identifier is needed to associate a tool call request with a tool
    call result in events when multiple concurrent tool calls are made.
    """
        # TODO: 未来可考虑将其设为 NotRequired[str]。

    name: str | None
    """要调用的工具名称。"""

    args: str | None
    """工具调用的参数。"""

    index: NotRequired[int | str]
    """工具调用在序列中的索引。"""

    extras: NotRequired[dict[str, Any]]
    """提供方特定的元数据。"""


class InvalidToolCall(TypedDict):
    """允许大语言模型产生的错误。

    在此处添加 `error` 键以暴露生成过程中发生的错误
    （例如，无效的 JSON 参数）。
    """

        # 注意：`title` 和 `context` 是可用于提供文件额外信息的字段，例如描述或内容摘要。

    type: Literal["invalid_tool_call"]
    """用于区分。"""

    id: str | None
    """与工具调用关联的标识符。

    An identifier is needed to associate a tool call request with a tool
    call result in events when multiple concurrent tool calls are made.
    """
        # 例如，使用 Claude 时，可以为文件提供上下文，该上下文将传递给模型。

    name: str | None
    """要调用的工具名称。"""

    args: str | None
    """工具调用的参数。"""

    error: str | None
    """与工具调用相关的错误消息。"""

    index: NotRequired[int | str]
    """在聚合响应中的块索引。用于流式传输。"""

    extras: NotRequired[dict[str, Any]]
    """提供方特定的元数据。"""


class ServerToolCall(TypedDict):
    """在服务端执行的工具调用。

    For example: 例如：代码执行、网络搜索等。
    """

    type: Literal["server_tool_call"]
    """用于区分。"""

    id: str
    """与工具调用关联的标识符。"""

    name: str
    """要调用的工具名称。"""

    args: dict[str, Any]
    """工具调用的参数。"""

    index: NotRequired[int | str]
    """聚合响应中块的索引。用于流式处理。"""

    extras: NotRequired[dict[str, Any]]
    """提供商特定的元数据。"""


class ServerToolCallChunk(TypedDict):
    """服务器端工具调用的一个数据块（流式处理时生成）。"""

    type: Literal["server_tool_call_chunk"]
    """用于区分类型。"""

    name: NotRequired[str]
    """要调用的工具名称。"""

    args: NotRequired[str]
    """工具调用参数的 JSON 子字符串。"""

    id: NotRequired[str]
    """此服务器工具调用块的唯一标识符。

    以下之一：

    - 由提供商生成
    - LangChain 在创建时生成（以 `'lc_'` 为前缀的 `UUID4`）
    """

    index: NotRequired[int | str]
    """聚合响应中块的索引。用于流式处理。"""

    extras: NotRequired[dict[str, Any]]
    """提供商特定的元数据。"""


class ServerToolResult(TypedDict):
    """服务器端工具调用的结果。"""

    type: Literal["server_tool_result"]
    """用于区分类型。"""

    id: NotRequired[str]
    """此服务器工具结果的唯一标识符。

    以下之一：

    - 由提供商生成
    - LangChain 在创建时生成（以 `'lc_'` 为前缀的 `UUID4`）
    """

    tool_call_id: str
    """对应服务器工具调用的 ID。"""

    status: Literal["success", "error"]
    """服务器端工具的执行状态。"""

    output: NotRequired[Any]
    """已执行工具的输出。"""

    index: NotRequired[int | str]
    """聚合响应中块的索引。用于流式处理。"""

    extras: NotRequired[dict[str, Any]]
    """提供商特定的元数据。"""


class ReasoningContentBlock(TypedDict):
    """来自 LLM 的推理输出。

    !!! note "工厂函数"

        `create_reasoning_block` 也可作为工厂函数来创建一个
        `ReasoningContentBlock`。优点包括：

        * 自动 ID 生成（当未提供时）
        * 创建时严格验证必需参数
    """

    type: Literal["reasoning"]
    """内容块的类型。用于区分类型。"""

    id: NotRequired[str]
    """此内容块的唯一标识符。

    以下之一：

    - 由提供商生成
    - LangChain 在创建时生成（以 `'lc_'` 为前缀的 `UUID4`）
    """

    reasoning: NotRequired[str]
    """推理文本。

    通常是思维摘要或原始推理文本本身。

    通常从模型响应中的 `<think>` 标签解析而来。
    """

    index: NotRequired[int | str]
    """聚合响应中块的索引。用于流式处理。"""

    extras: NotRequired[dict[str, Any]]
    """提供商特定的元数据。"""


# 未来可考虑的模态：
# - 3D 模型
# - 表格数据
class ImageContentBlock(TypedDict):
    """图像数据。

    !!! note "工厂函数"

        `create_image_block` 也可作为工厂函数来创建一个
        `ImageContentBlock`。优点包括：

        * 自动 ID 生成（当未提供时）
        * 创建时严格验证必需参数
    """

    type: Literal["image"]
    """内容块的类型。用于区分类型。"""

    id: NotRequired[str]
    """此内容块的唯一标识符。

    以下之一：

    - 由提供商生成
    - LangChain 在创建时生成（以 `'lc_'` 为前缀的 `UUID4`）
    """

    file_id: NotRequired[str]
    """对外部文件存储系统中图像的引用。

    例如，OpenAI 或 Anthropic 的 Files API。
    """

    mime_type: NotRequired[str]
    """图像的 MIME 类型。

    对 base64 数据是必需的。

    [IANA 示例](https://www.iana.org/assignments/media-types/media-types.xhtml#image)
    """

    index: NotRequired[int | str]
    """聚合响应中块的索引。用于流式处理。"""

    url: NotRequired[str]
    """图像的 URL。"""

    base64: NotRequired[str]
    """以 base64 字符串形式的数据。"""

    extras: NotRequired[dict[str, Any]]
    """提供商特定的元数据。此字段不应用于图像数据本身。"""


class VideoContentBlock(TypedDict):
    """视频数据。

    !!! note "工厂函数"

        `create_video_block` 也可作为工厂函数来创建一个
        `VideoContentBlock`。优点包括：

        * 自动 ID 生成（当未提供时）
        * 创建时严格验证必需参数
    """

    type: Literal["video"]
    """内容块的类型。用于区分类型。"""

    id: NotRequired[str]
    """此内容块的唯一标识符。

    以下之一：

    - 由提供方生成
    - 由 LangChain 在创建时生成（`UUID4` 前缀为 `'lc_'`）
    """

    file_id: NotRequired[str]
    """对外部文件存储系统中视频的引用。

    例如，OpenAI 或 Anthropic 的 Files API。
    """

    mime_type: NotRequired[str]
    """视频的 MIME 类型。

    base64 数据所必需。

    [IANA 示例](https://www.iana.org/assignments/media-types/media-types.xhtml#video)
    """

    index: NotRequired[int | str]
    """在聚合响应中的块索引。用于流式传输期间。"""

    url: NotRequired[str]
    """视频的 URL。"""

    base64: NotRequired[str]
    """以 base64 字符串形式的数据。"""

    extras: NotRequired[dict[str, Any]]
    """提供方特定的元数据。不应用于视频数据本身。"""


class AudioContentBlock(TypedDict):
    """音频数据。

    !!! note "工厂函数"

        `create_audio_block` 也可用作创建 `AudioContentBlock` 的工厂函数。优势包括：

        * 未提供时自动生成 ID
        * 创建时严格验证必填参数
    """

    type: Literal["audio"]
    """内容块的类型。用于区分。"""

    id: NotRequired[str]
    """此内容块的唯一标识符。

    以下之一：

    - 由提供方生成
    - 由 LangChain 在创建时生成（`UUID4` 前缀为 `'lc_'`）
    """

    file_id: NotRequired[str]
    """对外部文件存储系统中音频文件的引用。

    例如，OpenAI 或 Anthropic 的 Files API。
    """

    mime_type: NotRequired[str]
    """音频的 MIME 类型。

    base64 数据所必需。

    [IANA 示例](https://www.iana.org/assignments/media-types/media-types.xhtml#audio)
    """

    index: NotRequired[int | str]
    """在聚合响应中的块索引。用于流式传输期间。"""

    url: NotRequired[str]
    """音频的 URL。"""

    base64: NotRequired[str]
    """以 base64 字符串形式的数据。"""

    extras: NotRequired[dict[str, Any]]
    """提供方特定的元数据。不应用于音频数据本身。"""


class PlainTextContentBlock(TypedDict):
    """纯文本数据（例如来自 `.txt` 或 `.md` 文档）。

    !!! note

        A `PlainTextContentBlock` existed in `langchain-core<1.0.0`. Although the
        name has carried over, the structure has changed significantly. The only shared
        keys between the old and new versions are `type` and `text`, though the
        `type` value has changed from `'text'` to `'text-plain'`.

    !!! note

        标题和上下文是可选字段，可传递给模型。参见 Anthropic
        [示例](https://platform.claude.com/docs/en/build-with-claude/citations#citable-vs-non-citable-content)。

    !!! note "工厂函数"

        `create_plaintext_block` 也可用作创建 `PlainTextContentBlock`
        的工厂函数。优势包括：

        * 未提供时自动生成 ID
        * 创建时严格验证必填参数
    """

    type: Literal["text-plain"]
    """内容块的类型。用于区分。"""

    id: NotRequired[str]
    """此内容块的唯一标识符。

    以下之一：

    - 由提供方生成
    - 由 LangChain 在创建时生成（`UUID4` 前缀为 `'lc_'`）
    """

    file_id: NotRequired[str]
    """对外部文件存储系统中纯文本文件的引用。

    例如，OpenAI 或 Anthropic 的 Files API。
    """

    mime_type: Literal["text/plain"]
    """文件的 MIME 类型。

    base64 数据所必需。
    """

    index: NotRequired[int | str]
    """在聚合响应中的块索引。用于流式传输期间。"""

    url: NotRequired[str]
    """纯文本的 URL。"""

    base64: NotRequired[str]
    """以 base64 字符串形式的数据。"""

    text: NotRequired[str]
    """纯文本内容。若数据以 base64 提供，则此项可选。"""

    title: NotRequired[str]
    """文本数据的标题，例如文档的标题。"""

    context: NotRequired[str]
    """文本的上下文，例如对文本内容的描述或摘要。"""

    extras: NotRequired[dict[str, Any]]
    """提供方特定的元数据。这不应用于存储数据本身。"""


class FileContentBlock(TypedDict):
    """无法归入其他多模态块类型的文件数据。

    此块适用于非图像、音频或纯文本的文件。
    例如，可用于 PDF、Word 文档等。

    如果文件是图像、音频或纯文本，请使用相应的
    内容块类型（例如 `ImageContentBlock`、`AudioContentBlock`、
    `PlainTextContentBlock`）。

    !!! note "工厂函数"

        `create_file_block` 也可用作创建 `FileContentBlock` 的工厂函数。优点包括：

        * 自动生成 ID（当未提供时）
        * 创建时严格验证必需参数
    """

    type: Literal["file"]
    """内容块的类型。用于区分类型。"""

    id: NotRequired[str]
    """此内容块的唯一标识符。

    用于追踪和引用特定块（例如在流式传输期间）。

    Not to be confused with `file_id`, which references an external file in a
    storage system.

    以下之一：

    - 由提供方生成
    - 由 LangChain 在创建时生成（以 `'lc_'` 为前缀的 `UUID4`）
    """

    file_id: NotRequired[str]
    """对外部文件存储系统中文件的引用。

    例如，OpenAI Files API 中的文件 ID 或其他云存储提供商的文件 ID。
    这与 `id` 不同，`id` 用于标识内容块本身。
    """

    mime_type: NotRequired[str]
    """文件的 MIME 类型。

    对 base64 数据为必填项。

    [IANA 示例](https://www.iana.org/assignments/media-types/media-types.xhtml)
    """

    index: NotRequired[int | str]
    """块在聚合响应中的索引。用于流式传输期间。"""

    url: NotRequired[str]
    """文件的 URL。"""

    base64: NotRequired[str]
    """以 base64 字符串形式的数据。"""

    extras: NotRequired[dict[str, Any]]
    """提供方特定的元数据。这不应用于文件数据本身。"""


# --- 别名 ---
# 文本输出
# 工具


class NonStandardContentBlock(TypedDict):
    """提供方特定的内容数据。

    此块包含尚无标准类型的 data。

    此块的用途应仅为简单地持有提供方特定的负载。
    如果提供方的非标准输出包含推理和工具调用，应由适配器负责解析该负载，并发出对应的标准
    `ReasoningContentBlock` 和 `ToolCalls`。

    无 `extras` 字段，因为提供方特定数据应包含在 `value` 字段中。

    !!! note "工厂函数"

        `create_non_standard_block` 也可用作创建 `NonStandardContentBlock`
        的工厂函数。优点包括：

        * 自动生成 ID（当未提供时）
        * 创建时严格验证必需参数
    """

    type: Literal["non_standard"]
    """内容块的类型。用于区分类型。"""

    id: NotRequired[str]
    """此内容块的唯一标识符。

    以下之一：

    - 由提供方生成
    - 由 LangChain 在创建时生成（以 `'lc_'` 为前缀的 `UUID4`）
    """

    value: dict[str, Any]
    """提供方特定的内容数据。"""

    index: NotRequired[int | str]
    """块在聚合响应中的索引。用于流式传输期间。"""


# 多模态数据
DataContentBlock = (
    ImageContentBlock
    | VideoContentBlock
    | AudioContentBlock
    | PlainTextContentBlock
    | FileContentBlock
)
"""A union of all defined multimodal data `ContentBlock` types."""

ToolContentBlock = (
    ToolCall | ToolCallChunk | ServerToolCall | ServerToolCallChunk | ServerToolResult
)

ContentBlock = (
    TextContentBlock
    | InvalidToolCall
    | ReasoningContentBlock
    | NonStandardContentBlock
    | DataContentBlock
    | ToolContentBlock
)
"""A union of all defined `ContentBlock` types and aliases."""


KNOWN_BLOCK_TYPES = {
        # 服务端工具调用
    "text",
    "reasoning",
        # 通用捕获
    "tool_call",
    "invalid_tool_call",
    "tool_call_chunk",
        # 引用和 non_standard_annotation 故意省略
    "image",
    "audio",
    "file",
    "text-plain",
    "video",
        # 这是一个 Literal 类型，获取字面值
    "server_tool_call",
    "server_tool_call_chunk",
    "server_tool_result",
        # 类型有效且至少有一个数据字段存在
    "non_standard",
        # (接受旧式图像和音频 URLContentBlock)
}
"""These are block types known to `langchain-core >= 1.0.0`.

If a block has a type not in this set, it is considered to be provider-specific.
"""


def _get_data_content_block_types() -> tuple[str, ...]:
    """动态获取 DataContentBlock 联合类型成员的类型字面量。

    Example: ("image", "video", "audio", "text-plain", "file")

    注意，旧式多模态块类型字面量与新式块类型存在兼容性问题。
    特别是 "image"、"audio" 和 "file"。

    参见 `language_models._utils` 中 `_normalize_messages` 的 docstring 以获取详细信息。
    """
    data_block_types = []

    for block_type in get_args(DataContentBlock):
        hints = get_type_hints(block_type)
        if "type" in hints:
            type_annotation = hints["type"]
            if hasattr(type_annotation, "__args__"):
                                # 'text' 被检查以支持 v0 PlainTextContentBlock 类型
                literal_value = type_annotation.__args__[0]
                data_block_types.append(literal_value)

    return tuple(data_block_types)


def is_data_content_block(block: dict[str, Any]) -> bool:
    """检查提供的内容块是否为数据内容块。

    对 v0（旧式）和 v1（新式）多模态数据块均返回 True。

    Args:
        block: 要检查的内容块。

    Returns:
        如果内容块是数据内容块则返回 `True`，否则返回 `False`。
    """
    if block.get("type") not in _get_data_content_block_types():
        return False

    if any(key in block for key in ("url", "base64", "file_id", "text")):
                # 我们必须防范新式 TextContentBlock 也具有 'text' `type` 的情况
                # 通过确保 `source_type` 的存在

                # 旧式内容块可能的类型为 'image', 'audio', 和 'file'
                # 这些类型未被之前的检查所涵盖
                # 通过确保 `source_type` 的存在
        if block["type"] == "text" and "source_type" not in block:  # noqa: SIM103  # This is more readable
            return False

        return True

    if "source_type" in block:
                # 旧式内容块可能的类型有 'image'、'audio' 和 'file'
                # 这在之前的检查中未被捕捉
        source_type = block["source_type"]
        if (source_type == "url" and "url" in block) or (
            source_type == "base64" and "data" in block
        ):
            return True
        if (source_type == "id" and "id" in block) or (
            source_type == "text" and "url" in block
        ):
            return True

    return False


def create_text_block(
    text: str,
    *,
    id: str | None = None,
    annotations: list[Annotation] | None = None,
    index: int | str | None = None,
    **kwargs: Any,
) -> TextContentBlock:
    """创建一个 `TextContentBlock`。

    Args:
        text: 块的文本内容。
        id: 内容块标识符。

            若未提供则自动生成。
        annotations: 文本的 `Citation` 及其他注释。
        index: 在聚合响应中的块索引。

            流式传输期间使用。

    Returns:
        一个格式正确的 `TextContentBlock`。

    !!! note

        The `id` is generated automatically if not provided, using a UUID4 format
        prefixed with `'lc_'` to indicate it is a LangChain-generated ID.
    """
    block = TextContentBlock(
        type="text",
        text=text,
        id=ensure_id(id),
    )
    if annotations is not None:
        block["annotations"] = annotations
    if index is not None:
        block["index"] = index

    extras = {k: v for k, v in kwargs.items() if v is not None}
    if extras:
        block["extras"] = extras

    return block


def create_image_block(
    *,
    url: str | None = None,
    base64: str | None = None,
    file_id: str | None = None,
    mime_type: str | None = None,
    id: str | None = None,
    index: int | str | None = None,
    **kwargs: Any,
) -> ImageContentBlock:
    """创建一个 `ImageContentBlock`。

    Args:
        url: 图像的 URL。
        base64: Base64 编码的图像数据。
        file_id: 文件存储系统中图像文件的 ID。
        mime_type: 图像的 MIME 类型。

            对于 Base64 数据必须提供。
        id: 内容块标识符。

            若未提供则自动生成。
        index: 在聚合响应中的块索引。

            流式传输期间使用。

    Returns:
        一个格式正确的 `ImageContentBlock`。

    Raises:
        ValueError: If no image source is provided or if `base64` is used without
            `mime_type`.

    !!! note

        The `id` is generated automatically if not provided, using a UUID4 format
        prefixed with `'lc_'` to indicate it is a LangChain-generated ID.
    """
    if not any([url, base64, file_id]):
        msg = "Must provide one of: url, base64, or file_id"
        raise ValueError(msg)

    block = ImageContentBlock(type="image", id=ensure_id(id))

    if url is not None:
        block["url"] = url
    if base64 is not None:
        block["base64"] = base64
    if file_id is not None:
        block["file_id"] = file_id
    if mime_type is not None:
        block["mime_type"] = mime_type
    if index is not None:
        block["index"] = index

    extras = {k: v for k, v in kwargs.items() if v is not None}
    if extras:
        block["extras"] = extras

    return block


def create_video_block(
    *,
    url: str | None = None,
    base64: str | None = None,
    file_id: str | None = None,
    mime_type: str | None = None,
    id: str | None = None,
    index: int | str | None = None,
    **kwargs: Any,
) -> VideoContentBlock:
    """创建一个 `VideoContentBlock`。

    Args:
        url: 视频的 URL。
        base64: Base64 编码的视频数据。
        file_id: 文件存储系统中视频文件的 ID。
        mime_type: 视频的 MIME 类型。

            对于 Base64 数据必须提供。
        id: 内容块标识符。

            若未提供则自动生成。
        index: 在聚合响应中的块索引。

            流式传输期间使用。

    Returns:
        一个格式正确的 `VideoContentBlock`。

    Raises:
        ValueError: If no video source is provided or if `base64` is used without
            `mime_type`.

    !!! note

        The `id` is generated automatically if not provided, using a UUID4 format
        prefixed with `'lc_'` to indicate it is a LangChain-generated ID.
    """
    if not any([url, base64, file_id]):
        msg = "Must provide one of: url, base64, or file_id"
        raise ValueError(msg)

    if base64 and not mime_type:
        msg = "mime_type is required when using base64 data"
        raise ValueError(msg)

    block = VideoContentBlock(type="video", id=ensure_id(id))

    if url is not None:
        block["url"] = url
    if base64 is not None:
        block["base64"] = base64
    if file_id is not None:
        block["file_id"] = file_id
    if mime_type is not None:
        block["mime_type"] = mime_type
    if index is not None:
        block["index"] = index

    extras = {k: v for k, v in kwargs.items() if v is not None}
    if extras:
        block["extras"] = extras

    return block


def create_audio_block(
    *,
    url: str | None = None,
    base64: str | None = None,
    file_id: str | None = None,
    mime_type: str | None = None,
    id: str | None = None,
    index: int | str | None = None,
    **kwargs: Any,
) -> AudioContentBlock:
    """创建一个 `AudioContentBlock`。

    Args:
        url: 音频的 URL。
        base64: Base64 编码的音频数据。
        file_id: 文件存储系统中音频文件的 ID。
        mime_type: 音频的 MIME 类型。

            对于 Base64 数据必须提供。
        id: 内容块标识符。

            若未提供则自动生成。
        index: 在聚合响应中的块索引。

            流式传输期间使用。

    Returns:
        一个格式正确的 `AudioContentBlock`。

    Raises:
        ValueError: If no audio source is provided or if `base64` is used without
            `mime_type`.

    !!! note

        The `id` is generated automatically if not provided, using a UUID4 format
        prefixed with `'lc_'` to indicate it is a LangChain-generated ID.
    """
    if not any([url, base64, file_id]):
        msg = "Must provide one of: url, base64, or file_id"
        raise ValueError(msg)

    if base64 and not mime_type:
        msg = "mime_type is required when using base64 data"
        raise ValueError(msg)

    block = AudioContentBlock(type="audio", id=ensure_id(id))

    if url is not None:
        block["url"] = url
    if base64 is not None:
        block["base64"] = base64
    if file_id is not None:
        block["file_id"] = file_id
    if mime_type is not None:
        block["mime_type"] = mime_type
    if index is not None:
        block["index"] = index

    extras = {k: v for k, v in kwargs.items() if v is not None}
    if extras:
        block["extras"] = extras

    return block


def create_file_block(
    *,
    url: str | None = None,
    base64: str | None = None,
    file_id: str | None = None,
    mime_type: str | None = None,
    id: str | None = None,
    index: int | str | None = None,
    **kwargs: Any,
) -> FileContentBlock:
    """创建一个 `FileContentBlock`。

    Args:
        url: 文件的 URL。
        base64: Base64 编码的文件数据。
        file_id: 文件存储系统中的文件 ID。
        mime_type: 文件的 MIME 类型。

            Base64 数据必需。
        id: 内容块标识符。

            如果未提供，则自动生成。
        index: 在聚合响应中的块索引。

            流式传输期间使用。

    Returns:
        一个格式正确的 `FileContentBlock`。

    Raises:
        ValueError: If no file source is provided or if `base64` is used without
            `mime_type`.

    !!! note

        The `id` is generated automatically if not provided, using a UUID4 format
        prefixed with `'lc_'` to indicate it is a LangChain-generated ID.
    """
    if not any([url, base64, file_id]):
        msg = "Must provide one of: url, base64, or file_id"
        raise ValueError(msg)

    if base64 and not mime_type:
        msg = "mime_type is required when using base64 data"
        raise ValueError(msg)

    block = FileContentBlock(type="file", id=ensure_id(id))

    if url is not None:
        block["url"] = url
    if base64 is not None:
        block["base64"] = base64
    if file_id is not None:
        block["file_id"] = file_id
    if mime_type is not None:
        block["mime_type"] = mime_type
    if index is not None:
        block["index"] = index

    extras = {k: v for k, v in kwargs.items() if v is not None}
    if extras:
        block["extras"] = extras

    return block


def create_plaintext_block(
    text: str | None = None,
    url: str | None = None,
    base64: str | None = None,
    file_id: str | None = None,
    title: str | None = None,
    context: str | None = None,
    id: str | None = None,
    index: int | str | None = None,
    **kwargs: Any,
) -> PlainTextContentBlock:
    """创建一个 `PlainTextContentBlock`。

    Args:
        text: 纯文本内容。
        url: 纯文本文件的 URL。
        base64: Base64 编码的纯文本数据。
        file_id: 文件存储系统中的纯文本文件 ID。
        title: 文本数据的标题。
        context: 文本内容的上下文或描述。
        id: 内容块标识符。

            如果未提供，则自动生成。
        index: 在聚合响应中的块索引。

            流式传输期间使用。

    Returns:
        一个格式正确的 `PlainTextContentBlock`。

    !!! note

        The `id` is generated automatically if not provided, using a UUID4 format
        prefixed with `'lc_'` to indicate it is a LangChain-generated ID.
    """
    block = PlainTextContentBlock(
        type="text-plain",
        mime_type="text/plain",
        id=ensure_id(id),
    )

    if text is not None:
        block["text"] = text
    if url is not None:
        block["url"] = url
    if base64 is not None:
        block["base64"] = base64
    if file_id is not None:
        block["file_id"] = file_id
    if title is not None:
        block["title"] = title
    if context is not None:
        block["context"] = context
    if index is not None:
        block["index"] = index

    extras = {k: v for k, v in kwargs.items() if v is not None}
    if extras:
        block["extras"] = extras

    return block


def create_tool_call(
    name: str,
    args: dict[str, Any],
    *,
    id: str | None = None,
    index: int | str | None = None,
    **kwargs: Any,
) -> ToolCall:
    """创建一个 `ToolCall`。

    Args:
        name: 要调用的工具名称。
        args: 工具调用的参数。
        id: 工具调用的标识符。

            如果未提供，则自动生成。
        index: 在聚合响应中的块索引。

            流式传输期间使用。

    Returns:
        一个格式正确的 `ToolCall`。

    !!! note

        The `id` is generated automatically if not provided, using a UUID4 format
        prefixed with `'lc_'` to indicate it is a LangChain-generated ID.
    """
    block = ToolCall(
        type="tool_call",
        name=name,
        args=args,
        id=ensure_id(id),
    )

    if index is not None:
        block["index"] = index

    extras = {k: v for k, v in kwargs.items() if v is not None}
    if extras:
        block["extras"] = extras

    return block


def create_reasoning_block(
    reasoning: str | None = None,
    id: str | None = None,
    index: int | str | None = None,
    **kwargs: Any,
) -> ReasoningContentBlock:
    """创建一个 `ReasoningContentBlock`。

    Args:
        reasoning: 推理文本或思维摘要。
        id: 内容块标识符。

            如果未提供，则自动生成。
        index: 在聚合响应中的块索引。

            流式传输期间使用。

    Returns:
        一个格式正确的 `ReasoningContentBlock`。

    !!! note

        The `id` is generated automatically if not provided, using a UUID4 format
        prefixed with `'lc_'` to indicate it is a LangChain-generated ID.
    """
    block = ReasoningContentBlock(
        type="reasoning",
        reasoning=reasoning or "",
        id=ensure_id(id),
    )

    if index is not None:
        block["index"] = index

    extras = {k: v for k, v in kwargs.items() if v is not None}
    if extras:
        block["extras"] = extras

    return block


def create_citation(
    *,
    url: str | None = None,
    title: str | None = None,
    start_index: int | None = None,
    end_index: int | None = None,
    cited_text: str | None = None,
    id: str | None = None,
    **kwargs: Any,
) -> Citation:
    """创建一个 `Citation`。

    Args:
        url: 文档来源的 URL。
        title: 源文档标题。
        start_index: 引用在响应文本中开始的位置索引。
        end_index: 引用在响应文本中结束的位置索引。
        cited_text: 被引用的源文本摘录。
        id: 内容块标识符。

            如果未提供，则自动生成。

    Returns:
        一个格式正确的 `Citation`。

    !!! note

        The `id` is generated automatically if not provided, using a UUID4 format
        prefixed with `'lc_'` to indicate it is a LangChain-generated ID.
    """
    block = Citation(type="citation", id=ensure_id(id))

    if url is not None:
        block["url"] = url
    if title is not None:
        block["title"] = title
    if start_index is not None:
        block["start_index"] = start_index
    if end_index is not None:
        block["end_index"] = end_index
    if cited_text is not None:
        block["cited_text"] = cited_text

    extras = {k: v for k, v in kwargs.items() if v is not None}
    if extras:
        block["extras"] = extras

    return block


def create_non_standard_block(
    value: dict[str, Any],
    *,
    id: str | None = None,
    index: int | str | None = None,
) -> NonStandardContentBlock:
    """创建一个 `NonStandardContentBlock`。

    Args:
        value: 提供商特定的内容数据。
        id: 内容块标识符。

            如果未提供，则自动生成。
        index: 块在聚合响应中的索引。

            用于流式处理。

    Returns:
        一个格式正确的 `NonStandardContentBlock`。

    !!! note

        The `id` is generated automatically if not provided, using a UUID4 format
        prefixed with `'lc_'` to indicate it is a LangChain-generated ID.
    """
    block = NonStandardContentBlock(
        type="non_standard",
        value=value,
        id=ensure_id(id),
    )

    if index is not None:
        block["index"] = index

    return block

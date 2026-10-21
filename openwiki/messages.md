---
type: "Architecture"
title: "Message Types and Content Representation"
description: "Document the message abstraction, standardized content blocks for multimodal LLM I/O, message hierarchy, and provider-specific block translators."
tags: [messages, content-blocks, chat-models, streaming, multimodal, provider-adapters]
sources:
  - id: openwiki-source-77dc1fb726463969f9d53658
    resource: repo://libs/core/langchain_core/messages/ai.py
  - id: openwiki-source-b32b84365d17276620c41ebc
    resource: repo://libs/core/langchain_core/messages/base.py
  - id: openwiki-source-2e77747f30fe980d17d5d1a2
    resource: repo://libs/core/langchain_core/messages/block_translators/__init__.py
  - id: openwiki-source-b0f0ae0889f60e428f2f1b96
    resource: repo://libs/core/langchain_core/messages/block_translators/anthropic.py
  - id: openwiki-source-ad04883edeb0ba80d9ebcb7e
    resource: repo://libs/core/langchain_core/messages/block_translators/langchain_v0.py
  - id: openwiki-source-ac2e0f8b0fb1cb3b223672b7
    resource: repo://libs/core/langchain_core/messages/block_translators/openai.py
  - id: openwiki-source-fc874ddb29b9c5840565397f
    resource: repo://libs/core/langchain_core/messages/content.py
  - id: openwiki-source-8bb392f5dbc1fe7faaf52430
    resource: repo://libs/core/langchain_core/messages/human.py
  - id: openwiki-source-dad8cfeb38a829e03e165986
    resource: repo://libs/core/langchain_core/messages/system.py
  - id: openwiki-source-9861ba5cf0c42c142cf732f9
    resource: repo://libs/core/langchain_core/messages/tool.py
  - id: openwiki-source-498a9586e021b126ab8a8b42
    resource: repo://libs/core/langchain_core/messages/utils.py
generated: { by: "openwiki/0.5.0", at: "2026-09-08T08:27:09.597Z" }
verified:
  - by: openwiki/0.5.0
    at: 2026-09-28T08:35:20.640Z
---
## 概述

LangChain 的消息抽象提供了一个统一、与提供方无关的接口，用于表示大型语言模型的对话输入和输出。其核心是 **`BaseMessage`**，一个可序列化的容器，可容纳纯文本字符串或结构化的 **内容块列表**——代表文本、图像、音频、视频、工具调用、推理等内容的 TypedDict 对象。

关键创新在于 **内容块**：不再使用特定于提供方的模式（如 OpenAI 的 `image_url` 与 Anthropic 的 `document` 源块），LangChain 将所有内容标准化为统一格式。这使得应用程序能够以可移植的方式处理多模态消息，仅在调用时通过适配器（块转换器）将内容转换为提供方特定的格式。

## BaseMessage 层级与核心字段

**位置**：`repo://libs/core/langchain_core/messages/base.py#L93-L180`

`BaseMessage` 是所有消息类型的抽象基类。核心字段包括：

- **`content`**：`str | list[str | dict[Any, Any]]`  
  可以包含纯文本，或由字符串（视为文本块）和字典（内容块字典）组成的混合列表。

- **`type`**：`str`（架构中必需的字段）  
  唯一标识消息类型（`"human"`、`"ai"`、`"system"`、`"tool"`、`"chat"`、`"function"`，或其分块变体）。

- **`additional_kwargs`**：`dict[Any, Any]`  
  保留用于尚未映射到标准字段的提供方特定数据（例如 Ollama 或 DeepSeek 的 `reasoning_content`）。

- **`response_metadata`**：`dict[Any, Any]`  
  关于响应的元数据：请求头、token 数量、模型名称、提供方名称、输出版本。

- **`name`**（可选）：消息的人类可读标识符；大多数模型不使用。

- **`id`**（可选）：唯一标识符，通常由模型提供方分配。

### 核心消息类型

**`HumanMessage`** (`repo://libs/core/langchain_core/messages/human.py#L9-L61`)  
表示用户输入。用于提示词、问题以及来自用户的对话轮次。具有分块变体 `HumanMessageChunk`，支持流式传输。

**`AIMessage`** (`repo://libs/core/langchain_core/messages/ai.py#L160-L305`)  
表示模型输出。包含专用字段：
- **`tool_calls`**：`ToolCall` 字典列表（结构化工具调用请求）。
- **`invalid_tool_calls`**：解析失败的 `ToolCall` 字典（如 JSON 参数格式错误等）。
- **`usage_metadata`**：`UsageMetadata` 字典，包含标准化的 token 统计（`input_tokens`、`output_tokens`、`total_tokens`，以及可选的按类别细分统计）。

`AIMessageChunk` 变体用于流式传输，持有 `tool_call_chunks` 而非完整的 `tool_calls`。

**`SystemMessage`** (`repo://libs/core/langchain_core/messages/system.py#L9-L61`)  
用于引导模型行为；通常是对话中的第一条消息。也通过 `SystemMessageChunk` 支持分块。

**`ToolMessage`** (`repo://libs/core/langchain_core/messages/tool.py#L26-L164`)  
表示工具调用的结果。必需字段包括：
- **`tool_call_id`**：将此结果与发起请求的 `AIMessage.tool_calls[].id` 关联。
- **`content`**：工具的输出（字符串或内容块列表）。
- **`status`**：`"success"` 或 `"error"`。
- **`artifact`**（可选）：未发送给模型的完整工具输出（例如，当仅摘要在 `content` 中时的原始数据）。

**`ChatMessage`** 和 **`FunctionMessage`**  
遗留/专用的消息类型。`ChatMessage` 是通用类型，包含 `role` 字段；`FunctionMessage` 表示已弃用的函数调用格式。

**`RemoveMessage`** (`repo://libs/core/langchain_core/messages/modifier.py#L8-L33`)  
一种特殊消息类型，用于从对话历史中删除其他消息。仅接受一个 `id` 参数（要删除的消息 ID），不包含内容。在智能体工作流中用于修剪之前的交互回合。

### 消息块与流式处理

**位置**：`repo://libs/core/langchain_core/messages/base.py#L450-L500`（BaseMessageChunk）

在流式处理过程中，模型会逐个输出 `AIMessageChunk` 对象。这些块设计为**可合并**：通过 `+` 运算符组合时，它们会累积内容，按索引合并工具调用块，并聚合 token 使用量。

**`AIMessageChunk` 字段**：
- **`tool_call_chunks`**：部分工具调用对象，其中 `name` 和 `args` 可为空（JSON 字符串片段）。具有相同 `index` 的块会在聚合时被合并。
- **`chunk_position`**：最终块上为 `"last"`，用于触发完成信号（例如，将部分工具调用块解析为完整的 `tool_calls`）。

**通过 `add_ai_message_chunks()` 进行块聚合** (`repo://libs/core/langchain_core/messages/ai.py#L652-L700`)：

当使用 `+` 或该函数合并 `AIMessageChunk` 对象时：
1. **内容合并**：字符串内容进行拼接；列表内容则按顺序合并块。
2. **工具调用块合并**：根据 `index` 字段合并 `tool_call_chunks`，对共享相同索引的块拼接 `name` 和 `args` 字符串。
3. **token 使用量聚合**：对各块的 `usage_metadata`（输入 token、输出 token、各类别细分）进行求和。
4. **块位置处理**：当 `chunk_position="last"`（最终块）时，累积的 `tool_call_chunks` 通过 `parse_partial_json()` 解析为完整的 `ToolCall` 字典。
5. **服务端工具调用完成**：当 `chunk_position="last"` 且 `output_version="v1"` 时，`server_tool_call_chunk` 块会被解析为带有完整 `args` 对象的 `server_tool_call` 块。

合并示例序列：

```python
chunk1 = AIMessageChunk(tool_call_chunks=[ToolCallChunk(name="search", args='{"q":', index=0)])
chunk2 = AIMessageChunk(tool_call_chunks=[ToolCallChunk(name=None, args='"hello"}', index=0)])
final = chunk1 + chunk2
# final.tool_call_chunks[0].args == '{"q": "hello"}'
# After chunk_position="last", parses to: {"type": "tool_call", "name": "search", "args": {"q": "hello"}, ...}
```

## 内容块：统一的多模态表示

**位置**：`repo://libs/core/langchain_core/messages/content.py#L1-L878`

内容块是**TypedDict 对象**，代表消息的不同内容类型。它们提供了一种与提供方无关的抽象，由块转换器将其转换为具体提供方的格式。

### 标准块类型

**TextContentBlock**

```python
{
    "type": "text",
    "text": str,
    "id": str (optional, auto-generated),
    "annotations": list[Annotation] (optional, citations/metadata),
    "index": int | str (optional, for streaming),
    "extras": dict (optional, provider-specific fields),
}
```

模型输出的纯文本内容。注解支持指向源文档的引用。

**ReasoningContentBlock**

```python
{
    "type": "reasoning",
    "reasoning": str (optional),
    "id": str (optional),
    "index": int | str (optional),
    "extras": dict (optional),
}
```

来自如 o1、o3 等模型的链式思考或中间推理过程。通常从 `<think>` 标签或 `additional_kwargs` 中的特定提供方字段提取。

**ToolCall**

```python
{
    "type": "tool_call",
    "id": str | None,
    "name": str,
    "args": dict,
    "index": int | str (optional),
    "extras": dict (optional),
}
```

模型请求调用某个工具的指令。每个消息中的 ID 必须唯一，以便与 `ToolMessage` 响应匹配。

**ToolCallChunk**（流式变体）

```python
{
    "type": "tool_call_chunk",
    "id": str | None,
    "name": str | None,
    "args": str | None,
    "index": int | str (optional),
    "extras": dict (optional),
}
```

部分工具调用（在流式传输时发出）。`args` 字段会累积 JSON 内容。具有相同 `index` 的块会在到达时合并。

**InvalidToolCall**

```python
{
    "type": "invalid_tool_call",
    "id": str | None,
    "name": str | None,
    "args": str | None,
    "error": str | None,
    "index": int | str (optional),
    "extras": dict (optional),
}
```

解析失败的工具调用。`error` 字段捕获异常消息。

### 多模态数据块

**ImageContentBlock**

```python
{
    "type": "image",
    "url": str (optional),
    "base64": str (optional),
    "file_id": str (optional),
    "mime_type": str (optional, required for base64),
    "id": str (optional),
    "index": int | str (optional),
    "extras": dict (optional),
}
```

通过 URL、base64 编码或云文件引用（例如 OpenAI Files API）提供的图像数据。

**AudioContentBlock**、**VideoContentBlock**  
结构与 `ImageContentBlock` 类似，但 `type` 字段分别为 `"audio"` 和 `"video"`。

**FileContentBlock**

```python
{
    "type": "file",
    "url": str (optional),
    "base64": str (optional),
    "file_id": str (optional),
    "mime_type": str (optional),
    "id": str (optional),
    "index": int | str (optional),
    "extras": dict (optional),
}
```

不包含在图像、音频或纯文本类型中的通用文件数据（如 PDF、Word 文档等）。

**PlainTextContentBlock**

```python
{
    "type": "text-plain",
    "text": str (optional),
    "base64": str (optional),
    "url": str (optional),
    "file_id": str (optional),
    "mime_type": Literal["text/plain"],
    "title": str (optional),
    "context": str (optional),
    "id": str (optional),
    "index": int | str (optional),
    "extras": dict (optional),
}
```

带有可选标题和上下文的纯文本文档，用于模型理解。

### 服务端工具调用

**ServerToolCall**、**ServerToolCallChunk**、**ServerToolResult**  
支持在服务端执行的工具调用（例如代码执行、网络搜索）。模型通过这些类型发出请求，无需本地处理代码。

### NonStandardContentBlock

```python
{
    "type": "non_standard",
    "value": dict,
    "id": str (optional),
    "index": int | str (optional),
}
```

存储不映射到标准块类型的特定提供方内容。块转换器会在评估 `content_blocks` 属性时尝试解析非标准块。

### 内容块类型概览

LangChain 定义了以下标准块类型：

- **文本输出**：`TextContentBlock`（`type: "text"`），`ReasoningContentBlock`（`type: "reasoning"`，用于链式思维）
- **工具调用**：`ToolCall`（`type: "tool_call"`），`ToolCallChunk`（`type: "tool_call_chunk"`，用于流式传输），`InvalidToolCall`（`type: "invalid_tool_call"`，用于解析失败）
- **服务端工具**：`ServerToolCall`（`type: "server_tool_call"`），`ServerToolCallChunk`（`type: "server_tool_call_chunk"`），`ServerToolResult`（`type: "server_tool_result"`）
- **多模态数据**：`ImageContentBlock`（`type: "image"`），`AudioContentBlock`（`type: "audio"`），`VideoContentBlock`（`type: "video"`），`FileContentBlock`（`type: "file"`），`PlainTextContentBlock`（`type: "text-plain"`）
- **提供方特定**：`NonStandardContentBlock`（`type: "non_standard"`）

联合类型 `DataContentBlock` 表示所有多模态数据块，定义如下：

```python
DataContentBlock = (
    ImageContentBlock | VideoContentBlock | AudioContentBlock | 
    PlainTextContentBlock | FileContentBlock
)
```

所有块类型均支持一个可选的 `extras: dict[str, Any]` 字段，用于提供方特定的元数据。

### 访问内容块

**位置**：`repo://libs/core/langchain_core/messages/base.py#L199-L260`

`content_blocks` 属性将消息内容规范化为一组带类型的块字典列表：

```python
@property
def content_blocks(self) -> list[types.ContentBlock]:
```

**行为**：
1. 如果 `content` 是字符串，则将其包装为 `{"type": "text", "text": content}`。
2. 解析列表项：字符串转为文本块，具有已知 `type` 值的字典保持不变，其他项转为 `{"type": "non_standard", "value": ...}`。
3. 尝试通过一系列提供方特定的解析器解包非标准块（v0 块、Chat Completions 格式、Anthropic 格式、Google GenAI 格式、Bedrock 格式）。
4. 对于 `AIMessage`，检查 `response_metadata["model_provider"]`，若注册了对应提供方的翻译器（如 OpenAI、Anthropic），则使用该翻译器。
5. 若无翻译器存在，则采用最佳努力方式解析。
6. 对于 `AIMessage`，将尚未包含在 `content` 中的 `tool_calls` 附加为工具调用块。
7. 若存在 `additional_kwargs["reasoning_content"]`，则从中提取推理内容，并插入到块列表的开头。

## 块转换器：适配提供方格式

**位置**：`repo://libs/core/langchain_core/messages/block_translators/__init__.py` 以及各提供方模块

块转换器用于在 LangChain 的标准块格式与提供方特定格式之间进行转换。每个提供方模块会注册转换函数，当访问 `AIMessage.content_blocks` 时，若 `response_metadata["model_provider"]` 匹配，则会调用这些函数。

### 注册系统

**`register_translator`** 和 **`get_translator`**：

```python
def register_translator(
    provider: str,
    translate_content: Callable[[AIMessage], list[ContentBlock]],
    translate_content_chunk: Callable[[AIMessageChunk], list[ContentBlock]],
) -> None
```

转换器存储在 `PROVIDER_TRANSLATORS` 中，并通过 `_register_translators()` 在模块加载时自动初始化。

### 主要转换器

**OpenAI** (`repo://libs/core/langchain_core/messages/block_translators/openai.py`)  
处理聊天补全（Chat Completions）格式：
- 将 OpenAI 的 `image_url` 块转换为标准的 `ImageContentBlock`。
- 将 `tool_calls`（来自函数调用）解析为 `ToolCall` 块。
- 支持 Responses API 中的 `input_audio`、`input_file` 和 `input_image` 类型。
- `convert_to_openai_image_block()` 和 `convert_to_openai_data_block()` 是公开的实用工具，供模型和集成使用。

**Anthropic** (`repo://libs/core/langchain_core/messages/block_translators/anthropic.py`)  
处理 Anthropic 的格式：
- 将带有 `source` 字段（指定类型为 `base64`、`url`、`file` 或 `text`）的 `document` 块转换为标准的文件/纯文本块。
- 将具有多种源类型的 `image` 块转换为 `ImageContentBlock`。
- 将提供方特定字段（如 `cache_control`）填充至 `extras`。

**Google GenAI 与 Google VertexAI**  
将 Google 的格式块转换为标准类型。

**Bedrock（经典版）与 Bedrock Converse**  
将 AWS Bedrock 的格式块转换为标准类型。

**Groq**  
处理 Groq 的 API 响应格式。

**LangChain v0（向后兼容性）** (`repo://libs/core/langchain_core/messages/block_translators/langchain_v0.py`)  
解析基于旧版 `source_type` 的块（例如 `{"type": "image", "source_type": "url", "url": "..."}`），将其转换为 v1 块格式。这确保了使用 v0 块格式构造消息的旧代码仍能正常工作。

**自动注册**：所有转换器在 `block_translators` 模块加载时通过 `_register_translators()` 自动初始化，该函数在模块导入时被调用。外部集成可在运行时调用 `register_translator()` 来添加自定义提供方转换器。

### `content_blocks` 中的转换流程

当访问 `AIMessage.content_blocks` 时：
1. 检查 `response_metadata["output_version"]` 是否为 `"v1"`（已标准化，直接短路返回）。
2. 若 `response_metadata["model_provider"]` 已设置，尝试进行提供方特定的转换。
3. 回退到 `BaseMessage.content_blocks` 的最佳努力解析。
4. 对于 `AIMessage` 特别地，将任何尚未包含在内容中的工具调用追加，并从 kwargs 中提取推理信息。

## 消息操作实用工具

**位置**：`repo://libs/core/langchain_core/messages/utils.py`

该工具模块提供了操作消息的辅助函数：

**`get_buffer_string(messages, format="prefix")`** (`repo://libs/core/langchain_core/messages/utils.py#L287-L370`)  
将一系列消息转换为单个字符串，用于日志记录、提示生成或调试：
- **`format="prefix"`**（默认）：角色前缀格式，如 `"Human: ...\nAI: ..."`. 多模态内容块会被跳过；仅包含文本和 `text` 块。
- **`format="xml"`**：XML 格式输出，采用 `<message type="role">content</message>` 结构。支持对复杂多模态内容（图像、音频、视频、推理、工具调用）的安全渲染，并进行适当的字符转义。Base64 编码数据将被跳过。当消息内容可能包含类似角色前缀的内容，可能导致歧义时非常有用。

**`convert_to_messages` 和 `convert_to_openai_messages`**  
将各种输入格式（字典、字符串、`MessageLikeRepresentation` 联合类型）强制转换为有类型的消息对象。

**`filter_messages(messages, include_types=..., exclude_types=...)`**  
根据类型、名称或 ID 过滤消息序列。

**`trim_messages(messages, max_tokens=..., strategy="...")`**  
使用多种策略（保留开头、保留结尾、保留首尾等）将消息序列截断，以适应指定的 token 预算。

**`merge_message_runs(messages)`**  
合并并去重连续相同类型的 messages（例如多个 `AIMessage` 连续出现的情况）。

**`message_chunk_to_message(chunk: BaseMessageChunk) -> BaseMessage`**  
将一个消息块（或块列表）转换为完整的消息。

**`AnyMessage` 联合类型** (`repo://libs/core/langchain_core/messages/utils.py#L86-L100`)  

```python
AnyMessage = Annotated[
    Annotated[AIMessage, Tag(tag="ai")]
    | Annotated[HumanMessage, Tag(tag="human")]
    | Annotated[ChatMessage, Tag(tag="chat")]
    | Annotated[SystemMessage, Tag(tag="system")]
    | Annotated[FunctionMessage, Tag(tag="function")]
    | Annotated[ToolMessage, Tag(tag="tool")]
    | Annotated[AIMessageChunk, Tag(tag="AIMessageChunk")]
    | Annotated[HumanMessageChunk, Tag(tag="HumanMessageChunk")]
    | Annotated[ChatMessageChunk, Tag(tag="ChatMessageChunk")]
    | Annotated[SystemMessageChunk, Tag(tag="SystemMessageChunk")]
    | Annotated[FunctionMessageChunk, Tag(tag="FunctionMessageChunk")]
    | Annotated[ToolMessageChunk, Tag(tag="ToolMessageChunk")],
    Field(discriminator=Discriminator(_get_type)),
]
```

用于 Pydantic 反序列化的标签联合类型。包含所有消息类型及其块变体。`_get_type()` 判别函数从每个消息中提取 `type` 字段，在反序列化时路由到正确的类。

## 内容表示：字符串 vs. 块列表

消息支持两种内容形式：

**字符串内容**：

```python
AIMessage(content="Hello, world!")
```

简单且向后兼容。在内部被视为单个文本块。

**块列表内容**：

```python
AIMessage(
    content=[
        {"type": "text", "text": "What is this?"},
        {"type": "image", "url": "https://example.com/img.png"},
    ]
)
```

或使用类型化的 `content_blocks` 关键字参数：

```python
AIMessage(
    content_blocks=[
        create_text_block("What is this?"),
        create_image_block(url="https://example.com/img.png"),
    ]
)
```

### `text` 属性

**位置**: `repo://libs/core/langchain_core/messages/base.py#L262-L292`

`text` 属性返回一个 `TextAccessor`（字符串子类），用于从所有文本类型的内容块中提取拼接后的文本：

```python
msg = AIMessage(content=[
    {"type": "text", "text": "Hello"},
    {"type": "image", "url": "..."},
    {"type": "text", "text": " World"},
])
print(msg.text)  # "Hello World"
```

为保持向后兼容性，`TextAccessor` 同时支持属性和方法访问：
- **现代用法**（v1.0+）：`msg.text`（属性访问）
- **旧版用法**（pre-1.0）：`msg.text()`（方法调用，已弃用，会发出警告）

在提取文本时，非文本块（图像、音频、视频、工具调用、推理内容）会自动被跳过。

## 与聊天模型的集成

聊天模型通过消息抽象来规范化消息输入和输出：

- **输入**：用户提供的消息（字符串、字典或 `MessageLikeRepresentation`）。模型调用 `_normalize_messages()` 将其转换为 `BaseMessage` 对象，并可选地展开多模态内容以适配目标提供方。
- **输出**：模型返回 `AIMessage`，包含：
  - `content`：模型的文本响应（若为多模态则为块列表）。
  - `response_metadata`：填充了 `model_provider`、`output_version`、token 数量等信息。
  - `tool_calls`：从提供方格式解析为结构化的 `ToolCall` 字典。
  - `usage_metadata`：标准化的 token 统计数据。

有关模型调用和流式生命周期的详细信息，请参阅 `/openwiki/chat-models.md`。

## 提供方特定扩展

内容块中的 `extras` 字段允许添加提供方元数据，而不会破坏标准结构：

```python
{
    "type": "text",
    "text": "Response text",
    "extras": {
        "thought_signature": "EpoWCpc...",  # Google
        "cache_control": {"type": "ephemeral"},  # Anthropic
    },
}
```

该方法在保持类型安全的同时，支持新兴提供方功能。

## 消息版本化与向后兼容性

LangChain v1.0 引入了 v1 内容块格式，取代了 v0 的 `source_type` 风格。系统会透明地处理这两种格式：

### v0 块识别与转换

**LangChain v0 块**（例如 `{"type": "image", "source_type": "url", "url": "..."}`）通过是否存在 `source_type` 字段来识别。在 `content_blocks` 正常化过程中：
1. v0 块最初被包装为 `{"type": "non_standard", "value": ...}`。
2. `_convert_v0_multimodal_input_to_v1()` 解析器将其解包为 v1 格式。
3. 然后该块作为标准 v1 块进行处理。

### 提供商特定的块解包

原始提供商块（例如 OpenAI 的 `{"type": "image_url", "image_url": {"url": "..."}}`）在初始解析时会被包装为非标准格式。随后，通过 `PROVIDER_TRANSLATORS` 注册的提供商特定翻译器会在访问 `content_blocks` 属性时将其解包为标准类型。

### 输出版本追踪

`response_metadata["output_version"]` 字段用于指示内容归一化的状态：
- **`"v1"`**：内容已归一化为 v1 块（标准字典列表）。当满足以下条件时，`content_blocks` 属性直接返回内容而无需重新解析：
  - `output_version == "v1"` **且** `content` 是列表（而非字符串）
  - 此短路优化避免了对已符合 v1 格式的模型响应进行重复解析。
- **`None` 或缺失**：内容需要通过提供商翻译器和回退解析进行归一化。

对于 `AIMessageChunk`，此 v1 短路机制尤为重要：即使 `output_version="v1"`，如果 `content` 是字符串（例如仅文本的流式输出），内容仍会传递给提供商翻译器，以从 `tool_call_chunks` 构建 `ContentBlock` 字典。

## 示例工作流

### 发送多模态消息

```python
from langchain_core.messages import HumanMessage, create_text_block, create_image_block

message = HumanMessage(
    content_blocks=[
        create_text_block("Describe this chart."),
        create_image_block(url="https://example.com/chart.png", mime_type="image/png"),
    ]
)

# Access text
print(message.text)  # "Describe this chart."

# Get normalized blocks
for block in message.content_blocks:
    print(block["type"])  # "text", "image"
```

### 处理模型返回的工具调用

```python
ai_msg = model.invoke([...])
# ai_msg.tool_calls = [
#     {"type": "tool_call", "id": "call_1", "name": "search", "args": {"query": "..."}},
# ]

for tool_call in ai_msg.tool_calls:
    result = invoke_tool(tool_call["name"], tool_call["args"])
    tool_response = ToolMessage(
        content=str(result),
        tool_call_id=tool_call["id"],
    )
```

### 流式传输与分块聚合

```python
chunks = []
for chunk in model.stream(input_msg):
    chunks.append(chunk)
    print(f"Received: {chunk.content}")

# Aggregate all chunks
final = chunks[0]
for chunk in chunks[1:]:
    final = final + chunk

# final.tool_calls are now complete (parsed from tool_call_chunks)
```

### 使用块翻译器

当模型设置 `response_metadata["model_provider"]` 时，块翻译器会自动调用：

```python
# OpenAI model
ai_msg = openai_model.invoke(msg)
# response_metadata contains model_provider="openai"

blocks = ai_msg.content_blocks
# If content is from OpenAI's API, translator converts image_url → ImageContentBlock
```

自定义提供商集成可注册自己的翻译器：

```python
from langchain_core.messages.block_translators import register_translator

def my_translate_content(msg: AIMessage) -> list[ContentBlock]:
    # Custom logic
    pass

def my_translate_content_chunk(chunk: AIMessageChunk) -> list[ContentBlock]:
    # Custom logic
    pass

register_translator("my_provider", my_translate_content, my_translate_content_chunk)
```

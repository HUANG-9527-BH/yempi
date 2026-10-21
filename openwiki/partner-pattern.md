---
type: Integration Pattern
title: Adding a New Chat Model Provider
description: Step-by-step guide to integrate a new LLM provider into LangChain's monorepo, including package structure, ChatModel implementation, streaming, function calling, structured output, and standard tests. Covers message conversion, error handling, model profiles, and optional advanced API modes like Responses API.
tags: [chat-models, provider-integration, llm, function-calling, structured-output, streaming]
sources:
  - id: openwiki-source-c52037e7b642f7ac5a7642a8
    resource: repo://libs/core/langchain_core/language_models/chat_models.py
  - id: openwiki-source-b32b84365d17276620c41ebc
    resource: repo://libs/core/langchain_core/messages/base.py
  - id: openwiki-source-c479d4fffee5cf62576699e4
    resource: repo://libs/langchain_v1/langchain/chat_models/base.py
  - id: openwiki-source-e0b95eafb4bbd52f491c8cee
    resource: repo://libs/model-profiles/README.md
  - id: openwiki-source-7de1ace618efbdfd8bacb5cb
    resource: repo://libs/partners/anthropic/langchain_anthropic/chat_models.py
  - id: openwiki-source-f416dbc063b474398e38ff3c
    resource: repo://libs/partners/anthropic/langchain_anthropic/data/_profiles.py
  - id: openwiki-source-d14c2b8060843a8a89b74733
    resource: repo://libs/partners/anthropic/langchain_anthropic/data/profile_augmentations.toml
  - id: openwiki-source-04e3ac4f56ff2adb2b02de7d
    resource: repo://libs/partners/anthropic/pyproject.toml
  - id: openwiki-source-8641a971af4f11b852966d77
    resource: repo://libs/partners/openai/langchain_openai/chat_models/__init__.py
  - id: openwiki-source-3bc725a9a39d534be6f46d18
    resource: repo://libs/partners/openai/langchain_openai/chat_models/_compat.py
  - id: openwiki-source-738512768ef81ae009b097ac
    resource: repo://libs/partners/openai/langchain_openai/chat_models/base.py
  - id: openwiki-source-df762860acfcc6abf0ce804b
    resource: repo://libs/partners/openai/pyproject.toml
  - id: openwiki-source-e14c13faa5b908b1c95925d3
    resource: repo://libs/partners/openai/tests/integration_tests/chat_models/test_responses_api.py
  - id: openwiki-source-3953aa29dbaaf738e6efc09d
    resource: repo://libs/partners/openai/tests/unit_tests/chat_models/test_base_standard.py
  - id: openwiki-source-4065d8687d2708ee826c7818
    resource: repo://libs/partners/openai/tests/unit_tests/chat_models/test_responses_standard.py
  - id: openwiki-source-025cad4ae99967890152b7e0
    resource: repo://libs/standard-tests/README.md
generated: { by: "openwiki/0.5.0", at: "2026-09-22T08:27:06.345Z" }
verified:
  - by: openwiki/0.5.0
    at: 2026-10-16T08:25:55.096Z
---
## 概览

本指南记录了将新的聊天模型提供商（LLM 服务）集成到 LangChain 单体仓库中的模式。一个**提供商**代表一个 LLM 服务（如 OpenAI、Anthropic、Mistral），拥有自己的客户端库、模型列表和 API 约定。每个提供商的集成都位于 `/libs/partners/` 下的独立包中，并提供一个 `ChatModel` 子类，用于在 LangChain 的消息抽象与提供商的 API 之间建立桥梁。

集成过程包括：

1. **包结构**：创建 `/libs/partners/provider_name/` 目录，包含 Python 模块、测试文件和配置
2. **ChatModel 实现**：继承 `BaseChatModel` 并实现生成/流式输出方法
3. **消息转换**：在 LangChain 统一的消息格式与提供商特定的 API 模型之间进行转换
4. **提供商注册**：将提供商添加到内置的 `init_chat_model` 工厂注册表中
5. **模型配置文件**：发布能力数据（上下文窗口大小、工具调用、结构化输出等）
6. **标准测试**：继承单元测试和集成测试套件以验证实现正确性

## 1. 包结构

在 `/libs/partners/` 下创建一个以小写命名的目录，必要时使用连字符分隔：


```
/libs/partners/provider_name/
├── langchain_provider_name/              # Python package
│   ├── __init__.py                       # Exports: ChatProviderModel, version
│   ├── _version.py                       # Version constant
│   ├── chat_models/                      # Chat model implementation
│   │   ├── __init__.py
│   │   └── base.py                       # ChatProviderModel class
│   ├── data/                             # Model profiles and augmentations
│   │   ├── __init__.py
│   │   ├── _profiles.py                  # Auto-generated profiles from models.dev
│   │   └── profile_augmentations.toml    # Provider-specific overrides
│   ├── py.typed                          # PEP 561 marker for type checking
│   └── middleware/                       # (Optional) Custom middleware
├── tests/
│   ├── unit_tests/
│   │   ├── __init__.py
│   │   └── chat_models/
│   │       ├── test_standard.py          # Standard unit test suite
│   │       └── test_*.py                 # Provider-specific unit tests
│   └── integration_tests/
│       ├── __init__.py
│       └── chat_models/
│           ├── test_standard.py          # Standard integration test suite
│           └── test_*.py                 # Provider-specific integration tests
├── pyproject.toml                        # Package metadata and dependencies
├── Makefile                              # Common build/test targets
├── README.md                             # User-facing documentation
├── LICENSE                               # MIT license
└── uv.lock                               # Locked dependency versions

```


### 包元数据 (pyproject.toml)

提供商包的关键配置（参考：`/libs/partners/openai/pyproject.toml#L1-L76`）：


```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "langchain-provider-name"              # PyPI package name
description = "LangChain integration for Provider Name"
requires-python = ">=3.10.0,<4.0.0"
version = "0.1.0"

dependencies = [
    "langchain-core>=1.6.2,<2.0.0",           # Required: base LangChain
    "provider-client-library>=2.45.0,<4.0.0", # Provider's own SDK (pinned version)
    "certifi>=2024.6.2",                      # SSL certificates
]

[dependency-groups]
test = [
    "pytest>=9.0.3,<10.0.0",
    "pytest-asyncio>=1.3.0,<2.0.0",
    "langchain>=1.0.0,<2.0.0",
    "langchain-tests>=1.1.9,<2.0.0",          # Standard test suite with version constraint
]
lint = ["ruff>=0.13.1,<0.17.0"]
dev = []
test_integration = [
    # Additional integration test dependencies (API clients, test fixtures)
]
typing = [
    "mypy>=2.1.0,<2.2.0",
]

[tool.uv.sources]
langchain-core = { path = "../../core", editable = true }
langchain-tests = { path = "../../standard-tests", editable = true }
langchain = { path = "../../langchain_v1", editable = true }

[tool.uv]
constraint-dependencies = ["urllib3>=2.6.3", "pygments>=2.20.0"]
```


**关键规范：**
- **锁定提供商 SDK 版本**，防止因 API 变更导致的破坏性更新（例如 `openai>=2.45.0,<4.0.0`）
- **对所有测试依赖项添加版本约束**，确保可复现性
- **使用 `[tool.uv.sources]`** 引用单体仓库中的本地 LangChain 包
- **将集成测试依赖项单独归类**（仅在 CI 中需要）

## 2. ChatModel 实现

### BaseChatModel 与核心要求

所有提供商实现必须继承 **`BaseChatModel`**（`repo://libs/core/langchain_core/language_models/chat_models.py#L284-L2400`），该类定义了调用和流式处理聊天模型的契约。

**核心职责**（位置：`repo://libs/partners/anthropic/langchain_anthropic/chat_models.py#L1-L150`）：

1. **继承 `BaseChatModel`** 并指定类型参数为 `[AIMessage]`
2. **实现 `_generate` 方法**（必需同步）：将消息转换为 `ChatResult`，其中包含包装 `AIMessage` 输出的 `ChatGeneration` 对象
3. **实现 `_stream` 方法**（可选，用于支持流式输出）：逐个生成 `ChatGenerationChunk` 对象，其中包含带有增量 token 的 `AIMessageChunk`
4. **实现 `_agenerate` 方法**（`_generate` 的异步变体）或 `_astream` 方法（`_stream` 的异步变体）
5. **设置 `_llm_type` 属性**：返回用于识别的提供商标识符字符串

### 最小化 ChatModel 模板


```python
"""Provider chat model integration."""

from typing import Any, Iterator
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import LanguageModelInput
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult

class ChatProviderModel(BaseChatModel):
    """Chat model for Provider Name."""
    
    model: str                           # Model identifier (e.g., "model-123")
    api_key: str | None = None           # Provider API key
    temperature: float = 1.0             # Temperature parameter
    max_tokens: int | None = None        # Max output tokens
    
    @property
    def _llm_type(self) -> str:
        """Return provider identifier."""
        return "provider_name"
    
    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Generate chat completion synchronously.
        
        Args:
            messages: Conversation history and user input
            stop: Optional stop sequences
            run_manager: Callback manager for logging
            **kwargs: Additional provider-specific parameters
        
        Returns:
            ChatResult containing one or more ChatGeneration objects
        """
        # 1. Convert LangChain messages to provider format
        provider_messages = self._convert_messages_to_provider_format(messages)
        
        # 2. Build request payload
        payload = {
            "model": self.model,
            "messages": provider_messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stop": stop,
            **kwargs,
        }
        
        # 3. Call provider API
        response = self._client.chat.completions.create(**payload)
        
        # 4. Extract and convert response to AIMessage
        content = response.choices[0].message.content
        message = AIMessage(
            content=content,
            response_metadata={
                "model": response.model,
                "stop_reason": response.choices[0].finish_reason,
            },
        )
        
        # 5. Return ChatResult with generation info
        return ChatResult(
            generations=[ChatGeneration(message=message)],
            llm_output={"usage": response.usage.model_dump()} if response.usage else None,
        )
    
    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        """Stream chat completion tokens in real time.
        
        This method is called when `stream=True` or streaming callbacks are attached.
        
        Args:
            messages: Conversation history
            stop: Optional stop sequences
            run_manager: Callback manager for per-token callbacks
            **kwargs: Additional parameters
        
        Yields:
            ChatGenerationChunk objects containing AIMessageChunk with partial content
        """
        # 1. Build streaming request
        payload = {
            "model": self.model,
            "messages": self._convert_messages_to_provider_format(messages),
            "stream": True,
            "temperature": self.temperature,
            "stop": stop,
            **kwargs,
        }
        
        # 2. Stream from API
        accumulated_content = ""
        for event in self._client.chat.completions.create(**payload):
            # 3. Extract token delta
            delta = event.choices[0].delta
            if delta.content:
                accumulated_content += delta.content
                
                # 4. Yield chunk with incremental token
                chunk_message = AIMessageChunk(content=delta.content)
                chunk = ChatGenerationChunk(message=chunk_message)
                
                # 5. Notify run_manager of new token
                if run_manager:
                    run_manager.on_llm_new_token(delta.content, chunk=chunk)
                
                yield chunk
    
    def _convert_messages_to_provider_format(
        self, messages: list[BaseMessage]
    ) -> list[dict[str, Any]]:
        """Translate LangChain messages to provider API format.
        
        Provider APIs often use a different schema for messages (e.g., different
        role names, content representation). This method maps the unified LangChain
        format to the provider's specific requirements.
        """
        # Implementation: map LangChain message types to provider format
        # Handle HumanMessage, AIMessage, SystemMessage, ToolMessage
        pass
```


### 消息转换与内容块

LangChain 消息采用一种**统一且与提供商无关的内容格式**，使用**内容块**（参考：`/openwiki/messages.md`）。每个提供商都必须在输入和输出时，将其自身 API 模式与这一统一格式进行相互转换。

**关键消息字段：**

- **`content`**: `str | list[dict]` - 既可以是纯文本，也可以是结构化的内容块
- **`tool_calls`**: `list[ToolCall]` - 模型发出的结构化工具调用请求
- **`usage_metadata`**: Token 数量及类别细分
- **`response_metadata`**: 供应商特定的响应数据（模型名称、使用情况、结束原因等）

**统一的内容块类型：**
- `{"type": "text", "text": "..."}` - 纯文本
- `{"type": "text", "text": "...", "annotations": [...]}` - 带有引用/注释的文本（Responses API）
- `{"type": "image", "source": {...}}` - 图像（支持多种源格式：base64、URL 等）
- `{"type": "tool_use", "id": "...", "name": "...", "input": {...}}` - 工具调用
- `{"type": "tool_result", ...}` - 工具执行结果
- `{"type": "reasoning", ...}` - 推理内容（高级模型）
- `{"type": "audio", ...}` - 音频输入/输出（多模态）

**转换职责：**

1. **输入（消息 → 供应商格式）**：
   - 将 LangChain 消息类型映射为供应商角色（HumanMessage → "user"，AIMessage → "assistant" 等）
   - 将统一的内容块转换为供应商特定的格式
   - 处理工具调用和结构化输出模式
   - 保留特殊注释和附加信息（例如提示词缓存、推理参数）

2. **输出（供应商响应 → AIMessage）**：
   - 解析供应商特定的响应格式（JSON、流式分块、工具结果等）
   - 将供应商内容类型映射为统一块
   - 提取并标准化工具调用为 `ToolCall` 对象
   - 填充 `usage_metadata`（输入/输出 token 数量）
   - 将供应商特定的元数据存储在 `response_metadata` 中

**示例：消息转换（Anthropic 参考）：**

Anthropic 供应商同时处理输入和输出转换（参考：`repo://libs/partners/anthropic/langchain_anthropic/chat_models.py#L1-L150`）：

```python
def _convert_messages_to_provider_format(
    self, messages: list[BaseMessage]
) -> list[dict]:
    """Convert LangChain messages to Anthropic API format."""
    provider_messages = []
    
    for msg in messages:
        if isinstance(msg, HumanMessage):
            # Convert HumanMessage to Anthropic user role
            provider_messages.append({
                "role": "user",
                "content": self._format_content(msg.content),
            })
        elif isinstance(msg, AIMessage):
            # Convert AIMessage to Anthropic assistant role, including tool calls
            content = self._format_content(msg.content)
            if msg.tool_calls:
                # Append structured tool_use blocks
                content.extend([
                    {
                        "type": "tool_use",
                        "id": tc["id"],
                        "name": tc["name"],
                        "input": tc["args"],
                    }
                    for tc in msg.tool_calls
                ])
            provider_messages.append({
                "role": "assistant",
                "content": content,
            })
        elif isinstance(msg, SystemMessage):
            provider_messages.append({
                "role": "user",
                "content": msg.content,
            })
    
    return provider_messages
```

### 将供应商响应格式转换为统一消息

在从供应商 API 接收响应后，将其转换回 LangChain 的统一 `AIMessage` 格式。这是消息转换的反向过程。

**关键转换步骤：**

1. **解析供应商响应** - 从供应商特定格式中提取 choices/outputs
2. **创建统一内容块** - 将供应商内容类型映射为 LangChain 块
3. **处理工具调用** - 将供应商工具结果解析为 `ToolCall` 对象
4. **填充元数据** - 捕获使用信息和供应商特定的响应元数据
5. **创建 AIMessage** - 使用所有组件构造输出消息

**示例：OpenAI 响应转换**（参考：`repo://libs/partners/openai/langchain_openai/chat_models/base.py#L217-L282`）：

```python
from langchain_core.messages import AIMessage, UsageMetadata
from langchain_core.output_parsers.openai_tools import parse_tool_call, make_invalid_tool_call

def _convert_response_to_message(self, response: dict) -> AIMessage:
    """Convert provider API response to AIMessage."""
    # Parse the response choice (provider-specific format)
    choice = response["choices"][0]
    message_data = choice["message"]
    
    # Extract content, tool calls, and metadata
    content = message_data.get("content", "")
    
    # Convert tool calls from provider format to LangChain ToolCall
    tool_calls = []
    invalid_tool_calls = []
    if "tool_calls" in message_data:
        for raw_tc in message_data["tool_calls"]:
            try:
                # Parse provider tool call into ToolCall
                tool_calls.append(parse_tool_call(raw_tc, return_id=True))
            except Exception as e:
                # Store malformed tool calls for inspection
                invalid_tool_calls.append(
                    make_invalid_tool_call(raw_tc, str(e))
                )
    
    # Extract usage metadata (normalize across providers)
    usage_metadata = {}
    if "usage" in response:
        usage_metadata = {
            "input_tokens": response["usage"].get("prompt_tokens", 0),
            "output_tokens": response["usage"].get("completion_tokens", 0),
        }
    
    # Build response metadata with provider-specific fields
    response_metadata = {
        "model_name": response.get("model"),
        "finish_reason": choice.get("finish_reason"),
    }
    
    # Create unified AIMessage
    return AIMessage(
        content=content,
        tool_calls=tool_calls,
        invalid_tool_calls=invalid_tool_calls,
        usage_metadata=UsageMetadata(**usage_metadata) if usage_metadata else None,
        response_metadata=response_metadata,
    )
```

**处理供应商特定的响应字段：**

某些供应商会返回标准格式之外的额外字段。请将这些字段存储在 `response_metadata` 中：

- OpenAI: `model_name`，`service_tier`，`incomplete_details`（Responses API），`id`（响应 ID）
- Anthropic: `stop_reason`，`usage`，`content`（结构化块）
- 自定义字段：应用程序可能需要的任何供应商特定数据

## 3. 流式架构

### 流式实现模式

流式返回 `AIMessageChunk` 对象，随着模型生成 token 逐步递增输出。实现必须：

1. **在请求时启用流式**，通过在提供方 API 上设置流式标志
2. **遍历提供方事件**（例如，SSE 数据块、迭代器）
3. **从每个事件中提取增量/令牌内容**
4. **创建带有增量内容的 `AIMessageChunk`**
5. **用 `ChatGenerationChunk` 包装**以符合生成抽象层
6. **通过 `on_llm_new_token` 回调通知 run_manager**，实现可观测性

**核心模式**（Anthropic 参考：`repo://libs/partners/anthropic/langchain_anthropic/chat_models.py#L1862-L1910`）：

```python
def _stream(
    self,
    messages: list[BaseMessage],
    stop: list[str] | None = None,
    run_manager: CallbackManagerForLLMRun | None = None,
    **kwargs: Any,
) -> Iterator[ChatGenerationChunk]:
    # Enable streaming in API payload
    kwargs["stream"] = True
    payload = self._get_request_payload(messages, stop=stop, **kwargs)
    
    # Stream from API
    raw_response = self._client.create(payload)
    for event in raw_response.parse():
        # Convert Anthropic streaming event to AIMessageChunk
        msg_chunk = self._make_message_chunk_from_anthropic_event(event)
        
        if msg_chunk is not None:
            chunk = ChatGenerationChunk(message=msg_chunk)
            
            # Notify callbacks
            if run_manager and isinstance(msg_chunk.content, str):
                run_manager.on_llm_new_token(msg_chunk.content, chunk=chunk)
            
            yield chunk
```

### 异步流式

将 `_astream` 实现为 `_stream` 的异步版本，使用 `async for` 替代 `for`：

```python
async def _astream(
    self,
    messages: list[BaseMessage],
    stop: list[str] | None = None,
    run_manager: AsyncCallbackManagerForLLMRun | None = None,
    **kwargs: Any,
) -> AsyncIterator[ChatGenerationChunk]:
    kwargs["stream"] = True
    payload = self._get_request_payload(messages, stop=stop, **kwargs)
    
    raw_response = await self._acreate(payload)
    async for event in await _aparse(raw_response):
        msg_chunk = self._make_message_chunk_from_anthropic_event(event)
        if msg_chunk is not None:
            chunk = ChatGenerationChunk(message=msg_chunk)
            if run_manager and isinstance(msg_chunk.content, str):
                await run_manager.on_llm_new_token(msg_chunk.content, chunk=chunk)
            yield chunk
```

## 4. 提供方特定功能

### 函数调用 / 工具使用

实现 `bind_tools()`（继承自 `BaseChatModel`）以支持工具调用。工具在发送至 API 前会被转换为提供方的 schema 格式（OpenAI、Anthropic、JSON Schema 等）。模型响应中包含工具调用，这些调用会被提取并填充到 `AIMessage.tool_calls` 中。

**实现方式：**

1. **通过 `bind_tools()` 接受 `BaseTool` 对象、Pydantic 模型或字典**
2. **使用工具函数转换为提供方 schema**：
   - `convert_to_openai_tool()` - 用于兼容 OpenAI 的 API
   - `convert_to_json_schema()` - 用于 JSON Schema 格式
   - 提供方特定的转换器，用于自定义格式
3. **将工具包含在 API 请求载荷中**
4. **从响应中解析工具调用**为 `ToolCall` 对象
5. **处理无效/格式错误的工具调用**，将其存储在 `invalid_tool_calls` 中

### 结构化输出

实现 `with_structured_output()`，强制模型返回与 Pydantic 模型或 JSON Schema 匹配的响应。这通常映射到提供方的结构化输出或 JSON 模式功能。

**模式：**

1. 接收一个 Pydantic 模型或 JSON Schema
2. 转换为提供方的结构化输出格式
3. 包含在 API 请求中
4. 解析响应并根据 schema 验证
5. 返回解析后的模型实例或字典

### 视觉 / 多模态输入

通过内容块支持图像、视频和音频输入：

- **图像**：`{"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": "..."}}`
- **视频/音频**：使用适当的媒体类型，结构类似

将这些内容转换为提供方特定格式（例如，OpenAI 的 `image_url`，Anthropic 的 `source` 块）。

## 5. 在 init_chat_model 中注册提供方

将你的提供方添加到 **内置注册表** 中，以支持自动工厂实例化（`repo://libs/langchain_v1/langchain/chat_models/base.py#L56-L97`）：

```python
_BUILTIN_PROVIDERS: dict[str, tuple[str, str, Callable[..., BaseChatModel]]] = {
    # ... existing providers ...
    
    "provider_name": (
        "langchain_provider_name",           # Module path
        "ChatProviderModel",                 # Class name
        _call,                               # Instantiation function (_call is standard)
    ),
    
    # Special case: custom instantiation function (e.g., IBM Watson)
    # "ibm": ("langchain_ibm", "ChatWatsonx", lambda cls, model, **kwargs: cls(model_id=model, **kwargs)),
}
```

**注册后，用户可直接实例化你的模型：**

```python
from langchain.chat_models import init_chat_model

# With explicit provider prefix
model = init_chat_model("provider_name:model-id", temperature=0.5)

# With inferred provider (if model name starts with unique prefix)
model = init_chat_model("unique-prefix-model-id")
```

**提供方推断启发式规则** 定义在 `_attempt_infer_model_provider()` 中：

| 模型前缀 | 推断出的提供方 |
|---|---|
| `gpt-`, `o1`, `o3` | `openai` |
| `claude` | `anthropic` |
| `mistral`, `mixtral` | `mistralai` |

将你的提供者前缀添加到推理函数中，以启用裸模型名称注册。

## 6. 模型配置文件

**模型配置文件**通过 `model.profile` 属性暴露能力数据（如上下文窗口大小、支持的模态、工具调用等）。用户和集成系统会查询此属性以确定模型的能力。

### 配置文件结构与数据源

配置文件是存储在 `data/_profiles.py` 中的字典，由开源项目 [models.dev](https://github.com/sst/models.dev) 通过 `langchain-model-profiles` CLI 工具生成。

**示例配置文件**（参考：`repo://libs/partners/anthropic/langchain_anthropic/data/_profiles.py#L18-L52`）：

```python
_PROFILES: dict[str, dict[str, Any]] = {
    "claude-opus-4-7": {
        "name": "Claude Opus 4.7",
        "release_date": "2025-09-01",
        "max_input_tokens": 200000,
        "max_output_tokens": 4096,
        "text_inputs": True,
        "image_inputs": True,
        "audio_inputs": False,
        "pdf_inputs": True,
        "tool_calling": True,
        "structured_output": True,
        "tool_call_streaming": True,
        "reasoning_output": True,
    },
    # ... more models ...
}
```

### 更新配置文件

使用 `langchain-model-profiles` CLI 工具从 models.dev 刷新配置文件：

```bash
uv add langchain-model-profiles  # Install once globally or in dev dependencies

# Refresh profiles for your provider
langchain-model-profiles refresh \
    --provider provider_name \
    --data-dir ./langchain_provider_name/data
```

该命令会下载最新的模型数据，合并来自 `profile_augmentations.toml` 的提供者特定增强信息，并生成 `_profiles.py` 文件。

### 提供者增强项

为 LangChain 特定的能力覆盖创建 `data/profile_augmentations.toml`（参考：`repo://libs/partners/anthropic/langchain_anthropic/data/profile_augmentations.toml`）：

```toml
provider = "provider_name"

[overrides]
# Global overrides for all models
tool_call_streaming = true

[overrides."specific-model-id"]
# Model-specific overrides
structured_output = true
reasoning_effort_levels = ["low", "medium", "high"]
reasoning_effort_default = "high"
```

## 7. 标准测试

LangChain 通过 `langchain-tests` 包提供针对聊天模型的标准测试套件。提供者必须通过继承基础测试类来实现单元测试和集成测试。

### 单元测试

创建 `tests/unit_tests/chat_models/test_standard.py`（参考：`/libs/standard-tests/README.md`）：

```python
"""Standard LangChain interface tests for ChatProviderModel."""

from typing import Type

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_tests.unit_tests import ChatModelUnitTests

from langchain_provider_name import ChatProviderModel


class TestProviderModelStandard(ChatModelUnitTests):
    """Standard unit tests for ChatProviderModel."""
    
    @pytest.fixture
    def chat_model_class(self) -> Type[BaseChatModel]:
        return ChatProviderModel
    
    @pytest.fixture
    def chat_model_params(self) -> dict:
        """Parameters to instantiate the chat model.
        
        Must include all required constructor arguments (e.g., api_key if it's required).
        """
        return {
            "model": "model-123",
            "api_key": "test-key",  # Use environment variable in real tests
        }
```

**可配置的测试夹具**（来自 `langchain-tests` README）：

- `chat_model_class`（必需）：要测试的 `BaseChatModel` 子类
- `chat_model_params`：实例化时使用的关键字参数（默认为空字典）
- `chat_model_has_tool_calling`：模型是否支持 `bind_tools()`（自动检测）
- `chat_model_has_structured_output`：模型是否支持 `with_structured_output()`（自动检测）

### 集成测试

创建 `tests/integration_tests/chat_models/test_standard.py`：

```python
"""Standard integration tests for ChatProviderModel."""

from typing import Type

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_tests.integration_tests import ChatModelIntegrationTests

from langchain_provider_name import ChatProviderModel


class TestProviderModelIntegration(ChatModelIntegrationTests):
    """Standard integration tests for ChatProviderModel."""
    
    @pytest.fixture
    def chat_model_class(self) -> Type[BaseChatModel]:
        return ChatProviderModel
    
    @pytest.fixture
    def chat_model_params(self) -> dict:
        """Live API credentials (loaded from environment)."""
        return {
            "model": "model-123",
            # API key loaded from PROVIDER_NAME_API_KEY environment variable
        }
```

### 测试覆盖率

标准测试套件验证以下内容：

- **调用/流式方法**：同步与异步两种方式
- **消息处理**：所有消息类型和内容块
- **工具调用**：如果实现了 `bind_tools()`
- **结构化输出**：如果实现了 `with_structured_output()`
- **回调**：令牌计数、错误处理
- **模型配置文件**：存在性与有效性

### 高级 API 测试（可选）

如果你的提供者支持高级响应模式（例如 OpenAI 的 Responses API），请为这些模式创建额外的测试类：

**示例：Responses API 单元测试**（参考：`repo://libs/partners/openai/tests/unit_tests/chat_models/test_responses_standard.py`）：

```python
"""Standard tests with Responses API enabled."""

from langchain_core.language_models import BaseChatModel
from langchain_tests.unit_tests import ChatModelUnitTests

from langchain_provider_name import ChatProviderModel


class TestProviderModelResponsesAPI(ChatModelUnitTests):
    """Standard tests using the advanced Responses API mode."""
    
    @property
    def chat_model_class(self) -> type[BaseChatModel]:
        return ChatProviderModel
    
    @property
    def chat_model_params(self) -> dict:
        """Enable advanced API mode and any special configuration."""
        return {
            "model": "model-123",
            "use_responses_api": True,  # or provider-specific equivalent
            "stream_usage": True,  # Common for advanced modes
        }
```

高级模式的集成测试应验证：
- 响应格式兼容性（内容块、工具输出等）
- 保留结构化输出的流式传输
- 在高级格式中执行工具
- 不完整/截断的响应（如适用）

## 8. 高级 API 模式与响应格式

### 响应 API（提供方特定的高级功能）

部分提供方（例如 OpenAI）在标准聊天补全 API 之外，还提供了高级响应 API。这些 API 通常支持增强功能，如流式工具执行、推理输出以及结构化响应格式。

**示例：OpenAI 响应 API**（参考：`/openwiki/openai-provider.md`，`repo://libs/partners/openai/tests/integration_tests/chat_models/test_responses_api.py`）：

`ChatOpenAI` 提供方支持一个可选的 `use_responses_api` 参数，用于在聊天补全 API 与响应 API 之间切换：

```python
from langchain_openai import ChatOpenAI

# Standard Chat Completions API (default)
model = ChatOpenAI(model="gpt-4o-mini")

# Advanced Responses API (for streaming, reasoning models, etc.)
model = ChatOpenAI(
    model="gpt-4o",
    use_responses_api=True,
    reasoning={"effort": "medium"},  # Reasoning support
    include=["web_search_call.results"],  # Include tool outputs
)

# Invoke the model
response = model.invoke("What is the capital of France?")
# Returns AIMessage with structured content including reasoning, tool calls
```

**响应 API 的关键差异：**

1. **响应格式**：返回结构化内容块（文本、推理、工具调用、web_search_calls 等），而非纯文本
2. **消息内容**：`response.content` 是包含 `type`、`id` 和元数据的内容块列表
3. **流式行为**：流式输出会生成部分块；最终聚合时保留结构
4. **工具执行**：工具调用及结果作为内容块处理
5. **输出格式版本**：使用 `output_version="responses/v1"` 来控制解析

**响应 API 的消息转换：**

提供方实现必须将响应 API 输出（包含工具执行结果、推理内容及其他结构化项）转换回统一的 `AIMessage` 格式。OpenAI 提供方使用 `_convert_from_v1_to_responses()` 方法（参考：`repo://libs/partners/openai/langchain_openai/chat_models/_compat.py#L420-L514`）来翻译内容块：

```python
# Responses API returns:
{
    "type": "text",
    "text": "Paris is the capital of France",
    "annotations": [{"type": "web_search", "title": "..."}],
    "id": "msg_123"
}

# Converted to unified AIMessage content block:
{
    "type": "text",
    "text": "Paris is the capital of France",
    "annotations": [{"type": "web_search", "title": "..."}]
}
```

**建议**：响应 API 支持为可选且提供方特定功能。仅当您的提供方 SDK 支持时才实现。参考实现请见 OpenAI 的 `ChatOpenAI.use_responses_api` 及其在 `test_responses_standard.py` 中对应的测试用例。

## 9. 错误处理

将提供方特定异常映射到 LangChain 的统一异常层级（参考：`repo://libs/partners/openai/langchain_openai/chat_models/base.py#L54-L64`）：

| 提供方异常 | LangChain 异常 |
|---|---|
| `ProviderAPIError` | `ModelAPIError` |
| `ProviderAuthenticationError` | `ModelAuthenticationError` |
| `ProviderRateLimitError` | `ModelRateLimitError` |
| `ProviderTimeoutError` | `ModelTimeoutError` |
| `ProviderConnectionError` | `ModelConnectionError` |

**实现模式：**

```python
def _generate(self, messages, **kwargs):
    try:
        response = self._client.chat.create(...)
    except provider_sdk.AuthenticationError as e:
        raise ModelAuthenticationError(str(e)) from e
    except provider_sdk.RateLimitError as e:
        raise ModelRateLimitError(str(e)) from e
    except provider_sdk.APIError as e:
        raise ModelAPIError(str(e)) from e
    # ... rest of generation logic
```

## 10. 示例：OpenAI 提供方参考

OpenAI 提供方（`repo://libs/partners/openai/langchain_openai/chat_models/base.py`）是一个全面的参考实现，展示了以下功能：

- **消息转换**：支持图像、函数调用、推理内容
- **流式处理**：正确提取 delta 并计算 token 数量
- **工具调用**：转换为 OpenAI 格式，解析结构化响应
- **结构化输出**：JSON Schema 验证与解析
- **错误映射**：详细的提供方特定错误处理
- **异步支持**：所有方法均实现完整的 async/await 支持

## 11. 维护与更新

### 依赖项更新

在 `pyproject.toml` 中锁定提供方 SDK 版本，以防止引入破坏性变更。定期查阅提供方的发布说明，了解新模型和 API 变更。

### 模型配置文件更新

定期运行 CLI 工具，从 models.dev 获取新模型：

```bash
langchain-model-profiles refresh --provider provider_name --data-dir ./langchain_provider_name/data
```

### 测试

在发布更新前运行标准测试：

```bash
# Unit tests (no API credentials required)
pytest tests/unit_tests/

# Integration tests (requires provider API credentials)
pytest tests/integration_tests/
```

## 添加新提供方的检查清单

- [ ] 在 `/libs/partners/provider_name/` 中创建包结构
- [ ] 实现继承自 `BaseChatModel` 的 `ChatProviderModel`
- [ ] 实现 `_generate` 方法以支持同步生成
- [ ] 实现 `_stream` 方法以支持令牌流式输出
- [ ] 实现 `_agenerate` 或 `_astream` 以支持异步操作
- [ ] 将 LangChain 格式的消息转换为提供方 API 的数据结构
- [ ] 解析并转换提供方响应为 `AIMessage` / `AIMessageChunk`
- [ ] 实现 `bind_tools()` 以支持函数调用（如支持）
- [ ] 实现 `with_structured_output()` 以支持结构化输出（如支持）
- [ ] 将提供方异常映射到 LangChain 异常层级
- [ ] 通过 `langchain-model-profiles` CLI 获取并存储模型配置文件
- [ ] 将提供方添加至 `init_chat_model` 中的 `_BUILTIN_PROVIDERS` 注册表
- [ ] 创建继承自 `ChatModelUnitTests` 的单元测试套件
- [ ] 创建继承自 `ChatModelIntegrationTests` 的集成测试套件
- [ ] 在 docstrings 和 README 中记录公共 API
- [ ] （如适用）将提供方加入模型名称推断启发式规则
- [ ] 更新集成文档和变更日志

## 相关文档

- [聊天模型接口](/openwiki/chat-models.md) - 聊天模型核心协议
- [消息类型](/openwiki/messages.md) - 消息抽象与内容块
- [模型初始化](/openwiki/model-initialization.md) - `init_chat_model` 工厂详细信息
- [OpenAI 提供方](/openwiki/openai-provider.md) - 参考实现
- [LangChain 集成文档](https://docs.langchain.com/oss/python/integrations/providers/overview) - 用户导向指南

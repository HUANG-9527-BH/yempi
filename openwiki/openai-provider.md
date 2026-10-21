---
type: "ChatModel Integration"
title: "OpenAI Integration: ChatOpenAI and Azure Support"
description: "ChatOpenAI integration for OpenAI's Chat Completions and Responses APIs, with support for tool calling, structured output, vision, streaming, and Azure deployment."
tags: ["openai", "chat-models", "tool-calling", "structured-output", "vision", "azure"]
sources:
  - id: openwiki-source-1e66a9da38565f8901e651f4
    resource: repo://libs/partners/openai/langchain_openai/__init__.py
  - id: openwiki-source-f32b395707eda97cd743f4e5
    resource: repo://libs/partners/openai/langchain_openai/chat_models/azure.py
  - id: openwiki-source-738512768ef81ae009b097ac
    resource: repo://libs/partners/openai/langchain_openai/chat_models/base.py
  - id: openwiki-source-74e5bef080f1af7da12371cf
    resource: repo://libs/partners/openai/langchain_openai/data/_profiles.py
  - id: openwiki-source-b65307f1d05bd74532af2eee
    resource: repo://libs/partners/openai/langchain_openai/decisions/base.py
  - id: openwiki-source-122b01e9aef52272c0a77739
    resource: repo://libs/partners/openai/langchain_openai/decisions/types.py
generated: { by: "openwiki/0.5.0", at: "2026-10-16T08:25:55.096Z" }
verified:
  - by: openwiki/0.5.0
    at: 2026-10-16T08:25:55.096Z
---
## 概览

OpenAI 集成（`langchain-openai`）为 OpenAI 的 API 以及 OpenAI 兼容端点提供了生产就绪的聊天模型支持。`ChatOpenAI` 是封装 OpenAI 聊天补全与响应 API 的主要类，全面支持以下功能：

- **聊天补全 API**：用于标准生成和函数调用
- **响应 API**：支持流式传输、推理模型及增强功能
- **结构化输出**：通过工具调用（`json_schema`）、JSON 模式或函数调用实现
- **工具调用**：使用 `bind_tools()` 和 `tool_choice` 参数
- **视觉支持**：对 gpt-4-vision 和 gpt-4o 模型支持图像输入
- **流式 token**：通过回调集成实现按块超时的流式传输
- **模型配置文件**：包含能力元数据（输入/输出模态、最大 token 数、工具支持）

**核心原则**：`ChatOpenAI` 仅遵循 [官方 OpenAI API 规范](https://github.com/openai/openai-openapi)。第三方提供商添加的非标准响应字段（如 vLLM 的 `reasoning_content`、DeepSeek 的 `reasoning_details`）**不会被提取或保留**。如需使用特定提供商的功能，请使用对应的提供商专用 LangChain 包（例如 `ChatDeepSeek`、`ChatOpenRouter`）。

## 位置

**包路径**：`repo://libs/partners/openai/langchain_openai/`

**主类**：`repo://libs/partners/openai/langchain_openai/chat_models/base.py#L2904-L4266`

**导出项**：`repo://libs/partners/openai/langchain_openai/__init__.py`

相关类：
- `BaseChatOpenAI`：与 Azure OpenAI 共享的基础实现
- `AzureChatOpenAI`：Azure 特定子类，位于 `repo://libs/partners/openai/langchain_openai/chat_models/azure.py`
- `OpenAI`（已弃用）：仅支持补全的旧版模型，位于 `repo://libs/partners/openai/langchain_openai/llms/`

## ChatOpenAI 类

### 构造函数参数

**API 配置：**

- **`model`** (`str`, 默认值 `"gpt-3.5-turbo"`): OpenAI 模型标识符（例如 `"gpt-4o"`、`"gpt-4-turbo"`、`"gpt-3.5-turbo"`）。
- **`api_key`** (`str | Callable[[], str] | Callable[[], Awaitable[str]] | None`): 用于身份验证的 API 密钥。可为：
  - 字符串值
  - 返回字符串的同步可调用对象
  - 返回字符串的异步可调用对象
  - 若未提供，则从 `OPENAI_API_KEY` 环境变量中推断

  **示例**：用于动态密钥轮换的可调用对象

  ```python
  def get_api_key() -> str:
      return fetch_from_secrets_manager()
  
  model = ChatOpenAI(api_key=get_api_key)
  ```

- **`base_url`** (`str | None`): 自定义 API 基础 URL，适用于 OpenAI 兼容端点。解析顺序（以首次匹配为准）：
  1. 显式传入的 `base_url` 关键字参数
  2. 环境变量 `OPENAI_API_BASE`（LangChain 在初始化时读取）
  3. 环境变量 `OPENAI_BASE_URL`（由底层 OpenAI SDK 读取）

  设置后，默认禁用 `stream_usage`，因为许多非 OpenAI 端点不支持流式 token 使用统计。

- **`organization`** (`str | None`): OpenAI 组织 ID。若未提供，将从 `OPENAI_ORG_ID` 环境变量中推断。

**生成参数：**

- **`temperature`** (`float | None`): 采样温度（0–2，通常为 0–1）。控制随机性；值越高，结果越随机。
- **`max_tokens`** (`int | None`): 响应中生成的最大 token 数量。
- **`top_p`** (`float | None`): 核心采样概率。用于 token 选择的累积概率阈值。
- **`top_logprobs`** (`int | None`): 在每个位置返回具有 log 概率的最可能 token 数量（需配合 `logprobs=True` 使用）。
- **`logprobs`** (`bool | None`): 是否在响应中返回 token 的 log 概率。
- **`seed`** (`int | None`): 确定性生成种子（若模型支持）。
- **`presence_penalty`** (`float | None`): 对已提及 token 进行惩罚（−2 到 2）。
- **`frequency_penalty`** (`float | None`): 根据响应中 token 出现频率进行惩罚（−2 到 2）。
- **`logit_bias`** (`dict[int, int] | None`): 修改特定 token ID 出现的可能性。
- **`n`** (`int | None`): 为每个提示生成的补全数量。

**流式传输与延迟：**

- **`streaming`** (`bool`，默认值 `False`)：通过 `stream()` 和 `astream()` 启用流式输出。
- **`stream_usage`** (`bool | None`)：在流式数据块中包含 token 使用量元数据。
  - `None`（默认）：默认 OpenAI 接口启用，当设置 `base_url` 或提供自定义客户端时禁用
  - 设置为 `True`/`False` 可覆盖默认行为
- **`stream_chunk_timeout`** (`float | None`，默认值 `120.0`)：异步流式传输的每个数据块的墙钟超时时间（秒）。在解析出的数据块之间无内容时触发（不受 OpenAI keepalive SSE 注释影响）。设为 `None` 或 `0` 可禁用。可通过环境变量 `LANGCHAIN_OPENAI_STREAM_CHUNK_TIMEOUT_S` 覆盖。

**请求处理：**

- **`timeout`** (`float | tuple[float, float] | None`)：请求超时时间（秒），或 `(connect_timeout, read_timeout)` 元组形式。
- **`max_retries`** (`int | None`)：针对临时失败的最大重试次数。
- **`http_client`** (`httpx.Client | None`)：自定义同步 HTTP 客户端。用于异步使用时必须与 `http_async_client` 配对。
- **`http_async_client`** (`httpx.AsyncClient | None`)：自定义异步 HTTP 客户端。
- **`http_socket_options`** (`Sequence[tuple[int, int, int]] | None`)：应用于 httpx 传输层的 TCP 套接字选项 `(level, option, value)`。默认采用保守的 TCP 保活 + `TCP_USER_TIMEOUT` 配置（约 2 分钟挂起上限）。设为空元组 `()` 可禁用。可通过环境变量覆盖：`LANGCHAIN_OPENAI_TCP_KEEPALIVE`、`LANGCHAIN_OPENAI_TCP_KEEPIDLE`、`LANGCHAIN_OPENAI_TCP_KEEPINTVL`、`LANGCHAIN_OPENAI_TCP_KEEPCNT`、`LANGCHAIN_OPENAI_TCP_USER_TIMEOUT_MS`。

**高级功能：**

- **`reasoning_effort`** (`str | None`)：针对推理模型，限制推理努力程度。取值：`'minimal'`、`'low'`、`'medium'`、`'high'`。（仅限 Chat Completions API。）
- **`reasoning`** (`dict[str, Any] | None`)：推理模型的推理参数（仅 Responses API 有效）。结构为：`{"effort": None | "low" | "medium" | "high", "summary": "auto" | "concise" | "detailed"}`。
- **`verbosity`** (`str | None`)：推理模型的详细程度（Responses API）。取值：`'low'`、`'medium'`、`'high'`。
- **`service_tier`** (`str | None`)：请求的延迟层级。选项：`'auto'`、`'default'`、`'flex'`。适用于 OpenAI 规模层级服务用户。
- **`store`** (`bool | None`)：OpenAI 是否可存储响应数据。Responses API 默认为 `True`，Chat Completions API 默认为 `False`。
- **`include_response_headers`** (`bool`，默认值 `False`)：将响应头信息包含在消息的 `response_metadata` 中。可用于捕获提供方元数据（例如推理提供方返回的模型名称）。
- **`extra_body`** (`dict[str, Any] | None`)：为 OpenAI 兼容 API（如 vLLM、LM Studio 等）添加额外 JSON 属性。建议使用此字段替代 `model_kwargs` 来传递特定提供方参数。
- **`prompt_cache_options`** (`dict[str, Any] | None`)：OpenAI 提示词缓存配置。
- **`include`** (`list[str] | None`)：从 Responses API 生成结果中包含的额外字段。示例：`'file_search_call.results'`、`'message.input_image.image_url'`、`'reasoning.encrypted_content'`。
- **`truncation`** (`str | None`)：Responses API 的截断策略。`'auto'`（丢弃中间项）或 `'disabled'`（默认）。
- **`context_management`** (`list[dict[str, Any]] | None`)：[上下文压缩](https://developers.openai.com/api/docs/guides/compaction) 的配置。
- **`disabled_params`** (`dict[str, Any] | None`)：禁用模型的某些参数。结构为：`{"param": None | ['val1', 'val2']}`。用于防止不兼容参数（例如，对旧模型禁用 `{"parallel_tool_calls": None}`）。

**其他：**

- **`stop`** (`list[str] | str | None`)：默认停止序列。
- **`tiktoken_model_name`** (`str | None`)：用于 tiktoken 令牌计数的模型名称（如果与 `model` 不同）。
- **`model_kwargs`** (`dict[str, Any]`)：传递给 API 的额外参数（由 `extra_body` 覆盖提供方特定参数）。
- **`default_headers`** (`dict[str, str] | None`)：请求使用的自定义 HTTP 头。
- **`default_query`** (`dict[str, object] | None`)：自定义查询参数。

### 初始化示例

**基本用法（API 密钥来自环境变量）：**

```python
from langchain_openai import ChatOpenAI

model = ChatOpenAI(model="gpt-4o")
response = model.invoke("What is 2 + 2?")
```

**自定义 API 基地址（OpenAI 兼容端点）：**

```python
model = ChatOpenAI(
    model="gpt-4-turbo",
    base_url="https://api.custom-openai-provider.com/v1",
    api_key="your-custom-api-key"
)
```

**支持流式传输和超时设置：**

```python
model = ChatOpenAI(
    model="gpt-4o",
    streaming=True,
    timeout=30.0,
    stream_chunk_timeout=60.0
)

for chunk in model.stream("Hello, what is your name?"):
    print(chunk.content, end="", flush=True)
```

**动态 API 密钥：**

```python
async def get_api_key() -> str:
    return await fetch_from_secret_store()

model = ChatOpenAI(
    model="gpt-4o",
    api_key=get_api_key
)

# Use async methods (ainvoke, astream)
response = await model.ainvoke("Hi")
```

## BaseChatOpenAI 与初始化

`ChatOpenAI` 继承自 `BaseChatOpenAI`，后者是 `AzureChatOpenAI` 共享的基础类。在初始化时，`BaseChatOpenAI`：

1. **解析 API 认证信息**，从参数、环境变量或可调用对象中获取
2. **构建 HTTP 客户端**（同步与异步），支持可选的套接字选项以管理连接
3. **注册模型配置文件**，用于存储功能元数据
4. **验证参数**，例如 `stream_chunk_timeout`（负值将回退到默认值并发出警告）
5. **初始化 OpenAI 客户端实例**（`self.client`、`self.async_client`），使用 OpenAI SDK

**客户端初始化细节：**

- **同步客户端**（`self.client`）：由同步 `httpx.Client` 构建，或内部创建。用于同步 `invoke()` 和 `stream()` 方法。
- **异步客户端**（`self.async_client`）：由异步 `httpx.AsyncClient` 构建，或内部创建。用于异步 `ainvoke()` 和 `astream()` 方法。
- **根客户端**（`self.root_client`、`self.root_async_client`）：缓存的 OpenAI 客户端实例，用于实际 API 调用。

如果为 `api_key` 提供了 **异步可调用对象**，则同步客户端不可用，同步方法会抛出 `ValueError`。应改用异步方法：

```python
async def get_key() -> str:
    return await fetch_secret()

model = ChatOpenAI(model="gpt-4o", api_key=get_key)
# await model.ainvoke(...) works
# model.invoke(...) raises ValueError
```

## 模型配置文件与功能特性

模型配置文件是自动创建的元数据，用于描述模型的功能。它们存储在 `repo://libs/partners/openai/langchain_openai/data/_profiles.py` 中，并通过 `ModelProfileRegistry` 获取。

**配置文件字段：**
- **`text_inputs` / `text_outputs`**：文本支持。
- **`image_inputs`**：视觉支持（gpt-4o、gpt-4-vision、gpt-4-turbo 启用视觉功能）。
- **`audio_inputs` / `audio_outputs`**：音频支持（gpt-4o、未来模型）。
- **`video_inputs`**：视频支持（待发布）。
- **`tool_calling`**：模型是否支持函数/工具调用。
- **`structured_output`**：模型是否支持 JSON Schema 结构化输出。
- **`max_input_tokens` / `max_output_tokens`**：上下文窗口与生成长度限制。
- **`tool_call_streaming`**：工具调用是否支持增量流式输出。
- **`tool_choice`**：是否支持 `tool_choice` 参数。

**访问配置文件：**

```python
from langchain_openai import ChatOpenAI
from langchain_core.language_models import ModelProfileRegistry

model = ChatOpenAI(model="gpt-4o")
# Profiles are used internally by LangChain for capability checks
```

## 响应 API

`ChatOpenAI` 会根据模型、参数和配置自动在聊天补全 API 与响应 API 之间切换。**响应 API** 提供了增强功能，包括：

- **流式推理** 用于推理模型（例如 o1-preview）
- **结合工具的结构化输出**
- **上下文管理**（消息压缩）通过 `context_management` 参数实现
- **截断策略** 控制通过 `truncation` 参数
- **推理参数**（努力程度、摘要）通过 `reasoning` 字典配置
- **前次响应追踪** 通过 `use_previous_response_id` 参数

**自动 API 选择**：当满足以下任一条件时，将自动使用 Responses API：
- 模型名称以 `gpt-5`（专业版）开头或包含 `codex`
- 显式设置 `use_responses_api=True`
- 提供了 `reasoning` 或 `context_management` 参数
- 设置了 `truncation` 或 `include` 参数
- 设置了 `use_previous_response_id=True`
- 模型名称以 `gpt-6` 开头且提供了工具

**显式控制：**

```python
# Force Responses API
model = ChatOpenAI(model="gpt-4o", use_responses_api=True)

# Force Chat Completions API
model = ChatOpenAI(model="gpt-4o", use_responses_api=False)

# Auto-detect (default)
model = ChatOpenAI(model="gpt-4o", use_responses_api=None)
```

**带有推理功能的 Responses API：**

```python
model = ChatOpenAI(
    model="o1-preview",
    use_responses_api=True,
    reasoning={
        "effort": "high",
        "summary": "detailed"
    }
)

response = model.invoke("Analyze this complex system design")
# Response includes reasoning content and analysis
```

**上下文管理（仅 Responses API 支持）：**

```python
model = ChatOpenAI(
    model="gpt-4o",
    use_responses_api=True,
    context_management=[
        {"type": "auto", "min_tokens": 1000}
    ]
)
# Model will automatically drop older messages to fit context window
```

## 视觉支持

视觉功能支持 `gpt-4-vision`、`gpt-4o` 和 `gpt-4-turbo` 等模型。图像可通过以下方式提供：

1. **基于 URL 的方式（`image_url`）：**

   ```python
   from langchain_core.messages import HumanMessage
   
   message = HumanMessage(
       content=[
           {"type": "text", "text": "What's in this image?"},
           {
               "type": "image_url",
               "image_url": {
                   "url": "https://example.com/image.jpg",
                   "detail": "low"  # or "high", "auto"
               }
           }
       ]
   )
   
   model = ChatOpenAI(model="gpt-4o")
   response = model.invoke(message)
   ```

2. **Base64 编码格式：**

   ```python
   import base64
   
   with open("image.jpg", "rb") as f:
       image_data = base64.b64encode(f.read()).decode("utf-8")
   
   message = HumanMessage(
       content=[
           {"type": "text", "text": "Describe this image"},
           {
               "type": "image_url",
               "image_url": {
                   "url": f"data:image/jpeg;base64,{image_data}",
                   "detail": "auto"
               }
           }
       ]
   )
   ```

图像的 token 计算为近似值：低细节 = 85 tokens，高细节 = ~170 + 每图像块 255 tokens（根据分辨率而定）。

## 函数调用

OpenAI 的 [函数调用 API](https://platform.openai.com/docs/guides/function-calling)（现称为 API 中的“工具”）允许模型调用您定义的函数。

### `bind_tools()` 方法

将一个或多个工具绑定到模型上：

```python
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

@tool
def get_weather(location: str) -> str:
    """Get weather for a location."""
    return f"Sunny in {location}"

model = ChatOpenAI(model="gpt-4o")
bound_model = model.bind_tools([get_weather])

response = bound_model.invoke("What's the weather in Boston?")
print(response.tool_calls)
# [ToolCall(id='call_123', name='get_weather', args={'location': 'Boston'}, type='tool_call')]
```

**`bind_tools()` 签名：**

```python
def bind_tools(
    self,
    tools: Sequence[dict | type | Callable | BaseTool],
    *,
    tool_choice: dict | str | bool | None = None,
    strict: bool | None = None,
    parallel_tool_calls: bool | None = None,
    response_format: dict | type | None = None,
    **kwargs: Any,
) -> Runnable[LanguageModelInput, AIMessage]
```

**参数：**

- **`tools`**：工具列表。支持：
  - `@tool` 装饰器生成的 `BaseTool` 实例
  - Pydantic `BaseModel` 类
  - 带有类型提示的可调用对象
  - 字典（OpenAI 工具模式 schema）

- **`tool_choice`** (`dict | str | bool | None`)：强制选择哪个工具：
  - `str`（工具名称）：强制使用特定工具（例如 `"get_weather"`）
  - `'auto'`：自动选择工具或不调用（默认）
  - `'none'`：禁止工具调用
  - `'any'` / `'required'` / `True`：强制至少调用一个工具
  - `dict`：OpenAI 工具选择字典 `{"type": "function", "function": {"name": "tool_name"}}`
  - `False` / `None`：无影响，保持默认行为
  - `WellKnownTools` 字符串（如 `'file_search'`、`'web_search'`、`'tool_search'` 等）：内置工具

  **示例：**

  ```python
  # Force specific tool
  bound = model.bind_tools([get_weather, get_time], tool_choice="get_weather")
  
  # Force any tool
  bound = model.bind_tools([get_weather, get_time], tool_choice=True)
  
  # Prevent tool use
  bound = model.bind_tools([get_weather, get_time], tool_choice="none")
  
  # Allow web search alongside tools
  bound = model.bind_tools([get_weather], tool_choice="web_search")
  ```

- **`parallel_tool_calls`** (`bool | None`)：允许模型在一次响应中调用多个工具。默认：`None`（允许并行）。设为 `False` 可禁用。

  ```python
  # Disable parallel tool calls (one at a time)
  bound = model.bind_tools([get_weather, get_time], parallel_tool_calls=False)
  ```

- **`strict`** (`bool | None`)：若为 `True`，模型输出必须严格匹配工具 schema。schema 将按照 OpenAI 支持的 [模式](https://platform.openai.com/docs/guides/structured-outputs/supported-schemas) 进行验证。若为 `False`，则不进行验证。若为 `None`，则无严格要求。当通过 Chat Completions API 提供 `response_format` 时，`strict` 默认为 `True`，除非显式设置为 `False`。

- **`response_format`** (`dict | type | None`): 可选的 Chat Completions API 响应格式。当与 tools 一起使用时，需要设置 `strict=True`（例外：Responses API 不需要此要求）。

### 工具调用处理

当模型调用工具时，响应包含 `AIMessage.tool_calls`：

```python
response = bound_model.invoke("What's the weather in Boston and New York?")

# response.tool_calls:
# [
#   ToolCall(id='call_1', name='get_weather', args={'location': 'Boston'}),
#   ToolCall(id='call_2', name='get_weather', args={'location': 'New York'})
# ]
```

**在智能体循环中处理工具调用：**

```python
from langchain_core.messages import ToolMessage

messages = [HumanMessage("What's the weather in Boston?")]

while True:
    response = model.invoke(messages)
    
    if not response.tool_calls:
        print("Final response:", response.content)
        break
    
    messages.append(response)
    
    for tool_call in response.tool_calls:
        tool_result = get_weather(location=tool_call.args["location"])
        messages.append(ToolMessage(content=tool_result, tool_call_id=tool_call.id))
```

## 结构化输出

`with_structured_output()` 方法可将模型输出限制为特定模式。提供三种方法：

### 方法：`'function_calling'`（默认）

使用 OpenAI 的 [工具调用 API](https://platform.openai.com/docs/guides/function-calling)。模型必须调用特定工具，并且参数需匹配该模式。

**优点**：大多数模型支持（如 gpt-3.5-turbo、gpt-4 等）。

**缺点**：需要工具调用支持。严格性不如 `json_schema`。

**用法：**

```python
from pydantic import BaseModel
from langchain_openai import ChatOpenAI

class Joke(BaseModel):
    setup: str
    punchline: str

model = ChatOpenAI(model="gpt-4o")
structured = model.with_structured_output(Joke, method="function_calling")

result = structured.invoke("Tell me a joke")
print(result)
# Joke(setup='...', punchline='...')
```

### 方法：`'json_schema'`

使用 OpenAI 的 [结构化输出 API](https://platform.openai.com/docs/guides/structured-outputs)。模型生成的 JSON 必须严格符合指定模式。

**优点**：保证输出完全符合模式。适用于现代模型（如 gpt-4o-2024-08-06+、gpt-4-turbo-2024-04-09+）。

**缺点**：仅限于 profile 中 `structured_output=True` 的模型。需要有效的 JSON Schema。

**用法：**

```python
structured = model.with_structured_output(
    Joke, 
    method="json_schema",
    strict=True  # Validate schema and output
)

result = structured.invoke("Tell me a joke")
print(result)  # Pydantic instance if schema is BaseModel, else dict
```

### 方法：`'json_mode'`

使用 OpenAI 的 [JSON 模式](https://platform.openai.com/docs/guides/structured-outputs/json-mode)。模型生成 JSON，但不进行严格的模式验证。

**优点**：支持更多模型。比 `json_schema` 更简单。

**缺点**：输出可能不严格匹配模式。需要手动提示词工程。

**用法：**

```python
structured = model.with_structured_output(
    Joke,
    method="json_mode"
)

# Must include instructions in your prompt
result = structured.invoke(
    "Tell me a joke. Return as JSON: {setup: ..., punchline: ...}"
)
```

### 常用参数

```python
def with_structured_output(
    self,
    schema: dict | BaseModel | type | None = None,
    *,
    method: Literal["function_calling", "json_mode", "json_schema"] = "function_calling",
    include_raw: bool = False,
    strict: bool | None = None,
    tools: list | None = None,
    **kwargs: Any,
) -> Runnable[LanguageModelInput, dict | BaseModel]
```

- **`schema`**: 输出模式。支持：
  - Pydantic `BaseModel`（输出为该类的实例）
  - JSON Schema 字典
  - `TypedDict`
  - OpenAI 工具模式字典

- **`method`**: 约束输出的方法。默认为 `"function_calling"`。对于不兼容的方法请覆盖：

  ```python
  # For older models, auto-downgrade json_schema to function_calling
  structured = model.with_structured_output(
      Joke,
      method="json_schema"  # Auto-downgrades to function_calling if model doesn't support it
  )
  ```

- **`include_raw`** (`bool`, 默认 `False`): 返回原始模型响应和解析后的输出，以字典形式返回：

  ```python
  structured = model.with_structured_output(
      Joke,
      include_raw=True
  )
  
  result = structured.invoke("Tell me a joke")
  # {
  #   'raw': AIMessage(...),
  #   'parsed': Joke(...),
  #   'parsing_error': None
  # }
  ```

  若解析失败，`parsed` 为 `None`，`parsing_error` 为异常信息。

- **`strict`** (`bool | None`): 验证模式并强制输出完全匹配。默认值：`None`（不强制）。仅对 `json_schema` 和 `function_calling` 方法有效。

- **`tools`** (`list | None`): 模型可调用的额外工具（与结构化输出共存）。要求：
  - `method="json_schema"`
  - `strict=True`
  - `include_raw=True`
  
  当模型调用工具而非生成结构化输出时：

  ```python
  structured = model.with_structured_output(
      ResponseSchema,
      method="json_schema",
      tools=[get_weather, search_web],
      strict=True,
      include_raw=True
  )
  
  result = structured.invoke("Should I bring an umbrella to Boston?")
  # {
  #   'raw': AIMessage(tool_calls=[ToolCall(name='get_weather', ...)]),
  #   'parsed': None,
  #   'parsing_error': None
  # }
  ```

## 流式与回调

### 基础流式处理

```python
model = ChatOpenAI(model="gpt-4o", streaming=True)

for chunk in model.stream("Tell me a story"):
    print(chunk.content, end="", flush=True)
```

### 令牌回调集成

流式回调通过 `run_manager.on_llm_new_token()` 在每个数据块触发：

```python
from langchain_core.callbacks import StreamingStdOutCallbackHandler

model = ChatOpenAI(model="gpt-4o", streaming=True)

# Callbacks are invoked during stream
for chunk in model.stream(
    "Hello",
    config={"callbacks": [StreamingStdOutCallbackHandler()]}
):
    pass  # Callback prints tokens as they arrive
```

**自定义流式回调：**

```python
from langchain_core.callbacks import BaseCallbackHandler

class CustomTokenCallback(BaseCallbackHandler):
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        print(f"[TOKEN] {token}")

model = ChatOpenAI(model="gpt-4o", streaming=True)
model.invoke(
    "Hi",
    config={"callbacks": [CustomTokenCallback()]}
)
```

### 异步流式处理与分块超时

异步流式操作应用 `stream_chunk_timeout`（默认 120 秒）：

```python
async def stream_response():
    model = ChatOpenAI(
        model="gpt-4o",
        streaming=True,
        stream_chunk_timeout=30.0  # 30-second per-chunk timeout
    )
    
    async for chunk in model.astream("Tell me a long story"):
        print(chunk.content, end="", flush=True)

import asyncio
asyncio.run(stream_response())
```

如果某个数据块在超时时间内未到达，则会抛出 `StreamChunkTimeoutError`。此错误与 `httpx` 的读取超时不同——它衡量的是*已解析数据块之间*的静默时间，而非字节间的静默时间。

## 错误处理与重试

`ChatOpenAI` 将 OpenAI SDK 的异常映射到 LangChain 的标准化错误层级：

| OpenAI 异常 | LangChain 类 | 含义 |
|---|---|---|
| `AuthenticationError` | `ModelAuthenticationError` | 无效的 API 密钥 |
| `PermissionDeniedError` | `ModelPermissionDeniedError` | API 密钥缺少权限 |
| `BadRequestError` (context_length_exceeded) | `ContextOverflowError` | 输入超出模型的上下文窗口 |
| `BadRequestError` (response_format 验证失败) | `ModelInvalidRequestError` | 结构化输出的模式无效 |
| `RateLimitError` | `ModelRateLimitError` | 速率限制已超出 |
| `NotFoundError` | `ModelNotFoundError` | 模型不存在或不可用 |
| `APIError` / `InternalServerError` | `ModelAPIError` | OpenAI 服务器错误 |
| `APIConnectionError` | `ModelConnectionError` | 网络连接问题 |
| `APITimeoutError` | `ModelTimeoutError` | 请求超时 |

**错误处理示例：**

```python
from langchain_core.exceptions import (
    ContextOverflowError,
    ModelAuthenticationError,
    ModelRateLimitError,
    ModelTimeoutError,
)

model = ChatOpenAI(model="gpt-4o")

try:
    response = model.invoke(messages)
except ContextOverflowError as e:
    print(f"Message too long: {e}")
except ModelAuthenticationError as e:
    print(f"Auth failed: {e}")
except ModelRateLimitError as e:
    print(f"Rate limited, retry later")
except ModelTimeoutError as e:
    print(f"Request timed out")
```

**重试配置：**

对临时性故障的自动重试通过 `max_retries` 配置（默认值：None）。OpenAI SDK 会在某些临时性错误（429、500-599 状态码）上自动重试：

```python
model = ChatOpenAI(
    model="gpt-4o",
    max_retries=3,  # Retry up to 3 times on transient failures
    timeout=30.0   # Request timeout in seconds
)

# Or with tuple for separate connect/read timeouts
model = ChatOpenAI(
    model="gpt-4o",
    timeout=(10.0, 30.0)  # (connect_timeout, read_timeout)
)
```

**流式数据块超时（异步流式传输）：**

当异步流式传输在已解析的数据块之间出现停滞（非心跳保持）时，会抛出 `StreamChunkTimeoutError`：

```python
from langchain_openai import StreamChunkTimeoutError

model = ChatOpenAI(
    model="gpt-4o",
    stream_chunk_timeout=60.0  # Timeout per chunk
)

try:
    async for chunk in model.astream("Hello"):
        print(chunk.content, end="")
except StreamChunkTimeoutError as e:
    print(f"Stream stalled: {e}")
```

## 高级配置

### 代理与网络

```python
# Explicit proxy
model = ChatOpenAI(
    model="gpt-4o",
    openai_proxy="http://proxy.example.com:8080"
)

# Or via environment: OPENAI_PROXY=...
```

### 自定义 HTTP 客户端

```python
import httpx

http_client = httpx.Client(
    timeout=30.0,
    limits=httpx.Limits(max_connections=10)
)

model = ChatOpenAI(
    model="gpt-4o",
    http_client=http_client
)
```

### 提示词缓存

```python
# Cache long system prompts or large context
model = ChatOpenAI(
    model="gpt-4o",
    prompt_cache_options={
        "type": "ephemeral"
    }
)
```

### Logit 偏置

```python
# Encourage specific tokens
model = ChatOpenAI(
    model="gpt-4o",
    logit_bias={
        20: 50,    # Boost token ID 20
        100: -100  # Suppress token ID 100
    }
)
```

## 消息处理与生成

### 消息转换

`ChatOpenAI` 将 LangChain 的消息类型转换为 OpenAI API 格式，并反向转换：

**输入消息类型**（转换为 OpenAI 格式）：
- `HumanMessage`：用户角色
- `AIMessage`：助手角色（含 tool_calls 和 additional_kwargs）
- `SystemMessage`：系统角色（若标记为 `__openai_role__`，则为 "developer"）
- `ToolMessage`：工具角色（含 tool_call_id）
- `FunctionMessage`：函数角色（旧版）

**输出**：`AIMessage` 包含：
- `content`：文本响应
- `tool_calls`：若模型调用了工具，则为 `ToolCall` 对象列表
- `invalid_tool_calls`：无法解析的格式错误的工具调用
- `additional_kwargs`：音频数据（若启用音频输出）、function_call（旧版）等
- `response_metadata`：token 使用量、结束原因、系统指纹、logprobs 等
- `usage_metadata`：标准化的使用统计（input_tokens、output_tokens、total_tokens）

### 生成流程

1. **输入归一化**：将字符串或消息列表转换为 `ChatPromptValue`
2. **消息格式化**：根据 API 要求格式化内容块（文本、图像、工具使用标记）
3. **负载构建**：构建包含 model、messages、参数、tools、response_format 等的请求字典
4. **API 选择**：根据模型和参数决定使用 Chat Completions 还是 Responses API
5. **API 调用**：调用 OpenAI SDK（同步或异步）
6. **响应解析**：提取消息内容、工具调用、使用量、元数据
7. **消息创建**：将结果封装为带所有元数据的 `AIMessage`
8. **回调触发**：调用 LLM 回调以实现日志记录、流式输出等功能

### 内容块处理

当消息包含多模态内容（文本 + 图片，文本 + 工具引用）时，`ChatOpenAI` 会根据 API 要求进行格式化：

```python
from langchain_core.messages import HumanMessage

# Multi-modal message
message = HumanMessage(
    content=[
        {"type": "text", "text": "Analyze this chart"},
        {
            "type": "image_url",
            "image_url": {"url": "https://example.com/chart.png", "detail": "high"}
        }
    ]
)

response = model.invoke([message])
```

- 对于 **聊天补全 API**，某些内容块类型会被过滤（例如 `thinking`、`tool_use`）。
- 对于 **响应 API**，内容块会被展开以支持推理、计算机使用、文件搜索等。

## Azure OpenAI 集成

`AzureChatOpenAI` 是用于 Azure OpenAI 部署的专用子类。它继承了 `ChatOpenAI` 的所有功能（工具调用、结构化输出、流式传输、视觉支持），但具备 Azure 特有的认证方式、端点路由和响应元数据处理。

### Azure 设置

首先，使用 [快速入门指南](https://learn.microsoft.com/en-us/azure/ai-foundry/openai/chatgpt-quickstart) 创建一个 Azure OpenAI 部署。

安装相关包并设置环境变量：

```bash
pip install -U langchain-openai

export AZURE_OPENAI_API_KEY="your-api-key"
export AZURE_OPENAI_ENDPOINT="https://your-resource-name.openai.azure.com/"
export OPENAI_API_VERSION="2024-05-01-preview"  # Optional; can be passed to constructor
```

### 基本用法

```python
from langchain_openai import AzureChatOpenAI

model = AzureChatOpenAI(
    azure_deployment="my-deployment",
    api_version="2024-05-01-preview",
    temperature=0,
    max_tokens=None,
)

response = model.invoke("What is 2 + 2?")
print(response.usage_metadata)  # Token counts
```

### 关键 Azure 参数

**认证与端点：**

- **`azure_deployment`** (`str`)：Azure OpenAI 部署名称。将请求 URL 设置为 `/deployments/{azure_deployment}`。
- **`azure_endpoint`** (`str`)：完整的 Azure 端点 URL（例如 `https://resource-name.openai.azure.com/`）。可从 `AZURE_OPENAI_ENDPOINT` 环境变量自动推断。
- **`api_key`** (`str | Callable`)：Azure API 密钥。可从 `AZURE_OPENAI_API_KEY` 环境变量自动推断。
- **`azure_ad_token`** (`str`)：Azure Active Directory 令牌（作为 API 密钥的替代方案）。
- **`api_version`** (`str`)：Azure OpenAI REST API 版本（不同于模型版本）。示例：`"2024-05-01-preview"`、`"2024-02-15-preview"`。详见 [API 版本控制](https://learn.microsoft.com/en-us/azure/ai-services/openai/reference#rest-api-versioning)。

**模型配置（仅用于追踪与 token 计数）：**

- **`model`** (`str`)：底层 OpenAI 模型名称（例如 `"gpt-4o"`、`"gpt-35-turbo"`）。**不会影响**生成结果；实际使用的是 `azure_deployment`。
- **`model_version`** (`str`)：模型版本（例如 `'0125'`、`'0125-preview'`），用于 token 计数。

**其他参数：**

所有标准的 `ChatOpenAI` 参数均受支持：`temperature`、`max_tokens`、`top_p`、`frequency_penalty`、`presence_penalty`、`timeout`、`max_retries`、`streaming`、`logprobs` 等。

### Azure 响应示例

Azure 在响应中包含额外的元数据：

```python
model = AzureChatOpenAI(azure_deployment="my-deployment", api_version="2024-05-01-preview")
response = model.invoke("Translate to French: Hello")

print(response.usage_metadata)
# {'input_tokens': 28, 'output_tokens': 6, 'total_tokens': 34}

print(response.response_metadata)
# {
#   'token_usage': {
#     'completion_tokens': 6, 'prompt_tokens': 28, 'total_tokens': 34
#   },
#   'model_name': 'gpt-4o',
#   'system_fingerprint': 'fp_...',
#   'prompt_filter_results': [...],      # Content safety filtering
#   'content_filter_results': {...},     # Safety categorization
#   'finish_reason': 'stop',
# }
```

**内容安全过滤**：Azure 在 `response_metadata` 中包含 `prompt_filter_results` 和 `content_filter_results`，详细说明对仇恨言论、自残内容、色情内容及暴力内容的过滤情况。

### Azure 工具调用

使用 `AzureChatOpenAI` 进行工具调用的方式与 `ChatOpenAI` 完全相同：

```python
from pydantic import BaseModel, Field

class GetWeather(BaseModel):
    '''Get current weather'''
    location: str = Field(description="City and state, e.g. Boston, MA")

model = AzureChatOpenAI(azure_deployment="my-deployment", api_version="2024-05-01-preview")
model_with_tools = model.bind_tools([GetWeather])
response = model_with_tools.invoke("What's the weather in Boston?")
print(response.tool_calls)
```

### Azure 流式传输

使用 `AzureChatOpenAI` 的流式传输支持所有标准功能（回调、分块超时、分块中的 token 使用量）：

```python
model = AzureChatOpenAI(
    azure_deployment="my-deployment",
    api_version="2024-05-01-preview",
    streaming=True,
    stream_chunk_timeout=60.0
)

for chunk in model.stream("Translate to French: Hello"):
    print(chunk.content, end="")
```

### Azure 结构化输出

所有 `with_structured_output()` 方法均受支持：

```python
from pydantic import BaseModel

class Translation(BaseModel):
    french: str
    confidence: float

model = AzureChatOpenAI(azure_deployment="my-deployment", api_version="2024-05-01-preview")
structured = model.with_structured_output(Translation, method="json_schema")
result = structured.invoke("Translate to French: Hello world")
print(result.french)
```

## 模型名称示例

**当前推荐模型：**
- **`gpt-4o`**：最新版，支持多模态，速度最快（适用于大多数场景）
- **`gpt-4o-mini`**：轻量级，成本更低
- **`gpt-4-turbo`**：功能强大，早于 gpt-4o
- **`gpt-4`**：原始 GPT-4（已弃用）
- **`gpt-3.5-turbo`**：旧版，仍较便宜（已弃用）

请参阅 [OpenAI 模型页面](https://platform.openai.com/docs/models) 以获取最新列表。

## 测试

单元测试和集成测试位于 `repo://libs/partners/openai/tests/` 目录下。

### 单元测试

关键的单元测试文件包括：
- `repo://libs/partners/openai/tests/unit_tests/chat_models/test_base.py`：主要针对 ChatOpenAI 的测试，涵盖：
  - API 初始化与参数验证
  - 消息转换与内容块处理
  - 错误处理与异常映射
  - 工具调用与结构化输出方法
  - 带回调的流式处理
- `repo://libs/partners/openai/tests/unit_tests/chat_models/test_base_standard.py`：ChatOpenAI（聊天补全 API）的标准测试套件
- `repo://libs/partners/openai/tests/unit_tests/chat_models/test_responses_standard.py`：Responses API 的标准测试套件
- `repo://libs/partners/openai/tests/unit_tests/chat_models/test_azure.py`：Azure 特定的测试
- `repo://libs/partners/openai/tests/unit_tests/chat_models/test_client_utils.py`：客户端工具（socket 选项、代理、HTTP 客户端）

### 集成测试

包含真实 API 调用的集成测试位于 `repo://libs/partners/openai/tests/integration_tests/chat_models/`。

### 标准测试套件

`ChatOpenAI`（聊天补全）和 Responses API 均继承自 `langchain-tests` 的标准测试套件，用于验证：
- 基本的 invoke 和流式处理
- 工具调用语义
- 结构化输出一致性
- 回调集成
- Token 计数准确性

**示例单元测试：**

```python
import pytest
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

class TestSchema(BaseModel):
    name: str
    value: int

@pytest.mark.asyncio
async def test_structured_output_function_calling():
    model = ChatOpenAI(model="gpt-4o")
    structured = model.with_structured_output(TestSchema, method="function_calling")
    result = await structured.ainvoke("Return {name: 'test', value: 42}")
    assert isinstance(result, TestSchema)
    assert result.name == "test"
    assert result.value == 42

@pytest.mark.asyncio
async def test_streaming_with_callback():
    from langchain_core.callbacks import StreamingStdOutCallbackHandler
    
    model = ChatOpenAI(model="gpt-4o", streaming=True)
    chunks = []
    async for chunk in model.astream("Hello", config={"callbacks": []}):
        chunks.append(chunk)
    assert len(chunks) > 0
```

## 扩展与自定义

### 继承 BaseChatOpenAI

高级用例可通过继承 `BaseChatOpenAI` 来自定义行为：

```python
from langchain_openai.chat_models.base import BaseChatOpenAI
from langchain_core.outputs import ChatResult

class CustomChatOpenAI(BaseChatOpenAI):
    """Custom OpenAI wrapper with additional logging."""
    
    custom_param: str = "default"
    
    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        # Custom pre-processing
        print(f"Custom param: {self.custom_param}")
        
        # Call parent
        result = super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)
        
        # Custom post-processing
        result.llm_output["custom_field"] = "custom_value"
        
        return result

# Use custom class
model = CustomChatOpenAI(model="gpt-4o", custom_param="my_value")
response = model.invoke("Hello")
```

### 中间件与钩子

可通过 `RunnablePassthrough`、`RunnableLambda` 或装饰器模式添加自定义中间件：

```python
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

def log_input(input_val):
    print(f"User input: {input_val}")
    return input_val

def log_output(output):
    print(f"Model output: {output.content}")
    return output

model = ChatOpenAI(model="gpt-4o")
chain = (
    RunnableLambda(log_input)
    | model
    | RunnableLambda(log_output)
)

response = chain.invoke("What is 2+2?")
```

### 自定义客户端配置

对于高级网络控制，可提供完全配置好的 httpx 客户端：

```python
import httpx
from langchain_openai import ChatOpenAI

http_client = httpx.Client(
    timeout=httpx.Timeout(30.0),
    limits=httpx.Limits(max_connections=5, max_keepalive_connections=2),
    verify=certifi.where(),
)

http_async_client = httpx.AsyncClient(
    timeout=httpx.Timeout(30.0),
    limits=httpx.Limits(max_connections=5, max_keepalive_connections=2),
)

model = ChatOpenAI(
    model="gpt-4o",
    http_client=http_client,
    http_async_client=http_async_client,
)
```

## 已知限制与注意事项

1. **同步可调用 API 密钥**：若 `api_key` 为同步可调用对象，异步方法仍可使用，但会在线程池中解析密钥。
2. **供应商特定字段**：响应中非 OpenAI 字段（如来自 vLLM、DeepSeek）不会被保留。建议使用供应商专用包。
3. **Responses API 限制**：并非所有聊天补全参数均支持 Responses API（例如 `n` 参数不被支持）。
4. **结构化输出模式验证**：`json_schema` 方法要求模式满足 OpenAI 支持的模式约束。
5. **Azure API 版本耦合**：Azure 需显式指定 `api_version`，且版本与功能可用性绑定（例如，结构化输出仅在较新版本中支持）。

## OpenAI 决策 API

`OpenAIDecisions` 是一个专门的 Runnable（标记为 `@beta`），用于通过 OpenAI 的决策 API 对文本和图像提出类型化问题。与生成自由文本的 `ChatOpenAI` 不同，`OpenAIDecisions` 将查询结构化为谓词、选择项和评分，返回用于决策的概率和置信度分数。

**主要特性：**

- **`@beta` 装饰器**：表示不稳定的、早期阶段的 API，可能随时变更
- **结构化查询**：在共享输入上，将谓词（是/否）、选择（多选）和评分（量表）合并到一次请求中
- **置信度感知**：为每种答案类型返回概率和置信度
- **共享输入**：所有问题基于同一输入（文本或图像）以保证一致性
- **令牌使用追踪**：响应中包含令牌使用情况，用于成本/配额追踪

**基本用法：**

```python
from langchain_openai.decisions import (
    Choice,
    OpenAIDecisions,
    Predicate,
    Score,
)

decisions = OpenAIDecisions(model="gpt-6-luna")
response = decisions.invoke(
    {
        "input": "Stripe has failed to connect for three days. Help ASAP.",
        "questions": {
            "department": Choice(
                instructions="Which team should handle this request?",
                choices={
                    "billing": "Payment or subscription issues.",
                    "technical": "Product bugs or integration failures.",
                },
            ),
            "urgent": Predicate(instructions="Is this request urgent?"),
            "frustration": Score(
                instructions="How frustrated does the customer appear?",
                levels=["Calm", "Concerned", "Angry"],
            ),
        },
    }
)

# Access results by question type and key
print(response.choices["department"].choice)          # e.g., "technical"
print(response.predicates["urgent"].probability)      # e.g., 0.95
print(response.scores["frustration"].score)           # e.g., 2
```

**构造函数参数：**

- **`model`** (`str`)：决策模型标识符（例如 `"gpt-6-luna"`）。
- **`api_key`** (`SecretStr | None | Callable`)：OpenAI API 密钥。可自动从 `OPENAI_API_KEY` 推断，或通过 `LANGSMITH_GATEWAY_API_KEY` 在经 LangSmith 网关路由时获取。
- **`base_url`** (`str | None`)：自定义 API 基地址。解析顺序如下：
  1. 显式传入的 `base_url` 关键字参数
  2. `OPENAI_API_BASE` 环境变量
  3. 若设置了 `LANGSMITH_GATEWAY`，则使用 LangSmith 网关
  4. `OPENAI_BASE_URL` 环境变量
- **`organization`** (`str | None`)：OpenAI 组织 ID（来自 `OPENAI_ORG_ID` 环境变量）。
- **`timeout`** (`float | tuple[float, float] | None`)：请求超时时间（秒）。
- **`max_retries`** (`int | None`)：临时失败时的最大重试次数。
- **`default_headers`** 和 **`default_query`**：自定义 HTTP 头部和查询参数。
- **`http_client`** 和 **`http_async_client`**：用于同步/异步请求的自定义 httpx 客户端。

**答案类型：**

1. **`Predicate`**：回答是/否问题，返回概率（0–1）和置信度。

   ```python
   Predicate(instructions="Is this urgent?")
   # Response: PredicateAnswer with probability and confidence
   ```

2. **`Choice`**：从多个选项中选择，返回所选选项和置信度。

   ```python
   Choice(
       instructions="Which category?",
       choices={"A": "Description A", "B": "Description B"}
   )
   # Response: ChoiceAnswer with choice and confidence
   ```

3. **`Score`**：在量表上评分，返回数值分数和置信度。

   ```python
   Score(
       instructions="Rate from 1–5",
       levels=["Very Low", "Low", "Medium", "High", "Very High"]
   )
   # Response: ScoreAnswer with score and confidence
   ```

4. **`RefusalAnswer`**：模型拒绝回答该问题（无相关数据）

**响应结构：**

`DecisionResponse` 包含：

- **`predicates`** (`dict[str, PredicateAnswer]`)：谓词类问题的结果
- **`choices`** (`dict[str, ChoiceAnswer]`)：选择类问题的结果
- **`scores`** (`dict[str, ScoreAnswer]`)：评分类问题的结果
- **`usage`** (`Usage`)：令牌数量（input_tokens、output_tokens、total_tokens）

**带图像输入的示例：**

```python
import base64

with open("chart.png", "rb") as f:
    image_data = base64.b64encode(f.read()).decode("utf-8")

response = decisions.invoke(
    {
        "input": f"data:image/png;base64,{image_data}",
        "questions": {
            "trend": Predicate(instructions="Is the trend upward?"),
            "category": Choice(
                instructions="Which sector?",
                choices={"stocks": "Financial", "crypto": "Digital Assets"}
            ),
        },
    }
)
```

**同步与异步：**

客户端在首次调用 `invoke()` 或 `ainvoke()` 时才被惰性创建：

```python
# Sync
response = decisions.invoke(...)

# Async
response = await decisions.ainvoke(...)
```

在同一实例上混合使用同步和异步调用是受支持的；LangChain 内部会管理独立的客户端实例。

**错误处理：**

当未提供 API 密钥（既未显式传入，也未在 `OPENAI_API_KEY` 中，且无法通过 LangSmith 网关访问）时，`OpenAIDecisions` 会抛出 `ValueError`。OpenAI API 错误会被映射为标准化的 LangChain 异常，类似于 `ChatOpenAI` 的处理方式。

## 相关页面

- `/openwiki/model-initialization.md`：提供无供应商依赖的模型选择的工厂函数 `init_chat_model()`
- `/openwiki/chat-models.md`：核心 `BaseChatModel` 接口及生命周期
- `/openwiki/messages.md`：消息类型和内容块（文本、图像、工具调用）
- `/openwiki/partner-pattern.md`：合作伙伴包模式和扩展点
- `/openwiki/structured-output.md`：结构化输出模式和模式验证

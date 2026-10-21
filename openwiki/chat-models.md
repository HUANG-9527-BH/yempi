---
type: "Architecture"
title: "Chat Model Interface and Lifecycle"
description: "Document BaseChatModel protocol, input/output handling, streaming, and integration points with callbacks and model profiling. Covers the init_chat_model() factory, provider registry, and model instantiation."
tags: [chat-models, llm-integration, streaming, structured-output, model-capabilities, model-initialization]
sources:
  - id: openwiki-source-132f3183693cd9cf79d029a5
    resource: repo://libs/core/langchain_core/language_models/base.py
  - id: openwiki-source-5f8bc32563177d89fbab9b2f
    resource: repo://libs/core/langchain_core/language_models/chat_model_stream.py
  - id: openwiki-source-c52037e7b642f7ac5a7642a8
    resource: repo://libs/core/langchain_core/language_models/chat_models.py
  - id: openwiki-source-a0aef6917b7e1f4a06e6db95
    resource: repo://libs/core/langchain_core/language_models/model_profile.py
  - id: openwiki-source-c479d4fffee5cf62576699e4
    resource: repo://libs/langchain_v1/langchain/chat_models/base.py
generated: { by: "openwiki/0.5.0", at: "2026-09-28T08:35:20.640Z" }
verified:
  - by: openwiki/0.5.0
    at: 2026-10-16T08:25:55.096Z
---
## 概述

**聊天模型系统**是将大型语言模型集成到 LangChain 应用中的核心接口。`BaseChatModel` 是所有聊天模型实现所继承的抽象协议，定义了同步与异步调用/流式输出行为、回调集成、速率限制、结构化输出绑定以及通过模型配置文件进行能力发现的契约。

聊天模型将对话消息历史转换为 AI 响应，支持简单的生成（`invoke`）和流式输出（`stream`）。该框架统一了同步/异步模式，透明处理缓存，根据配置和附加回调路由至流式或非流式后端，并通过方法重写提供自定义行为的扩展点。

`init_chat_model()` 工厂函数提供了一个统一接口，可通过提供者名称（如 `'openai:gpt-4o'`）实例化任何受支持的聊天模型，具备自动提供者检测、依赖管理及回退策略功能。

## 核心接口：BaseChatModel

**位置**: `repo://libs/core/langchain_core/language_models/chat_models.py#L284-L2400`

`BaseChatModel` 继承自 `BaseLanguageModel[AIMessage]`，是一个接受 `LanguageModelInput` 输入并产生 `AIMessage` 输出的 `Runnable`。它专为子类继承而设计；实现必须重写 `_generate`（必需），可选地重写 `_llm_type`、`_stream` 和 `_agenerate`。

### 输入与输出类型

**LanguageModelInput** (`repo://libs/core/langchain_core/language_models/base.py#L140`) 是一个联合类型：

```python
LanguageModelInput = PromptValue | str | Sequence[MessageLikeRepresentation]
```

- **字符串**：转换为 `StringPromptValue`（简单用户消息）
- **消息列表**：转换为 `ChatPromptValue`（完整对话历史）
- **PromptValue**：已是结构化提示词（直接传递）

`_convert_input` 方法将所有输入形式规范化为 `PromptValue` 以供下游处理。

**输出**：所有 invoke/stream 方法返回 `AIMessage` 或 `AIMessageChunk`（用于流式）。聊天结果被封装在 `ChatGeneration` 对象中（包含消息与生成元数据），聚合为 `ChatResult`。

### 同步方法

**`invoke`** (`repo://libs/core/langchain_core/language_models/chat_models.py#L474-L499`) 是主要的同步入口点：

```python
def invoke(
    self,
    input: LanguageModelInput,
    config: RunnableConfig | None = None,
    *,
    stop: list[str] | None = None,
    **kwargs: Any,
) -> AIMessage
```

- 将输入转换为 `PromptValue`，再转换为消息
- 调用 `generate_prompt`（内部调用 `_generate_with_cache`）
- 提取并返回首个生成结果的消息
- 从配置中传播 `run_id`、回调、标签和元数据

**`stream`** (`repo://libs/core/langchain_core/language_models/chat_models.py#L726-L856`) 在响应到达时逐个产出 `AIMessageChunk` 对象：

```python
def stream(
    self,
    input: LanguageModelInput,
    config: RunnableConfig | None = None,
    *,
    stop: list[str] | None = None,
    **kwargs: Any,
) -> Iterator[AIMessageChunk]
```

- 检查是否启用流式且由 `_should_stream()` 实现
- 若流式未启用或未实现，则回退至 `invoke`
- 对于支持流式的模型，直接调用 `_stream()` 并逐个产出分块
- 使用回调生命周期包装输出：`on_chat_model_start`、`on_llm_new_token`（每块）、`on_llm_end` 或 `on_llm_error`
- 若已配置则应用速率限制
- 正常化消息并处理流式特定的输出格式（例如 `output_version="v1"`）
- 流式完成时产出一个空分块，其中 `chunk_position="last"`

### 异步方法

**`ainvoke`** (`repo://libs/core/langchain_core/language_models/chat_models.py#L501-L523`) 是异步变体：

```python
async def ainvoke(
    self,
    input: LanguageModelInput,
    config: RunnableConfig | None = None,
    *,
    stop: list[str] | None = None,
    **kwargs: Any,
) -> AIMessage
```

- 等待 `agenerate_prompt`
- 否则与 `invoke` 行为一致

**`astream`** (`repo://libs/core/langchain_core/language_models/chat_models.py#L857-L990`) 是异步流式变体：

```python
async def astream(
    self,
    input: LanguageModelInput,
    config: RunnableConfig | None = None,
    *,
    stop: list[str] | None = None,
    **kwargs: Any,
) -> AsyncIterator[AIMessageChunk]
```

- 检查 `_should_stream(async_api=True)` 以决定路由到 `_astream` 或回退
- 否则与 `stream` 行为一致，使用异步回调分发

## 流式架构

### 流式决策逻辑

**`_should_stream()`** (`repo://libs/core/langchain_core/language_models/chat_models.py#L549-L585`) 决定是否使用流式代码路径：

```python
def _should_stream(
    self,
    *,
    async_api: bool,
    run_manager: CallbackManagerForLLMRun | AsyncCallbackManagerForLLMRun | None = None,
    **kwargs: Any,
) -> bool
```

返回 `True` 的条件：
1. 流式未被禁用（`_streaming_disabled()` 返回 `False`）
2. 所请求的变体（同步/异步）已实现流式方法
3. 以下任一为真：
   - 显式传入 `stream=True` 关键字参数
   - 实例级属性 `streaming=True`
   - 绑定了 v1 风格的 `_StreamingCallbackHandler`

返回 `False`（回退至非流式）的条件：
- `disable_streaming=True`（强制禁用）
- `disable_streaming="tool_calling"` 且提供了工具
- 显式设置 `stream=False`
- 流式未实现，且异步回退至同步

### 流式实现方法

**`_stream()`** (`repo://libs/core/langchain_core/language_models/chat_models.py#L2255-L2273`) 是同步流式钩子（可选重写）：

```python
def _stream(
    self,
    messages: list[BaseMessage],
    stop: list[str] | None = None,
    run_manager: CallbackManagerForLLMRun | None = None,
    **kwargs: Any,
) -> Iterator[ChatGenerationChunk]
```

- 子类可重写以实现原生流式
- 默认抛出 `NotImplementedError`（回退至 `_generate`）
- 接收 run_manager 用于每标记回调

**`_astream()`** (`repo://libs/core/langchain_core/language_models/chat_models.py#L2275-L2311`) 是异步流式钩子（可选重写）：

```python
async def _astream(
    self,
    messages: list[BaseMessage],
    stop: list[str] | None = None,
    run_manager: AsyncCallbackManagerForLLMRun | None = None,
    **kwargs: Any,
) -> AsyncIterator[ChatGenerationChunk]
```

- 默认实现是在执行器中运行 `_stream()` 并生成结果
- 子类可重写以实现原生异步流式

### ChatModelStream 与 AsyncChatModelStream

**位置**：`repo://libs/core/langchain_core/language_models/chat_model_stream.py`

对于 v3 事件协议（`stream_events(version="v3")`），模型返回 `ChatModelStream`（同步）或 `AsyncChatModelStream`（异步），其暴露 **类型化投影** 用于增量内容：

- **`.text`**：累积文本内容块
- **`.reasoning`**：累积推理/思维链内容
- **`.tool_calls`**：累积解析后的工具调用块
- **`.usage`**：累积 token 使用信息
- **`.output`**：最终组装的 `AIMessage`

每个投影均可迭代获取增量值，或等待获取最终值。内部这些累加器会跟踪传入的协议事件，并将其合并为结构化输出。

## 生成与缓存

### 核心生成方法

**`_generate()`** (`repo://libs/core/langchain_core/language_models/chat_models.py#L2208-L2226`) 是所有子类必须实现的 **必需抽象方法**：

```python
@abstractmethod
def _generate(
    self,
    messages: list[BaseMessage],
    stop: list[str] | None = None,
    run_manager: CallbackManagerForLLMRun | None = None,
    **kwargs: Any,
) -> ChatResult
```

- 调用底层模型的 API
- 返回包含 `ChatGeneration` 对象列表的 `ChatResult`
- 必须在内部处理错误或向上抛出
- 接收归一化的消息和用于回调的 run_manager

**`_agenerate()`** (`repo://libs/core/langchain_core/language_models/chat_models.py#L2228-L2253`) 是可选的异步覆盖方法：

```python
async def _agenerate(
    self,
    messages: list[BaseMessage],
    stop: list[str] | None = None,
    run_manager: AsyncCallbackManagerForLLMRun | None = None,
    **kwargs: Any,
) -> ChatResult
```

- 默认实现是在执行器中运行 `_generate`
- 子类可通过重写以支持原生异步 API

### 缓存生成

**`_generate_with_cache()`** 和 **`_agenerate_with_cache()`** 会包装核心方法，包含以下逻辑：

1. **提示词缓存**：检查 `self.cache` 或全局 `get_llm_cache()` 是否已缓存输入对应的生成结果
2. **缓存命中**：返回缓存的生成结果，并在启用 v2 处理器时将其作为事件重放
3. **缓存未命中**：按流式或非流式路径继续处理
4. **协议路由**：根据 `_should_use_protocol_streaming` 分发至 v2 事件，或根据 `_should_stream` 路由至 v1 回调路径

### 批量处理方法

**`generate()`** 和 **`agenerate()`** 接受消息列表的列表，并使用内部缓存/流式机制批量处理提示词：

```python
def generate(
    self,
    messages: list[list[BaseMessage]],
    stop: list[str] | None = None,
    callbacks: Callbacks = None,
    **kwargs: Any,
) -> LLMResult
```

返回一个 `LLMResult`，其中生成结果按输入提示词分组，并合并了 `llm_output`。

## 回调生命周期

聊天模型与回调系统集成，可在执行过程中发出结构化事件：

### LLM 运行生命周期

1. **`on_chat_model_start`**（或回退的 `on_llm_start`）：
   - 在 `invoke`、`stream` 或 `generate` 开始时触发
   - 接收序列化的模型配置、格式化后的输入消息、调用参数和批处理大小
   - 返回绑定到该操作的运行管理器

2. **`on_llm_new_token`**（仅限流式）：
   - 每次流式传输一个 token 或数据块时触发
   - 接收 token 字符串和 `ChatGenerationChunk` 的元数据
   - 支持实时输出捕获

3. **`on_llm_end`**：
   - 生成成功完成时触发
   - 接收包含所有生成结果和元数据的最终 `LLMResult`

4. **`on_llm_error`**：
   - 生成过程中抛出异常时触发
   - 接收异常和部分 `LLMResult`（如有）
   - `_generate_response_from_error()` 可从 HTTP 错误中提取响应元数据

5. **`on_stream_event`**（v2/v3 协议）：
   - 流式过程中每个内容块协议事件触发一次
   - 支持对高级追踪进行细粒度事件观察

### 回调配置

通过 `RunnableConfig` 配置回调：

```python
config = {
    "callbacks": [my_handler],  # Callbacks for this run
    "tags": ["agent", "tools"],  # Labels for filtering
    "metadata": {"user_id": "123"},  # Context data
    "run_name": "my_run",  # Human-readable run name
    "run_id": uuid.uuid4(),  # Explicit run ID (optional)
}
result = model.invoke(input, config=config)
```

可继承的元数据和 LangSmith 参数通过 `_get_invocation_params()` 和 `_get_ls_params()` 提取。

## 结构化输出与工具绑定

### with_structured_output()

**位置**：`repo://libs/core/langchain_core/language_models/chat_models.py#L2385-L2565`

`with_structured_output()` 将聊天模型包装为约束输出符合指定模式：

```python
def with_structured_output(
    self,
    schema: dict[str, Any] | type,
    *,
    include_raw: bool = False,
    **kwargs: Any,
) -> Runnable[LanguageModelInput, dict[str, Any] | BaseModel]
```

**工作原理**：

1. 委托给 `bind_tools([schema], tool_choice="any", ...)`
2. 将结果链式传递至输出解析器：
   - 若模式为 Pydantic 类：使用 `PydanticToolsParser` → Pydantic 实例
   - 若模式为字典：使用 `JsonOutputKeyToolsParser` → 字典
3. 若 `include_raw=True`：将输出包装为 `{"raw": AIMessage, "parsed": ..., "parsing_error": ...}`
4. 若解析失败且 `include_raw=False`：抛出异常

**前置条件**：要求模型实现 `bind_tools()`（并非所有模型都支持此功能）。

### bind_tools()

**位置**：`repo://libs/core/langchain_core/language_models/chat_models.py#L2366-L2383`

```python
def bind_tools(
    self,
    tools: Sequence[dict[str, Any] | type | Callable[..., Any] | BaseTool],
    *,
    tool_choice: str | None = None,
    **kwargs: Any,
) -> Runnable[LanguageModelInput, AIMessage]
```

- 抽象方法；支持工具调用的子类必须实现
- 将工具列表绑定到模型
- 返回一个绑定了工具定义的 Runnable，该定义会包含在 API 请求中
- `tool_choice="any"` 强制模型至少调用一个工具

## 模型配置文件与能力

**位置**：`repo://libs/core/langchain_core/language_models/model_profile.py`

`BaseChatModel` 上的 `profile` 字段包含关于模型能力的元数据：

```python
class ModelProfile(TypedDict, total=False):
    # Metadata
    name: str  # Human-readable model name
    status: str  # 'active', 'deprecated', etc.
    release_date: str  # ISO 8601
    last_updated: str  # ISO 8601
    open_weights: bool  # Weights publicly available?
    
    # Input constraints
    max_input_tokens: int  # Context window size
    text_inputs: bool
    image_inputs: bool
    image_url_inputs: bool
    pdf_inputs: bool
    audio_inputs: bool
    video_inputs: bool
    image_tool_message: bool  # Images in ToolMessage?
    pdf_tool_message: bool  # PDFs in ToolMessage?
    
    # Output constraints
    max_output_tokens: int
    text_outputs: bool
    image_outputs: bool
    audio_outputs: bool
    video_outputs: bool
    
    # Capabilities
    tool_calling: bool  # Supports function calling?
    tool_choice: bool  # Supports tool_choice parameter?
    tool_call_streaming: bool  # Returns structured tool_call_chunks when streaming?
    structured_output: bool  # Native structured output support?
    reasoning_output: bool  # Reasoning/chain-of-thought?
    reasoning_effort_levels: list[str]  # ['low', 'medium', 'high']
    reasoning_effort_default: str
    temperature: bool  # Supports temperature parameter?
    attachment: bool  # Supports file attachments?
```

**自动加载**：配置文件通过 `_resolve_model_profile()`（子类重写）解析，并缓存在 `profile` 字段中。未知键名会通过 `_warn_unknown_profile_keys()` 触发警告。

### 合作方模式集成

合作包（如 `langchain-openai`）重写 `_resolve_model_profile()`，从其自身的配置数据中加载特定于模型的元数据。基础验证器 `_set_model_profile`（Pydantic mode="after"）会在未显式设置时自动填充该字段。

## 模型初始化：init_chat_model()

**位置**：`repo://libs/langchain_v1/langchain/chat_models/base.py#L197-L533`

`init_chat_model()` 是一个统一的工厂函数，可根据提供者名称实例化任意受支持的聊天模型，具备自动依赖管理、提供者推断和回退策略。

### 基本用法

```python
from langchain.chat_models import init_chat_model

# Fixed model with explicit provider
model = init_chat_model("openai:gpt-4o", temperature=0.7)

# Bare model name with inferred provider
model = init_chat_model("claude-3-sonnet", api_key="...")

# Configurable model (deferred instantiation)
configurable = init_chat_model(configurable_fields=("model", "model_provider"))
result = configurable.invoke("Hello", config={"configurable": {"model": "gpt-4o"}})
```

### 函数签名

```python
def init_chat_model(
    model: str | None = None,
    *,
    model_provider: str | None = None,
    configurable_fields: Literal["any"] | list[str] | tuple[str, ...] | None = None,
    config_prefix: str | None = None,
    **kwargs: Any,
) -> BaseChatModel | _ConfigurableModel
```

**参数**：

- **`model`**：带可选提供者前缀的模型名称字符串（例如 `'openai:gpt-4o'` 或仅 `'gpt-4o'`）。若为 `None` 且 `configurable_fields` 未设置，则默认进入可配置模式。
- **`model_provider`**：若未从 `model` 前缀推断出提供者，则需指定提供者名称。示例：`'openai'`、`'anthropic'`、`'bedrock'`、`'google_vertexai'`。
- **`configurable_fields`**：哪些参数可在运行时配置：
  - `None`：固定模型（当 `model` 被指定时的默认值）
  - `"any"`：所有字段均可配置（对不受信任的配置存在安全风险）
  - `list[str] | tuple[str, ...]`：仅指定字段（例如 `["model", "temperature"]`）
  - 当 `model` 为 `None` 时的默认值：`("model", "model_provider")`
- **`config_prefix`**：配置键的前缀（例如 `"llm"` → `config["configurable"]["llm_model"]`）
- **`**kwargs`**：模型特定参数（如 `temperature`、`max_tokens`、`api_key`、`base_url` 等）

**返回**：

- `BaseChatModel`：若指定了 `model` 且 `configurable_fields` 为 `None`，则返回一个可直接使用的聊天模型
- `_ConfigurableModel`：若字段可配置，则返回一个延迟初始化的 Runnable；在调用时执行 `_init_chat_model_helper()`

### 提供者推断

当 `model` 指定但无提供者前缀时，`_attempt_infer_model_provider()` 会根据模型名称模式猜测提供者：

| 模式 | 提供者 | 示例 |
|------|--------|------|
| `gpt-...`, `o1...`, `o3...` | `openai` | `gpt-4o`, `o1-pro` |
| `claude...` | `anthropic` | `claude-3-sonnet` |
| `command...` | `cohere` | `command-light` |
| `accounts/fireworks...` | `fireworks` | `accounts/fireworks/models/mixtral-8x7b` |
| `mistral...`, `mixtral...` | `mistralai` | `mistral-large` |
| `deepseek...` | `deepseek` | `deepseek-chat` |
| `grok...` | `xai` | `grok-2` |
| `sonar...` | `perplexity` | `sonar-pro` |
| `solar...` | `upstage` | `solar-1-mini` |
| `gemini...` | `google_vertexai`（默认） | `gemini-2.0-flash` |
| `amazon...`, `anthropic...`, `meta...` | `bedrock` | `amazon.nova-pro` |

当可能时，优先使用显式的提供者前缀形式（例如 `'openai:gpt-4o'`），以避免推断歧义。

### 内置提供者注册表

**位置**: `repo://libs/langchain_v1/langchain/chat_models/base.py#L56-L99`

`_BUILTIN_PROVIDERS` 字典将提供者名称映射到其导入路径和实例化函数：

```python
_BUILTIN_PROVIDERS: dict[str, tuple[str, str, Callable[..., BaseChatModel]]] = {
    "anthropic": ("langchain_anthropic", "ChatAnthropic", _call),
    "anthropic_bedrock": ("langchain_aws", "ChatAnthropicBedrock", _call),
    "azure_ai": ("langchain_azure_ai.chat_models", "AzureAIOpenAIApiChatModel", _call),
    "azure_openai": ("langchain_openai", "AzureChatOpenAI", _call),
    "baseten": ("langchain_baseten", "ChatBaseten", _call),
    "bedrock": ("langchain_aws", "ChatBedrock", _call),
    "bedrock_converse": ("langchain_aws", "ChatBedrockConverse", _call),
    "bedrock_mantle_anthropic": ("langchain_aws", "ChatAnthropicMantle", _call),
    "bedrock_mantle_openai": ("langchain_aws", "ChatOpenAIMantle", _call),
    "cohere": ("langchain_cohere", "ChatCohere", _call),
    "deepseek": ("langchain_deepseek", "ChatDeepSeek", _call),
    "fireworks": ("langchain_fireworks", "ChatFireworks", _call),
    "google_anthropic_vertex": ("langchain_google_vertexai.model_garden", "ChatAnthropicVertex", _call),
    "google_genai": ("langchain_google_genai", "ChatGoogleGenerativeAI", _call),
    "google_vertexai": ("langchain_google_vertexai", "ChatVertexAI", _call),
    "groq": ("langchain_groq", "ChatGroq", _call),
    "huggingface": ("langchain_huggingface", "ChatHuggingFace", lambda cls, model, **kwargs: cls.from_model_id(...)),
    "ibm": ("langchain_ibm", "ChatWatsonx", lambda cls, model, **kwargs: cls(model_id=model, ...)),
    "langsmith": ("langchain_openai", "ChatOpenAI", _init_langsmith),
    "litellm": ("langchain_litellm", "ChatLiteLLM", _call),
    "meta": ("langchain_meta", "ChatMetaModel", _call),
    "mistralai": ("langchain_mistralai", "ChatMistralAI", _call),
    "nvidia": ("langchain_nvidia_ai_endpoints", "ChatNVIDIA", _call),
    "ollama": ("langchain_ollama", "ChatOllama", _call),
    "openai": ("langchain_openai", "ChatOpenAI", _call),
    "openrouter": ("langchain_openrouter", "ChatOpenRouter", _call),
    "perplexity": ("langchain_perplexity", "ChatPerplexity", _call),
    "together": ("langchain_together", "ChatTogether", _call),
    "upstage": ("langchain_upstage", "ChatUpstage", _call),
    "xai": ("langchain_xai", "ChatXAI", _call),
}
```

每条记录是一个元组：`(module_path, class_name, creator_func)`：
- **`module_path`**：完整的 Python 模块路径（例如 `'langchain_openai'` 或 `'langchain_azure_ai.chat_models'`）
- **`class_name`**：要实例化的模型类名称
- **`creator_func`**：接受 `cls` 和模型关键字参数的可调用对象，并返回一个 `BaseChatModel` 实例（通常是 `_call` 或自定义包装器）

### 动态加载与错误处理

**`_get_chat_model_creator()`** (`repo://libs/langchain_v1/langchain/chat_models/base.py#L152-L193`) 用于获取并缓存创建函数：

1. 验证 `provider` 是否存在于 `_BUILTIN_PROVIDERS` 中
2. 通过 `_import_module()` 导入模块
3. 从模块中获取类
4. 返回绑定到该类的偏函数
5. 缓存结果以避免重复导入

**`_import_module()`** 对 `importlib.import_module()` 进行封装，并提供有用的错误信息：

- 若导入失败：抛出 `ImportError`，并建议执行 `pip install <package>`
- 从模块路径中提取包名（例如 `'langchain_openai'` → `'langchain-openai'`）
- 错误消息格式：`"Initializing {class} requires the {pkg} package. Please install it with 'pip install {pkg}'"`

**向后兼容的备用方案**：
- `ollama` 提供者会先尝试 `langchain-ollama`，若失败则回退至 `langchain-community` 以支持旧用户
- 其他提供者在包未安装时立即抛出异常

### 可配置模型

当指定了 `configurable_fields` 时，`init_chat_model()` 返回一个 `_ConfigurableModel` 实例，该实例延迟模型实例化：

```python
# Fixed model instantiated immediately
fixed = init_chat_model("gpt-4o", temperature=0.5)

# Configurable model deferred until invoke
configurable = init_chat_model(
    "gpt-4o",
    configurable_fields=("temperature", "max_tokens"),
    config_prefix="llm"
)

# Actual instantiation happens here with runtime config
result = configurable.invoke(
    "Hello",
    config={
        "configurable": {
            "llm_temperature": 0.7,  # Override default
            "llm_max_tokens": 100,
        }
    }
)
```

`_ConfigurableModel` 实现了：
- **`invoke()`, `ainvoke()`, `stream()`, `astream()`**：调用 `_model(config)` 实例化底层模型，然后进行委托
- **`batch()`, `abatch()`**：若仅有一个配置，则使用底层模型的批量方法；否则使用 `Runnable.batch`（并行化）
- **声明式方法**：`bind_tools()`, `with_structured_output()` 将操作排队，并在模型实例化后应用
- **配置合并**：`with_config()` 应用每次调用的配置，并返回一个合并了默认值的新 `_ConfigurableModel`

### 模型特定的初始化包装器

某些提供者需要自定义实例化方式：

- **`huggingface`**：使用 `ChatHuggingFace.from_model_id(model_id=model, **kwargs)` 而非 `__init__`
- **`ibm`**：使用 `ChatWatsonx(model_id=model, **kwargs)`（参数名称与标准 `model` 不同）
- **`langsmith`**：使用 `_init_langsmith()` 包装器，应用 LangSmith 网关配置，并设置 `use_responses_api=True`

### 错误处理与回退策略

| 场景 | 行为 | 恢复措施 |
|------|------|----------|
| **提供者不在注册表中** | 抛出 `ValueError`，附带受支持提供者的列表 | 若使用未列出的提供者，请显式传递 `model_provider` |
| **模块导入失败** | 抛出 `ImportError`，附带 `pip install` 建议 | 安装所需的集成包 |
| **模块中找不到类** | 抛出 `AttributeError`（隐式） | 确认提供者已正确注册 |
| **无效的关键字参数** | 由模型类抛出 `TypeError` 或 `ValidationError` | 检查模型特定的参数名称和类型 |
| **缺少 API 凭据** | 模型初始化成功；调用失败发生在 `invoke` 时 | 设置环境变量或在关键字参数中传入凭据 |

## 配置与状态

### 核心字段


```python
class BaseChatModel(BaseLanguageModel[AIMessage], ABC):
    rate_limiter: BaseRateLimiter | None = Field(default=None, exclude=True)
    
    disable_streaming: bool | Literal["tool_calling"] = False
    # False: use streaming if available
    # True: always use non-streaming (invoke)
    # "tool_calling": use non-streaming only when tools are passed
    
    output_version: str | None = None
    # 'v0': provider-specific format (lazy-parse via content_blocks)
    # 'v1': standardized format (merged into content)
    
    profile: ModelProfile | None = Field(default=None, exclude=True)
    # Capability metadata (auto-loaded if not provided)
    
    cache: BaseCache | None = None  # Inherited from BaseLanguageModel
    callbacks: list[BaseCallbackHandler] | None = None
    verbose: bool = False
    tags: list[str] | None = None
    metadata: dict[str, Any] | None = None
```


### 必需属性

- **`_llm_type`** (属性，抽象): 唯一的模型类型标识符（例如 `"openai"`、`"anthropic"`）
- **`_identifying_params`** (属性，可选): 用于追踪的模型配置字典（例如 `{"model": "gpt-4", "temperature": 0.7}`）

## 令牌计数

**位置**: `repo://libs/core/langchain_core/language_models/base.py#L434-L463`

`BaseLanguageModel` 通过 `get_token_ids()` 和 `get_num_tokens()` 提供令牌计数功能：

```python
def get_token_ids(self, text: str) -> list[int]:
    """Return token IDs for the given text."""
    if self.custom_get_token_ids is not None:
        return self.custom_get_token_ids(text)
    return _get_token_ids_default_method(text)  # GPT-2 fallback

def get_num_tokens(self, text: str) -> int:
    """Return token count for the given text."""
    return len(self.get_token_ids(text))
```

**默认行为**：
- 使用 GPT-2 分词器作为回退方案（需要 `transformers` 包）
- 仅警告一次，指出非 GPT-2 模型的计数可能不准确
- 自定义实现可通过覆盖 `custom_get_token_ids` 字段以获得模型特定的准确性

**合作伙伴包支持**：
- 实现类（如 `ChatOpenAI`）会重写 `get_token_ids()` 以使用原生分词器
- 确保在定价和上下文窗口管理中实现准确计数

## 实现要求

子类必须实现以下内容：

| 方法/属性 | 描述 | 是否必需 | 备注 |
|---|---|---|---|
| `_generate()` | 核心生成逻辑 | ✓ | 调用提供方 API，返回 `ChatResult` |
| `_llm_type` | 模型类型标识符 | ✓ | 如 `"openai"`、`"anthropic"` 的字符串 |
| `_identifying_params` | 用于追踪的配置字典 | ✗ | 由 `_get_llm_string()` 和序列化使用 |
| `_stream()` | 同步流式输出 | ✗ | 可选；若未实现，流式将回退至 `invoke` |
| `_agenerate()` | 原生异步生成 | ✗ | 可选；默认在执行器中运行 `_generate` |
| `_astream()` | 原生异步流式输出 | ✗ | 可选；默认在执行器中运行 `_stream` |
| `bind_tools()` | 工具绑定以支持结构化输出 | ✗ | 仅当需要 `with_structured_output()` 时才需实现 |

## 示例：自定义聊天模型


```python
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, AIMessage
from langchain_core.outputs import ChatResult, ChatGeneration
from langchain_core.callbacks import CallbackManagerForLLMRun

class MyCustomChatModel(BaseChatModel):
    """Custom chat model for demonstration."""
    
    model_name: str = "my-model"
    temperature: float = 0.7
    
    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Generate a response from the messages."""
        # Call your model API here
        response_text = f"Echo: {messages[-1].content}"
        
        message = AIMessage(content=response_text)
        generation = ChatGeneration(message=message)
        return ChatResult(generations=[generation])
    
    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        """Stream tokens from the model."""
        text = f"Echo: {messages[-1].content}"
        for char in text:
            chunk = ChatGenerationChunk(
                message=AIMessageChunk(content=char)
            )
            yield chunk
    
    @property
    def _llm_type(self) -> str:
        """Return the model type identifier."""
        return "my-custom-model"
    
    @property
    def _identifying_params(self) -> dict[str, Any]:
        """Return identifying parameters for tracing."""
        return {
            "model_name": self.model_name,
            "temperature": self.temperature,
        }
```


## 高级模式

### 带回调的流式输出


```python
from langchain_core.callbacks import StreamingStdOutCallbackHandler

handler = StreamingStdOutCallbackHandler()
config = {"callbacks": [handler]}

# Streams token-by-token to stdout
for chunk in model.stream("Tell me a joke", config=config):
    pass  # Handler prints as chunks arrive
```


### 带验证的结构化输出


```python
from pydantic import BaseModel

class Answer(BaseModel):
    text: str
    confidence: float

structured_model = model.with_structured_output(Answer)
result = structured_model.invoke("What is 2+2?")  # -> Answer(text="4", confidence=0.99)
```


### 缓存与速率限制


```python
from langchain_core.caches import InMemoryCache
from langchain_core.rate_limiters import InMemoryRateLimiter

model = ChatOpenAI(
    model="gpt-4",
    cache=InMemoryCache(),  # Cache results
    rate_limiter=InMemoryRateLimiter(requests_per_second=10)  # Limit requests
)

# Subsequent identical calls hit the cache
result1 = model.invoke("Hello")
result2 = model.invoke("Hello")  # Cached, no API call
```


### 条件性流式输出


```python
model_with_fallback = ChatOpenAI().with_fallbacks([ChatAnthropic()])

# Use streaming only when a handler requests it
config = {"callbacks": [MyStreamingHandler()]}
model_with_fallback.invoke("Prompt", config=config)
```


### 在智能体中使用 init_chat_model()


```python
from langchain.chat_models import init_chat_model
from langchain.agents import create_tool_calling_agent
from langchain.agents import AgentExecutor

# Create a configurable model that can be swapped at runtime
llm = init_chat_model(
    configurable_fields=("model", "model_provider"),
    temperature=0.7
)

# Build agent with the model
agent = create_tool_calling_agent(llm, tools, prompt)
executor = AgentExecutor(agent=agent, tools=tools)

# Default with OpenAI
result = executor.invoke({"input": "What is the weather?"})

# Switch to Anthropic at runtime
result = executor.invoke(
    {"input": "What is the weather?"},
    config={"configurable": {"model": "claude-3-sonnet"}}
)
```


## 关键不变量与保证

1. **输入归一化**：所有输入形式（字符串、消息列表、PromptValue）在调用 `_generate`/`_stream` 前均被归一化为消息格式。

2. **消息 ID**：每个流式输出的消息块及最终消息均分配唯一 ID（源自 run_id），用于追踪。

3. **回调顺序**：回调按顺序触发：`on_chat_model_start` → `on_llm_new_token`（每块）→ `on_llm_end` 或 `on_llm_error`。

4. **流式回退机制**：若未实现或禁用流式输出，`stream` 将无缝回退至 `invoke`，并将结果作为单个块输出。

5. **缓存透明性**：缓存命中完全透明——与缓存未命中时具有相同的生命周期回调触发。

6. **异步/同步等价性**：异步方法与同步行为一致；默认异步实现会在执行器中运行同步方法。

7. **响应元数据**：每次生成都会在 `message.response_metadata` 中累积元数据（如令牌数、终止原因等）。

8. **错误处理**：生成过程中发生的异常会触发 `on_llm_error` 并向调用者传播；若可用，错误元数据将从 HTTP 响应中提取。

9. **提供方隔离**：每个提供方集成均可独立安装和加载；缺少的依赖包会以清晰的 `pip install` 指令报告。

10. **可配置的延迟绑定**：可在可配置模型上使用声明式方法（如 `bind_tools`、`with_structured_output`）来排队操作，并在实例化时应用，确保在运行时模型切换时行为一致。

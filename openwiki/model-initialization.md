---
type: Factory
title: Chat Model Initialization with init_chat_model
description: Factory function for instantiating chat models from provider strings with unified configuration and runtime model switching.
tags: [chat-models, factory-pattern, initialization, model-parameters, configuration, provider-registry]
sources:
  - id: openwiki-source-c479d4fffee5cf62576699e4
    resource: repo://libs/langchain_v1/langchain/chat_models/base.py
generated: { by: "openwiki/0.5.0", at: "2026-09-28T08:35:20.640Z" }
verified:
  - by: openwiki/0.5.0
    at: 2026-09-28T08:35:20.640Z
---
## 概述

`init_chat_model` 是一个工厂函数，用于通过统一接口创建聊天模型实例。它在所有受支持的提供商集成（OpenAI、Anthropic、Bedrock、Google Vertex AI 等）中集中管理模型实例化过程，处理参数路由至特定提供商的构造函数，并通过 LangChain 的 `Runnable` 配置系统支持运行时模型配置。

该工厂函数接受一个 **带有可选提供商前缀的模型名称**（例如 `"openai:gpt-4"`、`"anthropic:claude-opus-4-7"`），在未指定前缀时推断提供商，获取对应提供商的集成包，并使用转换后的关键字参数实例化相应的聊天模型类。

**核心职责：**
- 接收并解析带或不带提供商前缀的模型标识符
- 使用启发式方法从模型名称前缀推断提供商
- 动态导入提供商集成包和类
- 将特定于提供商的关键字参数路由至提供商构造函数
- 支持固定（不可配置）和运行时可配置（在调用时切换模型/提供商）两种初始化模式
- 当提供商为 `langsmith` 时，通过 LangSmith 网关路由请求

## 位置

**文件**: `repo://libs/langchain_v1/langchain/chat_models/base.py`

**公开导出**: `repo://libs/langchain_v1/langchain/chat_models/__init__.py#L5`

## 函数签名


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


**参数：**

- `model` (`str | None`): 模型标识符，可选地包含提供商前缀（`"provider:model-name"`）。如果为 `None`，则返回一个可在运行时配置的模型，需在调用时提供模型名称。示例：
  - `"openai:gpt-5.5"`（明确前缀）
  - `"gpt-5.5"`（推断为 OpenAI）
  - `None`（在运行时配置）

- `model_provider` (`str | None`): 作为前缀格式的替代方式提供的提供商名称。当提供商是动态的或需要独立配置时使用。规范化为小写并用下划线分隔（如 `"azure-openai"` → `"azure_openai"`）。

- `configurable_fields` (`Literal["any"] | list[str] | tuple[str, ...] | None`): 
  - `None`: 无字段可配置（固定模型，若 `model` 已指定则为默认值）
  - `"any"`: 所有参数在运行时均可配置（⚠️ 安全风险：包括 `api_key`、`base_url` 等敏感信息）
  - `list[str] | tuple[str, ...]`: 指定的参数名称（如 `("temperature", "max_tokens")`）可配置
  - 若 `model` 为 `None`，默认值为 `("model", "model_provider")`

- `config_prefix` (`str | None`): 可选的运行时配置键命名空间前缀。当同一应用中存在多个可配置模型时使用。配置通过 `config["configurable"]["{config_prefix}_{param}"]` 访问。

- `**kwargs`: 传递给底层聊天模型构造函数的特定于提供商的参数。常见参数：
  - `temperature` (`float`): 随机性控制（0–1 或特定于提供商的范围）
  - `max_tokens` (`int`): 最大输出 token 数量
  - `timeout` (`float`): 请求超时时间（秒）
  - `max_retries` (`int`): 重试次数上限
  - `base_url` (`str`): 自定义 API 端点（OpenAI 兼容）
  - `rate_limiter` (`BaseRateLimiter`): 速率限制实例
  - 特定于提供商：`openai_api_key`、`anthropic_api_url`、`bedrock_region` 等

**返回值：**

- `BaseChatModel`：当 `configurable_fields` 为 `None` 时（若指定了 `model`，则为默认值），返回一个固定（不可配置）的聊天模型
- `_ConfigurableModel`：一个包装了 Runnable 的可配置模型，延迟到调用 `invoke`/`stream` 并传入配置时才实例化模型，从而支持运行时模型选择

**异常：**

- `TypeError`：如果 `model` 不是字符串类型（例如传入了模型对象）
- `ValueError`：如果无法推断提供方或提供方不受支持
- `ImportError`：如果提供方的集成包未安装

## 模型名称解析与提供方推断

### 显式提供方前缀

如果模型字符串中包含冒号（`:`），且前缀为已注册的提供方，则会将其提取出来：

```python
init_chat_model("openai:gpt-5.5")     # provider='openai', model='gpt-5.5'
init_chat_model("anthropic:claude-opus-4-7")  # provider='anthropic', model='claude-opus-4-7'
```

### 无前缀的模型名称推断

若未显式指定前缀，`_attempt_infer_model_provider` 将使用不区分大小写的前缀匹配：

| 模型前缀 | 推断出的提供方 |
|---|---|
| `gpt-`, `o1`, `o3`, `chatgpt`, `text-davinci` | `openai` |
| `claude` | `anthropic` |
| `command` | `cohere` |
| `accounts/fireworks` | `fireworks` |
| `gemini` | `google_vertexai`（⚠️ 已弃用默认值；将在下一个主版本中改为 `google_genai`） |
| `amazon.`, `anthropic.`, `meta.` | `bedrock` |
| `mistral`, `mixtral` | `mistralai` |
| `deepseek` | `deepseek` |
| `grok` | `xai` |
| `sonar` | `perplexity` |
| `solar` | `upstage` |

**示例：**

```python
init_chat_model("gpt-4")  # inferred as openai:gpt-4
init_chat_model("claude-sonnet-4-5-20250929")  # inferred as anthropic
```

如果推断失败且未提供 `model_provider`，将抛出 `ValueError`，列出受支持的提供方并建议查阅文档。

## 内置提供方注册表

`_BUILTIN_PROVIDERS` 字典将提供方名称映射到模块路径、类名和实例化函数。每项均为元组格式：`(module_path, class_name, creator_func)`。

**所有 30 个内置提供方**（repo://libs/langchain_v1/langchain/chat_models/base.py#L56-L99）：

| 提供方 | 包 | 类 | 模块 | 备注 |
|---|---|---|---|---|
| `openai` | `langchain-openai` | `ChatOpenAI` | `langchain_openai` | |
| `anthropic` | `langchain-anthropic` | `ChatAnthropic` | `langchain_anthropic` | |
| `azure_openai` | `langchain-openai` | `AzureChatOpenAI` | `langchain_openai` | |
| `azure_ai` | `langchain-azure-ai` | `AzureAIOpenAIApiChatModel` | `langchain_azure_ai.chat_models` | 子模块导入 |
| `google_vertexai` | `langchain-google-vertexai` | `ChatVertexAI` | `langchain_google_vertexai` | |
| `google_genai` | `langchain-google-genai` | `ChatGoogleGenerativeAI` | `langchain_google_genai` | |
| `google_anthropic_vertex` | `langchain-google-vertexai` | `ChatAnthropicVertex` | `langchain_google_vertexai.model_garden` | 通过 Google Vertex 使用 Anthropic |
| `anthropic_bedrock` | `langchain-aws` | `ChatAnthropicBedrock` | `langchain_aws` | Bedrock 托管的 Anthropic |
| `bedrock` | `langchain-aws` | `ChatBedrock` | `langchain_aws` | 通用 Bedrock 模型 |
| `bedrock_converse` | `langchain-aws` | `ChatBedrockConverse` | `langchain_aws` | Bedrock Converse API |
| `bedrock_mantle_anthropic` | `langchain-aws` | `ChatAnthropicMantle` | `langchain_aws` | Bedrock Mantle（Anthropic） |
| `bedrock_mantle_openai` | `langchain-aws` | `ChatOpenAIMantle` | `langchain_aws` | Bedrock Mantle（OpenAI） |
| `cohere` | `langchain-cohere` | `ChatCohere` | `langchain_cohere` | |
| `deepseek` | `langchain-deepseek` | `ChatDeepSeek` | `langchain_deepseek` | |
| `fireworks` | `langchain-fireworks` | `ChatFireworks` | `langchain_fireworks` | |
| `groq` | `langchain-groq` | `ChatGroq` | `langchain_groq` | |
| `huggingface` | `langchain-huggingface` | `ChatHuggingFace` | `langchain_huggingface` | 使用 `from_model_id()` |
| `ibm` | `langchain-ibm` | `ChatWatsonx` | `langchain_ibm` | 使用 `model_id=` 参数 |
| `litellm` | `langchain-litellm` | `ChatLiteLLM` | `langchain_litellm` | |
| `meta` | `langchain-meta` | `ChatMetaModel` | `langchain_meta` | |
| `mistralai` | `langchain-mistralai` | `ChatMistralAI` | `langchain_mistralai` | |
| `nvidia` | `langchain-nvidia-ai-endpoints` | `ChatNVIDIA` | `langchain_nvidia_ai_endpoints` | |
| `ollama` | `langchain-ollama` | `ChatOllama` | `langchain_ollama` | 降级至 `langchain_community` |
| `openrouter` | `langchain-openrouter` | `ChatOpenRouter` | `langchain_openrouter` | |
| `perplexity` | `langchain-perplexity` | `ChatPerplexity` | `langchain_perplexity` | |
| `together` | `langchain-together` | `ChatTogether` | `langchain_together` | |
| `upstage` | `langchain-upstage` | `ChatUpstage` | `langchain_upstage` | |
| `xai` | `langchain-xai` | `ChatXAI` | `langchain_xai` | |
| `baseten` | `langchain-baseten` | `ChatBaseten` | `langchain_baseten` | |
| `langsmith` | `langchain-openai` | `ChatOpenAI` | `langchain_openai` | 通过 LangSmith 网关路由 |

**设计说明：**

- 该注册表**并非完整列表**。即使未列出的提供方，只要安装了相应的集成包，仍可使用，但模型名称推断将无法工作；必须显式指定 `model_provider`。
- 大多数条目使用标准的 `_call` 创建函数，直接实例化类。
- 特殊创建函数：`huggingface` 使用 `from_model_id(model_id=...)`，`ibm` 使用 `model_id=...`，`langsmith` 在实例化时包裹网关配置。

## 参数映射与创建函数

`_get_chat_model_creator` 函数会获取提供方的创建函数，并返回一个部分应用的可调用对象：

```python
@functools.lru_cache(maxsize=len(_BUILTIN_PROVIDERS))
def _get_chat_model_creator(provider: str) -> Callable[..., BaseChatModel]:
    # Look up provider in registry
    pkg, class_name, creator_func = _BUILTIN_PROVIDERS[provider]
    # Import module and get class
    module = _import_module(pkg, class_name)
    cls = getattr(module, class_name)
    # Return partial with class bound
    return functools.partial(creator_func, cls=cls)
```

**标准创建函数（`_call`）**：

```python
def _call(cls: type[BaseChatModel], **kwargs: Any) -> BaseChatModel:
    return cls(**kwargs)
```

将所有 `kwargs` 直接转发至提供方的 `__init__` 方法。

**特殊创建函数**：

- **HuggingFace**：`lambda cls, model, **kwargs: cls.from_model_id(model_id=model, **kwargs)`
  - 使用类方法 `from_model_id` 而非直接实例化
  - `model` 参数变为 `model_id=`

- **IBM**：`lambda cls, model, **kwargs: cls(model_id=model, **kwargs)`
  - 将 `model` 映射为构造函数中的 `model_id=`

- **LangSmith 网关**（`_init_langsmith`）：
  - 调用 `_apply_gateway_config` 以从环境变量（或 `LANGSMITH_GATEWAY` URL）注入网关凭据和基础 URL
  - 设置 `use_responses_api=True` 以启用与网关的兼容性
  - 若未设置 `LANGSMITH_GATEWAY_API_KEY`，则回退使用 `LANGSMITH_API_KEY`

```python
def _init_langsmith(cls: type[BaseChatModel], **kwargs: Any) -> BaseChatModel:
    _apply_gateway_config(
        kwargs,
        cls,
        base_url_field="openai_api_base",
        api_key_field="openai_api_key",
        provider_path="v1",
        api_key_env=("LANGSMITH_GATEWAY_API_KEY", "LANGSMITH_API_KEY"),
        default_base_url="https://gateway.smith.langchain.com/v1",
    )
    kwargs["use_responses_api"] = True
    return cls(**kwargs)
```

**参数路由**：

传递给 `init_chat_model` 的所有 `**kwargs` 均会被转发至提供方的构造函数。提供方的参数验证机制决定了哪些 `kwargs` 是被接受的。通用跨提供方参数（如 `temperature`、`max_tokens`、`timeout`、`max_retries`）在大多数提供方上均有效；提供方特定参数（如 `openai_api_key`、`anthropic_api_url`）仅对目标提供方有效。

## 固定模型初始化

当 `model` 被指定且 `configurable_fields` 为 `None`（默认值）时，`init_chat_model` 会立即实例化并返回一个 `BaseChatModel`：

```python
init_chat_model("gpt-4", temperature=0.7, max_tokens=500)
# Returns ChatOpenAI instance, ready to invoke
```

**控制流**（repo://libs/langchain_v1/langchain/chat_models/base.py#L515-L520）：

```python
if not configurable_fields:
    return _init_chat_model_helper(
        cast("str", model),
        model_provider=model_provider,
        **kwargs,
    )
```

`_init_chat_model_helper` 函数解析模型，获取创建函数，并完成实例化：

```python
def _init_chat_model_helper(
    model: str,
    *,
    model_provider: str | None = None,
    **kwargs: Any,
) -> BaseChatModel:
    model, model_provider = _parse_model(model, model_provider)
    creator_func = _get_chat_model_creator(model_provider)
    return creator_func(model=model, **kwargs)
```

**错误处理：**

- 若缺少依赖包：抛出 `ImportError`，并建议执行 `pip install <package>`
- 若提供方未知：抛出 `ValueError`，列出所有支持的提供方
- 若 `model_provider` 推断失败：抛出 `ValueError`，附带文档链接

## 运行时可配置模型初始化

当 `configurable_fields` 不为 `None` 或 `model` 为 `None` 时，`init_chat_model` 返回一个 `_ConfigurableModel`，这是一个 `Runnable` 包装器，将在调用时提供配置后才延迟实例化。

**使用场景：**

1. **无默认模型** – 在运行时选择模型：

   ```python
   model = init_chat_model()  # No model specified
   model.invoke("hello", config={"configurable": {"model": "gpt-4"}})
   model.invoke("hello", config={"configurable": {"model": "claude-opus-4-7"}})
   ```

2. **有默认模型，可切换参数** – 在运行时覆盖特定字段：

   ```python
   model = init_chat_model(
       "gpt-4",
       configurable_fields=("temperature", "max_tokens"),
       temperature=0.5,
       max_tokens=100
   )
   model.invoke(
       "hello",
       config={"configurable": {"temperature": 0.9, "max_tokens": 500}}
   )
   ```

3. **有默认模型，完全可配置** – 在运行时切换模型或任意参数：

   ```python
   model = init_chat_model(
       "gpt-4",
       configurable_fields="any",  # All fields configurable
       config_prefix="my_model"
   )
   model.invoke("hello")  # Uses gpt-4, temperature=None
   model.invoke(
       "hello",
       config={
           "configurable": {
               "my_model_model": "claude-opus-4-7",
               "my_model_temperature": 0.8
           }
       }
   )
   ```


### _ConfigurableModel

**位置**: `repo://libs/langchain_v1/langchain/chat_models/base.py#L657-L1050`

`_ConfigurableModel` 是一个 `Runnable[LanguageModelInput, Any]`，它会将模型初始化和操作延迟执行，直到提供配置为止：

**状态：**
- `_default_config`：默认参数值的字典（例如，`{"model": "gpt-4", "temperature": 0.5}`）
- `_configurable_fields`：可在运行时覆盖的字段（`"any"`，或字段名称列表）
- `_config_prefix`：配置键的命名空间（例如，`"my_model_"`）
- `_queued_declarative_operations`：待应用的方法调用列表（例如，`bind_tools`、`with_structured_output`），在模型实例化后按顺序执行

**生命周期：**

1. **实例化**：`init_chat_model(...)` 创建 `_ConfigurableModel`，包含默认配置和已排队的操作
2. **声明式操作**（例如，`.bind_tools(...)`）：操作被排队；返回一个新的 `_ConfigurableModel`，不修改原对象
3. **调用**（例如，`.invoke(..., config=...)`）：调用 `_model(config)` 来：
   - 合并默认配置与运行时配置
   - 调用 `_init_chat_model_helper` 实例化实际模型
   - 按顺序应用所有已排队的操作
   - 返回配置后的模型实例
4. **流式/批量操作**：委托给已实例化的模型处理

**配置合并**（repo://libs/langchain_v1/langchain/chat_models/base.py#L711-L727）：


```python
def _model(self, config: RunnableConfig | None = None) -> Runnable[Any, Any]:
    params = {**self._default_config, **self._model_params(config)}
    model = _init_chat_model_helper(**params)
    for name, args, kwargs in self._queued_declarative_operations:
        model = getattr(model, name)(*args, **kwargs)
    return model

def _model_params(self, config: RunnableConfig | None) -> dict[str, Any]:
    config = ensure_config(config)
    # Extract configurable params and remove prefix
    model_params = {
        _remove_prefix(k, self._config_prefix): v
        for k, v in config.get("configurable", {}).items()
        if k.startswith(self._config_prefix)
    }
    # Filter to only allowed fields if not "any"
    if self._configurable_fields != "any":
        model_params = {k: v for k, v in model_params.items() if k in self._configurable_fields}
    return model_params
```


**声明式操作**（repo://libs/langchain_v1/langchain/chat_models/base.py#L681-L702）：

`bind_tools` 和 `with_structured_output` 等方法会被拦截并排队，而非立即执行：


```python
def __getattr__(self, name: str) -> Any:
    if name in _DECLARATIVE_METHODS:
        def queue(*args: Any, **kwargs: Any) -> _ConfigurableModel:
            queued_declarative_operations = list(self._queued_declarative_operations)
            queued_declarative_operations.append((name, args, kwargs))
            return _ConfigurableModel(
                default_config=dict(self._default_config),
                configurable_fields=self._configurable_fields,
                config_prefix=self._config_prefix,
                queued_declarative_operations=queued_declarative_operations,
            )
        return queue
    # ... delegate to default model if one exists
```


**缓存**：创建函数使用 `@functools.lru_cache` 缓存，以避免重复导入模块。

## 常见参数映射示例

### Temperature 和 max_tokens

这些参数几乎适用于所有提供商，但不同提供商的默认值和取值范围可能不同：


```python
# OpenAI: temperature 0–2 (default 1)
init_chat_model("gpt-4", temperature=0.7, max_tokens=500)

# Anthropic: temperature 0–1 (default 1)
init_chat_model("claude-opus-4-7", temperature=0.7, max_tokens=500)

# Google Vertex AI: temperature 0–2
init_chat_model("google_vertexai:gemini-1.5-pro", temperature=0.7)
```


请查阅对应提供商的集成文档以获取确切的取值范围和默认值。

### API 密钥和基础 URL

不同提供商的参数名称各不相同：


```python
# OpenAI: openai_api_key, openai_api_base
init_chat_model("gpt-4", openai_api_key="...", openai_api_base="https://custom.com/v1")

# Anthropic: anthropic_api_key, anthropic_api_url
init_chat_model("claude-opus-4-7", anthropic_api_key="...", anthropic_api_url="https://custom.com")

# Vertex AI: uses GCP credentials from environment, or project_id, location
init_chat_model("google_vertexai:gemini-1.5-pro", project_id="my-project")
```


环境变量回退机制是提供商特定的；请查阅集成包的文档。

### 重试和超时

这些是跨提供商通用的参数：


```python
init_chat_model(
    "gpt-4",
    max_retries=3,
    timeout=30.0,
)
```


### Bedrock 区域和模型 ID

AWS Bedrock 需要指定区域，并使用完整的模型 ID：


```python
init_chat_model(
    "bedrock:amazon.titan-text-express-v1",
    region_name="us-east-1",
)
```


## 测试与示例模式

### 固定模型初始化


```python
from langchain.chat_models import init_chat_model

# Explicit provider prefix
llm = init_chat_model("openai:gpt-4", temperature=0)
response = llm.invoke("What is 2+2?")

# Inferred provider
llm = init_chat_model("gpt-4", temperature=0)
response = llm.invoke("What is 2+2?")

# Separate model_provider parameter
llm = init_chat_model("gpt-4", model_provider="openai", temperature=0)
```


### 可配置模型（部分覆盖）


```python
from langchain.chat_models import init_chat_model

model = init_chat_model(
    "gpt-4",
    configurable_fields=("temperature", "max_tokens"),
    temperature=0.5,
    max_tokens=100,
)

# Use defaults
result = model.invoke("hello")

# Override at runtime
result = model.invoke(
    "hello",
    config={
        "configurable": {
            "temperature": 0.9,
            "max_tokens": 500,
        }
    }
)
```


### 可配置模型（无默认值）


```python
from langchain.chat_models import init_chat_model

model = init_chat_model(temperature=0.5)  # No model specified

# Select model at runtime
result = model.invoke(
    "hello",
    config={"configurable": {"model": "gpt-4"}}
)

result = model.invoke(
    "hello",
    config={"configurable": {"model": "claude-opus-4-7"}}
)
```


### 与提示词链式调用


```python
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate

model = init_chat_model("gpt-4", temperature=0)
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("user", "{input}"),
])

chain = prompt | model
response = chain.invoke({"input": "What is 2+2?"})
```


### 绑定工具


```python
from langchain.chat_models import init_chat_model
from pydantic import BaseModel, Field

class Calculator(BaseModel):
    """Perform arithmetic."""
    a: int = Field(..., description="First number")
    b: int = Field(..., description="Second number")

model = init_chat_model("gpt-4")
model_with_tools = model.bind_tools([Calculator])

result = model_with_tools.invoke("What is 2+2?")
```


### 可配置模型（绑定工具）


```python
from langchain.chat_models import init_chat_model
from pydantic import BaseModel, Field

class Calculator(BaseModel):
    """Perform arithmetic."""
    a: int = Field(..., description="First number")
    b: int = Field(..., description="Second number")

model = init_chat_model(
    "gpt-4",
    configurable_fields=("model", "model_provider"),
)
model_with_tools = model.bind_tools([Calculator])

# Use with default gpt-4
result = model_with_tools.invoke("What is 2+2?")

# Switch to Claude at runtime
result = model_with_tools.invoke(
    "What is 2+2?",
    config={"configurable": {"model": "claude-opus-4-7"}}
)
```


## LangSmith 网关集成

`langsmith` 提供商可连接至 LangSmith 的统一 LLM 网关，使多个非 OpenAI 模型可通过通用的 OpenAI 兼容 API 进行服务：


```python
init_chat_model("langsmith:moonshotai/kimi-k3")
```


**配置流程**：

1. 调用 `_init_langsmith` 而非标准的 `_call`
2. `_apply_gateway_config` 读取：
   - `LANGSMITH_GATEWAY` 或默认为 `https://gateway.smith.langchain.com`
   - `LANGSMITH_GATEWAY_API_KEY`（优先）或 `LANGSMITH_API_KEY`
   - 将其注入为 `openai_api_base` 和 `openai_api_key`
3. 设置 `use_responses_api=True` 以保证兼容性
4. 返回指向网关的 `ChatOpenAI` 实例

**示例**：

```python
import os
os.environ["LANGSMITH_GATEWAY_API_KEY"] = "..."
model = init_chat_model("langsmith:moonshotai/kimi-k3")
# Routes to https://gateway.smith.langchain.com/v1/chat/completions
```

## 扩展与添加新提供方

要为新的提供方集成添加支持，请更新 `repo://libs/langchain_v1/langchain/chat_models/base.py` 中的 `_BUILTIN_PROVIDERS`：

1. 添加条目：`"provider_name": (module_path, ClassName, creator_func)`
2. 如果使用标准实例化方式，使用 `_call`
3. 如果构造函数使用非标准参数来指定模型名称，则创建自定义创建函数
4. 确保提供方模块在指定的模块路径下导出了该类
5. 集成包必须可通过 pip 安装，并命名为 `langchain-<provider-name>`（将下划线转换为连字符）

针对一个假设的 "myai" 提供方的示例：

```python
_BUILTIN_PROVIDERS = {
    ...
    "myai": ("langchain_myai", "ChatMyAI", _call),
}
```

然后通过 `pip install langchain-myai` 安装并使用：

```python
init_chat_model("myai:my-model-v1")
# or
init_chat_model("my-model-v1", model_provider="myai")
```

如果存在稳定且无歧义的前缀（例如，所有 MyAI 模型均以 `myai-` 开头），请更新 `_attempt_infer_model_provider` 中的模型名称前缀推断逻辑。

## 安全注意事项

**API 密钥泄露**：当 `configurable_fields="any"` 时，所有参数（包括 `api_key`、`openai_api_key`、`anthropic_api_key` 和 `base_url`）都变为运行时可配置。在生产环境中，应将可配置字段限制为安全参数：

```python
# ❌ Unsafe: accepts any field, including secrets
model = init_chat_model("gpt-4", configurable_fields="any")

# ✅ Safe: whitelist only model switching and temperature
model = init_chat_model(
    "gpt-4",
    configurable_fields=("temperature", "max_tokens"),
)
```

**运行时配置来源**：验证配置字典来自可信源。如果配置源自用户输入，请过滤键名以防止意外的参数注入。

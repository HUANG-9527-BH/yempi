---
type: "Reference"
title: "Structured Output and Response Formatting"
description: "Document the structured output system: response format specification, strategy selection (auto, provider, tool-based), parsing, validation, and middleware integration."
tags: [structured-output, response-format, agent, schema, validation, middleware]
verified:
  - by: openwiki/0.5.0
    at: 2026-09-28T08:35:20.640Z
sources:
  - id: openwiki-source-71e882e1ac9757ea8e959a7c
    resource: repo://libs/langchain_v1/langchain/agents/factory.py
  - id: openwiki-source-ec30ab6256dd50cc670919f6
    resource: repo://libs/langchain_v1/langchain/agents/structured_output.py
generated: { by: "openwiki/0.5.0", at: "2026-09-28T08:35:20.640Z" }
---
## 概述

结构化输出是一种机制，确保语言模型返回的响应符合特定的 JSON 模式。与接收未解析的文本或工具调用不同，智能体可以强制模型输出符合 Pydantic 模型、数据类、TypedDict 或原始 JSON 模式的格式。该工厂配置了三种策略之一——基于工具的、提供方原生的或自动的，每种策略在兼容性、验证和重试行为方面各有权衡。

### 核心概念

当使用 `response_format` 参数创建智能体时，它会告知模型：“你所有的响应都必须匹配此模式”。该智能体：

1. **注册**该模式为一个虚拟工具（用于基于工具调用的策略）或发送给模型提供方的原生 API（用于提供方原生策略）
2. **检测**模型是否支持原生结构化输出（AutoStrategy）
3. **解析**模型的响应，并使用 Pydantic 的 `TypeAdapter` 依据模式进行验证
4. **自动重试**验证错误（如果通过 `handle_errors` 启用）
5. **存储**解析后的结果到 `structured_response` 状态字段

这与工具调用不同：结构化输出约束的是响应本身，而非模型的工具调用。

## 响应格式策略

三种策略控制结构化输出的强制方式：

### ToolStrategy：基于工具的结构化输出

将模型呈现为一个虚拟工具，其名称和参数与模式匹配。当模型调用此工具时，其参数将被解析并根据模式进行验证。

**生命周期：**

1. 将模式包装为一个 `StructuredTool`，并将模式的 JSON 模式作为 `args_schema`
2. 将工具添加到模型的工具列表中，并设置 `tool_choice="any"` 以强制使用
3. 模型生成一个带有模式名称的工具调用
4. 通过 `_parse_with_schema` 使用 Pydantic 的 `TypeAdapter` 解析工具调用参数
5. 解析结果存储在 `structured_response` 中
6. 返回空的 `ToolMessage`（该工具无实际执行）

**优点：**

- 适用于所有支持工具调用的模型
- 对非字典模式提供完整的验证
- 验证失败时可自动重试（可配置）
- 支持 Union 类型（多种模式变体）

**局限性：**

- 需要具备工具调用能力
- 会向工具列表中添加工具（某些模型可能占用工具槽位）
- 原始 JSON 模式字典跳过验证，导致 `handle_errors` 失效

**错误处理：**

`handle_errors` 参数控制验证失败时的重试行为：

- `True`（默认）：捕获所有错误，使用默认错误模板重试
- `False`：让异常直接传播，不重试
- `str`：捕获所有错误，使用自定义消息重试
- `type[Exception]` 或 `tuple[type[Exception], ...]`：仅对特定异常类型重试
- `Callable[[Exception], str]`：自定义函数，按异常返回重试消息

解析失败会生成一个包含错误信息的 `ToolMessage`，使模型能够修正其输出。

### ProviderStrategy：原生结构化输出

该模式通过模型提供商的原生结构化输出 API（例如 OpenAI 的 `response_format` 配合 `type: "json_schema"`）发送。提供商在自身侧强制执行模式合规性；智能体仅需解析返回的 JSON 响应。

**生命周期：**

1. 通过 Pydantic 的 `model_json_schema()` 将模式转换为 JSON Schema  
2. 包装为提供方特定格式：`{"type": "json_schema", "json_schema": {"name": ..., "schema": ..., "strict": ...}}`  
3. 通过 `model.bind(..., response_format={...})` 传递给模型  
4. 模型返回 JSON 文本（由提供方保证有效）  
5. 文本通过 `json.loads()` 解析，并根据模式进行验证  
6. 解析结果存储于 `structured_response`

**优势：**

- 提供方强制执行模式——不可能产生无效 JSON  
- 不消耗工具槽位  
- 可与工具调用并行使用  
- 支持严格模式（仅限支持的提供方）

**限制：**

- 仅限具备原生结构化输出能力的模型（如 OpenAI、Claude 等）  
- 验证错误时无自动重试机制（提供方侧严格）  
- 必须显式测试模型功能支持情况

**功能检测：**

若满足以下条件，则模型支持提供方原生结构化输出：

1. 其配置文件包含 `"structured_output": True`（通过 `model.profile` 检查），且  
2. 非前 3 系列 Gemini 模型（此类模型无法同时使用工具与结构化输出），或  
3. 模型名称匹配回退模式，如 `gpt-4o`、`claude-opus` 等

### AutoStrategy：自动策略选择

将策略选择推迟至模型调用时刻。工厂在绑定模型后进行检查，并选择：

1. **ProviderStrategy**：若模型支持原生结构化输出  
2. **ToolStrategy**：作为其他所有模型的回退方案

**生命周期：**

1. 用户传入原始模式或 `AutoStrategy(schema=...)`  
2. 工厂提前转换为 `ToolStrategy` 以预先构建工具  
3. 在模型调用时，`_supports_provider_strategy()` 检查模型能力  
4. 若支持，则创建 `ProviderStrategy` 并绑定模型参数  
5. 否则，使用 `ToolStrategy`（工具已准备就绪）

**优势：** 两者兼得——可用提供方原生结构时使用原生，否则回退至工具方式。

```python
from langchain.agents import create_agent
from pydantic import BaseModel

class WeatherResponse(BaseModel):
    """Weather forecast response."""
    location: str
    temperature: int
    condition: str

# AutoStrategy (recommended)
agent = create_agent(
    model="openai:gpt-4o",
    response_format=WeatherResponse,  # Wrapped in AutoStrategy automatically
)

# Explicit strategies
from langchain.agents.structured_output import ProviderStrategy, ToolStrategy

agent_native = create_agent(
    model="openai:gpt-4o",
    response_format=ProviderStrategy(schema=WeatherResponse),
)

agent_tools = create_agent(
    model="openai:gpt-4o",
    response_format=ToolStrategy(
        schema=WeatherResponse,
        handle_errors=True,  # Retry on validation failure
    ),
)
```

## 模式类型

支持的结构化输出模式类型：

| 类型 | 示例 | 验证方式 | 工具使用 |
|------|---------|-----------|----------|
| **Pydantic 模型** | `class Response(BaseModel): ...` | 完整验证 | 是（通过 TypeAdapter） |
| **数据类** | `@dataclass class Response: ...` | 完整验证 | 是（通过 TypeAdapter） |
| **TypedDict** | `class Response(TypedDict): ...` | 完整验证 | 是（通过 TypeAdapter） |
| **JSON Schema 字典** | `{"type": "object", "properties": {...}}` | 无验证（字典模式跳过验证） | 直接返回字典 |

工厂通过 `_SchemaSpec` 统一处理所有类型，其功能包括：

- 提取模式名称（类名、`title` 字段，或生成的 UUID 片段）  
- 提取描述（文档字符串、`description` 字段，或空值）  
- 计算用于工具绑定的 JSON Schema 表示形式  
- 跟踪 `strict` 模式标志，用于提供方侧强制执行

## 与智能体工厂的集成

`create_agent()` 函数通过以下方式实现结构化输出的集成：

### 1. 提前注册模式

```python
# At agent creation time
if tool_strategy_for_setup:
    for response_schema in tool_strategy_for_setup.schema_specs:
        structured_tool_info = OutputToolBinding.from_schema_spec(response_schema)
        structured_output_tools[structured_tool_info.tool.name] = structured_tool_info
```

预先构建 `OutputToolBinding` 实例，将模式包装为 `StructuredTool` 实例。这些绑定会存储原始模式、其分类（如 `pydantic`、`dataclass` 等）以及用于后续解析的工具。

### 2. 调用期间的模型绑定

`_get_bound_model()` 函数（在每次模型调用时被调用）执行自动检测：

```python
# Determine effective response format (auto-detect if needed)
effective_response_format: ResponseFormat[Any] | None
if isinstance(response_format, AutoStrategy):
    if _supports_provider_strategy(request.model, tools=request.tools):
        effective_response_format = ProviderStrategy(schema=response_format.schema)
    else:
        effective_response_format = ToolStrategy(schema=response_format.schema)
else:
    effective_response_format = response_format
```

然后绑定模型：

- **ProviderStrategy**：`model.bind(..., response_format={...})`
- **ToolStrategy**：`model.bind_tools(final_tools, tool_choice="any", ...)`

### 3. 响应解析

模型调用完成后，`_handle_model_output()` 将调度到相应的解析器：

**对于 ProviderStrategy：**

```python
if isinstance(effective_response_format, ProviderStrategy):
    if not output.tool_calls:
        provider_strategy_binding = ProviderStrategyBinding.from_schema_spec(...)
        structured_response = provider_strategy_binding.parse(output)
        return {"messages": [output], "structured_response": structured_response}
```

**对于 ToolStrategy：**

```python
if isinstance(effective_response_format, ToolStrategy):
    structured_tool_calls = [tc for tc in output.tool_calls if tc["name"] in structured_output_tools]
    if structured_tool_calls:
        # Single call: parse args, handle errors, return response
        structured_response = structured_output_tools[tool_call["name"]].parse(tool_call["args"])
        return {"messages": [...], "structured_response": structured_response}
```

## OutputToolBinding：模式到工具的转换

`OutputToolBinding` 是模式与工具之间的桥梁。它存储：

- **schema**：原始模式（Pydantic、dataclass、TypedDict 或 dict）
- **schema_kind**：分类（`'pydantic'`、`'dataclass'`、`'typeddict'`、`'json_schema'`）
- **tool**：带有 `args_schema` 绑定至 JSON 模式的 `StructuredTool` 实例

`parse()` 方法根据工具调用参数重建原始类型：

```python
def parse(self, tool_args: dict[str, Any]) -> SchemaT | dict[str, Any]:
    return _parse_with_schema(self.schema, self.schema_kind, tool_args)
```

**解析流程：**

1. 对于字典模式：直接返回参数（不进行验证）
2. 对于类型化模式：使用 Pydantic 的 `TypeAdapter` 验证 Python 类型
3. 验证失败时：抛出 `ValueError`，包含模式名称和错误详情

这使得工厂能够在智能体生命周期内维护一个单一的结构化输出工具名称到其绑定元数据的映射，从而在响应处理期间实现快速查找。

## 错误处理与验证

### 错误类型

**StructuredOutputError**（基类）：

- 保存引发错误的 `AIMessage`
- 其他具体错误类型的父类

**MultipleStructuredOutputsError**：

当预期仅有一个结构化输出模式，但模型调用了多个结构化输出工具时抛出。

```python
tool_names = [tc["name"] for tc in structured_tool_calls]
exception = MultipleStructuredOutputsError(tool_names, output)
```

**StructuredOutputValidationError**：

当工具调用参数无法根据模式解析时抛出。

```python
exception = StructuredOutputValidationError(tool_name, source_exception, output)
```

### 重试逻辑（仅适用于 ToolStrategy）

当启用 `handle_errors` 且在工具解析过程中发生验证错误时：

1. `_handle_structured_output_error()` 判断是否应重试
2. 返回 `(should_retry: bool, error_message: str)`
3. 若需重试：将错误消息封装在 `ToolMessage` 中并追加到对话中
4. 模型接收错误上下文，可修正其响应

**错误回调：**

```python
should_retry, error_message = _handle_structured_output_error(
    exception, effective_response_format
)
if not should_retry:
    raise exception from exc

# Return error message to model
return {
    "messages": [
        output,
        ToolMessage(
            content=error_message,
            tool_call_id=tool_call["id"],
            name=tool_call["name"],
        ),
    ],
}
```

模型的下一轮将接收到该错误，并可尝试修正输出。

### 验证限制

**字典模式跳过验证**：原始 JSON 模式字典（非 Pydantic、dataclass 或 TypedDict）直接返回参数：

```python
if schema_kind == "json_schema":
    return data  # No validation, no retry possible
```

若要启用验证和重试，请将模式表示为 Pydantic 模型或 TypedDict。

**提供方策略不支持重试**：原生结构化输出由提供方强制执行；智能体接收有效的 JSON 或 API 错误。在对话过程中，对于模式不匹配的情况无法进行重试。

## 状态与生命周期

### 状态字段

智能体状态通过以下方式处理结构化输出：

- **messages**：包含结构化输出调用产生的工具调用和工具消息
- **structured_response**：存储已解析的模式实例（当输出有效时设置，错误重试或未生成结构化输出时清除）

### 结构化响应清除

`_build_commands()` 函数确保在以下情况下显式清除 `structured_response` 字段：

1. 智能体配置了 `response_format`（`has_structured_output = True`）
2. 模型调用未产生结构化响应

这可防止持久化智能体状态中前一检查点轮次的过期值。清除操作发生在传递给图的 state 更新中：

```python
state: dict[str, Any] = {"messages": messages}
if model_response.structured_response is not None:
    state["structured_response"] = model_response.structured_response
elif has_structured_output:
    state["structured_response"] = None  # Explicitly clear to avoid stale values
```

### 生命周期事件

```
User input
  ↓
[pre-model middleware]
  ↓
_get_bound_model() → detect strategy, bind model with tools or provider format
  ↓
model.invoke() → model returns AIMessage with tool_calls (ToolStrategy)
                 or text (ProviderStrategy)
  ↓
_handle_model_output() → parse response, validate against schema
  ↓
[structured_response set] or [error + retry message]
  ↓
[post-model middleware]
  ↓
Return to user or continue loop
```

## 配置与中间件

原始模式和策略决策通过 `ModelRequest.response_format` 传递给中间件。中间件可在调用时通过 `ModelRequest.override()` 覆盖 `response_format`：

```python
class MyMiddleware(AgentMiddleware):
    def wrap_model_call(self, request, handler):
        # Narrow union response format to a specific variant
        narrow_format = ToolStrategy(schema=request.response_format.schema_specs[0])
        return handler(request.override(response_format=narrow_format))
```

智能体在每次调用时重新检测策略并重建工具绑定，从而支持动态模式变更。然而，**所有结构化输出模式必须提前声明**——中间件无法添加初始 `response_format` 中不存在的新模式。

## 示例：带结构化输出的天气智能体

```python
from pydantic import BaseModel
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

class WeatherResponse(BaseModel):
    """Current weather forecast."""
    location: str
    temperature_f: int
    condition: str
    humidity_percent: int

# Create agent with structured output
agent = create_agent(
    model=ChatOpenAI(model="gpt-4o"),
    response_format=WeatherResponse,  # AutoStrategy
    system_prompt="You are a weather forecaster. Return accurate weather data.",
)

# Invoke
result = agent.invoke({
    "messages": [{"role": "user", "content": "What's the weather in San Francisco?"}]
})

# Access structured response
weather: WeatherResponse = result["structured_response"]
print(f"Temperature: {weather.temperature_f}°F, Condition: {weather.condition}")
```

带有显式错误处理：

```python
from langchain.agents.structured_output import ToolStrategy

agent = create_agent(
    model="openai:gpt-4o",
    response_format=ToolStrategy(
        schema=WeatherResponse,
        handle_errors=True,  # Retry on validation errors
        tool_message_content="Invalid weather data format. Please provide: location, temperature_f, condition, humidity_percent.",
    ),
)
```

## 参见

- [智能体工厂](/openwiki/agent-factory.md)：创建智能体的入口点；负责模式注册和策略绑定
- [智能体执行流程](/openwiki/agent-execution.md)：运行时循环，在此过程中解析和验证结构化输出
- [LangChain 结构化输出文档](https://python.langchain.com/docs/guides/structured_output/)

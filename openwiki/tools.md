---
type: "Reference"
title: "Tools and Tool Binding"
description: "LangChain's tool system enables agents and language models to execute structured actions through schema-aware components with automatic validation, error handling, and callback integration."
tags: ["tool", "agent", "schema", "runnable", "execution"]
sources:
  - id: openwiki-source-9861ba5cf0c42c142cf732f9
    resource: repo://libs/core/langchain_core/messages/tool.py
  - id: openwiki-source-4ff475d7b00540f962384251
    resource: repo://libs/core/langchain_core/tools/base.py
  - id: openwiki-source-9c422fcb5ac12738f17d1cd1
    resource: repo://libs/core/langchain_core/tools/convert.py
  - id: openwiki-source-1ab4436ccb637ddf41e35732
    resource: repo://libs/core/langchain_core/tools/render.py
  - id: openwiki-source-80e84f93417c922f44011393
    resource: repo://libs/core/langchain_core/tools/simple.py
  - id: openwiki-source-b816e651a5890bde13cf8013
    resource: repo://libs/core/langchain_core/tools/structured.py
generated: { by: "openwiki/0.5.0", at: "2026-09-21T08:30:16.745Z" }
verified:
  - by: openwiki/0.5.0
    at: 2026-10-16T08:25:55.096Z
---
## 概述

LangChain 的工具系统使智能体和语言模型能够通过将 Python 函数与 Runnable 转换为具有模式感知能力的组件，执行结构化操作。工具构成了智能工作流的核心执行机制，提供自动参数验证、错误处理以及与回调系统的集成。

该工具生态系统包含三个层级：

1. **BaseTool**：核心抽象接口，定义了工具协议与执行语义
2. **工具类型**：针对不同输入模式的具体实现（StructuredTool、Tool）
3. **工具创建**：装饰器与工厂函数（@tool、convert_runnable_to_tool），用于从函数和 Runnable 生成工具

## BaseTool 协议与核心职责

BaseTool 是继承自 RunnableSerializable 的抽象基类，定义了所有工具的契约。每个工具都携带三个基本描述符及执行控制配置。

**必需属性：**
- `name: str` — 唯一标识符，清晰传达用途；由智能体和模型用于选择工具
- `description: str` — 人类可读文本，说明何时以及为何使用该工具；指导模型决策
- `args_schema: TypeBaseModel | dict | None` — Pydantic 模型或 JSON 模式字典，指定有效输入参数

**执行控制：**
- `return_direct: bool` — 当为 True 时，智能体在工具执行后立即停止循环（终端动作）
- `response_format: "content" | "content_and_artifact"` — 若为 "content_and_artifact"，工具必须返回一个二元组 (content, artifact)，以支持带可选附件的结构化输出
- `handle_tool_error: bool | str | Callable` — 处理 ToolException 的策略：False（重新抛出）、True（使用异常消息）、str（固定消息）、或 callable（自定义处理器）
- `handle_validation_error: bool | str | Callable` — 处理输入解析期间发生的 pydantic ValidationError 的策略

**回调与元数据：**
- `callbacks: Callbacks` — 生命周期回调（on_tool_start、on_tool_end、on_tool_error），用于追踪与监控
- `tags: list[str]` — 可选的语义标签，附加到所有调用上，用于过滤与指标统计
- `metadata: dict` — 传递给回调的自定义应用特定元数据
- `verbose: bool` — 是否记录工具执行进度

**提供商集成：**
- `extras: dict[str, Any]` — 提供商特定配置（如 Anthropic 的 cache_control、defer_loading），在工具渲染时传递给聊天模型

## 输入模式生成与验证

工具的输入验证基于从函数签名生成的 Pydantic 模型。模式生成流程处理自动推断与显式指定两种情况。

**模式来源（按优先级）：**
1. 工具装饰器或工厂函数中提供的显式 `args_schema` 参数
2. 若 `args_schema` 已是字典，则直接使用其作为 JSON 模式字典
3. 通过 `create_schema_from_function()` 从函数签名推断

**推断过程：**

当 `infer_schema=True`（默认值）时，工具会检查函数签名以生成 Pydantic 模型：

- 类型提示通过 `get_type_hints()` 提取，支持 `Annotated` 类型
- 函数文档字符串（若设置 `parse_docstring=True`）将按照 Google 风格解析，以提取参数描述
- 描述信息合并顺序为：Annotated 元数据 → 文档字符串的 Args 部分 → 无
- 被注入的参数（即使用 `InjectedToolArg`、`InjectedToolCallId` 或 `ToolRuntime` 注解的参数）会自动从发送给模型的模式中排除，但在运行时重新注入
- 保留的参数名（`run_manager`、`callbacks`、`config`）会被过滤，不包含在用户可见的模式中

**缓存机制：** `tool_call_schema` 属性为每个工具实例构建并缓存一个子模型类，排除被注入的参数。该模式类的 `model_json_schema()` 方法被修补以缓存生成的字典，防止在每次智能体循环中重复进行昂贵的重构。

**输入解析与验证：**

执行期间，工具输入由 `_parse_input()` 解析：
- 字符串输入在模式仅定义一个字段时映射到单个参数
- 字典输入通过 Pydantic 进行验证，其中 Annotated 描述提供字段文档
- 被注入的参数通过签名检查识别，并从工具元数据或调用上下文中重新注入（例如 `tool_call_id`、`ToolRuntime`）
- 验证错误根据 `handle_validation_error` 配置被捕获并处理

**注解驱动的描述：**

参数描述可来自 Annotated 字段元数据：

```python
from typing import Annotated
from pydantic import Field
from langchain_core.tools import tool

@tool
def my_function(
    query: Annotated[str, Field(description="The search query")],
    limit: Annotated[int, "Maximum number of results"] = 10
) -> str:
    return f"search: {query}"
```

同时支持 `Field(description=...)` 和直接的字符串注解，二者将合并至生成的模式中。

## ToolCall 与 ToolMessage：请求-响应协议

工具通过 ToolCall 对象调用，并以 ToolMessage 对象返回结果，从而在智能体循环中实现结构化通信。

**ToolCall（来自 messages/tool.py）：**

ToolCall 是一个 TypedDict，表示模型请求执行工具：

```python
{
    "name": "search_tool",        # Tool name to invoke
    "args": {"query": "python"},  # Validated arguments as dict
    "id": "call_abc123",          # Unique ID for pairing with response
    "type": "tool_call"           # Discriminator
}
```

多个 ToolCall 可通过 `AIMessageChunk` 流式传输并合并，流式输出产生 `ToolCallChunk` 对象，逐步构建参数的 JSON 字符串。

**ToolMessage（来自 messages/tool.py）：**

工具返回以向模型传达结果：

```python
ToolMessage(
    content="Result of the tool execution",
    tool_call_id="call_abc123",  # Must match ToolCall.id
    name="search_tool",           # Tool name (optional)
    artifact={"raw": "data"},     # Unshown to model (optional)
    status="success"              # "success" or "error"
)
```

- `artifact`：当仅向模型发送摘要时，存储完整的工具输出
- `status`：允许工具报告错误而不抛出异常（例如当 `handle_tool_error=True` 时）
- 内容支持丰富格式：纯文本或消息内容块列表（图像、JSON、搜索结果、文档等）

**ToolOutputMixin：** 一个空的混合类，用于标识工具可直接返回的自定义对象，无需强制转换为字符串。工具可返回 ToolOutputMixin 实例或其列表，绕过自动的 ToolMessage 包装。

## 执行：run() 与 arun() 方法

同步与异步执行遵循相同的生命周期：

1. **配置：** 合并来自工具配置、调用参数和 Runnable 配置的回调
2. **解析：** 通过 `_parse_input()` 和 `_to_args_and_kwargs()` 将工具输入（str/dict/ToolCall）转换为函数参数/关键字参数
3. **注入：** 若函数签名声明了相关参数，则注入运行时值（run_manager、callbacks、RunnableConfig）
4. **执行：** 在回调上下文中调用 `_run()` 或 `_arun()`，通过上下文变量传递配置
5. **格式化：** 若以 `tool_call_id` 调用，则将输出转换为 ToolMessage，保留状态和工件
6. **错误处理：** 捕获 ToolException 和 ValidationError，应用处理器策略（重新抛出、返回消息，或调用自定义处理器）

**回调生命周期：**

- `on_tool_start()`：在执行前触发，传入过滤后的输入（已移除注入参数）、工具元数据和追踪 ID
- `on_tool_end()`：成功后触发，传入格式化后的输出
- `on_tool_error()`：发生异常时触发，传入异常对象和追踪 ID

**配置传播：**

传递给 `invoke`/`ainvoke` 的 `RunnableConfig` 会被补丁处理，注入子级回调，并嵌入到工具执行上下文中，从而支持嵌套工具以及通过 `ToolRuntime` 参数访问状态/存储。

## 工具类型：Tool 与 StructuredTool

LangChain 提供了两种具体的工具实现，具有不同的输入处理语义。

**Tool（simple.py）：**
- 单输入工具，期望输入为字符串或可转换为字符串的字典
- 不需要显式定义参数模式；默认为 `{"tool_input": {"type": "string"}}`
- 用于简单函数包装器及旧版兼容性
- 在模式解析后验证恰好传入一个参数

**StructuredTool（structured.py）：**
- 多参数工具，采用显式模式驱动的解析方式
- 每个函数参数均成为独立的模式字段（除非被注入）
- 同时支持 `func`（同步）和 `coroutine`（异步）
- 若未定义异步协程，则回退至执行器进行同步调用
- 推荐作为具有多个命名参数的智能体工具使用模式

两者均继承自 `BaseTool`，并重写 `_run()` 和 `_arun()` 方法，以将调用委托给被包装的函数，同时保留配置和回调。

## 工具创建：@tool 装饰器与工厂方法

`@tool` 装饰器提供了将函数转换为工具的主要用户接口，支持多种使用模式的重载。

**装饰器形式：**

```python
# Form 1: No arguments (name from function)
@tool
def search(query: str) -> str:
    """Search the API."""
    return f"Results for {query}"

# Form 2: With parameters
@tool(description="Custom description", return_direct=True)
def calculate(expression: str) -> str:
    return str(eval(expression))

# Form 3: With explicit name
@tool("my_search")
def search(query: str) -> str:
    return query

# Form 4: With Runnable
tool_obj = tool("math_tool", my_runnable, description="...")
```

**关键行为：**

- 默认名称为 `function.__name__`，除非被覆盖
- 描述优先级：显式参数 → 函数文档字符串 → args_schema 描述
- `parse_docstring=True` 会提取 Google 风格的 Args 部分作为参数描述（并验证文档中的参数与签名一致）
- `infer_schema=True`（默认）会自动根据类型提示生成模式
- `infer_schema=False` 要求显式提供描述，并创建 Tool（字符串输入）而非 StructuredTool
- `response_format="content_and_artifact"` 期望函数返回 `(content, artifact)` 元组

**Runnable 转换：**

当装饰一个 Runnable 时，工具会自动：
- 包装同步/异步的 invoke 方法以注入回调
- 使用 Runnable 的 `input_schema` 作为工具的 `args_schema`
- 若未提供描述，则从 `input schema` 生成描述
- 对多参数 Runnable 调用 `StructuredTool.from_function()`，对字符串模式则使用 Tool

**异步支持：**

装饰器会检测协程，并创建包含 `func` 和 `coroutine` 的 StructuredTool，支持真正的异步执行。混合同步/异步模式可通过执行器回退机制正常工作。

## 模型用的模式渲染

工具通过 `render.py` 中的实用函数渲染为语言模型可用的格式：

- `render_text_description(tools: list[BaseTool]) -> str` — 返回 `"name - description\n..."` 格式，适用于提示词
- `render_text_description_and_args(tools: list[BaseTool]) -> str` — 包含参数信息：`"name - description, args: {...}"`

模型接收工具模式（tool schemas）时，采用提供方特定的格式（如 OpenAI 的 function_calling、Anthropic 的 tool_use 等），这些模式由 `function_calling.py` 工具生成，将 `tool_call_schema` 转换为带有 JSON 模式参数的 `FunctionDescription` 字典。

`tool_call_schema` 属性确保模型永远不会看到注入的参数或保留的参数名称，从而保护工具实现细节。

## 高级模式：注入参数与 ToolRuntime

工具可以通过注入参数接收运行时值，这些值不受模型控制。

**InjectedToolArg：** 用于标记应在运行时注入的参数的标记类：

```python
from typing import Annotated
from langchain_core.tools import tool, InjectedToolArg

@tool
def my_tool(
    user_query: str,
    context_var: Annotated[str, InjectedToolArg]
) -> str:
    # context_var is injected; user only provides user_query
    return f"{user_query} in {context_var}"
```

**InjectedToolCallId：** 专门用于注入 `tool_call_id` 的标记：

```python
@tool
def track_call(query: str, call_id: InjectedToolCallId) -> str:
    # call_id automatically populated with tool_call_id from invocation
    return f"Call {call_id}: {query}"
```

**ToolRuntime：** 直接注入的参数类型，提供对状态、上下文和存储的访问：

```python
from langchain_core.tools import tool, ToolRuntime

@tool
def stateful_tool(query: str, runtime: ToolRuntime) -> str:
    # Access application state, context, and LangGraph store
    state = runtime.state
    context = runtime.context
    store = runtime.store
    return f"State: {state}, Context: {context}"
```

注入参数具有以下特性：
- 不包含在发送给模型的 `tool_call_schema` 中
- 通过 `_get_injected_args_keys_from_signature()` 的签名检查识别
- 在 `_parse_input()` 过程中，从工具元数据或调用上下文中重新注入
- 通过 `_filter_injected_args()` 从回调输入中过滤掉

## 错误处理策略

工具支持灵活的错误处理机制，以在智能体循环中实现优雅恢复。

**ToolException：** 用于可控工具错误的自定义异常：

```python
from langchain_core.tools import tool, ToolException

@tool
def validate_input(value: str) -> str:
    if not value:
        raise ToolException("Value cannot be empty")
    return f"Valid: {value}"
```

**验证错误：** Pydantic 验证失败会被捕获并按 `handle_validation_error` 处理：

```python
@tool(handle_validation_error="Invalid input format")
def my_tool(count: int) -> str:
    return f"Count: {count}"

# If user passes non-integer, returns "Invalid input format" instead of raising
```

**工具错误：** 按 `handle_tool_error` 处理 `ToolException`：

```python
@tool(handle_tool_error=True)  # Use exception message
def risky_operation() -> str:
    raise ToolException("Operation failed")
    
# Returns ToolMessage with status="error", content="Operation failed"
```

自定义处理器接收异常，并返回字符串或消息内容块列表：

```python
def my_error_handler(e: ToolException) -> str:
    logger.error(f"Tool failed: {e}")
    return "Operation failed. Please try again later."

@tool(handle_tool_error=my_error_handler)
def operation() -> str:
    raise ToolException("Internal error")
```

处理后的错误在使用 `tool_call_id` 调用时返回 `ToolMessage`，且 `status="error"`，使智能体能够观察并响应失败，而不会中断循环。

## BaseToolkit：组织相关工具

对于复杂系统，可通过 `BaseToolkit` 将工具组织成工具包：

```python
from langchain_core.tools import BaseToolkit, tool

class MathToolkit(BaseToolkit):
    """Toolkit for mathematical operations."""
    
    @property
    def description(self) -> str:
        return "Tools for arithmetic and algebra"
    
    def get_tools(self) -> list[BaseTool]:
        @tool
        def add(a: int, b: int) -> int:
            return a + b
        
        @tool
        def multiply(a: int, b: int) -> int:
            return a * b
        
        return [add, multiply]

toolkit = MathToolkit()
tools = toolkit.get_tools()  # Retrieve all related tools
```

工具包支持：
- 相关功能的逻辑分组
- 条件性工具可用性（根据运行时状态返回子集）
- 动态工具生成
- 与智能体初始化流水线集成

## 将 Runnable 转换为工具

可通过 `tool()` 装饰器或 `convert_runnable_to_tool()` 函数将 Runnable 转换为工具：

```python
from langchain_core.runnables import RunnablePassthrough
from langchain_core.tools import convert_runnable_to_tool

my_runnable = RunnablePassthrough()

# Via convert function
tool_obj = convert_runnable_to_tool(
    my_runnable,
    name="passthrough",
    description="Passes input through unchanged"
)

# Via decorator
tool_obj = tool("passthrough", my_runnable)
```

转换过程包括：
- 从 `Runnable.get_input_jsonschema()` 提取 `input_schema`
- 验证模式为对象类型（多参数工具所需）
- 包装 `invoke`/`ainvoke`，将回调注入配置
- 使用包装函数委托给 `StructuredTool.from_function()`
- 对字符串输入的 Runnable 回退至 `Tool`

## 生命周期与不变性

**工具实例生命周期：**

1. **构造阶段：** 在 `__setattr__` 或 `model_copy()` 时清除模式缓存，若名称、描述或 args_schema 发生变化
2. **首次访问模式：** 构建子集模型，修补类以缓存 JSON 模式
3. **执行阶段：** 配置回调，解析输入，识别注入参数，调用函数，格式化输出
4. **序列化阶段：** 清除模式缓存（动态类无法通过引用序列化）；下次访问时重建

**模式缓存不变性：**

- 如果名称/描述/参数模式未更改，记忆化子集模型类永远不会重新生成
- 对子集类调用 `Pydantic model_json_schema()` 时，后续调用返回缓存的字典
- 缓存失效通过私有 `_TOOL_CALL_SCHEMA_FIELDS` 检查显式触发
- 在高频智能体循环下仍保持性能

**执行不变性：**

- 回调始终按顺序触发：`on_tool_start` → (`on_tool_error` | `on_tool_end`)
- 执行期间设置配置上下文，允许嵌套工具访问状态/存储
- 仅当提供 `tool_call_id` 时才进行 `ToolMessage` 包装
- 注入的参数对模型不可见，也不会出现在回调输入中
- 仅当 `handle_tool_error` 将异常转换为消息时，状态才设为 "error"

## 扩展点

**继承 `BaseTool`：**

自定义工具实现需重写：
- `_run(self, *args, **kwargs) -> Any` — 同步执行逻辑
- `_arun(self, *args, **kwargs) -> Any` — 异步执行逻辑（默认通过执行器委托给 `_run`）
- `get_input_schema()` — 重写模式来源（默认使用 `args_schema` 或根据 `_run` 签名创建）

**自定义错误处理器：**

作为可调用对象传入 `handle_tool_error` 和 `handle_validation_error`：

```python
def custom_validation_handler(e: ValidationError) -> str:
    # Extract user-friendly message from Pydantic error
    return ", ".join(f"{err['loc'][0]}: {err['msg']}" for err in e.errors())

@tool(handle_validation_error=custom_validation_handler)
def my_tool(count: int) -> str:
    return str(count)
```

**回调管理器：**

工具注入 `CallbackManager` / `AsyncCallbackManager` 以支持：
- 自定义事件处理器（日志、指标、追踪）
- 嵌套工具执行及回调传播
- `on_tool_start` / `on_tool_end` 钩子用于可观测性

工具在 `_run()` 签名中暴露 `run_manager`，以允许直接调用回调。

## 配置与操作注意事项

**保留参数名称：**

名为 `config`、`run_manager` 或 `callbacks` 的参数会从工具模式中过滤掉，因为它们与 LangChain 运行时注入冲突。如需访问运行时状态，请使用 `ToolRuntime` 注解。

**提供方额外配置：**

`extras` 字典允许传递提供方特定的配置：

```python
@tool(extras={"cache_control": {"type": "ephemeral"}})
def cached_operation(query: str) -> str:
    return query
```

聊天模型会检查 `extras`，并在渲染工具时应用提供方特定行为。

**文档字符串解析：**

当 `parse_docstring=True` 时，会解析 Google 风格的文档字符串以获取参数描述：

```python
@tool(parse_docstring=True)
def process(name: str, count: int) -> str:
    """Process items by name.
    
    Args:
        name: The item name
        count: Number of items to process
    """
    return f"{name}: {count}"
```

若 `error_on_invalid_docstring=True`，无效文档字符串（缺少 Args 部分、参数不在签名中、格式错误）将引发 `ValueError`。

**详细输出：**

设置 `verbose=True` 以记录工具执行。结合回调管理器可实现全面可观测性。

## 总结：各模式适用场景

- **`@tool` 装饰器：** 将函数转为工具的主要模式；配合类型提示可实现自动模式推断
- **`StructuredTool.from_function()`：** 装饰器语法不便利或需要程序化创建工具时的直接工厂方法
- **`Tool`（简单）：** 兼容旧版本或单字符串输入工具
- **`BaseToolkit`：** 组织相关工具或动态生成工具
- **`convert_runnable_to_tool()`：** 将现有 Runnable 包装为工具，保持一致调用方式
- **注入参数：** 在不向模型暴露的前提下共享运行时上下文（状态、存储、调用 ID）
- **自定义错误处理器：** 将 Pydantic 或工具错误转换为智能体友好的消息

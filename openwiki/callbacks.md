---
type: "Reference"
title: "Callback System and Handler Integration"
description: "Document the callback handler architecture, integration with runnables and chat models, and patterns for tracking execution events, streaming, and instrumentation."
tags: ["callbacks", "observability", "handlers", "tracing", "streaming", "langsmith"]
sources:
  - id: openwiki-source-c9313cf42f0120d86b20245f
    resource: repo://libs/core/langchain_core/callbacks/base.py
  - id: openwiki-source-b15ceb5ed590ce6a2b569ed6
    resource: repo://libs/core/langchain_core/callbacks/file.py
  - id: openwiki-source-1c233fccf5a66b84d0045366
    resource: repo://libs/core/langchain_core/callbacks/manager.py
  - id: openwiki-source-55497dabc655f803e40dc13a
    resource: repo://libs/core/langchain_core/callbacks/stdout.py
  - id: openwiki-source-c7a2c3ef4ec61c3e28011205
    resource: repo://libs/core/langchain_core/callbacks/streaming_stdout.py
  - id: openwiki-source-2685400b7962e4c90cefe9df
    resource: repo://libs/core/langchain_core/callbacks/usage.py
  - id: openwiki-source-079792f059657900794e2955
    resource: repo://libs/core/langchain_core/runnables/config.py
  - id: openwiki-source-bfd8b1aa6ad00852a2e99762
    resource: repo://libs/core/langchain_core/tracers/context.py
generated: { by: "openwiki/0.5.0", at: "2026-09-28T08:35:20.640Z" }
verified:
  - by: openwiki/0.5.0
    at: 2026-10-16T08:25:55.096Z
---
## 概述

LangChain 的回调系统提供了一种统一机制，用于可观测性、日志记录和追踪所有操作——包括 LLM 调用、链执行、工具使用以及检索。回调使开发者能够监控执行流程、收集指标、流式输出，并与 LangSmith 等外部可观测性平台集成，而无需修改核心应用代码。

该系统基于分层运行结构，通过运行 ID 追踪父-子关系，从而支持完整的追踪树。每次操作都会生成事件，并分发给一个或多个已注册的处理器，这些处理器可同步或异步处理事件。

## 架构

### 核心组件

**BaseCallbackHandler** (`repo://libs/core/langchain_core/callbacks/base.py#L496-L546`) 是所有回调实现的基础类。它继承自多个混入（mixin），定义了不同操作类型的事件方法：

- **LLMManagerMixin**: `on_llm_new_token`, `on_llm_end`, `on_llm_error`, `on_stream_event`
- **ChainManagerMixin**: `on_chain_end`, `on_chain_error`, `on_agent_action`, `on_agent_finish`
- **ToolManagerMixin**: `on_tool_end`, `on_tool_error`
- **RetrieverManagerMixin**: `on_retriever_end`, `on_retriever_error`
- **RunManagerMixin**: `on_text`, `on_retry`, `on_custom_event`
- **CallbackManagerMixin**: `on_llm_start`, `on_chat_model_start`, `on_chain_start`, `on_tool_start`, `on_retriever_start`

每个处理器还支持 `raise_error` 和 `run_inline` 属性，用于控制错误传播和执行上下文。

**BaseCallbackManager** (`repo://libs/core/langchain_core/callbacks/base.py#L1004-L1227`) 负责管理处理器集合及其生命周期。它维护以下内容：

- **handlers**: 当前操作的非继承型回调
- **inheritable_handlers**: 传递给子操作的回调
- **tags**: 用于过滤和组织运行的标签（可继承）
- **metadata**: 可序列化为 JSON 的上下文信息（可继承）
- **parent_run_id**: 指向父操作的引用，用于层级结构

**CallbackManager**（同步，`repo://libs/core/langchain_core/callbacks/manager.py#L1377-L1726`）和 **AsyncCallbackManager**（异步，`repo://libs/core/langchain_core/callbacks/manager.py#L1859`）是主要实现，负责将事件分发给处理器。它们提供 `on_llm_start`、`on_chat_model_start`、`on_chain_start`、`on_tool_start` 和 `on_retriever_start` 方法，返回与特定操作绑定的专用运行管理器。

**运行管理器**（Run Managers）由开始事件返回，提供绑定上下文的方法以处理结束和错误事件：

- **CallbackManagerForLLMRun**: `on_llm_new_token`, `on_llm_end`, `on_llm_error`, `on_stream_event`
- **CallbackManagerForChainRun**: `on_chain_end`, `on_chain_error`, `on_agent_action`, `on_agent_finish`
- **CallbackManagerForToolRun**: `on_tool_end`, `on_tool_error`
- **CallbackManagerForRetrieverRun**: `on_retriever_end`, `on_retriever_error`

每个运行管理器都持有操作的 **run_id** 和 **parent_run_id**，确保追踪层级结构得以保留。

### 事件分发

回调系统使用两个分发函数：

**handle_event** (`repo://libs/core/langchain_core/callbacks/manager.py#L285-L368`, 同步) 会遍历所有处理器，并调用对应的事件方法。它处理以下内容：

- 根据忽略条件跳过处理器（例如 `ignore_llm`、`ignore_chain`）
- 捕获并记录异常（尊重 `raise_error` 设置）
- 收集异步协程，并通过执行器池或事件循环运行
- 当未实现时，将 `on_chat_model_start` 转换为 `on_llm_start` 的回退机制

**ahandle_event** (`repo://libs/core/langchain_core/callbacks/manager.py#L453-L488`, 异步) 将内联（顺序）和非内联（并发）处理器分开，使用 `asyncio.gather()` 实现并行。

**shielded** 装饰器 (`repo://libs/core/langchain_core/callbacks/manager.py#L221-L254`) 在异步处理器中发生取消时保留上下文变量，避免任务取消死锁。

## 内置处理器

### StdOutCallbackHandler

**位置**: `repo://libs/core/langchain_core/callbacks/stdout.py`

将可读性良好的消息打印到标准输出，用于链的进入/退出、智能体动作、工具观察结果以及自由文本。适用于交互式 CLI 应用程序。

```python
from langchain_core.callbacks.stdout import StdOutCallbackHandler

handler = StdOutCallbackHandler(color=None)
chain.invoke(input, config={"callbacks": [handler]})
# > Entering new SequentialChain chain...
# > Finished chain.
```

方法重写：
- `on_chain_start`: 打印 "Entering new {name} chain"
- `on_chain_end`: 打印 "Finished chain"
- `on_agent_action`: 打印动作日志
- `on_tool_end`: 打印工具观察结果，支持可选前缀
- `on_text`: 打印任意文本

### FileCallbackHandler

**位置**: `repo://libs/core/langchain_core/callbacks/file.py`

将回调事件写入文件，支持可选着色。同时支持上下文管理器（推荐）和直接实例化两种模式。

```python
from langchain_core.callbacks.file import FileCallbackHandler

# Context manager (recommended)
with FileCallbackHandler("output.txt") as handler:
    chain.invoke(input, config={"callbacks": [handler]})

# Direct instantiation (deprecated)
handler = FileCallbackHandler("output.txt")
try:
    chain.invoke(input, config={"callbacks": [handler]})
finally:
    handler.close()
```

该处理器默认以追加模式打开文件，并为链、智能体和工具写入格式化输出。

### StreamingStdOutCallbackHandler

**位置**: `repo://libs/core/langchain_core/callbacks/streaming_stdout.py`

在流式生成过程中，将 LLM 的每个令牌实时流式输出到标准输出，实现实时输出可视化。

```python
from langchain_core.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

handler = StreamingStdOutCallbackHandler()
llm.stream("Hello", config={"callbacks": [handler]})  # Prints tokens as they arrive
```

仅适用于通过 `on_llm_new_token` 支持流式的 LLM。

### UsageMetadataCallbackHandler

**位置**: `repo://libs/core/langchain_core/callbacks/usage.py`

聚合跨 LLM 调用的令牌使用元数据，按模型名称收集输入/输出令牌数及成本信息。使用内部锁保证线程安全。

```python
from langchain_core.callbacks.usage import UsageMetadataCallbackHandler

callback = UsageMetadataCallbackHandler()
llm_1.invoke("Hello", config={"callbacks": [callback]})
llm_2.invoke("Hello", config={"callbacks": [callback]})
print(callback.usage_metadata)
# {"openai:gpt-4": UsageMetadata(...), "anthropic:claude-3": UsageMetadata(...)}
```

## LangSmith 集成

LangSmith 是 LangChain 的可观测性平台。通过环境变量或上下文管理器启用后，所有回调事件将自动发送至 LangSmith，用于可视化与分析。

### 自动追踪

**通过环境变量启用**：

```bash
export LANGCHAIN_TRACING_V2=true
export LANGSMITH_API_KEY=<your-key>
export LANGSMITH_PROJECT=<project-name>
```

当启用 `LANGCHAIN_TRACING_V2` 时，回调管理器会自动创建一个 **LangChainTracer** (`repo://libs/core/langchain_core/tracers/langchain.py`) 并将其注册为处理器。该追踪器：

- 捕获所有运行生命周期事件（开始、结束、错误）
- 使用 `parent_run_id` 构建层级化的追踪树
- 将追踪数据发送至 LangSmith 后端
- 提供运行 URL 和唯一运行 ID

**上下文管理器方式**：

```python
from langchain_core.tracers.context import tracing_v2_enabled

with tracing_v2_enabled(project_name="my_project") as tracer:
    chain.invoke("hello")
    run_url = tracer.get_run_url()
    print(f"View trace at: {run_url}")
```

**自动追踪回调注入**（`repo://libs/core/langchain_core/tracers/context.py#L105-L130`）通过 `_get_trace_callbacks` 实现，其流程如下：

1. 通过 `_tracing_v2_is_enabled()` 检查是否启用了追踪
2. 创建或复用现有的 LangChainTracer
3. 将其添加到回调管理器中，避免重复
4. 返回配置好的回调管理器以供操作使用

### 运行数据收集

用于程序化访问追踪数据，无需依赖 LangSmith：

```python
from langchain_core.tracers.context import collect_runs

with collect_runs() as runs_cb:
    chain.invoke("hello")
    for run in runs_cb.traced_runs:
        print(f"Run ID: {run.id}, Type: {run.run_type}")
```

**RunCollectorCallbackHandler** 会将所有 Run 对象收集到一个列表中，支持离线分析。

## 配置与注册

### 通过 RunnableConfig

Runnable 通过 `config` 参数接收回调：

```python
from langchain_core.runnables.config import RunnableConfig

config = RunnableConfig(
    callbacks=[handler1, handler2],
    tags=["production", "v1"],
    metadata={"user_id": "123", "session": "abc"},
    run_name="my_run",
)
chain.invoke(input, config=config)
```

**RunnableConfig** TypedDict（`repo://libs/core/langchain_core/runnables/config.py#L57-L129`）支持以下字段：

- **callbacks**：回调处理器列表或回调管理器
- **tags**：可继承的标签，用于过滤
- **metadata**：可继承的上下文信息（需 JSON 可序列化）
- **run_name**：覆盖默认的操作名称
- **run_id**：显式运行标识符（UUID）

### 管理器配置

静态方法 `configure` 用于创建一个完全初始化的回调管理器：

```python
from langchain_core.callbacks.manager import CallbackManager

manager = CallbackManager.configure(
    inheritable_callbacks=[tracer],
    local_callbacks=[stdout_handler],
    inheritable_tags=["app"],
    local_tags=["expensive_op"],
    inheritable_metadata={"env": "prod"},
    verbose=True,
)
chain.invoke(input, config={"callbacks": manager})
```

### 处理器管理

回调管理器 API 支持动态注册处理器：

```python
manager = CallbackManager(handlers=[])
manager.add_handler(handler, inherit=True)   # Add to both handler lists
manager.add_handler(handler, inherit=False)  # Add only to current operation
manager.remove_handler(handler)
manager.set_handlers([h1, h2], inherit=True) # Replace all handlers
```

### 标签与元数据管理

```python
manager.add_tags(["tag1", "tag2"], inherit=True)
manager.remove_tags(["tag1"])

manager.add_metadata({"key": "value"}, inherit=True)
manager.remove_metadata(["key"])
```

标签和元数据通过事件方法中的关键字参数传递给所有回调。

## 回调生命周期与执行

### 运行层级结构

操作形成父子层级关系，每个子操作都会收到一个 **parent_run_id**，用于关联父级：

```
LLMResult -> parent_run_id=run_id_chain -> parent_run_id=run_id_outer
```

这使得 LangSmith 能够重建完整的调用树。**ParentRunManager**（`repo://libs/core/langchain_core/callbacks/manager.py#L599-L619`）提供 `get_child()` 方法，用于创建继承标签和可继承处理器的子回调管理器。

### 继承规则

- **inheritable_handlers**：传递给所有子操作
- **handlers**：仅用于当前操作（不会传递给子级）
- **inheritable_tags**：传递给子级，并累积
- **inheritable_metadata**：传递给子级，并合并

这允许全局处理器（如 LangSmith 追踪器）存在，同时支持操作级别的特定处理器。

### 异步执行控制

处理器上的 `run_inline` 属性决定其执行方式：

- **run_inline=True**：在当前异步上下文中执行（顺序执行）
- **run_inline=False**：调度到线程池或并发任务中

```python
class MyInlineHandler(BaseCallbackHandler):
    run_inline = True  # Always runs in caller's context

class MyConcurrentHandler(BaseCallbackHandler):
    run_inline = False  # Runs in executor or concurrently
```

内联处理器会遵循调用者的执行上下文（ContextVar），而非内联处理器则使用 `copy_context().run()` 来跨线程边界保留上下文变量。

## 线程安全与上下文变量

### 上下文保持

回调系统使用 Python 的 `contextvars` 模块来跨同步/异步边界传播上下文：

- **tracing_v2_callback_var**：当前 LangChainTracer（若启用了追踪）
- **run_collector_var**：当前 RunCollectorCallbackHandler（若正在收集）
- **var_child_runnable_config**：来自父级的子 Runnable 配置

当 `handle_event` 从同步上下文运行异步协程时，会通过 `copy_context().run()` 保留上下文变量。

### 线程安全

如果多个处理器会并发访问，处理器应实现线程安全的状态管理。**UsageMetadataCallbackHandler** 使用 `threading.Lock()` 进行聚合。

## 流式传输与令牌事件

### LLM 令牌流式传输

当 LLM 支持流式传输时，回调管理器会为每个令牌调用 `on_llm_new_token`：

```python
class TokenCollector(BaseCallbackHandler):
    def __init__(self):
        self.tokens = []
    
    def on_llm_new_token(self, token, **kwargs):
        self.tokens.append(token)

collector = TokenCollector()
llm.stream("Hello", config={"callbacks": [collector]})
print("".join(collector.tokens))
```

`chunk` 参数提供完整的 `GenerationChunk` 或 `ChatGenerationChunk` 对象，包含额外元数据。

### 协议事件（v3）

对于使用 v3 协议的原生流式传输提供方：

```python
class ProtocolEventHandler(BaseCallbackHandler):
    def on_stream_event(self, event, **kwargs):
        # event is MessagesData: message-start, content-block-start, etc.
        print(f"Event type: {event['type']}")
```

协议事件的触发粒度比 `on_llm_new_token` 更细，并具有明确的生命周期边界。

## 错误处理

### 异常传播

默认情况下，处理器异常会被记录并吞没：

```python
class BuggyHandler(BaseCallbackHandler):
    def on_chain_start(self, **kwargs):
        raise ValueError("Oops!")  # Logged but doesn't stop execution

handler = BuggyHandler()
chain.invoke(input, config={"callbacks": [handler]})  # Still runs
```

若需强制执行严格的错误检查：

```python
handler = BuggyHandler()
handler.raise_error = True
chain.invoke(input, config={"callbacks": [handler]})  # Raises ValueError
```

### 忽略条件

处理器可选择性跳过特定事件类型：

```python
class LLMOnlyHandler(BaseCallbackHandler):
    @property
    def ignore_chain(self):
        return True  # Skip all chain events
    
    @property
    def ignore_retriever(self):
        return True  # Skip all retriever events
```

可用属性：`ignore_llm`、`ignore_chain`、`ignore_agent`、`ignore_retriever`、`ignore_retry`、`ignore_chat_model`、`ignore_custom_event`。

## 自定义事件分发

### on_custom_event

用于处理超出 LLM/链/工具范围的应用特定事件：

```python
manager = CallbackManager(handlers=[custom_handler])
manager.on_custom_event(
    name="user_interaction",
    data={"user_id": 123, "action": "clicked_button"},
)
```

处理器接收：

```python
def on_custom_event(self, name, data, run_id, tags, metadata, **kwargs):
    print(f"Event: {name}, Data: {data}")
```

## 链组

用于将多个独立调用组合为单一逻辑操作：

```python
from langchain_core.callbacks.manager import trace_as_chain_group

with trace_as_chain_group("data_processing", tags=["batch"]) as manager:
    result1 = llm.invoke("query1", config={"callbacks": manager})
    result2 = chain.invoke(input2, config={"callbacks": manager})
    # Both treated as a single chain in LangSmith trace
```

管理器会跟踪完成状态，并在退出时调用父级的 `on_chain_end` 或 `on_chain_error`。

## 自定义处理器实现

### BaseCallbackHandler 子类模式

通过继承 `BaseCallbackHandler` 并重写相关事件方法来实现自定义处理器：

```python
from langchain_core.callbacks import BaseCallbackHandler
from typing import Any

class CustomMetricsHandler(BaseCallbackHandler):
    """Custom handler for collecting application metrics."""
    
    def __init__(self):
        self.metrics = {}
        self.run_inline = True  # Execute in caller's context
    
    @property
    def ignore_llm(self) -> bool:
        """Skip LLM events for this handler."""
        return False
    
    def on_llm_start(self, serialized: dict[str, Any], prompts: list[str], 
                     run_id, tags=None, metadata=None, **kwargs: Any) -> None:
        """Called when LLM starts."""
        self.metrics[str(run_id)] = {"start_time": time.time(), "prompt": prompts[0]}
    
    def on_llm_end(self, response, run_id, **kwargs: Any) -> None:
        """Called when LLM completes."""
        if str(run_id) in self.metrics:
            self.metrics[str(run_id)]["end_time"] = time.time()
            self.metrics[str(run_id)]["duration"] = (
                self.metrics[str(run_id)]["end_time"] - 
                self.metrics[str(run_id)]["start_time"]
            )
    
    def on_llm_error(self, error: BaseException, run_id, **kwargs: Any) -> None:
        """Called when LLM errors."""
        if str(run_id) in self.metrics:
            self.metrics[str(run_id)]["error"] = str(error)
```

关键模式：

1. **设置 `run_inline`** 以控制执行上下文（同步 vs 异步线程池）
2. **实现 `ignore_*` 属性** 以跳过无关事件类型
3. **使用 `**kwargs` 处理可选参数** 以保证向前兼容
4. **在聚合或关联事件时保留 `run_id`** 以维护层级关系
5. **利用传入 kwargs 的元数据/标签** 实现上下文和过滤功能

## 最佳实践

1. **文件处理器使用上下文管理器**：`FileCallbackHandler` 应配合 `with` 语句使用，以确保正确清理。

2. **通过 inheritable_callbacks 注册全局处理器**：使用 `CallbackManager.configure(inheritable_callbacks=[...])` 使处理器在所有位置生效。

3. **生产环境启用 LangSmith**：设置 `LANGCHAIN_TRACING_V2=true` 和 `LANGSMITH_API_KEY` 以自动收集追踪数据。

4. **使用标签进行过滤**：添加语义标签（如 "production"、"experiment"、"expensive"）以便在 LangSmith 中筛选运行记录。

5. **包含元数据上下文**：在元数据中嵌入用户 ID、会话 ID 和环境信息，以提升可观测性。

6. **实现线程安全的处理器**：若处理器访问共享状态，应使用锁或线程局部存储。

7. **优雅地处理异常**：仅在关键处理器中设置 `raise_error=True`；其他处理器应记录错误并继续执行。

8. **尊重忽略条件**：如果你的处理器仅关注 LLM 调用，请设置 `ignore_chain=True` 以跳过无关事件。

9. **使用流式处理器实现实时反馈**：`StreamingStdOutCallbackHandler` 支持交互式的逐令牌输出。

10. **收集 token 使用量**：使用 `UsageMetadataCallbackHandler` 跟踪多模型应用中的成本。

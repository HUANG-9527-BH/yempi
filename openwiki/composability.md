---
type: "Concept"
title: "Composability and LCEL Chains"
description: "How Runnable components compose through LCEL operators, creating reusable workflows with automatic async, batch, and streaming support."
tags: ["composability", "LCEL", "runnables", "chaining", "operators"]
sources:
  - id: openwiki-source-a1981e868973f6fd7f71e12e
    resource: repo://libs/core/langchain_core/runnables/base.py
  - id: openwiki-source-48e94bbe49ab4f33ba87e9cb
    resource: repo://libs/core/langchain_core/runnables/branch.py
  - id: openwiki-source-f9f4c1dc4f9cdf80d824ce15
    resource: repo://libs/core/langchain_core/runnables/fallbacks.py
  - id: openwiki-source-de6c904bd0171642bd50f6d9
    resource: repo://libs/core/langchain_core/runnables/router.py
generated: { by: "openwiki/0.5.0", at: "2026-09-28T08:35:20.640Z" }
verified:
  - by: openwiki/0.5.0
    at: 2026-09-28T08:35:20.640Z
---
## 概述

可组合性是 LangChain 的 Runnable 协议的核心特性：能够以声明式方式链式连接、并行执行以及条件路由组件。每个组合后的链都会自动继承同步（`invoke`）、异步（`ainvoke`）、批量（`batch`/`abatch`）和流式（`stream`/`astream`）能力——并具备高效的优化。

两种主要的组合原语是 **`RunnableSequence`**（通过 `|` 运算符实现的顺序链式）和 **`RunnableParallel`**（通过字典语法实现的并行执行）。条件路由则通过 **`RunnableBranch`** 和 **`RouterRunnable`** 实现。

## 顺序组合：`|` 运算符

**管道运算符**（`|`）按顺序链式连接 Runnables，每一步的输出成为下一步的输入。这是最常见的组合模式。

```python
from langchain_core.runnables import RunnableLambda

add_one = RunnableLambda(lambda x: x + 1)
mul_two = RunnableLambda(lambda x: x * 2)

sequence = add_one | mul_two
sequence.invoke(1)  # (1 + 1) * 2 = 4
```

`|` 运算符创建一个 **`RunnableSequence`**，其特性包括：
- 按顺序调用每一步，并将输出传递给下一步输入
- 为提高效率而扁平化嵌套的序列
- 若所有步骤均支持 `transform` 方法，则自动保留流式属性
- 支持同步与异步执行

### 数据流

```
Input → Step 1 → Step 2 → Step 3 → Output
```

当一个字典被传入序列时，它会转变为一个 **`RunnableParallel`**：

```python
sequence = add_one | {
    "mul_2": RunnableLambda(lambda x: x * 2),
    "mul_5": RunnableLambda(lambda x: x * 5),
}
sequence.invoke(1)  # {'mul_2': 4, 'mul_5': 10}
```

## 并行组合：使用 `+` 和字典语法进行分支

并行执行在同一输入上并发调用多个 Runnables。这可以通过序列中的字典字面量或直接使用 **`RunnableParallel`** 实现。

### 字典字面量语法

```python
from langchain_core.runnables import RunnableLambda, RunnableParallel

add_one = RunnableLambda(lambda x: x + 1)
mul_two = RunnableLambda(lambda x: x * 2)
mul_three = RunnableLambda(lambda x: x * 3)

# Dict syntax creates a RunnableParallel
sequence = add_one | {
    "mul_2": mul_two,
    "mul_3": mul_three,
}
sequence.invoke(1)
# Output: {'mul_2': 4, 'mul_3': 6}
```

### 显式使用 RunnableParallel

```python
parallel = RunnableParallel(
    mul_2=mul_two,
    mul_3=mul_three,
)
parallel.invoke(2)
# Output: {'mul_2': 4, 'mul_3': 6}
```

### 并发执行

- **`RunnableParallel`** 使用 `atee`（异步）或 `safetee`（同步）为每个分支创建独立的输入副本
- 每个分支并发执行，块按完成顺序产出
- 对于异步流式处理，任务通过 `asyncio.wait(return_when=FIRST_COMPLETED)` 管理，一旦任一分支产生数据块即刻发出输出
- 最终结果是一个包含所有分支输出的字典

## 批量处理：对多个输入的并行调用

批量处理通过管道高效地处理多个输入。与并行分支不同，批量处理是对每个输入并行应用**相同的序列**。

### 同步批量

```python
sequence = add_one | mul_two
results = sequence.batch([1, 2, 3])
# [4, 6, 8]  # Each input processed in parallel via thread pool
```

### 异步批量

```python
results = await sequence.abatch([1, 2, 3])
# [4, 6, 8]
```

### 实现方式

- 默认 `batch` 使用线程池执行器，通过 `get_executor_for_config` 获取
- `abatch` 使用 `asyncio.gather`，并通过 `max_concurrency` 控制并发度
- 序列中的每一步独立对输入进行批量处理
- **`RunnableSequence`** 按顺序在每一步上调用 `batch`，并将输出传递给下一步

## 流式输出：逐令牌生成

流式输出在生成过程中逐块发出结果，使大语言模型（LLM）及其他顺序生成器能够实现实时响应。

### 流式方法

```python
for chunk in sequence.stream(1):
    print(chunk)  # Intermediate outputs as they become available
```

### Astream 方法（异步）

```python
async for chunk in sequence.astream(1):
    print(chunk)  # Non-blocking iteration
```

### 流式管道

一个 **`RunnableSequence`** 会保持流式属性：
- 若所有步骤都实现了 `transform`（即处理 `Iterator[Input] → Iterator[Output]`），则流式传输贯穿整个管道
- 若任意步骤不支持 `transform`，流式将阻塞直到该步骤完成，之后恢复
- **`RunnableLambda`** 默认不实现 `transform`；如需自定义流式逻辑，请使用 **`RunnableGenerator`**

### 示例：提示词 → 模型 → 解析器

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

prompt = ChatPromptTemplate.from_template("What is {topic}?")
model = ChatOpenAI()
parser = StrOutputParser()

chain = prompt | model | parser

# Stream tokens as the model generates them
for chunk in chain.stream({"topic": "composability"}):
    print(chunk, end="", flush=True)
```

在此链中：
1. `ChatPromptTemplate` 将输入字典格式化为字符串提示词
2. `ChatOpenAI` 以流式方式接收并逐个传递 API 返回的标记（token）
3. `StrOutputParser` 原样传递所有标记，不作修改

标记在整个流程中端到端流动，无需等待完整响应。

## 条件路由：RunnableBranch 与 RouterRunnable

条件逻辑根据谓词将输入路由到不同的分支。

### RunnableBranch：基于谓词的路由

**`RunnableBranch`** 按顺序评估条件，并执行第一个匹配的分支：

```python
from langchain_core.runnables import RunnableBranch, RunnableLambda

branch = RunnableBranch(
    (lambda x: isinstance(x, int), RunnableLambda(lambda x: x * 2)),
    (lambda x: isinstance(x, str), RunnableLambda(lambda x: x.upper())),
    RunnableLambda(lambda x: "unknown"),
)

branch.invoke(5)        # 10
branch.invoke("hello")  # "HELLO"
branch.invoke(None)     # "unknown"
```

条件按顺序评估；首个真值结果将选择其对应的 Runnable。若无条件匹配，则执行默认分支。

### RouterRunnable：基于键的路由

**`RouterRunnable`** 根据输入中的字符串键进行路由：

```python
from langchain_core.runnables.router import RouterRunnable

add = RunnableLambda(lambda x: x + 1)
square = RunnableLambda(lambda x: x ** 2)

router = RouterRunnable(runnables={"add": add, "square": square})
router.invoke({"key": "square", "input": 3})  # 9
router.invoke({"key": "add", "input": 3})     # 4
```

输入是一个字典，包含 `"key"`（指定要路由到的 Runnable）和 `"input"`（待处理的数据）。

## 使用 RunnablePassthrough 进行组合

**`RunnablePassthrough`** 不改变输入，或仅添加额外键，适用于在并行分支中保留上下文：

```python
from langchain_core.runnables import RunnablePassthrough

chain = (
    RunnableLambda(lambda x: x + 1)
    | {
        "original": RunnablePassthrough(),
        "modified": RunnableLambda(lambda x: x * 2),
    }
)

chain.invoke(5)
# {'original': 6, 'modified': 12}
```

此处，透传机制保留了中间结果，供其他分支复用。

## 异步等价方法

每个方法都有对应的异步版本：

| 同步 | 异步 |
|------|-------|
| `invoke(input)` | `ainvoke(input)` |
| `batch(inputs)` | `abatch(inputs)` |
| `stream(input)` | `astream(input)` |
| `transform(Iterator[Input])` | `atransform(AsyncIterator[Input])` |

异步方法与回调系统集成，并通过 `asyncio.gather` 实现并发感知的批处理。

## 变量绑定与上下文传递

在组合链中，数据连同执行上下文沿步骤流动。每一步都接收前一步的输出作为输入。

### 上下文传播

当链被调用时，**`RunnableSequence`** 会创建一个用于追踪的回调层级结构：
- 每一步均使用 `run_manager.get_child(f"seq:step:{i + 1}")` 标记为子运行
- 回调、标签和元数据通过 `RunnableConfig` 在链中传递
- `patch_config` 会在保持父级上下文的同时，更新每一步的配置

```python
from langchain_core.runnables import RunnableLambda

# Context flows through each step
step1 = RunnableLambda(lambda x: x + 1)
step2 = RunnableLambda(lambda x: x * 2)
chain = step1 | step2

# Invoke with tracing config
result = chain.invoke(
    5, 
    config={
        "run_name": "my_chain",
        "callbacks": [my_tracer],
        "tags": ["prod"],
    }
)
# Each step runs with inherited config while reporting to callbacks
```

### 字典组合与键选择

在序列中使用字典语法时，每个字典键都会成为独立的分支上下文：

```python
chain = step1 | {
    "result_a": step2,
    "result_b": step3,
}

# Output combines results from both branches
output = chain.invoke(input)  # {'result_a': ..., 'result_b': ...}
```

每个分支（`result_a`、`result_b`）在回调追踪中均表现为独立的子运行。

## 失败降级模式

失败降级通过在主组件失败时切换到备用组件来提供容错能力，而不会重试同一组件。

### RunnableWithFallbacks 机制

**`RunnableWithFallbacks`** 包装一个主 Runnable 和一组降级项：

1. 首先尝试执行主 Runnable
2. 当抛出与 `exceptions_to_handle` 匹配的异常时，按顺序尝试每个降级项
3. 第一个成功执行的降级项将停止后续尝试
4. 若所有降级项均失败，则抛出首次捕获的异常

关键属性：
- **`runnable`**：要执行的主 Runnable
- **`fallbacks`**：降级 Runnable 的有序列表
- **`exceptions_to_handle`**：触发降级的异常类型元组（默认：所有异常）
- **`exception_key`**：可选字符串键，用于将捕获的异常传递给降级项的输入字典

### 组件级别的回退机制

```python
from langchain_core.runnables import RunnableLambda

primary_llm = ChatOpenAI(model="gpt-4")
fallback_llm = ChatAnthropic(model="claude-3-sonnet")

resilient_llm = primary_llm.with_fallbacks(
    [fallback_llm],
    exceptions_to_handle=(APIConnectionError,),
)

output = resilient_llm.invoke("What is composability?")
# Uses primary_llm; falls back to fallback_llm if APIConnectionError occurs
```

### 多重回退与提供者故障转移

回退项按顺序尝试，直到其中一个成功为止：

```python
model = ChatOpenAI().with_fallbacks([
    ChatAnthropic(),        # Try second
    ChatClaude(),          # Try third
    ChatCohere(),          # Try fourth
    RunnableLambda(default_response),  # Final fallback
])
```

链会依次尝试每个回退项，直到有一个返回成功，或所有回退均耗尽。

### 链级别的回退机制

```python
# Construct a chain with fallback
chain_with_fallback = (
    prompt 
    | resilient_llm 
    | parser
).with_fallbacks([
    RunnableLambda(lambda x: "Service unavailable")
])

output = chain_with_fallback.invoke({"topic": "composability"})
# If the entire chain fails, returns fallback response
```

### 异常传递至回退项

使用 `exception_key` 可将捕获的异常作为输入的一部分传递给回退项：

```python
def fallback_with_context(input_dict):
    error = input_dict.get("error")
    if isinstance(error, APIConnectionError):
        return "API is currently unavailable. Please try again later."
    return "An unexpected error occurred."

chain = (
    prompt | llm | parser
).with_fallbacks(
    [RunnableLambda(fallback_with_context)],
    exception_key="error",  # Pass exception under "error" key
)

# Input becomes {"topic": "...", "error": <caught exception>}
```

这要求所有 Runnable（主项和回退项）都接受字典作为输入。

## 重试模式

重试逻辑在失败时自动重新调用 Runnable，支持可配置的退避策略和异常过滤。

### 组件级别的重试

```python
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables.retry import ExponentialJitterParams

llm = ChatOpenAI(model="gpt-4")

resilient_llm = llm.with_retry(
    retry_if_exception_type=(APIConnectionError, TimeoutError),
    max_attempt_number=3,
    wait_exponential_jitter=True,
    exponential_jitter_params={"initial": 1, "max": 10},
)

output = resilient_llm.invoke("What is composability?")
# Retries up to 3 times on transient errors with exponential backoff
```

### RunnableRetry 实现

**`RunnableRetry`** 包装任意 Runnable，并使用 `tenacity` 应用重试逻辑：

- **`retry_exception_types`**：要重试的异常类型元组（默认：所有异常）
- **`max_attempt_number`**：最大重试次数（默认：3）
- **`wait_exponential_jitter`**：启用带抖动的指数退避（默认：True）
- **`exponential_jitter_params`**：自定义退避参数（`initial`、`max`、`exp_base`、`jitter`）

重试操作会在回调中被追踪，标签如 `retry:attempt:2` 可在追踪中可见。

### 重试策略：瞬态错误与致命错误

最佳实践：仅对瞬态错误进行重试，而非致命错误：

```python
model = ChatOpenAI().with_retry(
    retry_if_exception_type=(
        APIConnectionError,    # Transient: network issues
        TimeoutError,          # Transient: slow API
        RateLimitError,        # Transient: quota exceeded
    ),
    max_attempt_number=5,
    exponential_jitter_params={"initial": 0.5, "max": 30},
)

# Do NOT retry on invalid input or authentication errors—these are fatal
```

### 链级别的重试

```python
chain = (
    prompt 
    | llm.with_retry(max_attempt_number=3)  # Retry at LLM level
    | parser
)

# Better than retrying the whole chain, which wastes time on non-failing steps
```

## 链接模式

### 常见模式：提示词 → 模型 → 解析器

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

chain = (
    ChatPromptTemplate.from_template("What is {topic}?")
    | ChatOpenAI()
    | StrOutputParser()
)

# Single invoke
output = chain.invoke({"topic": "LLMs"})

# Batch process
outputs = chain.batch([{"topic": "LLMs"}, {"topic": "Vectors"}])

# Stream tokens
for chunk in chain.stream({"topic": "LLMs"}):
    print(chunk, end="", flush=True)
```

### 分叉/汇聚：并行处理

```python
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

chain = (
    RunnablePassthrough()
    | {
        "summary": RunnableLambda(summarize),
        "entities": RunnableLambda(extract_entities),
        "sentiment": RunnableLambda(analyze_sentiment),
    }
)

result = chain.invoke(text)
# {'summary': '...', 'entities': [...], 'sentiment': 'positive'}
```

### 条件执行

```python
from langchain_core.runnables import RunnableBranch

route_logic = RunnableBranch(
    (lambda x: "math" in x.lower(), math_chain),
    (lambda x: "code" in x.lower(), code_chain),
    general_chain,
)

output = route_logic.invoke("How do I calculate factorial?")
```

## 类型安全与模式推断

链会从其组件中推断输入和输出类型：

```python
sequence = add_one | mul_two

# Access inferred schemas
print(sequence.input_schema)   # Pydantic model for input
print(sequence.output_schema)  # Pydantic model for output
print(sequence.input_schema.model_json_schema())
```

这使得无需显式类型注解即可实现验证和文档生成。

## 优化与扁平化

**`RunnableSequence`** 自动扁平化嵌套序列：

```python
# These are equivalent:
chain1 = step1 | step2 | step3
chain2 = step1 | (step2 | step3)
chain3 = (step1 | step2) | step3
```

所有形式最终生成一个单一的扁平序列，包含步骤 `[step1, step2, step3]`，避免不必要的嵌套开销。

## 序列化与调试

组合型链支持通过 LangChain 序列化系统进行序列化，实现：
- **持久化**：保存和加载链
- **追踪**：通过 LangSmith 自动集成回调以用于调试
- **检查**：使用 `get_graph()` 可可视化链结构

启用调试输出：

```python
from langchain_core.globals import set_debug

set_debug(True)  # Print intermediate results
chain.invoke(input)

# Or use callbacks:
from langchain_core.tracers import ConsoleCallbackHandler

chain.invoke(input, config={"callbacks": [ConsoleCallbackHandler()]})
```

## 扩展：自定义 Runnable

实现 **`Runnable`** 接口以创建自定义组件：

```python
from langchain_core.runnables import Runnable, RunnableConfig
from typing import Iterator

class CustomRunnable(Runnable[str, int]):
    def invoke(self, input: str, config: RunnableConfig | None = None) -> int:
        return len(input)
    
    async def ainvoke(self, input: str, config: RunnableConfig | None = None) -> int:
        return len(input)
    
    def stream(self, input: str, config: RunnableConfig | None = None) -> Iterator[int]:
        # For streaming support, implement transform
        for char in input:
            yield 1
    
    async def astream(self, input: str, config: RunnableConfig | None = None):
        for char in input:
            yield 1

# Immediately composable
chain = CustomRunnable() | another_step
```

自定义 Runnable 会自动兼容所有组合操作符。

## 总结表格

| 操作符 | 效果 | 示例 |
|--------|------|------|
| `\|` | 顺序链接 | `step1 \| step2` |
| 字典在序列中 | 并行分支 | `step1 \| {key1: step2, key2: step3}` |
| `RunnableBranch` | 条件路由 | `RunnableBranch((cond, runnable), default)` |
| `RouterRunnable` | 键值路由 | `RouterRunnable({"key": runnable})` |
| `.batch()` / `.abatch()` | 并行处理输入 | `chain.batch([in1, in2])` |
| `.stream()` / `.astream()` | 按标记逐个输出 | `for chunk in chain.stream(input):` |

有关协议详情和方法签名，请参阅 [Runnables](runnables.md) 页面。

---
type: "Concept"
title: "Prompt Templates and Few-Shot Learning"
description: "Prompt templates define message sequences and variable substitution patterns for chat models. Few-shot learning selects examples dynamically to teach models by example."
tags: [prompt, template, few-shot, example-selection, variable-substitution, structured-output]
sources:
  - id: openwiki-source-1f4e0a5b877db4f050f2a34c
    resource: repo://libs/core/langchain_core/example_selectors/base.py
  - id: openwiki-source-d533a177a8d9a5dd46f561d9
    resource: repo://libs/core/langchain_core/example_selectors/length_based.py
  - id: openwiki-source-5e027af8cc764d2750129cf1
    resource: repo://libs/core/langchain_core/example_selectors/semantic_similarity.py
  - id: openwiki-source-03d7415879ed05a392edd62d
    resource: repo://libs/core/langchain_core/prompts/base.py
  - id: openwiki-source-15fdd645c1ee76ae559799c1
    resource: repo://libs/core/langchain_core/prompts/chat.py
  - id: openwiki-source-bc32774051e0e8a931a6fecd
    resource: repo://libs/core/langchain_core/prompts/few_shot.py
  - id: openwiki-source-5549894302ea4dfd5b8f4278
    resource: repo://libs/core/langchain_core/prompts/prompt.py
  - id: openwiki-source-cf81d0ba0a387a7cd9b5dfb8
    resource: repo://libs/core/langchain_core/prompts/string.py
  - id: openwiki-source-204b5e61a019044332bd2dd4
    resource: repo://libs/core/langchain_core/prompts/structured.py
generated: { by: "openwiki/0.5.0", at: "2026-09-03T15:18:34.589Z" }
verified:
  - by: openwiki/0.5.0
    at: 2026-09-28T08:35:20.640Z
---
## 概述

LangChain 的 **提示词模板系统** 提供了一种灵活且可组合的方式，用于为语言模型构建消息。提示词接受输入变量，将其格式化为消息序列，并可选地解析结构化输出。该系统区分了 **字符串模板**（用于原始文本）和 **聊天模板**（消息序列，带有类型）。**少样本提示词模板** 还具备动态选择并注入示例的能力，通过示范方式教会模型。

## 基础概念

### 提示词类型

LangChain 提供了两类主要的提示词：

#### PromptTemplate（StringPromptTemplate）

`PromptTemplate` 包装一个带有变量占位符的单一字符串模板。模板使用以下三种引擎之一进行格式化：

- **f-string**（默认）：Python f-string 语法。速度快，支持在 `{...}` 中使用任意表达式，并通过 `{{` 和 `}}` 进行正确转义。
- **mustache**：使用 `{{variable}}` 的 Mustache 语法。对于用户控制的模板更安全。
- **jinja2**：完整的 Jinja2 模板功能。支持条件、循环和过滤器，但如果模板来自不可信来源，则存在安全风险；LangChain 默认使用 `SandboxedEnvironment` 以实现纵深防御。

**关键属性：**
- `template`：模板字符串。
- `input_variables`：格式化时必须提供的变量名称列表。
- `partial_variables`：预先填充的变量；减少格式化时所需的输入。
- `template_format`：使用的引擎（`f-string`、`mustache` 或 `jinja2`）。

```python
from langchain_core.prompts import PromptTemplate

# Simple f-string prompt
prompt = PromptTemplate.from_template("Tell me about {topic}")
output = prompt.format(topic="machine learning")

# Jinja2 with conditionals
prompt = PromptTemplate(
    template="{% if detailed %}Detailed:{% endif %} {content}",
    template_format="jinja2",
    input_variables=["content"],
    partial_variables={"detailed": True}
)

# Partial variables reduce required inputs
prompt = PromptTemplate(
    template="User: {name}, Topic: {topic}",
    input_variables=["topic"],
    partial_variables={"name": "Alice"}
)
result = prompt.format(topic="AI")  # name is already set
```

#### ChatPromptTemplate

`ChatPromptTemplate` 将消息提示词模板组合成对话结构。每条消息具有角色（system、human、ai、tool 等）和内容。这与 GPT-4、Claude 等聊天模型的消息式 API 保持一致。

**构造器模式：**

```python
from langchain_core.prompts import ChatPromptTemplate

# Tuple shorthand
template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("human", "Hello, {name}"),
    ("ai", "Hi {name}! How can I help?"),
    ("human", "{user_input}"),
])

# Or direct list construction
template = ChatPromptTemplate([
    ("system", "You are a helpful assistant."),
    ("human", "Hello, {name}"),
])
```

简写语法中支持的消息类型：
- `"system"` → `SystemMessagePromptTemplate`
- `"human"` → `HumanMessagePromptTemplate`
- `"ai"` → `AIMessagePromptTemplate`
- `"user"` → `"human"` 的别名
- `"assistant"` → `"ai"` 的别名
- `"tool"` / `"function"` → `ToolMessagePromptTemplate` / `FunctionMessagePromptTemplate`
- `"placeholder"` → `MessagesPlaceholder`，用于动态消息列表

**关键方法：**
- `format_messages(**kwargs)`：返回 `BaseMessage` 对象列表。
- `invoke(dict)`：Runnable 接口，返回包含格式化消息的 `ChatPromptValue`。
- `format(**kwargs)`：将消息列表转换为单个字符串（适用于调试或非聊天类 API）。

#### MessagesPlaceholder

`MessagesPlaceholder` 在提示词的特定位置注入一个预格式化的消息列表。这对于维护对话历史至关重要。

```python
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    MessagesPlaceholder("chat_history", optional=True),  # optional=True allows empty list
    ("human", "{question}"),
])

# Pass conversation history
result = template.invoke({
    "chat_history": [
        ("human", "What's 2+2?"),
        ("ai", "4"),
    ],
    "question": "And 3+3?",
})
# Messages: [system, human, ai, human]
```

`optional=True` 标志允许该占位符从输入中省略；若未提供，则用空列表替代。`n_messages` 参数限制包含的最近消息数量（对控制 token 预算很有用）。

### 模板变量替换

提示词会自动从模板语法中检测变量名，并在格式化时要求提供这些变量。

```python
from langchain_core.prompts import PromptTemplate

prompt = PromptTemplate.from_template("Q: {question}\nA: {answer}")
print(prompt.input_variables)  # ['answer', 'question']

# Variables are validated at runtime
try:
    prompt.format(question="What?")  # Missing 'answer'
except KeyError as e:
    print(f"Error: {e}")
```

**部分应用（Partial application）** 会预先填充部分变量，从而减少所需的输入集合：

```python
prompt = PromptTemplate.from_template("User: {name}, Question: {question}")
partial_prompt = prompt.partial(name="Bob")  # Bind 'name'
output = partial_prompt.format(question="How are you?")
# Only 'question' is required now
```

当提示词恰好包含 **一个** 输入变量时，模板可以直接接受非字典类型的参数：

```python
template = ChatPromptTemplate.from_messages([
    ("system", "You are a bot."),
    ("human", "{input}"),
])
result = template.invoke("Hello!")  # Auto-wraps as {"input": "Hello!"}
```

## 少样本提示词模板

少样本学习通过在用户实际查询前提供输入-输出示例来教导模型。LangChain 提供了两种模式：一种用于字符串提示词，另一种用于基于聊天的提示词。

### FewShotPromptTemplate

`FewShotPromptTemplate` 将示例格式化为单个字符串提示词。

**结构：**

```
[prefix]

[formatted example 1]

[formatted example 2]

...

[suffix]
```

**组件：**
- `prefix`：示例之前的文本（可选）。
- `example_prompt`：一个 `PromptTemplate`，指定每个示例的格式化方式。
- `examples` 或 `example_selector`：示例的来源（可以是固定列表或动态选择器）。
- `suffix`：示例之后的文本。通常包含实际任务以及新输入的占位符。
- `example_separator`：连接 prefix、examples 和 suffix 的字符串（默认值：`"\n\n"`）。

```python
from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate

examples = [
    {"input": "happy", "output": "sad"},
    {"input": "tall", "output": "short"},
]

example_prompt = PromptTemplate(
    template="Input: {input}\nOutput: {output}",
    input_variables=["input", "output"],
)

prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    suffix="Input: {input}\nOutput:",
    input_variables=["input"],
)

output = prompt.format(input="big")
# Output:
# Input: happy
# Output: sad
#
# Input: tall
# Output: short
#
# Input: big
# Output:
```

### FewShotChatMessagePromptTemplate

`FewShotChatMessagePromptTemplate` 将示例作为消息对嵌入到聊天序列中。

```python
from langchain_core.prompts import (
    ChatPromptTemplate,
    FewShotChatMessagePromptTemplate,
)

examples = [
    {"input": "2+2", "output": "4"},
    {"input": "2+3", "output": "5"},
]

example_prompt = ChatPromptTemplate.from_messages([
    ("human", "What is {input}?"),
    ("ai", "{output}"),
])

few_shot = FewShotChatMessagePromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
)

template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful math tutor."),
    few_shot,
    ("human", "What is {input}?"),
])

result = template.invoke({"input": "4+4"})
# Messages: [system, human(2+2?), ai(4), human(2+3?), ai(5), human(4+4?)]
```

## 示例选择器

与其使用固定的示例列表，不如使用 **示例选择器**，它可根据输入动态挑选相关的示例。这能优化提示词长度并提升相关性。

### BaseExampleSelector 接口

所有选择器都实现以下接口：

```python
class BaseExampleSelector:
    def add_example(self, example: dict[str, str]) -> Any:
        """Add a new example to the store."""
        
    def select_examples(self, input_variables: dict[str, str]) -> list[dict[str, Any]]:
        """Select which examples to use based on inputs."""
```

### SemanticSimilarityExampleSelector

将示例和输入嵌入到向量空间中，检索出最相似的 `k` 个示例。需要一个 `VectorStore` 和嵌入模型。

```python
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_core.embeddings import OpenAIEmbeddings
from langchain_core.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate

examples = [
    {"input": "happy", "output": "sad"},
    {"input": "tall", "output": "short"},
    {"input": "energetic", "output": "lethargic"},
    {"input": "sunny", "output": "gloomy"},
]

# Create vector store from example texts
to_vectorize = [" ".join(example.values()) for example in examples]
embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_texts(to_vectorize, embeddings, metadatas=examples)

selector = SemanticSimilarityExampleSelector(
    vectorstore=vectorstore,
    k=2,  # Always return 2 examples
)

example_prompt = PromptTemplate(
    template="Input: {input}\nOutput: {output}",
    input_variables=["input", "output"],
)

prompt = FewShotPromptTemplate(
    example_selector=selector,
    example_prompt=example_prompt,
    suffix="Input: {input}\nOutput:",
    input_variables=["input"],
)

# When formatting, the selector retrieves the 2 most similar examples to "bright"
output = prompt.format(input="bright")
```

**关键参数：**
- `vectorstore`：存储嵌入示例的向量存储。
- `k`：返回的示例数量（默认值：4）。
- `input_keys`：可选过滤器，仅使用特定键进行相似度搜索（例如，仅使用 "input" 字段，不包括 "output"）。
- `example_keys`：可选过滤器，仅包含返回示例中的某些键。
- `vectorstore_kwargs`：传递给向量存储的 `similarity_search` 方法的额外参数。

### LengthBasedExampleSelector

按贪婪策略选择示例，直到达到最大 token 数或单词数限制，防止提示词过长。在 token 预算紧张时非常有用。

```python
from langchain_core.example_selectors import LengthBasedExampleSelector
from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate

examples = [
    {"input": "happy", "output": "sad"},
    {"input": "tall", "output": "short"},
    {"input": "energetic", "output": "lethargic"},
]

example_prompt = PromptTemplate(
    template="Input: {input}\nOutput: {output}",
    input_variables=["input", "output"],
)

selector = LengthBasedExampleSelector(
    examples=examples,
    example_prompt=example_prompt,
    max_length=50,  # Limit prompt to ~50 words
    get_text_length=lambda x: len(x.split()),  # Custom length function
)

prompt = FewShotPromptTemplate(
    example_selector=selector,
    example_prompt=example_prompt,
    suffix="Input: {input}\nOutput:",
    input_variables=["input"],
)

# Selector returns only as many examples as fit within max_length
output = prompt.format(input="fast")
```

**关键参数：**
- `examples`：所有可用示例的列表。
- `max_length`：提示词的最大长度（以 token 或单词计，由 `get_text_length` 决定）。
- `get_text_length`：用于测量提示词长度的函数；默认通过正则表达式计算单词数。

**行为：** 示例按顺序遍历；当下一个示例会使总长度超过 `max_length` 时停止添加。这是贪婪策略，而非最优解，但速度快且行为可预测。

## 结构化输出提示词

`StructuredPrompt`（beta 版）将 `ChatPromptTemplate` 与 Pydantic 模式结合，使模型能够生成符合特定模式的 JSON 输出。

```python
from pydantic import BaseModel
from langchain_core.prompts import StructuredPrompt

class QuestionAnswer(BaseModel):
    question: str
    answer: str

template = StructuredPrompt.from_messages_and_schema(
    messages=[
        ("system", "You are a helpful assistant."),
        ("human", "{input}"),
    ],
    schema=QuestionAnswer,
)

# When invoked with a model supporting structured output,
# the model is instructed to return JSON matching QuestionAnswer
result = template.invoke({"input": "What is LangChain?"})
```

这在需要一致且可解析输出的任务中非常有用（例如，事实提取、数据分类）。

## Runnable 接口与链式调用

所有提示词均继承自 `RunnableSerializable`，使其与 LangChain 的链式构建系统兼容。

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

template = ChatPromptTemplate.from_messages([
    ("system", "You are a poet."),
    ("human", "Write a poem about {topic}"),
])

model = ChatOpenAI(model="gpt-4o")
parser = StrOutputParser()

# Chain: prompt → model → parser
chain = template | model | parser

result = chain.invoke({"topic": "the internet"})
print(result)
```

**关键方法：**
- `invoke(dict) -> PromptValue`：同步格式化。
- `ainvoke(dict) -> PromptValue`：异步格式化。
- `stream(dict)`：流式模式（提示词中较少使用，下游更常见）。
- `batch(list[dict])`：批量格式化多个输入。

## 提示词组合

提示词通过 `+` 操作符进行组合，合并消息和变量。

```python
from langchain_core.prompts import ChatPromptTemplate

system = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Your name is {bot_name}."),
])

conversation = ChatPromptTemplate.from_messages([
    ("human", "{user_input}"),
])

combined = system + conversation
# Equivalent to:
# ChatPromptTemplate([
#     ("system", "You are a helpful assistant. Your name is {bot_name}."),
#     ("human", "{user_input}"),
# ])

result = combined.invoke({"bot_name": "Alice", "user_input": "Hello!"})
```

**规则：**
- 合并 `ChatPromptTemplate` 实例时，消息将被连接。
- 两个模板的输入变量将被合并。
- 部分变量将被合并；存在冲突键时会抛出错误。
- 模板必须具有兼容的格式（均为 f-string，或均为 mustache 等）。

## 从文件加载提示词

**注意：** 使用旧的 `save()` / `load_prompt_from_config()` API 进行提示词序列化与加载已弃用，建议改用 `langchain_core.load` 中的 `dumpd()` / `loads()`。

要从 JSON 或 YAML 文件加载提示词，请使用现代 LangChain 序列化 API：

```python
from langchain_core.load import loads
import json

with open("prompt.json") as f:
    prompt_dict = json.load(f)

prompt = loads(prompt_dict)
# Returns a deserialized PromptTemplate or ChatPromptTemplate
```

可以使用 `langchain_core.load` 中的 `dumpd()` 将提示词序列化为 JSON，从而支持版本控制和共享：

```python
from langchain_core.load import dumpd
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are helpful."),
    ("human", "{input}"),
])

prompt_dict = dumpd(prompt)
# Contains nested structure compatible with loads()
```

## 与智能体工厂的集成

提示词是 **智能体工厂**（`create_agent`）的核心输入之一，为智能体推理和工具使用提供对话上下文。

```python
from langchain.agents import create_agent
from langchain_core.prompts import ChatPromptTemplate

system_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful weather assistant."),
])

# Or use a simple string
agent = create_agent(
    model="openai:gpt-4o",
    tools=[weather_tool],
    system_prompt="You are a helpful weather assistant."
)
```

智能体工厂内部会将提示词与模型及工具绑定编译，并通过状态机管理消息流转。中间件可通过 `wrap_model_call` 钩子在模型调用前拦截并修改提示词，支持提示词优化或安全过滤等用例。

## 安全性考虑

### 模板注入

在从用户输入构造提示词时，应使用 **部分变量** 或 **输入变量**，而非字符串拼接：

```python
# UNSAFE: Vulnerable to prompt injection
user_input = input("Enter text: ")
template = f"User said: {user_input}"  # Don't do this

# SAFE: Use variable substitution
from langchain_core.prompts import PromptTemplate
prompt = PromptTemplate.from_template("User said: {user_input}")
output = prompt.format(user_input=user_input)
```

### Jinja2沙箱机制

使用 Jinja2 模板时，LangChain 默认应用 `SandboxedEnvironment`，阻止对双下划线属性（如 `__class__`、`__globals__` 等）的访问。但请注意：

- **不要接受来自不可信来源的 Jinja2 模板。** 沙箱机制仅为尽力而为，并非绝对安全。
- 普通方法调用和属性访问仍被允许（例如 `obj.method()`）。
- 若必须使用 Jinja2，对于不可信输入，优先选择 `f-string` 或 `mustache`。

```python
# Safe: f-string template from user, validated at construction time
from langchain_core.prompts import PromptTemplate
prompt = PromptTemplate(
    template="Hello {name}",  # User-provided, but simple variable syntax
    input_variables=["name"],
)
```

## 生命周期与状态

提示词模板在函数式意义上是**不可变**的：调用 `format()` 或 `invoke()` 不会改变模板本身。类似 `partial()` 的方法会返回新实例。

```python
original = PromptTemplate.from_template("Say {text}")
partial = original.partial(text="hello")  # Returns a NEW PromptTemplate

# original is unchanged
print(original.input_variables)  # ['text']
print(partial.input_variables)   # []
```

这种不可变性确保了在管道中安全地组合与缓存。

## 可观测性与追踪

所有提示词均支持 LangChain 的标准追踪与可观测性钩子：

```python
template = ChatPromptTemplate.from_messages([
    ("system", "You are helpful."),
    ("human", "{input}"),
])

# Add metadata for tracing
template_with_metadata = template.with_config({
    "metadata": {"version": "1.0"},
    "tags": ["important"],
})

result = template_with_metadata.invoke({"input": "hello"})
# The invoke is traced with the given metadata
```

元数据和标签会传递到 LangSmith 及其他可观测性后端，从而支持调试和性能分析。

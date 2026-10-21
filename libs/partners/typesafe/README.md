# langchain-typesafe

[![PyPI - Version](https://img.shields.io/pypi/v/langchain-typesafe?label=%20)](https://pypi.org/project/langchain-typesafe/#history)
[![PyPI - License](https://img.shields.io/pypi/l/langchain-typesafe)](https://opensource.org/licenses/MIT)

## 安装

```bash
uv add langchain-typesafe
```

在发起请求之前，请设置 `TYPESAFE_API_KEY` 环境变量。

## 用法

`TypeSafeClassifier` 是一个 LangChain `Runnable`，可通过 TypeSafe 进行概率分类与评分。

```python
from langchain_typesafe import Choice, Noul, Score, TypeSafeClassifier

classifier = TypeSafeClassifier()

result = classifier.invoke(
    {
        "state": "Stripe has failed to connect for three days. Help ASAP.",
        "questions": {
            "department": Choice(
                instructions="Which team should handle this?",
                criteria={
                    "billing": "Payment or subscription issues",
                    "technical": "Product or integration issues",
                },
            ),
            "urgent": Noul(instructions="Does this message express urgency?"),
            "frustration": Score(
                instructions="How frustrated does the customer appear?",
                criteria=["calm", "frustrated", "angry"],
            ),
        },
    }
)
print(result.choices["department"].choice)
print(result.nouls["urgent"].noul)
print(result.scores["frustration"].score)
```

将完整的 `ClassifierRequest` 映射传递给 `invoke` 或 `ainvoke`。在 Runnable 输入中同时保留 `state` 和 `questions`，可以让完整的分类请求用于组合、批处理、回调与追踪。异步应用请使用 `await classifier.ainvoke(...)`：

```python
from langchain_typesafe import ClassifierRequest

request: ClassifierRequest = {
    "state": "Stripe has failed to connect for three days. Help ASAP.",
    "questions": {"urgent": Noul(instructions="Is this urgent?")},
}
result = classifier.invoke(request)
```

### 实验性中间件

安装实验性附加依赖以使用由 TypeSafe 驱动的智能体中间件。`langchain_typesafe.experimental` 下的 API 可能会随时变更，恕不另行通知。

```bash
uv add "langchain-typesafe[experimental]"
```

#### `ModelRouterMiddleware`

`ModelRouterMiddleware` 会将智能体路由到由 TypeSafe `Choice` 问题选出的模型：


```python
from langchain.agents import create_agent
from langchain_typesafe.experimental.middleware import (
    ModelChoice,
    ModelRouterMiddleware,
)

router = ModelRouterMiddleware(
    choices={
        "fast": ModelChoice(
            model="openai:gpt-5-mini",
            criteria="Simple, well-scoped tasks.",
        ),
        "powerful": ModelChoice(
            model=powerful_model,
            criteria="Complex tasks requiring deeper reasoning.",
        ),
    },
    instructions="Choose the least costly model suited to the task.",
)
agent = create_agent("openai:gpt-5-mini", middleware=[router])
```

模型路由器会在每次智能体运行时对最新的人类消息进行一次分类，并将完整的 `ChoiceAnswer` 存储在智能体状态中，使其概率与置信度可供应用和追踪使用。

#### `AutoModeMiddleware`

`AutoModeMiddleware` 会对显式配置的工具调用进行分类，并在执行前拦截有风险的调用：

```python
from langchain_typesafe import NoulCriteria
from langchain_typesafe.experimental.middleware import AutoModeMiddleware

auto_mode = AutoModeMiddleware(
    tools=[delete_file],
    criteria=NoulCriteria(
        true="The call writes, deletes, publishes, or changes access.",
        false="The call only reads public or user-provided data.",
    ),
)
agent = create_agent(
    model,
    tools=[read_file, delete_file],
    middleware=[auto_mode],
)
```

`tools` 接受工具名称或 `BaseTool` 实例。可通过 `instructions` 自定义整体风险问题，通过 `criteria` 自定义特定应用的有风险与安全结果。被配置的工具调用中，风险概率达到或超过阈值的调用将返回一个错误 `ToolMessage`。

### 以 LangChain 消息作为状态

`BaseMessage` 对象和消息序列可以出现在 JSON 状态的根层级或任意位置。该集成会将它们递归转换为包含 `role` 和 `content` 字段的对象，同时保留周围的应用数据：

```python
from langchain_core.messages import HumanMessage, SystemMessage

response = classifier.invoke(
    {
        "state": {
            "conversation": [
                SystemMessage("You are reviewing a customer support conversation."),
                HumanMessage("My payouts have failed for three days. Help!"),
            ],
            "account_tier": "enterprise",
        },
        "questions": {
            "urgent": Noul(instructions="Does this customer need urgent help?")
        },
    }
)
```

### 自定义 HTTP 客户端

在未提供客户端时，分类器会创建同步与异步的 `httpx2` 客户端。需要自定义传输层、代理或共享连接池的应用可以独立注入其中任意一种客户端：

```python
import httpx2

classifier = TypeSafeClassifier(
    client=httpx2.Client(proxy="http://proxy.internal"),
    async_client=httpx2.AsyncClient(proxy="http://proxy.internal"),
)
```

注入的客户端将按原样使用，其生命周期由应用自行负责。

### 错误处理

提供商错误同样继承自 LangChain 的标准模型错误层级。因此，应用可以在需要提供商元数据时捕获 TypeSafe 专属错误，或在统一处理多个模型提供商时捕获 LangChain 错误：

```python
from langchain_core.exceptions import ModelAuthenticationError, ModelRateLimitError
from langchain_typesafe import TypeSafeRateLimitError

try:
    response = classifier.invoke(
        {
            "state": "Classify this message.",
            "questions": {"urgent": Noul(instructions="Is this urgent?")},
        }
    )
except TypeSafeRateLimitError as error:
    print(error.request_id, error.retry_after_ms)
except (ModelAuthenticationError, ModelRateLimitError):
    handle_model_error()
```

`TypeSafeAPIError` 会暴露响应状态码、解析后的响应体、响应头、脱敏后的端点以及请求 ID。连接、超时、响应校验及特定状态码等子类的命名与 TypeSafe Python SDK 保持一致。

## 文档

模型与问题的语义请参阅 [TypeSafe 文档](https://docs.typesafe.ai/)。LangChain API 参考文档可在 [reference.langchain.com](https://reference.langchain.com/python/integrations/langchain_typesafe/) 查看。

## 参与贡献

有关贡献方式，请参阅 [LangChain 贡献指南](https://docs.langchain.com/oss/python/contributing/overview)。

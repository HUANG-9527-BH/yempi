---
type: "Testing & QA"
title: "Unit Testing and Mocking"
description: "Guide to writing unit tests: test structure, mocking chat models, fixtures, assertions, and test coverage for agents and components."
tags: [unit-tests, pytest, testing, fixtures, mocking, chat-models, tools, agents, integration, assertions]
sources:
  - id: openwiki-source-8f1875229ad4a704c8e20a06
    resource: repo://libs/core/Makefile
  - id: openwiki-source-043c2520f819900dc753650e
    resource: repo://libs/core/tests/unit_tests/callbacks/test_async_callback_manager.py
  - id: openwiki-source-727aef6a92fb635fdbb41cd6
    resource: repo://libs/core/tests/unit_tests/conftest.py
  - id: openwiki-source-5e13d2c899eb5925ef28fddf
    resource: repo://libs/core/tests/unit_tests/fake/callbacks.py
  - id: openwiki-source-e8916b46b41eee662deabd17
    resource: repo://libs/core/tests/unit_tests/fake/test_fake_chat_model.py
  - id: openwiki-source-f0e376fe9b6befdcc2505465
    resource: repo://libs/core/tests/unit_tests/pydantic_utils.py
  - id: openwiki-source-344bd4b667096c3c45c8fa82
    resource: repo://libs/core/tests/unit_tests/runnables/conftest.py
  - id: openwiki-source-4717abc86db20c5c76bbf23a
    resource: repo://libs/core/tests/unit_tests/runnables/test_runnable.py
  - id: openwiki-source-5839db669f618a6d604790ca
    resource: repo://libs/core/tests/unit_tests/stubs.py
  - id: openwiki-source-fffa23f844d6fa8ec39f3357
    resource: repo://libs/langchain_v1/tests/unit_tests/agents/any_str.py
  - id: openwiki-source-09fd21baed875d7f9c53a492
    resource: repo://libs/langchain_v1/tests/unit_tests/agents/conftest.py
  - id: openwiki-source-18a5b32f36bc87fa2d5237e3
    resource: repo://libs/langchain_v1/tests/unit_tests/agents/messages.py
  - id: openwiki-source-6904a2d923d0abd0c1af93bd
    resource: repo://libs/langchain_v1/tests/unit_tests/agents/model.py
  - id: openwiki-source-ff9b926753e6cc6fe87acfe7
    resource: repo://libs/langchain_v1/tests/unit_tests/agents/test_agent_streaming.py
  - id: openwiki-source-08ccfc598e3f10e8eea9ded3
    resource: repo://libs/langchain_v1/tests/unit_tests/agents/test_invalid_tool_calls.py
  - id: openwiki-source-bd29e79613d5f366a00068f5
    resource: repo://libs/standard-tests/langchain_tests/base.py
  - id: openwiki-source-3eb9100e02f9d70098d1b30d
    resource: repo://libs/standard-tests/langchain_tests/unit_tests/chat_models.py
  - id: openwiki-source-54e69c0cb7aa4a73b87cf97d
    resource: repo://libs/standard-tests/langchain_tests/unit_tests/embeddings.py
  - id: openwiki-source-a6b31954b6df57580d0f3ed0
    resource: repo://libs/standard-tests/langchain_tests/unit_tests/tools.py
generated: { by: "openwiki/0.5.0", at: "2026-09-28T08:35:20.640Z" }
verified:
  - by: openwiki/0.5.0
    at: 2026-10-16T08:25:55.096Z
---
## 概述

LangChain 中的单元测试在不进行网络调用或依赖外部 API 的情况下，对组件进行孤立测试。测试文件位于 `tests/unit_tests/` 目录中，并通过 `make test` 或 `uv run --group test pytest` 运行，同时施加严格的套接字限制并启用并行化。

本页涵盖测试基础设施、用于聊天模型和工具的标准测试类、智能体特定的测试模式（状态转换、中间件行为、工具调用）、测试夹具、模拟策略、回调机制以及用于测试智能体和组件的断言方法。

## 测试结构与组织

### 目录布局

每个 LangChain 包都以一致的方式组织其测试：


```
libs/core/
├── tests/
│   ├── unit_tests/       # No network calls; run via make test
│   ├── integration_tests/ # Live API calls; require credentials and API keys
│   └── benchmarks/       # Performance measurement tests
├── Makefile              # Task automation
└── pyproject.toml        # Dependencies
```


单元测试与源代码结构一一对应：位于 `langchain_core/runnables/base.py` 的模块，其测试文件位于 `tests/unit_tests/runnables/test_runnable.py`。

### 运行单元测试

所有单元测试均在套接字受限的环境下并行运行，以防止意外的网络访问：


```bash
# Run all unit tests in a package
make test

# Run a specific test file or directory
make test TEST_FILE=tests/unit_tests/runnables/test_runnable.py

# Run using uv directly
uv run --group test pytest tests/unit_tests/

# Watch mode: auto-rerun on code changes
make test_watch

# Extended tests (marked with @pytest.mark.requires)
make extended_tests
```


Makefile 的 test 目标设置了 `--disable-socket --allow-unix-socket`，并使用 `pytest-xdist`（`-n auto`）实现并行执行。LangSmith 追踪相关的环境变量（如 `LANGCHAIN_TRACING_V2`、`LANGSMITH_API_KEY` 等）被显式清除，以确保测试彼此独立。

## 标准测试类

`langchain-tests` 包（位于 `/libs/standard-tests/`）提供了可复用的基础测试类，用于集成测试。这些类确保了聊天模型、嵌入和工具等组件的测试一致性。

### ChatModelUnitTests

**位置**：`langchain_tests.unit_tests.ChatModelUnitTests`

对于任何聊天模型，应创建一个继承自 `ChatModelUnitTests` 的测试类，并实现两个必需的属性：


```python
# tests/unit_tests/test_standard.py
from typing import Type

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_tests.unit_tests import ChatModelUnitTests

from my_package.chat_models import MyChatModel


class TestMyChatModelUnit(ChatModelUnitTests):
    @property
    def chat_model_class(self) -> Type[BaseChatModel]:
        return MyChatModel

    @property
    def chat_model_params(self) -> dict:
        return {"model": "my-model-001", "temperature": 0}
```


**测试内容**：
- **初始化**：使用标准参数实例化模型
- **同步/异步调用**：在同步和异步上下文中处理单条消息
- **流式输出**：分块流式响应及分块累积
- **工具绑定**：`bind_tools()` 接口（如果支持）
- **结构化输出**：`with_structured_output()` 用于模式校验（如果支持）
- **序列化**：通过 LangChain 的序列化 API 进行模型的保存与加载
- **消息类型**：单条与多条消息对话、系统提示、工具消息
- **工具调用**：正确调用工具并处理结果（如果支持）

**可配置功能**（可通过属性覆盖）：

- `has_tool_calling` (bool)：模型的 `bind_tools` 方法是否被重写；自动检测但可显式设置
- `has_tool_choice` (bool)：`bind_tools` 是否接受 `tool_choice` 参数以强制调用工具
- `has_structured_output` (bool)：是否实现了 `with_structured_output()` 或 `bind_tools()`
- `structured_output_kwargs` (dict)：`with_structured_output()` 的额外参数（例如 `{"method": "json_schema"}`）
- `supports_json_mode` (bool)：模型是否支持在结构化输出中使用 `method='json_mode'`
- `supports_image_inputs` (bool)：模型是否接受图像内容块
- `supports_image_urls` (bool)：模型是否接受内容中的图像 URL
- `supports_pdf_inputs` (bool)：模型是否接受 PDF 文件内容
- `supports_audio_inputs` (bool)：模型是否接受音频内容
- `returns_usage_metadata` (bool)：`invoke()` 和 `stream()` 是否返回使用量的 token 统计（默认为 True）
- `supports_model_override` (bool)：模型在 `invoke()` 中是否接受 `model` 参数以在运行时覆盖（默认为 True）
- `model_override_value` (str)：用于测试动态模型选择的替代模型名称（当 `supports_model_override=True` 时必需）

示例使用功能标志：

```python
class TestOpenAIChatModel(ChatModelUnitTests):
    @property
    def chat_model_class(self) -> Type[BaseChatModel]:
        return ChatOpenAI

    @property
    def chat_model_params(self) -> dict:
        return {"model": "gpt-4"}

    @property
    def has_tool_calling(self) -> bool:
        return True

    @property
    def structured_output_kwargs(self) -> dict:
        return {"method": "json_schema"}

    @property
    def supports_image_inputs(self) -> bool:
        return True

    @property
    def model_override_value(self) -> str:
        return "gpt-4-turbo"
```

### EmbeddingsUnitTests

**位置**：`langchain_tests.unit_tests.EmbeddingsUnitTests`

类似地测试嵌入模型：

```python
from typing import Type

from langchain_core.embeddings import Embeddings
from langchain_tests.unit_tests import EmbeddingsUnitTests

from my_package.embeddings import MyEmbeddings


class TestMyEmbeddingsUnit(EmbeddingsUnitTests):
    @property
    def embeddings_class(self) -> Type[Embeddings]:
        return MyEmbeddings

    @property
    def embedding_model_params(self) -> dict:
        return {"model": "embedding-v1"}
```

**测试内容**：
- 模型初始化
- 单个文本字符串的嵌入
- 一批文本字符串的嵌入
- 从环境变量初始化（若设置了 `init_from_env_params`）

**可配置项**：
- `init_from_env_params` (tuple)：返回 `(env_vars, init_args, expected_attrs)` 以测试基于环境变量的初始化

### ToolsUnitTests

**位置**：`langchain_tests.unit_tests.ToolsUnitTests`

测试自定义工具：

```python
from langchain_core.tools import BaseTool
from langchain_tests.unit_tests import ToolsUnitTests

from my_package.tools import MyTool


class TestMyToolUnit(ToolsUnitTests):
    @property
    def tool_constructor(self) -> type[BaseTool] | BaseTool:
        return MyTool

    @property
    def tool_constructor_params(self) -> dict:
        return {"api_key": "test-key"}

    @property
    def tool_invoke_params_example(self) -> dict:
        return {"query": "example query"}
```

**测试内容**：
- 工具初始化
- 使用示例参数调用工具
- 工具模式生成（JSON Schema）
- 从环境变量初始化

## 用于智能体测试的聊天模型模拟

### FakeToolCallingModel

为在不进行 API 调用的情况下测试智能体和结构化工作流，可使用测试工具中的 `FakeToolCallingModel`：

```python
from tests.unit_tests.agents.model import FakeToolCallingModel
from langchain_core.messages import ToolCall

# Script predictable tool calls
model = FakeToolCallingModel(
    tool_calls=[
        [ToolCall(name="get_weather", args={"city": "Paris"}, id="call_1")],
        [],  # No tool call on second turn (finish)
    ]
)
```

**特性**：
- **tool_calls**：列表的列表，每个内层列表代表一轮中的工具调用
- **structured_response**：结构化输出模式下的响应（BaseModel、dataclass 或 dict）
- **index**：用于多轮测试的自动递增计数器
- **tool_style**：`bind_tools()` 输出风格 —— "openai"（默认）或 "anthropic"

**结构化输出示例**：

```python
from pydantic import BaseModel

class WeatherResponse(BaseModel):
    city: str
    temperature: int

model = FakeToolCallingModel(
    structured_response=WeatherResponse(city="Paris", temperature=20)
)
result = model.invoke([...], response_format=...)
```

### 无效工具调用处理

测试当模型返回格式错误的工具调用时，智能体的行为：

```python
from langchain_core.messages import AIMessage

# Simulate a model returning invalid JSON in tool args
message = AIMessage(
    content="",
    invalid_tool_calls=[
        {
            "name": "get_weather",
            "args": '{"city":',  # Truncated JSON
            "id": "call_1",
            "error": "Invalid JSON",
        }
    ],
)
```

智能体的错误处理会生成一个 `ToolMessage`，其 `status="error"` 并包含错误详情，使智能体可在下一轮中恢复。

## 共享固定项与配置

### conftest.py 模式

`tests/unit_tests/` 目录下的根 `conftest.py` 提供共享固定项和 pytest 钩子。智能体测试套件（如 `libs/langchain_v1/tests/unit_tests/agents/conftest.py`）会在此基础上扩展专用固定项。

**来自 `/libs/core/tests/unit_tests/conftest.py`**：

```python
@pytest.fixture(autouse=True)
def blockbuster() -> Iterator[BlockBuster]:
    """Blockbuster fixture prevents blocking I/O in async code."""
    with blockbuster_ctx("langchain_core") as bb:
        # Allow specific blocking operations in specific locations
        bb.functions["os.stat"].can_block_in(
            "langchain_core/_api/internal.py", "is_caller_internal"
        ).can_block_in(
            "langchain_core/runnables/base.py", "__repr__"
        ).can_block_in(
            "langsmith/client.py", "_default_retry_config"
        )
        bb.functions["os.path.abspath"].can_block_in(
            "langchain_core/_api/internal.py", "is_caller_internal"
        ).can_block_in(
            "langchain_core/runnables/base.py", "__repr__"
        )
        bb.functions["io.TextIOWrapper.read"].can_block_in(
            "langsmith/client.py", "_default_retry_config"
        )
        for bb_function in bb.functions.values():
            bb_function.can_block_in("freezegun/api.py", "_get_cached_module_attributes")
        yield bb
```

### 智能体测试固定项

智能体测试套件使用参数化固定项来跨多个检查点和状态存储后端进行测试：

```python
from collections.abc import AsyncIterator, Iterator
from uuid import UUID
from pytest_mock import MockerFixture
import pytest

# Deterministic UUIDs for snapshot testing
@pytest.fixture
def deterministic_uuids(mocker: MockerFixture) -> MockerFixture:
    side_effect = (UUID(f"00000000-0000-4000-8000-{i:012}", version=4) for i in range(10000))
    return mocker.patch("uuid.uuid4", side_effect=side_effect)

# Parametrized checkpointer fixture (multiple backends)
@pytest.fixture(params=["memory", "sqlite", "postgres"])
def sync_checkpointer(request: pytest.FixtureRequest) -> Iterator[BaseCheckpointSaver]:
    checkpointer_name = request.param
    # Yield appropriate checkpointer based on backend
    ...

# Parametrized async checkpointer fixture
@pytest.fixture(params=["memory", "sqlite_aio", "postgres_aio"])
async def async_checkpointer(request: pytest.FixtureRequest) -> AsyncIterator[BaseCheckpointSaver]:
    checkpointer_name = request.param
    # Yield appropriate async checkpointer
    ...
```

设置 `LANGGRAPH_TEST_FAST=true` 环境变量，仅针对内存后端运行以实现快速迭代。

**自定义标记**：

```python
def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--only-extended",
        action="store_true",
        help="Only run extended tests marked with @pytest.mark.requires",
    )
    parser.addoption(
        "--only-core",
        action="store_true",
        help="Only run core tests (skip extended tests)",
    )

def pytest_collection_modifyitems(config: pytest.Config, items) -> None:
    """Automatically skip tests marked with @pytest.mark.requires if dependencies are missing."""
    for item in items:
        requires_marker = item.get_closest_marker("requires")
        if requires_marker:
            for pkg in requires_marker.args:
                if util.find_spec(pkg) is None:
                    item.add_marker(pytest.mark.skip(reason=f"Requires pkg: {pkg}"))
```

**用于确定性 UUID 的固定项**：

```python
@pytest.fixture
def deterministic_uuids(mocker):
    """Replace random UUIDs with deterministic values for snapshot testing."""
    side_effect = (UUID(f"00000000-0000-4000-8000-{i:012}", version=4) for i in range(10000))
    return mocker.patch("uuid.uuid4", side_effect=side_effect)
```

在需要 UUID 在多次运行中保持稳定的情况下，使用 `deterministic_uuids` 固定项：

```python
def test_runnable_with_trace(deterministic_uuids):
    # UUIDs will be predictable now
    ...
```

### 标记模式

```python
# Skip test if dependency is missing
@pytest.mark.requires("anthropic")
def test_anthropic_tool_calling():
    from anthropic import Anthropic
    ...

# Extended tests (run with make extended_tests or --only-extended)
@pytest.mark.requires("openai")
def test_openai_structured_output():
    ...

# Parametrized tests
@pytest.mark.parametrize("model_name,expected_tokens", [
    ("small", 100),
    ("large", 1000),
])
def test_model_sizes(model_name, expected_tokens):
    ...

# Skip on Python version
@pytest.mark.skipif(sys.version_info < (3, 11), reason="Requires 3.11+")
def test_new_feature():
    ...

# Expected failure
@pytest.mark.xfail(reason="Feature not yet implemented")
def test_future_feature():
    ...
```

## 用于测试的假实现

LangChain 提供了假/模拟的聊天模型及其他组件，以避免在单元测试中进行 API 调用。

### FakeChatModel 类

位于 `langchain_core.language_models` 中：

```python
from langchain_core.language_models import (
    FakeListChatModel,
    FakeMessagesListChatModel,
    GenericFakeChatModel,
    ParrotFakeChatModel,
)
from langchain_core.messages import AIMessage, HumanMessage
```

**FakeListChatModel**：在一组固定的字符串响应中循环。

```python
from langchain_core.language_models import FakeListChatModel

model = FakeListChatModel(responses=["Hello", "Hi", "Hey"])
response = model.invoke("How are you?")
# Returns AIMessage(content="Hello")

response = model.invoke("What's up?")
# Returns AIMessage(content="Hi")
```

**GenericFakeChatModel**：在 AIMessage 对象中循环；适用于测试流式输出。

```python
from itertools import cycle
from langchain_core.messages import AIMessage
from langchain_core.language_models import GenericFakeChatModel

messages = cycle([AIMessage(content="response1"), AIMessage(content="response2")])
model = GenericFakeChatModel(messages=messages)

# Test streaming
chunks = list(model.stream("query"))
# Chunks are character-level splits of "response1"
```

**ParrotFakeChatModel**：将输入消息原样回显。

```python
from langchain_core.language_models import ParrotFakeChatModel

model = ParrotFakeChatModel()
response = model.invoke("Hello!")
# Returns AIMessage(content="Hello!")
```


**FakeListLLM 和 FakeStreamingListLLM**：旧版 LLM 接口（文本输入，文本输出）。

```python
from langchain_core.language_models import FakeListLLM

llm = FakeListLLM(responses=["Response 1", "Response 2"])
output = llm.invoke("Query")
```


### FakeEmbeddings

```python
from langchain_core.embeddings import FakeEmbeddings

embeddings = FakeEmbeddings(model="fake-model", size=1536)

# Embed a single string
vector = embeddings.embed_query("hello")
# Returns a list of 1536 float values (deterministic based on input hash)

# Embed a batch
vectors = embeddings.embed_documents(["hello", "world"])
# Returns list of vectors, one per input
```


### FakeCallbackHandler

位于 `tests.unit_tests.fake.callbacks`，一个用于测试的回调处理器，用于统计事件：

```python
from tests.unit_tests.fake.callbacks import FakeCallbackHandler

handler = FakeCallbackHandler()

# Track various events
assert handler.llm_starts == 0
assert handler.chain_starts == 0

# After invoke on a chain with LLM calls:
model.invoke("query", callbacks=[handler])

assert handler.llm_starts == 1
assert handler.llm_ends == 1
assert handler.starts == 1  # Total starts

# Fine-grained counters
assert handler.llm_streams == 0  # for streaming models
assert handler.tool_starts == 0
assert handler.tool_ends == 0
assert handler.chain_starts == 1
assert handler.chain_ends == 1
```


## 常见测试模式

### 智能体执行测试

测试智能体在多轮交互、状态更新和工具调用中的行为：

```python
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.checkpoint.memory import InMemorySaver
from tests.unit_tests.agents.model import FakeToolCallingModel

def test_agent_tool_invocation():
    # Script tool calls for multiple turns
    model = FakeToolCallingModel(
        tool_calls=[
            [{"name": "get_weather", "args": {"city": "Paris"}, "id": "tc1"}],
            [],  # No tool call on second turn
        ]
    )
    
    agent = create_agent(model, [get_weather], checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "test-1"}}
    
    # First turn: invoke with user message
    result = agent.invoke({"messages": [HumanMessage("Weather in Paris?")]}, config)
    
    # Verify response structure
    assert len(result["messages"]) >= 3  # human, ai with tool call, tool result
    assert isinstance(result["messages"][-1], AIMessage) or result["messages"][-1].name == "get_weather"
```


### 状态转换与检查点测试

测试智能体状态在检查点边界处的更新情况：

```python
def test_agent_state_persistence(sync_checkpointer):
    model = FakeToolCallingModel(tool_calls=[[], []])
    agent = create_agent(model, [], checkpointer=sync_checkpointer)
    config = {"configurable": {"thread_id": "thread-1"}}
    
    # First invoke
    result1 = agent.invoke({"messages": [HumanMessage("hi")]}, config)
    
    # Load checkpoint and verify state
    checkpoint = sync_checkpointer.get_tuple(config)
    assert checkpoint is not None
    assert checkpoint.checkpoint["channel_values"]["messages"] == result1["messages"]
    assert checkpoint.metadata["step"] == 1
    
    # Second invoke from checkpoint
    result2 = agent.invoke({"messages": [HumanMessage("hello again")]}, config)
    
    # Verify continuity
    assert len(result2["messages"]) > len(result1["messages"])
```


### 流式事件测试

通过流式事件（v3 协议）测试智能体执行追踪：

```python
def test_agent_stream_events():
    model = FakeToolCallingModel(
        tool_calls=[
            [{"name": "echo", "args": {"text": "x"}, "id": "tc1"}],
            [],
        ]
    )
    agent = create_agent(model, [echo])
    
    # Stream events with built-in tool_calls projection
    run = agent.stream_events(
        {"messages": [HumanMessage("hi")]},
        version="v3"
    )
    
    # Collect tool calls via projection
    tool_calls = list(run.tool_calls)  # type: ignore[attr-defined]
    assert len(tool_calls) == 1
    assert tool_calls[0].tool_name == "echo"
    assert tool_calls[0].tool_call_id == "tc1"
    assert tool_calls[0].completed is True
```


### 消息匹配（忽略 ID）

测试消息内容，而不依赖生成的 UUID：

```python
from tests.unit_tests.agents.any_str import AnyStr
from tests.unit_tests.agents.messages import _AnyIdHumanMessage, _AnyIdToolMessage

def test_agent_message_flow():
    model = FakeToolCallingModel()
    agent = create_agent(model, [])
    
    result = agent.invoke({"messages": [HumanMessage("hi")]})
    
    # Match messages regardless of their generated IDs
    assert result["messages"][0] == _AnyIdHumanMessage(content="hi")
    assert isinstance(result["messages"][1], AIMessage)
```


### 固定用例使用

将可复用组件定义为 pytest 固定用例：

```python
import pytest
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate

@pytest.fixture
def system_prompt():
    return SystemMessage(content="You are a helpful assistant.")

@pytest.fixture
def chat_prompt():
    return ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant."),
        ("human", "{user_input}"),
    ])

def test_with_prompt(chat_prompt):
    # Use the fixture
    assert chat_prompt is not None
```


`conftest.py` 中的固定用例会自动被发现，并对当前目录及其子目录中的所有测试可用。

### 参数化测试

测试多种输入/输出组合：

```python
import pytest

@pytest.mark.parametrize("input_text,expected_length", [
    ("hello", 5),
    ("world", 5),
    ("testing", 7),
    ("", 0),
])
def test_text_length(input_text, expected_length):
    assert len(input_text) == expected_length
```


使用固定用例进行参数化：

```python
@pytest.fixture(params=["gpt-3.5-turbo", "gpt-4"])
def model_name(request):
    return request.param

def test_model_response(model_name):
    # Test runs twice, once for each model
    model = ChatOpenAI(model=model_name)
    response = model.invoke("Hello")
    assert response.content is not None
```


### 使用 pytest-mock 进行模拟

`pytest-mock` 库提供了一个 `mocker` 固定用例：

```python
from unittest import mock

def test_with_mock(mocker):
    # Mock a function
    mock_api_call = mocker.patch("my_module.api_call")
    mock_api_call.return_value = {"status": "success"}

    # Call code that uses api_call
    result = my_function()
    
    assert mock_api_call.called
    assert result == {"status": "success"}

    # Check call arguments
    mock_api_call.assert_called_with("expected_arg")
```


模拟环境变量：

```python
from unittest import mock
import os

def test_env_initialization(mocker):
    mocker.patch.dict(os.environ, {"API_KEY": "test-key"})
    
    # Code that reads API_KEY will get "test-key"
    model = MyModel()  # reads from os.environ
    assert model.api_key == "test-key"
```


### 回调测试

测试回调是否以正确数据被调用：

```python
from langchain_core.callbacks.manager import CallbackManager
from tests.unit_tests.fake.callbacks import FakeCallbackHandler

def test_callbacks_on_chain():
    handler = FakeCallbackHandler()
    
    # Create a chain
    prompt = ChatPromptTemplate.from_template("Say hello to {name}")
    model = FakeChatModel(responses=["Hello Alice"])
    chain = prompt | model
    
    # Invoke with callbacks
    result = chain.invoke(
        {"name": "Alice"},
        config={"callbacks": [handler]}
    )
    
    # Verify callbacks were fired
    assert handler.starts == 2  # prompt + model
    assert handler.ends == 2
    assert handler.chain_starts == 0  # Only LLM runs were tracked
    assert handler.llm_starts == 1
    assert handler.llm_ends == 1
```


### 异步测试

使用 `async def` 标记异步测试，`pytest` 会自动处理：

```python
import pytest

@pytest.mark.asyncio
async def test_async_invoke():
    model = ChatOpenAI()
    result = await model.ainvoke("Hello")
    assert result.content is not None

@pytest.mark.asyncio
async def test_async_streaming():
    model = ChatOpenAI()
    chunks = []
    async for chunk in model.astream("Hello"):
        chunks.append(chunk)
    assert len(chunks) > 0
```


### 使用 Syrupy 的快照测试

快照测试用于捕获输出并与其基线快照进行比较。适用于复杂结构、追踪和序列化对象。

```python
from syrupy.assertion import SnapshotAssertion

def test_runnable_serialization(snapshot: SnapshotAssertion):
    prompt = ChatPromptTemplate.from_template("Say {msg}")
    model = ChatOpenAI(model="gpt-4")
    chain = prompt | model
    
    # Dump to serializable form
    dumped = dumpd(chain)
    
    # Compare against snapshot
    assert dumped == snapshot
```


快照存储在 `__snapshots__/` 目录中。可通过以下命令更新：

```bash
make test_watch  # Auto-updates snapshots
# or
pytest --snapshot-update
```


### 消息测试的辅助桩函数

在测试带有生成 ID 的消息时，可使用 `tests.unit_tests.stubs` 中的辅助函数来匹配任意 ID：

```python
from tests.unit_tests.stubs import (
    _any_id_ai_message,
    _any_id_ai_message_chunk,
    _any_id_human_message,
    AnyStr,
)

def test_message_response():
    model = GenericFakeChatModel(messages=cycle([AIMessage(content="hello")]))
    response = model.invoke("hi")
    
    # Matches any ID
    assert response == _any_id_ai_message(content="hello")

def test_message_streaming():
    model = GenericFakeChatModel(messages=cycle([AIMessage(content="hello")]))
    chunks = list(model.stream("hi"))
    
    assert chunks[0] == _any_id_ai_message_chunk(content="h")
    assert chunks[1] == _any_id_ai_message_chunk(content="ello", chunk_position="last")
```


`AnyStr` 类在用作值时，可匹配任意字符串：

```python
message.id = AnyStr()  # Now message.id == any_other_id is True
```


## 智能体测试的断言方法

### 消息类型断言

验证智能体输出中的消息类型和属性：

```python
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

def test_agent_response_types():
    agent = create_agent(model, [tool])
    result = agent.invoke({"messages": [HumanMessage("query")]})
    
    messages = result["messages"]
    assert isinstance(messages[0], HumanMessage)
    assert isinstance(messages[1], AIMessage)
    if len(messages) > 2:
        assert isinstance(messages[2], ToolMessage)
```


### 工具调用断言

验证 AI 消息中的工具调用：

```python
def test_agent_tool_calls():
    model = FakeToolCallingModel(
        tool_calls=[[{"name": "get_weather", "args": {"city": "NYC"}, "id": "tc1"}]]
    )
    agent = create_agent(model, [get_weather])
    result = agent.invoke({"messages": [HumanMessage("weather")]})
    
    ai_message = result["messages"][1]
    assert len(ai_message.tool_calls) == 1
    assert ai_message.tool_calls[0].name == "get_weather"
    assert ai_message.tool_calls[0].args == {"city": "NYC"}
```


### 状态更新断言

验证状态与检查点的一致性：

```python
def test_state_updates(sync_checkpointer):
    agent = create_agent(model, [tool], checkpointer=sync_checkpointer)
    config = {"configurable": {"thread_id": "t1"}}
    
    result = agent.invoke({"messages": [HumanMessage("hello")]}, config)
    
    # Load checkpoint
    checkpoint = sync_checkpointer.get_tuple(config)
    assert checkpoint.metadata["step"] == 1
    assert len(checkpoint.checkpoint["channel_values"]["messages"]) > 1
```


### 错误与无效工具调用断言

测试对格式错误的工具调用的错误处理：

```python
def test_invalid_tool_call_recovery():
    model = InvalidToolCallingModel()  # Returns invalid JSON in args
    agent = create_agent(model, [get_weather])
    config = {"configurable": {"thread_id": "t1"}}
    
    agent.invoke({"messages": [HumanMessage("weather")]}, config)
    model.invalid_tool_call_id = None
    result = agent.invoke({"messages": [HumanMessage("retry")]}, config)
    
    # Verify error ToolMessage was added
    tool_messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert any(m.status == "error" for m in tool_messages)
```


## 使用 mypy 进行类型检查

类型检查是标准 lint 工作流的一部分：

```bash
# Full type checking
make type

# Or directly with mypy
mypy libs/core/langchain_core/

# Type check specific file
mypy libs/core/langchain_core/runnables/base.py
```


Makefile 在运行 `make lint` 时会执行 `mypy`，该命令同时也会运行 ruff 和格式化检查：

```bash
make lint  # runs: ruff check, ruff format --diff, mypy
```


### 类型检查模式

全程使用类型提示：

```python
from typing import Any, Sequence
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage

def create_chain(
    model: BaseChatModel,
    messages: Sequence[BaseMessage],
    temperature: float = 0.7,
) -> str:
    """Create and invoke a chain.
    
    Args:
        model: The language model to use.
        messages: Input messages.
        temperature: Sampling temperature.
        
    Returns:
        The model's response as a string.
    """
    response = model.invoke(messages, {"temperature": temperature})
    return response.content
```


使用 `TYPE_CHECKING` 处理复杂类型：

```python
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from langchain_tests.unit_tests import ChatModelUnitTests
```


必要时使用注释抑制类型错误（应谨慎使用）：

```python
# mypy cannot infer this type from the lambda
my_dict: dict[str, Any] = {}  # type: ignore[assignment]

# Intentional override
result = chain.invoke(message)  # type: ignore[return-value]
```


## 测试覆盖率与报告

生成覆盖率报告：

```bash
make coverage

# Reports generated:
# - coverage.xml (for CI)
# - term-missing (terminal output with uncovered lines)
```


## 关键测试基础设施文件

- **conftest.py** (`libs/core/tests/unit_tests/conftest.py`)：共享的夹具、标记和大规模配置
- **agents/conftest.py** (`libs/langchain_v1/tests/unit_tests/agents/conftest.py`)：用于检查点、存储和确定性 UUID 的智能体专用夹具
- **agents/model.py** (`libs/langchain_v1/tests/unit_tests/agents/model.py`)：`FakeToolCallingModel`，用于在不进行 API 调用的情况下测试智能体
- **agents/any_str.py** (`libs/langchain_v1/tests/unit_tests/agents/any_str.py`)：`AnyStr` 类，用于匹配前缀但与具体值无关的字符串
- **agents/messages.py** (`libs/langchain_v1/tests/unit_tests/agents/messages.py`)：消息匹配辅助工具，如 `_AnyIdHumanMessage`
- **stubs.py** (`libs/core/tests/unit_tests/stubs.py`)：用于带通配符 ID 的消息测试的辅助函数
- **pydantic_utils.py** (`libs/core/tests/unit_tests/pydantic_utils.py`)：跨版本 Pydantic 兼容性的模式规范化
- **fake/callbacks.py** (`libs/core/tests/unit_tests/fake/callbacks.py`)：`FakeCallbackHandler`，用于追踪事件
- **fake/test_fake_chat_model.py**：假模型测试示例

## 最佳实践

1. **隔离测试**：每个测试应独立，不依赖其他测试的状态。

2. **使用夹具**：将设置代码提取为夹具以实现复用和清晰性。

3. **模拟外部依赖**：模拟 API 调用、文件 I/O 和网络操作。

4. **测试行为而非实现**：测试组件的功能表现，而非其实现方式。

5. **参数化以减少重复**：使用 `@pytest.mark.parametrize` 处理多种输入情况。

6. **快照测试复杂结构**：使用 Syrupy 测试追踪、序列化对象和大型输出。

7. **记录测试意图**：使用清晰的测试名称和文档字符串。

8. **提交前运行测试**：使用预提交钩子或本地执行 `make test`。

9. **开发过程中持续类型检查**：开发时运行 `make lint` 或 `make type`。

10. **使用标记进行分类**：使用 `@pytest.mark.requires` 或自定义标记对测试进行分类，支持选择性执行。

11. **测试状态转换**：对于智能体和有状态组件，验证检查点边界上的状态更新。

12. **可预测地脚本工具调用**：使用 `FakeToolCallingModel` 控制工具调用，测试多轮交互流程。

13. **断言消息类型和内容**：验证智能体响应、工具调用和状态更新的结构。

14. **测试错误恢复**：验证智能体能从无效工具调用和其他错误中优雅恢复。

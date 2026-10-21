---
type: "Testing & QA"
title: "Integration Testing: Live API Tests and VCR Cassettes"
description: "How to write integration tests that call real model APIs with VCR cassette recording for CI compatibility, including environment setup, cassette management, and parameterization patterns."
tags: [integration-tests, vcr, cassettes, api-testing, pytest, ci-cd, model-testing]
verified:
  - by: openwiki/0.5.0
    at: 2026-09-28T08:35:20.640Z
sources:
  - id: openwiki-source-ff76574b014ac8b5c67560a6
    resource: repo://libs/langchain_v1/tests/integration_tests/chat_models/test_base.py
  - id: openwiki-source-bcf7be66f36f862f639f3c7a
    resource: repo://libs/langchain_v1/tests/integration_tests/conftest.py
  - id: openwiki-source-bae620f3bb8d2668c69ac079
    resource: repo://libs/langchain/tests/integration_tests/.env.example
  - id: openwiki-source-9a9f29414e5bd045f28c0081
    resource: repo://libs/partners/openai/Makefile
  - id: openwiki-source-df762860acfcc6abf0ce804b
    resource: repo://libs/partners/openai/pyproject.toml
  - id: openwiki-source-ed8ac34fd0216d557729038e
    resource: repo://libs/partners/openai/tests/conftest.py
  - id: openwiki-source-ce605960b642234837131282
    resource: repo://libs/partners/openai/tests/integration_tests/chat_models/conftest.py
  - id: openwiki-source-711811fb8df6b32329091d23
    resource: repo://libs/partners/openai/tests/integration_tests/chat_models/test_base_standard.py
  - id: openwiki-source-d619d2c7cfe653fb9b712740
    resource: repo://libs/partners/openai/tests/integration_tests/chat_models/test_base.py
  - id: openwiki-source-e758f7cc743f83c081224f08
    resource: repo://libs/partners/openai/tests/integration_tests/embeddings/test_base.py
  - id: openwiki-source-db02c1dda8563ab005cd9d62
    resource: repo://libs/standard-tests/langchain_tests/conftest.py
generated: { by: "openwiki/0.5.0", at: "2026-09-28T08:35:20.640Z" }
---
## 概述

LangChain 中的集成测试与单元测试有本质区别：它们会调用真实的模型 API（如 OpenAI、Anthropic 等），需要网络访问权限和有效的凭证。为了在不暴露 API 密钥或依赖外部服务的前提下，使这些测试可复现且适合 CI 环境，LangChain 使用 **VCR 录像带**——即录制的 HTTP 交互记录，在测试执行时进行重放。

本页涵盖集成测试的完整生命周期：环境设置、理解 VCR 模式、录像带组织方式、编写可在实时模式和录制模式下运行的测试、处理参数化问题，以及管理录像带刷新工作流。

## 环境设置：API 密钥与 .env 文件

集成测试需要有效的 API 凭证来录制录像带（仅一次）以及在计划或按需场景中运行实时集成测试。请将凭证存储在集成测试目录根路径下的 `.env` 文件中。

### .env 文件位置与格式

对于大多数包（例如 OpenAI 合作伙伴），结构如下：

```
libs/partners/openai/tests/.env
```

示例 `.env.example`（在代码库中可见）：

```bash
# openai
# your api key from https://platform.openai.com/account/api-keys
OPENAI_API_KEY=your_openai_api_key_here

# searchapi
SEARCHAPI_API_KEY=your_searchapi_api_key_here
```

**重要提示**：`.env` 文件已被加入 `.gitignore`，永远不会提交。请将 `.env.example` 复制为 `.env` 并本地填充有效密钥。

### 无密钥时自动跳过

集成测试不会在缺少 API 密钥时显式跳过。相反：

1. 标记为 `@pytest.mark.scheduled` 的测试（用于实时 API 调用）仅在计划的 CI 或显式选择时运行
2. 标记为 `@pytest.mark.vcr` 的测试通过 `make test_vcr` 在 CI 中运行，无需凭证
3. 未标记但使用 API 密钥的测试，若没有 `.env` 文件存在，将在运行时失败——这是开发环境中的有意设计

当本地运行 `make integration_tests` 时，必须在 `.env` 中拥有有效的 `OPENAI_API_KEY`。如果没有，测试将报错，提醒你需要凭证。

### 环境加载

测试 conftest 会在导入时自动加载 `.env`：

```python
from pathlib import Path
from dotenv import load_dotenv

PROJECT_DIR = Path(__file__).resolve().parent.parent

def _load_env() -> None:
    dotenv_path = PROJECT_DIR / "tests" / ".env"
    if dotenv_path.exists():
        load_dotenv(dotenv_path)

_load_env()
```

随后，凭证可通过 `os.environ["OPENAI_API_KEY"]` 或类似方式在测试中访问。

## VCR 模式：录制与回放

**VCR**（视频录像带，由 `vcrpy` 库实现）会记录 HTTP 交互——首次运行测试并连接真实 API 时，会捕获所有请求与响应。后续运行时，VCR 将使用已录制的录像带回放，而非发起实时网络调用。

### 录制阶段（仅由开发者执行一次）

当编写新集成测试或现有测试行为发生变化时：

1. 使用 `--record-mode=new_episodes`（或默认的 `once`）运行测试，并在 `.env` 中提供有效的 API 密钥
2. VCR 拦截测试所发出的所有 HTTP 调用
3. **在录制前**，请求/响应体中的敏感头信息和令牌会被 **清除**（替换为 `PLACEHOLDER` 或 `**REDACTED**`）
4. 录像带以 YAML 文件形式保存（可选地压缩为 `.yaml.gz`）
5. 录像带被提交到代码仓库

### 回放阶段（在 CI 和本地开发中始终启用）

当运行测试时：

1. VCR 加载录像带文件  
2. 不再进行真实的 API 调用，VCR 会拦截你的 HTTP 客户端并返回预先录制的响应  
3. 无需 API 密钥；不会暴露任何凭证  
4. 测试速度快（无网络延迟）且可确定性执行（每次返回相同结果）

### 录像带位置

录像带相对于测试模块存储在 `cassettes/` 子目录中：

```
libs/partners/openai/tests/
  integration_tests/
    chat_models/
      test_base.py
      cassettes/
        test_base/
          TestChatOpenAICodexStandard.test_invoke.yaml.gz
          test_chat_openai.yaml.gz
```

或存储在测试根目录下：

```
libs/partners/openai/tests/cassettes/
  test_langchain_openai_embeddings_equivalent_to_raw.yaml.gz
  test_streaming_tool_call_v1_v2_parity.yaml.gz
```

fixture `vcr_cassette_dir`（来自 `conftest.py`）会为每个测试模块计算正确的目录：

```python
@pytest.fixture(scope="module")
def vcr_cassette_dir(request: pytest.FixtureRequest) -> str:
    module = Path(request.module.__file__)
    return str(module.parent / "cassettes" / module.stem)
```

## 安全性：敏感数据清除

VCR 录像带是**提交到 git**的，并且在公共仓库中对全世界可见。为防止意外暴露 API 密钥、JWT、OAuth 令牌及其他秘密信息，LangChain 使用多层清除管道。

### 被清除的内容

**请求头**（在 `conftest.py` 中配置）：

```python
_EXTRA_HEADERS = [
    ("openai-organization", "PLACEHOLDER"),
    ("user-agent", "PLACEHOLDER"),
    ("authorization", "PLACEHOLDER"),
    ("cookie", "PLACEHOLDER"),
]
```

**请求体和响应体**（OAuth 秘密字段）：

```python
_OAUTH_SECRET_FIELDS = frozenset({
    "access_token",
    "refresh_token",
    "id_token",
    "code",
    "device_code",
    "client_secret",
})
```

清除操作通过专用处理器应用：

- **JSON 请求体**：递归遍历解析后的 JSON 树，对匹配 `_OAUTH_SECRET_FIELDS` 的字段进行清除  
- **表单编码体**：按 `&` 分割，对匹配的键进行清除  
- **JWT 模式**：使用正则表达式检测并清除体内的任意 JWT 格式字符串  

**二进制负载**（PNG、JPEG、PDF、音频等）被**跳过**——其魔数会被识别，清除栈将绕过以提升性能（JWT 和 OAuth 秘密均为 ASCII，因此二进制负载无法携带它们）。

### conftest.py 中的清除配置

```python
def remove_request_headers(request: Any) -> Any:
    """Remove sensitive headers and OAuth secrets from the request."""
    for k in request.headers:
        request.headers[k] = "**REDACTED**"
    request.uri = "**REDACTED**"
    request.body = _scrub_oauth_secrets(request.body)
    return request

@pytest.fixture(scope="session")
def vcr_config() -> dict:
    """Extend the default configuration coming from langchain_tests."""
    config = base_vcr_config()
    config["match_on"] = ["json_body"]  # Don't match on URI (it's redacted)
    config.setdefault("filter_headers", []).extend(_EXTRA_HEADERS)
    config["before_record_request"] = remove_request_headers
    config["before_record_response"] = remove_response_headers
    config["serializer"] = "yaml.gz"  # Compress cassettes
    return config
```

## 编写集成测试

### 基本结构

集成测试位于 `tests/integration_tests/` 目录中（与单元测试布局一致）：

```
libs/partners/openai/tests/
  unit_tests/
  integration_tests/
    chat_models/
      __init__.py
      conftest.py
      test_base.py
      cassettes/
```

一个使用 pytest-recording（自动钩入 VCR）的最小集成测试示例：

```python
import pytest
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

@pytest.mark.vcr  # Mark test to use VCR
def test_chat_openai_invoke():
    chat = ChatOpenAI(model="gpt-4o-mini")
    response = chat.invoke([HumanMessage(content="Hello")])
    assert response.content  # Assert the response is non-empty
```

`@pytest.mark.vcr` 标记告诉 pytest-recording 自动执行以下操作：

1. 在模块的 `cassettes/` 目录中查找名为 `test_chat_openai_invoke.yaml.gz` 的录像带文件  
2. 使用 VCR 拦截 HTTP 调用  
3. 若录像带存在则回放；若不存在或传入 `--record-mode=new_episodes` 则录制新录像带

### 测试分类标记

**`@pytest.mark.scheduled`**：标记用于**实时 API 调用**的测试（不使用录像带）。这些测试在带有真实凭证的定时 CI 工作流中运行。

```python
@pytest.mark.scheduled
def test_chat_openai_streaming_live():
    """Test streaming with a live API call."""
    chat = ChatOpenAI(model="gpt-4o-mini", streaming=True)
    response = chat.invoke("Hello")
    assert response.content
```

这些测试在 CI 中默认被跳过，除非显式选择或在定时任务中运行。本地运行时需要有效的 API 密钥。

**`@pytest.mark.vcr`**：标记用于**录像带回放**的测试。启用 VCR，回放录像带。CI 中无需 API 密钥。

```python
@pytest.mark.vcr
def test_chat_openai_with_tools():
    """Test tool calling with a cassette."""
    chat = ChatOpenAI(model="gpt-4o-mini")
    tools = [...]
    response = chat.invoke(..., tools=tools)
    assert response.tool_calls
```

**无标记**：未加标记的测试以**普通单元测试模式**运行，常用于初始化、验证或离线场景。不需要 API 密钥。

```python
def test_chat_openai_model_name():
    """Test model name is set correctly."""
    chat = ChatOpenAI(model="gpt-4o-mini")
    assert chat.model_name == "gpt-4o-mini"
```

### 参数化集成测试

当需要在同一场景下测试多个模型或配置时，使用 `pytest.mark.parametrize`：

```python
import pytest
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

MODELS = ["gpt-4o-mini", "gpt-4o"]

@pytest.mark.vcr
@pytest.mark.parametrize("model_name", MODELS)
def test_chat_models_invoke(model_name: str):
    """Test invoke on different models."""
    chat = ChatOpenAI(model=model_name)
    response = chat.invoke([HumanMessage(content="Hello")])
    assert response.content
```

VCR 会为每种参数组合自动生成独立的录像带文件：

```
cassettes/
  test_chat_models_invoke[gpt-4o-mini].yaml.gz
  test_chat_models_invoke[gpt-4o].yaml.gz
```

当测试以 `model_name="gpt-4o-mini"` 运行时，VCR 会加载匹配 `[gpt-4o-mini]` 的录像带。这使得可以使用独立记录的交互来测试多种模型。

### 流式测试

流式测试得益于录像带，因为它们能完整捕获整个流式序列：

```python
@pytest.mark.vcr
def test_chat_openai_streaming():
    """Test streaming behavior."""
    chat = ChatOpenAI(model="gpt-4o-mini")
    chunks = []
    for chunk in chat.stream("Hello"):
        chunks.append(chunk)
    assert len(chunks) > 0
    assert chunks[-1].content  # Final chunk has content
```

录像带记录了所有的流式 HTTP 数据块，VCR 在回放时按顺序重放这些数据。

### 错误路径测试

为了测试错误处理（如速率限制、格式错误的输入、API 错误），你可以选择以下任一方式：

1. **仅录制一次真实错误**（例如使用无效的 API 密钥或端点），然后重放错误响应  
2. **手动创建一个录像带**，其中包含预设的错误响应用于测试

示例：测试 `RateLimitError`：

```python
@pytest.mark.vcr
def test_chat_openai_handles_rate_limit():
    """Test that rate limit errors are handled gracefully."""
    chat = ChatOpenAI(model="gpt-4o-mini")
    try:
        response = chat.invoke("Hello")
    except Exception as e:
        assert "rate" in str(e).lower()
```

此测试的录像带应包含来自 API 的一条记录的 429（Too Many Requests）错误响应。

## 录像带管理

### 录制新录像带

要录制新的录像带或重新录制现有录像带：

```bash
cd libs/partners/openai

# Record new cassettes (default --record-mode=once)
uv run --group test --group test_integration pytest \
  tests/integration_tests/chat_models/test_base.py::test_chat_openai_invoke

# Or explicitly with --record-mode=new_episodes to overwrite
uv run --group test --group test_integration pytest \
  --record-mode=new_episodes \
  tests/integration_tests/chat_models/test_base.py::test_chat_openai_invoke
```

**前提条件**：
- `tests/.env` 中存在有效的 `OPENAI_API_KEY`
- 具备网络访问权限（显然）

**执行过程**：
1. pytest-recording 启动 VCR 于录制模式
2. 你的测试运行并发起对 API 的真实 HTTP 调用
3. VCR 拦截所有请求与响应
4. 清理函数移除敏感头信息和密钥
5. 录像带被序列化为 YAML 格式，压缩为 `.yaml.gz` 并保存

**提交录像带**：

```bash
git add libs/partners/openai/tests/cassettes/test_chat_openai_invoke.yaml.gz
git commit -m "Add cassette for chat_openai_invoke integration test"
```

### 刷新过期的录像带

当测试代码发生变化时（例如更改提示词、模型或参数），录像带可能变得过期。VCR 会尝试将新请求体与旧录像带进行匹配；若不匹配，则在回放模式下抛出 `CannotOverwriteExistingCassetteException` 错误，或建议重新录制。

要刷新：

```bash
cd libs/partners/openai
uv run --group test --group test_integration pytest \
  --record-mode=new_episodes \
  tests/integration_tests/chat_models/test_base.py::test_chat_openai_invoke
```

这将用新的交互覆盖原有录像带。请重新验证录像带是否已正确清理，然后提交。

### 录像带格式

录像带是 YAML 文件（可选 gzip 压缩）。典型的录像带结构如下：

```yaml
interactions:
  - request:
      body: null
      headers: {}
      method: POST
      uri: https://api.openai.com/v1/chat/completions
    response:
      body:
        string: '{"choices": [{"message": {"content": "Hello!", "role": "assistant"}}]}'
      headers:
        content-type:
          - application/json
      status:
        code: 200
        message: OK
version: 1
```

头信息和密钥会被替换为 `PLACEHOLDER` 或 `**REDACTED**`：

```yaml
request:
  headers:
    authorization:
      - "**REDACTED**"
    openai-organization:
      - "PLACEHOLDER"
```

### 检查录像带

要检查录像带内容，请解压并阅读：

```bash
gunzip -c libs/partners/openai/tests/cassettes/test_invoke.yaml.gz | head -50
```

或者，为了永久查看：

```bash
gunzip libs/partners/openai/tests/cassettes/test_invoke.yaml.gz
# Now test_invoke.yaml is readable
```

（记得在提交前重新压缩或重新录制。）

## 运行集成测试

### 完整的集成测试套件

在真实 API 上运行所有集成测试（需要凭证）：

```bash
cd libs/partners/openai
make integration_tests  # Runs full suite with live API calls
```

这会调用：

```bash
uv run --with 'openai>=2.45.0,<3.0.0' --group test --group test_integration pytest \
  -v --tb=short tests/integration_tests/

uv run --with 'openai>=3.0.0,<4.0.0' --group test --group test_integration pytest \
  -v --tb=short tests/integration_tests/
```

同时还会对每个受支持的 OpenAI SDK 主版本（2.x 和 3.x）进行单次实时请求的冒烟测试。

### VCR 录像带测试（CI 模式）

仅使用录像带运行集成测试（不进行实时 API 调用，无需凭证）：

```bash
cd libs/partners/openai
make test_vcr
```

这会调用：

```bash
uv run --group test pytest --record-mode=none -m vcr tests/integration_tests/
```

`--record-mode=none` 标志告诉 VCR **仅回放**录像带，并在录像带缺失时失败（从不尝试实时调用或录制新内容）。

这是 CI 所运行的内容；它确保录制文件（cassettes）是最新的，并且在不暴露密钥的情况下使测试可重复。

### 定期集成测试

定期的 CI 工作流（每日或按需触发）会使用真实凭证运行实时集成测试：

```bash
cd libs/partners/openai
make integration_tests
```

仅标记为 `@pytest.mark.scheduled` 的测试会在这些工作流中运行。详情请参见 [CI/CD 工作流](repo:///openwiki/ci-workflows.md#integration-test-compilation)。

## 流式与异步集成测试

### 异步测试

异步集成测试的工作方式与同步测试相同：

```python
@pytest.mark.vcr
async def test_chat_openai_ainvoke():
    """Test async invoke."""
    chat = ChatOpenAI(model="gpt-4o-mini")
    response = await chat.ainvoke("Hello")
    assert response.content
```

VCR 会拦截底层的 `httpx`（或 `requests`）库，因此同步和异步的 HTTP 调用都会被相同地录制和回放。

### 使用 `stream_events` 的流式处理

使用 `stream_events`（LangChain 的事件流式 API）的测试同样可以与录制文件配合使用：

```python
@pytest.mark.vcr
def test_chat_openai_stream_events():
    """Test streaming events."""
    chat = ChatOpenAI(model="gpt-4o-mini")
    events = []
    for event in chat.stream_events("Hello", version="v2"):
        events.append(event)
    assert len(events) > 0
```

底层的 HTTP 调用会被记录在录制文件中；事件结构将被精确回放。

## VCR 配置细节

### conftest.py 设置

每个包的 `conftest.py` 文件全局配置 VCR：

```python
from langchain_tests.conftest import CustomPersister, CustomSerializer, base_vcr_config

@pytest.fixture(scope="session")
def vcr_config() -> dict:
    """Extend the default configuration coming from langchain_tests."""
    config = base_vcr_config()
    # Base config:
    # - record_mode: "once"
    # - filter_headers: ["authorization", "x-api-key", "api-key"]
    # - match_on: ["method", "uri", "body"]
    # - cassette_library_dir: "tests/cassettes"
    
    # Extend for OpenAI:
    config["match_on"] = ["json_body"]  # Custom JSON matching
    config["serializer"] = "yaml.gz"     # Compress cassettes
    config["before_record_request"] = remove_request_headers
    config["before_record_response"] = remove_response_headers
    return config

def pytest_recording_configure(config: dict, vcr: VCR) -> None:
    """Register custom serializer, persister, and matchers."""
    vcr.register_persister(CustomPersister())
    vcr.register_serializer("yaml.gz", CustomSerializer())
    vcr.register_matcher("json_body", _json_body_matcher)
```

### 录制模式

VCR 支持多种录制模式：

- **`once`**（默认）：录制新录制文件；回放已存在的录制文件。适用于初始开发阶段。
- **`new_episodes`**：即使录制文件已存在，也进行录制或覆盖。用于刷新过期的录制文件。
- **`none`**：从不录制；仅回放。用于 CI 环境。若录制文件缺失则测试失败。
- **`all`**：始终录制，即使录制文件已存在。应谨慎使用。

### 请求匹配规则

默认情况下，VCR 按以下方式匹配请求：

```python
match_on: ["method", "uri", "body"]
```

这意味着：“当 HTTP 方法、URI 和请求体完全相同时，请求即与录制文件匹配。”

对于请求体非确定性（例如包含时间戳）的 API，你可以自定义匹配器，或使用 `allow_playback_repeats` 来对多个请求变体复用同一响应。

OpenAI 的测试使用了自定义的 `json_body` 匹配器：

```python
config["match_on"] = ["json_body"]  # Only match on JSON-parsed body (ignores whitespace/key order)
```

这使得即使 JSON 键的顺序不同，录制文件也能成功匹配。

## 与 CI/CD 的集成

### CI 工作流

LangChain 的 CI 系统在多个环节运行集成测试：

1. **拉取请求检查**（`check_diffs.yml`）：编译集成测试但不运行
2. **VCR 录制文件测试**（`_test_vcr.yml`）：以仅回放模式运行基于录制文件的集成测试
3. **定期集成测试**（`integration_tests.yml`）：每日远程 API 测试，使用真实凭证对接合作伙伴库

完整详情请参见 [CI/CD 工作流：GitHub Actions 与发布流程](repo:///openwiki/ci-workflows.md)，包括定期工作流和录制模式管理。

### CI 中的录制文件验证

`test_vcr` 任务以 `--record-mode=none` 运行，这意味着：

- 录制文件在 CI 中**从不被录制或覆盖**
- 若录制文件缺失，测试**立即失败**
- 若请求体不匹配，测试**立即失败**

这能捕获因测试代码变更导致的过期录制文件，而无需执行实际调用。

## 扩展点与自定义

### 自定义 VCR 匹配器

对于高级匹配逻辑（例如忽略某些请求字段），请在 `conftest.py` 中注册自定义匹配器：

```python
def _custom_matcher(r1: Any, r2: Any) -> None:
    """Match requests with custom logic."""
    # Example: match method and URI, ignore timestamps in body
    assert r1.method == r2.method
    assert r1.uri == r2.uri
    # Don't check body

@pytest.fixture(scope="session")
def vcr_config() -> dict:
    config = base_vcr_config()
    config["match_on"] = ["custom"]
    return config

def pytest_recording_configure(config: dict, vcr: VCR) -> None:
    vcr.register_matcher("custom", _custom_matcher)
```

### 自定义清理器

用于添加对专有密钥的额外清理：

```python
def remove_proprietary_header(request: Any) -> Any:
    """Remove custom header."""
    request.headers.pop("X-Proprietary-Token", None)
    return request

@pytest.fixture(scope="session")
def vcr_config() -> dict:
    config = base_vcr_config()
    config["before_record_request"] = remove_proprietary_header
    return config
```

## 标准测试套件继承

LangChain 合作伙伴包通过 `langchain_tests.integration_tests` 中的 `ChatModelIntegrationTests` 继承标准集成测试套件。这使得所有聊天模型提供商都能运行一组通用测试，而无需重复编写测试代码。

### 工作原理

每个合作伙伴包定义一个继承自 `ChatModelIntegrationTests` 的测试类：

```python
# libs/partners/openai/tests/integration_tests/chat_models/test_base_standard.py
from langchain_tests.integration_tests import ChatModelIntegrationTests
from langchain_openai import ChatOpenAI

class TestOpenAIStandard(ChatModelIntegrationTests):
    @property
    def chat_model_class(self) -> type[BaseChatModel]:
        return ChatOpenAI

    @property
    def chat_model_params(self) -> dict:
        return {"model": "gpt-4o-mini"}

    @property
    def supports_image_inputs(self) -> bool:
        return True

    @property
    def has_tool_calling(self) -> bool:
        return True
```

基类（`ChatModelIntegrationTests`）实现了数十个测试方法，用于测试标准聊天模型功能：`invoke`、`ainvoke`、`stream`、`astream`、工具调用、结构化输出、图像输入等。每个测试方法都自动标记为 `@pytest.mark.vcr`，以便录制和回放录像带。

### 可定制点

合作伙伴实现通过重写属性来声明：

- **`chat_model_class`**：要实例化的模型类
- **`chat_model_params`**：默认参数（模型名称、API 密钥等）
- **`supports_*` 属性**：声明模型支持的功能（图像、工具、JSON 模式等）
- **`has_*` 属性**：声明模型能力（工具调用、结构化输出等）
- **自定义方法**：重写测试方法以提供特定于供应商的行为（例如处理音频输入、推理模型）

### 与标准测试的 VCR 集成

标准测试方法继承了 VCR 集成：

1. 每个测试方法已预先标记为 `@pytest.mark.vcr`
2. 录像带会自动生成，文件名如 `test_invoke.yaml.gz`、`test_invoke[model0].yaml.gz`（用于参数化变体）
3. 通过 `make test_vcr --record-mode=none` 在 CI 中运行测试，以验证录像带是否最新
4. 当合作伙伴更新测试（例如向测试中添加新模型）时，使用 `--record-mode=new_episodes` 记录新的录像带

这种方法确保所有 LangChain 合作伙伴保持一致的测试覆盖率，并且标准测试套件的任何更改（新增测试方法、新增功能）都会被所有合作伙伴自动采纳。

## 最佳实践

1. **始终清理录像带**：使用 `before_record_request` 和 `before_record_response` 来屏蔽所有敏感数据。提交前检查录像带内容。

2. **提交录像带**：录像带是测试套件的一部分。将其提交到 git，以便他人无需实时 API 访问即可运行测试。

3. **保持录像带小巧**：避免录制大型响应（例如长文档的嵌入）。如需，可使用手动创建录像带并加入预设数据。

4. **正确使用标记**：
   - `@pytest.mark.vcr`：用于基于录像带的测试
   - `@pytest.mark.scheduled`：用于仅限实时的测试
   - 无标记：用于离线测试

5. **针对多个模型/版本进行测试**：使用 `@pytest.mark.parametrize` 测试多种配置，并为每种配置生成独立的录像带。

6. **记录文档回放依赖项**：如果一个回放依赖于特定模型行为（例如推理模型），请在测试中进行说明。

7. **代码变更时刷新回放**：如果修改了测试（如新的提示词、新参数），请使用 `--record-mode=new_episodes` 刷新回放。

8. **审查回放差异**：提交回放更新时，请在 PR 中审查差异，确保没有泄露敏感信息。

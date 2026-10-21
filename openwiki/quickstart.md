---
type: "Getting Started"
title: "LangChain Repository Quick Start"
description: "Entry point for engineers: orient to the monorepo structure, run first tests, understand what to edit for common tasks, and route to major development areas."
tags: [quickstart, getting-started, monorepo, setup, development, first-steps, cli-reference]
sources:
  - id: openwiki-source-4d1645cb6317345817452838
    resource: repo://.pre-commit-config.yaml
  - id: openwiki-source-8e384445ccfbaf00747b3e18
    resource: repo://libs/core/langchain_core/__init__.py
  - id: openwiki-source-96c73d2c05223b0b46abdbe9
    resource: repo://libs/core/langchain_core/callbacks/__init__.py
  - id: openwiki-source-c52037e7b642f7ac5a7642a8
    resource: repo://libs/core/langchain_core/language_models/chat_models.py
  - id: openwiki-source-0f6ea1dd09fb4675ff4112b1
    resource: repo://libs/core/langchain_core/messages/__init__.py
  - id: openwiki-source-e7908d069731cffec228727e
    resource: repo://libs/core/langchain_core/output_parsers/__init__.py
  - id: openwiki-source-c46a1d181ab64e61460c84c6
    resource: repo://libs/core/langchain_core/prompts/__init__.py
  - id: openwiki-source-65071982a626569c8820a34b
    resource: repo://libs/core/langchain_core/runnables/__init__.py
  - id: openwiki-source-7c4bed110359f4f5c7847c8b
    resource: repo://libs/core/langchain_core/tools/__init__.py
  - id: openwiki-source-8f1875229ad4a704c8e20a06
    resource: repo://libs/core/Makefile
  - id: openwiki-source-3486a94e6eb23a78271a5bfb
    resource: repo://libs/core/pyproject.toml
  - id: openwiki-source-47db752fe27393d5d4825827
    resource: repo://libs/langchain_v1/langchain/__init__.py
  - id: openwiki-source-71e882e1ac9757ea8e959a7c
    resource: repo://libs/langchain_v1/langchain/agents/factory.py
  - id: openwiki-source-07e634f5cd5f00c636010306
    resource: repo://libs/langchain_v1/langchain/agents/middleware/__init__.py
  - id: openwiki-source-b09b1477098d69af6abaa5b4
    resource: repo://libs/langchain_v1/langchain/chat_models/__init__.py
  - id: openwiki-source-49fbcc45434b619b68220bf9
    resource: repo://libs/Makefile
  - id: openwiki-source-1e66a9da38565f8901e651f4
    resource: repo://libs/partners/openai/langchain_openai/__init__.py
  - id: openwiki-source-48ce5ee900993294d349b4e8
    resource: repo://libs/standard-tests/langchain_tests/__init__.py
generated: { by: "openwiki/0.5.0", at: "2026-10-19T08:31:23.057Z" }
verified:
  - by: openwiki/0.5.0
    at: 2026-10-19T08:31:23.057Z
---
## 欢迎使用 LangChain 开发

LangChain 是一个智能体工程平台——一个用于构建基于大语言模型（LLM）的应用程序的框架，具备可组合的抽象结构、提供商集成以及编排原语。本页面将引导您了解仓库的目录结构、必要的设置流程、常见的开发任务，并指引您深入查阅更详细的文档。

**首次接触此仓库？** 请先阅读 [安装与设置](#installation--setup)，然后跳转至 [快速导航](#quick-navigation-to-major-areas) 查找您需要工作的部分。

**想要完整概览？** 参见 [架构概览](/openwiki/architecture.md) 了解系统设计，[开发命令](/openwiki/dev-commands.md) 获取详细的 CLI 参考，以及 [源码地图](/openwiki/source-map.md) 根据主题定位代码位置。

## 单体仓库概览

LangChain 采用 **三层架构**，位于 `/libs/` 目录下：

```
/libs/
├── core/              # langchain-core (v1.6.12): Base abstractions (Runnable, BaseChatModel, tools, prompts, messages)
├── langchain_v1/      # langchain (v1.4.6): Agent orchestration, factory, middleware
├── partners/          # Provider-specific integrations (OpenAI, Anthropic, Ollama, etc.)
├── standard-tests/    # Shared test suites for component conformance
├── text-splitters/    # Text splitting utilities
├── model-profiles/    # LLM metadata and capability profiles
└── Makefile           # Monorepo-level build targets
```

### 各层编辑场景说明

| 层级 | 编辑场景 | 关键文件 |
|------|--------|----------|
| **core** | 添加或修改基础抽象和核心接口：`Runnable`、`BaseChatModel`、消息、工具、提示词、回调、输出解析器。 | `libs/core/langchain_core/` |
| **langchain_v1** | 构建智能体工厂功能（`create_agent`）、中间件组合、模型初始化（`init_chat_model`），或高层编排逻辑。 | `libs/langchain_v1/langchain/agents/factory.py`、`libs/langchain_v1/langchain/chat_models/base.py` |
| **partners/{name}** | 添加新的大语言模型提供商（如 OpenAI、Anthropic 等）、实现 `ChatModel`、处理消息转换，或添加提供商特定功能（流式、工具调用、结构化输出）。 | `libs/partners/{provider}/langchain_{provider}/chat_models/base.py` |
| **standard-tests** | 定义可复用的测试套件和测试固定项，用于跨所有提供商评估聊天模型、嵌入和工具。 | `libs/standard-tests/langchain_tests/` |
| **model-profiles** | 发布模型元数据、能力配置、上下文窗口及支持的功能信息，供 `init_chat_model` 发现使用。 | `libs/model-profiles/langchain_model_profiles/` |

## 安装与设置

### 克隆仓库

```bash
git clone https://github.com/langchain-ai/langchain.git
cd langchain
```

### 使用 `uv` 安装依赖

该单体仓库使用 `uv` 实现快速且确定性的依赖解析。首次安装一次即可：

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Or via Homebrew
brew install uv
```

随后在您的包中同步所有依赖：

```bash
# From any libs/ subdirectory, install all groups (test, lint, type, typing, dev)
uv sync --all-groups

# Or install only what you need
uv sync --group test          # For running tests
uv sync --group test_integration  # For integration tests with VCR cassettes
uv sync --group lint          # For ruff formatting
uv sync --group typing        # For mypy type checking
uv sync --group dev           # For dev tools (Jupyter, setuptools, etc.)
```

### 预提交钩子

安装 Git 钩子以自动强制执行代码质量规范：

```bash
pre-commit install

# To manually run all hooks
pre-commit run --all-files

# To run a specific hook
pre-commit run ruff --all-files
```

预提交钩子会执行以下操作：
- YAML/TOML 语法验证
- 文本规范化与尾部空白修复
- **按包格式化与 linting**（ruff、mypy）
- **版本一致性检查**，覆盖所有 `pyproject.toml` 文件

## 常见开发任务

### 运行单元测试

```bash
# From any package directory (libs/core, libs/langchain_v1, etc.)
make test

# Run a specific test file
make test TEST_FILE=tests/unit_tests/agents/test_factory.py

# Run tests in watch mode (auto-rerun on file changes)
make test_watch

# Run extended tests (marked @pytest.mark.requires)
make extended_tests
```

**关键细节：**
- 测试运行时启用 **套接字限制**（`--disable-socket`），防止意外网络调用
- 测试通过 pytest-xdist **并行执行**（`-n auto`）
- 清除 LangSmith 追踪变量，确保测试环境隔离
- 通常测试路径与源码路径一致：`langchain_core/runnables/base.py` → `tests/unit_tests/runnables/test_base.py`

### 格式化与代码检查

```bash
# Format all Python files (ruff)
make format

# Check linting issues (ruff, mypy)
make lint

# Type checking only (mypy)
make type

# Format only changed files (git diff against main)
make format_diff
```

**使用工具：**
- **ruff**：快速的 Python 代码检查与格式化工具（替代 black、isort、flake8）。通过 `uv run --group lint` 运行
- **mypy**：静态类型检查器。通过 `uv run --group typing` 运行
- 两者均已集成至 pre-commit 钩子和 make 目标中

### 运行集成测试

集成测试会调用真实的模型 API，并使用录制的响应（VCR 比较文件）：

```bash
# From any package directory
make integration_tests

# Run a specific integration test
make integration_tests TEST_FILE=tests/integration_tests/test_specific.py

# Record new cassettes (requires API credentials in .env)
make integration_tests RECORD=true
```

有关比较文件的详细管理，请参阅 [集成测试](/openwiki/integration-tests.md)。

### 完整的本地验证

在提交 PR 前请运行以下命令：

```bash
# From your package directory
make format && make lint && make test
```

或一行命令执行：

```bash
cd libs/core && make format lint test
```

若需包含集成测试：

```bash
cd libs/langchain_v1 && make format lint test integration_tests
```

## 快速导航至主要功能区

使用下表可快速跳转至详细文档：

| 任务 | 入门点 | 关键概念 |
|------|-----------|--------------|
| **构建一个智能体** | [智能体工厂](/openwiki/agent-factory.md) | create_agent, AgentState, 中间件组合, 图执行 |
| **添加新的 LLM 提供商** | [添加聊天模型提供商](/openwiki/partner-pattern.md) | ChatModel 实现, 消息转换, 提供商注册, 标准测试 |
| **集成 OpenAI（ChatGPT、o1 等）** | [OpenAI 集成](/openwiki/openai-provider.md) | ChatOpenAI, 响应 API, 视觉支持, 流式输出, 工具调用, Azure |
| **理解架构设计** | [架构概览](/openwiki/architecture.md) | 三层设计, 依赖流向, 核心组件 vs. 协调组件 vs. 合作伙伴组件 |
| **使用聊天模型** | [聊天模型接口](/openwiki/chat-models.md) | BaseChatModel 协议, 流式输出, 工具绑定, 结构化输出 |
| **动态初始化模型** | [模型初始化](/openwiki/model-initialization.md) | init_chat_model 工厂函数, provider:model 语法, 回退链 |
| **组合组件（链、管道）** | [Runnable 与可组合性](/openwiki/runnables.md), [可组合性](/openwiki/composability.md) | Runnable 协议, \| 操作符, 分支逻辑, 重试机制, 回退策略 |
| **使用工具** | [工具](/openwiki/tools.md) | BaseTool, 模式生成, 工具调用, 结果处理 |
| **流式输出响应** | [流式输出](/openwiki/streaming.md) | 逐个 token 输出, 跨组件流式传输 |
| **强制响应格式** | [结构化输出](/openwiki/structured-output.md) | JSON 模式, 响应校验, 类型化输出 |
| **编写中间件** | [智能体中间件](/openwiki/middleware.md) | 中间件类型, 组合方式, 自定义钩子 |
| **追踪智能体执行过程** | [智能体执行流程](/openwiki/agent-execution.md) | 运行时生命周期, 循环控制, 状态转换 |
| **添加可观测性** | [回调与追踪](/openwiki/callbacks.md) | 回调管理器, LangSmith 集成, 日志记录 |
| **编写单元/集成测试** | [单元测试](/openwiki/unit-tests.md), [集成测试](/openwiki/integration-tests.md) | 测试结构, 固定数据, 模拟, VCR 比较文件 |
| **处理提示词** | [提示词](/openwiki/prompts.md) | 模板, 少样本学习, 变量替换, 图像处理 |
| **理解消息类型** | [消息](/openwiki/messages.md) | AIMessage, ToolMessage, 内容块, 提供商转换 |
| **使用模型上下文协议** | [MCP 集成](/openwiki/mcp-integration.md) | MCP 服务器, 工具适配器, 诱导机制 |
| **参考所有文件路径** | [源码映射](/openwiki/source-map.md) | 概念到路径的查找表, 目录结构 |
| **检查 CI/CD 工作流** | [CI/CD 工作流](/openwiki/ci-workflows.md) | GitHub Actions, 测试, 代码检查, 发布流程 |
| **开发命令参考** | [开发命令](/openwiki/dev-commands.md) | 详细的 make 目标, uv 语法, 环境配置 |

## 仓库结构概览

### 根目录层级

```
/
├── .github/              # GitHub Actions workflows (CI/CD)
├── .pre-commit-config.yaml # Pre-commit hooks definition
├── .vscode/              # VS Code settings
├── libs/                 # Main monorepo workspace
├── AGENTS.md             # Contributing guide (READ THIS BEFORE PR)
└── README.md             # Top-level project overview
```

### `/libs/` 目录内

**core/** — 基础抽象（langchain-core 包）

```
core/
├── langchain_core/
│   ├── language_models/  # BaseChatModel and language model contracts
│   ├── messages/         # Message types and content blocks
│   ├── runnables/        # Runnable protocol and operators
│   ├── tools/            # BaseTool and tool utilities
│   ├── prompts/          # Prompt templates and few-shot
│   ├── callbacks/        # Callback manager and handlers
│   └── output_parsers/   # Output parsing and validation
├── tests/unit_tests/     # Unit tests (no network)
├── tests/integration_tests/ # Integration tests (live APIs)
├── Makefile              # Build targets (test, lint, format)
└── pyproject.toml        # Package deps and metadata
```

**langchain_v1/** — 智能体编排（langchain 包）

```
langchain_v1/
├── langchain/
│   ├── agents/
│   │   ├── factory.py    # create_agent function
│   │   ├── middleware/   # Pluggable middleware hooks
│   │   └── structured_output.py # Response schema
│   ├── chat_models/
│   │   └── base.py       # init_chat_model factory
│   ├── mcp/              # Model Context Protocol
│   └── ...
├── tests/unit_tests/
├── tests/integration_tests/
├── tests/cassettes/      # VCR cassettes for HTTP mocking
├── Makefile
└── pyproject.toml
```

**partners/** — 服务商集成

```
partners/
├── openai/               # ChatOpenAI (Chat Completions & Responses APIs), embeddings
├── anthropic/            # ChatAnthropic (Claude)
├── ollama/               # ChatOllama (local models)
├── groq/                 # ChatGroq
├── mistralai/            # ChatMistralAI
├── huggingface/          # HuggingFace models/embeddings
├── deepseek/             # ChatDeepSeek
└── ... (20+ more providers)
```

每个服务商都具有相同的结构：

```
provider/
├── langchain_{provider}/
│   ├── __init__.py       # Exports ChatModel class
│   ├── chat_models/
│   │   └── base.py       # ChatModel implementation
│   └── data/             # Model profiles
├── tests/
│   ├── unit_tests/       # Standard tests + custom
│   └── integration_tests/
├── pyproject.toml
├── Makefile
└── uv.lock
```

部分服务商（如 OpenAI）支持高级 API 模式——详情请参见 [OpenAI 集成](/openwiki/openai-provider.md)，了解 Responses API 和结构化输出的流式处理。

## 你的首个 PR：工作流程

### 1. 选择任务

使用上方的 [快速导航](#quick-navigation-to-major-areas) 表决定你要处理的内容。对于首次贡献者：
- **简单**：添加测试、修复类型错误、改进文档
- **中等**：添加新的中间件钩子、扩展工具接口
- **困难**：添加新的服务商集成（请参考 [添加聊天模型服务商](/openwiki/partner-pattern.md)）

### 2. 阅读贡献指南

在编码前，请阅读：
- **[AGENTS.md](repo://AGENTS.md)** — 贡献规范、风格及 PR 期望
- **相关维基页面** — 你所在领域的深入背景（参见上表）

### 3. 设置你的包环境

```bash
cd libs/{core|langchain_v1|partners/provider}
uv sync --all-groups
pre-commit install
```

### 4. 进行修改

遵循代码库中已有的风格与模式。使用类型提示；编写代码的同时一并编写测试。

### 5. 运行本地检查

```bash
make format lint test
```

所有检查必须通过后才能推送。

### 6. 提交并推送

遵循 [Conventional Commits](https://www.conventionalcommits.org/) 格式，并包含作用域（必需）：

```bash
git add .
# Format: type(scope): description
# Example: feat(core): add streaming support to BaseChatModel
# Example: fix(openai): handle timeout errors gracefully
git commit -m "type(scope): description"
git push origin your-branch
```

详见 [AGENTS.md](repo://AGENTS.md) 中的提交规范和分支命名规则（`<username>/<scope>/<description>`）。

预提交钩子会自动运行。若失败，请修复后再次提交。

### 7. 打开拉取请求（PR）

将 PR 关联到相关问题，并在描述中引用你阅读过的维基页面。LangChain 团队将进行审查并提供反馈。

## 必须了解的关键文件

| 文件 | 用途 |
|------|------|
| `AGENTS.md` | 贡献指南、风格与规范 |
| `libs/Makefile` | 单体仓库级别的 make 目标（lock，check-lock） |
| `libs/{core,langchain_v1,partners/*/Makefile` | 各包的测试、lint、格式化目标 |
| `.pre-commit-config.yaml` | Git 钩子，用于代码质量控制 |
| `pyproject.toml`（各包内） | 包元数据、依赖项、构建配置 |

## 故障排除

### 测试因套接字错误而失败
默认情况下，测试会限制套接字访问。如需网络访问：
- 在 `tests/integration_tests/` 中编写集成测试（参见 [集成测试](/openwiki/integration-tests.md)）
- 或本地禁用套接字限制：`uv run --group test pytest --disable-socket=false ...`

### 导入错误或版本不匹配
重新生成锁定文件：

```bash
cd libs
make lock
```

或在单个包中操作：

```bash
cd libs/core
uv lock
```

### 类型检查失败
运行 mypy 查看详细错误信息：

```bash
make type
```

请查阅 [聊天模型](/openwiki/chat-models.md) 或 [Runnable](/openwiki/runnables.md) 页面，了解类型签名模式。

### 提交前钩子：阻止提交
提交前钩子会自动修复格式和部分问题。请重新暂存并提交：

```bash
git add .
git commit -m "..."  # Try again
```

如果静态检查仍然失败，请运行 `make lint` 查看详细信息并手动修复。

## 快速命令参考

```bash
# Setup
uv sync --all-groups          # Install all dependencies
pre-commit install            # Setup git hooks

# Testing
make test                      # Run unit tests
make test TEST_FILE=path/     # Run specific test file
make test_watch               # Watch mode (auto-rerun)
make integration_tests        # Run integration tests

# Code Quality
make format                   # Format code (ruff)
make lint                     # Check linting (ruff, mypy)
make type                     # Type check only (mypy)

# Lockfile Management
cd libs && make lock          # Regenerate all lockfiles
cd libs && make check-lock    # Verify lockfiles are up-to-date

# All Before PR
make format && make lint && make test
```

## 下一步

1. **阅读 [AGENTS.md](repo://AGENTS.md)** 了解贡献规范  
2. **从 [快速导航](#quick-navigation-to-major-areas) 中选择一个与你任务匹配的维基页面**  
3. **克隆仓库，配置环境，并进行首次修改**  
4. **运行 `make format lint test`** 在本地验证  
5. **打开 Pull Request（PR）并参与团队协作**

欢迎加入 LangChain！🚀

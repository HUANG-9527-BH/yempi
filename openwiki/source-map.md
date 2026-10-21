---
type: "Reference"
title: "Source Map: Repository File Organization"
description: "Quick reference for locating code by topic, mapping LangChain concepts to their implementation paths across the monorepo including core abstractions, agents, middleware, partners, and configuration files."
tags: [reference, file-organization, monorepo, codebase-map, pathfinding]
sources:
  - id: openwiki-source-4d1645cb6317345817452838
    resource: repo://.pre-commit-config.yaml
  - id: openwiki-source-1c233fccf5a66b84d0045366
    resource: repo://libs/core/langchain_core/callbacks/manager.py
  - id: openwiki-source-c52037e7b642f7ac5a7642a8
    resource: repo://libs/core/langchain_core/language_models/chat_models.py
  - id: openwiki-source-b32b84365d17276620c41ebc
    resource: repo://libs/core/langchain_core/messages/base.py
  - id: openwiki-source-03d7415879ed05a392edd62d
    resource: repo://libs/core/langchain_core/prompts/base.py
  - id: openwiki-source-a1981e868973f6fd7f71e12e
    resource: repo://libs/core/langchain_core/runnables/base.py
  - id: openwiki-source-4ff475d7b00540f962384251
    resource: repo://libs/core/langchain_core/tools/base.py
  - id: openwiki-source-3486a94e6eb23a78271a5bfb
    resource: repo://libs/core/pyproject.toml
  - id: openwiki-source-a690cf632a02205e3f555be8
    resource: repo://libs/core/tests/unit_tests/test_tools.py
  - id: openwiki-source-71e882e1ac9757ea8e959a7c
    resource: repo://libs/langchain_v1/langchain/agents/factory.py
  - id: openwiki-source-03e8ca0eebe37feda8566793
    resource: repo://libs/langchain_v1/langchain/agents/middleware/types.py
  - id: openwiki-source-ec30ab6256dd50cc670919f6
    resource: repo://libs/langchain_v1/langchain/agents/structured_output.py
  - id: openwiki-source-c479d4fffee5cf62576699e4
    resource: repo://libs/langchain_v1/langchain/chat_models/base.py
  - id: openwiki-source-0a3228970b0eadc4bcadbb5d
    resource: repo://libs/langchain_v1/langchain/mcp/adapter.py
  - id: openwiki-source-ba4876d385d4d18ed4fa0342
    resource: repo://libs/langchain_v1/pyproject.toml
  - id: openwiki-source-94d5218d78dfd52679adc96b
    resource: repo://libs/langchain_v1/tests/unit_tests/test_imports.py
  - id: openwiki-source-49fbcc45434b619b68220bf9
    resource: repo://libs/Makefile
  - id: openwiki-source-77f5d6298c73161b4d4f697e
    resource: repo://libs/model-profiles/langchain_model_profiles/__init__.py
  - id: openwiki-source-738512768ef81ae009b097ac
    resource: repo://libs/partners/openai/langchain_openai/chat_models/base.py
  - id: openwiki-source-b65307f1d05bd74532af2eee
    resource: repo://libs/partners/openai/langchain_openai/decisions/base.py
  - id: openwiki-source-bd29e79613d5f366a00068f5
    resource: repo://libs/standard-tests/langchain_tests/base.py
generated: { by: "openwiki/0.5.0", at: "2026-10-19T08:31:23.057Z" }
verified:
  - by: openwiki/0.5.0
    at: 2026-10-19T08:31:23.057Z
---
## 概览

本页提供在 LangChain 代码仓库中按主题快速定位代码的参考指南。该仓库采用多包工作区结构，具有三层架构：**langchain-core**（基础抽象）、**langchain**（编排与智能体）和 **partners**（提供商集成）。使用此地图可直接导航至负责特定概念的代码。

**当前版本**：langchain-core v1.6.12，langchain v1.4.6

## 概念到路径映射

| 概念 | 主要路径 | 目的 |
|------|----------|------|
| **智能体工厂** | `repo://libs/langchain_v1/langchain/agents/factory.py` | 为智能体循环构建已编译的 LangGraph 状态机，支持模型绑定、工具执行和中间件组合 |
| **智能体中间件** | `repo://libs/langchain_v1/langchain/agents/middleware/` | 可插拔的钩子，用于模型调用、工具调用和生命周期事件（重试、人工介入、数据脱敏等） |
| **聊天模型** | `repo://libs/core/langchain_core/language_models/chat_models.py` | `BaseChatModel` 抽象基类及统一的提供商接口，支持流式传输、批量处理和 token 计数 |
| **聊天模型（核心接口）** | `repo://libs/core/langchain_core/language_models/` | 基础语言模型类、用于测试的假模型以及兼容性桥接 |
| **聊天模型（合作伙伴实现）** | `repo://libs/partners/*/`（例如 `openai/`、`anthropic/`、`ollama/`） | 服务商特定实现：ChatOpenAI、ChatAnthropic、ChatOllama 等 |
| **决策（OpenAI）** | `repo://libs/partners/openai/langchain_openai/decisions/` | 集成 OpenAI 结构化输出 API，实现确定性响应序列化与路由 |
| **回调与追踪** | `repo://libs/core/langchain_core/callbacks/` | 回调管理器、基础处理器、流式输出以及 LangSmith 集成 |
| **消息** | `repo://libs/core/langchain_core/messages/` | 消息类型（AIMessage、HumanMessage、SystemMessage、ToolMessage）和内容块 |
| **模型初始化** | `repo://libs/langchain_v1/langchain/chat_models/base.py` | `init_chat_model()` 工厂函数，支持通过提供商:模型标识符动态发现并加载模型 |
| **模型配置文件** | `repo://libs/model-profiles/langchain_model_profiles/` | LLM 行为、能力与配置的元数据和配置文件 |
| **MCP（模型上下文协议）** | `repo://libs/langchain_v1/langchain/mcp/` | MCP 基于模型集成的适配器、工具与引导机制 |
| **输出解析器** | `repo://libs/core/langchain_core/output_parsers/` | 结构化输出解析、Pydantic 模型序列化与输出验证 |
| **提示词** | `repo://libs/core/langchain_core/prompts/` | 聊天与字符串提示模板、少量示例及图像提示 |
| **可运行对象（LCEL）** | `repo://libs/core/langchain_core/runnables/` | 基础 `Runnable[Input, Output]` 协议、组合操作符与控制流（管道、分支、回退、重试） |
| **结构化输出** | `repo://libs/langchain_v1/langchain/agents/structured_output.py` | 类型化智能体输出的模式定义与响应序列化 |
| **工具** | `repo://libs/core/langchain_core/tools/` | `BaseTool` 抽象、函数/Pydantic 转换为工具，以及工具渲染 |
| **测试（核心单元）** | `repo://libs/core/tests/unit_tests/` | 抽象组件的单元测试：可运行对象、消息、提示词、工具、回调 |
| **测试（核心集成）** | `repo://libs/core/tests/integration_tests/` | 与实时提供商及外部服务的集成测试 |
| **测试（LangChain 单元）** | `repo://libs/langchain_v1/tests/unit_tests/` | 智能体工厂、中间件、聊天模型与编排的单元测试 |
| **测试（LangChain 集成）** | `repo://libs/langchain_v1/tests/integration_tests/` | 智能体执行、工具绑定与提供商降级的集成测试 |
| **测试（合作伙伴单元）** | `repo://libs/partners/*/tests/unit_tests/` | 服务商特定的单元测试 |
| **测试（合作伙伴集成）** | `repo://libs/partners/*/tests/integration_tests/` | 服务商特定的集成测试 |
| **标准测试** | `repo://libs/standard-tests/langchain_tests/` | 生态系统内组件一致性验证的共享测试套件与契约 |
| **配置（核心）** | `repo://libs/core/pyproject.toml` | langchain-core 包元数据、依赖项（langsmith、httpx、tenacity、pydantic）及构建配置 |
| **配置（LangChain）** | `repo://libs/langchain_v1/pyproject.toml` | langchain 包元数据、核心依赖项及可选服务商组 |
| **配置（仓库级）** | `repo:///.pre-commit-config.yaml` | 所有包通用的 Git 钩子，用于格式化、 lint 和验证 |
| **构建系统（库）** | `repo://libs/Makefile` | 仓库级别的构建目标、依赖锁定与跨包任务 |

## 关键目录结构

```
/libs/
├── core/                           # langchain-core: Base abstractions (v1.6.12)
│   ├── langchain_core/
│   │   ├── language_models/        # BaseChatModel and language model abstractions
│   │   ├── messages/               # Message types and content blocks
│   │   ├── runnables/              # Runnable protocol and composition
│   │   ├── tools/                  # BaseTool and tool conversion
│   │   ├── prompts/                # Prompt templates and few-shot
│   │   ├── output_parsers/         # Output parsing and validation
│   │   ├── callbacks/              # Callback manager and handlers
│   │   ├── retrievers.py           # Retriever abstraction
│   │   └── ... (other modules)
│   ├── tests/
│   │   ├── unit_tests/
│   │   └── integration_tests/
│   ├── Makefile
│   └── pyproject.toml
│
├── langchain_v1/                   # langchain: Orchestration and agents (v1.4.6)
│   ├── langchain/
│   │   ├── agents/
│   │   │   ├── factory.py          # Agent factory and graph construction
│   │   │   ├── middleware/         # Pluggable middleware system
│   │   │   ├── structured_output.py # Response schema and marshaling
│   │   │   └── _subagent_transformer.py # Sub-agent utilities
│   │   ├── chat_models/
│   │   │   └── base.py             # init_chat_model factory
│   │   ├── mcp/                    # Model Context Protocol adapter
│   │   ├── messages/               # v1-specific message utilities
│   │   ├── embeddings/             # Embedding utilities
│   │   ├── tools/                  # v1-specific tool utilities
│   │   └── rate_limiters/          # Rate limiting implementations
│   ├── tests/
│   │   ├── unit_tests/
│   │   ├── integration_tests/
│   │   ├── benchmarks/
│   │   └── cassettes/              # VCR cassettes for HTTP mocking
│   ├── Makefile
│   └── pyproject.toml
│
├── partners/                       # Provider-specific integrations
│   ├── openai/                     # ChatOpenAI, embeddings, etc.
│   ├── anthropic/                  # ChatAnthropic (Claude)
│   ├── ollama/                     # ChatOllama (local models)
│   ├── groq/                       # ChatGroq
│   ├── mistralai/                  # ChatMistralAI
│   ├── huggingface/                # HuggingFace embeddings and models
│   ├── deepseek/                   # ChatDeepSeek
│   ├── xai/                        # XAI (Grok)
│   ├── perplexity/                 # Perplexity models
│   ├── fireworks/                  # Fireworks inference
│   ├── openrouter/                 # OpenRouter aggregator
│   ├── typesafe/                   # Typesafe schema generation
│   ├── chroma/                     # Chroma vector store
│   ├── qdrant/                     # Qdrant vector store
│   ├── exa/                        # Exa search
│   ├── nomic/                      # Nomic embeddings
│   └── Makefile
│
├── model-profiles/                 # LLM behavior and capability profiles
│   ├── langchain_model_profiles/
│   ├── Makefile
│   └── pyproject.toml
│
├── standard-tests/                 # Cross-ecosystem test contracts
│   ├── langchain_tests/
│   ├── tests/
│   ├── Makefile
│   └── pyproject.toml
│
├── text-splitters/                 # Text splitting utilities
│
├── Makefile                        # Multi-package build coordination
└── README.md
```

## 常见工作流程

### 查找与智能体相关的代码
- **智能体构建**：`repo://libs/langchain_v1/langchain/agents/factory.py`
- **中间件钩子**：`repo://libs/langchain_v1/langchain/agents/middleware/types.py` 用于类型定义；具体中间件位于 `repo://libs/langchain_v1/langchain/agents/middleware/` 子目录中
- **结构化输出**：`repo://libs/langchain_v1/langchain/agents/structured_output.py`
- **测试**：`repo://libs/langchain_v1/tests/unit_tests/` 和 `repo://libs/langchain_v1/tests/integration_tests/`

### 查找 LLM 集成相关代码
- **提供商实现**：`repo://libs/partners/<provider>/`（例如 `repo://libs/partners/openai/`）
- **模型发现**：`repo://libs/langchain_v1/langchain/chat_models/base.py`（init_chat_model）
- **基础接口**：`repo://libs/core/langchain_core/language_models/chat_models.py`
- **模型配置文件**：`repo://libs/model-profiles/langchain_model_profiles/`

### 查找核心抽象
- **Runnable 协议**：`repo://libs/core/langchain_core/runnables/base.py`
- **消息**：`repo://libs/core/langchain_core/messages/`
- **工具**：`repo://libs/core/langchain_core/tools/base.py`
- **提示词**：`repo://libs/core/langchain_core/prompts/`
- **回调**：`repo://libs/core/langchain_core/callbacks/`

### 查找测试代码
- **核心抽象**：`repo://libs/core/tests/`
- **智能体与编排**：`repo://libs/langchain_v1/tests/`
- **特定提供商**：`repo://libs/partners/<provider>/tests/`
- **共享测试契约**：`repo://libs/standard-tests/`

### 配置与构建
- **代码检查与格式化**：仓库根目录下的 `.pre-commit-config.yaml`
- **核心依赖**：`repo://libs/core/pyproject.toml`
- **LangChain 依赖**：`repo://libs/langchain_v1/pyproject.toml`
- **构建任务**：`repo://libs/Makefile` 提供多包命令

## 构建与开发命令

所有包目录（`libs/core/`、`libs/langchain_v1/`、`libs/partners/<provider>/` 等）均包含一个本地 `Makefile`，提供标准目标：

```bash
# Format code (ruff)
make -C <package> format

# Lint code (ruff)
make -C libs/core lint

# Run all tests
make -C libs/langchain_v1 test

# Lock dependencies
make -C libs/core lock

# Check lockfile consistency
make -C libs/core check-lock
```

根目录的 `repo://libs/Makefile` 负责协调多包操作：

```bash
# Lock all packages at once
make -C libs lock

# Check all lockfiles
make -C libs check-lock
```

## 关键实现构件

### 智能体工厂图
`repo://libs/langchain_v1/langchain/agents/factory.py` 中的智能体构建流水线会生成一个 LangGraph 状态机，包含以下节点：
- **入口**：仅运行一次 `before_agent` 中间件
- **循环入口**：每次模型迭代开始时运行，执行 `before_model` 钩子
- **模型**：使用消息历史调用语言模型
- **模型后处理**：运行 `after_model` 钩子以处理响应
- **工具**：执行工具调用（如有）
- **退出**：在完成时仅运行一次 `after_agent` 钩子

中间件可在模型边界、工具边界以及生命周期钩子（`before_agent`、`before_model`、`after_model`、`after_tool_call`、`after_agent`）处注入钩子。

### 智能体中间件实现
中间件目录包含 17+ 种中间件实现，涵盖：
- **模型级别**：`ModelRetryMiddleware`、`ModelCallLimitMiddleware`、`ModelFallbackMiddleware`
- **工具级别**：`ToolRetryMiddleware`、`ToolCallLimitMiddleware`、`ToolErrorMiddleware`、`ToolEmulator`
- **上下文**：`HumanInTheLoopMiddleware`、`ContextEditingMiddleware`、`FilesystemFileSearchMiddleware`
- **安全/脱敏**：`PIIMiddleware`、`RedactionMiddleware`（内部使用）
- **执行控制**：`ShellToolMiddleware`、`ProviderToolSearchMiddleware`
- **优化**：`SummarizationMiddleware`、`TodoListMiddleware`
- **选择**：`LLMToolSelectorMiddleware`
- **生命周期**：`TracePolicy` 配置及执行追踪

每个中间件均可独立配置，并可组合用于智能体构建链。

### Runnable 组合
`repo://libs/core/langchain_core/runnables/base.py` 中的 Runnable 协议通过操作符实现声明式链式调用：
- **管道** (`|`)：顺序组合
- **并行** (`+`)：并行执行分支
- **分支** (`.pipe()` 配合路由)：条件执行路径
- **回退** (`.with_fallback()`)：错误恢复，提供备选方案
- **重试** (`.with_retry()`)：自动重试并带退避机制

所有组合均自动支持 `invoke()`、`ainvoke()`、`batch()`、`stream()` 及其异步变体。

### 消息协议
`repo://libs/core/langchain_core/messages/` 中的消息抽象定义了：
- **消息类型**：`AIMessage`、`HumanMessage`、`SystemMessage`、`ToolMessage`、`FunctionMessage`
- **内容块**：`TextBlock`、`ImageBlock`、`ToolUseBlock`、`ToolResultBlock`，以及自定义块
- **消息工具**：序列化、合并、角色映射、模型特定转换

### 工具抽象
`repo://libs/core/langchain_core/tools/base.py` 中的 `BaseTool` 提供：
- **工具协议**：同步/异步调用，从文档字符串或 Pydantic 生成模式
- **转换**：辅助函数，将 Python 函数包装为工具
- **渲染**：将工具格式化为模型上下文中的描述或结构化模式

## 依赖流

```
User Applications
  ├─→ langchain (v1.4.6)
  │    ├─→ langchain-core (v1.6.11)
  │    └─→ LangGraph (state machines)
  │
  ├─→ langchain-core (direct use)
  │
  └─→ Partner Packages (langchain-openai, langchain-anthropic, etc.)
       └─→ Implement langchain-core abstractions
```

**版本控制**：Core 独立发布，采用严格的语义化版本控制。LangChain 及合作伙伴固定 Core 版本。各合作包按提供商独立发布。

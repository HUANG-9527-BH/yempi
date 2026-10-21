---
type: "Developer Tools & Commands"
title: "Development Commands and Local Setup"
description: "Quick reference for uv, make, lint, test, and type-checking commands in the LangChain monorepo, including environment setup, pre-commit hooks, and testing workflows."
tags: [development, build, testing, linting, typing, uv, make, pre-commit, local-setup]
sources:
  - id: openwiki-source-4d1645cb6317345817452838
    resource: repo://.pre-commit-config.yaml
  - id: openwiki-source-8037e2358a2c4f9b2c722a11
    resource: repo://AGENTS.md
  - id: openwiki-source-8f1875229ad4a704c8e20a06
    resource: repo://libs/core/Makefile
  - id: openwiki-source-3486a94e6eb23a78271a5bfb
    resource: repo://libs/core/pyproject.toml
  - id: openwiki-source-7c96a74af67942d40559bf7d
    resource: repo://libs/langchain_v1/Makefile
  - id: openwiki-source-f708a9db48bfcf1b154e4708
    resource: repo://libs/langchain/Makefile
  - id: openwiki-source-49fbcc45434b619b68220bf9
    resource: repo://libs/Makefile
  - id: openwiki-source-a6e669bb11f217c6fbd06670
    resource: repo://libs/partners/anthropic/Makefile
generated: { by: "openwiki/0.5.0", at: "2026-09-03T15:18:34.589Z" }
verified:
  - by: openwiki/0.5.0
    at: 2026-09-28T08:35:20.640Z
---
## 概览

LangChain Python 单体仓库使用 `uv` 进行依赖管理，使用 `make` 进行任务自动化，并使用 `ruff`/`mypy` 进行代码质量检查。本页提供常见开发命令和设置流程的快速参考。

## 初始设置

### 安装依赖

`libs/` 目录中的每个包都有自己的 `pyproject.toml` 和 `uv.lock` 文件。在运行测试或进行修改之前，请先设置依赖：

```bash
# Install all dependency groups (lint, typing, test, dev)
uv sync --all-groups

# Or install only a specific group
uv sync --group test
uv sync --group lint
```

`--all-groups` 标志确保你拥有用于 linting、类型检查和测试的工具。有关详细的开发规范和 PR 指南，请参阅 [AGENTS.md 中的贡献指南](repo://AGENTS.md)。

### 预提交设置

该仓库使用预提交钩子（pre-commit hooks）在提交时强制执行代码质量。请一次性安装并配置它们：

```bash
pre-commit install
```

预提交钩子会在每次提交前自动对暂存文件运行。如需手动触发钩子：

```bash
# Run all hooks on all files
pre-commit run --all-files

# Run a specific hook
pre-commit run ruff --all-files
```

## 预提交钩子

`.pre-commit-config.yaml` 定义了以下钩子，用于强制执行：

- **标准验证**：YAML/TOML 语法检查、正确的文件结尾、无尾随空白
- **文本规范化**：修复花括号引号和非标准空格
- **按包格式与 lint**：`libs/` 中的每个包（core、langchain、partners/*）都会运行 `make format lint`
- **版本一致性检查**：确保 `pyproject.toml` 中的版本与 `langchain-core`、`langchain` 及合作伙伴包的源代码版本一致

这些钩子会自动阻止因 lint 失败或格式问题而提交的更改。它们使用了下文所述的相同 Makefiles。

## 测试命令

### 运行所有单元测试

```bash
# From any package directory
make test

# Example: test langchain_v1
cd libs/langchain_v1 && make test
```

单元测试位于 `tests/unit_tests/` 目录中（不允许网络调用）。测试目标使用 `pytest` 配合 `xdist` 实现并行化，并通过套接字限制防止意外的网络调用。

### 运行特定测试文件

```bash
# Using make
make test TEST_FILE=tests/unit_tests/agents/test_agent.py

# Or using uv directly
uv run --group test pytest tests/unit_tests/agents/test_agent.py
```

### 集成测试

集成测试位于 `tests/integration_tests/` 目录中，需要网络访问和 API 密钥。请单独运行它们：

```bash
make integration_tests
```

某些包（如 `langchain_v1`）在集成测试中使用 Docker 服务（PostgreSQL、Redis）：

```bash
cd libs/langchain_v1
make test  # starts services, runs tests, stops services
```

### 侦听模式下运行测试

编辑代码时自动重新运行测试：

```bash
make test_watch
```

此功能使用 `pytest-watcher`（通过 `ptw` 命令）实现，并会自动更新快照。

### 生成覆盖率报告

生成代码覆盖率报告：

```bash
make coverage
```

这将生成 `xml` 和终端缺失报告，有助于理解未被覆盖的代码路径。

## 代码检查与格式化

### 运行完整的检查套件

```bash
make lint
```

此命令按顺序执行三项检查：
1. **Ruff 检查**：用于逻辑错误、命名规范和代码异味的 linter
2. **Ruff 格式检查（diff）**：格式检查（不修改文件）
3. **Mypy**：静态类型检查

### 格式化代码

```bash
make format
```

此命令应用 `ruff format` 和 `ruff check --fix`，自动修复格式和逻辑问题（例如未排序的导入、未使用的变量）。

### 仅 Ruff 命令

为加快迭代速度，可分别运行格式化和 linter 工具：

```bash
# Check formatting without fixing
ruff check .
ruff format . --diff

# Fix formatting and lint issues
ruff check . --fix
ruff format .
```

从包目录内或仓库根目录运行。Ruff 会处理 Python 文件和 Jupyter 笔记本。

### 类型检查

使用 mypy 进行完整的类型检查：

```bash
mypy .
```

或者使用 make 目标：

```bash
make type
```

此操作将检查所有 Python 文件中的类型错误（例如，参数类型不正确、缺少类型提示）。对于大型包，类型检查可能较慢；可使用以下命令运行：

```bash
# Type check specific file or directory
mypy libs/core/langchain_core/runnables.py
```

## 单个包的测试与开发

在 monorepo 中开发并测试单个包：

```bash
# Example: work on langchain_v1 core agent system
cd libs/langchain_v1

# Install all dependencies for this package
uv sync --all-groups

# Run all unit tests
make test

# Test specific file or with custom pytest options
make test TEST_FILE=tests/unit_tests/agents/test_create_agent.py

# Format and lint
make format
make lint

# Type check
make type
```

每个包都有自己的 Makefile，包含一致的目标。monorepo 的 `/libs/Makefile` 提供了适用于所有包的通用命令，例如重新生成锁文件。

## 锁文件管理

每个包中的 `uv.lock` 文件精确锁定依赖版本，以确保构建的可重现性。当依赖项发生变化时，需更新锁文件：

```bash
# From a package directory
uv lock

# Or regenerate all package locks
cd libs && make lock

# Verify all locks are up-to-date (CI check)
cd libs && make check-lock
```

`.pre-commit-config.yaml` 包含 `UV_FROZEN = true`，这可在常规开发中防止意外的锁文件变更。仅在有意更新依赖项时使用上述命令。

## Make 命令参考

所有包遵循相同的 Makefile 结构：

| 命令 | 用途 |
|------|------|
| `make test` | 运行所有单元测试（pytest） |
| `make test TEST_FILE=<path>` | 运行特定文件或目录中的测试 |
| `make test_watch` | 以监视模式运行测试（更改后自动重跑） |
| `make integration_tests` | 运行集成测试（需要 API 密钥） |
| `make extended_tests` | 仅运行标记为 `@pytest.mark.extended` 的测试 |
| `make lint` | 运行 ruff 检查 + ruff format --diff + mypy |
| `make format` | 应用 ruff 格式化并运行 ruff check --fix |
| `make type` | 运行 mypy 类型检查 |
| `make coverage` | 运行测试并生成覆盖率报告 |
| `make help` | 显示所有可用目标 |

包特定命令（请参见各目录中的 Makefile）：

- `langchain_v1`: `make test_fast`, `make coverage_agents`, `make start_services`, `make stop_services`
- `core`: `make check_imports`, `make benchmark`
- `partners/*`: `make test TEST_FILE=tests/integration_tests/`

## 常用工作流

### 提交前准备

```bash
# 1. Format code
make format

# 2. Run linting and type checks
make lint

# 3. Run tests
make test

# 4. Commit (pre-commit hooks will run automatically)
git commit
```

或让 pre-commit 自动完成格式化/检查：

```bash
make test
git add .
pre-commit run --all-files  # or just commit and let hooks run
git commit
```

### 迭代式开发

开发过程中快速获得反馈：

```bash
# Terminal 1: Watch tests
make test_watch

# Terminal 2: Edit code and format
# Changes auto-trigger re-run in Terminal 1
make format
```

### 检查已修改文件的代码风格

```bash
# Lint only files changed in the current branch
make lint_diff
make format_diff
```

这些目标会在 `git diff` 输出与 `master` 对比的基础上运行。

### 针对特定模块的类型检查

```bash
# Type check a module while developing
mypy libs/langchain_v1/langchain/agents/agent.py

# Type check tests (faster, uses test group)
cd libs/langchain_v1
make lint_tests
```

## 环境变量

Makefiles 使用若干环境变量来控制行为：

| 变量 | 用途 | 默认值 |
|------|------|--------|
| `UV_FROZEN` | 在 `uv sync` 期间防止锁文件变更 | Makefiles 中为 `true` |
| `TEST_FILE` | 测试文件或目录路径 | `tests/unit_tests/` |
| `PYTEST_EXTRA` | 额外的 pytest 选项 | （空） |
| `LANGGRAPH_TEST_FAST` | 使用内存服务而非 Docker | `1`（快速）或 `0`（完整） |

示例：以额外的 pytest 详细级别运行快速测试：

```bash
make test PYTEST_EXTRA="-vv" TEST_FILE=tests/unit_tests/agents
```

## 故障排除

### 锁文件不同步

```bash
# Regenerate lock
cd libs/<package>
uv lock

# Or check if lock is up-to-date
uv lock --check
```

### 依赖项未安装

```bash
# Ensure all groups are installed
uv sync --all-groups

# Or just the test group
uv sync --group test
```

### 测试因“无网络”错误而失败

这是有意为之的——单元测试具有套接字限制。如需进行集成测试：

```bash
make integration_tests
```

### Ruff 或 mypy 找不到

```bash
# Install lint and typing groups
uv sync --group lint --group typing
```

### 本地预提交钩子失败，但在 CI 中通过

请确保使用相同的 Python 版本，并已安装所有依赖组：

```bash
python --version
uv sync --all-groups
pre-commit run --all-files
```

## 相关文档

- [贡献指南](repo://AGENTS.md)：详细的开发规范、PR 模板和代码标准
- [系统架构](repo:///openwiki/architecture.md)：三层设计及模块职责
- [CI/CD 工作流](repo:///openwiki/ci-workflows.md)：GitHub Actions 自动化与发布流程

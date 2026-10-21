---
type: "Reference"
title: "CI/CD Workflows: GitHub Actions and Release Process"
description: "LangChain's GitHub Actions-based CI/CD system automating testing, linting, and release management across a monorepo with intelligent change detection, parallel matrix testing, and strict release gates."
tags: [ci-cd, github-actions, testing, linting, release, pypi, monorepo, automation]
sources:
  - id: openwiki-source-34e57b5a3a0c875639ab72a7
    resource: repo://.github/scripts/check_diff.py
  - id: openwiki-source-f35e7c44cc1805709393a581
    resource: repo://.github/workflows/_lint.yml
  - id: openwiki-source-c92cc62695c6def991956428
    resource: repo://.github/workflows/_release.yml
  - id: openwiki-source-c9c292f4ecabe180cdae27ce
    resource: repo://.github/workflows/_test_pydantic.yml
  - id: openwiki-source-d8a8900818f4abab719bd1b7
    resource: repo://.github/workflows/_test_vcr.yml
  - id: openwiki-source-4d9cccca7700db7220ec055e
    resource: repo://.github/workflows/_test.yml
  - id: openwiki-source-7330cb37457ccdb62d7c41c7
    resource: repo://.github/workflows/auto-label-by-package.yml
  - id: openwiki-source-6e3a52c89729b5704dbd7eec
    resource: repo://.github/workflows/check_diffs.yml
  - id: openwiki-source-9069a5dd5fbb579fbd5470ce
    resource: repo://.github/workflows/integration_tests.yml
  - id: openwiki-source-6d4b4e707b8d60b6ccfa3425
    resource: repo://.github/workflows/openwiki-update.yml
  - id: openwiki-source-f8781d847f6481a966a44a68
    resource: repo://.github/workflows/pr_labeler.yml
  - id: openwiki-source-12805fbf767dc2a3e238645e
    resource: repo://.github/workflows/pr_lint.yml
generated: { by: "openwiki/0.5.0", at: "2026-09-28T08:35:20.640Z" }
verified:
  - by: openwiki/0.5.0
    at: 2026-10-16T08:25:55.096Z
---
# CI/CD 工作流：GitHub Actions 与发布流程

LangChain 使用基于 GitHub Actions 构建的复杂 CI/CD 系统，该系统在单体仓库（monorepo）结构中自动化执行测试、代码风格检查、质量检测以及发布管理。系统通过智能变更检测、并行矩阵测试和严格的发布门禁机制，强调效率优化。

## 架构概览

CI/CD 系统包含三个层级：

1. **拉取请求 / 推送 CI（`check_diffs.yml`）**：检测已更改的包，并对受影响部分运行针对性的测试、代码风格检查及兼容性验证  
2. **定时集成测试（`integration_tests.yml`）**：每日使用真实凭证对合作库进行远程 API 测试  
3. **手动发布工作流（`_release.yml`）**：全面的预发布验证、PyPI 发布以及依赖包测试

## 主要 CI 工作流（拉取请求与主分支推送）

主要入口为 `.github/workflows/check_diffs.yml`，该工作流在每次拉取请求、推送到主分支或合并组事件时触发。

### 变更检测与矩阵生成

工作流从变更检测阶段开始：

1. 一个 Python 脚本（`.github/scripts/check_diff.py`）分析哪些文件发生了变更  
2. 将变更映射到对应的包目录（如 `libs/core`、`libs/partners/*` 等）  
3. 构建依赖图，当核心组件变更时自动包含其依赖包  
4. 为代码风格检查、单元测试、Pydantic 兼容性测试、集成测试编译、VCR 录像带测试以及扩展测试套件分别生成独立的测试矩阵  
5. 输出结果以 JSON 格式传递给下游任务，通过矩阵策略（matrix strategy）使用

此检测机制确保仅对受影响的包进行测试，从而优化 CI 运行时间。脚本在枚举过程中跳过 `libs/standard-tests` 目录，并将某些合作方（如 huggingface）视为 CI 不稳定，移除其依赖链，但允许直接编辑内容被测试。

### 代码风格检查流水线（`_lint.yml`）

在受影响的包上运行，支持 Python 3.11（可配置）：

- **Ruff 分析**：代码风格、导入排序与规则强制执行，通过 `RUFF_OUTPUT_FORMAT: github` 在 GitHub 中生成内联注释  
- **MyPy 类型检查**：静态类型验证  
- **Markdown 代码风格检查**：文档质量检查（通过 `.markdownlint.json` 配置）

工具来源于依赖组：`lint` 和 `typing`。工作流会安装包代码与测试代码的依赖项，运行 `make lint_package` 和 `make lint_tests` 目标。合作包会单独安装测试依赖项，包括集成测试依赖。

### 单元测试（`_test.yml`）

在多个 Python 版本上运行矩阵测试，并验证依赖约束：

**矩阵维度**：
- `libs/core`：Python 3.10、3.11、3.12、3.13、3.14  
- 其他包：Python 3.10 与 3.14  
- 当前锁定的依赖版本（来自 `uv.lock`）  
- 最低支持的依赖版本

**双阶段测试**：

1. **当前依赖版本**：使用 `uv.lock` 中的版本运行完整测试套件，通过 `make test PYTEST_EXTRA=-q` 执行  
2. **最低依赖版本**：通过 `get_min_versions.py` 脚本从 `pyproject.toml` 的约束中计算出最小版本，使用 pip 降级后重新运行测试，通过 `make tests PYTEST_EXTRA=-q` 确保兼容性

工作流在测试后验证工作目录保持干净（无未跟踪的生成文件）。

### Pydantic 兼容性测试（`_test_pydantic.yml`）

针对可配置的 Pydantic 版本（例如 v2.0、v2.1、v2.2）测试受影响的包：

- 在 Pydantic 版本约束或依赖代码变更时触发
- 通过查询 `uv.lock` 获取最大 Pydantic 版本，以及通过 `pyproject.toml` 获取核心和目标包中的最小版本，以确定测试矩阵
- 对每个 Pydantic 版本运行 `make test`
- 默认使用 Python 3.12，支持覆盖

### VCR 录制匣测试（`_test_vcr.yml`）

验证基于已录制 HTTP 录制匣的集成测试：

- 以回放模式运行，使用假凭证（无需真实 API 密钥）
- 在不重新录制的情况下检测因测试输入变更导致的过期录制匣
- 执行 `make test_vcr` 目标
- 实现快速、可重复的集成测试反馈

仅对包含 VCR 录制匣的包（当前为 `libs/partners/openai`）触发，由 `check_diff.py` 中的 `VCR_PACKAGES` 集合追踪。

### 集成测试编译（`_compile_integration_test.yml`）

执行浅层集成测试验证：

- 通过 `pytest -m compile tests/integration_tests` 编译测试模块但不执行
- 捕获导入错误和明显的语法问题
- 提供快速反馈循环，无需运行昂贵的外部 API 调用
- 安装测试和集成测试依赖组

### 扩展测试套件

对于定义了 `extended_testing_deps.txt` 的包，运行额外测试：

- 通过该文件安装超出标准测试组的额外依赖
- 执行 `make extended_tests` 目标
- 支持性能基准测试、压力测试或重量级验证
- 位于 `check_diffs.yml` 中的 `extended-tests` 作业内

### 发布选项验证

工作流包含一个 `check-release-options` 作业：

- 验证 `.github/workflows/_release.yml` 下拉选项与实际包目录保持同步
- 防止过期的发布选项阻塞有效发布

## 发布工作流（`_release.yml`）

发布工作流通过 GitHub Actions UI 手动触发（或可作为可重用工作流调用）。它处理版本管理、构建、测试，并将包发布到 PyPI。

### 发布模式与调用方式

**手动分发**（`workflow_dispatch`）：
- 从下拉菜单中选择要发布的包（core、langchain、langchain_v1、text-splitters、standard-tests、model-profiles，或 17+ 个合作伙伴包）
- 手动输入版本号（默认值为 `0.1.0`）
- 可选覆盖为完整路径（例如 `libs/partners/partner-xyz`）
- 危险标志：`dangerous-nonmaster-release`（热修复）、`allow-prereleases`、`skip-prior-published-package-checks`

**可重用工作流**（`workflow_call`）：
- 接受 `working-directory`、`release-version` 和安全绕过标志
- 内部用于多包发布编排

### 发布门禁：构建与版本检查

**作业：`build`**（具有隔离权限以保障安全）：

1. **版本验证**：从 `pyproject.toml` 提取版本，并与输入版本进行比较，使用 PEP 440 标准化处理（将 `0.1.0-rc1` 和 `0.1.0rc1` 视为等价）；若不匹配则失败
2. **PyPI 可用性检查**：查询 `https://pypi.org/pypi/{pkg}/{version}/json` 确保该版本尚未发布；若 PyPI 不可达或返回意外状态，则关闭失败
3. **构建**：运行 `uv build` 以创建 wheel 和 sdist 分发包
4. **构件上传**：将 `dist/` 目录存储供下游作业使用

安全设计原理：将构建（无凭证）与发布（受信任的发布令牌）分离，以防止被破坏的依赖项访问 PyPI 凭证。

### 发布说明生成

**作业：`release-notes`**：

1. **标签检测**：通过 git 历史记录查找上一个发布标签
   - 对于预发布版本（包含连字符）：匹配基础版本；若无则回退到最新发布标签
   - 对于稳定版本：搜索前一个补丁版本；若无则回退到最新版本
   - 首次发布：使用 git 根目录起始的完整提交历史
2. **变更日志提取**：运行 `git log --format="%s" <prev-tag>..HEAD -- <working-dir>` 以收集提交信息
3. **标签验证**：在继续之前确认上一个标签存在于 git 仓库中

### 预发布检查

**作业：`pre-release-checks`**（不启用缓存以捕获缺失依赖项）：

1. **直接 wheel 安装**：通过 `uv pip install dist/*.whl` 直接安装构建好的 wheel（验证元数据和可安装性）
2. **包导入测试**：验证主模块能否成功导入
3. **单元测试**：对 wheel 运行完整的 `make tests`
4. **最低版本测试**：重新计算最低版本要求，通过 pip 降级后，使用 `make tests PYTEST_EXTRA="-q -k 'not test_serdes'"` 重新运行测试（为加快速度跳过序列化测试）
5. **预发布依赖检测**：如果任何依赖声明了预发布版本，则失败（除非当前发布本身也是预发布版本）
6. **集成测试**：仅针对合作包，使用实时 API 凭证运行 `make integration_tests`

### PyPI 发布

**作业：`test-pypi-publish`**（TestPyPI）：
- 使用 GitHub OpenID Connect（受信任的发布机制）
- 发布至 test.pypi.org 用于预发布验证
- 通过 `skip-existing: true` 允许重复版本（仅限 CI 安全性）

**作业：`publish`**（生产环境 PyPI）：
- 使用受信任的发布机制发布至生产 PyPI
- 仅在所有前置检查通过后运行
- 创建带有自动生成发布说明的 GitHub Release

### 兼容性测试

**作业：`test-prior-published-packages-against-new-core`**：
- 仅在 `libs/core` 发布时运行
- 测试此前已发布的合作包（目前为 anthropic、openai）与新核心库的兼容性
- 从 git 中获取最新的未被移除的合作包标签，安装新的 core wheel，运行测试
- 可通过 `skip-prior-published-package-checks` 输入参数跳过特定合作包（选项：none、anthropic、openai、all）

**作业：`test-dependents`**：
- 仅在 `libs/core` 或 `libs/langchain_v1` 发布时运行
- 检查外部依赖包（目前为 deepagents）
- 测试 Python 3.11 和 3.13
- 确保在发布前捕获破坏性变更

## 集成测试（`integration_tests.yml`）

每日定时执行（UTC 时间下午 1 点），支持手动触发覆盖。

### 测试矩阵生成

**作业：`compute-matrix`**：

- **默认范围**：测试 9 个合作库（OpenAI、Anthropic、Fireworks、Groq、MistralAI、XAI、Google VertexAI、Google GenAI、AWS）
- **Python 版本**：默认为 3.10 和 3.14；可通过输入参数覆盖
- **选择性测试**：可选择单个库、排除某些库或覆盖 Python 版本
- **范围安全**：仅在主仓库上运行；允许从分支手动触发

### 集成测试执行

**任务：`integration-tests`**：

- 检出主仓库以及外部的 google-genai、google-vertexai 和 langchain-aws 仓库
- 将外部仓库重新组织到本地的 partner 目录中，以实现统一测试
- 使用 Google Cloud 和 AWS 凭证进行身份验证
- 对每个包运行 `make integration_tests`，并注入所有实时 API 凭证
- 使用基于 (包, Python 版本) 的并发锁，对同一包的运行进行序列化，防止凭证冲突
- 包含特殊安装逻辑：在检出的合作伙伴版本之上叠加本地可编辑的核心包和标准测试包

**凭据**：接收超过 30 个环境变量，涵盖 OpenAI、Anthropic、Google、AWS、Azure、Groq、MistralAI、HuggingFace、Mistral、Together、Cohere 等。

## 自动标注工作流

### 问题自动标注（`auto-label-by-package.yml`）

在问题被打开或编辑时触发：

1. 解析问题正文中的 `## Package` 部分
2. 支持下拉菜单（单选）和复选框（多选）两种格式
3. 通过 JSON 映射表将包名称（如 "langchain-openai"）映射为标签（如 "openai"）
4. 根据所选包添加或移除标签

### PR 标题校验（`pr_lint.yml`）

强制要求所有拉取请求标题符合 Conventional Commits 1.0.0 格式：

- **格式**：`<type>[optional scope]: <description>`（例如，`feat(core): add multi-tenant support`）
- **允许的类型**：feat、fix、docs、style、refactor、perf、test、build、ci、chore、revert、release、hotfix
- **可选作用域**：特定包的作用域（core、langchain、anthropic、openai 等）或跨领域关注点（infra、deps、partners）
- **破坏性变更**：在类型/作用域后附加 `!`（例如，`feat!: remove deprecated API`）
- **发布提交**：必须为 `release(scope): x.y.z` 格式
- **验证**：使用 `amannn/action-semantic-pull-request` 并拒绝空作用域

空作用域括号会被拒绝；PR 必须要么省略括号（无作用域），要么提供有效作用域。

### PR 标签分配（`pr_labeler.yml`）

统一的 PR 标签系统，应用于大小、文件级、标题级及贡献者分类：

- 文件级标签：将更改的文件路径映射为包标签
- 大小标签：根据差异统计计算 PR 大小（small、medium、large）
- 标题级标签：检测 PR 标题中的特定模式
- 贡献者分类：检查组织成员资格，以标记外部贡献（通过 GitHub App token）
- 使用并发锁防止竞争条件
- 将多个先前的工作流整合为单一顺序执行的运行

## OpenWiki 自动更新（`openwiki-update.yml`）

按计划（每日 UTC 时间上午 8 点）或手动触发运行：

1. 通过 `fetch-depth: 0` 检出完整的仓库历史（用于与 HEAD 进行差异对比）
2. 安装 Node.js 和 OpenWiki CLI（@0.5.0），可选启用 Mermaid 图表验证
3. 运行 `openwiki code --update --print` 以重新生成文档
4. 删除临时状态文件（`.run.json`）
5. 通过 `peter-evans/create-pull-request@v8.1.1` 创建/更新拉取请求
6. 故障时保留部分进度：如果 OpenWiki 执行失败，PR 会故意仅保留失败前已完成的页面，使其成为下一次计划运行的基线

使用 LangSmith 追踪功能进行可观测性（OPENWIKI_LANGSMITH_API_KEY, LANGSMITH_API_KEY）。

## 依赖项锁定与版本管理

### 冻结的依赖项锁

所有 CI 作业均设置 `UV_FROZEN=true` 和 `UV_NO_SYNC=true`（适用时）：

- 确保基于 `uv.lock` 中锁定版本的可复现构建
- 防止 CI 中出现传递性依赖意外
- 每个作业显式锁定 Python 版本和依赖项修订版本

### 最低版本测试

`get_min_versions.py` 脚本从 `pyproject.toml` 提取版本约束，并查询 PyPI 以获取满足这些约束的最早发布版本。

示例：若约束为 `langchain-core>=0.3.0,<1.0`，该脚本将找到并安装最早的 0.3.* 版本。

两种模式：
- `pull_request`：在 PR CI 中对最低版本进行一定程度宽松测试
- `release`：在发布验证中采用更严格测试并拒绝预发布版本

## 发布策略

### 语义化版本控制

**核心库**（`libs/core`）遵循严格的语义化版本控制：
- 主版本：破坏性变更
- 次版本：新功能（向后兼容）
- 补丁版本：错误修复

**合作伙伴包**及其他库与核心版本保持一致：
- LangChain 与核心版本保持同步以实现紧密集成
- 合作伙伴维护独立版本号，但与核心发布进行协调

### 发布分支策略

- 发布仅从 `master` 分支（默认）或通过 `dangerous-nonmaster-release` 标志明确触发（仅限热修复）
- 版本必须与 `pyproject.toml` 匹配，或由操作员提供覆盖
- PyPI 可用性需双重确认，防止意外重复发布

### 预发布支持

- 支持 alpha/beta/rc 版本（如 `0.1.0-rc1`, `0.1.0a1`）
- 预发布检测会根据 PEP 440 标准规范化连字符/下划线变体
- 可选 `allow-prereleases` 标志允许在 alpha 周期中包含传递性预发布依赖
- 正式发布默认阻止预发布依赖，除非显式允许

## 配置与运维

### 环境变量

**冻结依赖控制**：
- `UV_FROZEN`：禁止自动依赖解析
- `UV_NO_SYNC`：跳过构建步骤中的 uv sync（改用手动 sync）

**代码检查与格式化**：
- `RUFF_OUTPUT_FORMAT: github`：在 GitHub 上内联标注 linter 违规信息

**LangSmith 追踪**（可选）：
- `LANGSMITH_API_KEY`：可选地追踪 CI 工作流自身
- `LANGCHAIN_TRACING_V2: true`：启用追踪
- `LANGCHAIN_PROJECT: openwiki`：LangSmith 项目名称

### GitHub Actions 权限

工作流遵循最小权限原则：

- **默认**：`contents: read`（只读）
- **PR 标签器**：`pull-requests: write`, `issues: write`
- **发布**：`id-token: write`（可信发布），`contents: write`（创建 GitHub Release）
- **OpenWiki 更新**：`contents: write`, `pull-requests: write`

隔离任务（构建、测试）不授予写权限；发布任务在单独任务中运行且权限受限。

### 自定义动作

**`uv_setup`**（`.github/actions/uv_setup`）：
- 通过官方 `setup-python` 动作设置 Python
- 配置 `uv` 工具并支持可选缓存
- 支持按包名缓存后缀，避免交叉污染
- 参数：`python-version`, `cache-suffix`, `working-directory`, `enable-cache`

## 重要不变量与失败模式

1. **发布前检查中不缓存**：缺少依赖项会被缓存的虚拟环境掩盖，导致损坏的发布被错误地发布
2. **最低版本降级隔离**：最低版本测试在全新的虚拟环境上下文中重新安装包，而非仅通过约束松弛实现
3. **构建/发布任务分离**：构建任务无 PyPI 凭证；发布任务无构建工具，防止供应链攻击
4. **变更检测范围**：VCR 和扩展测试矩阵仅包含具有适当标记的包；添加无标记的测试文件不会触发相应的测试套件
5. **预发布版本阻断**：稳定版本拒绝任何预发布依赖，防止下游用户出现版本解析问题
6. **标签/版本同步**：发布工作流在发布前验证 Git 标签是否符合预期版本格式，捕捉手动标签漂移

## 扩展点

1. **添加新的包类型**：更新 `check_diff.py` 以识别新目录，并将其映射到适当的测试矩阵
2. **向发布测试添加合作伙伴**：更新 `test-prior-published-packages-against-new-core` 矩阵和 `skip-prior-published-package-checks` 输入选项（保持同步）
3. **添加新的 linting/类型检查器**：扩展 `_lint.yml` 作业步骤和依赖组；确保存在 `make lint_package` 目标
4. **添加集成测试凭证**：在 `integration_tests.yml` 作业中添加环境变量，并确保 `make integration_tests` 目标可处理可选凭证
5. **自定义测试套件**：在包目录中创建 `extended_testing_deps.txt` 并定义 `make extended_tests` 目标
6. **OpenWiki 页面**：添加至 `openwiki/` 目录；在每次计划运行时自动更新

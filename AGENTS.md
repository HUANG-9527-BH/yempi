# LangChain 单体仓库的全局开发指南

本文档提供上下文信息，帮助理解 LangChain Python 项目并支持开发工作。

<corridor>

## Corridor 安全分析

当 Corridor 的 `analyzePlan` 工具可用时，请先创建一个计划，并使用该工具对其进行分析，再生成或修改代码。在编写代码前，请应用分析结果提供的安全指导。

</corridor>

## 项目架构与上下文

### 单体仓库结构

这是一个使用 `uv` 管理依赖的 Python 单体仓库，包含多个独立版本管理的包。

```txt
langchain/
├── libs/
│   ├── core/             # `langchain-core` 原语与基础抽象
│   ├── langchain/        # `langchain-classic`（历史版本，不再新增特性）
│   ├── langchain_v1/     # 积极维护的 `langchain` 包
│   ├── partners/         # 第三方集成
│   │   ├── openai/       # OpenAI 模型与嵌入
│   │   ├── anthropic/    # Anthropic（Claude）集成
│   │   ├── ollama/       # 本地模型支持
│   │   └── ...（其他由 LangChain 团队维护的集成）
│   ├── text-splitters/   # 文档分块工具
│   ├── standard-tests/   # 面向集成的共享测试套件
│   └── model-profiles/   # 模型配置档案
├── .github/              # CI/CD workflows and templates
├── .vscode/              # VSCode IDE standard settings and recommended extensions
├── openwiki/             # Generated just-in-time evidence index (optional reading)
└── README.md             # Information about LangChain
```

- **核心层**（`langchain-core`）：基础抽象、接口和协议。用户不应直接了解此层。
- **实现层**（`langchain`）：具体实现和高层公共工具
- **集成层**（`partners/`）：第三方服务集成。请注意，此单体仓库并非 LangChain 所有集成的完整集合；部分集成维护在独立仓库中，例如 `langchain-ai/langchain-google` 和 `langchain-ai/langchain-aws`。这些仓库通常与本单体仓库同级克隆，因此如需查阅其代码，可从本仓库导航至 `../langchain-google/` 直接访问。
- **测试层**（`standard-tests/`）：针对合作伙伴集成的标准化集成测试

### 开发工具与命令

- `uv` – 快速的 Python 包安装器和依赖解析器（替代 pip/poetry）
- `make` – 任务运行器，用于执行常见开发命令。`libs/` 下每个包都有自己的 `Makefile`；可自由查看以了解可用命令和使用模式。
- `ruff` – 快速的 Python 代码检查器和格式化工具
- `mypy` – 静态类型检查
- `pytest` – 测试框架

此单体仓库使用 `uv` 进行依赖管理。本地开发通过在各包的 `pyproject.toml` 文件中 `[tool.uv.sources]` 表声明的可编辑安装实现，使路径依赖指向你的工作树而非 PyPI。

`libs/` 中的每个包都有自己的 `pyproject.toml`、`uv.lock` 和 `Makefile`。仓库根目录没有工作区级别的 `pyproject.toml` 或 `Makefile` —— 始终在运行以下命令前进入你正在工作的包目录（例如 `cd libs/langchain_v1` 或 `cd libs/core`）。

在运行测试前，请通过以下命令设置所有包：

```bash
# For all groups
uv sync --all-groups

# or, to install a specific group only:
uv sync --group test
```

```bash
# Run unit tests (no network)
make test

# Run specific test file
uv run --group test pytest tests/unit_tests/test_specific.py
```

```bash
# Lint code
make lint

# Format code
make format

# Type checking
uv run --group lint mypy .
```

#### 环境与依赖管理

在此单体仓库中，所有环境和依赖操作均应使用 `uv`。不要直接调用 `pip`、`poetry` 或 `conda`。

- 让 `uv` 管理解释器和虚拟环境 —— `uv sync` 和 `uv run` 无需手动执行 `source .venv/bin/activate`。不要在包目录外创建临时虚拟环境。
- 每个包通过其 `pyproject.toml` 指定自身的支持 Python 版本范围；不要固定全局 Python 版本。如需明确指定解释器，应参考包的 `requires-python` 字段，而非假设使用系统 Python。
- 通过 `uv sync` 显式安装依赖（可选 `--group <name>` / `--all-groups`）；绝不允许隐式安装依赖。
- 不要在会话内混用环境，且仅在绝对必要时才添加新依赖 —— 若确实需要，必须说明理由（最新发布/提交、采用情况）。

#### 关键配置文件

仓库根目录没有单一的工作区配置。配置信息按包存放：

- `libs/<package>/pyproject.toml`：包元数据和依赖组（`test`、`lint`、`typing`、`dev` 等）
- `libs/<package>/uv.lock`：用于可复现构建的锁定依赖
- `libs/<package>/Makefile`：该包的开发任务（`test`、`lint`、`format`、`type` 等）
- `libs/Makefile`：仅用于跨包的 `lock` / `check-lock`
- `.pre-commit-config.yaml`（仓库根目录）：在每个包上运行的 Git 钩子，用于格式化/检查

#### PR 和提交标题

遵循 Conventional Commits 规范。详见 `.github/workflows/pr_lint.yml` 中允许的类型和作用域。所有标题必须包含作用域，**无一例外**——即使是对主 `langchain` 包的修改也是如此。

- `type(scope):` 之后的内容应以小写字母开头，除非首词是专有名词（如 `Azure`、`GitHub`、`OpenAI`）或命名实体（类、函数、方法、参数或变量名）。
- 将命名实体用反引号包裹，使其以代码样式渲染。专有名词保持原样不加修饰。
- 标题应简短且描述清晰——细节请留到正文说明。

示例：

```txt
feat(langchain): add new chat completion feature
fix(core): resolve type hinting issue in vector store
chore(anthropic): update infrastructure dependencies
feat(langchain): `ls_agent_type` tag on `create_agent` calls
fix(openai): infer Azure chat profiles from model name
```

#### 分支命名

分支名称应以 `<github-username>/<scope>/<short-description>` 为前缀：

- `<github-username>` —— 作者的 GitHub 用户名（例如 `mdrxy`）。
- `<scope>` —— 与 Conventional Commit 标题中使用的作用域一致（如 `core`、`langchain`、合作伙伴名称、`infra`、`docs` 等）。
- `<short-description>` —— 使用连字符连接的小写形式，简明扼要，末尾不加斜杠。

示例：

```txt
mdrxy/anthropic/normalize-tool-call-ids
mdrxy/core/vector-store-type-hints
mdrxy/infra/agents-md-branch
```

#### PR 描述

描述 *就是* 概要——不要添加 `# Summary` 标题。

- 当 PR 关闭某个问题时，请在最顶部单独一行写出关闭关键字，随后是一条水平线，再接正文：

  ```txt
  Closes #123

  ---

  <rest of description>
  ```

  仅 `Closes`、`Fixes` 和 `Resolves` 在合并时会自动关闭关联问题。`Related:` 或类似标签仅为信息提示，不会触发关闭。

- 解释“为什么”：谁将从中受益，他们遇到了什么问题，以及本变更如何解决它。优先使用简单的用户故事，而非冗长的摘要。
- 写给可能不熟悉该代码库部分的读者。避免使用内部术语，采用对公众友好的语言——这有助于提升可读性。
- **不要**引用行号；一旦文件变动，这些行号就会失效。
- 极少包含完整文件路径或文件名。应通过名称引用受影响的符号、类或子系统。
- 将类、函数、方法、参数和变量名用反引号包裹。
- 对于全新功能或行为变更型的 bug 修复，PR 描述中应包含一个 `## Release note` 部分，使用适合发布说明的语言陈述用户可见的变化。
- 大多数情况下无需专门设置“测试计划”或“测试”章节。仅在覆盖率不明显、风险较高或特别值得注意时提及测试。
- 明确指出需要仔细审查的变更部分。
- 添加简短声明，注明该贡献涉及 AI 智能体的协助。

## 核心开发原则

### 保持公共接口的稳定性

重要：始终尝试保留导出/公开方法的函数签名、参数位置和名称。不要进行破坏性更改。  
对于任何函数签名的变更，无论是否看起来具有破坏性，都应向开发者发出警告。

**在对公共 API 进行任何更改之前：**

- 检查该函数/类是否在 `__init__.py` 中被导出
- 查看测试和示例中已有的使用模式
- 为新参数使用关键字参数：`*, new_param: str = "default"`
- 使用文档字符串中的警告明确标记实验性功能（使用 MkDocs Material 的提示块，如 `!!! warning`）

提问：“如果有人上周使用了这个代码，这次更改会不会导致他们的代码崩溃？”

### 代码质量标准

所有 Python 代码必须包含类型注解和返回类型。

```python title="Example"
def filter_unknown_users(users: list[str], known_users: set[str]) -> list[str]:
    """Single line description of the function.

    Any additional context about the function can go here.

    Args:
        users: List of user identifiers to filter.
        known_users: Set of known/valid user identifiers.

    Returns:
        List of users that are not in the `known_users` set.
    """
```

- 使用描述性强、自解释的变量名。
- 遵循你正在修改的代码库中的现有模式。
- 在合理的情况下，将复杂的函数（超过 20 行）拆分为更小、更专注的函数。

### 测试要求

每个新功能或缺陷修复都必须有单元测试覆盖。

- 单元测试：`tests/unit_tests/`（不允许网络调用）
- 集成测试：`tests/integration_tests/`（允许网络调用）
- 我们使用 `pytest` 作为测试框架；如有疑问，请参考其他现有测试以获取示例。
- 测试文件结构应与源代码结构保持一致。

**检查清单：**

- [ ] 当你的新逻辑出错时，测试会失败
- [ ] 正常流程已被覆盖
- [ ] 边界情况和错误条件已被测试
- [ ] 使用 fixture/mock 处理外部依赖
- [ ] 测试是确定性的（无不稳定测试）
- [ ] 如果你的新逻辑出错，测试套件是否会失败？

### 安全性与风险评估

- 不要在用户可控输入上使用 `eval()`、`exec()` 或 `pickle`
- 正确处理异常（禁止使用裸露的 `except:`），并使用 `msg` 变量存储错误信息
- 提交前移除不可达或注释掉的代码
- 避免竞争条件或资源泄漏（文件句柄、套接字、线程）
- 确保正确清理资源（文件句柄、连接）

### 文档标准

所有公共函数均使用 Google 风格的 docstring，并包含 Args 部分。

```python title="Example"
def send_email(to: str, msg: str, *, priority: str = "normal") -> bool:
    """Send an email to a recipient with specified priority.

    Any additional context about the function can go here.

    Args:
        to: The email address of the recipient.
        msg: The message body to send.
        priority: Email priority level.

    Returns:
        `True` if email was sent successfully, `False` otherwise.

    Raises:
        InvalidEmailError: If the email address format is invalid.
        SMTPConnectionError: If unable to connect to email server.
    """
```

- 类型应放在函数签名中，而不是 docstring 中
  - 如果存在默认值，除非有后处理或条件设置，否则不要在 docstring 中重复
- 重点描述“为什么”而非“做什么”
- 记录所有参数、返回值和异常
- 描述应简洁但清晰
- 使用美式英语拼写（例如，“behavior”，而非“behaviour”）
- 不要使用 Sphinx 风格的双反引号格式（`` ``code`` ``）。在 docstring 和注释中，使用单反引号（`` `code` ``）表示内联代码引用。

#### 文档和示例中的模型引用

在 docstring 和示例代码片段中引用 LLM 时，始终使用最新的通用可用（GA）模型。除非某个模型没有 GA 版本，否则避免使用预览或测试版标识符。过时的模型名称会传递代码陈旧的信号，容易误导用户。

在编写或更新模型引用之前，请对照提供方的官方文档验证当前的模型 ID。不要依赖记忆或缓存的模型名称——它们会很快过时。

修改代码中的**已发布默认参数值**（例如类构造函数中的 `model=` 关键字参数默认值）可能构成破坏性变更——请参见上文“保持公共接口稳定”的说明。此建议适用于文档和示例，不适用于代码默认值。

对于模型的*配置数据*（功能标志、上下文窗口等），请使用下文所述的 `langchain-profiles` CLI 工具。

## 模型配置

模型配置通过位于 `libs/model-profiles` 中的 `langchain-profiles` CLI 生成。`--data-dir` 必须指向包含 `profile_augmentations.toml` 的目录，而非顶层包目录。

```bash
# Run from libs/model-profiles
cd libs/model-profiles

# Refresh profiles for a partner in this repo
uv run langchain-profiles refresh --provider openai --data-dir ../partners/openai/langchain_openai/data

# Refresh profiles for a partner in an external repo (requires echo y to confirm)
echo y | uv run langchain-profiles refresh --provider google --data-dir /path/to/langchain-google/libs/genai/langchain_google_genai/data
```

本仓库中具有配置的合作伙伴示例：

- `libs/partners/openai/langchain_openai/data/`（提供方：`openai`）
- `libs/partners/anthropic/langchain_anthropic/data/`（提供方：`anthropic`）
- `libs/partners/perplexity/langchain_perplexity/data/`（提供方：`perplexity`）

当 `--data-dir` 位于 `libs/model-profiles` 工作目录之外时，必须使用 `echo y |` 管道。

## CI/CD 基础设施

### 发布流程

每个合作伙伴包独立发布。完整流程如下：

1. **版本号提升 PR**。创建一个 PR，将三个文件各提升一行：
   - `langchain_<partner>/_version.py` — `__version__`
   - `pyproject.toml` — `version`
   - `uv.lock` — 从包目录运行 `uv lock`。如果差异中包含无关更改（例如不同本地 Python 版本导致的环境相关标记行），请撤销这些更改，仅保留该包的 `version = "..."` 行

   标题遵循常规提交规范：`release(<partner>): <version>`（例如 `release(openrouter): 0.2.6`）。使用分支名 `release/<partner>-<version>`。

   补丁版本与次要版本的提升依据仓库内已有惯例：在 `0.x` 系列中，修复和新增功能采用补丁版本提升（例如 `session_id` 字段 → 0.2.1→0.2.2，`parallel_tool_calls` → 0.2.3→0.2.4）。

2. **合并 PR** 至 `master`。

3. **触发发布工作流**。对 “🚀 Package Release” 工作流（文件 `_release.yml`，文件 ID `63880841`）运行 `gh workflow run`：

   ```bash
   gh workflow run 63880841 --repo langchain-ai/langchain \
     -f working-directory=<partner> -f release-version=<version>
   ```

`working-directory` 是工作流下拉菜单中的短合作伙伴名称（例如 `openrouter`，而非 `libs/partners/openrouter`）。

4. **工作流自动处理其余所有步骤**——请**不要**手动创建 GitHub 发布或标签。`mark-release` 作业（使用 `ncipollo/release-action`）在 PyPI 发布成功后创建 GitHub 发布、标签和发布说明。发布说明正文由前一个标签与 HEAD 之间的提交历史自动生成。

   监控运行状态：

   ```bash
   gh run view <run-id> --repo langchain-ai/langchain
   ```

完整的作业链为：构建 → 发布说明 → 预发布检查 → TestPyPI 发布 → PyPI 发布 → 创建 GitHub 发布标签。

### PR 标签与 linting

**标题 linting**（`.github/workflows/pr_lint.yml`）

**自动标注：**

- `.github/workflows/pr_labeler.yml` – 统一的 PR 标注器（大小、文件、标题、外部/内部、贡献者等级）
- `.github/workflows/pr_labeler_backfill.yml` – 手动为开放的 PR 回填标签
- `.github/workflows/auto-label-by-package.yml` – 按包进行问题标注
- `.github/workflows/tag-external-issues.yml` – 问题的外部/内部分类

### 集成测试追踪（LangSmith）

定时和手动触发的集成测试（`integration_tests.yml`）会将每次运行追踪至 LangSmith，以便失败情况能回溯到原始的 Actions 运行。`_release.yml` 也会运行集成测试，但目前未配置 LangSmith 追踪。

**CI 设置的环境变量：**

- `LANGSMITH_API_KEY` — 用于认证 LangSmith（仓库密钥，限定在 `integration_tests.yml` 中的“定时测试” GitHub 环境）。
- `LANGSMITH_TRACING: "true"` — 启用测试进程的追踪。
- `LANGSMITH_PROJECT` — 追踪数据发送的目标项目。默认通过仓库变量覆盖为 `scheduled-testing-py`：`${{ vars.LANGSMITH_PROJECT || 'scheduled-testing-py' }}`。如需更改项目，请在 GitHub 设置中修改 `LANGSMITH_PROJECT` 仓库变量——不要在工作流中硬编码。
- `LANGSMITH_TAGS` — 以逗号分隔的标签，用于标识运行：`github-actions`、矩阵工作目录（例如 `libs/partners/openai`）、Python 版本和提交 SHA。
- `LANGSMITH_METADATA` — 由“构建 LangSmith 元数据”步骤生成的 JSON 对象，包含 `github_sha`、`github_run_id`、`github_run_attempt`、`github_run_url`、`github_workflow`、`github_event`、`github_ref`、`working_directory` 和 `python_version`。

**追踪桥接插件：** LangSmith SDK 并不原生读取环境中的 `LANGSMITH_TAGS` 或 `LANGSMITH_METADATA`。位于 `libs/standard-tests/langchain_tests/_langsmith_plugin.py` 的 pytest 插件通过在测试会话期间注入 `langsmith.run_helpers.tracing_context` 来弥补这一缺口。该插件仅在 `GITHUB_ACTIONS=true` 时激活，因此本地开发不受影响。通过任何依赖 `langchain-tests` 包的 `pytest11` 入口点自动发现。

**单元测试隔离：** 单元测试绝不能发起网络调用或发送追踪数据。`libs/core` 目录下的 `Makefile` 中的 `make test` 目标使用 `env -u` 取消设置追踪相关变量（`LANGCHAIN_TRACING_V2`、`LANGCHAIN_API_KEY`、`LANGSMITH_API_KEY`、`LANGSMITH_TRACING`、`LANGCHAIN_PROJECT`），再运行 pytest。此外，`libs/core/tests/unit_tests/runnables/conftest.py` 中有一个会话范围的自动应用 fixture，显式禁用可运行单元测试的追踪，并在之后恢复原始环境。

### 在 CI 中添加新合作伙伴

当添加新的合作伙伴包时，请更新以下文件：

- `.github/ISSUE_TEMPLATE/*.yml` – 添加至包下拉菜单
- `.github/dependabot.yml` – 添加依赖项更新条目
- `.github/scripts/pr-labeler-config.json` – 添加文件规则及作用域到标签的映射
- `.github/workflows/_release.yml` – 如需，添加 API 密钥密钥
- `.github/workflows/auto-label-by-package.yml` – 添加包标签
- `.github/workflows/check_diffs.yml` – 添加至变更检测
- `.github/workflows/integration_tests.yml` – 添加集成测试配置
- `.github/workflows/pr_lint.yml` – 添加至允许的作用域列表

## GitHub Actions & Workflows

此仓库要求操作必须固定到完整的提交 SHA。尝试使用标签将导致失败。请使用 `gh` CLI 工具进行查询。验证标签不是带注解的标签对象（这类对象需要解引用）。

## 额外资源

- **文档：** https://docs.langchain.com/oss/python/langchain/overview 及源码位于 https://github.com/langchain-ai/docs 或 `../docs/`。建议本地安装，并使用文件搜索工具以获得最佳效果。如需程序化访问，可使用 `.mcp.json` 中定义的 docs MCP 服务器。
- **贡献指南：** [贡献指南](https://docs.langchain.com/oss/python/contributing/overview)

<!-- OPENWIKI:START -->

## OpenWiki

此仓库生成了 `openwiki/` 证据索引。它作为可选的即时上下文，非启动时必需阅读内容。

- 以源代码和测试为权威依据。简要说明中的未知项与审查事项是验证缺口，而非自动要求。
- 优先采用最窄且安静的验证方式来证明变更行为。保留完整的失败输出。

计划中的 OpenWiki GitHub Actions 工作流会刷新仓库的维基页面。除非明确要求，否则不要手动编辑生成的 OpenWiki 页面；应优先更新源代码或文档，让 OpenWiki 自动重新生成。

<!-- OPENWIKI:END -->

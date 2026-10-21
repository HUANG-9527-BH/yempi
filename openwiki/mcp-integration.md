---
type: "Reference"
title: "MCP (Model Context Protocol) Integration"
description: "MCPAdapter bridges MCP servers to LangChain agents, discovering tools, handling protocol negotiation via FastMCP, managing multiple transports, and supporting mid-call user input via LangGraph interrupts."
tags: [mcp, protocol, tools, adapter, integration, langgraph]
verified:
  - by: openwiki/0.5.0
    at: 2026-09-28T08:35:20.640Z
sources:
  - id: openwiki-source-6d1e3478d5b63988ee177552
    resource: repo://libs/langchain_v1/examples/mcp/auth_bearer.py
  - id: openwiki-source-4bad19dc422af3ebb00e7f2f
    resource: repo://libs/langchain_v1/examples/mcp/auth_oauth.py
  - id: openwiki-source-46fd56b09fa62a41e3c41f08
    resource: repo://libs/langchain_v1/examples/mcp/destructive_interrupt.py
  - id: openwiki-source-8b95e4b88972026f6c7678a3
    resource: repo://libs/langchain_v1/examples/mcp/graph_factory.py
  - id: openwiki-source-cfb2965ed32b54e99ffb6328
    resource: repo://libs/langchain_v1/examples/mcp/multi_server.py
  - id: openwiki-source-71f7ffcbb69cda81c2e3f940
    resource: repo://libs/langchain_v1/examples/mcp/protocol_eras.py
  - id: openwiki-source-37b31519003157eb1b1bdaef
    resource: repo://libs/langchain_v1/examples/mcp/README.md
  - id: openwiki-source-6df781509d081d60a037331b
    resource: repo://libs/langchain_v1/examples/mcp/tool_errors.py
  - id: openwiki-source-caa1f747bb1ba9b6514eeaac
    resource: repo://libs/langchain_v1/examples/mcp/transports.py
  - id: openwiki-source-0a3228970b0eadc4bcadbb5d
    resource: repo://libs/langchain_v1/langchain/mcp/adapter.py
  - id: openwiki-source-b4c5eca79ce58abf486c2776
    resource: repo://libs/langchain_v1/langchain/mcp/elicitation.py
  - id: openwiki-source-4715c337e9b93b9d00846133
    resource: repo://libs/langchain_v1/langchain/mcp/tools.py
generated: { by: "openwiki/0.5.0", at: "2026-09-28T08:35:20.640Z" }
---
## 概述

Model Context Protocol (MCP) 是一种用于 LLM 应用程序发现并调用外部服务器暴露的工具的标准。`langchain.mcp` 提供了 `MCPAdapter` 类，该类可发现 MCP 工具，并将其转换为适用于智能体使用的 LangChain `BaseTool` 对象。该适配器处理以下功能：

- **协议协商** 通过 FastMCP（客户端会话管理、版本握手）
- **多种传输方式**（进程内、stdio 子进程、HTTP 流式）
- **多服务器路由** 通过 `ClientGroup` 实现，支持自动工具名称前缀添加
- **调用过程中的用户输入** 通过 LangGraph 中断机制（征询）
- **错误恢复** 将工具报告的错误传递给模型以实现重试

## MCP 概念

MCP 是一种请求-响应协议，其特点如下：

- **客户端**（如 LangChain 智能体）发现并调用服务器上的工具
- **服务器** 暴露工具目录，定义工具模式，并执行调用
- **工具** 是具有类型参数的命名、文档化函数
- **工具返回的内容** 以内容块形式表示（文本、图像、文件、结构化数据）
- **协议版本** 随时间演进（例如，2025-11-25 使用 `initialize` 握手；2026-07-28 使用 `server/discover`）

当为每个服务器使用独立适配器时，来自不同协议时代的服务器可在同一智能体中共存。

## 架构：MCPAdapter 与 FastMCP

`MCPAdapter` 是主要的用户入口点。它包装一个或多个 FastMCP 客户端，并将发现的工具暴露为 LangChain 工具：

```python
async with MCPAdapter(target) as adapter:
    tools = await adapter.list_tools()
    agent = create_agent("anthropic:claude-sonnet-5", tools)
```

### 目标类型与传输推断

FastMCP 根据目标推断传输方式。`MCPAdapter` 接受以下类型：

- **`str`（仅限 HTTP/HTTPS URL）** — 通过可流式的 HTTP 访问。字符串验证会拒绝非 URL 字符串（如本地文件路径），以防止静默执行本地代码。

  ```python
  async with MCPAdapter("https://api.example.com/mcp") as adapter:
      tools = await adapter.list_tools()
  ```

- **`Path`** — 通过 stdio 作为子进程启动。脚本必须存在。

  ```python
  async with MCPAdapter(Path("server.py")) as adapter:
      tools = await adapter.list_tools()
  ```

- **`FastMCP` 实例** — 进程内服务器，无需网络或子进程。适用于测试和嵌入式部署。

  ```python
  server = FastMCP("weather")
  @server.tool
  def forecast(city: str) -> str:
      return f"{city}: sunny"
  
  async with MCPAdapter(server) as adapter:
      tools = await adapter.list_tools()
  ```

- **`Client` 或 `ClientGroup`** — 预构建的 FastMCP 客户端。允许自定义配置（认证、缓存、传输方式）。

  ```python
  client = Client("https://api.example.com/mcp", auth="bearer-token")
  async with MCPAdapter(client) as adapter:
      tools = await adapter.list_tools()
  ```

- **`MCPConfig`（字典）** — 多个服务器，每个拥有独立的传输和认证方式。FastMCP 使用配置键对工具进行前缀标记，避免冲突。

  ```python
  config = {
      "mcpServers": {
          "weather": {"command": "python", "args": ["weather_server.py"]},
          "calc": {"url": "https://api.example.com/mcp"},
      }
  }
  async with MCPAdapter(config) as adapter:
      tools = await adapter.list_tools()  # ["weather_forecast", "calc_add", ...]
  ```

- **`ClientTransport`** — 显式传输方式（HTTP、stdio 或自定义），用于细粒度控制。

## 工具发现与转换

`adapter.list_tools()` 调用 `fastmcp.Client.list_tools()` 获取远程工具，然后通过 `as_langchain_tool()` 进行转换：

### 发现流程

1. **连接**：进入适配器上下文（连接底层客户端）
2. **获取**：调用 `client.list_tools(cache_mode=...)` 获取工具定义
3. **转换**：对每个 MCP 工具，调用 `as_langchain_tool(tool, client)` 生成 LangChain 工具

### 转换细节

每个 MCP 工具均转换为一个 `StructuredTool`，包含：

- **name、description、args_schema** — 来自 MCP 工具定义
- **coroutine** — 异步函数，通过客户端调用 MCP 工具
- **response_format="content_and_artifact"** — 返回模型可见的内容块和结构化数据
- **metadata["mcp"]** — 携带工具注解、服务器身份标识及破坏性提示
- **handle_tool_error=_handle_mcp_tool_error** — 错误处理器，将 MCP 工具错误传递给模型

### 多服务器工具前缀

使用 `ClientGroup` 或 `MCPConfig` 时，工具名称会以服务器键为前缀（例如 `weather_forecast` = `weather` + `forecast`）。这可防止命名冲突，并使工具来源清晰可见。适配器内部的路由机制确保每个调用都能到达正确的服务器：

```python
adapter = MCPAdapter(group)
tools = await adapter.list_tools()  # Tool names are prefixed
# e.g., ["weather_forecast", "weather_current_conditions", "calc_add"]

# When a tool is called, the router resolves it to the correct server member
for tool in tools:
    if tool.name == "weather_forecast":
        # Routed to the "weather" client by resolve_tool()
        result = await tool.ainvoke({"city": "Oslo"})
```

### 模式规范化

`_normalize_mcp_schema()` 在提供方模式转换过程中保持开放对象参数的开放性，对未显式定义属性的对象属性添加 `additionalProperties: True`。这允许模型在模式未对其施加约束时传递任意键。

## 工具调用与结果转换

当调用工具时：

1. **征询检测**：适配器检查底层客户端是否具备驱动 LangGraph 中断的能力
2. **工具调用路由**：
   - 对于 `ClientGroup`，调用 `resolve_tool(name)` 以查找成员客户端及其上游名称
   - 若客户端具备中断能力且服务器为现代版本（2026-07-28 及以后），则调用 `_call_tool_with_interrupts()`
   - 否则直接调用 `fastmcp.Client.call_tool()`
3. **结果转换**：将 MCP 内容块转换为 LangChain 内容块
4. **错误处理**：若服务器报告 `isError=True`，抛出 `_MCPToolExecutionError`（即 `ToolException`），该异常将变为带有 `status="error"` 的 `ToolMessage`，使模型能够感知并重试
5. **产物提取**：将 `structured_content`（JSON、表格等）提取至独立的产物字段

### 内容块转换

支持的内容类型：

- **文本** — 纯字符串 → `TextContentBlock`
- **图像** — base64 编码（`ImageContent`）或 URL 引用（`ResourceLink`）→ `ImageContentBlock`
- **文件** — base64 或 URL 引用 → `FileContentBlock`
- **资源** — 嵌入的二进制/文本数据（`EmbeddedResource`）或 URL 链接（`ResourceLink`）→ 若 MIME 类型为 image/* 则转为 `ImageContentBlock`，否则转为 `FileContentBlock`
- **音频** — 尚不支持（抛出 `NotImplementedError`）

## 征询：调用过程中的用户输入

某些 MCP 工具在调用过程中需要向用户提问才能完成（例如：“确认此操作？”）。为避免挂起或报错，服务器返回一个 `InputRequiredResult`，描述其所需内容。适配器将其转换为 LangGraph 的 `interrupt()`，使人类可进行响应，运行可无缝恢复。

### 征询流程

**中断准备：**

1. 构造时，`MCPAdapter` 会调用每个底层客户端的 `_arm_for_interrupts()`
2. 此操作将 `_declare_elicitation_capability` 设置为征询回调，并标记客户端带有 `_ARMED_MARKER`
3. FastMCP 向服务器通告 `elicitation` 能力（仅限现代服务器）
4. 对已自带处理程序的预构建客户端，先进行克隆，确保调用者对象不会被修改

**中断循环：**

1. `as_langchain_tool()` 检查客户端是否已准备就绪（`_drives_interrupts()`）且服务器为现代版本（协议版本 ≥ 2026-07-28）
2. 若满足条件，调用 `_call_tool_with_interrupts()` 而非普通的 `call_tool()`
3. 循环流程：
   - 以 `allow_input_required=True` 发起工具调用
   - 若结果为 `InputRequiredResult`，则提取征询请求
   - 仅保留 `ElicitRequest` 类型（拒绝采样、根节点、续传请求）
   - 抛出 `interrupt(payload)`，暂停运行
   - 恢复后，接收以请求 ID 为键的 `responses` 字典中的答案
   - 通过 `_build_responses()` 构建响应载荷，验证每项回答
   - 使用 `input_responses=...` 和 `request_state=...`（不透明，原样回传）重新尝试调用
   - 重复直至工具返回终止结果（非 `InputRequiredResult`）

**请求类型：**

- **表单** — 服务器请求结构化数据（JSON，需匹配某个模式）
- **URL** — 服务器要求人类访问某个 URL（例如用于批准或身份验证）

**响应操作：**

- **接受** — 回答问题（表单：提供与模式匹配的 `content`；URL：无需额外操作）
- **拒绝** — 仅拒绝此请求（工具继续处理其他请求）
- **取消** — 完全拒绝（中止工具调用）

### 协议兼容性

中断循环仅在现代服务器（2026-07-28 及之后版本）上运行，这些服务器会返回 `InputRequiredResult`。旧版服务器（2025-11-25）从不触发该机制，因此其行为保持不变。已预构建的客户端若已有处理器，则会使用该处理器，从而允许自定义获取策略。

## 传输类型

三种主要传输方式，由 FastMCP 自动选择：

### 内存内

一个 `FastMCP` 服务器实例在同一流程中运行，无需子进程或网络通信：

```python
from langchain.mcp import MCPAdapter
from fastmcp import FastMCP

server = FastMCP("weather")
@server.tool
def get_forecast(city: str) -> str:
    return f"{city}: sunny"

async with MCPAdapter(server) as adapter:
    tools = await adapter.list_tools()
```

**适用场景**：测试、开发、具有完全控制权的单应用部署。

### Stdio

脚本（Python 或 Node.js）作为子进程启动，并通过 stdin/stdout 进行通信：

```python
from pathlib import Path
from langchain.mcp import MCPAdapter

script_path = Path("server.py")  # must exist
async with MCPAdapter(script_path) as adapter:
    tools = await adapter.list_tools()
```

**适用场景**：本地开发、私有工具、沙箱环境。每个适配器实例会启动一个子进程。

### HTTP（流式）

通过 HTTP(S) 使用流式传输连接远程 MCP 服务器：

```python
from langchain.mcp import MCPAdapter

url = "https://api.example.com/mcp"
async with MCPAdapter(url) as adapter:  # no auth
    tools = await adapter.list_tools()
```

**适用场景**：公共 MCP 服务器、云服务、第三方集成。

## 认证

MCP 服务器可要求凭证。适配器和客户端支持：

- **Bearer token** — 静态令牌，无发现或刷新机制
- **OAuth 2.1** — 完整流程，包含动态客户端注册、浏览器重定向和令牌交换
- **自定义认证** — 任意 `httpx2.Auth` 实现

```python
from fastmcp.client import Client
from langchain.mcp import MCPAdapter

# Bearer token
async with MCPAdapter(Client("https://api.example.com/mcp", auth="token-value")) as adapter:
    tools = await adapter.list_tools()

# OAuth (opens browser, auto-approves on demo server)
async with MCPAdapter(Client("https://api.example.com/mcp", auth="oauth")) as adapter:
    tools = await adapter.list_tools()
```

对于多服务器部署，可在 `MCPConfig` 中为每个服务器指定认证信息：

```python
config = {
    "mcpServers": {
        "api1": {
            "command": "python",
            "args": ["server.py"],
            "auth": {"type": "bearer", "token": "secret-1"},
        },
        "api2": {
            "command": "python",
            "args": ["server.py"],
            "auth": {"type": "oauth"},
        },
    }
}

async with MCPAdapter(config) as adapter:
    tools = await adapter.list_tools()
```

## 元数据与工具注解

MCP 工具可携带注解（例如 `destructiveHint=True` 表示删除操作）。这些注解会在 LangChain 工具中以 `metadata["mcp"]["tool"]["annotations"]` 形式呈现：

```python
from mcp.server.mcpserver import MCPServer
from mcp.types import ToolAnnotations

server = MCPServer("example")

@server.tool(annotations=ToolAnnotations(destructiveHint=True))
def delete_file(path: str) -> str:
    return f"Deleted {path}"
```

客户端可读取此信息，以在未经批准前阻止破坏性工具的执行，而无需硬编码工具名称：

```python
def _is_destructive(tool):
    annotations = (tool.metadata or {}).get("mcp", {}).get("tool", {}).get("annotations", {})
    return annotations.get("destructive_hint", False)

destructive_tools = [tool.name for tool in tools if _is_destructive(tool)]
# Pass to HumanInTheLoopMiddleware or similar approval gate
```

工具元数据还包含 `mcp.server`，其中包含服务器身份信息（名称、版本），使客户端能够根据来源区分工具。

## 错误处理与恢复

### MCP 工具错误

当服务器报告 `isError=True` 时：

- 转换为 `ToolMessage`，状态为 `status="error"`，并附带服务器消息
- 对模型可见，模型可修正输入并重试
- 示例：除零错误、文件未找到、远程服务器网络超时

### 传输错误

网络故障、子进程失败或响应格式错误：

- 抛出异常；运行失败
- 模型无法对此类错误作出响应，因此应在编排层进行重试
- 示例：无法访问的 URL、子进程崩溃、服务器返回无效 JSON

## 多服务器设置与配置

### MCPConfig 集群

要连接多个 MCP 服务器并将它们的所有工具暴露给单一智能体：

```python
from langchain.mcp import MCPAdapter

config = {
    "mcpServers": {
        "weather": {"command": "python", "args": ["weather_server.py"]},
        "calc": {"command": "python", "args": ["calc_server.py"]},
    }
}

async with MCPAdapter(config) as adapter:
    tools = await adapter.list_tools()  # ["weather_forecast", "calc_add", ...]
    agent = create_agent("anthropic:claude-sonnet-5", tools)
```

FastMCP 会自动根据配置键前缀化工具（例如 `weather_` + `forecast` = `weather_forecast`）。这可防止命名冲突，并使工具来源清晰可见。适配器内部的路由机制确保每个调用都能到达正确的服务器。服务器可在同一配置中混合使用多种传输方式：部分通过 stdio，部分通过 HTTP，部分为进程内调用。

### ClientGroup

适用于程序化多服务器设置（例如按请求进行身份验证）：

```python
from fastmcp import Client
from fastmcp.client.group import ClientGroup
from langchain.mcp import MCPAdapter

group = ClientGroup({
    "weather": Client("https://api1.example.com/mcp", auth="token-1"),
    "calc": Client("https://api2.example.com/mcp", auth="token-2"),
})

async with MCPAdapter(group) as adapter:
    tools = await adapter.list_tools()
```

## 协议时代与版本协商

一个智能体中可共存两个 MCP 协议时代，方法是为每个时代使用独立的适配器：

```python
from fastmcp import Client
from langchain.mcp import MCPAdapter

# Legacy era server (2025-11-25, `initialize` handshake)
legacy_client = Client(legacy_server(), mode="legacy")

# Modern era server (2026-07-28, `server/discover` handshake)
modern_client = Client(modern_server(), mode="auto")

async with MCPAdapter(legacy_client) as legacy_adapter, \
           MCPAdapter(modern_client) as modern_adapter:
    tools = await legacy_adapter.list_tools() + await modern_adapter.list_tools()
    agent = create_agent("anthropic:claude-sonnet-5", tools)
```

单个 `MCPConfig` 集群会在所有成员间协商确定一个时代：若任一成员仅支持旧时代，则整个集群将降级至该时代。使用独立适配器可确保每台服务器始终使用其连接所支持的最佳时代。

## 按请求设置的图工厂模式

适用于按请求设置服务器（例如按用户凭证）时，可在图工厂中创建工具：

```python
async def make_graph(runtime):
    user = runtime.user.identity
    auth = BearerAuth(token_for(user))
    group = ClientGroup({
        "api1": Client("https://api.example.com/mcp", auth=auth),
        "api2": Client("https://api.example.com/mcp", auth=auth),
    })
    tools = await MCPAdapter(group).list_tools()
    return create_agent("anthropic:claude-sonnet-5", tools)

# Use with langgraph dev or persistent deployments
agent = CompiledStateGraph(make_graph)
```

对于跨运行状态（如共享 HTTP 连接池、按用户划分的响应缓存分区），应在工厂外部实例化：

```python
import httpx2
from langchain.mcp import MCPAdapter
from fastmcp import Client
from fastmcp.client.group import ClientGroup
from mcp.client.caching import InMemoryResponseCacheStore, CacheConfig

_pool = httpx2.AsyncHTTPTransport()
_cache = InMemoryResponseCacheStore()

async def make_graph(runtime):
    user = runtime.user.identity
    group = ClientGroup({
        name: Client(
            url,
            httpx_client_factory=lambda: httpx2.AsyncClient(transport=_pool),
            cache=CacheConfig(store=_cache, partition=user),
        )
        for name, url in SERVERS.items()
    })
    tools = await MCPAdapter(group).list_tools(cache_mode="use")
    return create_agent("anthropic:claude-sonnet-5", tools)
```

## 响应缓存与缓存模式

FastMCP 在服务器端缓存工具列表，并支持按主体隔离。适配器的 `cache_mode` 参数控制缓存使用方式：

- **`"use"`**（默认）—— 若缓存新鲜（在服务器的 TTL 提示范围内），则从缓存提供
- **`"refresh"`** —— 从服务器刷新，重新填充缓存
- **`"bypass"`** —— 完全跳过缓存

```python
tools = await adapter.list_tools(cache_mode="refresh")
```

缓存及其按主体的隔离配置由客户端自身决定（`Client(cache=...)`）；此参数仅选择发现如何读取缓存。已配置的缓存会被尊重——注意，裸调用 `ClientGroup.list_tools()` 默认为 `refresh`，而 `MCPAdapter.list_tools()` 默认为 `use`。

## 示例

LangChain 在 `examples/mcp/` 中提供了可运行的示例：

| 示例 | 展示内容 | 模型 | 网络 |
|------|--------|:----:|:----:|
| `transports.py` | 内存、stdio、HTTP 传输方式 | | |
| `remote_server.py` | 公共 MCP 服务器（DeepWiki） | ✅ | ✅ |
| `multi_server.py` | 使用工具前缀的 `MCPConfig` 集群 | ✅ | |
| `graph_factory.py` | 图工厂中的按用户凭证 | | |
| `protocol_eras.py` | 旧时代与新时代服务器共存 | ✅ | |
| `tool_errors.py` | 工具失败与模型恢复 | ✅ | |
| `elicitation.py` | 服务器在调用过程中请求用户输入 | ✅ | |
| `destructive_interrupt.py` | 通过元数据控制破坏性工具 | ✅ | |
| `auth_bearer.py` | 静态 bearer token | | |
| `auth_oauth.py` | 支持动态客户端注册的 OAuth 2.1 | | |

通过以下命令运行示例：

```bash
uv sync --extra mcp --extra anthropic
export ANTHROPIC_API_KEY=...
uv run examples/mcp/transports.py
```

## 集成点

### `create_agent`

来自 `MCPAdapter.list_tools()` 的工具可直接传递给 `create_agent()`，该函数通过智能体的模型和执行器路由工具调用。即使适配器上下文退出后，这些工具仍可调用，因为它们持有对底层客户端的引用。

```python
async with MCPAdapter(target) as adapter:
    tools = await adapter.list_tools()
    agent = create_agent("anthropic:claude-sonnet-5", tools)
    
# Tools are still callable after adapter context exits
result = await agent.ainvoke({"messages": [...]})
```

### LangGraph 中断与检查点

由引出驱动的中断需要检查点，以便运行能够暂停并恢复：

```python
from langgraph.checkpoint.memory import InMemorySaver
from langchain.mcp.elicitation import ELICITATION_INTERRUPT_TYPE

agent = create_agent(
    "anthropic:claude-sonnet-5",
    tools,
    checkpointer=InMemorySaver(),
)

config = {"configurable": {"thread_id": "user-1"}}

# First invocation pauses on elicitation interrupt
paused = await agent.ainvoke(
    {"messages": [{"role": "user", "content": "Approve deletion of file X"}]},
    config,
)

# Check for interrupt
if paused.values.get("interrupt"):
    interrupt_data = paused.values["interrupt"]
    # Handle based on interrupt_data["type"] == ELICITATION_INTERRUPT_TYPE
    # ...
    
    # Resume with answers
    resumed = await agent.ainvoke(
        {"interrupt_answers": {...}},  # Depends on your graph's schema
        config,
    )
```


### 工具元数据与中间件

智能体可以对工具调用应用中间件，以进行权限控制或日志记录。MCP 工具元数据（例如 `destructiveHint`）可与 `HumanInTheLoopMiddleware` 集成：

```python
from langchain.agents.middleware import HumanInTheLoopMiddleware

def _is_destructive(tool):
    annotations = (tool.metadata or {}).get("mcp", {}).get("tool", {}).get("annotations", {})
    return annotations.get("destructive_hint", False)

interrupt_on = {
    tool.name: InterruptOnConfig(...)
    for tool in tools
    if _is_destructive(tool)
}

agent = create_agent(..., middleware=[HumanInTheLoopMiddleware(interrupt_on=interrupt_on)])
```


## 日志与可观测性

`MCPAdapter` 和 `as_langchain_tool()` 对 LangChain 的日志和可观测性钩子完全透明。工具调用会作为 `ToolMessage` 事件记录在智能体的消息历史中。诱发中断及其响应可通过 LangGraph 内置追踪，在运行状态转换中清晰可见。

## 不变性与失败语义

- **工具可用性**：一旦 `list_tools()` 完成，即使适配器上下文已退出，工具仍可调用（它们持有客户端）
- **诱发重试**：当工具以答案恢复时，整个调用将从头重新执行。若服务器在询问前执行了工作，则每次诱发轮次都会重复该工作
- **错误传播**：传输错误会作为异常传播；MCP 工具错误（`isError=True`）会变为模型可见的 `ToolMessage` 错误
- **客户端复用**：客户端是可重入的；即使连接已在别处持有，工具仍可打开其客户端
- **预构建客户端克隆**：若调用者传入一个已存在诱发处理器的客户端，该客户端会被克隆，确保调用者的对象不会被修改
- **组路由**：来自 `ClientGroup` 的工具会根据配置键添加前缀；路由器通过 `resolve_tool()` 将每个调用解析到正确的成员
- **无并发诱发**：诱发答案按顺序驱动，每轮仅一个 `interrupt()`，因此 LangGraph 可通过顺序匹配恢复值
- **字符串目标验证**：裸字符串目标必须是 http/https URL，以防止静默执行本地文件

## 扩展点

- **自定义传输**：传入任意 `fastmcp.ClientTransport` 以支持非标准协议
- **自定义认证**：实现 `httpx2.Auth` 以支持超出 bearer token 和 OAuth 的认证方案
- **自定义元数据处理器**：继承 `StructuredTool` 以自定义 MCP 元数据在 LangChain 工具上的暴露方式
- **自定义错误处理器**：重写 `_handle_mcp_tool_error()` 或为工具提供自己的 `handle_tool_error`
- **自定义诱发机制**：传入带有自定义 `elicitation_handler` 的预构建客户端，以覆盖默认的中断驱动机制

## 相关页面

- **[tools.md](/openwiki/tools.md)** — LangChain 工具抽象、`BaseTool`、`StructuredTool`、元数据
- **[agent-factory.md](/openwiki/agent-factory.md)** — 智能体编排、`create_agent`、工具路由、中间件

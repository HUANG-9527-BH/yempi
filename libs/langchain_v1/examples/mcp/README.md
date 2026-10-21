# `langchain.mcp` 示例

可直接运行、自包含的脚本。每个脚本都会自行启动所需的 MCP 服务器，因此无需单独启动任何东西。

```bash
uv sync --extra mcp --extra anthropic
export ANTHROPIC_API_KEY=...          # needed by the examples that run an agent
uv run examples/mcp/transports.py
```

| 示例 | 演示内容 | 模型 | 联网 |
|---|---|:-:|:-:|
| `transports.py` | 一个适配器覆盖内存（in-memory）、stdio 与 HTTP 三种传输 | | |
| `remote_server.py` | 将适配器指向一个公共 MCP 服务器 | ✅ | ✅ |
| `multi_server.py` | 一个适配器后接多个服务器，工具按服务器加前缀 | ✅ | |
| `graph_factory.py` | 由 `langgraph dev` 图的每次运行共享的一个长生命周期适配器 | | |
| `protocol_eras.py` | 一个智能体同时持有来自两个 MCP 协议时代的工具 | ✅ | |
| `tool_errors.py` | 让失败的工具触达模型，以便其重试 | ✅ | |
| `elicitation.py` | 服务器在调用过程中通过 `interrupt()` 请求人类介入 | ✅ | |
| `destructive_interrupt.py` | 依据工具元数据，将破坏性工具置于审批流程之后 | ✅ | |
| `auth_bearer.py` | 位于静态 bearer token 保护之后的服务器 | | |
| `auth_oauth.py` | 包含动态客户端注册的完整 OAuth 2.1 流程 | | |

`remote_server.py` 会调用 DeepWiki（一个公共 MCP 服务器），因此需要联网。`auth_oauth.py` 会打开一个浏览器标签页；演示用授权服务器会自动批准，因此会直接重定向回来。

`_servers.py` 保存着各示例共享的小型 MCP 服务器，`_stdio_server.py` 则是通过 stdio 作为子进程启动的入口点。两者都不属于所演示 API 的一部分。

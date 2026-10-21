"""运行图工厂所连接的两个受保护的 MCP 服务器。

`graph_factory.py` 将 `SERVERS` 指向固定端口；这会在每个端口上启动一个服务器。
两个服务器均使用共享密钥对（`MCP_DEMO_KEYFILE`）验证 token，该密钥对与 `langgraph dev`
生成的一致，因此在工厂进程中生成的 token 在此处也有效。

    MCP_DEMO_KEYFILE=/path/to/keys.json uv run examples/mcp/_fleet_servers.py
"""

from __future__ import annotations

import multiprocessing

from _servers import PUBLIC_KEY, run_guarded_server

PORTS = {"calendar": 8001, "docs": 8002}


def main() -> None:
    procs = [
        multiprocessing.Process(
            target=run_guarded_server,
            args=("127.0.0.1", port, name, PUBLIC_KEY),
            daemon=True,
        )
        for name, port in PORTS.items()
    ]
    for proc in procs:
        proc.start()
    for proc in procs:
        proc.join()


if __name__ == "__main__":
    main()

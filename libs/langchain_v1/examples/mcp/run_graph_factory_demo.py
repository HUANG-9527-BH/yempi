"""通过带有认证的 real `langgraph dev` 服务器驱动图工厂。

This is the end-to-end check the example describes but cannot show inline: two
users hit one deployment, each authenticates with their own `x-api-key`, and
each run's agent talks to the MCP fleet as that user. It:

  1. mints one shared keypair (so the token the factory mints validates on the
     servers) into a temp `MCP_DEMO_KEYFILE`,
  2. starts the two guarded MCP servers on the ports `graph_factory.SERVERS`
     names,
  3. starts `langgraph dev` over `langgraph.json` (which wires `auth.py`),
  4. runs the `fleet` graph once per user and checks each agent reports its own
     identity from the servers' `whoami` tools.

    ANTHROPIC_API_KEY=... uv run examples/mcp/run_graph_factory_demo.py
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from fastmcp.server.auth.providers.jwt import RSAKeyPair
from langgraph_sdk import get_client

HERE = Path(__file__).parent
DEV_URL = "http://127.0.0.1:2024"
USERS = ("alice", "bob")


def _write_keyfile() -> Path:
    """将一个密钥对序列化供每个进程读取，确保 token 在各进程间可验证。"""
    keys = RSAKeyPair.generate()
    path = Path(tempfile.mkstemp(suffix=".json", prefix="mcp-demo-keys-")[1])
    path.write_text(
        json.dumps(
            {"private_key": keys.private_key.get_secret_value(), "public_key": keys.public_key}
        )
    )
    return path


async def _wait_until_up(url: str, timeout: float = 60.0) -> None:
    """轮询直到开发服务器注册了 `fleet` 图，或超时。"""
    deadline = time.monotonic() + timeout
    client = get_client(url=url, headers={"x-user-id": "healthcheck"})
    while time.monotonic() < deadline:
        with contextlib.suppress(Exception):
            assistants = await client.assistants.search(graph_id="fleet")
            if assistants:
                return
        await asyncio.sleep(1.0)
    msg = f"dev server at {url} never registered the 'fleet' graph"
    raise RuntimeError(msg)


async def _run_as(user: str) -> str:
    """以 `user` 身份运行 fleet 图并返回智能体的最终文本。"""
        # `x-user-id` 头部是 `auth.py` 将其转换为本次运行身份的内容。
    client = get_client(url=DEV_URL, headers={"x-user-id": user})
    thread = await client.threads.create()
    result = await client.runs.wait(
        thread["thread_id"],
        "fleet",
        input={
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Call both whoami tools. Reply with exactly: calendar=<value> docs=<value>."
                    ),
                }
            ]
        },
    )
    return result["messages"][-1]["content"]


async def _drive() -> int:
    await _wait_until_up(DEV_URL)
    failures = 0
    for user in USERS:
        reply = await _run_as(user)
        text = reply if isinstance(reply, str) else json.dumps(reply)
        ok = f"calendar={user}" in text and f"docs={user}" in text
        print(f"{user:6} -> {'PASS' if ok else 'FAIL'}: {text!r}")
        failures += not ok
    return failures


def main() -> int:
    """启动舰队 + 开发服务器，让每个用户依次通过，然后报告结果。"""
    keyfile = _write_keyfile()
    env = {**os.environ, "MCP_DEMO_KEYFILE": str(keyfile)}
        # 使用与此解释器相邻的 `langgraph`，而非 PATH 上第一个找到的版本。
        # PATH：图导入此虚拟环境的依赖项（`httpx2`、`fastmcp`
    # 等），因此必须在此虚拟环境中运行，而非全局安装的 CLI。
        # `langgraph dev` 将其图注册表持久化存储于
    # `.langgraph_api`；之前运行遗留的旧注册表会遮蔽本配置中的 `fleet` 图。
    langgraph = Path(sys.executable).with_name("langgraph")
    if not langgraph.exists():
        msg = f"`langgraph` CLI not found at {langgraph}; install langgraph-cli[inmem]."
        raise RuntimeError(msg)

        # 一个传递依赖项（`jsonschema_specifications`）首次验证工具模式时会同步扫描其包数据。`langgraph
    # dev` 默认拒绝事件循环上的阻塞调用；此处允许它们，以便演示能正常运行。
        # 较早运行遗留的配置会遮蔽此配置的 `fleet` 图。
    shutil.rmtree(HERE / ".langgraph_api", ignore_errors=True)

    servers = subprocess.Popen(  # noqa: S603
        [sys.executable, str(HERE / "_fleet_servers.py")], env=env, cwd=HERE
    )
    devlog = Path("/tmp/lg-driver.log")  # noqa: S108
    print(f"(dev log -> {devlog})")
    dev = subprocess.Popen(  # noqa: S603
        [
            str(langgraph),
            "dev",
            "--no-browser",
            "--port",
            "2024",
            "--config",
            "langgraph.json",
                        # 一个传递依赖项（`jsonschema_specifications`）会扫描其
                        # 同步打包工具模式的首次使用
                        # 已验证。`langgraph dev` 会拒绝事件中的阻塞调用
                        # 默认循环；允许此处这样写，以便演示能运行。
            "--allow-blocking",
        ],
        env=env,
        cwd=HERE,
        stdout=devlog.open("w"),
        stderr=subprocess.STDOUT,
    )
    try:
        return asyncio.run(_drive())
    finally:
        for proc in (dev, servers):
            proc.terminate()
            with contextlib.suppress(Exception):
                proc.wait(timeout=10)
        keyfile.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())

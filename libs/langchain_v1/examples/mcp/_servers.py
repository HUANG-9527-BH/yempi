"""Tiny MCP 服务器，示例之间共享，因此每个示例都聚焦于一个核心思想。

这里没有任何内容是针对 LangChain 的——这些是普通的 FastMCP 服务器。 `run_*`
函数存在是因为通过 stdio 或 HTTP 连接的服务器必须作为独立进程启动，而该入口点不能是
lambda。
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from fastmcp import Context, FastMCP
from fastmcp.server.auth.providers.jwt import JWTVerifier, RSAKeyPair
from fastmcp.server.dependencies import get_access_token
from mcp.types import (
    ElicitRequest,
    ElicitRequestFormParams,
    ElicitResult,
    InputRequiredResult,
    TextContent,
    ToolAnnotations,
)
from pydantic import SecretStr


def weather_server() -> FastMCP[None]:
    """一个仅包含一个始终成功的工具的服务器。"""
    mcp: FastMCP[None] = FastMCP("weather")

    @mcp.tool
    def get_forecast(city: str) -> str:
        """报告某个城市的天气预报。"""
        return f"{city}: 18C and clear."

    return mcp


def calculator_server() -> FastMCP[None]:
    """一个其工具对无法处理的输入报告失败的服务器。"""
    mcp: FastMCP[None] = FastMCP("calculator")

    @mcp.tool
    def divide(numerator: float, denominator: float) -> str:
        """将两个数字相除。"""
        if denominator == 0:
                        # 在工具内部抛出异常会变成 MCP 错误结果（`isError=True`）
                        # 而不是传输失败，这使得智能体能够看到错误并重试。参见 `tool_errors.py`。
                        # 参见 `tool_errors.py`。
            msg = "Cannot divide by zero. Choose a non-zero denominator."
            raise ValueError(msg)
        return str(numerator / denominator)

    return mcp


def files_server() -> FastMCP[None]:
    """一个包含只读工具和破坏性工具的服务器。

    `delete_file` 设置了 `destructiveHint=True`；适配器将其作为
    `metadata["mcp"]["tool"]["annotations"]["destructive_hint"]` 表面化，
    `destructive_interrupt.py` 使用它来在人类确认后才允许调用该工具。
    """
    mcp: FastMCP[None] = FastMCP("files")

    @mcp.tool
    def list_files() -> list[str]:
        """列出工作区中的文件。"""
        return ["report.md", "notes.txt"]

    @mcp.tool(annotations=ToolAnnotations(destructiveHint=True))
    def delete_file(path: str) -> str:
        """从工作区删除一个文件。"""
        return f"Deleted {path}."

    return mcp


def booking_server() -> FastMCP[None]:
    """一个其工具在未获得人类回答前无法完成的服务器。

    Uses the guard pattern: the tool checks whether the answer it needs has
    arrived and, if not, returns an `InputRequiredResult` describing the
    question instead of doing any work. Returning early is what makes the call
    safe to replay when the run resumes.
    """
    mcp: FastMCP[None] = FastMCP("booking")

    @mcp.tool
    async def book_table(party_size: int, ctx: Context) -> list[TextContent] | InputRequiredResult:
        """预订餐厅座位。会向用户询问预订日期。"""
        answers = ctx.input_responses
        if not answers or "date" not in answers:
            return InputRequiredResult(
                input_requests={
                    "date": ElicitRequest(
                        method="elicitation/create",
                        params=ElicitRequestFormParams(
                            mode="form",
                            message="What date would you like to book?",
                            requested_schema={
                                "type": "object",
                                "properties": {"date": {"type": "string", "format": "date"}},
                                "required": ["date"],
                            },
                        ),
                    )
                },
                request_state="awaiting-date",
            )

        answer = answers["date"]
        if not isinstance(answer, ElicitResult) or answer.action != "accept" or not answer.content:
            return [TextContent(type="text", text="No date given, so nothing was booked.")]
        date = answer.content["date"]
        return [TextContent(type="text", text=f"Booked a table for {party_size} on {date}.")]

    return mcp


def run_weather_stdio() -> None:
    """通过 stdio 与天气服务器通信的入口点。"""
    weather_server().run()


def run_calculator_stdio() -> None:
    """通过 stdio 与计算器服务器通信的入口点。"""
    calculator_server().run()


def run_weather_http(host: str, port: int) -> None:
    """通过 HTTP 服务的天气服务器的入口点。"""
    weather_server().run(
        transport="http", host=host, port=port, show_banner=False, log_level="warning"
    )


def run_calculator_http(host: str, port: int) -> None:
    """通过 HTTP 服务的计算器服务器的入口点。"""
    calculator_server().run(
        transport="http", host=host, port=port, show_banner=False, log_level="warning"
    )


ISSUER = "https://demo.issuer"
AUDIENCE = "mcp-fleet"


def _load_or_generate_keys() -> RSAKeyPair:
    """返回一个跨进程共享的密钥对，或每个进程一个新密钥对。

    单个进程（如 `run_server_in_process`
    示例）可以生成自己的密钥对，并将公钥部分传递给服务器。多进程演示则不行： 生成令牌的
    `langgraph dev` 工作进程和验证令牌的服务器进程是不同的解释器。通过
    `MCP_DEMO_KEYFILE` 指向同一个 PEM
    文件，两者共享同一密钥对；若不设置，则每个进程保留自己的密钥对，与之前完全一致。
    """
    keyfile = os.environ.get("MCP_DEMO_KEYFILE")
    if keyfile:
        data = json.loads(Path(keyfile).read_text())
        return RSAKeyPair(private_key=SecretStr(data["private_key"]), public_key=data["public_key"])
    return RSAKeyPair.generate()


_KEYS = _load_or_generate_keys()
"""Stands in for an identity provider, so the examples can mint real tokens."""

PUBLIC_KEY = _KEYS.public_key
"""Pass this to `run_guarded_server`; see why in its docstring."""


def token_for(user: str) -> str:
    """生成标识 `user` 的访问令牌。

    The stand-in for whatever a deployment already has: an OAuth gateway that
    exchanges for a per-user token, or a provider from `fastmcp.client.auth` —
    `OAuth(mcp_url=..., token_storage=<per-user store>)` runs the full
    authorization-code flow per identity, which `auth_oauth.py` demonstrates.
    """
    return _KEYS.create_token(subject=user, issuer=ISSUER, audience=AUDIENCE)


def run_guarded_server(host: str, port: int, name: str, public_key: str) -> None:
    """提供一个受令牌保护的服务器，其唯一工具报告调用者身份。

    The key is a parameter rather than read from this module: the server runs
    in its own process, which re-imports this file and would otherwise mint a
    second, unrelated key pair.
    """
    mcp: FastMCP[None] = FastMCP(
        name,
        auth=JWTVerifier(public_key=public_key, issuer=ISSUER, audience=AUDIENCE),
                # 启用客户端响应缓存（SEP-2549）：服务器为每个可缓存的结果打上此 TTL 和
        # `private` 范围标签，因此持有 `CacheConfig` 的客户端可从自身缓存中提供重复的
        # `tools/list` 响应。
                # 每个可缓存的结果打上此 TTL 和 `private` 范围标签，因此持有 `CacheConfig`
        # 的客户端可从自身缓存中提供重复的 `tools/list` 响应。
                # 每个可缓存的结果打上此 TTL 和 `private` 范围标签，因此持有 `CacheConfig`
        # 的客户端可从自身缓存中提供重复的 `tools/list` 响应。
                # 存储而非通过线路。如果没有这些提示，服务器会发送
                # `ttlMs: 0`，且不会进行缓存，因此无法依赖任意第三方
                # 服务器。
        cache_ttl=60,
        cache_scope="private",
    )

    @mcp.tool
    def whoami() -> str:
        """报告服务器从访问令牌推导出的身份信息。"""
        token = get_access_token()
        return "unauthenticated" if token is None else str(token.claims.get("sub"))

    mcp.run(transport="http", host=host, port=port, show_banner=False, log_level="warning")

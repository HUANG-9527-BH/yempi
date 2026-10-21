"""Claude bash 工具的 Anthropic 特定中间件。"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from langchain.agents.middleware.shell_tool import ShellToolMiddleware
from langchain.agents.middleware.types import (
    ModelRequest,
    ModelResponse,
)

# Anthropic 的工具类型常量
BASH_TOOL_TYPE = "bash_20250124"
BASH_TOOL_NAME = "bash"


class ClaudeBashToolMiddleware(ShellToolMiddleware):
    """将 Anthropic 原生 bash 工具暴露给模型的中间件。"""

    def __init__(
        self,
        workspace_root: str | None = None,
        *,
        startup_commands: tuple[str, ...] | list[str] | str | None = None,
        shutdown_commands: tuple[str, ...] | list[str] | str | None = None,
        execution_policy: Any | None = None,
        redaction_rules: tuple[Any, ...] | list[Any] | None = None,
        tool_description: str | None = None,
        env: dict[str, Any] | None = None,
    ) -> None:
        """初始化 Claude 原生 bash 工具的中间件。

        Args:
            workspace_root: Shell 会话的基础目录。

                如果省略，则创建一个临时目录。
            startup_commands: 会话启动后可选执行的命令。
            shutdown_commands: 会话关闭前可选执行的命令。
            execution_policy: 控制超时和限制的执行策略。
            redaction_rules: 可选的脱敏规则，用于净化输出。
            tool_description: 可选的工具描述覆盖。
            env: 可选的 shell 会话环境变量。
        """
        super().__init__(
            workspace_root=workspace_root,
            startup_commands=startup_commands,
            shutdown_commands=shutdown_commands,
            execution_policy=execution_policy,
            redaction_rules=redaction_rules,
            tool_description=tool_description,
            tool_name=BASH_TOOL_NAME,
            shell_command=("/bin/bash",),
            env=env,
        )
                # 父类现在通过 tool_name 参数以名称 "bash" 创建工具

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        """用 Claude 的 bash 描述符替换父级的 shell 工具。"""
        filtered = [
            t for t in request.tools if getattr(t, "name", None) != BASH_TOOL_NAME
        ]
        tools = [*filtered, {"type": BASH_TOOL_TYPE, "name": BASH_TOOL_NAME}]
        return handler(request.override(tools=tools))

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        """Async: 异步：用 Claude 的 bash 描述符替换父级的 shell 工具。"""
        filtered = [
            t for t in request.tools if getattr(t, "name", None) != BASH_TOOL_NAME
        ]
        tools = [*filtered, {"type": BASH_TOOL_TYPE, "name": BASH_TOOL_NAME}]
        return await handler(request.override(tools=tools))


__all__ = ["ClaudeBashToolMiddleware"]

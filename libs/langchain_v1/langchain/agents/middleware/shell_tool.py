"""为智能体提供持久化 shell 工具的中间件。"""

from __future__ import annotations

import contextlib
import logging
import os
import queue
import signal
import subprocess
import tempfile
import threading
import time
import uuid
import weakref
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, Literal, cast, overload

from langchain_core.messages import ToolMessage
from langchain_core.runnables import run_in_executor
from langchain_core.tools.base import ToolException
from langgraph.channels.untracked_value import UntrackedValue
from pydantic import BaseModel, model_validator
from pydantic.json_schema import SkipJsonSchema
from typing_extensions import NotRequired, override

from langchain.agents.middleware._execution import (
    SHELL_TEMP_PREFIX,
    BaseExecutionPolicy,
    CodexSandboxExecutionPolicy,
    DockerExecutionPolicy,
    HostExecutionPolicy,
)
from langchain.agents.middleware._redaction import (
    PIIDetectionError,
    PIIMatch,
    RedactionRule,
    ResolvedRedactionRule,
)
from langchain.agents.middleware.types import (
    AgentMiddleware,
    AgentState,
    ContextT,
    PrivateStateAttr,
    ResponseT,
)
from langchain.tools import ToolRuntime, tool

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from langgraph.runtime import Runtime


LOGGER = logging.getLogger(__name__)
_DONE_MARKER_PREFIX = "__LC_SHELL_DONE__"

DEFAULT_TOOL_DESCRIPTION = (
    "Execute a shell command inside a persistent session. Before running a command, "
    "confirm the working directory is correct (e.g., inspect with `ls` or `pwd`) and ensure "
    "any parent directories exist. Prefer absolute paths and quote paths containing spaces, "
    'such as `cd "/path/with spaces"`. Chain multiple commands with `&&` or `;` instead of '
    "embedding newlines. Avoid unnecessary `cd` usage unless explicitly required so the "
    "session remains stable. Outputs may be truncated when they become very large, and long "
    "running commands will be terminated once their configured timeout elapses."
)
SHELL_TOOL_NAME = "shell"


def _cleanup_resources(
    session: ShellSession, tempdir: tempfile.TemporaryDirectory[str] | None, timeout: float
) -> None:
    with contextlib.suppress(Exception):
        session.stop(timeout)
    if tempdir is not None:
        with contextlib.suppress(Exception):
            tempdir.cleanup()


@dataclass
class _SessionResources:
    """用于存储每次运行时 shell 资源的容器。"""

    session: ShellSession
    tempdir: tempfile.TemporaryDirectory[str] | None
    policy: BaseExecutionPolicy
    finalizer: weakref.finalize = field(init=False, repr=False)  # type: ignore[type-arg]

    def __post_init__(self) -> None:
        self.finalizer = weakref.finalize(
            self,
            _cleanup_resources,
            self.session,
            self.tempdir,
            self.policy.termination_timeout,
        )


class ShellToolState(AgentState[ResponseT]):
    """用于跟踪 shell 会话资源的智能体状态扩展。

    类型参数:
        ResponseT: 结构化响应的类型。默认为 `Any`。
    """

    shell_session_resources: NotRequired[
        Annotated[_SessionResources | None, UntrackedValue, PrivateStateAttr]
    ]


@dataclass(frozen=True)
class CommandExecutionResult:
    """命令执行的结构化结果。"""

    output: str
    exit_code: int | None
    timed_out: bool
    truncated_by_lines: bool
    truncated_by_bytes: bool
    total_lines: int
    total_bytes: int


class ShellSession:
    """支持顺序命令执行的持久化 shell 会话。"""

    def __init__(
        self,
        workspace: Path,
        policy: BaseExecutionPolicy,
        command: tuple[str, ...],
        environment: Mapping[str, str],
    ) -> None:
        self._workspace = workspace
        self._policy = policy
        self._command = command
        self._environment = dict(environment)
        self._process: subprocess.Popen[str] | None = None
        self._stdin: Any = None
        self._queue: queue.Queue[tuple[str, str | None]] = queue.Queue()
        self._lock = threading.Lock()
        self._stdout_thread: threading.Thread | None = None
        self._stderr_thread: threading.Thread | None = None
        self._terminated = False

    def start(self) -> None:
        """启动 shell 子进程和读取线程。

        Raises:
            RuntimeError: 如果无法初始化 shell 会话的管道。
        """
        if self._process and self._process.poll() is None:
            return

        self._process = self._policy.spawn(
            workspace=self._workspace,
            env=self._environment,
            command=self._command,
        )
        if (
            self._process.stdin is None
            or self._process.stdout is None
            or self._process.stderr is None
        ):
            msg = "Failed to initialize shell session pipes."
            raise RuntimeError(msg)

        self._stdin = self._process.stdin
        self._terminated = False
        self._queue = queue.Queue()

        self._stdout_thread = threading.Thread(
            target=self._enqueue_stream,
            args=(self._process.stdout, "stdout"),
            daemon=True,
        )
        self._stderr_thread = threading.Thread(
            target=self._enqueue_stream,
            args=(self._process.stderr, "stderr"),
            daemon=True,
        )
        self._stdout_thread.start()
        self._stderr_thread.start()

    def restart(self) -> None:
        """重启 shell 进程。"""
        self.stop(self._policy.termination_timeout)
        self.start()

    def stop(self, timeout: float) -> None:
        """停止 shell 子进程。"""
        if not self._process:
            return

        if self._process.poll() is None and not self._terminated:
            try:
                self._stdin.write("exit\n")
                self._stdin.flush()
            except (BrokenPipeError, OSError):
                LOGGER.debug(
                    "Failed to write exit command; terminating shell session.",
                    exc_info=True,
                )

        try:
            if self._process.wait(timeout=timeout) is None:
                self._kill_process()
        except subprocess.TimeoutExpired:
            self._kill_process()
        finally:
            self._terminated = True
            with contextlib.suppress(Exception):
                self._stdin.close()
            self._process = None

    def execute(self, command: str, *, timeout: float) -> CommandExecutionResult:
        """在持久化 shell 中执行命令。"""
        if not self._process or self._process.poll() is not None:
            msg = "Shell session is not running."
            raise RuntimeError(msg)

        marker = f"{_DONE_MARKER_PREFIX}{uuid.uuid4().hex}"
        deadline = time.monotonic() + timeout

        with self._lock:
            self._drain_queue()
            payload = command if command.endswith("\n") else f"{command}\n"
            try:
                self._stdin.write(payload)
                self._stdin.write(f"printf '{marker} %s\\n' $?\n")
                self._stdin.flush()
            except (BrokenPipeError, OSError):
                                # The shell exited before we could write the marker command.
                                # This happens when commands like 'exit 1' terminate the shell.
                return self._collect_output_after_exit(deadline)

            return self._collect_output(marker, deadline, timeout)

    def _collect_output(
        self,
        marker: str,
        deadline: float,
        timeout: float,
    ) -> CommandExecutionResult:
        collected: list[str] = []
        total_lines = 0
        total_bytes = 0
        truncated_by_lines = False
        truncated_by_bytes = False
        exit_code: int | None = None
        timed_out = False

        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                timed_out = True
                break
            try:
                source, data = self._queue.get(timeout=remaining)
            except queue.Empty:
                timed_out = True
                break

            if data is None:
                continue

            if source == "stdout" and data.startswith(marker):
                _, _, status = data.partition(" ")
                exit_code = self._safe_int(status.strip())
                                # Drain any remaining stderr that may have arrived concurrently.
                                # The stderr reader thread runs independently, so output might
                                # still be in flight when the stdout marker arrives.
                self._drain_remaining_stderr(collected, deadline)
                break

            total_lines += 1
            encoded = data.encode("utf-8", "replace")
            total_bytes += len(encoded)

            if total_lines > self._policy.max_output_lines:
                truncated_by_lines = True
                continue

            if (
                self._policy.max_output_bytes is not None
                and total_bytes > self._policy.max_output_bytes
            ):
                truncated_by_bytes = True
                continue

            if source == "stderr":
                stripped = data.rstrip("\n")
                collected.append(f"[stderr] {stripped}")
                if data.endswith("\n"):
                    collected.append("\n")
            else:
                collected.append(data)

        if timed_out:
            LOGGER.warning(
                "Command timed out after %.2f seconds; restarting shell session.",
                timeout,
            )
            self.restart()
            return CommandExecutionResult(
                output="",
                exit_code=None,
                timed_out=True,
                truncated_by_lines=truncated_by_lines,
                truncated_by_bytes=truncated_by_bytes,
                total_lines=total_lines,
                total_bytes=total_bytes,
            )

        output = "".join(collected)
        return CommandExecutionResult(
            output=output,
            exit_code=exit_code,
            timed_out=False,
            truncated_by_lines=truncated_by_lines,
            truncated_by_bytes=truncated_by_bytes,
            total_lines=total_lines,
            total_bytes=total_bytes,
        )

    def _collect_output_after_exit(self, deadline: float) -> CommandExecutionResult:
        """收集 shell 异常退出后输出的内容。

        Called when a `BrokenPipeError` occurs while writing to stdin, indicating the
        shell process terminated (e.g., due to an 'exit' command).

        Args:
            deadline: 收集必须完成的绝对时间。

        Returns:
            包含收集到的输出和进程退出码的 `CommandExecutionResult`。
        """
        collected: list[str] = []
        total_lines = 0
        total_bytes = 0
        truncated_by_lines = False
        truncated_by_bytes = False

                # Give reader threads a brief moment to enqueue any remaining output.
        drain_timeout = 0.1
        drain_deadline = min(time.monotonic() + drain_timeout, deadline)

        while True:
            remaining = drain_deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                source, data = self._queue.get(timeout=remaining)
            except queue.Empty:
                break

            if data is None:
                                # EOF marker from a reader thread; continue draining.
                continue

            total_lines += 1
            encoded = data.encode("utf-8", "replace")
            total_bytes += len(encoded)

            if total_lines > self._policy.max_output_lines:
                truncated_by_lines = True
                continue

            if (
                self._policy.max_output_bytes is not None
                and total_bytes > self._policy.max_output_bytes
            ):
                truncated_by_bytes = True
                continue

            if source == "stderr":
                stripped = data.rstrip("\n")
                collected.append(f"[stderr] {stripped}")
                if data.endswith("\n"):
                    collected.append("\n")
            else:
                collected.append(data)

                # Get exit code from the terminated process.
        exit_code: int | None = None
        if self._process:
            exit_code = self._process.poll()

        output = "".join(collected)
        return CommandExecutionResult(
            output=output,
            exit_code=exit_code,
            timed_out=False,
            truncated_by_lines=truncated_by_lines,
            truncated_by_bytes=truncated_by_bytes,
            total_lines=total_lines,
            total_bytes=total_bytes,
        )

    def _kill_process(self) -> None:
        if not self._process:
            return

        if hasattr(os, "killpg"):
            try:
                child_pgid = os.getpgid(self._process.pid)
                                # 仅在子进程拥有独立的进程组时发送组终止信号。
                                # 如果子进程与我们共享进程组，killpg 也会终止调用者，
                                # 因此需继续执行下方的直接终止操作。该直接终止仅回收
                                # 直接子进程，因此其创建的任何后代可能成为孤儿进程。
                                # 这适用于 HostExecutionPolicy(create_process_group=False)，该
                                # 策略是唯一在调用者进程组中运行 shell 的策略。
                if child_pgid != os.getpgrp():
                    os.killpg(child_pgid, signal.SIGKILL)
                    return
            except ProcessLookupError:
                                # 进程已不存在；无需再进行终止操作。
                return
            except OSError:
                                # 例如，在查询或向组发送信号时出现 EPERM 错误。不要静默泄漏
                                # 子进程；应继续执行直接终止操作。
                LOGGER.warning(
                    "Group kill failed; falling back to direct kill.",
                    exc_info=True,
                )

        try:
            self._process.kill()
        except ProcessLookupError:
                        # 在上述检查与本次终止之间，进程已退出；无需执行任何操作。
            pass
        except OSError:
                        # 回退终止可能遇到与引导我们至此相同的条件（例如
            # EPERM）。应记录日志而非让其逃逸出会话关闭路径。
                        # 创建一个可直接执行的完整工具（无需拦截）
            LOGGER.warning(
                "Direct kill failed.",
                exc_info=True,
            )

    def _enqueue_stream(self, stream: Any, label: str) -> None:
        for line in iter(stream.readline, ""):
            self._queue.put((label, line))
        self._queue.put((label, None))

    def _drain_queue(self) -> None:
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break

    def _drain_remaining_stderr(
        self, collected: list[str], deadline: float, drain_timeout: float = 0.05
    ) -> None:
        """清空与结束标记同时到达的 stderr 输出。

        stdout 和 stderr 的读取线程独立运行。当命令在退出前向 stderr 写入时，stderr
        输出可能仍在传输过程中，而结束标记已到达
        stdout。此方法短暂轮询队列以捕获此类输出。

        Args:
            collected: 要追加收集到的 stderr 行的列表。
            deadline: 原始命令的截止时间（用作上限）。
            drain_timeout: 等待额外 stderr 输出的最大时间。
        """
        drain_deadline = min(time.monotonic() + drain_timeout, deadline)
        while True:
            remaining = drain_deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                source, data = self._queue.get(timeout=remaining)
            except queue.Empty:
                break
            if data is None or source != "stderr":
                continue
            stripped = data.rstrip("\n")
            collected.append(f"[stderr] {stripped}")
            if data.endswith("\n"):
                collected.append("\n")

    @staticmethod
    def _safe_int(value: str) -> int | None:
        with contextlib.suppress(ValueError):
            return int(value)
        return None


class _ShellToolInput(BaseModel):
    """持久化 shell 工具的输入模式。"""

    command: str | None = None
    """要执行的 shell 命令。"""

    restart: bool | None = None
    """是否重启 shell 会话。"""

    runtime: Annotated[Any, SkipJsonSchema()] = None
    """shell 工具的运行时。

    当前作为变通方案，因为 args_schema 与注入的 ToolRuntime 不兼容。
    """

    @model_validator(mode="after")
    def validate_payload(self) -> _ShellToolInput:
        if self.command is None and not self.restart:
            msg = "Shell tool requires either 'command' or 'restart'."
            raise ValueError(msg)
        if self.command is not None and self.restart:
            msg = "Specify only one of 'command' or 'restart'."
            raise ValueError(msg)
        return self


class ShellToolMiddleware(AgentMiddleware[ShellToolState[ResponseT], ContextT, ResponseT]):
    """注册持久化 shell 工具给智能体的中间件。

    该中间件暴露一个长期存在的 shell 会话。使用执行策略来匹配您的部署安全策略：

    * `HostExecutionPolicy` – full host access; best for trusted environments where the
        agent already runs inside a container or VM that provides isolation.
    * `CodexSandboxExecutionPolicy` – reuses the Codex CLI sandbox for additional
        syscall/filesystem restrictions when the CLI is available.
    * `DockerExecutionPolicy` – launches a separate Docker container for each agent run,
        providing harder isolation, optional read-only root filesystems, and user
        remapping.

    若未提供策略，默认使用 `HostExecutionPolicy`。
    """

    state_schema = ShellToolState  # type: ignore[assignment]

    def __init__(
        self,
        workspace_root: str | Path | None = None,
        *,
        startup_commands: tuple[str, ...] | list[str] | str | None = None,
        shutdown_commands: tuple[str, ...] | list[str] | str | None = None,
        execution_policy: BaseExecutionPolicy | None = None,
        redaction_rules: tuple[RedactionRule, ...] | list[RedactionRule] | None = None,
        tool_description: str | None = None,
        tool_name: str = SHELL_TOOL_NAME,
        shell_command: Sequence[str] | str | None = None,
        env: Mapping[str, Any] | None = None,
    ) -> None:
        """初始化 `ShellToolMiddleware` 的实例。

        Args:
            workspace_root: shell 会话的基目录。

                如果未指定，智能体启动时会创建一个临时目录，并在结束时删除。
                startup_commands: 会话启动后按顺序执行的可选命令。 shutdown_commands:
                会话关闭前执行的可选命令。 execution_policy:
                控制超时、输出限制和资源配置的执行策略。

                Defaults to `HostExecutionPolicy` for native execution.
            redaction_rules: Optional redaction rules to sanitize command output before
                returning it to the model.

                !!! warning 脱敏规则在执行后应用，使用 `HostExecutionPolicy`
                时无法防止秘密或敏感数据的外泄。

            tool_description: Optional override for the registered shell tool
                description.
            tool_name: Name for the registered shell tool.

                默认为 `"shell"`。 shell_command: 用于启动持久会话的可选 shell
                可执行文件（字符串）或参数序列。

                默认为实现定义的 bash 命令。
            env: 传递给 shell 会话的可选环境变量。

                值在命令执行前会被强制转换为字符串。如果未指定，会话将继承父进程的环境。
        """
        super().__init__()
        self._workspace_root = Path(workspace_root) if workspace_root else None
        self._tool_name = tool_name
        self._shell_command = self._normalize_shell_command(shell_command)
        self._environment = self._normalize_env(env)
        if execution_policy is not None:
            self._execution_policy = execution_policy
        else:
            self._execution_policy = HostExecutionPolicy()
        rules = redaction_rules or ()
        self._redaction_rules: tuple[ResolvedRedactionRule, ...] = tuple(
            rule.resolve() for rule in rules
        )
        self._startup_commands = self._normalize_commands(startup_commands)
        self._shutdown_commands = self._normalize_commands(shutdown_commands)

                # 资源从未创建，无需清理
        description = tool_description or DEFAULT_TOOL_DESCRIPTION

        @tool(self._tool_name, args_schema=_ShellToolInput, description=description)
        def shell_tool(
            *,
            runtime: ToolRuntime[None, ShellToolState],
            command: str | None = None,
            restart: bool = False,
        ) -> ToolMessage | str:
            resources = self._get_or_create_resources(runtime.state)
            return self._run_shell_tool(
                resources,
                {"command": command, "restart": restart},
                tool_call_id=runtime.tool_call_id,
            )

        self._shell_tool = shell_tool
        self.tools = [self._shell_tool]

    @staticmethod
    def _normalize_commands(
        commands: tuple[str, ...] | list[str] | str | None,
    ) -> tuple[str, ...]:
        if commands is None:
            return ()
        if isinstance(commands, str):
            return (commands,)
        return tuple(commands)

    @staticmethod
    def _normalize_shell_command(
        shell_command: Sequence[str] | str | None,
    ) -> tuple[str, ...]:
        if shell_command is None:
            return ("/bin/bash",)
        normalized = (shell_command,) if isinstance(shell_command, str) else tuple(shell_command)
        if not normalized:
            msg = "Shell command must contain at least one argument."
            raise ValueError(msg)
        return normalized

    @staticmethod
    def _normalize_env(env: Mapping[str, Any] | None) -> dict[str, str] | None:
        if env is None:
            return None
        normalized: dict[str, str] = {}
        for key, value in env.items():
            if not isinstance(key, str):
                msg = "Environment variable names must be strings."  # type: ignore[unreachable]
                raise TypeError(msg)
            normalized[key] = str(value)
        return normalized

    @override
    def before_agent(
        self, state: ShellToolState[ResponseT], runtime: Runtime[ContextT]
    ) -> dict[str, Any] | None:
        """启动 shell 会话并运行启动命令。

        Args:
            state: 当前智能体状态。
            runtime: 运行时上下文。

        Returns:
            存储在智能体状态中的 shell 会话资源。
        """
        resources = self._get_or_create_resources(state)
        return {"shell_session_resources": resources}

    async def abefore_agent(
        self, state: ShellToolState[ResponseT], runtime: Runtime[ContextT]
    ) -> dict[str, Any] | None:
        """异步启动 shell 会话并运行启动命令。

        Args:
            state: 当前智能体状态。
            runtime: 运行时上下文。

        Returns:
            存储在智能体状态中的 shell 会话资源。
        """
        return await run_in_executor(None, self.before_agent, state, runtime)

    @override
    def after_agent(self, state: ShellToolState[ResponseT], runtime: Runtime[ContextT]) -> None:
        """运行关闭命令并在智能体完成时释放资源。"""
        resources = state.get("shell_session_resources")
        if not isinstance(resources, _SessionResources):
                        # 强制转换以使状态字典类对象可被修改
            return
        try:
            self._run_shutdown_commands(resources.session)
        finally:
            resources.finalizer()

    async def aafter_agent(
        self, state: ShellToolState[ResponseT], runtime: Runtime[ContextT]
    ) -> None:
        """异步运行关闭命令并在智能体完成时释放资源。"""
        return self.after_agent(state, runtime)

    def _get_or_create_resources(self, state: ShellToolState[ResponseT]) -> _SessionResources:
        """从状态获取现有资源，若不存在则创建新资源。

        This method enables resumability by checking if resources already exist in the state
        (e.g., after an interrupt), and only creating new resources if they're not present.

        Args:
            state: 可能包含 shell 会话资源的智能体状态。

        Returns:
            从状态获取的会话资源，或新创建的资源。
        """
        resources = state.get("shell_session_resources")
        if isinstance(resources, _SessionResources):
            return resources

        new_resources = self._create_resources()
                # 强制转换以使状态字典类似可变
        cast("dict[str, Any]", state)["shell_session_resources"] = new_resources
        return new_resources

    def _create_resources(self) -> _SessionResources:
        workspace = self._workspace_root
        tempdir: tempfile.TemporaryDirectory[str] | None = None
        if workspace is None:
            tempdir = tempfile.TemporaryDirectory(prefix=SHELL_TEMP_PREFIX)
            workspace_path = Path(tempdir.name)
        else:
            workspace_path = workspace
            workspace_path.mkdir(parents=True, exist_ok=True)

        session = ShellSession(
            workspace_path,
            self._execution_policy,
            self._shell_command,
            self._environment or {},
        )
        try:
            session.start()
            LOGGER.info("Started shell session in %s", workspace_path)
            self._run_startup_commands(session)
        except BaseException:
            LOGGER.exception("Starting shell session failed; cleaning up resources.")
            session.stop(self._execution_policy.termination_timeout)
            if tempdir is not None:
                tempdir.cleanup()
            raise

        return _SessionResources(session=session, tempdir=tempdir, policy=self._execution_policy)

    def _run_startup_commands(self, session: ShellSession) -> None:
        if not self._startup_commands:
            return
        for command in self._startup_commands:
            result = session.execute(command, timeout=self._execution_policy.startup_timeout)
            if result.timed_out or (result.exit_code not in {0, None}):
                msg = f"Startup command '{command}' failed with exit code {result.exit_code}"
                raise RuntimeError(msg)

    def _run_shutdown_commands(self, session: ShellSession) -> None:
        if not self._shutdown_commands:
            return
        for command in self._shutdown_commands:
            try:
                result = session.execute(command, timeout=self._execution_policy.command_timeout)
                if result.timed_out:
                    LOGGER.warning("Shutdown command '%s' timed out.", command)
                elif result.exit_code not in {0, None}:
                    LOGGER.warning(
                        "Shutdown command '%s' exited with %s.", command, result.exit_code
                    )
            except (RuntimeError, ToolException, OSError) as exc:
                LOGGER.warning(
                    "Failed to run shutdown command '%s': %s", command, exc, exc_info=True
                )

    def _apply_redactions(self, content: str) -> tuple[str, dict[str, list[PIIMatch]]]:
        """对命令输出应用配置的脱敏规则。"""
        matches_by_type: dict[str, list[PIIMatch]] = {}
        updated = content
        for rule in self._redaction_rules:
            updated, matches = rule.apply(updated)
            if matches:
                matches_by_type.setdefault(rule.pii_type, []).extend(matches)
        return updated, matches_by_type

    @overload
    def _run_shell_tool(
        self,
        resources: _SessionResources,
        payload: dict[str, Any],
        *,
        tool_call_id: str,
    ) -> ToolMessage: ...

    @overload
    def _run_shell_tool(
        self,
        resources: _SessionResources,
        payload: dict[str, Any],
        *,
        tool_call_id: None,
    ) -> str: ...

    def _run_shell_tool(
        self,
        resources: _SessionResources,
        payload: dict[str, Any],
        *,
        tool_call_id: str | None,
    ) -> ToolMessage | str:
        session = resources.session

        if payload.get("restart"):
            LOGGER.info("Restarting shell session on request.")
            try:
                session.restart()
                self._run_startup_commands(session)
            except BaseException as err:
                LOGGER.exception("Restarting shell session failed; session remains unavailable.")
                msg = "Failed to restart shell session."
                raise ToolException(msg) from err
            message = "Shell session restarted."
            return self._format_tool_message(message, tool_call_id, status="success")

        command = payload.get("command")
        if not command or not isinstance(command, str):
            msg = "Shell tool expects a 'command' string when restart is not requested."
            raise ToolException(msg)

        LOGGER.info("Executing shell command: %s", command)
        result = session.execute(command, timeout=self._execution_policy.command_timeout)

        if result.timed_out:
            timeout_seconds = self._execution_policy.command_timeout
            message = f"Error: Command timed out after {timeout_seconds:.1f} seconds."
            return self._format_tool_message(
                message,
                tool_call_id,
                status="error",
                artifact={
                    "timed_out": True,
                    "exit_code": None,
                },
            )

        try:
            sanitized_output, matches = self._apply_redactions(result.output)
        except PIIDetectionError as error:
            LOGGER.warning("Blocking command output due to detected %s.", error.pii_type)
            message = f"Output blocked: detected {error.pii_type}."
            return self._format_tool_message(
                message,
                tool_call_id,
                status="error",
                artifact={
                    "timed_out": False,
                    "exit_code": result.exit_code,
                    "matches": {error.pii_type: error.matches},
                },
            )

        sanitized_output = sanitized_output or "<no output>"
        if result.truncated_by_lines:
            sanitized_output = (
                f"{sanitized_output.rstrip()}\n\n"
                f"... Output truncated at {self._execution_policy.max_output_lines} lines "
                f"(observed {result.total_lines})."
            )
        if result.truncated_by_bytes and self._execution_policy.max_output_bytes is not None:
            sanitized_output = (
                f"{sanitized_output.rstrip()}\n\n"
                f"... Output truncated at {self._execution_policy.max_output_bytes} bytes "
                f"(observed {result.total_bytes})."
            )

        if result.exit_code not in {0, None}:
            sanitized_output = f"{sanitized_output.rstrip()}\n\nExit code: {result.exit_code}"
            final_status: Literal["success", "error"] = "error"
        else:
            final_status = "success"

        artifact = {
            "timed_out": False,
            "exit_code": result.exit_code,
            "truncated_by_lines": result.truncated_by_lines,
            "truncated_by_bytes": result.truncated_by_bytes,
            "total_lines": result.total_lines,
            "total_bytes": result.total_bytes,
            "redaction_matches": matches,
        }

        return self._format_tool_message(
            sanitized_output,
            tool_call_id,
            status=final_status,
            artifact=artifact,
        )

    @overload
    def _format_tool_message(
        self,
        content: str,
        tool_call_id: str,
        *,
        status: Literal["success", "error"],
        artifact: dict[str, Any] | None = None,
    ) -> ToolMessage: ...

    @overload
    def _format_tool_message(
        self,
        content: str,
        tool_call_id: None,
        *,
        status: Literal["success", "error"],
        artifact: dict[str, Any] | None = None,
    ) -> str: ...

    def _format_tool_message(
        self,
        content: str,
        tool_call_id: str | None,
        *,
        status: Literal["success", "error"],
        artifact: dict[str, Any] | None = None,
    ) -> ToolMessage | str:
        artifact = artifact or {}
        if tool_call_id is None:
            return content
        return ToolMessage(
            content=content,
            tool_call_id=tool_call_id,
            name=self._tool_name,
            status=status,
            artifact=artifact,
        )


__all__ = [
    "CodexSandboxExecutionPolicy",
    "DockerExecutionPolicy",
    "HostExecutionPolicy",
    "RedactionRule",
    "ShellToolMiddleware",
]

"""Anthropic 文本编辑器和记忆工具中间件。

此模块使用无模式工具定义和工具调用拦截，提供 Anthropic 文本编辑器和
记忆工具的客户端实现。
"""

from __future__ import annotations

import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, cast

from langchain.agents.middleware.types import (
    AgentMiddleware,
    AgentState,
    ModelRequest,
    ModelResponse,
    _ModelRequestOverrides,
)
from langchain.tools import ToolRuntime, tool
from langchain_core.messages import SystemMessage, ToolMessage
from langgraph.types import Command
from typing_extensions import NotRequired, TypedDict

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Sequence


# 工具类型常量
TEXT_EDITOR_TOOL_TYPE = "text_editor_20250728"
TEXT_EDITOR_TOOL_NAME = "str_replace_based_edit_tool"
MEMORY_TOOL_TYPE = "memory_20250818"
MEMORY_TOOL_NAME = "memory"

MEMORY_SYSTEM_PROMPT = """IMPORTANT: ALWAYS VIEW YOUR MEMORY DIRECTORY BEFORE \
DOING ANYTHING ELSE.
MEMORY PROTOCOL:
1. Use the `view` command of your `memory` tool to check for earlier progress.
2. ... (work on the task) ...
   - As you make progress, record status / progress / thoughts etc in your memory.
ASSUME INTERRUPTION: Your context window might be reset at any moment, so you risk \
losing any progress that is not recorded in your memory directory."""


class FileData(TypedDict):
    """用于存储文件内容的数据结构。"""

    content: list[str]
    """文件的行。"""

    created_at: str
    """文件创建的 ISO 8601 时间戳。"""

    modified_at: str
    """文件最后修改的 ISO 8601 时间戳。"""


def files_reducer(
    left: dict[str, FileData] | None, right: dict[str, FileData | None]
) -> dict[str, FileData]:
    """自定义归约函数，用于合并文件更新。

    Args:
        left: 现有文件字典。
        right: 要合并的新文件字典（`None` 值表示删除文件）。

    Returns:
        合并后的 `dict`，right 对应键覆盖 left 中的同名键。
    """
    if left is None:
                # 初始化时过滤掉 None 值
        return {k: v for k, v in right.items() if v is not None}

        # 合并，同时过滤掉 None 值（删除项）
    result = {**left}
    for k, v in right.items():
        if v is None:
            result.pop(k, None)
        else:
            result[k] = v
    return result


class AnthropicToolsState(AgentState):
    """Anthropic 文本编辑器和记忆工具的状态模式。"""

    text_editor_files: NotRequired[Annotated[dict[str, FileData], files_reducer]]
    """文本编辑器工具的虚拟文件系统。"""

    memory_files: NotRequired[Annotated[dict[str, FileData], files_reducer]]
    """记忆工具的虚拟文件系统。"""


def _is_within_allowed_prefix(normalized: str, prefixes: Sequence[str]) -> bool:
    """检查规范化路径是否位于允许的前缀目录内。

    Uses a segment-boundary comparison rather than a raw string prefix test so
    that sibling directories sharing a textual prefix cannot escape the allowed
    directory. For example, with prefix `/memories` the path `/memories2/evil.txt`
    is rejected because it is not the prefix itself nor a descendant of it.

    Args:
        normalized: 一个规范化、正斜杠分隔、绝对风格的路径。
        prefixes: 用于比较的允许路径前缀列表。

    Returns:
        `True` if `normalized` exactly equals one of the prefix directories or is
        contained within one of them, `False` otherwise.
    """
    for prefix in prefixes:
                # 以与路径相同的规范化方式归一化前缀，以便比较一致（去除边界处的尾部斜杠）。
                # 以与路径相同的规范化方式归一化前缀，以便比较一致（去除边界处的尾部斜杠）。
        prefix_dir = prefix.rstrip("/")
        if normalized == prefix_dir or normalized.startswith(f"{prefix_dir}/"):
            return True
    return False


def _validate_path(path: str, *, allowed_prefixes: Sequence[str] | None = None) -> str:
    """对文件路径进行验证和规范化以确保安全。

    Args:
        path: 要验证的路径。
        allowed_prefixes: 可选的允许路径前缀列表。

    Returns:
        规范化后的标准路径。

    Raises:
        ValueError: 如果路径包含遍历序列或违反前缀规则。
    """
        # 拒绝包含路径遍历尝试的路径
    if ".." in path or path.startswith("~"):
        msg = f"Path traversal not allowed: {path}"
        raise ValueError(msg)

        # 归一化路径（解析 ., // 等）
    normalized = os.path.normpath(path)

        # 转换为正斜杠以保持一致性
    normalized = normalized.replace("\\", "/")

        # 确保路径以 / 开头
    if not normalized.startswith("/"):
        normalized = f"/{normalized}"

        # 如果指定了允许的前缀，则检查路径是否符合
    if allowed_prefixes is not None and not _is_within_allowed_prefix(
        normalized, allowed_prefixes
    ):
        msg = f"Path must start with one of {allowed_prefixes}: {path}"
        raise ValueError(msg)

    return normalized


def _list_directory(files: dict[str, FileData], path: str) -> list[str]:
    """列出目录中的文件。

    Args:
        files: 文件 `dict`。
        path: 规范化的目录路径。

    Returns:
        目录中文件路径的排序列表。
    """
        # 确保路径以 / 结尾，用于目录匹配
    dir_path = path if path.endswith("/") else f"{path}/"

    matching_files = []
    for file_path in files:
        if file_path.startswith(dir_path):
                        # 从目录获取相对路径
            relative = file_path[len(dir_path) :]
                        # 仅包含直接子项（不包括子目录）
            if "/" not in relative:
                matching_files.append(file_path)

    return sorted(matching_files)


class _StateClaudeFileToolMiddleware(AgentMiddleware):
    """基于状态的文件工具中间件基类（内部使用）。"""

    state_schema = AnthropicToolsState

    def __init__(
        self,
        *,
        tool_type: str,
        tool_name: str,
        state_key: str,
        allowed_path_prefixes: Sequence[str] | None = None,
        system_prompt: str | None = None,
    ) -> None:
        """初始化。

        Args:
            tool_type: 工具类型标识符。
            tool_name: 工具名称。
            state_key: 文件存储的状态键。
            allowed_path_prefixes: 可选的允许路径前缀列表。
            system_prompt: 可选的系统提示词，用于注入。
        """
        self.tool_type = tool_type
        self.tool_name = tool_name
        self.state_key = state_key
        self.allowed_prefixes = allowed_path_prefixes
        self.system_prompt = system_prompt

                # 创建将由工具节点执行的工具
        @tool(tool_name)
        def file_tool(
            runtime: ToolRuntime[None, AnthropicToolsState],
            command: str,
            path: str,
            file_text: str | None = None,
            old_str: str | None = None,
            new_str: str | None = None,
            insert_line: int | None = None,
            new_path: str | None = None,
            view_range: list[int] | None = None,
        ) -> Command | str:
            """在虚拟文件系统上执行文件操作。

            Args:
                runtime: 提供对状态访问的工具运行时。
                command: 要执行的操作。
                path: 要操作的文件路径。
                file_text: create 命令的完整文件内容。
                old_str: str_replace 命令中要替换的字符串。
                new_str: str_replace 命令中的替换字符串。
                insert_line: insert 命令的行号（从 0 开始）。
                new_path: rename 命令的新路径。
                view_range: view 命令的行范围 `[start, end]`。

            Returns:
                用于状态更新或字符串结果的命令。
            """
                        # 构建处理方法的参数字典
                        # `old_path` 在 `_handle_rename` 中被填充，该方法会读取源路径
                        # 而非 `path` 键下的路径。
            args: dict[str, Any] = {"path": path, "old_path": path}
            if file_text is not None:
                args["file_text"] = file_text
            if old_str is not None:
                args["old_str"] = old_str
            if new_str is not None:
                args["new_str"] = new_str
            if insert_line is not None:
                args["insert_line"] = insert_line
            if new_path is not None:
                args["new_path"] = new_path
            if view_range is not None:
                args["view_range"] = view_range

                        # 根据命令路由到相应的处理函数
            try:
                if command == "view":
                    return self._handle_view(args, runtime.state, runtime.tool_call_id)
                if command == "create":
                    return self._handle_create(
                        args, runtime.state, runtime.tool_call_id
                    )
                if command == "str_replace":
                    return self._handle_str_replace(
                        args, runtime.state, runtime.tool_call_id
                    )
                if command == "insert":
                    return self._handle_insert(
                        args, runtime.state, runtime.tool_call_id
                    )
                if command == "delete":
                    return self._handle_delete(
                        args, runtime.state, runtime.tool_call_id
                    )
                if command == "rename":
                    return self._handle_rename(
                        args, runtime.state, runtime.tool_call_id
                    )
                return f"Unknown command: {command}"
            except (ValueError, FileNotFoundError) as e:
                return str(e)

        self.tools = [file_tool]

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        """注入 Anthropic 工具描述符和可选的系统提示词。"""
                # 用 Anthropic 的原生工具描述符替换我们的 BaseTool
        tools = [
            t
            for t in (request.tools or [])
            if getattr(t, "name", None) != self.tool_name
        ] + [{"type": self.tool_type, "name": self.tool_name}]

                # 如果提供了系统提示，则注入
        overrides: _ModelRequestOverrides = {"tools": tools}
        if self.system_prompt:
            if request.system_message is not None:
                new_system_content = [
                    *request.system_message.content_blocks,
                    {"type": "text", "text": f"\n\n{self.system_prompt}"},
                ]
            else:
                new_system_content = [{"type": "text", "text": self.system_prompt}]
            new_system_message = SystemMessage(
                content=cast("list[str | dict[str, str]]", new_system_content)
            )
            overrides["system_message"] = new_system_message

        return handler(request.override(**overrides))

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        """注入 Anthropic 工具描述符和可选的系统提示词。"""
                # 用 Anthropic 的原生工具描述符替换我们的 BaseTool
        tools = [
            t
            for t in (request.tools or [])
            if getattr(t, "name", None) != self.tool_name
        ] + [{"type": self.tool_type, "name": self.tool_name}]

                # 如果提供了系统提示，则注入
        overrides: _ModelRequestOverrides = {"tools": tools}
        if self.system_prompt:
            if request.system_message is not None:
                new_system_content = [
                    *request.system_message.content_blocks,
                    {"type": "text", "text": f"\n\n{self.system_prompt}"},
                ]
            else:
                new_system_content = [{"type": "text", "text": self.system_prompt}]
            new_system_message = SystemMessage(
                content=cast("list[str | dict[str, str]]", new_system_content)
            )
            overrides["system_message"] = new_system_message

        return await handler(request.override(**overrides))

    def _handle_view(
        self, args: dict, state: AnthropicToolsState, tool_call_id: str | None
    ) -> Command:
        """处理 view 命令。"""
        path = args["path"]
        normalized_path = _validate_path(path, allowed_prefixes=self.allowed_prefixes)

        files = cast("dict[str, Any]", state.get(self.state_key, {}))
        file_data = files.get(normalized_path)

        if file_data is None:
                        # 尝试列出目录内容
            matching = _list_directory(files, normalized_path)

            if matching:
                content = "\n".join(matching)
                return Command(
                    update={
                        "messages": [
                            ToolMessage(
                                content=content,
                                tool_call_id=tool_call_id,
                                name=self.tool_name,
                            )
                        ]
                    }
                )

            msg = f"File not found: {path}"
            raise FileNotFoundError(msg)

                # 带行号格式化文件内容
        lines_content = file_data["content"]
        formatted_lines = [f"{i + 1}|{line}" for i, line in enumerate(lines_content)]
        content = "\n".join(formatted_lines)

        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content=content,
                        tool_call_id=tool_call_id,
                        name=self.tool_name,
                    )
                ]
            }
        )

    def _handle_create(
        self, args: dict, state: AnthropicToolsState, tool_call_id: str | None
    ) -> Command:
        """处理 create 命令。"""
        path = args["path"]
        file_text = args["file_text"]

        normalized_path = _validate_path(path, allowed_prefixes=self.allowed_prefixes)

                # 获取现有文件
        files = cast("dict[str, Any]", state.get(self.state_key, {}))
        existing = files.get(normalized_path)

                # 创建文件数据
        now = datetime.now(timezone.utc).isoformat()
        created_at = existing["created_at"] if existing else now

        content_lines = file_text.split("\n")

        return Command(
            update={
                self.state_key: {
                    normalized_path: {
                        "content": content_lines,
                        "created_at": created_at,
                        "modified_at": now,
                    }
                },
                "messages": [
                    ToolMessage(
                        content=f"File created: {path}",
                        tool_call_id=tool_call_id,
                        name=self.tool_name,
                    )
                ],
            }
        )

    def _handle_str_replace(
        self, args: dict, state: AnthropicToolsState, tool_call_id: str | None
    ) -> Command:
        """处理 str_replace 命令。"""
        path = args["path"]
        old_str = args["old_str"]
        new_str = args.get("new_str", "")

        normalized_path = _validate_path(path, allowed_prefixes=self.allowed_prefixes)

                # 读取文件
        files = cast("dict[str, Any]", state.get(self.state_key, {}))
        file_data = files.get(normalized_path)
        if file_data is None:
            msg = f"File not found: {path}"
            raise FileNotFoundError(msg)

        lines_content = file_data["content"]
        content = "\n".join(lines_content)

                # 替换字符串
        if old_str not in content:
            msg = f"String not found in file: {old_str}"
            raise ValueError(msg)

        new_content = content.replace(old_str, new_str, 1)
        new_lines = new_content.split("\n")

                # 更新文件
        now = datetime.now(timezone.utc).isoformat()

        return Command(
            update={
                self.state_key: {
                    normalized_path: {
                        "content": new_lines,
                        "created_at": file_data["created_at"],
                        "modified_at": now,
                    }
                },
                "messages": [
                    ToolMessage(
                        content=f"String replaced in {path}",
                        tool_call_id=tool_call_id,
                        name=self.tool_name,
                    )
                ],
            }
        )

    def _handle_insert(
        self, args: dict, state: AnthropicToolsState, tool_call_id: str | None
    ) -> Command:
        """处理 insert 命令。"""
        path = args["path"]
        insert_line = args["insert_line"]
        text_to_insert = args["new_str"]

        normalized_path = _validate_path(path, allowed_prefixes=self.allowed_prefixes)

                # 读取文件
        files = cast("dict[str, Any]", state.get(self.state_key, {}))
        file_data = files.get(normalized_path)
        if file_data is None:
            msg = f"File not found: {path}"
            raise FileNotFoundError(msg)

        lines_content = file_data["content"]
        new_lines = text_to_insert.split("\n")

                # 在 insert_line 后插入（从 0 开始索引）
        updated_lines = (
            lines_content[:insert_line] + new_lines + lines_content[insert_line:]
        )

                # 更新文件
        now = datetime.now(timezone.utc).isoformat()

        return Command(
            update={
                self.state_key: {
                    normalized_path: {
                        "content": updated_lines,
                        "created_at": file_data["created_at"],
                        "modified_at": now,
                    }
                },
                "messages": [
                    ToolMessage(
                        content=f"Text inserted in {path}",
                        tool_call_id=tool_call_id,
                        name=self.tool_name,
                    )
                ],
            }
        )

    def _handle_delete(
        self,
        args: dict,
        state: AnthropicToolsState,
        tool_call_id: str | None,
    ) -> Command:
        """处理 delete 命令。"""
        path = args["path"]

        normalized_path = _validate_path(path, allowed_prefixes=self.allowed_prefixes)

        return Command(
            update={
                self.state_key: {normalized_path: None},
                "messages": [
                    ToolMessage(
                        content=f"File deleted: {path}",
                        tool_call_id=tool_call_id,
                        name=self.tool_name,
                    )
                ],
            }
        )

    def _handle_rename(
        self, args: dict, state: AnthropicToolsState, tool_call_id: str | None
    ) -> Command:
        """处理 rename 命令。"""
        old_path = args["old_path"]
        new_path = args["new_path"]

        normalized_old = _validate_path(
            old_path, allowed_prefixes=self.allowed_prefixes
        )
        normalized_new = _validate_path(
            new_path, allowed_prefixes=self.allowed_prefixes
        )

                # 读取文件
        files = cast("dict[str, Any]", state.get(self.state_key, {}))
        file_data = files.get(normalized_old)
        if file_data is None:
            msg = f"File not found: {old_path}"
            raise ValueError(msg)

                # 更新时间戳
        now = datetime.now(timezone.utc).isoformat()
        file_data_copy = file_data.copy()
        file_data_copy["modified_at"] = now

        return Command(
            update={
                self.state_key: {
                    normalized_old: None,
                    normalized_new: file_data_copy,
                },
                "messages": [
                    ToolMessage(
                        content=f"File renamed: {old_path} -> {new_path}",
                        tool_call_id=tool_call_id,
                        name=self.tool_name,
                    )
                ],
            }
        )


class StateClaudeTextEditorMiddleware(_StateClaudeFileToolMiddleware):
    """基于状态的文本编辑器工具中间件。

    使用 LangGraph 状态作为存储，提供 Anthropic 的 `text_editor` 工具。
    文件在对话线程中持久存在。

    Example:
        ```python
        from langchain.agents import create_agent
        from langchain.agents.middleware import StateTextEditorToolMiddleware

        agent = create_agent(
            model=model,
            tools=[],
            middleware=[StateTextEditorToolMiddleware()],
        )
        ```
    """

    def __init__(
        self,
        *,
        allowed_path_prefixes: Sequence[str] | None = None,
    ) -> None:
        """初始化文本编辑器中间件。

        Args:
            allowed_path_prefixes: 可选的允许路径前缀列表。

                若指定，则仅允许以这些前缀开头的路径。
        """
        super().__init__(
            tool_type=TEXT_EDITOR_TOOL_TYPE,
            tool_name=TEXT_EDITOR_TOOL_NAME,
            state_key="text_editor_files",
            allowed_path_prefixes=allowed_path_prefixes,
        )


class StateClaudeMemoryMiddleware(_StateClaudeFileToolMiddleware):
    """基于状态的记忆工具中间件。

    使用 LangGraph 状态作为存储，提供 Anthropic 的记忆工具。
    文件在对话线程中持久存在。

    强制使用 `/memories` 前缀，并注入 Anthropic 推荐的系统提示词。

    Example:
        ```python
        from langchain.agents import create_agent
        from langchain.agents.middleware import StateMemoryToolMiddleware

        agent = create_agent(
            model=model,
            tools=[],
            middleware=[StateMemoryToolMiddleware()],
        )
        ```
    """

    def __init__(
        self,
        *,
        allowed_path_prefixes: Sequence[str] | None = None,
        system_prompt: str = MEMORY_SYSTEM_PROMPT,
    ) -> None:
        """初始化内存中间件。

        Args:
            allowed_path_prefixes: 可选的允许路径前缀列表。

                默认为 `['/memories']`。
            system_prompt: 要注入的系统提示词。

                默认为 Anthropic 推荐的内存提示词。
        """
        super().__init__(
            tool_type=MEMORY_TOOL_TYPE,
            tool_name=MEMORY_TOOL_NAME,
            state_key="memory_files",
            allowed_path_prefixes=allowed_path_prefixes or ["/memories"],
            system_prompt=system_prompt,
        )


class _FilesystemClaudeFileToolMiddleware(AgentMiddleware):
    """基于文件系统的文件工具中间件基类（内部使用）。"""

    def __init__(
        self,
        *,
        tool_type: str,
        tool_name: str,
        root_path: str,
        allowed_prefixes: list[str] | None = None,
        max_file_size_mb: int = 10,
        system_prompt: str | None = None,
    ) -> None:
        """初始化。

        Args:
            tool_type: 工具类型标识符。
            tool_name: 工具名称。
            root_path: 文件操作的根目录。
            allowed_prefixes: 可选的允许虚拟路径前缀列表。
            max_file_size_mb: 最大文件大小（MB）。
            system_prompt: 可选的系统提示词，用于注入。
        """
        self.tool_type = tool_type
        self.tool_name = tool_name
        self.root_path = Path(root_path).resolve()
        self.allowed_prefixes = allowed_prefixes or ["/"]
        self.max_file_size_bytes = max_file_size_mb * 1024 * 1024
        self.system_prompt = system_prompt

                # 如果不存在则创建根目录
        self.root_path.mkdir(parents=True, exist_ok=True)

                # 创建将由工具节点执行的工具
        @tool(tool_name)
        def file_tool(
            runtime: ToolRuntime,
            command: str,
            path: str,
            file_text: str | None = None,
            old_str: str | None = None,
            new_str: str | None = None,
            insert_line: int | None = None,
            new_path: str | None = None,
            view_range: list[int] | None = None,
        ) -> Command | str:
            """在文件系统上执行文件操作。

            Args:
                runtime: 提供 `tool_call_id` 的工具运行时。
                command: 要执行的操作。
                path: 要操作的文件路径。
                file_text: create 命令的完整文件内容。
                old_str: `str_replace` 命令中要替换的字符串。
                new_str: `str_replace` 命令中的替换字符串。
                insert_line: insert 命令的行号（从 0 开始）。
                new_path: rename 命令的新路径。
                view_range: view 命令的行范围 `[start, end]`。

            Returns:
                用于消息更新或字符串结果的命令。
            """
                        # 构建处理方法的参数字典
                        # `old_path` 在 `_handle_rename` 中被填充，该方法会读取源路径
                        # 而非 `path` 键下的路径。
            args: dict[str, Any] = {"path": path, "old_path": path}
            if file_text is not None:
                args["file_text"] = file_text
            if old_str is not None:
                args["old_str"] = old_str
            if new_str is not None:
                args["new_str"] = new_str
            if insert_line is not None:
                args["insert_line"] = insert_line
            if new_path is not None:
                args["new_path"] = new_path
            if view_range is not None:
                args["view_range"] = view_range

                        # 根据命令路由到相应的处理函数
            try:
                if command == "view":
                    return self._handle_view(args, runtime.tool_call_id)
                if command == "create":
                    return self._handle_create(args, runtime.tool_call_id)
                if command == "str_replace":
                    return self._handle_str_replace(args, runtime.tool_call_id)
                if command == "insert":
                    return self._handle_insert(args, runtime.tool_call_id)
                if command == "delete":
                    return self._handle_delete(args, runtime.tool_call_id)
                if command == "rename":
                    return self._handle_rename(args, runtime.tool_call_id)
                return f"Unknown command: {command}"
            except (ValueError, FileNotFoundError, PermissionError) as e:
                return str(e)

        self.tools = [file_tool]

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        """注入 Anthropic 工具描述符和可选的系统提示词。"""
                # 用 Anthropic 的原生工具描述符替换我们的 BaseTool
        tools = [
            t
            for t in (request.tools or [])
            if getattr(t, "name", None) != self.tool_name
        ] + [{"type": self.tool_type, "name": self.tool_name}]

                # 如果提供了系统提示，则注入
        overrides: _ModelRequestOverrides = {"tools": tools}
        if self.system_prompt:
            if request.system_message is not None:
                new_system_content = [
                    *request.system_message.content_blocks,
                    {"type": "text", "text": f"\n\n{self.system_prompt}"},
                ]
            else:
                new_system_content = [{"type": "text", "text": self.system_prompt}]
            new_system_message = SystemMessage(
                content=cast("list[str | dict[str, str]]", new_system_content)
            )
            overrides["system_message"] = new_system_message

        return handler(request.override(**overrides))

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        """注入 Anthropic 工具描述符和可选的系统提示词。"""
                # 用 Anthropic 的原生工具描述符替换我们的 BaseTool
        tools = [
            t
            for t in (request.tools or [])
            if getattr(t, "name", None) != self.tool_name
        ] + [{"type": self.tool_type, "name": self.tool_name}]

                # 如果提供了系统提示，则注入
        overrides: _ModelRequestOverrides = {"tools": tools}
        if self.system_prompt:
            if request.system_message is not None:
                new_system_content = [
                    *request.system_message.content_blocks,
                    {"type": "text", "text": f"\n\n{self.system_prompt}"},
                ]
            else:
                new_system_content = [{"type": "text", "text": self.system_prompt}]
            new_system_message = SystemMessage(
                content=cast("list[str | dict[str, str]]", new_system_content)
            )
            overrides["system_message"] = new_system_message

        return await handler(request.override(**overrides))

    def _validate_and_resolve_path(self, path: str) -> Path:
        """验证并解析虚拟路径为文件系统路径。

        Args:
            path: 虚拟路径（例如 `/file.txt` 或 `/src/main.py`）。

        Returns:
            在 `root_path` 内解析出的绝对文件系统路径。

        Raises:
            ValueError: 如果路径包含遍历尝试、逃逸根目录，
                或违反 `allowed_prefixes` 限制。
        """
                # 归一化路径
        if not path.startswith("/"):
            path = "/" + path

                # 检查路径遍历
        if ".." in path or "~" in path:
            msg = "Path traversal not allowed"
            raise ValueError(msg)

                # 将虚拟路径转换为文件系统路径
                # 移除开头的 / 并相对于根路径解析
        relative = path.lstrip("/")
        full_path = (self.root_path / relative).resolve()

                # 确保路径在根目录内
        try:
            full_path.relative_to(self.root_path)
        except ValueError:
            msg = f"Path outside root directory: {path}"
            raise ValueError(msg) from None

                # 检查允许的前缀
        virtual_path = "/" + str(full_path.relative_to(self.root_path))
        if self.allowed_prefixes and not _is_within_allowed_prefix(
            virtual_path, self.allowed_prefixes
        ):
            msg = f"Path must start with one of: {self.allowed_prefixes}"
            raise ValueError(msg)

        return full_path

    def _handle_view(self, args: dict, tool_call_id: str | None) -> Command:
        """处理 view 命令。"""
        path = args["path"]
        full_path = self._validate_and_resolve_path(path)

        if not full_path.exists() or not full_path.is_file():
            msg = f"File not found: {path}"
            raise FileNotFoundError(msg)

                # 检查文件大小
        if full_path.stat().st_size > self.max_file_size_bytes:
            max_mb = self.max_file_size_bytes / 1024 / 1024
            msg = f"File too large: {path} exceeds {max_mb}MB"
            raise ValueError(msg)

                # 读取文件
        try:
            content = full_path.read_text()
        except UnicodeDecodeError as e:
            msg = f"Cannot decode file {path}: {e}"
            raise ValueError(msg) from e

                # 按行号格式化
        lines = content.split("\n")
                # 移除末尾换行符，如果存在空字符串
        if lines and lines[-1] == "":
            lines = lines[:-1]
        formatted_lines = [f"{i + 1}|{line}" for i, line in enumerate(lines)]
        formatted_content = "\n".join(formatted_lines)

        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content=formatted_content,
                        tool_call_id=tool_call_id,
                        name=self.tool_name,
                    )
                ]
            }
        )

    def _handle_create(self, args: dict, tool_call_id: str | None) -> Command:
        """处理 create 命令。"""
        path = args["path"]
        file_text = args["file_text"]

        full_path = self._validate_and_resolve_path(path)

                # 创建父目录
        full_path.parent.mkdir(parents=True, exist_ok=True)

                # 写入文件
        full_path.write_text(file_text + "\n")

        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content=f"File created: {path}",
                        tool_call_id=tool_call_id,
                        name=self.tool_name,
                    )
                ]
            }
        )

    def _handle_str_replace(self, args: dict, tool_call_id: str | None) -> Command:
        """处理 `str_replace` 命令。"""
        path = args["path"]
        old_str = args["old_str"]
        new_str = args.get("new_str", "")

        full_path = self._validate_and_resolve_path(path)

        if not full_path.exists():
            msg = f"File not found: {path}"
            raise FileNotFoundError(msg)

                # 读取文件
        content = full_path.read_text()

                # 替换字符串
        if old_str not in content:
            msg = f"String not found in file: {old_str}"
            raise ValueError(msg)

        new_content = content.replace(old_str, new_str, 1)

                # 写回
        full_path.write_text(new_content)

        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content=f"String replaced in {path}",
                        tool_call_id=tool_call_id,
                        name=self.tool_name,
                    )
                ]
            }
        )

    def _handle_insert(self, args: dict, tool_call_id: str | None) -> Command:
        """处理 insert 命令。"""
        path = args["path"]
        insert_line = args["insert_line"]
        text_to_insert = args["new_str"]

        full_path = self._validate_and_resolve_path(path)

        if not full_path.exists():
            msg = f"File not found: {path}"
            raise FileNotFoundError(msg)

                # 读取文件
        content = full_path.read_text()
        lines = content.split("\n")
                # 处理末尾换行符
        if lines and lines[-1] == "":
            lines = lines[:-1]
            had_trailing_newline = True
        else:
            had_trailing_newline = False

        new_lines = text_to_insert.split("\n")

                # 在 insert_line 之后插入（0索引）
        updated_lines = lines[:insert_line] + new_lines + lines[insert_line:]

                # 写回
        new_content = "\n".join(updated_lines)
        if had_trailing_newline:
            new_content += "\n"
        full_path.write_text(new_content)

        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content=f"Text inserted in {path}",
                        tool_call_id=tool_call_id,
                        name=self.tool_name,
                    )
                ]
            }
        )

    def _handle_delete(self, args: dict, tool_call_id: str | None) -> Command:
        """处理 delete 命令。"""
        path = args["path"]
        full_path = self._validate_and_resolve_path(path)

        if full_path.is_file():
            full_path.unlink()
        elif full_path.is_dir():
            shutil.rmtree(full_path)
                # 如果不存在，静默成功

        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content=f"File deleted: {path}",
                        tool_call_id=tool_call_id,
                        name=self.tool_name,
                    )
                ]
            }
        )

    def _handle_rename(self, args: dict, tool_call_id: str | None) -> Command:
        """处理 rename 命令。"""
        old_path = args["old_path"]
        new_path = args["new_path"]

        old_full = self._validate_and_resolve_path(old_path)
        new_full = self._validate_and_resolve_path(new_path)

        if not old_full.exists():
            msg = f"File not found: {old_path}"
            raise ValueError(msg)

                # 为新路径创建父目录
        new_full.parent.mkdir(parents=True, exist_ok=True)

                # 重命名
        old_full.rename(new_full)

        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content=f"File renamed: {old_path} -> {new_path}",
                        tool_call_id=tool_call_id,
                        name=self.tool_name,
                    )
                ]
            }
        )


class FilesystemClaudeTextEditorMiddleware(_FilesystemClaudeFileToolMiddleware):
    """基于文件系统的文本编辑器工具中间件。

    使用本地文件系统作为存储，提供 Anthropic 的 `text_editor` 工具。
    用户通过卷、git 或其他机制自行处理持久化。

    Example:
        ```python
        from langchain.agents import create_agent
        from langchain.agents.middleware import FilesystemTextEditorToolMiddleware

        agent = create_agent(
            model=model,
            tools=[],
            middleware=[FilesystemTextEditorToolMiddleware(root_path="/workspace")],
        )
        ```
    """

    def __init__(
        self,
        *,
        root_path: str,
        allowed_prefixes: list[str] | None = None,
        max_file_size_mb: int = 10,
    ) -> None:
        """初始化文本编辑器中间件。

        Args:
            root_path: 文件操作的根目录。
            allowed_prefixes: 可选的允许虚拟路径前缀列表。

                默认为 `['/']`。
            max_file_size_mb: 最大文件大小（MB）

                默认为 `10`。
        """
        super().__init__(
            tool_type=TEXT_EDITOR_TOOL_TYPE,
            tool_name=TEXT_EDITOR_TOOL_NAME,
            root_path=root_path,
            allowed_prefixes=allowed_prefixes,
            max_file_size_mb=max_file_size_mb,
        )


class FilesystemClaudeMemoryMiddleware(_FilesystemClaudeFileToolMiddleware):
    """基于文件系统的记忆工具中间件。

    使用本地文件系统作为存储，提供 Anthropic 的记忆工具。
    用户通过卷、git 或其他机制自行处理持久化。

    强制使用 `/memories` 前缀，并注入 Anthropic 推荐的系统
    提示词。

    Example:
        ```python
        from langchain.agents import create_agent
        from langchain.agents.middleware import FilesystemMemoryToolMiddleware

        agent = create_agent(
            model=model,
            tools=[],
            middleware=[FilesystemMemoryToolMiddleware(root_path="/workspace")],
        )
        ```
    """

    def __init__(
        self,
        *,
        root_path: str,
        allowed_prefixes: list[str] | None = None,
        max_file_size_mb: int = 10,
        system_prompt: str = MEMORY_SYSTEM_PROMPT,
    ) -> None:
        """初始化内存中间件。

        Args:
            root_path: 文件操作的根目录。
            allowed_prefixes: 可选的允许虚拟路径前缀列表。

                默认为 `['/memories']`。
            max_file_size_mb: 最大文件大小（MB）

                默认为 `10`。
            system_prompt: 要注入的系统提示词。

                默认为 Anthropic 推荐的内存提示词。
        """
        super().__init__(
            tool_type=MEMORY_TOOL_TYPE,
            tool_name=MEMORY_TOOL_NAME,
            root_path=root_path,
            allowed_prefixes=allowed_prefixes or ["/memories"],
            max_file_size_mb=max_file_size_mb,
            system_prompt=system_prompt,
        )


__all__ = [
    "AnthropicToolsState",
    "FileData",
    "FilesystemClaudeMemoryMiddleware",
    "FilesystemClaudeTextEditorMiddleware",
    "StateClaudeMemoryMiddleware",
    "StateClaudeTextEditorMiddleware",
]

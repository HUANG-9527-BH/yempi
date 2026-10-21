"""Anthropic 文本编辑器和记忆工具的文件搜索中间件。

此模块提供 Glob 和 Grep 搜索工具，可在状态或文件系统中存储的文件上操作。
"""

from __future__ import annotations

import fnmatch
import re
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Literal, cast

if TYPE_CHECKING:
    from typing import Any

from langchain.agents.middleware.types import AgentMiddleware
from langchain.tools import ToolRuntime, tool

from langchain_anthropic.middleware.anthropic_tools import AnthropicToolsState


def _expand_include_patterns(pattern: str) -> list[str] | None:
    """将 `*.{py,pyi}` 等花括号模式展开为 glob 列表。"""
    if "}" in pattern and "{" not in pattern:
        return None

    expanded: list[str] = []

    def _expand(current: str) -> None:
        start = current.find("{")
        if start == -1:
            expanded.append(current)
            return

        end = current.find("}", start)
        if end == -1:
            msg = f"Unbalanced brace in pattern: '{current}' is missing a '}}'."
            raise ValueError(msg)

        prefix = current[:start]
        suffix = current[end + 1 :]
        inner = current[start + 1 : end]
        if not inner:
            msg = f"Empty brace expansion in pattern: '{current}'."
            raise ValueError(msg)

        for option in inner.split(","):
            _expand(prefix + option + suffix)

    try:
        _expand(pattern)
    except ValueError:
        return None

    return expanded


def _is_valid_include_pattern(pattern: str) -> bool:
    """验证用于包含过滤器的 glob 模式。"""
    if not pattern:
        return False

    if any(char in pattern for char in ("\x00", "\n", "\r")):
        return False

    expanded = _expand_include_patterns(pattern)
    if expanded is None:
        return False

    try:
        for candidate in expanded:
            re.compile(fnmatch.translate(candidate))
    except re.error:
        return False

    return True


def _match_include_pattern(basename: str, pattern: str) -> bool:
    """如果基名匹配包含模式，则返回 `True`。"""
    expanded = _expand_include_patterns(pattern)
    if not expanded:
        return False

    return any(fnmatch.fnmatch(basename, candidate) for candidate in expanded)


class StateFileSearchMiddleware(AgentMiddleware):
    """在基于状态的文件上提供 Glob 和 Grep 搜索。

    此中间件添加了两个搜索虚拟文件的工具：

    - Glob：通过文件路径进行快速文件模式匹配
    - Grep：使用正则表达式进行快速内容搜索

    Example:
        ```python
        from langchain.agents import create_agent
        from langchain.agents.middleware import (
            StateTextEditorToolMiddleware,
            StateFileSearchMiddleware,
        )

        agent = create_agent(
            model=model,
            tools=[],
            middleware=[
                StateTextEditorToolMiddleware(),
                StateFileSearchMiddleware(),
            ],
        )
        ```
    """

    state_schema = AnthropicToolsState

    def __init__(
        self,
        *,
        state_key: str = "text_editor_files",
    ) -> None:
        """初始化搜索中间件。

        Args:
            state_key: 要搜索的状态键

                使用 `'memory_files'` 搜索记忆工具文件。
        """
        self.state_key = state_key

                # 创建工具实例
        @tool
        def glob_search(  # noqa: D417
            runtime: ToolRuntime[None, AnthropicToolsState],
            pattern: str,
            path: str = "/",
        ) -> str:
            """适用于任意代码库规模的快速文件模式匹配工具。

            支持如 `**/*.js` 或 `src/**/*.ts` 的 glob 模式。

            返回按修改时间排序的匹配文件路径。

            当需要根据名称模式查找文件时使用此工具。

            Args:
                pattern: 用于匹配文件的 glob 模式。
                path: 要搜索的目录。

                    如果未指定，则从根目录开始搜索。

            Returns:
                匹配文件路径的换行分隔列表，按修改时间排序（最近修改的在前）。

                    如果无匹配项，则返回 `'No files found'`。
            """
            return self._handle_glob_search(pattern, path, runtime.state)

        @tool
        def grep_search(  # noqa: D417
            runtime: ToolRuntime[None, AnthropicToolsState],
            pattern: str,
            path: str = "/",
            include: str | None = None,
            output_mode: Literal[
                "files_with_matches", "content", "count"
            ] = "files_with_matches",
        ) -> str:
            """适用于任意代码库规模的快速内容搜索工具。

            使用正则表达式搜索文件内容。

            Supports full regex syntax and filters files by pattern with the include
            parameter.

            Args:
                pattern: 在文件内容中搜索的正则表达式模式。
                path: 要搜索的目录。如果未指定，则从根目录开始搜索。
                include: 文件模式过滤器（例如 `'*.js'`，`'*.{ts,tsx}'`）。
                output_mode: 输出格式。

                    选项：

                    - `'files_with_matches'`: 仅包含匹配项的文件路径
                    - `'content'`: 匹配行，格式为 file:line:content
                    - `'count'`: 每个文件中的匹配数量

            Returns:
                根据 `output_mode` 格式化的搜索结果。

                    如果没有匹配项，则返回 `'No matches found'`。
            """
            return self._handle_grep_search(
                pattern, path, include, output_mode, runtime.state
            )

        self.glob_search = glob_search
        self.grep_search = grep_search
        self.tools = [glob_search, grep_search]

    def _handle_glob_search(
        self,
        pattern: str,
        path: str,
        state: AnthropicToolsState,
    ) -> str:
        """处理 glob 搜索操作。

        Args:
            pattern: 用于匹配文件的 glob 模式。
            path: 要搜索的目录。
            state: 当前智能体状态。

        Returns:
            匹配文件路径的换行分隔列表，按修改时间排序（最近修改的在前）。

                如果无匹配项，则返回 `'No files found'`。
        """
                # 规范化基础路径
        base_path = path if path.startswith("/") else "/" + path

                # 从状态中获取文件
        files = cast("dict[str, Any]", state.get(self.state_key, {}))

                # 匹配文件
        matches = []
        for file_path, file_data in files.items():
            if file_path.startswith(base_path):
                                # 从基础路径获取相对路径
                if base_path == "/":
                    relative = file_path[1:]  # 移除开头的 /
                elif file_path == base_path:
                    relative = Path(file_path).name
                elif file_path.startswith(base_path + "/"):
                    relative = file_path[len(base_path) + 1 :]
                else:
                    continue

                                # 根据模式进行匹配
                                # 处理 ** 模式，需要特别注意
                                # PurePosixPath.match 不匹配单级路径
                                # 针对 **/pattern
                is_match = PurePosixPath(relative).match(pattern)
                if not is_match and pattern.startswith("**/"):
                                        # 同时也尝试不带 **/ 前缀匹配，用于基础目录中的文件
                    is_match = PurePosixPath(relative).match(pattern[3:])

                if is_match:
                    matches.append((file_path, file_data["modified_at"]))

        if not matches:
            return "No files found"

                # 按修改时间排序
        matches.sort(key=lambda x: x[1], reverse=True)
        file_paths = [path for path, _ in matches]

        return "\n".join(file_paths)

    def _handle_grep_search(
        self,
        pattern: str,
        path: str,
        include: str | None,
        output_mode: str,
        state: AnthropicToolsState,
    ) -> str:
        """处理 grep 搜索操作。

        Args:
            pattern: 在文件内容中搜索的正则表达式模式。
            path: 要搜索的目录。
            include: 文件模式过滤器（例如 `'*.js'`，`'*.{ts,tsx}'`）。
            output_mode: 输出格式。
            state: 当前智能体状态。

        Returns:
            根据 `output_mode` 格式化搜索结果。

                如果无结果，则返回 `'No matches found'`。
        """
                # 规范化基础路径
        base_path = path if path.startswith("/") else "/" + path

                # 编译正则表达式模式（用于验证）
        try:
            regex = re.compile(pattern)
        except re.error as e:
            return f"Invalid regex pattern: {e}"

        if include and not _is_valid_include_pattern(include):
            return "Invalid include pattern"

                # 搜索文件
        files = cast("dict[str, Any]", state.get(self.state_key, {}))
        results: dict[str, list[tuple[int, str]]] = {}

        for file_path, file_data in files.items():
            if (
                base_path != "/"  # noqa: PLR1714
                and file_path != base_path
                and not file_path.startswith(base_path + "/")
            ):
                continue

                        # 检查包含过滤器
            if include:
                basename = Path(file_path).name
                if not _match_include_pattern(basename, include):
                    continue

                        # 搜索文件内容
            for line_num, line in enumerate(file_data["content"], 1):
                if regex.search(line):
                    if file_path not in results:
                        results[file_path] = []
                    results[file_path].append((line_num, line))

        if not results:
            return "No matches found"

                # 根据模式格式化输出
        return self._format_grep_results(results, output_mode)

    def _format_grep_results(
        self,
        results: dict[str, list[tuple[int, str]]],
        output_mode: str,
    ) -> str:
        """根据输出模式格式化 grep 搜索结果。"""
        if output_mode == "files_with_matches":
                        # 仅返回文件路径
            return "\n".join(sorted(results.keys()))

        if output_mode == "content":
                        # 返回 file:line:content 格式
            lines = []
            for file_path in sorted(results.keys()):
                for line_num, line in results[file_path]:
                    lines.append(f"{file_path}:{line_num}:{line}")
            return "\n".join(lines)

        if output_mode == "count":
                        # 返回 file:count 格式
            lines = []
            for file_path in sorted(results.keys()):
                count = len(results[file_path])
                lines.append(f"{file_path}:{count}")
            return "\n".join(lines)

                # 默认使用 files_with_matches
        return "\n".join(sorted(results.keys()))


__all__ = [
    "StateFileSearchMiddleware",
]

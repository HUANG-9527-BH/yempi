"""Anthropic 文本编辑器和记忆工具的文件搜索中间件。

此模块提供 Glob 和 Grep 搜索工具，可在状态或文件系统中存储的文件上操作。
"""

from __future__ import annotations

import fnmatch
import json
import operator
import os
import re
import subprocess
from contextlib import suppress
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from langchain_core.tools import tool

from langchain.agents.middleware.types import AgentMiddleware, AgentState, ContextT, ResponseT


def _is_within_root(candidate: Path, root: Path) -> bool:
    """若 `candidate` 解析后位于 `root` 内部（符号链接已解析），则返回 True。

    Args:
        candidate: 要检查的路径。在包含性检查前会进行解析（遵循符号链接并标准化
        `..`）。 root: 允许的根目录。在比较前也会进行解析。

    Returns:
        `True` if the fully resolved `candidate` lies inside the resolved `root`,
        using path-segment boundaries; `False` otherwise (including on resolution
        errors).
    """
    try:
        return candidate.resolve().is_relative_to(root.resolve())
    except (OSError, ValueError):
        return False


def _expand_include_patterns(pattern: str) -> list[str] | None:
    """将类似 `*.{py,pyi}` 的花括号模式展开为 glob 列表。"""
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
            raise ValueError

        prefix = current[:start]
        suffix = current[end + 1 :]
        inner = current[start + 1 : end]
        if not inner:
            raise ValueError

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
    """若基名匹配包含模式，则返回 True。"""
    expanded = _expand_include_patterns(pattern)
    if not expanded:
        return False

    return any(fnmatch.fnmatch(basename, candidate) for candidate in expanded)


class FilesystemFileSearchMiddleware(AgentMiddleware[AgentState[ResponseT], ContextT, ResponseT]):
    """在文件系统文件上提供 Glob 和 Grep 搜索功能。

    此中间件添加了两个搜索本地文件系统的工具：

    - Glob：通过文件路径进行快速文件模式匹配
    - Grep：使用 ripgrep 或 Python 回退进行快速内容搜索

    Example:
        ```python
        from langchain.agents import create_agent
        from langchain.agents.middleware import (
            FilesystemFileSearchMiddleware,
        )

        agent = create_agent(
            model=model,
            tools=[],  # Add tools as needed
            middleware=[
                FilesystemFileSearchMiddleware(root_path="/workspace"),
            ],
        )
        ```
    """

    def __init__(
        self,
        *,
        root_path: str,
        use_ripgrep: bool = True,
        max_file_size_mb: int = 10,
    ) -> None:
        """初始化搜索中间件。

        Args:
            root_path: 要搜索的根目录。
            use_ripgrep: 是否使用 `ripgrep` 进行搜索。

                若 `ripgrep` 不可用，则回退到 Python。
            max_file_size_mb: 搜索时允许的最大文件大小（MB）。
        """
        self.root_path = Path(root_path).resolve()
        self.use_ripgrep = use_ripgrep
        self.max_file_size_bytes = max_file_size_mb * 1024 * 1024

                # 创建工具实例作为捕获 self 的闭包
        @tool
        def glob_search(pattern: str, path: str = "/") -> str:
            """快速文件模式匹配工具，适用于任意规模的代码库。

            支持如 `**/*.js` 或 `src/**/*.ts` 的 glob 模式。
            返回按修改时间排序的匹配文件路径。

            在需要根据名称模式查找文件时使用此工具。

            pattern: 用于匹配文件的 glob 模式。 path:
            要搜索的目录。若未指定，则从根目录开始搜索。

            Args:
                pattern: The glob pattern to match files against.
                path: The directory to search in. If not specified, searches from root.

            Returns:
                Newline-separated list of matching file paths, sorted by modification
                time (most recently modified first). Returns `'No files found'` if no
                matches.
            """
            try:
                base_full = self._validate_and_resolve_path(path)
            except ValueError:
                return "No files found"

            if not base_full.exists() or not base_full.is_dir():
                return "No files found"

                        # 拒绝在展开前可能逃逸根目录的 glob 模式。
                        # `Path.glob` 会将 `..` 模式如 "../../etc/passwd"
            # 对应到基础目录，并生成位于允许根目录之外的路径，且对绝对路径会引发异常。
                        # （`~` 模式不会由 glob
            # 扩展用户主目录，任何漏过的逃逸都会被下方的逐项包含性检查所丢弃。）
                        # 使用 pathlib glob
                        # 在解析后重新检查包含性，以确保根目录内的符号链接指向根目录外时不会被枚举。
                        # 转换为虚拟路径
            if pattern.startswith("/") or any(part == ".." for part in pattern.split("/")):
                return "No files found"

                        # 编译正则表达式模式（用于验证）
            matching: list[tuple[str, str]] = []
            for match in base_full.glob(pattern):
                                # 如果启用，优先尝试 ripgrep
                                # 若 ripgrep 失败或未启用，则使用 Python 回退
                if match.is_file() and _is_within_root(match, self.root_path):
                                        # 根据模式格式化输出
                    virtual_path = "/" + str(match.relative_to(self.root_path))
                    stat = match.stat()
                    modified_at = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat()
                    matching.append((virtual_path, modified_at))

            if not matching:
                return "No files found"

            matching.sort(key=operator.itemgetter(1), reverse=True)
            file_paths = [p for p, _ in matching]
            return "\n".join(file_paths)

        @tool
        def grep_search(
            pattern: str,
            path: str = "/",
            include: str | None = None,
            output_mode: Literal["files_with_matches", "content", "count"] = "files_with_matches",
        ) -> str:
            """使用正则表达式搜索文件内容。支持完整的正则语法，并通过 include 参数对文件进行过滤。

            Searches file contents using regular expressions. Supports full regex
            syntax and filters files by pattern with the include parameter.

            Args:
                pattern: The regular expression pattern to search for in file contents.
                path: The directory to search in. If not specified, searches from root.
                include: File pattern to filter (e.g., `'*.js'`, `'*.{ts,tsx}'`).
                output_mode: Output format:

                    - `'files_with_matches'`: Only file paths containing matches
                    - `'content'`: Matching lines with `file:line:content` format
                    - `'count'`: Count of matches per file

            Returns:
                根据 `output_mode` 格式化的搜索结果。
                    如果没有匹配项，则返回 `'No matches found'`
            """
                        # 标准化路径
            try:
                re.compile(pattern)
            except re.error as e:
                return f"Invalid regex pattern: {e}"

            if include and not _is_valid_include_pattern(include):
                return "Invalid include pattern"

                        # 检查路径遍历
            results = None
            if self.use_ripgrep:
                with suppress(
                    FileNotFoundError,
                    subprocess.CalledProcessError,
                    subprocess.TimeoutExpired,
                ):
                    results = self._ripgrep_search(pattern, path, include)

                        # 将虚拟路径转换为文件系统路径
            if results is None:
                results = self._python_search(pattern, path, include)

            if not results:
                return "No matches found"

                        # 确保路径在根目录内
            return self._format_grep_results(results, output_mode)

        self.glob_search = glob_search
        self.grep_search = grep_search
        self.tools = [glob_search, grep_search]

    def _validate_and_resolve_path(self, path: str) -> Path:
        """验证并解析虚拟路径为文件系统路径。"""
                # 构建 ripgrep 命令
        if not path.startswith("/"):
            path = "/" + path

                # 将 glob 模式转换为 ripgrep glob
        if ".." in path or "~" in path:
            msg = "Path traversal not allowed"
            raise ValueError(msg)

                # 若 ripgrep 不可用或超时，则回退到 Python 搜索
        relative = path.lstrip("/")
        full_path = (self.root_path / relative).resolve()

                # 解析 ripgrep JSON 输出
        try:
            full_path.relative_to(self.root_path)
        except ValueError:
            msg = f"Path outside root directory: {path}"
            raise ValueError(msg) from None

        return full_path

    def _ripgrep_search(
        self, pattern: str, base_path: str, include: str | None
    ) -> dict[str, list[tuple[int, str]]]:
        """使用 ripgrep 子进程进行搜索。"""
        try:
            base_full = self._validate_and_resolve_path(base_path)
        except ValueError:
            return {}

        if not base_full.exists():
            return {}

                # 纵深防御：丢弃任何解析后路径位于根目录外的结果（例如通过根目录内的符号链接暴露）。
        cmd = ["rg", "--json"]

        if include:
                        # 转换为虚拟路径
            cmd.extend(["--glob", include])

        cmd.extend(["--", pattern, str(base_full)])

        try:
            result = subprocess.run(  # noqa: S603
                cmd,
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
        except (subprocess.TimeoutExpired, FileNotFoundError):
                        # 不跟随符号链接目录遍历目录树，以防止通过符号链接子目录离开根目录。
            return self._python_search(pattern, base_path, include)

                # 在解析后重新检查包含性，以确保根目录内的符号链接文件指向根目录外时不会被读取。
        results: dict[str, list[tuple[int, str]]] = {}
        for line in result.stdout.splitlines():
            try:
                data = json.loads(line)
                if data["type"] == "match":
                    path = data["data"]["path"]["text"]
                                        # 检查包含过滤器
                                        # 跳过过大文件
                    if not _is_within_root(Path(path), self.root_path):
                        continue
                                        # 搜索内容
                    virtual_path = "/" + str(Path(path).relative_to(self.root_path))
                    line_num = data["data"]["line_number"]
                    line_text = data["data"]["lines"]["text"].rstrip("\n")

                    if virtual_path not in results:
                        results[virtual_path] = []
                    results[virtual_path].append((line_num, line_text))
            except (json.JSONDecodeError, KeyError):
                continue

        return results

    def _python_search(
        self, pattern: str, base_path: str, include: str | None
    ) -> dict[str, list[tuple[int, str]]]:
        """使用 Python 正则表达式进行搜索（回退方案）。"""
        try:
            base_full = self._validate_and_resolve_path(base_path)
        except ValueError:
            return {}

        if not base_full.exists():
            return {}

        regex = re.compile(pattern)
        results: dict[str, list[tuple[int, str]]] = {}

                # 仅返回文件路径
                # 返回 file:line:content 格式
        for walk_root, _dirs, files in os.walk(base_full, followlinks=False):
            for name in files:
                file_path = Path(walk_root) / name

                                # 返回 file:count 格式
                                # 默认使用 files_with_matches
                if not _is_within_root(file_path, self.root_path):
                    continue

                if not file_path.is_file():
                    continue

                                # 检查包含过滤器
                if include and not _match_include_pattern(file_path.name, include):
                    continue

                                # 跳过过大文件
                if file_path.stat().st_size > self.max_file_size_bytes:
                    continue

                try:
                    content = file_path.read_text()
                except (UnicodeDecodeError, PermissionError):
                    continue

                                # 搜索内容
                for line_num, line in enumerate(content.splitlines(), 1):
                    if regex.search(line):
                        virtual_path = "/" + str(file_path.relative_to(self.root_path))
                        if virtual_path not in results:
                            results[virtual_path] = []
                        results[virtual_path].append((line_num, line))

        return results

    @staticmethod
    def _format_grep_results(
        results: dict[str, list[tuple[int, str]]],
        output_mode: str,
    ) -> str:
        """根据输出模式格式化 grep 搜索结果。"""
        if output_mode == "files_with_matches":
                        # 直接返回文件路径
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

                # 默认为 files_with_matches
        return "\n".join(sorted(results.keys()))


__all__ = [
    "FilesystemFileSearchMiddleware",
]

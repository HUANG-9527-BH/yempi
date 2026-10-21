"""Markdown 文本分割器。"""

from __future__ import annotations

import re
from typing import Any, TypedDict

from langchain_core.documents import Document

from langchain_text_splitters.base import Language
from langchain_text_splitters.character import RecursiveCharacterTextSplitter


class MarkdownTextSplitter(RecursiveCharacterTextSplitter):
    """尝试沿 Markdown 格式的标题分割文本。"""

    def __init__(self, **kwargs: Any) -> None:
        """初始化一个 `MarkdownTextSplitter`。"""
        separators = self.get_separators_for_language(Language.MARKDOWN)
        super().__init__(separators=separators, **kwargs)


class MarkdownHeaderTextSplitter:
    """基于指定标题分割 Markdown 文件。"""

    def __init__(
        self,
        headers_to_split_on: list[tuple[str, str]],
        return_each_line: bool = False,  # noqa: FBT001,FBT002
        strip_headers: bool = True,  # noqa: FBT001,FBT002
        custom_header_patterns: dict[str, int] | None = None,
    ) -> None:
        """创建一个新的 `MarkdownHeaderTextSplitter`。

        Args:
            headers_to_split_on: Headers we want to track
            return_each_line: Return each line w/ associated headers
            strip_headers: Strip split headers from the content of the chunk
            custom_header_patterns: Optional dict mapping header patterns to their
                levels.

                For example: 例如：`{"**": 1, "***": 2}` 将 `**Header**` 视为一级标题，
                `***Header***` 视为二级标题。
        """
                # Output line-by-line or aggregated into chunks w/ common headers
        self.return_each_line = return_each_line
                # Given the headers we want to split on,
                # (e.g., "#, ##, etc") order by length
        self.headers_to_split_on = sorted(
            headers_to_split_on, key=lambda split: len(split[0]), reverse=True
        )
                # Strip headers split headers from the content of the chunk
        self.strip_headers = strip_headers
                # Custom header patterns with their levels
        self.custom_header_patterns = custom_header_patterns or {}

    def _is_custom_header(self, line: str, sep: str) -> bool:
        """检查行是否匹配自定义标题模式。

        Args:
            line: 要检查的行
            sep: 要匹配的分隔符模式

        Returns:
            如果行匹配自定义模式格式，则返回 `True`
        """
        if sep not in self.custom_header_patterns:
            return False

                # Escape special regex characters in the separator
        escaped_sep = re.escape(sep)
                # Create regex pattern to match exactly one separator at start and end
                # with content in between
        pattern = (
            f"^{escaped_sep}(?!{escaped_sep})(.+?)(?<!{escaped_sep}){escaped_sep}$"
        )

        match = re.match(pattern, line)
        if match:
                        # Extract the content between the patterns
            content = match.group(1).strip()
                        # Valid header if there's actual content (not just whitespace or separators)
                        # Check that content doesn't consist only of separator characters
            if content and not all(c in sep for c in content.replace(" ", "")):
                return True
        return False

    def aggregate_lines_to_chunks(self, lines: list[LineType]) -> list[Document]:
        """将具有共同元数据的行合并为分块。

        Args:
            lines: 带有相关标题元数据的文本行

        Returns:
            包含聚合共同元数据的 `Document` 对象列表。
        """
        aggregated_chunks: list[LineType] = []

        for line in lines:
            if (
                aggregated_chunks
                and aggregated_chunks[-1]["metadata"] == line["metadata"]
            ):
                                # If the last line in the aggregated list
                                # has the same metadata as the current line,
                                # append the current content to the last lines's content
                aggregated_chunks[-1]["content"] += "  \n" + line["content"]
            elif (
                aggregated_chunks
                and aggregated_chunks[-1]["metadata"] != line["metadata"]
                                # may be issues if other metadata is present
                and len(aggregated_chunks[-1]["metadata"]) < len(line["metadata"])
                and aggregated_chunks[-1]["content"].split("\n")[-1][0] == "#"
                and not self.strip_headers
            ):
                                # If the last line in the aggregated list
                                # has different metadata as the current line,
                                # and has shallower header level than the current line,
                                # and the last line is a header,
                                # and we are not stripping headers,
                                # append the current content to the last line's content
                aggregated_chunks[-1]["content"] += "  \n" + line["content"]
                                # and update the last line's metadata
                aggregated_chunks[-1]["metadata"] = line["metadata"]
            else:
                                # Otherwise, append the current line to the aggregated list
                aggregated_chunks.append(line)

        return [
            Document(page_content=chunk["content"], metadata=chunk["metadata"])
            for chunk in aggregated_chunks
        ]

    def split_text(self, text: str) -> list[Document]:
        """分割 Markdown 文件。

        Args:
            text: Markdown 文件

        Returns:
            `Document` 对象列表。
        """
                # Split the input text by newline character ("\n").
        lines = text.split("\n")

                # Final output
        lines_with_metadata: list[LineType] = []

                # Content and metadata of the chunk currently being processed
        current_content: list[str] = []

        current_metadata: dict[str, str] = {}

                # Keep track of the nested header structure
        header_stack: list[HeaderType] = []

        initial_metadata: dict[str, str] = {}

        in_code_block = False

        opening_fence = ""

        for line in lines:
            stripped_line = line.strip()
                        # Remove all non-printable characters from the string, keeping only visible
                        # text.
            stripped_line = "".join(filter(str.isprintable, stripped_line))
            if not in_code_block:
                                # Exclude inline code spans
                if stripped_line.startswith("```") and stripped_line.count("```") == 1:
                    in_code_block = True
                    opening_fence = "```"
                elif stripped_line.startswith("~~~"):
                    in_code_block = True
                    opening_fence = "~~~"
            elif stripped_line.startswith(opening_fence):
                in_code_block = False
                opening_fence = ""

            if in_code_block:
                current_content.append(stripped_line)
                continue

                        # Check each line against each of the header types (e.g., #, ##)
            for sep, name in self.headers_to_split_on:
                is_standard_header = stripped_line.startswith(sep) and (
                                        # Header with no text OR header is followed by space
                                        # Both are valid conditions that sep is being used a header
                    len(stripped_line) == len(sep) or stripped_line[len(sep)] == " "
                )
                is_custom_header = self._is_custom_header(stripped_line, sep)

                                # Check if line matches either standard or custom header pattern
                if is_standard_header or is_custom_header:
                                        # Preserve support for untyped callers using None to skip metadata.
                    if name is not None:  # ty: ignore[redundant-condition-strict]
                                                # Get the current header level
                        if sep in self.custom_header_patterns:
                            current_header_level = self.custom_header_patterns[sep]
                        else:
                            current_header_level = sep.count("#")

                                                # Pop out headers of lower or same level from the stack
                        while (
                            header_stack
                            and header_stack[-1]["level"] >= current_header_level
                        ):
                                                        # We have encountered a new header
                                                        # at the same or higher level
                            popped_header = header_stack.pop()
                                                        # Clear the metadata for the
                                                        # popped header in initial_metadata
                            if popped_header["name"] in initial_metadata:
                                initial_metadata.pop(popped_header["name"])

                                                # Push the current header to the stack
                                                # Extract header text based on header type
                        if is_custom_header:
                                                        # For custom headers like **Header**, extract text
                                                        # between patterns
                            header_text = stripped_line[len(sep) : -len(sep)].strip()
                        else:
                                                        # For standard headers like # Header, extract text
                                                        # after the separator
                            header_text = stripped_line[len(sep) :].strip()

                        header: HeaderType = {
                            "level": current_header_level,
                            "name": name,
                            "data": header_text,
                        }
                        header_stack.append(header)
                                                # Update initial_metadata with the current header
                        initial_metadata[name] = header["data"]

                                        # Add the previous line to the lines_with_metadata
                                        # only if current_content is not empty
                    if current_content:
                        lines_with_metadata.append(
                            {
                                "content": "\n".join(current_content),
                                "metadata": current_metadata.copy(),
                            }
                        )
                        current_content.clear()

                    if not self.strip_headers:
                        current_content.append(stripped_line)

                    break
            else:
                if stripped_line:
                    current_content.append(stripped_line)
                elif current_content:
                    lines_with_metadata.append(
                        {
                            "content": "\n".join(current_content),
                            "metadata": current_metadata.copy(),
                        }
                    )
                    current_content.clear()

            current_metadata = initial_metadata.copy()

        if current_content:
            lines_with_metadata.append(
                {
                    "content": "\n".join(current_content),
                    "metadata": current_metadata,
                }
            )

                # lines_with_metadata has each line with associated header metadata
                # aggregate these into chunks based on common metadata
        if not self.return_each_line:
            return self.aggregate_lines_to_chunks(lines_with_metadata)
        return [
            Document(page_content=chunk["content"], metadata=chunk["metadata"])
            for chunk in lines_with_metadata
        ]


class LineType(TypedDict):
    """行类型作为 `TypedDict`。"""

    metadata: dict[str, str]
    content: str


class HeaderType(TypedDict):
    """标题类型作为 `TypedDict`。"""

    level: int
    name: str
    data: str


class ExperimentalMarkdownSyntaxTextSplitter:
    """用于处理 Markdown 语法的实验性文本分割器。

    This splitter aims to retain the exact whitespace of the original text while
    extracting structured metadata, such as headers. It is a re-implementation of the
    `MarkdownHeaderTextSplitter` with notable changes to the approach and additional
    features.

    主要特性：

    * 保留 Markdown 文本的原始空白字符和格式。 * 提取标题、代码块和水平线作为元数据。 *
    分离代码块，并将语言包含在 "Code" 元数据键中。 * 也根据水平线（`---`）进行分割。 *
    默认采用合理的分割行为，可通过 `headers_to_split_on` 参数覆盖。

    Example:
        ```python
        headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
        ]
        splitter = ExperimentalMarkdownSyntaxTextSplitter(
            headers_to_split_on=headers_to_split_on
        )
        chunks = splitter.split(text)
        for chunk in chunks:
            print(chunk)
        ```

    This class is currently experimental and subject to change based on feedback and
    further development.
    """

    def __init__(
        self,
        headers_to_split_on: list[tuple[str, str]] | None = None,
        return_each_line: bool = False,  # noqa: FBT001,FBT002
        strip_headers: bool = True,  # noqa: FBT001,FBT002
    ) -> None:
        """使用标题分割和格式选项初始化文本分割器。

        This constructor sets up the required configuration for splitting text into
        chunks based on specified headers and formatting preferences.

        Args:
            headers_to_split_on: 由元组组成的列表，每个元组包含一个标题标签（例如
            "h1"）及其对应的元数据键。

                如果为 `None`，则使用默认标题。
            return_each_line: 是否将每行作为独立分块返回。

                默认为 `False`，即把行聚合为更大的分块。
            strip_headers: 是否从结果分块中排除标题。
        """
        self.chunks: list[Document] = []
        self.current_chunk = Document(page_content="")
        self.current_header_stack: list[tuple[int, str]] = []
        self.strip_headers = strip_headers
        if headers_to_split_on:
            self.splittable_headers = dict(headers_to_split_on)
        else:
            self.splittable_headers = {
                "#": "Header 1",
                "##": "Header 2",
                "###": "Header 3",
                "####": "Header 4",
                "#####": "Header 5",
                "######": "Header 6",
            }

        self.return_each_line = return_each_line

    def split_text(self, text: str) -> list[Document]:
        """将输入文本分割为结构化分块。

        此方法逐行处理输入文本，识别并处理特定模式，如标题、代码块和水平规则，根据这些模式将文本分割为结构化块。

        Args:
            text: 待分割的输入文本。

        Returns:
            一个 `Document` 对象列表，表示输入文本的结构化块。如果启用了
            `return_each_line`，则每行作为一个独立的 `Document` 返回。
        """
                # Reset the state for each new file processed
        self.chunks.clear()
        self.current_chunk = Document(page_content="")
        self.current_header_stack.clear()

        raw_lines = text.splitlines(keepends=True)

        while raw_lines:
            raw_line = raw_lines.pop(0)
            header_match = self._match_header(raw_line)
            code_match = self._match_code(raw_line)
            horz_match = self._match_horz(raw_line)
            if header_match:
                self._complete_chunk_doc()

                if not self.strip_headers:
                    self.current_chunk.page_content += raw_line

                                # add the header to the stack
                header_depth = len(header_match.group(1))
                header_text = header_match.group(2)
                self._resolve_header_stack(header_depth, header_text)
            elif code_match:
                self._complete_chunk_doc()
                self.current_chunk.page_content = self._resolve_code_chunk(
                    raw_line, raw_lines
                )
                self.current_chunk.metadata["Code"] = code_match.group(1)
                self._complete_chunk_doc()
            elif horz_match:
                self._complete_chunk_doc()
            else:
                self.current_chunk.page_content += raw_line

        self._complete_chunk_doc()
                # I don't see why `return_each_line` is a necessary feature of this splitter.
                # 在类外部完成也足够简单，调用方可以拥有更多控制权。
                # 控制权。
        if self.return_each_line:
            return [
                Document(page_content=line, metadata=chunk.metadata)
                for chunk in self.chunks
                for line in chunk.page_content.splitlines()
                if line and not line.isspace()
            ]
        return self.chunks

    def _resolve_header_stack(self, header_depth: int, header_text: str) -> None:
        for i, (depth, _) in enumerate(self.current_header_stack):
            if depth >= header_depth:
                                # 截断从此级别开始的所有内容
                self.current_header_stack = self.current_header_stack[:i]
                break
        self.current_header_stack.append((header_depth, header_text))

    def _resolve_code_chunk(self, current_line: str, raw_lines: list[str]) -> str:
        chunk = current_line
        while raw_lines:
            raw_line = raw_lines.pop(0)
            chunk += raw_line
            if self._match_code(raw_line):
                return chunk
        return ""

    def _complete_chunk_doc(self) -> None:
        chunk_content = self.current_chunk.page_content
                # 丢弃任何空文档
        if chunk_content and not chunk_content.isspace():
                        # 将标题栈作为元数据应用
            for depth, value in self.current_header_stack:
                header_key = self.splittable_headers.get("#" * depth)
                if header_key is not None:
                    self.current_chunk.metadata[header_key] = value
            self.chunks.append(self.current_chunk)
                # 重置当前块
        self.current_chunk = Document(page_content="")

        # 匹配方法
    def _match_header(self, line: str) -> re.Match[str] | None:
        match = re.match(r"^(#{1,6}) (.*)", line)
                # 仅匹配配置的标题
        if match and match.group(1) in self.splittable_headers:
            return match
        return None

    @staticmethod
    def _match_code(line: str) -> re.Match[str] | None:
        matches = [re.match(rule, line) for rule in [r"^```(.*)", r"^~~~(.*)"]]
        return next((match for match in matches if match), None)

    @staticmethod
    def _match_horz(line: str) -> re.Match[str] | None:
        matches = [
            re.match(rule, line) for rule in [r"^\*\*\*+\n", r"^---+\n", r"^___+\n"]
        ]
        return next((match for match in matches if match), None)

"""用于列表输出的解析器。"""

from __future__ import annotations

import csv
import re
from abc import abstractmethod
from collections import deque
from io import StringIO
from typing import TYPE_CHECKING, TypeVar

from typing_extensions import override

from langchain_core.messages import BaseMessage
from langchain_core.output_parsers.transform import BaseTransformOutputParser

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Iterator

T = TypeVar("T")


def droplastn(
    iter: Iterator[T],  # noqa: A002
    n: int,
) -> Iterator[T]:
    """从迭代器中丢弃最后的 `n` 个元素。

    Args:
        iter: 要丢弃元素的迭代器。
        n: 要丢弃的元素数量。

    Yields:
        迭代器中的元素，但不包括最后的 n 个元素。
    """
    buffer: deque[T] = deque()
    for item in iter:
        buffer.append(item)
        if len(buffer) > n:
            yield buffer.popleft()


class ListOutputParser(BaseTransformOutputParser[list[str]]):
    """将模型输出解析为列表。"""

    @property
    def _type(self) -> str:
        return "list"

    @abstractmethod
    def parse(self, text: str) -> list[str]:
        """解析 LLM 调用的输出。

        Args:
            text: LLM 调用的输出。

        Returns:
            一组字符串。
        """

    def parse_iter(self, text: str) -> Iterator[re.Match[str]]:
        """解析 LLM 调用的输出。

        Args:
            text: LLM 调用的输出。

        Yields:
            每个输出部分的匹配对象。
        """
        raise NotImplementedError

    @override
    def _transform(self, input: Iterator[str | BaseMessage]) -> Iterator[list[str]]:
        buffer = ""
        for chunk in input:
            if isinstance(chunk, BaseMessage):
                                # 提取文本
                chunk_content = chunk.content
                if not isinstance(chunk_content, str):
                    continue
                buffer += chunk_content
            else:
                                # 将当前块添加到缓冲区
                buffer += chunk
                        # 将缓冲区解析为多个部分的列表
            try:
                done_idx = 0
                                # 仅生成完整部分
                for m in droplastn(self.parse_iter(buffer), 1):
                    done_idx = m.end()
                    yield [m.group(1)]
                buffer = buffer[done_idx:]
            except NotImplementedError:
                parts = self.parse(buffer)
                                # 仅生成完整部分
                if len(parts) > 1:
                    for part in parts[:-1]:
                        yield [part]
                    buffer = parts[-1]
                # 生成最后一个部分
        for part in self.parse(buffer):
            yield [part]

    @override
    async def _atransform(
        self, input: AsyncIterator[str | BaseMessage]
    ) -> AsyncIterator[list[str]]:
        buffer = ""
        async for chunk in input:
            if isinstance(chunk, BaseMessage):
                                # 提取文本
                chunk_content = chunk.content
                if not isinstance(chunk_content, str):
                    continue
                buffer += chunk_content
            else:
                                # 将当前块添加到缓冲区
                buffer += chunk
                        # 将缓冲区解析为多个部分的列表
            try:
                done_idx = 0
                                # 仅生成完整部分
                for m in droplastn(self.parse_iter(buffer), 1):
                    done_idx = m.end()
                    yield [m.group(1)]
                buffer = buffer[done_idx:]
            except NotImplementedError:
                parts = self.parse(buffer)
                                # 仅生成完整部分
                if len(parts) > 1:
                    for part in parts[:-1]:
                        yield [part]
                    buffer = parts[-1]
                # 生成最后一个部分
        for part in self.parse(buffer):
            yield [part]


class CommaSeparatedListOutputParser(ListOutputParser):
    """将模型输出解析为逗号分隔的列表。"""

    @classmethod
    def is_lc_serializable(cls) -> bool:
        """返回 `True`，因为此类可序列化。"""
        return True

    @classmethod
    def get_lc_namespace(cls) -> list[str]:
        """获取 LangChain 对象的命名空间。

        Returns:
            `["langchain", "output_parsers", "list"]`
        """
        return ["langchain", "output_parsers", "list"]

    @override
    def get_format_instructions(self) -> str:
        """返回逗号分隔列表输出的格式说明。"""
        return (
            "Your response should be a list of comma separated values, "
            "eg: `foo, bar, baz` or `foo,bar,baz`"
        )

    @override
    def parse(self, text: str) -> list[str]:
        """解析 LLM 调用的输出。

        Args:
            text: LLM 调用的输出。

        Returns:
            一组字符串。
        """
        try:
            reader = csv.reader(
                StringIO(text), quotechar='"', delimiter=",", skipinitialspace=True
            )
            return [item for sublist in reader for item in sublist]
        except csv.Error:
                        # 保留旧逻辑作为备用
            return [part.strip() for part in text.split(",")]

    @property
    def _type(self) -> str:
        return "comma-separated-list"


class NumberedListOutputParser(ListOutputParser):
    """解析编号列表。"""

    pattern: str = r"\d+\.\s([^\n]+)"
    """匹配编号列表项的模式。"""

    @override
    def get_format_instructions(self) -> str:
        return (
            "Your response should be a numbered list with each item on a new line. "
            "For example: \n\n1. foo\n\n2. bar\n\n3. baz"
        )

    def parse(self, text: str) -> list[str]:
        """解析 LLM 调用的输出。

        Args:
            text: LLM 调用的输出。

        Returns:
            一组字符串。
        """
        return re.findall(self.pattern, text)

    @override
    def parse_iter(self, text: str) -> Iterator[re.Match[str]]:
        return re.finditer(self.pattern, text)

    @property
    def _type(self) -> str:
        return "numbered-list"


class MarkdownListOutputParser(ListOutputParser):
    """解析 Markdown 列表。"""

    pattern: str = r"^\s*[-*]\s([^\n]+)$"
    """匹配 Markdown 列表项的模式。"""

    @override
    def get_format_instructions(self) -> str:
        """返回 Markdown 列表输出的格式说明。"""
        return "Your response should be a markdown list, eg: `- foo\n- bar\n- baz`"

    def parse(self, text: str) -> list[str]:
        """解析 LLM 调用的输出。

        Args:
            text: LLM 调用的输出。

        Returns:
            一组字符串。
        """
        return re.findall(self.pattern, text, re.MULTILINE)

    @override
    def parse_iter(self, text: str) -> Iterator[re.Match[str]]:
        return re.finditer(self.pattern, text, re.MULTILINE)

    @property
    def _type(self) -> str:
        return "markdown-list"

"""用于处理流式输入的输出解析器的基础类。"""

from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
)

from typing_extensions import override

from langchain_core.messages import BaseMessage, BaseMessageChunk
from langchain_core.output_parsers.base import BaseOutputParser, T
from langchain_core.outputs import (
    ChatGeneration,
    ChatGenerationChunk,
    Generation,
    GenerationChunk,
)
from langchain_core.runnables.config import run_in_executor

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Iterator

    from langchain_core.runnables import RunnableConfig


class BaseTransformOutputParser(BaseOutputParser[T]):
    """可处理流式输入的输出解析器的基础类。"""

    def _transform(
        self,
        input: Iterator[str | BaseMessage],
    ) -> Iterator[T]:
        for chunk in input:
            if isinstance(chunk, BaseMessage):
                yield self.parse_result([ChatGeneration(message=chunk)])
            else:
                yield self.parse_result([Generation(text=chunk)])

    async def _atransform(
        self,
        input: AsyncIterator[str | BaseMessage],
    ) -> AsyncIterator[T]:
        async for chunk in input:
            if isinstance(chunk, BaseMessage):
                yield await run_in_executor(
                    None, self.parse_result, [ChatGeneration(message=chunk)]
                )
            else:
                yield await run_in_executor(
                    None, self.parse_result, [Generation(text=chunk)]
                )

    @override
    def transform(
        self,
        input: Iterator[str | BaseMessage],
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> Iterator[T]:
        """将输入转换为输出格式。

        Args:
            input: 要转换的输入。
            config: 用于转换的配置。
            **kwargs: 额外的关键字参数。

        Yields:
            转换后的输出。
        """
        yield from self._transform_stream_with_config(
            input, self._transform, config, run_type="parser"
        )

    @override
    async def atransform(
        self,
        input: AsyncIterator[str | BaseMessage],
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[T]:
        """异步将输入转换为输出格式。

        Args:
            input: 要转换的输入。
            config: 用于转换的配置。
            **kwargs: 额外的关键字参数。

        Yields:
            转换后的输出。
        """
        async for chunk in self._atransform_stream_with_config(
            input, self._atransform, config, run_type="parser"
        ):
            yield chunk


class BaseCumulativeTransformOutputParser(BaseTransformOutputParser[T]):
    """可处理流式输入的输出解析器的基础类。"""

    diff: bool = False
    """In streaming mode, whether to yield diffs between the previous and current parsed
    output, or just the current parsed output.
    """

    def _diff(
        self,
        prev: T | None,
        next: T,  # noqa: A002
    ) -> T:
        """将已解析的输出转换为差异格式。

        此行为由输出解析器自行定义。

        Args:
            prev: 上一次已解析的输出。
            next: 当前已解析的输出。

        Returns:
            上一次与当前已解析输出之间的差异。
        """
        raise NotImplementedError

    @override
    def _transform(self, input: Iterator[str | BaseMessage]) -> Iterator[Any]:
        prev_parsed = None
        acc_gen: GenerationChunk | ChatGenerationChunk | None = None
        for chunk in input:
            chunk_gen: GenerationChunk | ChatGenerationChunk
            if isinstance(chunk, BaseMessageChunk):
                chunk_gen = ChatGenerationChunk(message=chunk)
            elif isinstance(chunk, BaseMessage):
                chunk_gen = ChatGenerationChunk(
                    message=BaseMessageChunk(**chunk.model_dump())
                )
            else:
                chunk_gen = GenerationChunk(text=chunk)

            acc_gen = chunk_gen if acc_gen is None else acc_gen + chunk_gen  # type: ignore[operator]

            parsed = self.parse_result([acc_gen], partial=True)
            if parsed is not None and parsed != prev_parsed:
                if self.diff:
                    yield self._diff(prev_parsed, parsed)
                else:
                    yield parsed
                prev_parsed = parsed

    @override
    async def _atransform(
        self, input: AsyncIterator[str | BaseMessage]
    ) -> AsyncIterator[T]:
        prev_parsed = None
        acc_gen: GenerationChunk | ChatGenerationChunk | None = None
        async for chunk in input:
            chunk_gen: GenerationChunk | ChatGenerationChunk
            if isinstance(chunk, BaseMessageChunk):
                chunk_gen = ChatGenerationChunk(message=chunk)
            elif isinstance(chunk, BaseMessage):
                chunk_gen = ChatGenerationChunk(
                    message=BaseMessageChunk(**chunk.model_dump())
                )
            else:
                chunk_gen = GenerationChunk(text=chunk)

            acc_gen = chunk_gen if acc_gen is None else acc_gen + chunk_gen  # type: ignore[operator]

            parsed = await self.aparse_result([acc_gen], partial=True)
            if parsed is not None and parsed != prev_parsed:
                if self.diff:
                    yield await run_in_executor(None, self._diff, prev_parsed, parsed)
                else:
                    yield parsed
                prev_parsed = parsed

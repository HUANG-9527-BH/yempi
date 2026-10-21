from __future__ import annotations

from typing import Annotated, Any, TypeVar

from langchain_core.exceptions import OutputParserException
from langchain_core.output_parsers import BaseOutputParser, StrOutputParser
from langchain_core.prompts import BasePromptTemplate
from langchain_core.runnables import Runnable, RunnableSerializable
from pydantic import SkipValidation
from typing_extensions import TypedDict, override

from langchain_classic.output_parsers.prompts import NAIVE_FIX_PROMPT

T = TypeVar("T")


class OutputFixingParserRetryChainInput(TypedDict, total=False):
    """重试链的输出修复解析器的输入。"""

    instructions: str
    completion: str
    error: str


class OutputFixingParser(BaseOutputParser[T]):
    """包装一个解析器并尝试修复解析错误。"""

    @classmethod
    @override
    def is_lc_serializable(cls) -> bool:
        return True

    parser: Annotated[Any, SkipValidation()]
    """用于解析输出的解析器。"""
        # 应为 LLMChain，但为了避免顶层导入
        # langchain_classic.chains
    retry_chain: Annotated[
        RunnableSerializable[OutputFixingParserRetryChainInput, str] | Any,
        SkipValidation(),
    ]
    """用于重试完成的 RunnableSerializable（旧版：LLMChain）。"""
    max_retries: int = 1
    """最多重试解析的次数。"""
    legacy: bool = True
    """是否使用 retry_chain 的 run 或 arun 方法。"""

    @classmethod
    def from_llm(
        cls,
        llm: Runnable,
        parser: BaseOutputParser[T],
        prompt: BasePromptTemplate = NAIVE_FIX_PROMPT,
        max_retries: int = 1,
    ) -> OutputFixingParser[T]:
        """从语言模型和解析器创建 OutputFixingParser。

        Args:
            llm: 用于修复的语言模型
            parser: 用于解析的解析器
            prompt: 用于修复的提示词
            max_retries: 最大重试次数以进行解析。

        Returns:
            OutputFixingParser
        """
        chain = prompt | llm | StrOutputParser()
        return cls(parser=parser, retry_chain=chain, max_retries=max_retries)

    @override
    def parse(self, completion: str) -> T:
        retries = 0

        while retries <= self.max_retries:
            try:
                return self.parser.parse(completion)
            except OutputParserException as e:
                if retries == self.max_retries:
                    raise
                retries += 1
                if self.legacy and hasattr(self.retry_chain, "run"):
                    completion = self.retry_chain.run(
                        instructions=self.parser.get_format_instructions(),
                        completion=completion,
                        error=repr(e),
                    )
                else:
                    try:
                        completion = self.retry_chain.invoke(
                            {
                                "instructions": self.parser.get_format_instructions(),
                                "completion": completion,
                                "error": repr(e),
                            },
                        )
                    except (NotImplementedError, AttributeError):
                                                # 情况：self.parser 没有 get_format_instructions
                        completion = self.retry_chain.invoke(
                            {
                                "completion": completion,
                                "error": repr(e),
                            },
                        )

        msg = "Failed to parse"
        raise OutputParserException(msg)

    @override
    async def aparse(self, completion: str) -> T:
        retries = 0

        while retries <= self.max_retries:
            try:
                return await self.parser.aparse(completion)
            except OutputParserException as e:
                if retries == self.max_retries:
                    raise
                retries += 1
                if self.legacy and hasattr(self.retry_chain, "arun"):
                    completion = await self.retry_chain.arun(
                        instructions=self.parser.get_format_instructions(),
                        completion=completion,
                        error=repr(e),
                    )
                else:
                    try:
                        completion = await self.retry_chain.ainvoke(
                            {
                                "instructions": self.parser.get_format_instructions(),
                                "completion": completion,
                                "error": repr(e),
                            },
                        )
                    except (NotImplementedError, AttributeError):
                                                # 情况：self.parser 没有 get_format_instructions
                        completion = await self.retry_chain.ainvoke(
                            {
                                "completion": completion,
                                "error": repr(e),
                            },
                        )

        msg = "Failed to parse"
        raise OutputParserException(msg)

    @override
    def get_format_instructions(self) -> str:
        return self.parser.get_format_instructions()

    @property
    def _type(self) -> str:
        return "output_fixing"

    @property
    @override
    def OutputType(self) -> type[T]:
        return self.parser.OutputType

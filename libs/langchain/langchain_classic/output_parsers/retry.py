from __future__ import annotations

from typing import Annotated, Any, TypeVar

from langchain_core.exceptions import OutputParserException
from langchain_core.language_models import BaseLanguageModel
from langchain_core.output_parsers import BaseOutputParser, StrOutputParser
from langchain_core.prompt_values import PromptValue
from langchain_core.prompts import BasePromptTemplate, PromptTemplate
from langchain_core.runnables import RunnableSerializable
from pydantic import SkipValidation
from typing_extensions import TypedDict, override

NAIVE_COMPLETION_RETRY = """Prompt:
{prompt}
Completion:
{completion}

Above, the Completion did not satisfy the constraints given in the Prompt.
Please try again:"""

NAIVE_COMPLETION_RETRY_WITH_ERROR = """Prompt:
{prompt}
Completion:
{completion}

Above, the Completion did not satisfy the constraints given in the Prompt.
Details: {error}
Please try again:"""

NAIVE_RETRY_PROMPT = PromptTemplate.from_template(NAIVE_COMPLETION_RETRY)
NAIVE_RETRY_WITH_ERROR_PROMPT = PromptTemplate.from_template(
    NAIVE_COMPLETION_RETRY_WITH_ERROR,
)

T = TypeVar("T")


class RetryOutputParserRetryChainInput(TypedDict):
    """重试链输入，用于 RetryOutputParser。"""

    prompt: str
    completion: str


class RetryWithErrorOutputParserRetryChainInput(TypedDict):
    """重试链输入，用于 RetryWithErrorOutputParser。"""

    prompt: str
    completion: str
    error: str


class RetryOutputParser(BaseOutputParser[T]):
    """包装一个解析器，并尝试修复解析错误。

    通过将原始提示词和完成内容传递给另一个
    大语言模型，并告知该完成内容未满足提示词中的标准来实现。
    """

    parser: Annotated[BaseOutputParser[T], SkipValidation()]
    """用于解析输出的解析器。"""
        # 应为 LLMChain，但为避免顶层导入
        # langchain_classic.chains
    retry_chain: Annotated[
        RunnableSerializable[RetryOutputParserRetryChainInput, str] | Any,
        SkipValidation(),
    ]
    """用于重试完成内容的 RunnableSerializable（旧版：LLMChain）。"""
    max_retries: int = 1
    """解析失败时最多重试次数。"""
    legacy: bool = True
    """是否使用 retry_chain 的 run 或 arun 方法。"""

    @classmethod
    def from_llm(
        cls,
        llm: BaseLanguageModel,
        parser: BaseOutputParser[T],
        prompt: BasePromptTemplate = NAIVE_RETRY_PROMPT,
        max_retries: int = 1,
    ) -> RetryOutputParser[T]:
        """从语言模型和解析器创建 RetryOutputParser。

        Args:
            llm: 用于修复的模型
            parser: 用于解析的解析器
            prompt: 用于修复的提示词
            max_retries: 最大重试次数以进行解析。

        Returns:
            RetryOutputParser
        """
        chain = prompt | llm | StrOutputParser()
        return cls(parser=parser, retry_chain=chain, max_retries=max_retries)

    def parse_with_prompt(self, completion: str, prompt_value: PromptValue) -> T:
        """使用包装的解析器解析 LLM 调用的输出。

        Args:
            completion: 需要解析的链完成内容。
            prompt_value: 用于解析完成内容的提示词。

        Returns:
            解析后的完成内容。
        """
        retries = 0

        while retries <= self.max_retries:
            try:
                return self.parser.parse(completion)
            except OutputParserException:
                if retries == self.max_retries:
                    raise
                retries += 1
                if self.legacy and hasattr(self.retry_chain, "run"):
                    completion = self.retry_chain.run(
                        prompt=prompt_value.to_string(),
                        completion=completion,
                    )
                else:
                    completion = self.retry_chain.invoke(
                        {
                            "prompt": prompt_value.to_string(),
                            "completion": completion,
                        },
                    )

        msg = "Failed to parse"
        raise OutputParserException(msg)

    async def aparse_with_prompt(self, completion: str, prompt_value: PromptValue) -> T:
        """使用包装的解析器解析 LLM 调用的输出。

        Args:
            completion: 需要解析的链完成内容。
            prompt_value: 用于解析完成内容的提示词。

        Returns:
            解析后的完成内容。
        """
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
                        prompt=prompt_value.to_string(),
                        completion=completion,
                        error=repr(e),
                    )
                else:
                    completion = await self.retry_chain.ainvoke(
                        {
                            "prompt": prompt_value.to_string(),
                            "completion": completion,
                        },
                    )

        msg = "Failed to parse"
        raise OutputParserException(msg)

    @override
    def parse(self, completion: str) -> T:
        msg = "This OutputParser can only be called by the `parse_with_prompt` method."
        raise NotImplementedError(msg)

    @override
    def get_format_instructions(self) -> str:
        return self.parser.get_format_instructions()

    @property
    def _type(self) -> str:
        return "retry"

    @property
    @override
    def OutputType(self) -> type[T]:
        return self.parser.OutputType


class RetryWithErrorOutputParser(BaseOutputParser[T]):
    """包装一个解析器，并尝试修复解析错误。

    Does this by passing the original prompt, the completion, AND the error
    that was raised to another language model and telling it that the completion
    did not work, and raised the given error. Differs from RetryOutputParser
    in that this implementation provides the error that was raised back to the
    LLM, which in theory should give it more information on how to fix it.
    """

    parser: Annotated[BaseOutputParser[T], SkipValidation()]
    """用于解析输出的解析器。"""
        # 应为 LLMChain，但为避免顶层导入
        # langchain_classic.chains
    retry_chain: Annotated[
        RunnableSerializable[RetryWithErrorOutputParserRetryChainInput, str] | Any,
        SkipValidation(),
    ]
    """用于重试完成内容的 RunnableSerializable（旧版：LLMChain）。"""
    max_retries: int = 1
    """解析失败时最多重试次数。"""
    legacy: bool = True
    """是否使用 retry_chain 的 run 或 arun 方法。"""

    @classmethod
    def from_llm(
        cls,
        llm: BaseLanguageModel,
        parser: BaseOutputParser[T],
        prompt: BasePromptTemplate = NAIVE_RETRY_WITH_ERROR_PROMPT,
        max_retries: int = 1,
    ) -> RetryWithErrorOutputParser[T]:
        """从语言模型创建 RetryWithErrorOutputParser。

        Args:
            llm: 用于重试完成内容的语言模型。
            parser: 用于解析输出的解析器。
            prompt: 用于重试完成内容的提示词。
            max_retries: 重试完成内容的最大次数。

        Returns:
            一个 RetryWithErrorOutputParser。
        """
        chain = prompt | llm | StrOutputParser()
        return cls(parser=parser, retry_chain=chain, max_retries=max_retries)

    @override
    def parse_with_prompt(self, completion: str, prompt_value: PromptValue) -> T:
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
                        prompt=prompt_value.to_string(),
                        completion=completion,
                        error=repr(e),
                    )
                else:
                    completion = self.retry_chain.invoke(
                        {
                            "completion": completion,
                            "prompt": prompt_value.to_string(),
                            "error": repr(e),
                        },
                    )

        msg = "Failed to parse"
        raise OutputParserException(msg)

    async def aparse_with_prompt(self, completion: str, prompt_value: PromptValue) -> T:
        """使用包装的解析器解析 LLM 调用的输出。

        Args:
            completion: 需要解析的链完成内容。
            prompt_value: 用于解析完成内容的提示词。

        Returns:
            解析后的完成内容。
        """
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
                        prompt=prompt_value.to_string(),
                        completion=completion,
                        error=repr(e),
                    )
                else:
                    completion = await self.retry_chain.ainvoke(
                        {
                            "prompt": prompt_value.to_string(),
                            "completion": completion,
                            "error": repr(e),
                        },
                    )

        msg = "Failed to parse"
        raise OutputParserException(msg)

    @override
    def parse(self, completion: str) -> T:
        msg = "This OutputParser can only be called by the `parse_with_prompt` method."
        raise NotImplementedError(msg)

    @override
    def get_format_instructions(self) -> str:
        return self.parser.get_format_instructions()

    @property
    def _type(self) -> str:
        return "retry_with_error"

    @property
    @override
    def OutputType(self) -> type[T]:
        return self.parser.OutputType

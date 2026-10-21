"""语言模型输出的基类解析器。"""

from __future__ import annotations

import builtins
import contextlib
from abc import ABC, abstractmethod
from typing import (
    TYPE_CHECKING,
    Any,
    Generic,
    TypeVar,
    cast,
)

from typing_extensions import override

from langchain_core._api import deprecated
from langchain_core.language_models import LanguageModelOutput
from langchain_core.messages import AnyMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, Generation
from langchain_core.runnables import Runnable, RunnableConfig, RunnableSerializable
from langchain_core.runnables.config import run_in_executor

if TYPE_CHECKING:
    import builtins

    from langchain_core.prompt_values import PromptValue

T = TypeVar("T")
OutputParserLike = Runnable[LanguageModelOutput, T]


class BaseLLMOutputParser(ABC, Generic[T]):
    """用于解析模型输出的抽象基类。"""

    @abstractmethod
    def parse_result(self, result: list[Generation], *, partial: bool = False) -> T:
        """将一组候选模型 `Generation` 对象解析为特定格式。

        Args:
            result: 需要解析的 `Generation` 列表。

                The `Generation` objects are assumed to be different candidate outputs
                for a single model input.
            partial: Whether to parse the output as a partial result.

                这对能够解析部分结果的解析器很有用。

        Returns:
            结构化输出。
        """

    async def aparse_result(
        self, result: list[Generation], *, partial: bool = False
    ) -> T:
        """将一组候选模型 `Generation` 对象解析为特定格式。

        Args:
            result: 需要解析的 `Generation` 列表。

                The Generations are assumed to be different candidate outputs for a
                single model input.
            partial: Whether to parse the output as a partial result.

                这对能够解析部分结果的解析器很有用。

        Returns:
            结构化输出。
        """
        return await run_in_executor(None, self.parse_result, result, partial=partial)


class BaseGenerationOutputParser(
    BaseLLMOutputParser[T], RunnableSerializable[LanguageModelOutput, T]
):
    """用于解析 LLM 调用输出的基类。"""

    @property
    @override
    def InputType(self) -> Any:
        """返回解析器的输入类型。"""
        return str | AnyMessage

    @property
    @override
    def OutputType(self) -> type[T]:
        """返回解析器的输出类型。"""
                # 即使 mypy 报告此不合法，
                # 对 pydantic 来说已足够用于构建模式
        return cast("type[T]", T)  # type: ignore[misc]

    @override
    def invoke(
        self,
        input: str | BaseMessage,
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> T:
        if isinstance(input, BaseMessage):
            return self._call_with_config(
                lambda inner_input: self.parse_result(
                    [ChatGeneration(message=inner_input)]
                ),
                input,
                config,
                run_type="parser",
            )
        return self._call_with_config(
            lambda inner_input: self.parse_result([Generation(text=inner_input)]),
            input,
            config,
            run_type="parser",
        )

    @override
    async def ainvoke(
        self,
        input: str | BaseMessage,
        config: RunnableConfig | None = None,
        **kwargs: Any | None,
    ) -> T:
        if isinstance(input, BaseMessage):
            return await self._acall_with_config(
                lambda inner_input: self.aparse_result(
                    [ChatGeneration(message=inner_input)]
                ),
                input,
                config,
                run_type="parser",
            )
        return await self._acall_with_config(
            lambda inner_input: self.aparse_result([Generation(text=inner_input)]),
            input,
            config,
            run_type="parser",
        )


class BaseOutputParser(
    BaseLLMOutputParser[T], RunnableSerializable[LanguageModelOutput, T]
):
    """用于解析 LLM 调用输出的基类。

    输出解析器有助于结构化语言模型的响应。

    Example:
        ```python
        # Implement a simple boolean output parser


        class BooleanOutputParser(BaseOutputParser[bool]):
            true_val: str = "YES"
            false_val: str = "NO"

            def parse(self, text: str) -> bool:
                cleaned_text = text.strip().upper()
                if cleaned_text not in (
                    self.true_val.upper(),
                    self.false_val.upper(),
                ):
                    raise OutputParserException(
                        f"BooleanOutputParser expected output value to either be "
                        f"{self.true_val} or {self.false_val} (case-insensitive). "
                        f"Received {cleaned_text}."
                    )
                return cleaned_text == self.true_val.upper()

            @property
            def _type(self) -> str:
                return "boolean_output_parser"
        ```
    """

    @property
    @override
    def InputType(self) -> Any:
        """返回解析器的输入类型。"""
        return str | AnyMessage

    @property
    @override
    def OutputType(self) -> type[T]:
        """返回解析器的输出类型。

        此属性从类的第一个类型参数推断得出。

        Raises:
            TypeError: 如果类无法推断出 `OutputType`。
        """
        for base in self.__class__.mro():
            if hasattr(base, "__pydantic_generic_metadata__"):
                metadata = base.__pydantic_generic_metadata__
                if "args" in metadata and len(metadata["args"]) > 0:
                    return cast("type[T]", metadata["args"][0])

        msg = (
            f"Runnable {self.__class__.__name__} doesn't have an inferable OutputType. "
            "Override the OutputType property to specify the output type."
        )
        raise TypeError(msg)

    @override
    def invoke(
        self,
        input: str | BaseMessage,
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> T:
        if isinstance(input, BaseMessage):
            return self._call_with_config(
                lambda inner_input: self.parse_result(
                    [ChatGeneration(message=inner_input)]
                ),
                input,
                config,
                run_type="parser",
            )
        return self._call_with_config(
            lambda inner_input: self.parse_result([Generation(text=inner_input)]),
            input,
            config,
            run_type="parser",
        )

    @override
    async def ainvoke(
        self,
        input: str | BaseMessage,
        config: RunnableConfig | None = None,
        **kwargs: Any | None,
    ) -> T:
        if isinstance(input, BaseMessage):
            return await self._acall_with_config(
                lambda inner_input: self.aparse_result(
                    [ChatGeneration(message=inner_input)]
                ),
                input,
                config,
                run_type="parser",
            )
        return await self._acall_with_config(
            lambda inner_input: self.aparse_result([Generation(text=inner_input)]),
            input,
            config,
            run_type="parser",
        )

    @override
    def parse_result(self, result: list[Generation], *, partial: bool = False) -> T:
        """将一组候选模型 `Generation` 对象解析为特定格式。

        The return value is parsed from only the first `Generation` in the result, which
        is assumed to be the highest-likelihood `Generation`.

        Args:
            result: 需要解析的 `Generation` 列表。

                The `Generation` objects are assumed to be different candidate outputs
                for a single model input.
            partial: Whether to parse the output as a partial result.

                这对能够解析部分结果的解析器很有用。

        Returns:
            结构化输出。
        """
        return self.parse(result[0].text)

    @abstractmethod
    def parse(self, text: str) -> T:
        """将单个字符串形式的模型输出解析为某种结构。

        Args:
            text: 语言模型的字符串输出。

        Returns:
            结构化输出。
        """

    async def aparse_result(
        self, result: list[Generation], *, partial: bool = False
    ) -> T:
        """将一组候选模型 `Generation` 对象解析为特定格式。

        The return value is parsed from only the first `Generation` in the result, which
        is assumed to be the highest-likelihood `Generation`.

        Args:
            result: 需要解析的 `Generation` 列表。

                The `Generation` objects are assumed to be different candidate outputs
                for a single model input.
            partial: Whether to parse the output as a partial result.

                这对能够解析部分结果的解析器很有用。

        Returns:
            结构化输出。
        """
        return await run_in_executor(None, self.parse_result, result, partial=partial)

    async def aparse(self, text: str) -> T:
        """异步解析单个字符串形式的模型输出为某种结构。

        Args:
            text: 语言模型的字符串输出。

        Returns:
            结构化输出。
        """
        return await run_in_executor(None, self.parse, text)

        # TODO: 将 'completion' 重命名为 'text'。
    def parse_with_prompt(
        self,
        completion: str,
        prompt: PromptValue,  # noqa: ARG002
    ) -> Any:
        """使用输入提示词作为上下文，解析 LLM 调用的输出。

        The prompt is largely provided in the event the `OutputParser` wants to retry or
        fix the output in some way, and needs information from the prompt to do so.

        Args:
            completion: 语言模型的字符串输出。
            prompt: 输入的 `PromptValue`。

        Returns:
            结构化输出。
        """
        return self.parse(completion)

    def get_format_instructions(self) -> str:
        """关于如何格式化 LLM 输出的说明。"""
        raise NotImplementedError

    @property
    def _type(self) -> str:
        """返回用于序列化的输出解析器类型。"""
        msg = (
            f"_type property is not implemented in class {self.__class__.__name__}."
            " This is required for serialization."
        )
        raise NotImplementedError(msg)

    @deprecated("1.4.2", alternative="asdict", removal="2.0.0")
    @override
    def dict(self, **kwargs: Any) -> builtins.dict[str, Any]:
        """已弃用 - 请改用 `asdict()`。

        返回输出解析器的字典表示形式。
        """
        return self.asdict(**kwargs)

    def asdict(self, **kwargs: Any) -> builtins.dict[str, Any]:
        """返回输出解析器的字典表示形式。"""
        output_parser_dict = super().model_dump(**kwargs)
        with contextlib.suppress(NotImplementedError):
            output_parser_dict["_type"] = self._type
        return output_parser_dict

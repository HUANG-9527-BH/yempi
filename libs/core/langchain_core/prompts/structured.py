"""面向语言模型的结构化提示词模板。"""

from collections.abc import (
    AsyncIterator,
    Awaitable,
    Callable,
    Iterator,
    Mapping,
    Sequence,
)
from typing import (
    Any,
    overload,
)

from pydantic import BaseModel, Field
from typing_extensions import override

from langchain_core._api.beta_decorator import beta
from langchain_core.language_models.base import BaseLanguageModel
from langchain_core.prompt_values import PromptValue
from langchain_core.prompts.chat import (
    ChatPromptTemplate,
    MessageLikeRepresentation,
)
from langchain_core.prompts.string import PromptTemplateFormat
from langchain_core.runnables.base import (
    Other,
    Runnable,
    RunnableSequence,
    RunnableSerializable,
)
from langchain_core.utils import get_pydantic_field_names


@beta()
class StructuredPrompt(ChatPromptTemplate):
    """面向语言模型的结构化提示词模板。"""

    schema_: dict[str, Any] | type
    """结构化提示词的模式。"""

    structured_output_kwargs: dict[str, Any] = Field(default_factory=dict)

    def __init__(
        self,
        messages: Sequence[MessageLikeRepresentation],
        schema_: dict[str, Any] | type[BaseModel] | None = None,
        *,
        structured_output_kwargs: dict[str, Any] | None = None,
        template_format: PromptTemplateFormat = "f-string",
        **kwargs: Any,
    ) -> None:
        """创建一个结构化提示词模板。

        Args:
            messages: 消息序列。
            schema_: 结构化提示词的模式。
            structured_output_kwargs: 用于结构化输出的额外参数。
            template_format: 提示词的模板格式。

        Raises:
            ValueError: 如果未提供模式。
        """
        schema_ = schema_ or kwargs.pop("schema", None)
        if not schema_:
            err_msg = (
                "Must pass in a non-empty structured output schema. Received: "
                f"{schema_}"
            )
            raise ValueError(err_msg)
                # 避免在合并额外参数时修改调用方提供的字典。
        structured_output_kwargs = dict(structured_output_kwargs or {})
        for k in set(kwargs).difference(get_pydantic_field_names(self.__class__)):
            structured_output_kwargs[k] = kwargs.pop(k)
        super().__init__(
            messages=messages,
            schema_=schema_,
            structured_output_kwargs=structured_output_kwargs,
            template_format=template_format,
            **kwargs,
        )

    @classmethod
    def get_lc_namespace(cls) -> list[str]:
        """获取 LangChain 对象的命名空间。

        For example, if the class is `langchain.llms.openai.OpenAI`, then the namespace
        is `["langchain", "llms", "openai"]`

        Returns:
            LangChain 对象的命名空间。
        """
        return cls.__module__.split(".")

    @classmethod
    def from_messages_and_schema(
        cls,
        messages: Sequence[MessageLikeRepresentation],
        schema: dict[str, Any] | type,
        **kwargs: Any,
    ) -> ChatPromptTemplate:
        """从多种消息格式创建聊天提示词模板。

        Examples:
            从消息模板列表实例化：

            ```python
            from langchain_core.prompts import StructuredPrompt


            class OutputSchema(BaseModel):
                name: str
                value: int


            template = StructuredPrompt(
                [
                    ("human", "Hello, how are you?"),
                    ("ai", "I'm doing well, thanks!"),
                    ("human", "That's good to hear."),
                ],
                OutputSchema,
            )
            ```

        Args:
            messages: 消息表示的序列。

                消息可使用以下格式表示：

                1. `BaseMessagePromptTemplate`
                2. `BaseMessage`
                3. 2 元组形式 `(message type, template)`；例如，
                    `("human", "{user_input}")`
                4. 2 元组形式 `(message class, template)`
                5. 字符串，作为 `("human", template)` 的简写；例如，
                    `"{user_input}"`
            schema: 函数调用的字典表示，或 Pydantic 模型。
            **kwargs: 传递给
                `ChatModel.with_structured_output(schema, **kwargs)` 的任意额外参数。

        Returns:
            一个结构化提示词模板
        """
        return cls(messages, schema, **kwargs)

    @overload
    def __or__(
        self, other: Mapping[str, Any]
    ) -> RunnableSerializable[dict[str, Any], dict[str, Any]]: ...

    @overload
    def __or__(
        self,
        other: Callable[[PromptValue], Runnable[PromptValue, Other]]
        | Callable[[PromptValue], Awaitable[Runnable[PromptValue, Other]]],
    ) -> RunnableSerializable[dict[str, Any], Other]: ...

    @overload
    def __or__(
        self,
        other: Runnable[PromptValue, Other]
        | Callable[[Iterator[PromptValue]], Iterator[Other]]
        | Callable[[AsyncIterator[PromptValue]], AsyncIterator[Other]]
        | Callable[[PromptValue], Other],
    ) -> RunnableSerializable[dict[str, Any], Other]: ...

    @override
    def __or__(
        self,
        other: Runnable[PromptValue, Other]
        | Callable[[Iterator[PromptValue]], Iterator[Other]]
        | Callable[[AsyncIterator[PromptValue]], AsyncIterator[Other]]
        | Callable[[PromptValue], Other]
        | Mapping[str, Runnable[PromptValue, Any] | Callable[[PromptValue], Any] | Any],
    ) -> RunnableSerializable[dict[str, Any], Any]:
        return self.pipe(other)

    def pipe(
        self,
        *others: Runnable[Any, Other]
        | Callable[[Iterator[Any]], Iterator[Other]]
        | Callable[[AsyncIterator[Any]], AsyncIterator[Other]]
        | Callable[[Any], Other]
        | Mapping[str, Runnable[Any, Other] | Callable[[Any], Other] | Any],
        name: str | None = None,
    ) -> RunnableSerializable[dict[str, Any], Other]:
        """将结构化提示词传递给语言模型。

        Args:
            others: 要将结构化提示词传递的语言模型。
            name: 管道的名称。

        Returns:
            一个 `RunnableSequence` 对象。

        Raises:
            NotImplementedError: 如果 `others` 的第一个元素不是语言模型。
        """
        if (others and isinstance(others[0], BaseLanguageModel)) or hasattr(
            others[0], "with_structured_output"
        ):
            return RunnableSequence(
                self,
                others[0].with_structured_output(
                    self.schema_, **self.structured_output_kwargs
                ),
                *others[1:],
                name=name,
            )
        msg = "Structured prompts need to be piped to a language model."
        raise NotImplementedError(msg)

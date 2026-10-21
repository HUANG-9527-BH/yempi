"""包含少量示例的提示词模板。"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
)
from typing_extensions import override

from langchain_core._api import deprecated
from langchain_core.example_selectors import BaseExampleSelector
from langchain_core.messages import BaseMessage, get_buffer_string
from langchain_core.prompts.chat import BaseChatPromptTemplate
from langchain_core.prompts.message import BaseMessagePromptTemplate
from langchain_core.prompts.prompt import PromptTemplate
from langchain_core.prompts.string import (
    DEFAULT_FORMATTER_MAPPING,
    StringPromptTemplate,
    check_valid_template,
    get_template_variables,
)

if TYPE_CHECKING:
    from pathlib import Path

    from typing_extensions import Self


class _FewShotPromptTemplateMixin(BaseModel):
    """包含少量示例的提示词模板。"""

    examples: list[dict[str, Any]] | None = None
    """用于格式化到提示词中的示例。

    二者中必须提供其一：`examples` 或 `example_selector`。
    """

    example_selector: BaseExampleSelector | None = None
    """`ExampleSelector`，用于从示例中选择要格式化的部分。

    二者中必须提供其一：`examples` 或 `example_selector`。
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="forbid",
    )

    @model_validator(mode="before")
    @classmethod
    def check_examples_and_selector(cls, values: dict[str, Any]) -> Any:
        """检查 `examples` 和 `example_selector` 中仅提供其一。

        Args:
            values: 需要检查的值。

        Returns:
            如果值有效，则返回这些值。

        Raises:
            ValueError: 如果既未提供 `examples` 也未提供 `example_selector`。
            ValueError: 如果同时提供了 `examples` 和 `example_selector`。
        """
        examples = values.get("examples")
        example_selector = values.get("example_selector")
        if examples and example_selector:
            msg = "Only one of 'examples' and 'example_selector' should be provided"
            raise ValueError(msg)

        if examples is None and example_selector is None:
            msg = "One of 'examples' and 'example_selector' should be provided"
            raise ValueError(msg)

        return values

    def _get_examples(self, **kwargs: Any) -> list[dict[str, Any]]:
        """获取用于格式化提示词的示例。

        Args:
            **kwargs: 传递给示例选择器的关键字参数。

        Returns:
            示例列表。

        Raises:
            ValueError: 如果既未提供 `examples` 也未提供 `example_selector`。
        """
        if self.examples is not None:
            return self.examples
        if self.example_selector is not None:
            return self.example_selector.select_examples(kwargs)
        msg = "One of 'examples' and 'example_selector' should be provided"
        raise ValueError(msg)

    async def _aget_examples(self, **kwargs: Any) -> list[dict[str, Any]]:
        """异步获取用于格式化提示词的示例。

        Args:
            **kwargs: 传递给示例选择器的关键字参数。

        Returns:
            示例列表。

        Raises:
            ValueError: 如果既未提供 `examples` 也未提供 `example_selector`。
        """
        if self.examples is not None:
            return self.examples
        if self.example_selector is not None:
            return await self.example_selector.aselect_examples(kwargs)
        msg = "One of 'examples' and 'example_selector' should be provided"
        raise ValueError(msg)


class FewShotPromptTemplate(_FewShotPromptTemplateMixin, StringPromptTemplate):
    """包含少量示例的提示词模板。"""

    @classmethod
    def is_lc_serializable(cls) -> bool:
        """返回 `False`，因为此类不可序列化。"""
        return False

    validate_template: bool = False
    """是否尝试验证模板。"""

    example_prompt: PromptTemplate
    """`PromptTemplate`，用于格式化单个示例。"""

    suffix: str
    """在示例之后插入的提示词模板字符串。"""

    example_separator: str = "\n\n"
    """用于连接前缀、示例和后缀的字符串分隔符。"""

    prefix: str = ""
    """在示例之前插入的提示词模板字符串。"""

    template_format: Literal["f-string", "jinja2"] = "f-string"
    """提示词模板的格式。

    Options are: 可选项：`'f-string'`、`'jinja2'`。
    """

    def __init__(self, **kwargs: Any) -> None:
        """初始化少量示例提示词模板。"""
        if "input_variables" not in kwargs and "example_prompt" in kwargs:
            kwargs["input_variables"] = kwargs["example_prompt"].input_variables
        super().__init__(**kwargs)

    @model_validator(mode="after")
    def template_is_valid(self) -> Self:
        """检查前缀、后缀和输入变量的一致性。"""
        if self.validate_template:
            check_valid_template(
                self.prefix + self.suffix,
                self.template_format,
                self.input_variables + list(self.partial_variables),
            )
        elif self.template_format:
            self.input_variables = [
                var
                for var in get_template_variables(
                    self.prefix + self.suffix, self.template_format
                )
                if var not in self.partial_variables
            ]
        return self

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="forbid",
    )

    def format(self, **kwargs: Any) -> str:
        """使用输入格式化提示词并生成字符串。

        使用此方法生成提示词的字符串表示形式。

        Args:
            **kwargs: 用于格式化的关键字参数。

        Returns:
            提示词的字符串表示形式。
        """
        kwargs = self._merge_partial_and_user_variables(**kwargs)
                # 获取要使用的示例。
        examples = self._get_examples(**kwargs)
        examples = [
            {k: e[k] for k in self.example_prompt.input_variables} for e in examples
        ]
                # 格式化示例。
        example_strings = [
            self.example_prompt.format(**example) for example in examples
        ]
                # 创建整体模板。
        pieces = [self.prefix, *example_strings, self.suffix]
        template = self.example_separator.join([piece for piece in pieces if piece])

                # 使用输入变量格式化模板。
        return DEFAULT_FORMATTER_MAPPING[self.template_format](template, **kwargs)

    async def aformat(self, **kwargs: Any) -> str:
        """异步格式化提示词并生成字符串。

        使用此方法生成提示词的字符串表示形式。

        Args:
            **kwargs: 用于格式化的关键字参数。

        Returns:
            提示词的字符串表示形式。
        """
        kwargs = self._merge_partial_and_user_variables(**kwargs)
                # 获取要使用的示例。
        examples = await self._aget_examples(**kwargs)
        examples = [
            {k: e[k] for k in self.example_prompt.input_variables} for e in examples
        ]
                # 格式化示例。
        example_strings = [
            await self.example_prompt.aformat(**example) for example in examples
        ]
                # 创建整体模板。
        pieces = [self.prefix, *example_strings, self.suffix]
        template = self.example_separator.join([piece for piece in pieces if piece])

                # 使用输入变量格式化模板。
        return DEFAULT_FORMATTER_MAPPING[self.template_format](template, **kwargs)

    @property
    def _prompt_type(self) -> str:
        """返回提示词类型键。"""
        return "few_shot"

    @deprecated(
        since="1.2.21",
        removal="2.0.0",
        alternative="Use `dumpd`/`dumps` from `langchain_core.load` to serialize "
        "prompts and `load`/`loads` to deserialize them.",
    )
    def save(self, file_path: Path | str) -> None:
        """将提示词模板保存到文件。

        Args:
            file_path: 保存提示词模板的路径。

        Raises:
            ValueError: 如果提供了 `example_selector`。
        """
        if self.example_selector:
            msg = "Saving an example selector is not currently supported"
            raise ValueError(msg)
        return super().save(file_path)


class FewShotChatMessagePromptTemplate(
    BaseChatPromptTemplate, _FewShotPromptTemplateMixin
):
    """支持少量示例的聊天提示词模板。

    此提示词模板生成的结构是一个消息列表，包含前缀消息、示例消息和后缀消息。

    该结构可用于创建带有中间示例的对话，例如：

    ```txt
    System: You are a helpful AI Assistant

    Human: What is 2+2?

    AI: 4

    Human: What is 2+3?

    AI: 5

    Human: What is 4+4?
    ```

    此提示词模板可用于生成固定示例列表，或根据输入动态选择示例。

    Examples:
        Prompt template with a fixed list of examples (matching the sample
        conversation above):

        ```python
        from langchain_core.prompts import (
            FewShotChatMessagePromptTemplate,
            ChatPromptTemplate,
        )

        examples = [
            {"input": "2+2", "output": "4"},
            {"input": "2+3", "output": "5"},
        ]

        example_prompt = ChatPromptTemplate.from_messages(
            [
                ("human", "What is {input}?"),
                ("ai", "{output}"),
            ]
        )

        few_shot_prompt = FewShotChatMessagePromptTemplate(
            examples=examples,
            # This is a prompt template used to format each individual example.
            example_prompt=example_prompt,
        )

        final_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "You are a helpful AI Assistant"),
                few_shot_prompt,
                ("human", "{input}"),
            ]
        )
        final_prompt.format(input="What is 4+4?")
        ```

        使用动态选择示例的提示词模板：

        ```python
        from langchain_core.prompts import SemanticSimilarityExampleSelector
        from langchain_core.embeddings import OpenAIEmbeddings
        from langchain_core.vectorstores import Chroma

        examples = [
            {"input": "2+2", "output": "4"},
            {"input": "2+3", "output": "5"},
            {"input": "2+4", "output": "6"},
            # ...
        ]

        to_vectorize = [" ".join(example.values()) for example in examples]
        embeddings = OpenAIEmbeddings()
        vectorstore = Chroma.from_texts(to_vectorize, embeddings, metadatas=examples)
        example_selector = SemanticSimilarityExampleSelector(vectorstore=vectorstore)

        from langchain_core import SystemMessage
        from langchain_core.prompts import HumanMessagePromptTemplate
        from langchain_core.prompts.few_shot import FewShotChatMessagePromptTemplate

        few_shot_prompt = FewShotChatMessagePromptTemplate(
            # Which variable(s) will be passed to the example selector.
            input_variables=["input"],
            example_selector=example_selector,
            # Define how each example will be formatted.
            # In this case, each example will become 2 messages:
            # 1 human, and 1 AI
            example_prompt=(
                HumanMessagePromptTemplate.from_template("{input}")
                + AIMessagePromptTemplate.from_template("{output}")
            ),
        )
        # Define the overall prompt.
        final_prompt = (
            SystemMessagePromptTemplate.from_template("You are a helpful AI Assistant")
            + few_shot_prompt
            + HumanMessagePromptTemplate.from_template("{input}")
        )
        # Show the prompt
        print(final_prompt.format_messages(input="What's 3+3?"))  # noqa: T201

        # Use within an LLM
        from langchain_anthropic import ChatAnthropic

        chain = final_prompt | ChatAnthropic(model="claude-sonnet-5")
        chain.invoke({"input": "What's 3+3?"})
        ```
    """

    input_variables: list[str] = Field(default_factory=list)
    """A list of the names of the variables the prompt template will use to pass to
    the `example_selector`, if provided.
    """

    example_prompt: BaseMessagePromptTemplate | BaseChatPromptTemplate
    """用于格式化每个示例的类。"""

    @classmethod
    def is_lc_serializable(cls) -> bool:
        """返回 `False`，因为此类不可序列化。"""
        return False

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="forbid",
    )

    def format_messages(self, **kwargs: Any) -> list[BaseMessage]:
        """将关键字参数格式化为消息列表。

        Args:
            **kwargs: 用于填充消息中模板的关键词参数。

        Returns:
            所有模板变量填充后的格式化消息列表。
        """
                # 获取要使用的示例。
        examples = self._get_examples(**kwargs)
        examples = [
            {k: e[k] for k in self.example_prompt.input_variables} for e in examples
        ]
                # 格式化示例。
        return [
            message
            for example in examples
            for message in self.example_prompt.format_messages(**example)
        ]

    async def aformat_messages(self, **kwargs: Any) -> list[BaseMessage]:
        """异步将关键字参数格式化为消息列表。

        Args:
            **kwargs: 用于填充消息中模板的关键词参数。

        Returns:
            所有模板变量填充后的格式化消息列表。
        """
                # 获取要使用的示例。
        examples = await self._aget_examples(**kwargs)
        examples = [
            {k: e[k] for k in self.example_prompt.input_variables} for e in examples
        ]
                # 格式化示例。
        return [
            message
            for example in examples
            for message in await self.example_prompt.aformat_messages(**example)
        ]

    def format(self, **kwargs: Any) -> str:
        """使用输入格式化提示词，生成字符串。

        使用此方法生成由聊天消息组成的提示词的字符串表示形式。

        适用于传入基于字符串的补全语言模型或调试。

        Args:
            **kwargs: 用于格式化的关键词参数。

        Returns:
            提示词的字符串表示形式
        """
        messages = self.format_messages(**kwargs)
        return get_buffer_string(messages)

    async def aformat(self, **kwargs: Any) -> str:
        """异步使用输入格式化提示词，生成字符串。

        使用此方法生成由聊天消息组成的提示词的字符串表示形式。

        适用于传入基于字符串的补全语言模型或调试。

        Args:
            **kwargs: 用于格式化的关键词参数。

        Returns:
            提示词的字符串表示形式
        """
        messages = await self.aformat_messages(**kwargs)
        return get_buffer_string(messages)

    @override
    def pretty_repr(self, html: bool = False) -> str:
        """返回提示词模板的美化表示形式。

        Args:
            html: 是否返回 HTML 格式的字符串。

        Returns:
            提示词模板的美化表示形式。
        """
        raise NotImplementedError

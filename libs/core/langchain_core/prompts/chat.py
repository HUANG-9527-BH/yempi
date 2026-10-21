"""聊天提示词模板。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from pathlib import Path
from typing import (
    Annotated,
    Any,
    TypedDict,
    TypeVar,
    cast,
    overload,
)

from pydantic import (
    Field,
    PositiveInt,
    SkipValidation,
    model_validator,
)
from typing_extensions import Self, override

from langchain_core._api import deprecated
from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    BaseMessage,
    ChatMessage,
    HumanMessage,
    SystemMessage,
    convert_to_messages,
)
from langchain_core.messages.base import get_msg_title_repr
from langchain_core.prompt_values import ChatPromptValue
from langchain_core.prompts.base import BasePromptTemplate
from langchain_core.prompts.dict import DictPromptTemplate
from langchain_core.prompts.image import ImagePromptTemplate
from langchain_core.prompts.message import (
    BaseMessagePromptTemplate,
)
from langchain_core.prompts.prompt import PromptTemplate
from langchain_core.prompts.string import (
    PromptTemplateFormat,
    StringPromptTemplate,
    get_template_variables,
)
from langchain_core.utils import get_colored_text
from langchain_core.utils.interactive_env import is_interactive_env


class MessagesPlaceholder(BaseMessagePromptTemplate):
    """假设变量已经是消息列表的提示词模板。

    可用于传入消息列表的占位符。

    !!! example "直接使用"

        ```python
        from langchain_core.prompts import MessagesPlaceholder

        prompt = MessagesPlaceholder("history")
        prompt.format_messages()  # raises KeyError

        prompt = MessagesPlaceholder("history", optional=True)
        prompt.format_messages()  # returns empty list []

        prompt.format_messages(
            history=[
                ("system", "You are an AI assistant."),
                ("human", "Hello!"),
            ]
        )
        # -> [
        #     SystemMessage(content="You are an AI assistant."),
        #     HumanMessage(content="Hello!"),
        # ]
        ```

    !!! example "构建包含聊天历史的提示词"

        ```python
        from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "You are a helpful assistant."),
                MessagesPlaceholder("history"),
                ("human", "{question}"),
            ]
        )
        prompt.invoke(
            {
                "history": [("human", "what's 5 + 2"), ("ai", "5 + 2 is 7")],
                "question": "now multiply that by 4",
            }
        )
        # -> ChatPromptValue(messages=[
        #     SystemMessage(content="You are a helpful assistant."),
        #     HumanMessage(content="what's 5 + 2"),
        #     AIMessage(content="5 + 2 is 7"),
        #     HumanMessage(content="now multiply that by 4"),
        # ])
        ```

    !!! example "限制消息数量"

        ```python
        from langchain_core.prompts import MessagesPlaceholder

        prompt = MessagesPlaceholder("history", n_messages=1)

        prompt.format_messages(
            history=[
                ("system", "You are an AI assistant."),
                ("human", "Hello!"),
            ]
        )
        # -> [
        #     HumanMessage(content="Hello!"),
        # ]
        ```
    """

    variable_name: str
    """用作消息的变量名称。"""

    optional: bool = False
    """是否必须提供 `format_messages`。

    If `True` `format_messages` can be called with no arguments and will return an empty
    list.

    If `False` then a named argument with name `variable_name` must be passed in, even
    if the value is an empty list.
    """

    n_messages: PositiveInt | None = None
    """要包含的最大消息数。

    如果为 `None`，则包含所有消息。
    """

    def __init__(
        self, variable_name: str, *, optional: bool = False, **kwargs: Any
    ) -> None:
        """创建一个消息占位符。

        Args:
            variable_name: 用作消息的变量名称。
            optional: `format_messages` 是否必须提供。

                If `True` format_messages can be called with no arguments and will
                return an empty list.

                If `False` then a named argument with name `variable_name` must be
                passed in, even if the value is an empty list.
        """
                # mypy 无法检测到在父类中定义的初始化方法
                # 因为这些是 BaseModel 类。
        super().__init__(variable_name=variable_name, optional=optional, **kwargs)  # type: ignore[call-arg,unused-ignore]

    def format_messages(self, **kwargs: Any) -> list[BaseMessage]:
        """从 kwargs 中格式化消息。

        Args:
            kwargs: 用于格式化的关键字参数。

        Returns:
            BaseMessage 对象的列表。

        Raises:
            ValueError: 如果变量不是消息列表。
        """
        value = (
            kwargs.get(self.variable_name, [])
            if self.optional
            else kwargs[self.variable_name]
        )
        if not isinstance(value, list):
            msg = (
                f"variable {self.variable_name} should be a list of base messages, "
                f"got {value} of type {type(value)}"
            )
            raise ValueError(msg)  # noqa: TRY004
        value = convert_to_messages(value)
        if self.n_messages:
            value = value[-self.n_messages :]
        return value

    @property
    def input_variables(self) -> list[str]:
        """此提示词模板的输入变量。

        Returns:
            输入变量名称的列表。
        """
        return [self.variable_name] if not self.optional else []

    @override
    def pretty_repr(self, html: bool = False) -> str:
        """人类可读的表示形式。

        Args:
            html: 是否以 HTML 格式化。

        Returns:
            人类可读的表示形式。
        """
        var = "{" + self.variable_name + "}"
        if html:
            title = get_msg_title_repr("Messages Placeholder", bold=True)
            var = get_colored_text(var, "yellow")
        else:
            title = get_msg_title_repr("Messages Placeholder")
        return f"{title}\n\n{var}"


MessagePromptTemplateT = TypeVar(
    "MessagePromptTemplateT", bound="BaseStringMessagePromptTemplate"
)
"""Type variable for message prompt templates."""


class BaseStringMessagePromptTemplate(BaseMessagePromptTemplate, ABC):
    """使用字符串提示词模板的消息提示词模板基类。"""

    prompt: StringPromptTemplate
    """字符串提示词模板。"""

    additional_kwargs: dict[str, Any] = Field(default_factory=dict)
    """传递给提示词模板的额外关键字参数。"""

    @classmethod
    def from_template(
        cls,
        template: str,
        template_format: PromptTemplateFormat = "f-string",
        partial_variables: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Self:
        """从字符串模板创建一个类。

        Args:
            template: 模板。 template_format: 模板的格式。 partial_variables:
            可用于部分填充模板的字典。

                For example, if the template is `"{variable1} {variable2}"`, and
                `partial_variables` is `{"variable1": "foo"}`, then the final prompt
                will be `"foo {variable2}"`.

            **kwargs: 传递给构造函数的关键字参数。

        Returns:
            该类的新实例。
        """
        prompt = PromptTemplate.from_template(
            template,
            template_format=template_format,
            partial_variables=partial_variables,
        )
        return cls(prompt=prompt, **kwargs)

    @classmethod
    def from_template_file(
        cls,
        template_file: str | Path,
        **kwargs: Any,
    ) -> Self:
        """从模板文件创建一个类。

        Args:
            template_file: 模板文件的路径。
            **kwargs: 传递给构造函数的关键字参数。

        Returns:
            该类的新实例。
        """
        prompt = PromptTemplate.from_file(template_file)
        return cls(prompt=prompt, **kwargs)

    @abstractmethod
    def format(self, **kwargs: Any) -> BaseMessage:
        """格式化提示词模板。

        Args:
            **kwargs: 用于格式化的关键字参数。

        Returns:
            格式化后的消息。
        """

    async def aformat(self, **kwargs: Any) -> BaseMessage:
        """异步格式化提示词模板。

        Args:
            **kwargs: 用于格式化的关键字参数。

        Returns:
            格式化后的消息。
        """
        return self.format(**kwargs)

    def format_messages(self, **kwargs: Any) -> list[BaseMessage]:
        """从 kwargs 中格式化消息。

        Args:
            **kwargs: 用于格式化的关键字参数。

        Returns:
            BaseMessage 对象的列表。
        """
        return [self.format(**kwargs)]

    async def aformat_messages(self, **kwargs: Any) -> list[BaseMessage]:
        """异步从 kwargs 中格式化消息。

        Args:
            **kwargs: 用于格式化的关键字参数。

        Returns:
            BaseMessage 对象的列表。
        """
        return [await self.aformat(**kwargs)]

    @property
    def input_variables(self) -> list[str]:
        """此提示词模板的输入变量。

        Returns:
            输入变量名称的列表。
        """
        return self.prompt.input_variables

    @override
    def pretty_repr(self, html: bool = False) -> str:
        """人类可读的表示形式。

        Args:
            html: 是否以 HTML 格式化。

        Returns:
            人类可读的表示形式。
        """
                # TODO: 处理部分参数
        title = self.__class__.__name__.replace("MessagePromptTemplate", " Message")
        title = get_msg_title_repr(title, bold=html)
        return f"{title}\n\n{self.prompt.pretty_repr(html=html)}"


class ChatMessagePromptTemplate(BaseStringMessagePromptTemplate):
    """聊天消息提示词模板。"""

    role: str
    """消息的角色。"""

    def format(self, **kwargs: Any) -> BaseMessage:
        """格式化提示词模板。

        Args:
            **kwargs: 用于格式化的关键字参数。

        Returns:
            格式化后的消息。
        """
        text = self.prompt.format(**kwargs)
        return ChatMessage(
            content=text, role=self.role, additional_kwargs=self.additional_kwargs
        )

    async def aformat(self, **kwargs: Any) -> BaseMessage:
        """异步格式化提示词模板。

        Args:
            **kwargs: 用于格式化的关键字参数。

        Returns:
            格式化后的消息。
        """
        text = await self.prompt.aformat(**kwargs)
        return ChatMessage(
            content=text, role=self.role, additional_kwargs=self.additional_kwargs
        )


class _TextTemplateParam(TypedDict, total=False):
    text: str | dict[str, Any]


class _ImageTemplateParam(TypedDict, total=False):
    image_url: str | dict[str, Any]


class _StringImageMessagePromptTemplate(BaseMessagePromptTemplate):
    """用户消息提示词模板。这是从用户发送的消息。"""

    prompt: (
        StringPromptTemplate
        | list[StringPromptTemplate | ImagePromptTemplate | DictPromptTemplate]
    )
    """提示词模板。"""
    additional_kwargs: dict[str, Any] = Field(default_factory=dict)
    """传递给提示词模板的额外关键字参数。"""

    _msg_class: type[BaseMessage]

    @classmethod
    def from_template(
        cls: type[Self],
        template: str
        | Sequence[str | _TextTemplateParam | _ImageTemplateParam | dict[str, Any]],
        template_format: PromptTemplateFormat = "f-string",
        *,
        partial_variables: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Self:
        """从字符串模板创建一个类。

        Args:
            template: 模板。
            template_format: 模板的格式。

                Options are: 选项包括：`'f-string'`、`'mustache'`、`'jinja2'`。
            partial_variables: 可用于部分填充的变量字典。

            **kwargs: 传递给构造函数的关键字参数。

        Returns:
            该类的新实例。

        Raises:
            ValueError: 如果模板不是字符串或字符串列表。
        """
        prompt: (
            StringPromptTemplate
            | list[StringPromptTemplate | ImagePromptTemplate | DictPromptTemplate]
        )
        if isinstance(template, str):
            prompt = PromptTemplate.from_template(
                template,
                template_format=template_format,
                partial_variables=partial_variables,
            )
            return cls(prompt=prompt, **kwargs)
        if isinstance(template, Sequence):
            if (partial_variables is not None) and len(partial_variables) > 0:
                msg = "Partial variables are not supported for list of templates."
                raise ValueError(msg)
            prompt = []
            for tmpl in template:
                if isinstance(tmpl, str) or (
                    isinstance(tmpl, dict)
                    and "text" in tmpl
                    and set(tmpl.keys()) <= {"type", "text"}
                ):
                    if isinstance(tmpl, str):
                        text: str = tmpl
                    else:
                        text = cast("_TextTemplateParam", tmpl)["text"]  # type: ignore[assignment]
                    prompt.append(
                        PromptTemplate.from_template(
                            text, template_format=template_format
                        )
                    )
                elif (
                    isinstance(tmpl, dict)
                    and "image_url" in tmpl
                    and set(tmpl.keys())
                    <= {
                        "type",
                        "image_url",
                    }
                ):
                    img_template = cast("_ImageTemplateParam", tmpl)["image_url"]
                    input_variables = []
                    if isinstance(img_template, str):
                        variables = get_template_variables(
                            img_template, template_format
                        )
                        if variables:
                            if len(variables) > 1:
                                msg = (
                                    "Only one format variable allowed per image"
                                    f" template.\nGot: {variables}"
                                    f"\nFrom: {tmpl}"
                                )
                                raise ValueError(msg)
                            input_variables = [variables[0]]
                        img_template = {"url": img_template}
                        img_template_obj = ImagePromptTemplate(
                            input_variables=input_variables,
                            template=img_template,
                            template_format=template_format,
                        )
                    elif isinstance(img_template, dict):
                        img_template = dict(img_template)
                        for key in ["url", "path", "detail"]:
                            if key in img_template:
                                input_variables.extend(
                                    get_template_variables(
                                        img_template[key], template_format
                                    )
                                )
                        img_template_obj = ImagePromptTemplate(
                            input_variables=input_variables,
                            template=img_template,
                            template_format=template_format,
                        )
                    else:
                        msg = f"Invalid image template: {tmpl}"  # type: ignore[unreachable]
                        raise ValueError(msg)
                    prompt.append(img_template_obj)
                elif isinstance(tmpl, dict):
                    if template_format == "jinja2":
                        msg = (
                            "jinja2 is unsafe and is not supported for templates "
                            "expressed as dicts. Please use 'f-string' or 'mustache' "
                            "format."
                        )
                        raise ValueError(msg)
                    data_template_obj = DictPromptTemplate(
                        template=cast("dict[str, Any]", tmpl),
                        template_format=template_format,
                    )
                    prompt.append(data_template_obj)
                else:
                    msg = f"Invalid template: {tmpl}"  # type: ignore[unreachable]
                    raise ValueError(msg)
            return cls(prompt=prompt, **kwargs)
        msg = f"Invalid template: {template}"  # type: ignore[unreachable]
        raise ValueError(msg)

    @classmethod
    def from_template_file(
        cls: type[Self],
        template_file: str | Path,
        input_variables: list[str],
        **kwargs: Any,
    ) -> Self:
        """从模板文件创建一个类。

        Args:
            template_file: 模板文件的路径。
            input_variables: 输入变量的列表。
            **kwargs: 传递给构造函数的关键词参数。

        Returns:
            此类的新实例。
        """
        template = Path(template_file).read_text(encoding="utf-8")
        return cls.from_template(template, input_variables=input_variables, **kwargs)

    def format_messages(self, **kwargs: Any) -> list[BaseMessage]:
        """从 kwargs 格式化消息。

        Args:
            **kwargs: 用于格式化的关键词参数。

        Returns:
            `BaseMessage` 对象的列表。
        """
        return [self.format(**kwargs)]

    async def aformat_messages(self, **kwargs: Any) -> list[BaseMessage]:
        """异步从 kwargs 格式化消息。

        Args:
            **kwargs: 用于格式化的关键词参数。

        Returns:
            `BaseMessage` 对象的列表。
        """
        return [await self.aformat(**kwargs)]

    @property
    def input_variables(self) -> list[str]:
        """此提示词模板的输入变量。

        Returns:
            输入变量名称的列表。
        """
        prompts = self.prompt if isinstance(self.prompt, list) else [self.prompt]
        return [iv for prompt in prompts for iv in prompt.input_variables]

    def format(self, **kwargs: Any) -> BaseMessage:
        """格式化提示词模板。

        Args:
            **kwargs: 用于格式化的关键词参数。

        Returns:
            格式化后的消息。
        """
        if isinstance(self.prompt, StringPromptTemplate):
            text = self.prompt.format(**kwargs)
            return self._msg_class(
                content=text, additional_kwargs=self.additional_kwargs
            )
        content: list[str | dict[str, Any]] = []
        for prompt in self.prompt:
            inputs = {var: kwargs[var] for var in prompt.input_variables}
            if isinstance(prompt, StringPromptTemplate):
                formatted_text = prompt.format(**inputs)
                if formatted_text != "":
                    content.append({"type": "text", "text": formatted_text})
            elif isinstance(prompt, ImagePromptTemplate):
                formatted_image = prompt.format(**inputs)
                content.append({"type": "image_url", "image_url": formatted_image})
            elif isinstance(prompt, DictPromptTemplate):
                formatted_dict = prompt.format(**inputs)
                content.append(formatted_dict)
        return self._msg_class(
            content=content, additional_kwargs=self.additional_kwargs
        )

    async def aformat(self, **kwargs: Any) -> BaseMessage:
        """异步格式化提示词模板。

        Args:
            **kwargs: 用于格式化的关键词参数。

        Returns:
            格式化后的消息。
        """
        if isinstance(self.prompt, StringPromptTemplate):
            text = await self.prompt.aformat(**kwargs)
            return self._msg_class(
                content=text, additional_kwargs=self.additional_kwargs
            )
        content: list[str | dict[str, Any]] = []
        for prompt in self.prompt:
            inputs = {var: kwargs[var] for var in prompt.input_variables}
            if isinstance(prompt, StringPromptTemplate):
                formatted_text = await prompt.aformat(**inputs)
                if formatted_text != "":
                    content.append({"type": "text", "text": formatted_text})
            elif isinstance(prompt, ImagePromptTemplate):
                formatted_image = await prompt.aformat(**inputs)
                content.append({"type": "image_url", "image_url": formatted_image})
            elif isinstance(prompt, DictPromptTemplate):
                formatted_dict = prompt.format(**inputs)
                content.append(formatted_dict)
        return self._msg_class(
            content=content, additional_kwargs=self.additional_kwargs
        )

    @override
    def pretty_repr(self, html: bool = False) -> str:
        """可读性高的表示形式。

        Args:
            html: 是否以 HTML 格式表示。

        Returns:
            可读性高的表示形式。
        """
                # TODO: 处理部分参数
        title = self.__class__.__name__.replace("MessagePromptTemplate", " Message")
        title = get_msg_title_repr(title, bold=html)
        prompts = self.prompt if isinstance(self.prompt, list) else [self.prompt]
        prompt_reprs = "\n\n".join(prompt.pretty_repr(html=html) for prompt in prompts)
        return f"{title}\n\n{prompt_reprs}"


class HumanMessagePromptTemplate(_StringImageMessagePromptTemplate):
    """用户消息提示词模板。

    这是从用户发送的消息。
    """

    _msg_class: type[BaseMessage] = HumanMessage


class AIMessagePromptTemplate(_StringImageMessagePromptTemplate):
    """AI 消息提示词模板。

    这是从 AI 发送的消息。
    """

    _msg_class: type[BaseMessage] = AIMessage


class SystemMessagePromptTemplate(_StringImageMessagePromptTemplate):
    """系统消息提示词模板。

    这是不发送给用户的消息。
    """

    _msg_class: type[BaseMessage] = SystemMessage


class BaseChatPromptTemplate(BasePromptTemplate[str], ABC):
    """聊天提示词模板的基类。"""

    @property
    @override
    def lc_attributes(self) -> dict[str, Any]:
        return {"input_variables": self.input_variables}

    def format(self, **kwargs: Any) -> str:
        """将聊天模板格式化为字符串。

        Args:
            **kwargs: Keyword arguments to use for filling in template variables in all
                the template messages in this chat template.

        Returns:
            格式化后的字符串。
        """
        return self.format_prompt(**kwargs).to_string()

    async def aformat(self, **kwargs: Any) -> str:
        """异步将聊天模板格式化为字符串。

        Args:
            **kwargs: Keyword arguments to use for filling in template variables in all
                the template messages in this chat template.

        Returns:
            格式化后的字符串。
        """
        return (await self.aformat_prompt(**kwargs)).to_string()

    def format_prompt(self, **kwargs: Any) -> ChatPromptValue:
        """格式化提示词。

        应返回一个 `ChatPromptValue`。

        Args:
            **kwargs: 用于格式化的关键词参数。
        """
        messages = self.format_messages(**kwargs)
        return ChatPromptValue(messages=messages)

    async def aformat_prompt(self, **kwargs: Any) -> ChatPromptValue:
        """异步格式化提示词。

        应返回一个 `ChatPromptValue`。

        Args:
            **kwargs: 用于格式化的关键词参数。
        """
        messages = await self.aformat_messages(**kwargs)
        return ChatPromptValue(messages=messages)

    @abstractmethod
    def format_messages(self, **kwargs: Any) -> list[BaseMessage]:
        """将 kwargs 格式化为消息列表。

        Returns:
            `BaseMessage` 对象的列表。
        """

    async def aformat_messages(self, **kwargs: Any) -> list[BaseMessage]:
        """异步将 kwargs 格式化为消息列表。

        Returns:
            `BaseMessage` 对象的列表。
        """
        return self.format_messages(**kwargs)

    def pretty_repr(
        self,
        html: bool = False,  # noqa: FBT001,FBT002
    ) -> str:
        """可读性高的表示形式。

        Args:
            html: 是否以 HTML 格式表示。

        Returns:
            可读性高的表示形式。
        """
        raise NotImplementedError

    def pretty_print(self) -> None:
        """打印可读性高的表示形式。"""
        print(self.pretty_repr(html=is_interactive_env()))  # noqa: T201


MessageLike = BaseMessagePromptTemplate | BaseMessage | BaseChatPromptTemplate

MessageLikeRepresentation = (
    MessageLike
    | tuple[str | type, str | Sequence[dict[str, Any]] | Sequence[object]]
    | str
    | dict[str, Any]
)


class ChatPromptTemplate(BaseChatPromptTemplate):
    """聊天模型的提示词模板。

    用于为聊天模型创建灵活的模板化提示词。

    !!! example

        ```python
        from langchain_core.prompts import ChatPromptTemplate

        template = ChatPromptTemplate(
            [
                ("system", "You are a helpful AI bot. Your name is {name}."),
                ("human", "Hello, how are you doing?"),
                ("ai", "I'm doing well, thanks!"),
                ("human", "{user_input}"),
            ]
        )

        prompt_value = template.invoke(
            {
                "name": "Bob",
                "user_input": "What is your name?",
            }
        )
        # Output:
        # ChatPromptValue(
        #    messages=[
        #        SystemMessage(content='You are a helpful AI bot. Your name is Bob.'),
        #        HumanMessage(content='Hello, how are you doing?'),
        #        AIMessage(content="I'm doing well, thanks!"),
        #        HumanMessage(content='What is your name?')
        #    ]
        # )
        ```

    !!! note "消息占位符"

        ```python
        # In addition to Human/AI/Tool/Function messages,
        # you can initialize the template with a MessagesPlaceholder
        # either using the class directly or with the shorthand tuple syntax:

        template = ChatPromptTemplate(
            [
                ("system", "You are a helpful AI bot."),
                # Means the template will receive an optional list of messages under
                # the "conversation" key
                ("placeholder", "{conversation}"),
                # Equivalently:
                # MessagesPlaceholder(variable_name="conversation", optional=True)
            ]
        )

        prompt_value = template.invoke(
            {
                "conversation": [
                    ("human", "Hi!"),
                    ("ai", "How can I assist you today?"),
                    ("human", "Can you make me an ice cream sundae?"),
                    ("ai", "No."),
                ]
            }
        )

        # Output:
        # ChatPromptValue(
        #    messages=[
        #        SystemMessage(content='You are a helpful AI bot.'),
        #        HumanMessage(content='Hi!'),
        #        AIMessage(content='How can I assist you today?'),
        #        HumanMessage(content='Can you make me an ice cream sundae?'),
        #        AIMessage(content='No.'),
        #    ]
        # )
        ```

    !!! note "单变量模板"

        If your prompt has only a single input variable (i.e., one instance of
        `'{variable_nams}'`), and you invoke the template with a non-dict object, the
        prompt template will inject the provided argument into that variable location.

        ```python
        from langchain_core.prompts import ChatPromptTemplate

        template = ChatPromptTemplate(
            [
                ("system", "You are a helpful AI bot. Your name is Carl."),
                ("human", "{user_input}"),
            ]
        )

        prompt_value = template.invoke("Hello, there!")
        # Equivalent to
        # prompt_value = template.invoke({"user_input": "Hello, there!"})

        # Output:
        #  ChatPromptValue(
        #     messages=[
        #         SystemMessage(content='You are a helpful AI bot. Your name is Carl.'),
        #         HumanMessage(content='Hello, there!'),
        #     ]
        # )
        ```
    """

    messages: Annotated[list[MessageLike], SkipValidation()]
    """由消息提示词模板或消息组成的列表。"""

    validate_template: bool = False
    """是否尝试验证模板。"""

    def __init__(
        self,
        messages: Sequence[MessageLikeRepresentation],
        *,
        template_format: PromptTemplateFormat = "f-string",
        **kwargs: Any,
    ) -> None:
        """从多种消息格式创建聊天提示词模板。

        Args:
            messages: 消息表示的序列。

                消息可以使用以下格式表示：

                1. `BaseMessagePromptTemplate`
                2. `BaseMessage`
                3. 二元组 `(消息类型, 模板)`；例如，
                    `('human', '{user_input}')`
                4. 二元组 `(消息类, 模板)`
                5. 字符串，作为 `('human', 模板)` 的简写；例如，
                    `'{user_input}'`
            template_format: 模板的格式。
            **kwargs: 传递给 `BasePromptTemplate` 的额外关键词参数，
                包括（但不限于）：

                - `input_variables`: 需要作为输入值的变量名称列表。 -
                `optional_variables`: 可选占位符或 `MessagePlaceholder` 的变量名称列表。

                    These variables are auto inferred from the prompt and user need not
                    provide them.

                - `partial_variables`: 提示词模板携带的局部变量字典。

                    局部变量会填充到模板中，因此你无需每次调用提示词时都传入它们。

                - `validate_template`: 是否验证模板。 - `input_types`:
                提示词模板期望的变量类型字典。

                    如果未提供，则假设所有变量均为字符串。

        Examples:
            从消息模板列表实例化：

            ```python
            template = ChatPromptTemplate(
                [
                    ("human", "Hello, how are you?"),
                    ("ai", "I'm doing well, thanks!"),
                    ("human", "That's good to hear."),
                ]
            )
            ```

            从混合消息格式实例化：

            ```python
            template = ChatPromptTemplate(
                [
                    SystemMessage(content="hello"),
                    ("human", "Hello, how are you?"),
                ]
            )
            ```
        """
        messages_ = [
            _convert_to_message_template(message, template_format)
            for message in messages
        ]

                # 自动从消息中推断输入变量
        input_vars: set[str] = set()
        optional_variables: set[str] = set()
        partial_vars: dict[str, Any] = {}
        for message in messages_:
            if isinstance(message, MessagesPlaceholder) and message.optional:
                partial_vars[message.variable_name] = []
                optional_variables.add(message.variable_name)
            elif isinstance(
                message, (BaseChatPromptTemplate, BaseMessagePromptTemplate)
            ):
                input_vars.update(message.input_variables)

        kwargs = {
            "input_variables": sorted(input_vars),
            "optional_variables": sorted(optional_variables),
            "partial_variables": partial_vars,
            **kwargs,
        }
        cast("type[ChatPromptTemplate]", super()).__init__(messages=messages_, **kwargs)

    @classmethod
    def get_lc_namespace(cls) -> list[str]:
        """获取 LangChain 对象的命名空间。

        Returns:
            `["langchain", "prompts", "chat"]`
        """
        return ["langchain", "prompts", "chat"]

    def __add__(self, other: Any) -> ChatPromptTemplate:
        """合并两个提示词模板。

        Args:
            other: 另一个提示词模板。

        Returns:
            合并后的提示词模板。
        """
        partials = {**self.partial_variables}

                # 需要检查另一个对象是否具有部分变量，因为它可能不是
                # ChatPromptTemplate。
        if hasattr(other, "partial_variables") and other.partial_variables:
            partials.update(other.partial_variables)

                # 便于轻松组合
        if isinstance(other, ChatPromptTemplate):
            return ChatPromptTemplate(messages=self.messages + other.messages).partial(
                **partials
            )
        if isinstance(
            other, (BaseMessagePromptTemplate, BaseMessage, BaseChatPromptTemplate)
        ):
            return ChatPromptTemplate(messages=[*self.messages, other]).partial(
                **partials
            )
        if isinstance(other, (list, tuple)):
            other_ = ChatPromptTemplate.from_messages(other)
            return ChatPromptTemplate(messages=self.messages + other_.messages).partial(
                **partials
            )
        if isinstance(other, str):
            prompt = HumanMessagePromptTemplate.from_template(other)
            return ChatPromptTemplate(messages=[*self.messages, prompt]).partial(
                **partials
            )
        msg = f"Unsupported operand type for +: {type(other)}"
        raise NotImplementedError(msg)

    @model_validator(mode="before")
    @classmethod
    def validate_input_variables(cls, values: dict[str, Any]) -> Any:
        """验证输入变量。

        如果未设置 `input_variables`，它将被设为消息中所有输入变量的并集。

        Args:
            values: 需要验证的值。

        Returns:
            验证后的值。

        Raises:
            ValueError: 如果输入变量不匹配。
        """
        messages = values["messages"]
        input_vars: set[str] = set()
        optional_variables = set()
        input_types: dict[str, Any] = values.get("input_types", {})
        for message in messages:
            if isinstance(message, (BaseMessagePromptTemplate, BaseChatPromptTemplate)):
                input_vars.update(message.input_variables)
            if isinstance(message, MessagesPlaceholder):
                if "partial_variables" not in values:
                    values["partial_variables"] = {}
                if (
                    message.optional
                    and message.variable_name not in values["partial_variables"]
                ):
                    values["partial_variables"][message.variable_name] = []
                    optional_variables.add(message.variable_name)
                if message.variable_name not in input_types:
                    input_types[message.variable_name] = list[AnyMessage]
        if "partial_variables" in values:
            input_vars -= set(values["partial_variables"])
        if optional_variables:
            input_vars -= optional_variables
        if "input_variables" in values and values.get("validate_template"):
            if input_vars != set(values["input_variables"]):
                msg = (
                    "Got mismatched input_variables. "
                    f"Expected: {input_vars}. "
                    f"Got: {values['input_variables']}"
                )
                raise ValueError(msg)
        else:
            values["input_variables"] = sorted(input_vars)
        if optional_variables:
            values["optional_variables"] = sorted(optional_variables)
        values["input_types"] = input_types
        return values

    @classmethod
    def from_template(cls, template: str, **kwargs: Any) -> ChatPromptTemplate:
        """从模板字符串创建聊天提示词模板。

        Creates a chat template consisting of a single message assumed to be from the
        human.

        Args:
            template: 模板字符串
            **kwargs: 传递给构造函数的关键词参数。

        Returns:
            该类的新实例。
        """
        prompt_template = PromptTemplate.from_template(template, **kwargs)
        message = HumanMessagePromptTemplate(prompt=prompt_template)
        return cls.from_messages([message])

    @classmethod
    def from_messages(
        cls,
        messages: Sequence[MessageLikeRepresentation],
        template_format: PromptTemplateFormat = "f-string",
    ) -> ChatPromptTemplate:
        """从多种消息格式创建聊天提示词模板。

        Examples:
            从消息模板列表实例化：

            ```python
            template = ChatPromptTemplate.from_messages(
                [
                    ("human", "Hello, how are you?"),
                    ("ai", "I'm doing well, thanks!"),
                    ("human", "That's good to hear."),
                ]
            )
            ```

            从混合消息格式实例化：

            ```python
            template = ChatPromptTemplate.from_messages(
                [
                    SystemMessage(content="hello"),
                    ("human", "Hello, how are you?"),
                ]
            )
            ```
        Args:
            messages: 消息表示的序列。

                消息可以使用以下格式表示：

                1. `BaseMessagePromptTemplate`
                2. `BaseMessage`
                3. 2 元组 `(消息类型, 模板)`；例如，
                    `('human', '{user_input}')`
                4. 2 元组 `(消息类, 模板)`
                5. 字符串，作为 `('human', 模板)` 的简写；例如，
                    `'{user_input}'`
            template_format: 模板的格式。

        Returns:
            一个聊天提示词模板。

        """
        return cls(messages, template_format=template_format)

    def format_messages(self, **kwargs: Any) -> list[BaseMessage]:
        """将聊天模板格式化为一组已确定的消息。

        Args:
            **kwargs: Keyword arguments to use for filling in template variables
                in all the template messages in this chat template.

        Raises:
            ValueError: 如果消息类型不符合预期。

        Returns:
            格式化后的消息列表。
        """
        kwargs = self._merge_partial_and_user_variables(**kwargs)
        result = []
        for message_template in self.messages:
            if isinstance(message_template, BaseMessage):
                result.extend([message_template])
            elif isinstance(
                message_template, (BaseMessagePromptTemplate, BaseChatPromptTemplate)
            ):
                message = message_template.format_messages(**kwargs)
                result.extend(message)
            else:
                msg = f"Unexpected input: {message_template}"  # type: ignore[unreachable]
                raise ValueError(msg)  # noqa: TRY004
        return result

    async def aformat_messages(self, **kwargs: Any) -> list[BaseMessage]:
        """异步将聊天模板格式化为一组已确定的消息。

        Args:
            **kwargs: Keyword arguments to use for filling in template variables
                in all the template messages in this chat template.

        Returns:
            格式化后的消息列表。

        Raises:
            ValueError: 如果输入不正确。
        """
        kwargs = self._merge_partial_and_user_variables(**kwargs)
        result = []
        for message_template in self.messages:
            if isinstance(message_template, BaseMessage):
                result.extend([message_template])
            elif isinstance(
                message_template, (BaseMessagePromptTemplate, BaseChatPromptTemplate)
            ):
                message = await message_template.aformat_messages(**kwargs)
                result.extend(message)
            else:
                msg = f"Unexpected input: {message_template}"  # type: ignore[unreachable]
                raise ValueError(msg)  # noqa:TRY004
        return result

    def partial(self, **kwargs: Any) -> ChatPromptTemplate:
        """获取一个新的 `ChatPromptTemplate`，其中部分输入变量已预先填充。

        Args:
            **kwargs: 用于填充模板变量的关键字参数。

                应当是输入变量的子集。

        Returns:
            一个新的 `ChatPromptTemplate`。

        Example:
            ```python
            from langchain_core.prompts import ChatPromptTemplate

            template = ChatPromptTemplate.from_messages(
                [
                    ("system", "You are an AI assistant named {name}."),
                    ("human", "Hi I'm {user}"),
                    ("ai", "Hi there, {user}, I'm {name}."),
                    ("human", "{input}"),
                ]
            )
            template2 = template.partial(user="Lucy", name="R2D2")

            template2.format_messages(input="hello")
            ```
        """
        prompt_dict = self.__dict__.copy()
        prompt_dict["input_variables"] = list(
            set(self.input_variables).difference(kwargs)
        )
        prompt_dict["partial_variables"] = {**self.partial_variables, **kwargs}
        return type(self)(**prompt_dict)

    def append(self, message: MessageLikeRepresentation) -> None:
        """在聊天模板末尾追加一条消息。

        Args:
            message: 要追加的消息的表示形式。
        """
        self.messages.append(_convert_to_message_template(message))

    def extend(self, messages: Sequence[MessageLikeRepresentation]) -> None:
        """用一系列消息扩展聊天模板。

        Args:
            messages: 要追加的消息表示序列。
        """
        self.messages.extend(
            [_convert_to_message_template(message) for message in messages]
        )

    @overload
    def __getitem__(self, index: int) -> MessageLike: ...

    @overload
    def __getitem__(self, index: slice) -> ChatPromptTemplate: ...

    def __getitem__(self, index: int | slice) -> MessageLike | ChatPromptTemplate:
        """用于对聊天模板进行索引。

        Returns:
            如果索引是整数，返回该索引处的消息。

            如果索引是切片，返回一个包含该切片中消息的新 `ChatPromptTemplate`。
        """
        if isinstance(index, slice):
            start, stop, step = index.indices(len(self.messages))
            messages = self.messages[start:stop:step]
            return ChatPromptTemplate.from_messages(messages)
        return self.messages[index]

    def __len__(self) -> int:
        """返回聊天模板的长度。"""
        return len(self.messages)

    @property
    def _prompt_type(self) -> str:
        """提示词类型的名称。用于序列化。"""
        return "chat"

    @deprecated(
        since="1.2.21",
        removal="2.0.0",
        alternative="Use `dumpd`/`dumps` from `langchain_core.load` to serialize "
        "prompts and `load`/`loads` to deserialize them.",
    )
    def save(self, file_path: Path | str) -> None:
        """将提示词保存到文件。

        Args:
            file_path: 文件路径。
        """
        raise NotImplementedError

    @override
    def pretty_repr(self, html: bool = False) -> str:
        """可读性良好的表示形式。

        Args:
            html: 是否以 HTML 格式进行格式化。

        Returns:
            可读性良好的表示形式。
        """
                # TODO: 处理部分参数
        return "\n\n".join(msg.pretty_repr(html=html) for msg in self.messages)


def _create_template_from_message_type(
    message_type: str,
    template: str | list[str | dict[str, Any] | bool],
    template_format: PromptTemplateFormat = "f-string",
) -> BaseMessagePromptTemplate:
    """从消息类型和模板字符串创建一个消息提示词模板。

    Args:
        message_type: 消息模板的类型（例如 `'human'`、`'ai'` 等）
        template: 模板字符串。
        template_format: 模板格式。

    Returns:
        相应类型的提示词模板。

    Raises:
        ValueError: 如果消息类型不正确。
    """
    if message_type in {"human", "user"}:
        message: BaseMessagePromptTemplate = HumanMessagePromptTemplate.from_template(
            cast("str", template), template_format=template_format
        )
    elif message_type in {"ai", "assistant"}:
        message = AIMessagePromptTemplate.from_template(
            cast("str", template), template_format=template_format
        )
    elif message_type == "system":
        message = SystemMessagePromptTemplate.from_template(
            cast("str", template), template_format=template_format
        )
    elif message_type == "placeholder":
        if isinstance(template, str):
            if template[0] != "{" or template[-1] != "}":
                msg = (
                    f"Invalid placeholder template: {template}."
                    " Expected a variable name surrounded by curly braces."
                )
                raise ValueError(msg)
            var_name = template[1:-1]
            message = MessagesPlaceholder(variable_name=var_name, optional=True)
        else:
            try:
                var_name_wrapped, is_optional = template
            except ValueError as e:
                msg = (
                    "Unexpected arguments for placeholder message type."
                    " Expected either a single string variable name"
                    " or a list of [variable_name: str, is_optional: bool]."
                    f" Got: {template}"
                )
                raise ValueError(msg) from e

            if not isinstance(is_optional, bool):
                msg = f"Expected is_optional to be a boolean. Got: {is_optional}"
                raise ValueError(msg)  # noqa: TRY004

            if not isinstance(var_name_wrapped, str):
                msg = f"Expected variable name to be a string. Got: {var_name_wrapped}"
                raise ValueError(msg)  # noqa: TRY004
            if var_name_wrapped[0] != "{" or var_name_wrapped[-1] != "}":
                msg = (
                    f"Invalid placeholder template: {var_name_wrapped}."
                    " Expected a variable name surrounded by curly braces."
                )
                raise ValueError(msg)
            var_name = var_name_wrapped[1:-1]

            message = MessagesPlaceholder(variable_name=var_name, optional=is_optional)
    else:
        msg = (
            f"Unexpected message type: {message_type}. Use one of 'human',"
            f" 'user', 'ai', 'assistant', or 'system'."
        )
        raise ValueError(msg)
    return message


def _convert_to_message_template(
    message: MessageLikeRepresentation,
    template_format: PromptTemplateFormat = "f-string",
) -> BaseMessage | BaseMessagePromptTemplate | BaseChatPromptTemplate:
    """从多种消息格式实例化一个消息。

    消息可以使用以下格式表示：

    1. `BaseMessagePromptTemplate`
    2. `BaseMessage`
    3. 2 元组 `(消息类型, 模板)`；例如 `('human', '{user_input}')`
    4. 2 元组 `(消息类, 模板)`
    5. 作为 `('human', 模板)` 的简写字符串；例如 `'{user_input}'`

    Args:
        message: 一种支持格式的消息表示。
        template_format: 模板格式。

    Returns:
        消息或消息模板的实例。

    Raises:
        ValueError: 如果消息类型不正确。
        ValueError: 如果 2 元组元素不足两个。
    """
    if isinstance(message, (BaseMessagePromptTemplate, BaseChatPromptTemplate)):
        message_: BaseMessage | BaseMessagePromptTemplate | BaseChatPromptTemplate = (
            message
        )
    elif isinstance(message, BaseMessage):
        message_ = message
    elif isinstance(message, str):
        message_ = _create_template_from_message_type(
            "human", message, template_format=template_format
        )
    elif isinstance(message, (tuple, dict)):
        if isinstance(message, dict):
            if set(message.keys()) != {"content", "role"}:
                msg = (
                    "Expected dict to have exact keys 'role' and 'content'."
                    f" Got: {message}"
                )
                raise ValueError(msg)
            message_type_str = message["role"]
            template = message["content"]
        else:
            if len(message) != 2:  # noqa: PLR2004
                msg = f"Expected 2-tuple of (role, template), got {message}"  # type: ignore[unreachable]
                raise ValueError(msg)
            message_type_str, template = message

        if isinstance(message_type_str, str):
            message_ = _create_template_from_message_type(
                message_type_str, template, template_format=template_format
            )
        elif (
            hasattr(message_type_str, "model_fields")
            and "type" in message_type_str.model_fields
        ):
            message_type = message_type_str.model_fields["type"].default
            message_ = _create_template_from_message_type(
                message_type, template, template_format=template_format
            )
        else:
            message_ = message_type_str(
                prompt=PromptTemplate.from_template(
                    cast("str", template), template_format=template_format
                )
            )
    else:
        msg = f"Unsupported message type: {type(message)}"  # type: ignore[unreachable]
        raise NotImplementedError(msg)

    return message_


# 为了向后兼容：
_convert_to_message = _convert_to_message_template

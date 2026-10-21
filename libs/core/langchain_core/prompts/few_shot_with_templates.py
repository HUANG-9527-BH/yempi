"""包含少量示例的提示词模板。"""

from pathlib import Path
from typing import Any

from pydantic import ConfigDict, model_validator
from typing_extensions import Self

from langchain_core._api import deprecated
from langchain_core.example_selectors import BaseExampleSelector
from langchain_core.prompts.prompt import PromptTemplate
from langchain_core.prompts.string import (
    DEFAULT_FORMATTER_MAPPING,
    PromptTemplateFormat,
    StringPromptTemplate,
)


class FewShotPromptWithTemplates(StringPromptTemplate):
    """包含少量示例的提示词模板。"""

    examples: list[dict[str, Any]] | None = None
    """用于格式化到提示词中的示例。

    二者之一必须提供。
    """

    example_selector: BaseExampleSelector | None = None
    """`ExampleSelector`，用于选择要格式化到提示词中的示例。

    二者之一必须提供。
    """

    example_prompt: PromptTemplate
    """用于格式化单个示例的 `PromptTemplate`。"""

    suffix: StringPromptTemplate
    """在示例之后放置的 `PromptTemplate`。"""

    example_separator: str = "\n\n"
    """用于连接前缀、示例和后缀的字符串分隔符。"""

    prefix: StringPromptTemplate | None = None
    """在示例之前放置的 `PromptTemplate`。"""

    template_format: PromptTemplateFormat = "f-string"
    """提示词模板的格式。

    Options are: 可选项：`'f-string'`、`'jinja2'`、`'mustache'`。
    """

    validate_template: bool = False
    """是否尝试验证模板。"""

    @classmethod
    def get_lc_namespace(cls) -> list[str]:
        """获取 LangChain 对象的命名空间。

        Returns:
            `["langchain", "prompts", "few_shot_with_templates"]`
        """
        return ["langchain", "prompts", "few_shot_with_templates"]

    @model_validator(mode="before")
    @classmethod
    def check_examples_and_selector(cls, values: dict[str, Any]) -> Any:
        """检查是否仅提供了 `examples` 或 `example_selector` 中的一个。"""
        examples = values.get("examples")
        example_selector = values.get("example_selector")
        if examples and example_selector:
            msg = "Only one of 'examples' and 'example_selector' should be provided"
            raise ValueError(msg)

        if examples is None and example_selector is None:
            msg = "One of 'examples' and 'example_selector' should be provided"
            raise ValueError(msg)

        return values

    @model_validator(mode="after")
    def template_is_valid(self) -> Self:
        """检查前缀、后缀和输入变量是否一致。"""
        if self.validate_template:
            input_variables = self.input_variables
            expected_input_variables = set(self.suffix.input_variables)
            expected_input_variables |= set(self.partial_variables)
            if self.prefix is not None:
                expected_input_variables |= set(self.prefix.input_variables)
            missing_vars = expected_input_variables.difference(input_variables)
            if missing_vars:
                msg = (
                    f"Got input_variables={input_variables}, but based on "
                    f"prefix/suffix expected {expected_input_variables}"
                )
                raise ValueError(msg)
        else:
            self.input_variables = sorted(
                set(self.suffix.input_variables)
                | set(self.prefix.input_variables if self.prefix else [])
                - set(self.partial_variables)
            )
        return self

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="forbid",
    )

    def _get_examples(self, **kwargs: Any) -> list[dict[str, Any]]:
        if self.examples is not None:
            return self.examples
        if self.example_selector is not None:
            return self.example_selector.select_examples(kwargs)
        msg = "One of 'examples' and 'example_selector' should be provided"
        raise ValueError(msg)

    async def _aget_examples(self, **kwargs: Any) -> list[dict[str, Any]]:
        if self.examples is not None:
            return self.examples
        if self.example_selector is not None:
            return await self.example_selector.aselect_examples(kwargs)
        msg = "One of 'examples' and 'example_selector' should be provided"
        raise ValueError(msg)

    def format(self, **kwargs: Any) -> str:
        """使用输入格式化提示词。

        Args:
            **kwargs: 传递给提示词模板的任意参数。

        Returns:
            格式化后的字符串。

        Example:
            ```python
            prompt.format(variable1="foo")
            ```
        """
        kwargs = self._merge_partial_and_user_variables(**kwargs)
                # 获取要使用的示例。
        examples = self._get_examples(**kwargs)
                # 格式化示例。
        example_strings = [
            self.example_prompt.format(**example) for example in examples
        ]
                # 创建整体前缀。
        if self.prefix is None:
            prefix = ""
        else:
            prefix_kwargs = {
                k: v for k, v in kwargs.items() if k in self.prefix.input_variables
            }
            for k in prefix_kwargs:
                kwargs.pop(k)
            prefix = self.prefix.format(**prefix_kwargs)

                # 创建整体后缀
        suffix_kwargs = {
            k: v for k, v in kwargs.items() if k in self.suffix.input_variables
        }
        for k in suffix_kwargs:
            kwargs.pop(k)
        suffix = self.suffix.format(
            **suffix_kwargs,
        )

        pieces = [prefix, *example_strings, suffix]
        template = self.example_separator.join([piece for piece in pieces if piece])
                # 使用输入变量格式化模板。
        return DEFAULT_FORMATTER_MAPPING[self.template_format](template, **kwargs)

    async def aformat(self, **kwargs: Any) -> str:
        """异步使用输入格式化提示词。

        Args:
            **kwargs: 传递给提示词模板的任意参数。

        Returns:
            格式化后的字符串。
        """
        kwargs = self._merge_partial_and_user_variables(**kwargs)
                # 获取要使用的示例。
        examples = await self._aget_examples(**kwargs)
                # 格式化示例。
        example_strings = [
                        # 此处可以使用同步方法，因为 PromptTemplate 不会阻塞
            self.example_prompt.format(**example)
            for example in examples
        ]
                # 创建整体前缀。
        if self.prefix is None:
            prefix = ""
        else:
            prefix_kwargs = {
                k: v for k, v in kwargs.items() if k in self.prefix.input_variables
            }
            for k in prefix_kwargs:
                kwargs.pop(k)
            prefix = await self.prefix.aformat(**prefix_kwargs)

                # 创建整体后缀
        suffix_kwargs = {
            k: v for k, v in kwargs.items() if k in self.suffix.input_variables
        }
        for k in suffix_kwargs:
            kwargs.pop(k)
        suffix = await self.suffix.aformat(
            **suffix_kwargs,
        )

        pieces = [prefix, *example_strings, suffix]
        template = self.example_separator.join([piece for piece in pieces if piece])
                # 使用输入变量格式化模板。
        return DEFAULT_FORMATTER_MAPPING[self.template_format](template, **kwargs)

    @property
    def _prompt_type(self) -> str:
        """返回提示词类型键。"""
        return "few_shot_with_templates"

    @deprecated(
        since="1.2.21",
        removal="2.0.0",
        alternative="Use `dumpd`/`dumps` from `langchain_core.load` to serialize "
        "prompts and `load`/`loads` to deserialize them.",
    )
    def save(self, file_path: Path | str) -> None:
        """将提示词保存到文件。

        Args:
            file_path: 要保存提示词的路径。

        Raises:
            ValueError: 如果提供了 `example_selector`。
        """
        if self.example_selector:
            msg = "Saving an example selector is not currently supported"
            raise ValueError(msg)
        return super().save(file_path)

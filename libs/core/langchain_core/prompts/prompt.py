"""提示词模式定义。"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, model_validator
from typing_extensions import override

from langchain_core.prompts.string import (
    DEFAULT_FORMATTER_MAPPING,
    PromptTemplateFormat,
    StringPromptTemplate,
    check_valid_template,
    get_template_variables,
    mustache_schema,
)

if TYPE_CHECKING:
    from langchain_core.runnables.config import RunnableConfig


class PromptTemplate(StringPromptTemplate):
    """语言模型的提示词模板。

    提示词模板由一个字符串模板组成。它接受用户的一组参数，
    这些参数可用于生成语言模型的提示词。

    模板可以使用 f-strings（默认）、jinja2 或 mustache
    语法进行格式化。

    !!! warning "安全"

        Prefer using `template_format='f-string'` instead of `template_format='jinja2'`,
        or make sure to NEVER accept jinja2 templates from untrusted sources as they may
        lead to arbitrary Python code execution.

        从 LangChain 0.0.329 版本开始，jinja2 模板将默认使用 Jinja2 的
        SandboxedEnvironment
        进行渲染。这种沙箱机制应被视为一种尽力而为的安全措施，而非绝对保证，因为它是可选退出而非可选加入的方式。

        尽管有沙箱保护，我们仍建议永远不要使用来自不可信来源的 jinja2 模板。

    Example:
        ```python
        from langchain_core.prompts import PromptTemplate

        # Instantiation using from_template (recommended)
        prompt = PromptTemplate.from_template("Say {foo}")
        prompt.format(foo="bar")

        # Instantiation using initializer
        prompt = PromptTemplate(template="Say {foo}")
        ```
    """

    @property
    @override
    def lc_attributes(self) -> dict[str, Any]:
        return {
            "template_format": self.template_format,
        }

    @classmethod
    @override
    def get_lc_namespace(cls) -> list[str]:
        """获取 LangChain 对象的命名空间。

        Returns:
            `["langchain", "prompts", "prompt"]`
        """
        return ["langchain", "prompts", "prompt"]

    template: str
    """提示词模板。"""

    template_format: PromptTemplateFormat = "f-string"
    """提示词模板的格式。

    Options are: 可选项：`'f-string'`、`'mustache'`、`'jinja2'`。
    """

    validate_template: bool = False
    """是否尝试验证模板。"""

    @model_validator(mode="before")
    @classmethod
    def pre_init_validation(cls, values: dict[str, Any]) -> Any:
        """检查模板和输入变量是否一致。"""
        if values.get("template") is None:
                        # 如果未提供模板，将让 pydantic 抛出 ValidationError。
                        # 如果未提供模板，将让 pydantic 抛出 ValidationError。
            return values

                # 根据字段默认值设置一些默认值
        values.setdefault("template_format", "f-string")
        values.setdefault("partial_variables", {})

        if values.get("validate_template"):
            if values["template_format"] == "mustache":
                msg = "Mustache templates cannot be validated."
                raise ValueError(msg)

            if "input_variables" not in values:
                msg = "Input variables must be provided to validate the template."
                raise ValueError(msg)

            all_inputs = values["input_variables"] + list(values["partial_variables"])
            check_valid_template(
                values["template"], values["template_format"], all_inputs
            )

        if values["template_format"]:
            values["input_variables"] = [
                var
                for var in get_template_variables(
                    values["template"], values["template_format"]
                )
                if var not in values["partial_variables"]
            ]

        return values

    @override
    def get_input_schema(self, config: RunnableConfig | None = None) -> type[BaseModel]:
        """获取提示词的输入模式。

        Args:
            config: Runnable 配置。

        Returns:
            提示词的输入模式。
        """
        if self.template_format != "mustache":
            return super().get_input_schema(config)

        return mustache_schema(self.template)

    def __add__(self, other: Any) -> PromptTemplate:
        """重写 `+` 操作符以支持组合提示词模板。

        Raises:
            ValueError: If the template formats are not f-string or if there are
                conflicting partial variables.
            NotImplementedError: If the other object is not a `PromptTemplate` or str.

        Returns:
            一个新的 `PromptTemplate`，是两个模板的组合结果。
        """
                # 支持轻松组合
        if isinstance(other, PromptTemplate):
            if self.template_format != other.template_format:
                msg = "Cannot add templates of different formats"
                raise ValueError(msg)
            input_variables = list(
                set(self.input_variables) | set(other.input_variables)
            )
            template = self.template + other.template
                        # 如果某些项不希望进行验证，则不要进行验证
            validate_template = self.validate_template and other.validate_template
            partial_variables = dict(self.partial_variables.items())
            for k, v in other.partial_variables.items():
                if k in partial_variables:
                    msg = "Cannot have same variable partialed twice."
                    raise ValueError(msg)
                partial_variables[k] = v
            return PromptTemplate(
                template=template,
                input_variables=input_variables,
                partial_variables=partial_variables,
                template_format=self.template_format,
                validate_template=validate_template,
            )
        if isinstance(other, str):
            prompt = PromptTemplate.from_template(
                other,
                template_format=self.template_format,
            )
            return self + prompt
        msg = f"Unsupported operand type for +: {type(other)}"
        raise NotImplementedError(msg)

    @property
    def _prompt_type(self) -> str:
        """返回提示词类型键。"""
        return "prompt"

    def format(self, **kwargs: Any) -> str:
        """使用输入参数格式化提示词。

        Args:
            **kwargs: 传递给提示词模板的任意参数。

        Returns:
            格式化后的字符串。
        """
        kwargs = self._merge_partial_and_user_variables(**kwargs)
        return DEFAULT_FORMATTER_MAPPING[self.template_format](self.template, **kwargs)

    @classmethod
    def from_examples(
        cls,
        examples: list[str],
        suffix: str,
        input_variables: list[str],
        example_separator: str = "\n\n",
        prefix: str = "",
        **kwargs: Any,
    ) -> PromptTemplate:
        """使用列表格式的示例，包含前缀和后缀来创建提示词。

        旨在作为从示例动态创建提示词的一种方式。

        Args:
            examples: 用于提示词的示例列表。
            suffix: 在示例列表之后添加的字符串。

                通常用于设置用户的输入。 input_variables:
                最终提示词模板将期望的变量名称列表。 example_separator:
                示例之间的分隔符。 prefix: 在任何示例之前添加的字符串。

                通常包含示例。

        Returns:
            生成的最终提示词。
        """
        template = example_separator.join([prefix, *examples, suffix])
        return cls(input_variables=input_variables, template=template, **kwargs)

    @classmethod
    def from_file(
        cls,
        template_file: str | Path,
        encoding: str | None = None,
        **kwargs: Any,
    ) -> PromptTemplate:
        """从文件加载提示词。

        Args:
            template_file: 包含提示词模板的文件路径。
            encoding: 打开模板文件时使用的编码系统。

                如果未提供，将使用操作系统的默认编码。

        Returns:
            从文件加载的提示词。
        """
        template = Path(template_file).read_text(encoding=encoding)
        return cls.from_template(template=template, **kwargs)

    @classmethod
    def from_template(
        cls,
        template: str,
        *,
        template_format: PromptTemplateFormat = "f-string",
        partial_variables: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> PromptTemplate:
        """从模板加载提示词模板。

        !!! warning "安全"

            Prefer using `template_format='f-string'` instead of
            `template_format='jinja2'`, or make sure to NEVER accept jinja2 templates
            from untrusted sources as they may lead to arbitrary Python code execution.

            自 LangChain 0.0.329 版本起，Jinja2 模板将默认使用 Jinja2 的
            SandboxedEnvironment
            进行渲染。这种沙箱机制应被视为一种尽力而为的安全措施，而非安全保证，因为它是一种可选择退出而非选择加入的方式。

            尽管有沙箱保护，我们仍建议永远不要使用来自不受信任来源的 jinja2 模板。

        Args:
            template: 要加载的模板。
            template_format: 模板的格式。

                Use `jinja2` for jinja2, `mustache` for mustache, and `f-string` for
                f-strings.
            partial_variables: A dictionary of variables that can be used to partially
                fill in the template.

                For example, if the template is `'{variable1} {variable2}'`, and
                `partial_variables` is `{"variable1": "foo"}`, then the final prompt
                will be `'foo {variable2}'`.
            **kwargs: Any other arguments to pass to the prompt template.

        Returns:
            从模板加载的提示词模板。
        """
        input_variables = get_template_variables(template, template_format)
        partial_variables_ = partial_variables or {}

        if partial_variables_:
            input_variables = [
                var for var in input_variables if var not in partial_variables_
            ]

        return cls(
            input_variables=input_variables,
            template=template,
            template_format=template_format,
            partial_variables=partial_variables_,
            **kwargs,
        )

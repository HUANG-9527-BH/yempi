"""BasePrompt 模式定义。"""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from string import Formatter
from typing import TYPE_CHECKING, Any, Literal, cast

from pydantic import BaseModel, create_model
from typing_extensions import override

from langchain_core.prompt_values import PromptValue, StringPromptValue
from langchain_core.prompts.base import BasePromptTemplate
from langchain_core.utils import get_colored_text, mustache
from langchain_core.utils.formatting import formatter
from langchain_core.utils.interactive_env import is_interactive_env

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

try:
    from jinja2 import meta
    from jinja2.sandbox import SandboxedEnvironment

    _HAS_JINJA2 = True
except ImportError:
    _HAS_JINJA2 = False

PromptTemplateFormat = Literal["f-string", "mustache", "jinja2"]


def jinja2_formatter(template: str, /, **kwargs: Any) -> str:
    """使用 jinja2 格式化模板。

    !!! warning "安全"

        从 LangChain 0.0.329 版本开始，此方法默认使用 Jinja2 的
        `SandboxedEnvironment`。
        然而，这种沙箱机制应视为尽力而为的安全措施，而非安全保证。

        Do not accept jinja2 templates from untrusted sources as they may lead
        to arbitrary Python code execution.

        [更多信息。](https://jinja.palletsprojects.com/en/3.1.x/sandbox/)

    Args:
        template: 模板字符串。
        **kwargs: 用于格式化模板的变量。

    Returns:
        格式化后的字符串。

    Raises:
        ImportError: 如果未安装 jinja2。
    """
    if not _HAS_JINJA2:
        msg = (
            "jinja2 not installed, which is needed to use the jinja2_formatter. "
            "Please install it with `pip install jinja2`."
            "Please be cautious when using jinja2 templates. "
            "Do not expand jinja2 templates using unverified or user-controlled "
            "inputs as that can result in arbitrary Python code execution."
        )
        raise ImportError(msg)

        # 使用 Jinja2 的 SandboxedEnvironment，阻止对双下划线属性的访问
        # (例如 __class__, __globals__) 以防止沙箱逃逸。
        # 注意：常规属性访问（例如 {{obj.attr}}）和方法调用是
        # 仍允许。这是尽力而为的措施——不要与不可信的模板一起使用
        # 模板
    return SandboxedEnvironment().from_string(template).render(**kwargs)


def validate_jinja2(template: str, input_variables: list[str]) -> None:
    """验证输入变量是否适用于该模板。

    如果发现缺失或多余的变量，将发出警告。

    Args:
        template: 模板字符串。
        input_variables: 输入变量。
    """
    input_variables_set = set(input_variables)
    valid_variables = _get_jinja2_variables_from_template(template)
    missing_variables = valid_variables - input_variables_set
    extra_variables = input_variables_set - valid_variables

    warning_message = ""
    if missing_variables:
        warning_message += f"Missing variables: {missing_variables} "

    if extra_variables:
        warning_message += f"Extra variables: {extra_variables}"

    if warning_message:
        warnings.warn(warning_message.strip(), stacklevel=7)


def _get_jinja2_variables_from_template(template: str) -> set[str]:
    if not _HAS_JINJA2:
        msg = (
            "jinja2 not installed, which is needed to use the jinja2_formatter. "
            "Please install it with `pip install jinja2`."
        )
        raise ImportError(msg)
    env = SandboxedEnvironment()
    ast = env.parse(template)
    return meta.find_undeclared_variables(ast)


def mustache_formatter(template: str, /, **kwargs: Any) -> str:
    """使用 mustache 格式化模板。

    Args:
        template: 模板字符串。
        **kwargs: 用于格式化模板的变量。

    Returns:
        格式化后的字符串。
    """
    return mustache.render(template, kwargs)


def mustache_template_vars(
    template: str,
) -> set[str]:
    """获取 mustache 模板中的顶层变量。

    对于嵌套变量如 `{{person.name}}`，仅返回顶层键（`person`）。

    Args:
        template: 模板字符串。

    Returns:
        模板中的顶层变量。
    """
    variables: set[str] = set()
    section_depth = 0
    for type_, key in mustache.tokenize(template):
        if type_ == "end":
            section_depth -= 1
        elif (
            type_ in {"variable", "section", "inverted section", "no escape"}
            and key != "."
            and section_depth == 0
        ):
            variables.add(key.split(".")[0])
        if type_ in {"section", "inverted section"}:
            section_depth += 1
    return variables


Defs = dict[str, "Defs"]


def mustache_schema(template: str) -> type[BaseModel]:
    """获取 mustache 模板中的变量。

    Args:
        template: 模板字符串。

    Returns:
        模板中的变量，以 Pydantic 模型形式返回。
    """
    fields = {}
    prefix: tuple[str, ...] = ()
    section_stack: list[tuple[str, ...]] = []
    for type_, key in mustache.tokenize(template):
        if key == ".":
            continue
        if type_ == "end":
            if section_stack:
                prefix = section_stack.pop()
        elif type_ in {"section", "inverted section"}:
            section_stack.append(prefix)
            prefix += tuple(key.split("."))
            fields[prefix] = False
        elif type_ in {"variable", "no escape"}:
            fields[prefix + tuple(key.split("."))] = True

    for fkey, fval in fields.items():
        fields[fkey] = fval and not any(
            is_subsequence(fkey, k) for k in fields if k != fkey
        )
    defs: Defs = {}  # None 表示叶节点
    while fields:
        field, is_leaf = fields.popitem()
        current = defs
        for part in field[:-1]:
            current = current.setdefault(part, {})
        current.setdefault(field[-1], "" if is_leaf else {})  # type: ignore[arg-type]
    return _create_model_recursive("PromptInput", defs)


def _create_model_recursive(name: str, defs: Defs) -> type[BaseModel]:
    return cast(
        "type[BaseModel]",
        create_model(  # type: ignore[call-overload]
            name,
            **{
                k: (_create_model_recursive(k, v), None) if v else (type(v), None)
                for k, v in defs.items()
            },
        ),
    )


DEFAULT_FORMATTER_MAPPING: dict[str, Callable[..., str]] = {
    "f-string": formatter.format,
    "mustache": mustache_formatter,
    "jinja2": jinja2_formatter,
}

DEFAULT_VALIDATOR_MAPPING: dict[str, Callable[[str, list[str]], None]] = {
    "f-string": formatter.validate_input_variables,
    "jinja2": validate_jinja2,
}


def _parse_f_string_fields(template: str) -> list[tuple[str, str | None]]:
    fields: list[tuple[str, str | None]] = []
    for _, field_name, format_spec, _ in Formatter().parse(template):
        if field_name is not None:
            fields.append((field_name, format_spec))
    return fields


def validate_f_string_template(template: str) -> list[str]:
    """验证 f-string 模板并返回其输入变量。"""
    input_variables = set()
    for var, format_spec in _parse_f_string_fields(template):
        if "." in var or "[" in var or "]" in var:
            msg = (
                f"Invalid variable name {var!r} in f-string template. "
                f"Variable names cannot contain attribute "
                f"access (.) or indexing ([])."
            )
            raise ValueError(msg)

        if var.isdigit():
            msg = (
                f"Invalid variable name {var!r} in f-string template. "
                f"Variable names cannot be all digits as they are interpreted "
                f"as positional arguments."
            )
            raise ValueError(msg)

        if format_spec and ("{" in format_spec or "}" in format_spec):
            msg = (
                "Invalid format specifier in f-string template. "
                "Nested replacement fields are not allowed."
            )
            raise ValueError(msg)

        input_variables.add(var)

    return sorted(input_variables)


def check_valid_template(
    template: str, template_format: str, input_variables: list[str]
) -> None:
    """检查模板字符串是否有效。

    Args:
        template: 模板字符串。
        template_format: 模板格式。

            应为 `'f-string'` 或 `'jinja2'` 之一。
        input_variables: 输入变量。

    Raises:
        ValueError: 如果模板格式不受支持。
        ValueError: 如果提示词模式无效。
    """
    try:
        validator_func = DEFAULT_VALIDATOR_MAPPING[template_format]
    except KeyError as exc:
        msg = (
            f"Invalid template format {template_format!r}, should be one of"
            f" {list(DEFAULT_FORMATTER_MAPPING)}."
        )
        raise ValueError(msg) from exc
    if template_format == "f-string":
        validate_f_string_template(template)
    try:
        validator_func(template, input_variables)
    except (KeyError, IndexError) as exc:
        msg = (
            "Invalid prompt schema; check for mismatched or missing input parameters"
            f" from {input_variables}."
        )
        raise ValueError(msg) from exc


def get_template_variables(template: str, template_format: str) -> list[str]:
    """获取模板中的变量。

    Args:
        template: 模板字符串。
        template_format: 模板格式。

            应为 `'f-string'`、`'mustache'` 或 `'jinja2'` 之一。

    Returns:
        模板中的变量。

    Raises:
        ValueError: 如果模板格式不受支持。
    """
    input_variables: list[str] | set[str]
    if template_format == "jinja2":
                # 获取模板的变量
        input_variables = sorted(_get_jinja2_variables_from_template(template))
    elif template_format == "f-string":
        input_variables = validate_f_string_template(template)
    elif template_format == "mustache":
        input_variables = mustache_template_vars(template)
    else:
        msg = f"Unsupported template format: {template_format}"
        raise ValueError(msg)

    return sorted(input_variables)


class StringPromptTemplate(BasePromptTemplate[str], ABC):
    """暴露 format 方法的字符串提示词，返回一个提示词。"""

    @classmethod
    def get_lc_namespace(cls) -> list[str]:
        """获取 LangChain 对象的命名空间。

        Returns:
            `["langchain", "prompts", "base"]`
        """
        return ["langchain", "prompts", "base"]

    def format_prompt(self, **kwargs: Any) -> PromptValue:
        """使用输入格式化提示词。

        Args:
            **kwargs: 传递给提示词模板的任意参数。

        Returns:
            格式化后的字符串。
        """
        return StringPromptValue(text=self.format(**kwargs))

    async def aformat_prompt(self, **kwargs: Any) -> PromptValue:
        """异步使用输入格式化提示词。

        Args:
            **kwargs: 传递给提示词模板的任意参数。

        Returns:
            格式化后的字符串。
        """
        return StringPromptValue(text=await self.aformat(**kwargs))

    @override
    @abstractmethod
    def format(self, **kwargs: Any) -> str: ...

    def pretty_repr(
        self,
        html: bool = False,  # noqa: FBT001,FBT002
    ) -> str:
        """获取提示词的美化表示形式。

        Args:
            html: 是否返回 HTML 格式的字符串。

        Returns:
            提示词的美化表示形式。
        """
                # TODO: 处理部分模板
        dummy_vars = {
            input_var: "{" + f"{input_var}" + "}" for input_var in self.input_variables
        }
        if html:
            dummy_vars = {
                k: get_colored_text(v, "yellow") for k, v in dummy_vars.items()
            }
        return self.format(**dummy_vars)

    def pretty_print(self) -> None:
        """打印提示词的美化表示形式。"""
        print(self.pretty_repr(html=is_interactive_env()))  # noqa: T201


def is_subsequence(child: Sequence[Any], parent: Sequence[Any]) -> bool:
    """如果 child 是 parent 的子序列，则返回 True。"""
    if len(child) == 0 or len(parent) == 0:
        return False
    if len(parent) < len(child):
        return False
    return all(child[i] == parent[i] for i in range(len(child)))

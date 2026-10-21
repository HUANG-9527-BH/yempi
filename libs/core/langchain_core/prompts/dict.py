"""字典提示词模板。"""

import warnings
from functools import cached_property
from typing import Any, Literal, cast

from pydantic import model_validator
from typing_extensions import override

from langchain_core.load import dumpd
from langchain_core.prompts.string import (
    DEFAULT_FORMATTER_MAPPING,
    get_template_variables,
)
from langchain_core.runnables import RunnableConfig, RunnableSerializable
from langchain_core.runnables.config import ensure_config


class DictPromptTemplate(RunnableSerializable[dict[str, Any], dict[str, Any]]):
    """以字典表示的模板。

    识别 f-string 或 mustache 格式字符串字典值中的变量。

    不识别字典键中的变量。递归应用。

    Example:
        ```python
        prompt = DictPromptTemplate(
            template={
                "type": "text",
                "text": "Hello {name}",
                "metadata": {"source": "{source}"},
            },
            template_format="f-string",
        )
        prompt.format(name="Alice", source="docs")
        # {
        #     "type": "text",
        #     "text": "Hello Alice",
        #     "metadata": {"source": "docs"},
        # }
        ```
    """

    template: dict[str, Any]
    template_format: Literal["f-string", "mustache"]

    @model_validator(mode="after")
    def validate_template(self) -> "DictPromptTemplate":
        """验证模板结构仅包含安全变量。"""
        _get_input_variables(self.template, self.template_format)
        return self

    @property
    def input_variables(self) -> list[str]:
        """模板输入变量。"""
        return _get_input_variables(self.template, self.template_format)

    def format(self, **kwargs: Any) -> dict[str, Any]:
        """使用输入格式化提示词。

        Returns:
            格式化后的字典。
        """
        return _insert_input_variables(self.template, kwargs, self.template_format)

    async def aformat(self, **kwargs: Any) -> dict[str, Any]:
        """使用输入格式化提示词。

        Returns:
            格式化后的字典。
        """
        return self.format(**kwargs)

    @override
    def invoke(
        self, input: dict[str, Any], config: RunnableConfig | None = None, **kwargs: Any
    ) -> dict[str, Any]:
        return self._call_with_config(
            lambda x: self.format(**x),
            input,
            ensure_config(config),
            run_type="prompt",
            serialized=self._serialized,
            **kwargs,
        )

    @property
    def _prompt_type(self) -> str:
        return "dict-prompt"

    @cached_property
    def _serialized(self) -> dict[str, Any]:
                # self 在此情况下始终是 Serializable 对象，因此结果
                # 保证为字典，因为 dumpd 使用默认回调，该回调使用
                # obj.to_json，其始终返回 TypedDict 子类
        return cast("dict[str, Any]", dumpd(self))

    @classmethod
    def is_lc_serializable(cls) -> bool:
        """返回 `True`，因为此类可序列化。"""
        return True

    @classmethod
    def get_lc_namespace(cls) -> list[str]:
        """获取 LangChain 对象的命名空间。

        Returns:
            `["langchain_core", "prompts", "dict"]`
        """
        return ["langchain_core", "prompts", "dict"]

    def pretty_repr(self, *, html: bool = False) -> str:
        """可读性表示。

        Args:
            html: 是否以 HTML 格式化。

        Returns:
            可读性表示。
        """
        raise NotImplementedError


def _get_input_variables(
    template: dict[str, Any], template_format: Literal["f-string", "mustache"]
) -> list[str]:
    input_variables = []
    for v in template.values():
        if isinstance(v, str):
            input_variables += get_template_variables(v, template_format)
        elif isinstance(v, dict):
            input_variables += _get_input_variables(v, template_format)
        elif isinstance(v, (list, tuple)):
            for x in v:
                if isinstance(x, str):
                    input_variables += get_template_variables(x, template_format)
                elif isinstance(x, dict):
                    input_variables += _get_input_variables(x, template_format)
    return list(set(input_variables))


def _insert_input_variables(
    template: dict[str, Any],
    inputs: dict[str, Any],
    template_format: Literal["f-string", "mustache"],
) -> dict[str, Any]:
    formatted: dict[str, Any] = {}
    formatter = DEFAULT_FORMATTER_MAPPING[template_format]
    for k, v in template.items():
        if isinstance(v, str):
            formatted[k] = formatter(v, **inputs)
        elif isinstance(v, dict):
            if k == "image_url" and "path" in v:
                msg = (
                    "Specifying image inputs via file path in environments with "
                    "user-input paths is a security vulnerability. Out of an abundance "
                    "of caution, the utility has been removed to prevent possible "
                    "misuse."
                )
                warnings.warn(msg, stacklevel=2)
            formatted[k] = _insert_input_variables(v, inputs, template_format)
        elif isinstance(v, (list, tuple)):
            formatted_v: list[Any] = []
            for x in v:
                if isinstance(x, str):
                    formatted_v.append(formatter(x, **inputs))
                elif isinstance(x, dict):
                    formatted_v.append(
                        _insert_input_variables(x, inputs, template_format)
                    )
                else:
                    formatted_v.append(x)
            formatted[k] = type(v)(formatted_v)
        else:
            formatted[k] = v
    return formatted

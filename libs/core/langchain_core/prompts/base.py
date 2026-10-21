"""提示词模板的基类。"""

from __future__ import annotations

import builtins
import contextlib
import json
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, Any, Generic, TypeVar, cast

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator
from typing_extensions import Self, override

from langchain_core._api import deprecated, suppress_langchain_deprecation_warning
from langchain_core.exceptions import ErrorCode, create_message
from langchain_core.load import dumpd
from langchain_core.output_parsers.base import BaseOutputParser
from langchain_core.prompt_values import (
    ChatPromptValueConcrete,
    PromptValue,
    StringPromptValue,
)
from langchain_core.runnables.base import RunnableSerializable
from langchain_core.runnables.config import RunnableConfig, ensure_config
from langchain_core.utils.pydantic import create_model_v2

if TYPE_CHECKING:
    from langchain_core.documents import Document


FormatOutputType = TypeVar("FormatOutputType")


class BasePromptTemplate(
    RunnableSerializable[dict[str, Any], PromptValue], ABC, Generic[FormatOutputType]
):
    """所有提示词模板的基类，返回一个提示词。"""

    input_variables: list[str]
    """生成提示词所需的变量名称列表。
    """

    optional_variables: list[str] = Field(default=[])
    """A list of the names of the variables for placeholder or `MessagePlaceholder` that
    are optional.

    这些变量会自动从提示词中推断，用户无需提供。
    """

    input_types: builtins.dict[str, Any] = Field(default_factory=dict, exclude=True)
    """提示词模板期望的变量类型字典。

    如果不提供，则假设所有变量均为字符串。
    """

        # 理想情况下，我们应将 output_parser 类型设为 BaseOutputParser[Any]
        # 但这样会导致 Pydantic 失败（Pydantic 会尝试实例化 BaseOutputParser
        # 而不是使用提供的 output_parser...)
    output_parser: BaseOutputParser | None = None  # type: ignore[type-arg]
    """如何解析在该格式化提示词上调用 LLM 的输出。"""

    partial_variables: Mapping[str, Any] = Field(default_factory=dict)
    """提示词模板携带的局部变量字典。

    局部变量会填充到模板中，因此每次调用提示词时无需重复传入。
    """

    metadata: builtins.dict[str, Any] | None = None
    """用于追踪的元数据。"""

    tags: list[str] | None = None
    """用于追踪的标签。"""

    @model_validator(mode="after")
    def validate_variable_names(self) -> Self:
        """验证变量名称不包含受限名称。"""
        if "stop" in self.input_variables:
            msg = (
                "Cannot have an input variable named 'stop', as it is used internally,"
                " please rename."
            )
            raise ValueError(
                create_message(message=msg, error_code=ErrorCode.INVALID_PROMPT_INPUT)
            )
        if "stop" in self.partial_variables:
            msg = (
                "Cannot have an partial variable named 'stop', as it is used "
                "internally, please rename."
            )
            raise ValueError(
                create_message(message=msg, error_code=ErrorCode.INVALID_PROMPT_INPUT)
            )

        overall = set(self.input_variables).intersection(self.partial_variables)
        if overall:
            msg = f"Found overlapping input and partial variables: {overall}"
            raise ValueError(
                create_message(message=msg, error_code=ErrorCode.INVALID_PROMPT_INPUT)
            )
        return self

    @classmethod
    def get_lc_namespace(cls) -> list[str]:
        """获取 LangChain 对象的命名空间。

        Returns:
            `["langchain", "schema", "prompt_template"]`
        """
        return ["langchain", "schema", "prompt_template"]

    @classmethod
    def is_lc_serializable(cls) -> bool:
        """返回 `True`，因为此类是可序列化的。"""
        return True

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )

    @cached_property
    def _serialized(self) -> builtins.dict[str, Any]:
                # 在此情况下，self 始终是 Serializable 对象，因此结果
                # 保证是字典，因为 dumpd 使用默认回调，该回调使用
                # obj.to_json，其始终返回 TypedDict 子类
        return cast("builtins.dict[str, Any]", dumpd(self))

    @property
    @override
    def OutputType(self) -> Any:
        """返回提示词的输出类型。"""
        return StringPromptValue | ChatPromptValueConcrete

    @override
    def get_input_schema(self, config: RunnableConfig | None = None) -> type[BaseModel]:
        """获取提示词的输入模式。

        Args:
            config: 提示词的配置。

        Returns:
            提示词的输入模式。
        """
                # 这是正确的，但 pydantic 类型/ mypy 不这么认为。
        required_input_variables = {
            k: (self.input_types.get(k, str), ...) for k in self.input_variables
        }
        optional_input_variables = {
            k: (self.input_types.get(k, str), None) for k in self.optional_variables
        }
        return create_model_v2(
            "PromptInput",
            field_definitions={**required_input_variables, **optional_input_variables},
        )

    def _validate_input(self, inner_input: Any) -> builtins.dict[str, Any]:
        if not isinstance(inner_input, dict):
            if len(self.input_variables) == 1:
                var_name = self.input_variables[0]
                inner_input_ = {var_name: inner_input}

            else:
                msg = (
                    f"Expected mapping type as input to {self.__class__.__name__}. "
                    f"Received {type(inner_input)}."
                )
                raise TypeError(
                    create_message(
                        message=msg, error_code=ErrorCode.INVALID_PROMPT_INPUT
                    )
                )
        else:
            inner_input_ = inner_input
        missing = set(self.input_variables).difference(inner_input_)
        if missing:
            msg = (
                f"Input to {self.__class__.__name__} is missing variables {missing}. "
                f" Expected: {self.input_variables}"
                f" Received: {list(inner_input_.keys())}"
            )
            example_key = missing.pop()
            msg += (
                f"\nNote: if you intended {{{example_key}}} to be part of the string"
                " and not a variable, please escape it with double curly braces like: "
                f"'{{{{{example_key}}}}}'."
            )
            raise KeyError(
                create_message(message=msg, error_code=ErrorCode.INVALID_PROMPT_INPUT)
            )
        return inner_input_

    def _format_prompt_with_error_handling(
        self, inner_input: builtins.dict[str, Any]
    ) -> PromptValue:
        inner_input_ = self._validate_input(inner_input)
        return self.format_prompt(**inner_input_)

    async def _aformat_prompt_with_error_handling(
        self, inner_input: builtins.dict[str, Any]
    ) -> PromptValue:
        inner_input_ = self._validate_input(inner_input)
        return await self.aformat_prompt(**inner_input_)

    @override
    def invoke(
        self,
        input: builtins.dict[str, Any],
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> PromptValue:
        """调用提示词。

        Args:
            input: 提示词的输入。
            config: 提示词的配置。

        Returns:
            提示词的输出。
        """
        config = ensure_config(config)
        if self.metadata:
            config["metadata"] = {**config["metadata"], **self.metadata}
        if self.tags:
            config["tags"] += self.tags
        return self._call_with_config(
            self._format_prompt_with_error_handling,
            input,
            config,
            run_type="prompt",
            serialized=self._serialized,
        )

    @override
    async def ainvoke(
        self,
        input: builtins.dict[str, Any],
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> PromptValue:
        """异步调用提示词。

        Args:
            input: 提示词的输入。
            config: 提示词的配置。

        Returns:
            提示词的输出。
        """
        config = ensure_config(config)
        if self.metadata:
            config["metadata"].update(self.metadata)
        if self.tags:
            config["tags"].extend(self.tags)
        return await self._acall_with_config(
            self._aformat_prompt_with_error_handling,
            input,
            config,
            run_type="prompt",
            serialized=self._serialized,
        )

    @abstractmethod
    def format_prompt(self, **kwargs: Any) -> PromptValue:
        """创建 `PromptValue`。

        Args:
            **kwargs: 传递给提示词模板的任意参数。

        Returns:
            提示词的输出结果。
        """

    async def aformat_prompt(self, **kwargs: Any) -> PromptValue:
        """异步创建 `PromptValue`。

        Args:
            **kwargs: 传递给提示词模板的任意参数。

        Returns:
            提示词的输出结果。
        """
        return self.format_prompt(**kwargs)

    def partial(
        self, **kwargs: str | Callable[[], str]
    ) -> BasePromptTemplate[FormatOutputType]:
        """返回提示词模板的部分实例。

        Args:
            **kwargs: 要设置的部分变量。

        Returns:
            提示词模板的部分实例。
        """
        prompt_dict = self.__dict__.copy()
        prompt_dict["input_variables"] = list(
            set(self.input_variables).difference(kwargs)
        )
        prompt_dict["partial_variables"] = {**self.partial_variables, **kwargs}
        return type(self)(**prompt_dict)

    def _merge_partial_and_user_variables(
        self, **kwargs: Any
    ) -> builtins.dict[str, Any]:
                # 获取部分参数：
        partial_kwargs = {
            k: v if not callable(v) else v() for k, v in self.partial_variables.items()
        }
        return {**partial_kwargs, **kwargs}

    @abstractmethod
    def format(self, **kwargs: Any) -> FormatOutputType:
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

    async def aformat(self, **kwargs: Any) -> FormatOutputType:
        """异步使用输入格式化提示词。

        Args:
            **kwargs: 传递给提示词模板的任意参数。

        Returns:
            格式化后的字符串。

        Example:
            ```python
            await prompt.aformat(variable1="foo")
            ```
        """
        return self.format(**kwargs)

    @property
    def _prompt_type(self) -> str:
        """返回提示词类型键。"""
        raise NotImplementedError

    @deprecated("1.4.2", alternative="asdict", removal="2.0.0")
    @override
    def dict(self, **kwargs: Any) -> builtins.dict[str, Any]:
        """已弃用 - 请改用 `asdict()`。

        返回提示词的字典表示形式。
        """
        return self.asdict(**kwargs)

    def asdict(self, **kwargs: Any) -> builtins.dict[str, Any]:
        """返回提示词的字典表示形式。

        Args:
            **kwargs: 传递给字典的任意附加参数。

        Returns:
            提示词的字典表示形式。
        """
        prompt_dict = super().model_dump(**kwargs)
        with contextlib.suppress(NotImplementedError):
            prompt_dict["_type"] = self._prompt_type
        return prompt_dict

    def _dict_for_compat(self) -> builtins.dict[str, Any]:
        """返回保留已弃用覆盖项的提示词字典。"""
        with suppress_langchain_deprecation_warning():
            return self.dict()

    @deprecated(
        since="1.2.21",
        removal="2.0.0",
        alternative="Use `dumpd`/`dumps` from `langchain_core.load` to serialize "
        "prompts and `load`/`loads` to deserialize them.",
    )
    def save(self, file_path: Path | str) -> None:
        """保存提示词。

        Args:
            file_path: 保存提示词的目录路径。

        Raises:
            ValueError: 如果提示词包含部分变量。
            ValueError: 如果文件路径不是 json 或 yaml 格式。
            NotImplementedError: 如果提示词类型未实现。

        Example:
            ```python
            prompt.save(file_path="path/prompt.yaml")
            ```
        """
        if self.partial_variables:
            msg = "Cannot save prompt with partial variables."
            raise ValueError(msg)

                # 获取要保存的字典。保留已弃用的 `dict()` 覆盖项，直到
                # `dict()` 被移除。
        prompt_dict = self._dict_for_compat()
        if "_type" not in prompt_dict:
            msg = f"Prompt {self} does not support saving."
            raise NotImplementedError(msg)

                # 将文件路径转换为 Path 对象。
        save_path = Path(file_path)

        directory_path = save_path.parent
        directory_path.mkdir(parents=True, exist_ok=True)

        resolved_path = save_path.resolve()
        if resolved_path.suffix == ".json":
            with resolved_path.open("w", encoding="utf-8") as f:
                json.dump(prompt_dict, f, indent=4)
        elif resolved_path.suffix.endswith((".yaml", ".yml")):
            with resolved_path.open("w", encoding="utf-8") as f:
                yaml.dump(prompt_dict, f, default_flow_style=False)
        else:
            msg = f"{save_path} must be json or yaml"
            raise ValueError(msg)


def _get_document_info(
    doc: Document, prompt: BasePromptTemplate[str]
) -> dict[str, Any]:
    base_info = {"page_content": doc.page_content, **doc.metadata}
    missing_metadata = set(prompt.input_variables).difference(base_info)
    if len(missing_metadata) > 0:
        required_metadata = [
            iv for iv in prompt.input_variables if iv != "page_content"
        ]
        msg = (
            f"Document prompt requires documents to have metadata variables: "
            f"{required_metadata}. Received document with missing metadata: "
            f"{list(missing_metadata)}."
        )
        raise ValueError(
            create_message(message=msg, error_code=ErrorCode.INVALID_PROMPT_INPUT)
        )
    return {k: base_info[k] for k in prompt.input_variables}


def format_document(doc: Document, prompt: BasePromptTemplate[str]) -> str:
    """根据提示词模板将文档格式化为字符串。

    首先，此方法从文档中获取信息，来源如下：

    1. `page_content`: This takes the information from the `document.page_content` and
        assigns it to a variable named `page_content`.
    2. `metadata`: This takes information from `document.metadata` and assigns it to
        variables of the same name.

    然后将这些变量传递给 `prompt`，以生成格式化后的字符串。

    Args:
        doc: `Document`，`page_content` 和 `metadata` 将用于创建最终字符串。 prompt:
        `BasePromptTemplate`，将用于将 `page_content` 和 `metadata` 格式化为最终字符串。

    Returns:
        格式化后的文档字符串。

    Example:
        ```python
        from langchain_core.documents import Document
        from langchain_core.prompts import PromptTemplate

        doc = Document(page_content="This is a joke", metadata={"page": "1"})
        prompt = PromptTemplate.from_template("Page {page}: {page_content}")
        format_document(doc, prompt)
        # -> "Page 1: This is a joke"
        ```
    """
    return prompt.format(**_get_document_info(doc, prompt))


async def aformat_document(doc: Document, prompt: BasePromptTemplate[str]) -> str:
    """异步根据提示词模板将文档格式化为字符串。

    首先，此方法从文档中获取信息，来源如下：

    1. `page_content`: This takes the information from the `document.page_content` and
        assigns it to a variable named `page_content`.
    2. `metadata`: This takes information from `document.metadata` and assigns it to
        variables of the same name.

    然后将这些变量传递给 `prompt`，以生成格式化后的字符串。

    Args:
        doc: `Document`，`page_content` 和 `metadata` 将用于创建最终字符串。 prompt:
        `BasePromptTemplate`，将用于将 `page_content` 和 `metadata` 格式化为最终字符串。

    Returns:
        格式化后的文档字符串。
    """
    return await prompt.aformat(**_get_document_info(doc, prompt))

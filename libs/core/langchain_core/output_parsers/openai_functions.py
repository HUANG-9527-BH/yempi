"""OpenAI 函数输出的解析器。"""

import copy
import json
from typing import Any

import jsonpatch  # type: ignore[import-untyped]
from pydantic import BaseModel, model_validator
from pydantic.v1 import BaseModel as BaseModelV1
from typing_extensions import override

from langchain_core.exceptions import OutputParserException
from langchain_core.output_parsers import (
    BaseCumulativeTransformOutputParser,
    BaseGenerationOutputParser,
)
from langchain_core.output_parsers.json import parse_partial_json
from langchain_core.outputs import ChatGeneration, Generation
from langchain_core.utils.pydantic import PydanticBaseModel, TypeBaseModel


class OutputFunctionsParser(BaseGenerationOutputParser[Any]):
    """解析一个输出，该输出是值集合之一。"""

    args_only: bool = True
    """是否仅返回函数调用的参数。"""

    @override
    def parse_result(self, result: list[Generation], *, partial: bool = False) -> Any:
        """将 LLM 调用的结果解析为 JSON 对象。

        Args:
            result: LLM 调用的结果。
            partial: 是否解析部分 JSON 对象。

        Returns:
            解析后的 JSON 对象。

        Raises:
            OutputParserException: 如果输出不是有效的 JSON。
        """
        generation = result[0]
        if not isinstance(generation, ChatGeneration):
            msg = "This output parser can only be used with a chat generation."
            raise OutputParserException(msg)
        message = generation.message
        try:
            func_call = copy.deepcopy(message.additional_kwargs["function_call"])
        except KeyError as exc:
            msg = f"Could not parse function call: {exc}"
            raise OutputParserException(msg) from exc

        if self.args_only:
            return func_call["arguments"]
        return func_call


class JsonOutputFunctionsParser(BaseCumulativeTransformOutputParser[Any]):
    """将输出解析为 JSON 对象。"""

    strict: bool = False
    """是否允许非 JSON 兼容的字符串。

    See: 参见：https://docs.python.org/3/library/json.html#encoders-and-decoders

    当解析的输出可能包含 Unicode 字符或换行符时很有用。
    """

    args_only: bool = True
    """是否仅返回函数调用的参数。"""

    @property
    def _type(self) -> str:
        return "json_functions"

    @override
    def _diff(self, prev: Any | None, next: Any) -> Any:
        return jsonpatch.make_patch(prev, next).patch

    def parse_result(self, result: list[Generation], *, partial: bool = False) -> Any:
        """将 LLM 调用的结果解析为 JSON 对象。

        Args:
            result: LLM 调用的结果。
            partial: 是否解析部分 JSON 对象。

        Returns:
            解析后的 JSON 对象。

        Raises:
            OutputParserException: 如果输出不是有效的 JSON。
        """
        if len(result) != 1:
            msg = f"Expected exactly one result, but got {len(result)}"
            raise OutputParserException(msg)
        generation = result[0]
        if not isinstance(generation, ChatGeneration):
            msg = "This output parser can only be used with a chat generation."
            raise OutputParserException(msg)
        message = generation.message
        try:
            function_call = message.additional_kwargs["function_call"]
        except KeyError as exc:
            if partial:
                return None
            msg = f"Could not parse function call: {exc}"
            raise OutputParserException(msg) from exc
        try:
            if partial:
                try:
                    if self.args_only:
                        return parse_partial_json(
                            function_call["arguments"], strict=self.strict
                        )
                    return {
                        **function_call,
                        "arguments": parse_partial_json(
                            function_call["arguments"], strict=self.strict
                        ),
                    }
                except json.JSONDecodeError:
                    return None
            elif self.args_only:
                try:
                    return json.loads(function_call["arguments"], strict=self.strict)
                except (json.JSONDecodeError, TypeError) as exc:
                    msg = f"Could not parse function call data: {exc}"
                    raise OutputParserException(msg) from exc
            else:
                try:
                    return {
                        **function_call,
                        "arguments": json.loads(
                            function_call["arguments"], strict=self.strict
                        ),
                    }
                except (json.JSONDecodeError, TypeError) as exc:
                    msg = f"Could not parse function call data: {exc}"
                    raise OutputParserException(msg) from exc
        except KeyError:
            return None

        # 此方法将由 `parse_result` 的默认实现调用
        # 但因为我们重写了该方法，所以不需要它。
    def parse(self, text: str) -> Any:
        """将 LLM 调用的输出解析为 JSON 对象。

        Args:
            text: LLM 调用的输出。

        Returns:
            解析后的 JSON 对象。
        """
        raise NotImplementedError


class JsonKeyOutputFunctionsParser(JsonOutputFunctionsParser):
    """将输出解析为 JSON 对象的元素。"""

    key_name: str
    """要返回的键的名称。"""

    def parse_result(self, result: list[Generation], *, partial: bool = False) -> Any:
        """将 LLM 调用的结果解析为 JSON 对象。

        Args:
            result: LLM 调用的结果。
            partial: 是否解析部分 JSON 对象。

        Returns:
            解析后的 JSON 对象。
        """
        res = super().parse_result(result, partial=partial)
        if partial and res is None:
            return None
        return res.get(self.key_name) if partial else res[self.key_name]


class PydanticOutputFunctionsParser(OutputFunctionsParser):
    """将输出解析为 Pydantic 对象。

    此解析器用于解析使用 OpenAI 函数格式调用函数的聊天模型的输出。

    解析器提取函数调用并将其与提供的 Pydantic 模式匹配。

    如果函数调用与提供的模式不匹配，则会引发异常。

    Example:
        ```python
        message = AIMessage(
            content="This is a test message",
            additional_kwargs={
                "function_call": {
                    "name": "cookie",
                    "arguments": json.dumps({"name": "value", "age": 10}),
                }
            },
        )
        chat_generation = ChatGeneration(message=message)


        class Cookie(BaseModel):
            name: str
            age: int


        class Dog(BaseModel):
            species: str


        # Full output
        parser = PydanticOutputFunctionsParser(
            pydantic_schema={"cookie": Cookie, "dog": Dog}
        )
        result = parser.parse_result([chat_generation])
        ```

    """

    pydantic_schema: TypeBaseModel | dict[str, TypeBaseModel]
    """用于解析输出的 Pydantic 模式。

    如果提供了多个模式，则将使用函数名称来确定使用哪个模式。
    """

    @model_validator(mode="before")
    @classmethod
    def validate_schema(cls, values: dict[str, Any]) -> Any:
        """验证 Pydantic 模式。

        Args:
            values: 要验证的值。

        Returns:
            验证后的值。

        Raises:
            ValueError: 如果模式不是 Pydantic 模式。
        """
        schema = values["pydantic_schema"]
        if "args_only" not in values:
            values["args_only"] = isinstance(schema, type) and issubclass(
                schema, BaseModel
            )
        elif values["args_only"] and isinstance(schema, dict):
            msg = (
                "If multiple pydantic schemas are provided then args_only should be"
                " False."
            )
            raise ValueError(msg)
        return values

    @override
    def parse_result(self, result: list[Generation], *, partial: bool = False) -> Any:
        """将 LLM 调用的结果解析为 JSON 对象。

        Args:
            result: LLM 调用的结果。
            partial: 是否解析部分 JSON 对象。

        Raises:
            ValueError: 如果 Pydantic 模式无效。

        Returns:
            解析后的 JSON 对象。
        """
        result_ = super().parse_result(result)
        pydantic_args: PydanticBaseModel
        if self.args_only:
            if isinstance(self.pydantic_schema, dict):
                msg = (
                    "Dict Pydantic schema unsupported with args_only: "
                    f"{self.pydantic_schema}"
                )
                raise ValueError(msg)
            if issubclass(self.pydantic_schema, BaseModel):
                pydantic_args = self.pydantic_schema.model_validate_json(result_)
            elif issubclass(self.pydantic_schema, BaseModelV1):
                pydantic_args = self.pydantic_schema.parse_raw(result_)
            else:
                msg = (  # type: ignore[unreachable]
                    "Unsupported Pydantic schema with args_only: "
                    f"{self.pydantic_schema}"
                )
                raise ValueError(msg)
        else:
            fn_name = result_["name"]
            args = result_["arguments"]
            if isinstance(self.pydantic_schema, dict):
                pydantic_schema = self.pydantic_schema[fn_name]
            else:
                pydantic_schema = self.pydantic_schema
            if issubclass(pydantic_schema, BaseModel):
                pydantic_args = pydantic_schema.model_validate_json(args)
            elif issubclass(pydantic_schema, BaseModelV1):
                pydantic_args = pydantic_schema.parse_raw(args)
            else:
                msg = f"Unsupported Pydantic schema: {pydantic_schema}"  # type: ignore[unreachable]
                raise ValueError(msg)
        return pydantic_args


class PydanticAttrOutputFunctionsParser(PydanticOutputFunctionsParser):
    """将输出解析为 Pydantic 对象的属性。"""

    attr_name: str
    """要返回的属性的名称。"""

    @override
    def parse_result(self, result: list[Generation], *, partial: bool = False) -> Any:
        """将 LLM 调用的结果解析为 JSON 对象。

        Args:
            result: LLM 调用的结果。
            partial: 是否解析部分 JSON 对象。

        Returns:
            解析后的 JSON 对象。
        """
        result = super().parse_result(result)
        return getattr(result, self.attr_name)

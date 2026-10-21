"""用于格式化字符串的工具。"""

from collections.abc import Mapping, Sequence
from string import Formatter
from typing import Any


class StrictFormatter(Formatter):
    """一种强制使用关键字参数替换的字符串格式化器。

    This formatter extends Python's built-in `string.Formatter` to provide stricter
    validation for prompt template formatting. It ensures that all variable
    substitutions use keyword arguments rather than positional arguments, which improves
    clarity and reduces errors when formatting prompt templates.

    Example:
        >>> fmt = StrictFormatter()
        >>> fmt.format("Hello, {name}!", name="World")
        'Hello, World!'
        >>> fmt.format("Hello, {}!", "World")  # Raises ValueError
    """

    def vformat(
        self, format_string: str, args: Sequence[Any], kwargs: Mapping[str, Any]
    ) -> str:
        """使用关键字参数格式化字符串。

        覆盖基类 `vformat` 以拒绝位置参数，确保所有替换均为显式命名。

        Args:
            format_string: 包含替换字段的字符串（例如 `'{name}'`）。
            args: 位置参数（必须为空）。
            kwargs: 用于插入格式化字符串的关键字参数。

        Returns:
            替换所有占位符后的格式化字符串。

        Raises:
            ValueError: 如果提供了任何位置参数。
        """
        if len(args) > 0:
            msg = (
                "No arguments should be provided, "
                "everything should be passed as keyword arguments."
            )
            raise ValueError(msg)
        return super().vformat(format_string, args, kwargs)

    def validate_input_variables(
        self, format_string: str, input_variables: list[str]
    ) -> None:
        """验证输入变量是否与格式字符串中的占位符匹配。

        Checks that the provided input variables can be used to format the given string
        without missing or extra keys. This is useful for validating prompt templates
        before runtime.

        Args:
            format_string: 需要验证的包含替换字段的字符串 （例如 `'Hello, {name}!'`）。
            input_variables: 期望填充替换字段的变量名称列表。

        Raises:
            KeyError: 如果格式字符串包含未在 input_variables 中出现的占位符。

        Example:
            >>> fmt = StrictFormatter()
            >>> fmt.validate_input_variables("Hello, {name}!", ["name"])  # OK
            >>> fmt.validate_input_variables("Hello, {name}!", ["other"])  # Raises
        """
        dummy_inputs = dict.fromkeys(input_variables, "foo")
        super().format(format_string, **dummy_inputs)


#: LangChain 内部使用的默认 StrictFormatter 实例。
#: 在内部用于使用命名变量格式化提示词模板。
formatter = StrictFormatter()

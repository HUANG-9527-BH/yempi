import datetime
import warnings
from collections.abc import Sequence
from typing import Any, Literal

from langchain_core.utils import check_package_version
from typing_extensions import TypedDict

try:
    check_package_version("lark", gte_version="1.1.5")
    from lark import Lark, Transformer, v_args

    _HAS_LARK = True
except ImportError:

    def v_args(*_: Any, **__: Any) -> Any:  # type: ignore[misc]
        """当未安装 lark 时使用的空装饰器。"""
        return lambda _: None

    Transformer = object  # type: ignore[assignment,misc]
    Lark = object  # type: ignore[assignment,misc]
    _HAS_LARK = False

from langchain_core.structured_query import (
    Comparator,
    Comparison,
    FilterDirective,
    Operation,
    Operator,
)

GRAMMAR = r"""
    ?program: func_call
    ?expr: func_call
        | value

    func_call: CNAME "(" [args] ")"

    ?value: SIGNED_INT -> int
        | SIGNED_FLOAT -> float
        | DATE -> date
        | DATETIME -> datetime
        | list
        | string
        | ("false" | "False" | "FALSE") -> false
        | ("true" | "True" | "TRUE") -> true

    args: expr ("," expr)*
    DATE.2: /["']?(\d{4}-[01]\d-[0-3]\d)["']?/
    DATETIME.2: /["']?\d{4}-[01]\d-[0-3]\dT[0-2]\d:[0-5]\d:[0-5]\d[Zz]?["']?/
    string: /'[^']*'/ | ESCAPED_STRING
    list: "[" [args] "]"

    %import common.CNAME
    %import common.ESCAPED_STRING
    %import common.SIGNED_FLOAT
    %import common.SIGNED_INT
    %import common.WS
    %ignore WS
"""


class ISO8601Date(TypedDict):
    """ISO 8601 格式的日期（YYYY-MM-DD）。"""

    date: str
    type: Literal["date"]


class ISO8601DateTime(TypedDict):
    """ISO 8601 格式的日期时间（YYYY-MM-DDTHH:MM:SS）。"""

    datetime: str
    type: Literal["datetime"]


@v_args(inline=True)
class QueryTransformer(Transformer):
    """将查询字符串转换为中间表示形式。"""

    def __init__(
        self,
        *args: Any,
        allowed_comparators: Sequence[Comparator] | None = None,
        allowed_operators: Sequence[Operator] | None = None,
        allowed_attributes: Sequence[str] | None = None,
        **kwargs: Any,
    ):
        """初始化 QueryTransformer。

        Args:
            *args: 位置参数。
            allowed_comparators: 可选的允许比较符序列。
            allowed_operators: 可选的允许操作符序列。
            allowed_attributes: 可选的允许属性序列，用于比较符。
            **kwargs: 额外的关键字参数。
        """
        super().__init__(*args, **kwargs)
        self.allowed_comparators = allowed_comparators
        self.allowed_operators = allowed_operators
        self.allowed_attributes = allowed_attributes

    def program(self, *items: Any) -> tuple:
        """将项目转换为元组。"""
        return items

    def func_call(self, func_name: Any, args: list) -> FilterDirective:
        """将函数名和参数转换为 FilterDirective。

        Args:
            func_name: 函数名称。
            args: 传递给函数的参数。

        Returns:
            过滤指令。

        Raises:
            ValueError: 如果函数是比较符且第一个参数不在允许属性中。
        """
        func = self._match_func_name(str(func_name))
        if isinstance(func, Comparator):
            if self.allowed_attributes and args[0] not in self.allowed_attributes:
                msg = (
                    f"Received invalid attributes {args[0]}. Allowed attributes are "
                    f"{self.allowed_attributes}"
                )
                raise ValueError(msg)
            return Comparison(comparator=func, attribute=args[0], value=args[1])
        if len(args) == 1 and func in (Operator.AND, Operator.OR):
            return args[0]
        return Operation(operator=func, arguments=args)

    def _match_func_name(self, func_name: str) -> Operator | Comparator:
        if func_name in set(Comparator):
            if (
                self.allowed_comparators is not None
                and func_name not in self.allowed_comparators
            ):
                msg = (
                    f"Received disallowed comparator {func_name}. Allowed "
                    f"comparators are {self.allowed_comparators}"
                )
                raise ValueError(msg)
            return Comparator(func_name)
        if func_name in set(Operator):
            if (
                self.allowed_operators is not None
                and func_name not in self.allowed_operators
            ):
                msg = (
                    f"Received disallowed operator {func_name}. Allowed operators"
                    f" are {self.allowed_operators}"
                )
                raise ValueError(msg)
            return Operator(func_name)
        msg = (
            f"Received unrecognized function {func_name}. Valid functions are "
            f"{list(Operator) + list(Comparator)}"
        )
        raise ValueError(msg)

    def args(self, *items: Any) -> tuple:
        """将项目转换为元组。

        Args:
            items: 要转换的项目。
        """
        return items

    def false(self) -> bool:
        """返回 false。"""
        return False

    def true(self) -> bool:
        """返回 true。"""
        return True

    def list(self, item: Any) -> list:
        """将项目转换为列表。

        Args:
            item: 要转换的项目。
        """
        if item is None:
            return []
        return list(item)

    def int(self, item: Any) -> int:
        """将项目转换为整数。

        Args:
            item: 要转换的项目。
        """
        return int(item)

    def float(self, item: Any) -> float:
        """将项目转换为浮点数。

        Args:
            item: 要转换的项目。
        """
        return float(item)

    def date(self, item: Any) -> ISO8601Date:
        """将项目转换为 ISO8601Date 对象。

        Args:
            item: 要转换的项目。

        Raises:
            ValueError: 如果项目不是 ISO 8601 日期格式。
        """
        item = str(item).strip("\"'")
        try:
            datetime.datetime.strptime(item, "%Y-%m-%d")  # noqa: DTZ007
        except ValueError:
            warnings.warn(
                "Dates are expected to be provided in ISO 8601 date format "
                "(YYYY-MM-DD).",
                stacklevel=3,
            )
        return {"date": item, "type": "date"}

    def datetime(self, item: Any) -> ISO8601DateTime:
        """将项目转换为 ISO8601DateTime 对象。

        Args:
            item: 要转换的项目。

        Raises:
            ValueError: 如果项目不是 ISO 8601 日期时间格式。
        """
        item = str(item).strip("\"'")
        try:
                        # 解析完整的 ISO 8601 日期时间格式
            datetime.datetime.strptime(item, "%Y-%m-%dT%H:%M:%S%z")
        except ValueError:
            try:
                datetime.datetime.strptime(item, "%Y-%m-%dT%H:%M:%S")  # noqa: DTZ007
            except ValueError as e:
                msg = "Datetime values are expected to be in ISO 8601 format."
                raise ValueError(msg) from e
        return {"datetime": item, "type": "datetime"}

    def string(self, item: Any) -> str:
        """将项目转换为字符串。

        移除转义引号。

        Args:
            item: 要转换的项目。
        """
        return str(item).strip("\"'")


def get_parser(
    allowed_comparators: Sequence[Comparator] | None = None,
    allowed_operators: Sequence[Operator] | None = None,
    allowed_attributes: Sequence[str] | None = None,
) -> Lark:
    """返回查询语言的解析器。

    Args:
        allowed_comparators: 允许的比较符。
        allowed_operators: 允许的操作符。
        allowed_attributes: 允许的属性。

    Returns:
        查询语言的 Lark 解析器。
    """
    if not _HAS_LARK:
        msg = "Cannot import lark, please install it with 'pip install lark'."
        raise ImportError(msg)
    transformer = QueryTransformer(
        allowed_comparators=allowed_comparators,
        allowed_operators=allowed_operators,
        allowed_attributes=allowed_attributes,
    )
    return Lark(GRAMMAR, parser="lalr", transformer=transformer, start="program")

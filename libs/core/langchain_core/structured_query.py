"""结构化查询语言的内部表示。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

if TYPE_CHECKING:
    from collections.abc import Sequence


class Visitor(ABC):
    """使用访问者模式进行 IR 转换的接口定义。"""

    allowed_comparators: Sequence[Comparator] | None = None
    """访问者允许的比较器。"""

    allowed_operators: Sequence[Operator] | None = None
    """访问者允许的操作符。"""

    def _validate_func(self, func: Operator | Comparator) -> None:
        if (
            isinstance(func, Operator)
            and self.allowed_operators is not None
            and func not in self.allowed_operators
        ):
            msg = (
                f"Received disallowed operator {func}. Allowed "
                f"comparators are {self.allowed_operators}"
            )
            raise ValueError(msg)
        if (
            isinstance(func, Comparator)
            and self.allowed_comparators is not None
            and func not in self.allowed_comparators
        ):
            msg = (
                f"Received disallowed comparator {func}. Allowed "
                f"comparators are {self.allowed_comparators}"
            )
            raise ValueError(msg)

    @abstractmethod
    def visit_operation(self, operation: Operation) -> Any:
        """转换一个 Operation。

        Args:
            operation: 要转换的 Operation。
        """

    @abstractmethod
    def visit_comparison(self, comparison: Comparison) -> Any:
        """转换一个 Comparison。

        Args:
            comparison: 要转换的 Comparison。
        """

    @abstractmethod
    def visit_structured_query(self, structured_query: StructuredQuery) -> Any:
        """转换一个 StructuredQuery。

        Args:
            structured_query: 要转换的 StructuredQuery。
        """


def _to_snake_case(name: str) -> str:
    """将名称转换为 snake_case 格式。"""
    snake_case = ""
    for i, char in enumerate(name):
        if char.isupper() and i != 0:
            snake_case += "_" + char.lower()
        else:
            snake_case += char.lower()
    return snake_case


class Expr(BaseModel):
    """所有表达式的基类。"""

    def accept(self, visitor: Visitor) -> Any:
        """接受一个访问者。

        Args:
            visitor: 要接受的访问者。

        Returns:
            访问的结果。
        """
        return getattr(visitor, f"visit_{_to_snake_case(self.__class__.__name__)}")(
            self
        )


class Operator(str, Enum):
    """操作枚举。"""

    AND = "and"
    OR = "or"
    NOT = "not"


class Comparator(str, Enum):
    """比较操作符枚举。"""

    EQ = "eq"
    NE = "ne"
    GT = "gt"
    GTE = "gte"
    LT = "lt"
    LTE = "lte"
    CONTAIN = "contain"
    LIKE = "like"
    IN = "in"
    NIN = "nin"


class FilterDirective(Expr, ABC):
    """过滤表达式。"""


class Comparison(FilterDirective):
    """与值的比较。"""

    comparator: Comparator
    """要使用的比较器。"""

    attribute: str
    """要比较的属性。"""

    value: Any
    """要比较的值。"""

    def __init__(
        self, comparator: Comparator, attribute: str, value: Any, **kwargs: Any
    ) -> None:
        """创建一个 Comparison。

        Args:
            comparator: 要使用的比较器。
            attribute: 要比较的属性。
            value: 要比较的值。
        """
                # super exists from BaseModel
        super().__init__(
            comparator=comparator, attribute=attribute, value=value, **kwargs
        )


class Operation(FilterDirective):
    """对其他指令进行逻辑运算。"""

    operator: Operator
    """要使用的操作符。"""

    arguments: list[FilterDirective]
    """操作符的参数。"""

    def __init__(
        self, operator: Operator, arguments: list[FilterDirective], **kwargs: Any
    ) -> None:
        """创建一个 Operation。

        Args:
            operator: 要使用的操作符。
            arguments: 操作符的参数。
        """
                # super exists from BaseModel
        super().__init__(operator=operator, arguments=arguments, **kwargs)


class StructuredQuery(Expr):
    """结构化查询。"""

    query: str
    """查询字符串。"""

    filter: FilterDirective | None
    """过滤表达式。"""

    limit: int | None
    """结果数量限制。"""

    def __init__(
        self,
        query: str,
        filter: FilterDirective | None,  # noqa: A002
        limit: int | None = None,
        **kwargs: Any,
    ) -> None:
        """创建一个 StructuredQuery。

        Args:
            query: 查询字符串。
            filter: 过滤表达式。
            limit: 结果数量限制。
        """
                # super exists from BaseModel
        super().__init__(query=query, filter=filter, limit=limit, **kwargs)

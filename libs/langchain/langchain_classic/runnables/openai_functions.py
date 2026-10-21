from collections.abc import Callable, Mapping
from operator import itemgetter
from typing import Any, cast

from langchain_core.messages import BaseMessage
from langchain_core.output_parsers.openai_functions import JsonOutputFunctionsParser
from langchain_core.runnables import RouterInput, RouterRunnable, Runnable
from langchain_core.runnables.base import RunnableBindingBase
from typing_extensions import TypedDict


class OpenAIFunction(TypedDict):
    """ChatOpenAI 的函数说明。"""

    name: str
    """函数的名称。"""
    description: str
    """函数的描述。"""
    parameters: dict
    """函数的参数。"""


class OpenAIFunctionsRouter(RunnableBindingBase[BaseMessage, Any]):  # type: ignore[no-redef]
    """一个路由到选定函数的 Runnable。"""

    functions: list[OpenAIFunction] | None

    def __init__(
        self,
        runnables: Mapping[
            str,
            Runnable[dict, Any] | Callable[[dict], Any],
        ],
        functions: list[OpenAIFunction] | None = None,
    ):
        """初始化 `OpenAIFunctionsRouter`。

        Args:
            runnables: 函数名到 Runnable 的映射。
            functions: 可选的函数列表，用于与 Runnable 进行比对。
        """
        if functions is not None:
            if len(functions) != len(runnables):
                msg = "The number of functions does not match the number of runnables."
                raise ValueError(msg)
            if not all(func["name"] in runnables for func in functions):
                msg = "One or more function names are not found in runnables."
                raise ValueError(msg)
        router = cast(
            "Runnable[Any, RouterInput]",
            JsonOutputFunctionsParser(args_only=False)
            | {"key": itemgetter("name"), "input": itemgetter("arguments")},
        ) | RouterRunnable(runnables)

        super().__init__(bound=router, kwargs={}, functions=functions)

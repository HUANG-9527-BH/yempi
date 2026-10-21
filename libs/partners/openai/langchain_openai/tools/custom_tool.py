"""OpenAI 自定义工具的自定义工具装饰器。"""

import inspect
from collections.abc import Awaitable, Callable
from typing import Any

from langchain_core.tools import tool


def _make_wrapped_func(func: Callable[..., str]) -> Callable[..., list[dict[str, Any]]]:
    def wrapped(x: str) -> list[dict[str, Any]]:
        return [{"type": "custom_tool_call_output", "output": func(x)}]

    return wrapped


def _make_wrapped_coroutine(
    coroutine: Callable[..., Awaitable[str]],
) -> Callable[..., Awaitable[list[dict[str, Any]]]]:
    async def wrapped(*args: Any, **kwargs: Any) -> list[dict[str, Any]]:
        result = await coroutine(*args, **kwargs)
        return [{"type": "custom_tool_call_output", "output": result}]

    return wrapped


def custom_tool(*args: Any, **kwargs: Any) -> Any:
    """用于创建 OpenAI 自定义工具的装饰器。

    自定义工具允许使用（可能较长的）自由格式字符串输入。

    以下是使用 LangGraph 的示例：

    ```python
    @custom_tool
    def execute_code(code: str) -> str:
        \"\"\"Execute python code.\"\"\"
        return "27"


    model = ChatOpenAI(model="gpt-5", output_version="responses/v1")

    agent = create_react_agent(model, [execute_code])

    input_message = {"role": "user", "content": "Use the tool to calculate 3^3."}
    for step in agent.stream(
        {"messages": [input_message]},
        stream_mode="values",
    ):
        step["messages"][-1].pretty_print()
    ```

    You can also specify a format for a corresponding context-free grammar using the
    `format` kwarg:

    ```python
    from langchain_openai import ChatOpenAI, custom_tool
    from langgraph.prebuilt import create_react_agent

    grammar = \"\"\"
    start: expr
    expr: term (SP ADD SP term)* -> add
    | term
    term: factor (SP MUL SP factor)* -> mul
    | factor
    factor: INT
    SP: " "
    ADD: "+"
    MUL: "*"
    %import common.INT
    \"\"\"

    format = {"type": "grammar", "syntax": "lark", "definition": grammar}

    # highlight-next-line
    @custom_tool(format=format)
    def do_math(input_string: str) -> str:
        \"\"\"Do a mathematical operation.\"\"\"
        return "27"


    model = ChatOpenAI(model="gpt-5", output_version="responses/v1")

    agent = create_react_agent(model, [do_math])

    input_message = {"role": "user", "content": "Use the tool to calculate 3^3."}
    for step in agent.stream(
        {"messages": [input_message]},
        stream_mode="values",
    ):
        step["messages"][-1].pretty_print()
    ```
    """

    def decorator(func: Callable[..., Any]) -> Any:
        metadata = {"type": "custom_tool"}
        if "format" in kwargs:
            metadata["format"] = kwargs.pop("format")
        tool_obj = tool(infer_schema=False, **kwargs)(func)
        tool_obj.metadata = metadata
        tool_obj.description = func.__doc__
        if inspect.iscoroutinefunction(func):
            tool_obj.coroutine = _make_wrapped_coroutine(func)
        else:
            tool_obj.func = _make_wrapped_func(func)
        return tool_obj

    if args and callable(args[0]) and not kwargs:
        return decorator(args[0])

    return decorator

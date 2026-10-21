"""从字符串解析为 AgentAction 或 AgentFinish 的工具函数。

AgentAction 表示应执行某个操作。
这包含要使用的工具名称、传递给该工具的输入，
以及一个 `log` 变量（包含智能体思考的日志）。

AgentFinish 表示应返回一个响应。
这包含一个 `return_values` 字典。通常只包含一个
`output` 键，但也可以扩展以包含更多内容。
这也包含一个 `log` 变量（包含智能体思考的日志）。
"""

from langchain_classic.agents.output_parsers.json import JSONAgentOutputParser
from langchain_classic.agents.output_parsers.openai_functions import (
    OpenAIFunctionsAgentOutputParser,
)
from langchain_classic.agents.output_parsers.react_json_single_input import (
    ReActJsonSingleInputOutputParser,
)
from langchain_classic.agents.output_parsers.react_single_input import (
    ReActSingleInputOutputParser,
)
from langchain_classic.agents.output_parsers.self_ask import SelfAskOutputParser
from langchain_classic.agents.output_parsers.tools import ToolsAgentOutputParser
from langchain_classic.agents.output_parsers.xml import XMLAgentOutputParser

__all__ = [
    "JSONAgentOutputParser",
    "OpenAIFunctionsAgentOutputParser",
    "ReActJsonSingleInputOutputParser",
    "ReActSingleInputOutputParser",
    "SelfAskOutputParser",
    "ToolsAgentOutputParser",
    "XMLAgentOutputParser",
]

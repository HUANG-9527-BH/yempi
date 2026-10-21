import json
import re
from re import Pattern

from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.exceptions import OutputParserException
from typing_extensions import override

from langchain_classic.agents.agent import AgentOutputParser
from langchain_classic.agents.chat.prompt import FORMAT_INSTRUCTIONS

FINAL_ANSWER_ACTION = "Final Answer:"


class ReActJsonSingleInputOutputParser(AgentOutputParser):
    """解析具有单个工具输入的 JSON 格式的 ReAct 风格 LLM 调用。

    期望输出为以下两种格式之一。

    如果输出表示应采取行动， 应采用如下格式。这将导致返回一个 AgentAction。

    ```
    Thought: agent thought here
    Action:
    ```
    {
        "action": "search",
        "action_input": "what is the temperature in SF"
    }
    ```
    ```

    如果输出表示应给出最终答案， 应采用如下格式。这将导致返回一个 AgentFinish。

    ```
    Thought: agent thought here
    Final Answer: The temperature is 100 degrees
    ```

    """

    pattern: Pattern = re.compile(r"^.*?`{3}(?:json)?\n?(.*?)`{3}.*?$", re.DOTALL)
    """正则表达式模式，用于解析输出。"""

    @override
    def get_format_instructions(self) -> str:
        return FORMAT_INSTRUCTIONS

    @override
    def parse(self, text: str) -> AgentAction | AgentFinish:
        includes_answer = FINAL_ANSWER_ACTION in text
        try:
            found = self.pattern.search(text)
            if not found:
                                # 快速失败以解析最终答案。
                msg = "action not found"
                raise ValueError(msg)
            action = found.group(1)
            response = json.loads(action.strip())
            includes_action = "action" in response
            if includes_answer and includes_action:
                msg = (
                    "Parsing LLM output produced a final answer "
                    f"and a parse-able action: {text}"
                )
                raise OutputParserException(msg)
            return AgentAction(
                response["action"],
                response.get("action_input", {}),
                text,
            )

        except Exception as e:
            if not includes_answer:
                msg = f"Could not parse LLM output: {text}"
                raise OutputParserException(msg) from e
            output = text.rsplit(FINAL_ANSWER_ACTION, maxsplit=1)[-1].strip()
            return AgentFinish({"output": output}, text)

    @property
    def _type(self) -> str:
        return "react-json-single-input"

import re

from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.exceptions import OutputParserException
from typing_extensions import override

from langchain_classic.agents.agent import AgentOutputParser
from langchain_classic.agents.mrkl.prompt import FORMAT_INSTRUCTIONS

FINAL_ANSWER_ACTION = "Final Answer:"
MISSING_ACTION_AFTER_THOUGHT_ERROR_MESSAGE = (
    "Invalid Format: Missing 'Action:' after 'Thought:'"
)
MISSING_ACTION_INPUT_AFTER_ACTION_ERROR_MESSAGE = (
    "Invalid Format: Missing 'Action Input:' after 'Action:'"
)
FINAL_ANSWER_AND_PARSABLE_ACTION_ERROR_MESSAGE = (
    "Parsing LLM output produced both a final answer and a parse-able action:"
)


class ReActSingleInputOutputParser(AgentOutputParser):
    """解析具有单个工具输入的 ReAct 风格 LLM 调用。

    期望输出为以下两种格式之一。

    如果输出表明应采取行动， 格式如下。这将导致返回一个 AgentAction。

    ```
    Thought: agent thought here
    Action: search
    Action Input: what is the temperature in SF?
    ```

    如果输出表明应给出最终答案， 格式如下。这将导致返回一个 AgentFinish。

    ```
    Thought: agent thought here
    Final Answer: The temperature is 100 degrees
    ```

    """

    @override
    def get_format_instructions(self) -> str:
        return FORMAT_INSTRUCTIONS

    @override
    def parse(self, text: str) -> AgentAction | AgentFinish:
        includes_answer = FINAL_ANSWER_ACTION in text
        regex = r"Action\s*\d*\s*:[\s]*(.*?)Action\s*\d*\s*Input\s*\d*\s*:[\s]*(.*)"
        action_match = re.search(regex, text, re.DOTALL)
        if action_match:
            if includes_answer:
                msg = f"{FINAL_ANSWER_AND_PARSABLE_ACTION_ERROR_MESSAGE}: {text}"
                raise OutputParserException(msg)
            action = action_match.group(1).strip()
            action_input = action_match.group(2)
            tool_input = action_input.strip(" ")
            tool_input = tool_input.strip('"')

            return AgentAction(action, tool_input, text)

        if includes_answer:
            return AgentFinish(
                {"output": text.rsplit(FINAL_ANSWER_ACTION, maxsplit=1)[-1].strip()},
                text,
            )

        if not re.search(r"Action\s*\d*\s*:[\s]*(.*?)", text, re.DOTALL):
            msg = f"Could not parse LLM output: `{text}`"
            raise OutputParserException(
                msg,
                observation=MISSING_ACTION_AFTER_THOUGHT_ERROR_MESSAGE,
                llm_output=text,
                send_to_llm=True,
            )
        if not re.search(
            r"[\s]*Action\s*\d*\s*Input\s*\d*\s*:[\s]*(.*)",
            text,
            re.DOTALL,
        ):
            msg = f"Could not parse LLM output: `{text}`"
            raise OutputParserException(
                msg,
                observation=MISSING_ACTION_INPUT_AFTER_ACTION_ERROR_MESSAGE,
                llm_output=text,
                send_to_llm=True,
            )
        msg = f"Could not parse LLM output: `{text}`"
        raise OutputParserException(msg)

    @property
    def _type(self) -> str:
        return "react-single-input"

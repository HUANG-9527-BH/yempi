from __future__ import annotations

import logging

from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.exceptions import OutputParserException
from langchain_core.utils.json import parse_json_markdown
from typing_extensions import override

from langchain_classic.agents.agent import AgentOutputParser

logger = logging.getLogger(__name__)


class JSONAgentOutputParser(AgentOutputParser):
    """解析工具调用和最终答案的 JSON 格式。

    期望输出为以下两种格式之一。

    如果输出表示应采取某个动作， 应采用如下格式。这将导致返回一个 AgentAction。

    ```
    {"action": "search", "action_input": "2+2"}
    ```

    如果输出表示应给出最终答案， 应采用如下格式。这将导致返回一个 AgentFinish。

    ```
    {"action": "Final Answer", "action_input": "4"}
    ```
    """

    @override
    def parse(self, text: str) -> AgentAction | AgentFinish:
        try:
            response = parse_json_markdown(text)
            if isinstance(response, list):
                                # gpt turbo 经常忽略要求生成单一动作的指令
                logger.warning("Got multiple action responses: %s", response)
                response = response[0]
            if response["action"] == "Final Answer":
                return AgentFinish({"output": response["action_input"]}, text)
            action_input = response.get("action_input", {})
            if action_input is None:
                action_input = {}
            return AgentAction(response["action"], action_input, text)
        except Exception as e:
            msg = f"Could not parse LLM output: {text}"
            raise OutputParserException(msg) from e

    @property
    def _type(self) -> str:
        return "json-agent"

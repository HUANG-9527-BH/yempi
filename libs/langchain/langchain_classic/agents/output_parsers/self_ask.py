from collections.abc import Sequence

from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.exceptions import OutputParserException
from typing_extensions import override

from langchain_classic.agents.agent import AgentOutputParser


class SelfAskOutputParser(AgentOutputParser):
    """解析自问风格的 LLM 调用。

    期望输出为以下两种格式之一。

    如果输出表明应采取行动， 格式如下。这将导致返回一个 AgentAction。

    ```
    Thoughts go here...
    Follow up: what is the temperature in SF?
    ```

    如果输出表明应给出最终答案， 格式如下。这将导致返回一个 AgentFinish。

    ```
    Thoughts go here...
    So the final answer is: The temperature is 100 degrees
    ```

    """

    followups: Sequence[str] = ("Follow up:", "Followup:")
    finish_string: str = "So the final answer is: "

    @override
    def parse(self, text: str) -> AgentAction | AgentFinish:
        last_line = text.rsplit("\n", maxsplit=1)[-1]
        if not any(follow in last_line for follow in self.followups):
            if self.finish_string not in last_line:
                msg = f"Could not parse output: {text}"
                raise OutputParserException(msg)
            return AgentFinish({"output": last_line[len(self.finish_string) :]}, text)

        after_colon = text.rsplit(":", maxsplit=1)[-1].strip()
        return AgentAction("Intermediate Answer", after_colon, text)

    @property
    def _type(self) -> str:
        return "self_ask"

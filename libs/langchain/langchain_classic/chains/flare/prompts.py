from langchain_core.output_parsers import BaseOutputParser
from langchain_core.prompts import PromptTemplate
from typing_extensions import override


class FinishedOutputParser(BaseOutputParser[tuple[str, bool]]):
    """检查输出是否完成的输出解析器。"""

    finished_value: str = "FINISHED"
    """表示输出已完成的值。"""

    @override
    def parse(self, text: str) -> tuple[str, bool]:
        cleaned = text.strip()
        finished = self.finished_value in cleaned
        return cleaned.replace(self.finished_value, ""), finished


PROMPT_TEMPLATE = """\
Respond to the user message using any relevant context. \
If context is provided, you should ground your answer in that context. \
Once you're done responding return FINISHED.

>>> CONTEXT: {context}
>>> USER INPUT: {user_input}
>>> RESPONSE: {response}\
"""

PROMPT = PromptTemplate(
    template=PROMPT_TEMPLATE,
    input_variables=["user_input", "context", "response"],
)


QUESTION_GENERATOR_PROMPT_TEMPLATE = """\
Given a user input and an existing partial response as context, \
ask a question to which the answer is the given term/entity/phrase:

>>> USER INPUT: {user_input}
>>> EXISTING PARTIAL RESPONSE: {current_response}

The question to which the answer is the term/entity/phrase "{uncertain_span}" is:"""
QUESTION_GENERATOR_PROMPT = PromptTemplate(
    template=QUESTION_GENERATOR_PROMPT_TEMPLATE,
    input_variables=["user_input", "current_response", "uncertain_span"],
)

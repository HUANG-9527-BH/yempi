from collections.abc import Iterator

from langchain_core._api import deprecated
from langchain_core.language_models import BaseChatModel, BaseLanguageModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers.openai_functions import PydanticOutputFunctionsParser
from langchain_core.prompts.chat import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_core.runnables import Runnable
from pydantic import BaseModel, Field

from langchain_classic.chains.llm import LLMChain
from langchain_classic.chains.openai_functions.utils import get_llm_kwargs


class FactWithEvidence(BaseModel):
    """表示单个语句的类。

    每个事实都有一个主体和一组来源。
    如果有多个事实，请确保将它们拆分，
    使得每个事实仅使用与其相关的来源集合。
    """

    fact: str = Field(..., description="Body of the sentence, as part of a response")
    substring_quote: list[str] = Field(
        ...,
        description=(
            "Each source should be a direct quote from the context, "
            "as a substring of the original content"
        ),
    )

    def _get_span(self, quote: str, context: str, errs: int = 100) -> Iterator[str]:
        import regex

        minor = quote
        major = context

        errs_ = 0
        s = regex.search(f"({minor}){{e<={errs_}}}", major)
        while s is None and errs_ <= errs:
            errs_ += 1
            s = regex.search(f"({minor}){{e<={errs_}}}", major)

        if s is not None:
            yield from s.spans()

    def get_spans(self, context: str) -> Iterator[str]:
        """获取上下文中的子串引用的跨度。

        Args:
            context: 在其中查找子串引用跨度的上下文。

        Returns:
            子串引用在上下文中的跨度的迭代器。
        """
        for quote in self.substring_quote:
            yield from self._get_span(quote, context)


class QuestionAnswer(BaseModel):
    """一个问题及其答案，由一组事实组成。

    每个事实都应包含一个来源。
    每个句子包含一个主体和一组来源。
    """

    question: str = Field(..., description="Question that was asked")
    answer: list[FactWithEvidence] = Field(
        ...,
        description=(
            "Body of the answer, each fact should be "
            "its separate object with a body and a list of sources"
        ),
    )


def create_citation_fuzzy_match_runnable(llm: BaseChatModel) -> Runnable:
    """创建一个引用模糊匹配 Runnable。

    使用示例：

        ```python
        from langchain_classic.chains import create_citation_fuzzy_match_runnable
        from langchain_openai import ChatOpenAI

        model = ChatOpenAI(model="gpt-4o-mini")

        context = "Alice has blue eyes. Bob has brown eyes. Charlie has green eyes."
        question = "What color are Bob's eyes?"

        chain = create_citation_fuzzy_match_runnable(model)
        chain.invoke({"question": question, "context": context})
        ```

    Args:
        llm: 用于链的语言模型。必须实现 bind_tools。

    Returns:
        可用于带引用回答问题的 Runnable。

    """
    if type(llm).bind_tools is BaseChatModel.bind_tools:
        msg = "Language model must implement bind_tools to use this function."
        raise ValueError(msg)
    prompt = ChatPromptTemplate(
        [
            SystemMessage(
                "You are a world class algorithm to answer "
                "questions with correct and exact citations.",
            ),
            HumanMessagePromptTemplate.from_template(
                "Answer question using the following context."
                "\n\n{context}"
                "\n\nQuestion: {question}"
                "\n\nTips: Make sure to cite your sources, "
                "and use the exact words from the context.",
            ),
        ],
    )
    return prompt | llm.with_structured_output(QuestionAnswer)


@deprecated(
    since="0.2.13",
    removal="2.0.0",
    alternative="create_citation_fuzzy_match_runnable",
)
def create_citation_fuzzy_match_chain(llm: BaseLanguageModel) -> LLMChain:
    """创建一个引用模糊匹配链。

    Args:
        llm: 用于链的语言模型。

    Returns:
        可用于带引用回答问题的链（LLMChain）。
    """
    output_parser = PydanticOutputFunctionsParser(pydantic_schema=QuestionAnswer)
    schema = QuestionAnswer.model_json_schema()
    function = {
        "name": schema["title"],
        "description": schema["description"],
        "parameters": schema,
    }
    llm_kwargs = get_llm_kwargs(function)
    messages = [
        SystemMessage(
            content=(
                "You are a world class algorithm to answer "
                "questions with correct and exact citations."
            ),
        ),
        HumanMessage(content="Answer question using the following context"),
        HumanMessagePromptTemplate.from_template("{context}"),
        HumanMessagePromptTemplate.from_template("Question: {question}"),
        HumanMessage(
            content=(
                "Tips: Make sure to cite your sources, "
                "and use the exact words from the context."
            ),
        ),
    ]
    prompt = ChatPromptTemplate(messages=messages)  # type: ignore[arg-type]

    return LLMChain(
        llm=llm,
        prompt=prompt,
        llm_kwargs=llm_kwargs,
        output_parser=output_parser,
    )

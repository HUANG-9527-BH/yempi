from typing import Any, cast

from langchain_core._api import deprecated
from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import BaseLLMOutputParser
from langchain_core.output_parsers.openai_functions import (
    OutputFunctionsParser,
    PydanticOutputFunctionsParser,
)
from langchain_core.prompts import PromptTemplate
from langchain_core.prompts.chat import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_core.utils.pydantic import is_basemodel_subclass
from pydantic import BaseModel, Field

from langchain_classic.chains.llm import LLMChain
from langchain_classic.chains.openai_functions.utils import get_llm_kwargs


class AnswerWithSources(BaseModel):
    """对问题的回答，附带来源。"""

    answer: str = Field(..., description="Answer to the question that was asked")
    sources: list[str] = Field(
        ...,
        description="List of sources used to answer the question",
    )


@deprecated(
    since="0.2.13",
    removal="2.0.0",
    alternative="langchain.agents.create_agent",
    addendum=(
        "Build new RAG flows with `create_agent`, a retrieval tool, and "
        "`response_format` for structured responses. See "
        "https://docs.langchain.com/oss/python/langchain/rag and "
        "https://docs.langchain.com/oss/python/langchain/structured-output"
    ),
)
def create_qa_with_structure_chain(
    llm: BaseLanguageModel,
    schema: dict | type[BaseModel],
    output_parser: str = "base",
    prompt: PromptTemplate | ChatPromptTemplate | None = None,
    verbose: bool = False,  # noqa: FBT001,FBT002
) -> LLMChain:
    """创建一个带有结构的问答链。

    创建一个返回带来源的答案的问答链
    基于模式。

    Args:
        llm: 用于链的语言模型。
        schema: 用于输出的 Pydantic 模式。
        output_parser: 用于输出的解析器。应为 `'pydantic'` 或 `'base'` 之一。
        prompt: 可选的提示词，用于链。
        verbose: 是否以详细模式运行链。

    Returns:
        问答链。

    """
    if output_parser == "pydantic":
        if not (isinstance(schema, type) and is_basemodel_subclass(schema)):
            msg = (
                "Must provide a pydantic class for schema when output_parser is "
                "'pydantic'."
            )
            raise ValueError(msg)
        _output_parser: BaseLLMOutputParser = PydanticOutputFunctionsParser(
            pydantic_schema=schema,
        )
    elif output_parser == "base":
        _output_parser = OutputFunctionsParser()
    else:
        msg = (
            f"Got unexpected output_parser: {output_parser}. "
            f"Should be one of `pydantic` or `base`."
        )
        raise ValueError(msg)
    if isinstance(schema, type) and is_basemodel_subclass(schema):
        schema_dict = cast("dict", schema.model_json_schema())
    else:
        schema_dict = cast("dict", schema)
    function = {
        "name": schema_dict["title"],
        "description": schema_dict["description"],
        "parameters": schema_dict,
    }
    llm_kwargs = get_llm_kwargs(function)
    messages = [
        SystemMessage(
            content=(
                "You are a world class algorithm to answer "
                "questions in a specific format."
            ),
        ),
        HumanMessage(content="Answer question using the following context"),
        HumanMessagePromptTemplate.from_template("{context}"),
        HumanMessagePromptTemplate.from_template("Question: {question}"),
        HumanMessage(content="Tips: Make sure to answer in the correct format"),
    ]
    prompt = prompt or ChatPromptTemplate(messages=messages)  # type: ignore[arg-type]

    return LLMChain(
        llm=llm,
        prompt=prompt,
        llm_kwargs=llm_kwargs,
        output_parser=_output_parser,
        verbose=verbose,
    )


@deprecated(
    since="0.2.13",
    removal="2.0.0",
    alternative="langchain.agents.create_agent",
    addendum=(
        "Build new RAG flows with `create_agent`, a retrieval tool, and "
        "`response_format` for structured responses. See "
        "https://docs.langchain.com/oss/python/langchain/rag and "
        "https://docs.langchain.com/oss/python/langchain/structured-output"
    ),
)
def create_qa_with_sources_chain(
    llm: BaseLanguageModel,
    verbose: bool = False,  # noqa: FBT001,FBT002
    **kwargs: Any,
) -> LLMChain:
    """创建一个返回带来源答案的问答链。

    Args:
        llm: 用于链的语言模型。
        verbose: 是否打印链的详细信息
        **kwargs: 传递给 `create_qa_with_structure_chain` 的关键字参数。

    Returns:
        可用于回答问题并附带引用的链（LLMChain）。
    """
    return create_qa_with_structure_chain(
        llm,
        AnswerWithSources,
        verbose=verbose,
        **kwargs,
    )

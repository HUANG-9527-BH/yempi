"""检索器工具。"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, Field

# 无法将 Callbacks 和 Document 移至 TYPE_CHECKING，因为 StructuredTool 的
# 函数/协程参数注解在运行时才会被求值。
from langchain_core.callbacks import Callbacks  # noqa: TC001
from langchain_core.documents import Document  # noqa: TC001
from langchain_core.prompts import (
    BasePromptTemplate,
    PromptTemplate,
    aformat_document,
    format_document,
)
from langchain_core.tools.structured import StructuredTool

if TYPE_CHECKING:
    from langchain_core.retrievers import BaseRetriever


class RetrieverInput(BaseModel):
    """检索器的输入。"""

    query: str = Field(description="query to look up in retriever")


def create_retriever_tool(
    retriever: BaseRetriever,
    name: str,
    description: str,
    *,
    document_prompt: BasePromptTemplate[str] | None = None,
    document_separator: str = "\n\n",
    response_format: Literal["content", "content_and_artifact"] = "content",
) -> StructuredTool:
    r"""创建一个用于文档检索的工具。

    Args:
        retriever: 用于检索的检索器
        name: 工具的名称。

            This will be passed to the language model, so should be unique and somewhat
            descriptive.
        description: The description for the tool.

            这将传递给语言模型，因此应具有描述性。
        document_prompt: 用于文档的提示词。
        document_separator: 用于分隔文档的分隔符。
        response_format: 工具的响应格式。

            如果为 `'content'`，则工具的输出被解释为 `ToolMessage` 的内容。如果为
            `'content_and_artifact'`，则输出预期为一个二元组，对应 `ToolMessage` 的
            `(content, artifact)`（artifact 在此情况下为文档列表）。

    Returns:
        传递给智能体的工具类。
    """
    document_prompt_ = document_prompt or PromptTemplate.from_template("{page_content}")

    def func(
        query: str, callbacks: Callbacks = None
    ) -> str | tuple[str, list[Document]]:
        docs = retriever.invoke(query, config={"callbacks": callbacks})
        content = document_separator.join(
            format_document(doc, document_prompt_) for doc in docs
        )
        if response_format == "content_and_artifact":
            return (content, docs)
        return content

    async def afunc(
        query: str, callbacks: Callbacks = None
    ) -> str | tuple[str, list[Document]]:
        docs = await retriever.ainvoke(query, config={"callbacks": callbacks})
        content = document_separator.join(
            [await aformat_document(doc, document_prompt_) for doc in docs]
        )
        if response_format == "content_and_artifact":
            return (content, docs)
        return content

    return StructuredTool(
        name=name,
        description=description,
        func=func,
        coroutine=afunc,
        args_schema=RetrieverInput,
        response_format=response_format,
    )

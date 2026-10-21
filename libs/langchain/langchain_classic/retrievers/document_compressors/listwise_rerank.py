"""使用 LLM 进行列表级重排序并选择 top-k 的过滤器。"""

from collections.abc import Sequence
from typing import Any

from langchain_core.callbacks import Callbacks
from langchain_core.documents import BaseDocumentCompressor, Document
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import BasePromptTemplate, ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableLambda, RunnablePassthrough
from pydantic import BaseModel, ConfigDict, Field

_default_system_tmpl = """{context}

Sort the Documents by their relevance to the Query."""
_DEFAULT_PROMPT = ChatPromptTemplate.from_messages(
    [("system", _default_system_tmpl), ("human", "{query}")],
)


def _get_prompt_input(input_: dict) -> dict[str, Any]:
    """返回压缩链的输入。"""
    documents = input_["documents"]
    context = ""
    for index, doc in enumerate(documents):
        context += f"Document ID: {index}\n```{doc.page_content}```\n\n"
    document_range = "empty list"
    if len(documents) > 0:
        document_range = f"Document ID: 0, ..., Document ID: {len(documents) - 1}"
    context += f"Documents = [{document_range}]"
    return {"query": input_["query"], "context": context}


def _parse_ranking(results: dict) -> list[Document]:
    ranking = results["ranking"]
    docs = results["documents"]
    return [docs[i] for i in ranking.ranked_document_ids]


class LLMListwiseRerank(BaseDocumentCompressor):
    """使用 `零样本列表级文档重排序` 的文档压缩器。

    Adapted from: adapted from: https://arxiv.org/pdf/2305.02156.pdf

    `LLMListwiseRerank` 使用语言模型根据文档与查询的相关性对文档列表进行重排序。

    !!! note
        要求底层模型实现 `with_structured_output`。

    示例用法：
        ```python
        from langchain_classic.retrievers.document_compressors.listwise_rerank import (
            LLMListwiseRerank,
        )
        from langchain_core.documents import Document
        from langchain_openai import ChatOpenAI

        documents = [
            Document("Sally is my friend from school"),
            Document("Steve is my friend from home"),
            Document("I didn't always like yogurt"),
            Document("I wonder why it's called football"),
            Document("Where's waldo"),
        ]

        reranker = LLMListwiseRerank.from_llm(llm=ChatOpenAI(model="gpt-5.5"), top_n=3)
        compressed_docs = reranker.compress_documents(documents, "Who is steve")
        assert len(compressed_docs) == 3
        assert "Steve" in compressed_docs[0].page_content
        ```
    """

    reranker: Runnable[dict, list[Document]]
    """LLM-based reranker to use for filtering documents. Expected to take in a dict
        with 'documents: Sequence[Document]' and 'query: str' keys and output a
        List[Document]."""

    top_n: int = 3
    """要返回的文档数量。"""

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )

    def compress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Callbacks | None = None,
    ) -> Sequence[Document]:
        """根据文档与查询的相关性筛选文档。"""
        results = self.reranker.invoke(
            {"documents": documents, "query": query},
            config={"callbacks": callbacks},
        )
        return results[: self.top_n]

    @classmethod
    def from_llm(
        cls,
        llm: BaseLanguageModel,
        *,
        prompt: BasePromptTemplate | None = None,
        **kwargs: Any,
    ) -> "LLMListwiseRerank":
        """从语言模型创建一个 LLMListwiseRerank 文档压缩器。

        Args:
            llm: The language model to use for filtering. **Must implement
                BaseLanguageModel.with_structured_output().**
            prompt: The prompt to use for the filter.
            kwargs: Additional arguments to pass to the constructor.

        Returns:
            使用给定语言模型的 LLMListwiseRerank 文档压缩器。
        """
        if type(llm).with_structured_output == BaseLanguageModel.with_structured_output:
            msg = (
                f"llm of type {type(llm)} does not implement `with_structured_output`."
            )
            raise ValueError(msg)

        class RankDocuments(BaseModel):
            """根据文档与用户问题的相关性对文档进行排序。

            从最相关到最不相关排序。
            """

            ranked_document_ids: list[int] = Field(
                ...,
                description=(
                    "The integer IDs of the documents, sorted from most to least "
                    "relevant to the user question."
                ),
            )

        _prompt = prompt if prompt is not None else _DEFAULT_PROMPT
        reranker = RunnablePassthrough.assign(
            ranking=RunnableLambda(_get_prompt_input)
            | _prompt
            | llm.with_structured_output(RankDocuments),
        ) | RunnableLambda(_parse_ranking)
        return cls(reranker=reranker, **kwargs)

"""通过递归缩减多个文档来合并它们。"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol

from langchain_core._api import deprecated
from langchain_core.callbacks import Callbacks
from langchain_core.documents import Document
from pydantic import ConfigDict

from langchain_classic.chains.combine_documents.base import BaseCombineDocumentsChain


class CombineDocsProtocol(Protocol):
    """combine_docs 方法的接口。"""

    def __call__(self, docs: list[Document], **kwargs: Any) -> str:
        """combine_docs 方法的接口。"""


class AsyncCombineDocsProtocol(Protocol):
    """combine_docs 方法的接口。"""

    async def __call__(self, docs: list[Document], **kwargs: Any) -> str:
        """combine_docs 方法的异步接口。"""


def split_list_of_docs(
    docs: list[Document],
    length_func: Callable,
    token_max: int,
    **kwargs: Any,
) -> list[list[Document]]:
    """将 `Document` 对象分割为若干子集，每个子集的累积长度满足约束条件。

    Args:
        docs: The full list of `Document` objects.
        length_func: Function for computing the cumulative length of a set of `Document`
            objects.
        token_max: The maximum cumulative length of any subset of `Document` objects.
        **kwargs: Arbitrary additional keyword params to pass to each call of the
            `length_func`.

    Returns:
        一个 `list[list[Document]]`。
    """
    new_result_doc_list = []
    _sub_result_docs = []
    for doc in docs:
        _sub_result_docs.append(doc)
        _num_tokens = length_func(_sub_result_docs, **kwargs)
        if _num_tokens > token_max:
            if len(_sub_result_docs) == 1:
                msg = (
                    "A single document was longer than the context length,"
                    " we cannot handle this."
                )
                raise ValueError(msg)
            new_result_doc_list.append(_sub_result_docs[:-1])
            _sub_result_docs = _sub_result_docs[-1:]
    new_result_doc_list.append(_sub_result_docs)
    return new_result_doc_list


def collapse_docs(
    docs: list[Document],
    combine_document_func: CombineDocsProtocol,
    **kwargs: Any,
) -> Document:
    """对一组文档执行折叠函数，并合并它们的元数据。

    Args:
        docs: 需要合并的 `Document` 对象列表。 combine_document_func: 接收一组
        `Document` 对象并可选地接收额外关键字参数的函数，将其合并为单个字符串。
        **kwargs: 传递给 `combine_document_func` 的任意额外关键字参数。

    Returns:
        A single `Document` with the output of `combine_document_func` for the page
            content and the combined metadata's of all the input documents. All metadata
            values are strings, and where there are overlapping keys across documents
            the values are joined by `', '`.
    """
    result = combine_document_func(docs, **kwargs)
    combined_metadata = {k: str(v) for k, v in docs[0].metadata.items()}
    for doc in docs[1:]:
        for k, v in doc.metadata.items():
            if k in combined_metadata:
                combined_metadata[k] += f", {v}"
            else:
                combined_metadata[k] = str(v)
    return Document(page_content=result, metadata=combined_metadata)


async def acollapse_docs(
    docs: list[Document],
    combine_document_func: AsyncCombineDocsProtocol,
    **kwargs: Any,
) -> Document:
    """对一组文档执行折叠函数，并合并它们的元数据。

    Args:
        docs: 需要合并的 `Document` 对象列表。 combine_document_func: 接收一组
        `Document` 对象并可选地接收额外关键字参数的函数，将其合并为单个字符串。
        **kwargs: 传递给 `combine_document_func` 的任意额外关键字参数。

    Returns:
        A single `Document` with the output of `combine_document_func` for the page
            content and the combined metadata's of all the input documents. All metadata
            values are strings, and where there are overlapping keys across documents
            the values are joined by `', '`.
    """
    result = await combine_document_func(docs, **kwargs)
    combined_metadata = {k: str(v) for k, v in docs[0].metadata.items()}
    for doc in docs[1:]:
        for k, v in doc.metadata.items():
            if k in combined_metadata:
                combined_metadata[k] += f", {v}"
            else:
                combined_metadata[k] = str(v)
    return Document(page_content=result, metadata=combined_metadata)


@deprecated(
    since="0.3.1",
    removal="2.0.0",
    alternative="langchain.agents.create_agent",
    addendum=(
        "Build new RAG flows with `create_agent` and a retrieval tool. See "
        "https://docs.langchain.com/oss/python/langchain/rag"
    ),
)
class ReduceDocumentsChain(BaseCombineDocumentsChain):
    """通过递归缩减来合并文档。

    这涉及

    - `combine_documents_chain`
    - `collapse_documents_chain`

    `combine_documents_chain` 始终提供。这是最终调用的链。

    我们将所有先前的结果传递给此链，该链的输出作为最终结果返回。

    `collapse_documents_chain` is used if the documents passed in are too many to all
    be passed to `combine_documents_chain` in one go. In this case,
    `collapse_documents_chain` is called recursively on as big of groups of documents
    as are allowed.

    Example:
        ```python
        from langchain_classic.chains import (
            StuffDocumentsChain,
            LLMChain,
            ReduceDocumentsChain,
        )
        from langchain_core.prompts import PromptTemplate
        from langchain_openai import OpenAI

        # This controls how each document will be formatted. Specifically,
        # it will be passed to `format_document` - see that function for more
        # details.
        document_prompt = PromptTemplate(
            input_variables=["page_content"], template="{page_content}"
        )
        document_variable_name = "context"
        model = OpenAI()
        # The prompt here should take as an input variable the
        # `document_variable_name`
        prompt = PromptTemplate.from_template("Summarize this content: {context}")
        llm_chain = LLMChain(llm=model, prompt=prompt)
        combine_documents_chain = StuffDocumentsChain(
            llm_chain=llm_chain,
            document_prompt=document_prompt,
            document_variable_name=document_variable_name,
        )
        chain = ReduceDocumentsChain(
            combine_documents_chain=combine_documents_chain,
        )
        # If we wanted to, we could also pass in collapse_documents_chain
        # which is specifically aimed at collapsing documents BEFORE
        # the final call.
        prompt = PromptTemplate.from_template("Collapse this content: {context}")
        llm_chain = LLMChain(llm=model, prompt=prompt)
        collapse_documents_chain = StuffDocumentsChain(
            llm_chain=llm_chain,
            document_prompt=document_prompt,
            document_variable_name=document_variable_name,
        )
        chain = ReduceDocumentsChain(
            combine_documents_chain=combine_documents_chain,
            collapse_documents_chain=collapse_documents_chain,
        )
        ```
    """

    combine_documents_chain: BaseCombineDocumentsChain
    """用于合并文档的最终链。

    通常是 `StuffDocumentsChain`。
    """
    collapse_documents_chain: BaseCombineDocumentsChain | None = None
    """Chain to use to collapse documents if needed until they can all fit.
    If `None`, will use the `combine_documents_chain`.

    通常是 `StuffDocumentsChain`。
    """
    token_max: int = 3000
    """将文档分组的最大 token 数量。

    For example, if set to 3000 then documents will be grouped into chunks of no greater
    than 3000 tokens before trying to combine them into a smaller chunk.
    """
    collapse_max_retries: int | None = None
    """尝试将文档压缩至 `token_max` 范围内的最大重试次数。

    如果为 `None`，则会持续尝试压缩文档以适应 `token_max`。

    否则，在达到最大次数后，将抛出错误。
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="forbid",
    )

    @property
    def _collapse_chain(self) -> BaseCombineDocumentsChain:
        if self.collapse_documents_chain is not None:
            return self.collapse_documents_chain
        return self.combine_documents_chain

    def combine_docs(
        self,
        docs: list[Document],
        token_max: int | None = None,
        callbacks: Callbacks = None,
        **kwargs: Any,
    ) -> tuple[str, dict]:
        """递归地合并多个文档。

        Args:
            docs: List of documents to combine, assumed that each one is less than
                `token_max`.
            token_max: Recursively creates groups of documents less than this number
                of tokens.
            callbacks: Callbacks to be passed through
            **kwargs: additional parameters to be passed to LLM calls (like other
                input variables besides the documents)

        Returns:
            返回的第一个元素是单一字符串输出。第二个返回元素是其他键的字典。
        """
        result_docs, _ = self._collapse(
            docs,
            token_max=token_max,
            callbacks=callbacks,
            **kwargs,
        )
        return self.combine_documents_chain.combine_docs(
            docs=result_docs,
            callbacks=callbacks,
            **kwargs,
        )

    async def acombine_docs(
        self,
        docs: list[Document],
        token_max: int | None = None,
        callbacks: Callbacks = None,
        **kwargs: Any,
    ) -> tuple[str, dict]:
        """异步递归合并多个文档。

        Args:
            docs: List of documents to combine, assumed that each one is less than
                `token_max`.
            token_max: Recursively creates groups of documents less than this number
                of tokens.
            callbacks: Callbacks to be passed through
            **kwargs: additional parameters to be passed to LLM calls (like other
                input variables besides the documents)

        Returns:
            返回的第一个元素是单一字符串输出。第二个返回元素是其他键的字典。
        """
        result_docs, _ = await self._acollapse(
            docs,
            token_max=token_max,
            callbacks=callbacks,
            **kwargs,
        )
        return await self.combine_documents_chain.acombine_docs(
            docs=result_docs,
            callbacks=callbacks,
            **kwargs,
        )

    def _collapse(
        self,
        docs: list[Document],
        token_max: int | None = None,
        callbacks: Callbacks = None,
        **kwargs: Any,
    ) -> tuple[list[Document], dict]:
        result_docs = docs
        length_func = self.combine_documents_chain.prompt_length
        num_tokens = length_func(result_docs, **kwargs)

        def _collapse_docs_func(docs: list[Document], **kwargs: Any) -> str:
            return self._collapse_chain.run(
                input_documents=docs,
                callbacks=callbacks,
                **kwargs,
            )

        _token_max = token_max or self.token_max
        retries: int = 0
        while num_tokens is not None and num_tokens > _token_max:
            new_result_doc_list = split_list_of_docs(
                result_docs,
                length_func,
                _token_max,
                **kwargs,
            )
            result_docs = [
                collapse_docs(docs_, _collapse_docs_func, **kwargs)
                for docs_ in new_result_doc_list
            ]
            num_tokens = length_func(result_docs, **kwargs)
            retries += 1
            if self.collapse_max_retries and retries == self.collapse_max_retries:
                msg = f"Exceed {self.collapse_max_retries} tries to \
                        collapse document to {_token_max} tokens."
                raise ValueError(msg)
        return result_docs, {}

    async def _acollapse(
        self,
        docs: list[Document],
        token_max: int | None = None,
        callbacks: Callbacks = None,
        **kwargs: Any,
    ) -> tuple[list[Document], dict]:
        result_docs = docs
        length_func = self.combine_documents_chain.prompt_length
        num_tokens = length_func(result_docs, **kwargs)

        async def _collapse_docs_func(docs: list[Document], **kwargs: Any) -> str:
            return await self._collapse_chain.arun(
                input_documents=docs,
                callbacks=callbacks,
                **kwargs,
            )

        _token_max = token_max or self.token_max
        retries: int = 0
        while num_tokens is not None and num_tokens > _token_max:
            new_result_doc_list = split_list_of_docs(
                result_docs,
                length_func,
                _token_max,
                **kwargs,
            )
            result_docs = [
                await acollapse_docs(docs_, _collapse_docs_func, **kwargs)
                for docs_ in new_result_doc_list
            ]
            num_tokens = length_func(result_docs, **kwargs)
            retries += 1
            if self.collapse_max_retries and retries == self.collapse_max_retries:
                msg = f"Exceed {self.collapse_max_retries} tries to \
                        collapse document to {_token_max} tokens."
                raise ValueError(msg)
        return result_docs, {}

    @property
    def _chain_type(self) -> str:
        return "reduce_documents_chain"

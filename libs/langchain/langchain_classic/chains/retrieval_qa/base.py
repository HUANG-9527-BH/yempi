"""针对向量数据库的问答链。"""

from __future__ import annotations

import inspect
from abc import abstractmethod
from typing import Any

from langchain_core._api import deprecated
from langchain_core.callbacks import (
    AsyncCallbackManagerForChainRun,
    CallbackManagerForChainRun,
    Callbacks,
)
from langchain_core.documents import Document
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import PromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.vectorstores import VectorStore
from pydantic import ConfigDict, Field, model_validator
from typing_extensions import override

from langchain_classic.chains.base import Chain
from langchain_classic.chains.combine_documents.base import BaseCombineDocumentsChain
from langchain_classic.chains.combine_documents.stuff import StuffDocumentsChain
from langchain_classic.chains.llm import LLMChain
from langchain_classic.chains.question_answering import load_qa_chain
from langchain_classic.chains.question_answering.stuff_prompt import PROMPT_SELECTOR


@deprecated(
    since="0.2.13",
    removal="2.0.0",
    alternative="langchain.agents.create_agent",
    addendum=(
        "Build new RAG flows with `create_agent` and a retrieval tool. See "
        "https://docs.langchain.com/oss/python/langchain/rag"
    ),
)
class BaseRetrievalQA(Chain):
    """问答链的基类。"""

    combine_documents_chain: BaseCombineDocumentsChain
    """用于合并文档的链。"""
    input_key: str = "query"
    output_key: str = "result"
    return_source_documents: bool = False
    """是否返回源文档。"""

    model_config = ConfigDict(
        populate_by_name=True,
        arbitrary_types_allowed=True,
        extra="forbid",
    )

    @property
    def input_keys(self) -> list[str]:
        """输入键。"""
        return [self.input_key]

    @property
    def output_keys(self) -> list[str]:
        """输出键。"""
        _output_keys = [self.output_key]
        if self.return_source_documents:
            _output_keys = [*_output_keys, "source_documents"]
        return _output_keys

    @classmethod
    def from_llm(
        cls,
        llm: BaseLanguageModel,
        prompt: PromptTemplate | None = None,
        callbacks: Callbacks = None,
        llm_chain_kwargs: dict | None = None,
        **kwargs: Any,
    ) -> BaseRetrievalQA:
        """从 LLM 初始化。"""
        _prompt = prompt or PROMPT_SELECTOR.get_prompt(llm)
        llm_chain = LLMChain(
            llm=llm,
            prompt=_prompt,
            callbacks=callbacks,
            **(llm_chain_kwargs or {}),
        )
        document_prompt = PromptTemplate(
            input_variables=["page_content"],
            template="Context:\n{page_content}",
        )
        combine_documents_chain = StuffDocumentsChain(
            llm_chain=llm_chain,
            document_variable_name="context",
            document_prompt=document_prompt,
            callbacks=callbacks,
        )

        return cls(
            combine_documents_chain=combine_documents_chain,
            callbacks=callbacks,
            **kwargs,
        )

    @classmethod
    def from_chain_type(
        cls,
        llm: BaseLanguageModel,
        chain_type: str = "stuff",
        chain_type_kwargs: dict | None = None,
        **kwargs: Any,
    ) -> BaseRetrievalQA:
        """从链类型加载链。"""
        _chain_type_kwargs = chain_type_kwargs or {}
        combine_documents_chain = load_qa_chain(
            llm,
            chain_type=chain_type,
            **_chain_type_kwargs,
        )
        return cls(combine_documents_chain=combine_documents_chain, **kwargs)

    @abstractmethod
    def _get_docs(
        self,
        question: str,
        *,
        run_manager: CallbackManagerForChainRun,
    ) -> list[Document]:
        """获取用于问答的文档。"""

    def _call(
        self,
        inputs: dict[str, Any],
        run_manager: CallbackManagerForChainRun | None = None,
    ) -> dict[str, Any]:
        """对输入查询运行 get_relevant_text 和 llm。

        如果链的 'return_source_documents' 为 'True'，则在 'source_documents'
        键下返回检索到的文档。

        Example:
        ```python
        res = indexqa({"query": "This is my query"})
        answer, docs = res["result"], res["source_documents"]
        ```
        """
        _run_manager = run_manager or CallbackManagerForChainRun.get_noop_manager()
        question = inputs[self.input_key]
        accepts_run_manager = (
            "run_manager" in inspect.signature(self._get_docs).parameters
        )
        if accepts_run_manager:
            docs = self._get_docs(question, run_manager=_run_manager)
        else:
            docs = self._get_docs(question)  # type: ignore[call-arg]
        answer = self.combine_documents_chain.run(
            input_documents=docs,
            question=question,
            callbacks=_run_manager.get_child(),
        )

        if self.return_source_documents:
            return {self.output_key: answer, "source_documents": docs}
        return {self.output_key: answer}

    @abstractmethod
    async def _aget_docs(
        self,
        question: str,
        *,
        run_manager: AsyncCallbackManagerForChainRun,
    ) -> list[Document]:
        """获取用于问答的文档。"""

    async def _acall(
        self,
        inputs: dict[str, Any],
        run_manager: AsyncCallbackManagerForChainRun | None = None,
    ) -> dict[str, Any]:
        """对输入查询运行 get_relevant_text 和 llm。

        如果链的 'return_source_documents' 为 'True'，则在 'source_documents'
        键下返回检索到的文档。

        Example:
        ```python
        res = indexqa({"query": "This is my query"})
        answer, docs = res["result"], res["source_documents"]
        ```
        """
        _run_manager = run_manager or AsyncCallbackManagerForChainRun.get_noop_manager()
        question = inputs[self.input_key]
        accepts_run_manager = (
            "run_manager" in inspect.signature(self._aget_docs).parameters
        )
        if accepts_run_manager:
            docs = await self._aget_docs(question, run_manager=_run_manager)
        else:
            docs = await self._aget_docs(question)  # type: ignore[call-arg]
        answer = await self.combine_documents_chain.arun(
            input_documents=docs,
            question=question,
            callbacks=_run_manager.get_child(),
        )

        if self.return_source_documents:
            return {self.output_key: answer, "source_documents": docs}
        return {self.output_key: answer}


@deprecated(
    since="0.1.17",
    removal="2.0.0",
    alternative="langchain.agents.create_agent",
    addendum=(
        "Build new RAG flows with `create_agent` and a retrieval tool. See "
        "https://docs.langchain.com/oss/python/langchain/rag"
    ),
)
class RetrievalQA(BaseRetrievalQA):
    """针对索引的问答链。

    This class is deprecated. See below for an example implementation using
    `create_retrieval_chain`:

        ```python
        from langchain_classic.chains import create_retrieval_chain
        from langchain_classic.chains.combine_documents import (
            create_stuff_documents_chain,
        )
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_openai import ChatOpenAI


        retriever = ...  # Your retriever
        model = ChatOpenAI()

        system_prompt = (
            "Use the given context to answer the question. "
            "If you don't know the answer, say you don't know. "
            "Use three sentence maximum and keep the answer concise. "
            "Context: {context}"
        )
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("human", "{input}"),
            ]
        )
        question_answer_chain = create_stuff_documents_chain(model, prompt)
        chain = create_retrieval_chain(retriever, question_answer_chain)

        chain.invoke({"input": query})
        ```

    Example:
        ```python
        from langchain_openai import OpenAI
        from langchain_classic.chains import RetrievalQA
        from langchain_community.vectorstores import FAISS
        from langchain_core.vectorstores import VectorStoreRetriever

        retriever = VectorStoreRetriever(vectorstore=FAISS(...))
        retrievalQA = RetrievalQA.from_llm(llm=OpenAI(), retriever=retriever)
        ```
    """

    retriever: BaseRetriever = Field(exclude=True)

    def _get_docs(
        self,
        question: str,
        *,
        run_manager: CallbackManagerForChainRun,
    ) -> list[Document]:
        """获取文档。"""
        return self.retriever.invoke(
            question,
            config={"callbacks": run_manager.get_child()},
        )

    async def _aget_docs(
        self,
        question: str,
        *,
        run_manager: AsyncCallbackManagerForChainRun,
    ) -> list[Document]:
        """获取文档。"""
        return await self.retriever.ainvoke(
            question,
            config={"callbacks": run_manager.get_child()},
        )

    @property
    def _chain_type(self) -> str:
        """返回链类型。"""
        return "retrieval_qa"


@deprecated(
    since="0.2.13",
    removal="2.0.0",
    alternative="langchain.agents.create_agent",
    addendum=(
        "Build new RAG flows with `create_agent` and a retrieval tool. See "
        "https://docs.langchain.com/oss/python/langchain/rag"
    ),
)
class VectorDBQA(BaseRetrievalQA):
    """针对向量数据库的问答链。"""

    vectorstore: VectorStore = Field(exclude=True, alias="vectorstore")
    """要连接的向量数据库。"""
    k: int = 4
    """查询的文档数量。"""
    search_type: str = "similarity"
    """在向量存储上使用的搜索类型。`similarity` 或 `mmr`。"""
    search_kwargs: dict[str, Any] = Field(default_factory=dict)
    """额外的搜索参数。"""

    @model_validator(mode="before")
    @classmethod
    def validate_search_type(cls, values: dict) -> Any:
        """验证搜索类型。"""
        if "search_type" in values:
            search_type = values["search_type"]
            if search_type not in ("similarity", "mmr"):
                msg = f"search_type of {search_type} not allowed."
                raise ValueError(msg)
        return values

    @override
    def _get_docs(
        self,
        question: str,
        *,
        run_manager: CallbackManagerForChainRun,
    ) -> list[Document]:
        """获取文档。"""
        if self.search_type == "similarity":
            docs = self.vectorstore.similarity_search(
                question,
                k=self.k,
                **self.search_kwargs,
            )
        elif self.search_type == "mmr":
            docs = self.vectorstore.max_marginal_relevance_search(
                question,
                k=self.k,
                **self.search_kwargs,
            )
        else:
            msg = f"search_type of {self.search_type} not allowed."
            raise ValueError(msg)
        return docs

    async def _aget_docs(
        self,
        question: str,
        *,
        run_manager: AsyncCallbackManagerForChainRun,
    ) -> list[Document]:
        """获取文档。"""
        msg = "VectorDBQA does not support async"
        raise NotImplementedError(msg)

    @property
    def _chain_type(self) -> str:
        """返回链类型。"""
        return "vector_db_qa"

"""向量存储的存根，用于索引 API。"""

from typing import Any

from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseLanguageModel
from langchain_core.vectorstores import VectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter, TextSplitter
from pydantic import BaseModel, ConfigDict, Field

from langchain_classic.chains.qa_with_sources.retrieval import (
    RetrievalQAWithSourcesChain,
)
from langchain_classic.chains.retrieval_qa.base import RetrievalQA


def _get_default_text_splitter() -> TextSplitter:
    """返回用于拆分文档的默认文本分割器。"""
    return RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)


class VectorStoreIndexWrapper(BaseModel):
    """对 `VectorStore` 的包装，便于访问。"""

    vectorstore: VectorStore

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="forbid",
    )

    def query(
        self,
        question: str,
        llm: BaseLanguageModel | None = None,
        retriever_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> str:
        """使用提供的 LLM 查询 `VectorStore`。

        Args:
            question: 要查询的问题或提示词。
            llm: 要使用的语言模型。不能为 `None`。
            retriever_kwargs: 可选的关键字参数，用于检索器。
            **kwargs: 额外的关键字参数，将转发给链。

        Returns:
            RetrievalQA 链返回的结果字符串。
        """
        if llm is None:
            msg = (
                "This API has been changed to require an LLM. "
                "Please provide an llm to use for querying the vectorstore.\n"
                "For example,\n"
                "from langchain_openai import OpenAI\n"
                "model = OpenAI(temperature=0)"
            )
            raise NotImplementedError(msg)
        retriever_kwargs = retriever_kwargs or {}
        chain = RetrievalQA.from_chain_type(
            llm,
            retriever=self.vectorstore.as_retriever(**retriever_kwargs),
            **kwargs,
        )
        return chain.invoke({chain.input_key: question})[chain.output_key]

    async def aquery(
        self,
        question: str,
        llm: BaseLanguageModel | None = None,
        retriever_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> str:
        """异步使用提供的 LLM 查询 `VectorStore`。

        Args:
            question: 要查询的问题或提示词。
            llm: 要使用的语言模型。不能为 `None`。
            retriever_kwargs: 可选的关键字参数，用于检索器。
            **kwargs: 额外的关键字参数，将转发给链。

        Returns:
            RetrievalQA 链返回的异步结果字符串。
        """
        if llm is None:
            msg = (
                "This API has been changed to require an LLM. "
                "Please provide an llm to use for querying the vectorstore.\n"
                "For example,\n"
                "from langchain_openai import OpenAI\n"
                "model = OpenAI(temperature=0)"
            )
            raise NotImplementedError(msg)
        retriever_kwargs = retriever_kwargs or {}
        chain = RetrievalQA.from_chain_type(
            llm,
            retriever=self.vectorstore.as_retriever(**retriever_kwargs),
            **kwargs,
        )
        return (await chain.ainvoke({chain.input_key: question}))[chain.output_key]

    def query_with_sources(
        self,
        question: str,
        llm: BaseLanguageModel | None = None,
        retriever_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> dict:
        """查询 `VectorStore` 并获取答案及来源。

        Args:
            question: 要查询的问题或提示词。
            llm: 要使用的语言模型。不能为 `None`。
            retriever_kwargs: 可选的关键字参数，用于检索器。
            **kwargs: 额外的关键字参数，将转发给链。

        Returns:
            包含答案和源文档的 `dict`。
        """
        if llm is None:
            msg = (
                "This API has been changed to require an LLM. "
                "Please provide an llm to use for querying the vectorstore.\n"
                "For example,\n"
                "from langchain_openai import OpenAI\n"
                "model = OpenAI(temperature=0)"
            )
            raise NotImplementedError(msg)
        retriever_kwargs = retriever_kwargs or {}
        chain = RetrievalQAWithSourcesChain.from_chain_type(
            llm,
            retriever=self.vectorstore.as_retriever(**retriever_kwargs),
            **kwargs,
        )
        return chain.invoke({chain.question_key: question})

    async def aquery_with_sources(
        self,
        question: str,
        llm: BaseLanguageModel | None = None,
        retriever_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> dict:
        """异步查询 `VectorStore` 并获取答案和来源。

        Args:
            question: 要查询的问题或提示词。
            llm: 要使用的语言模型。不能为 `None`。
            retriever_kwargs: 可选的关键字参数，用于检索器。
            **kwargs: 额外的关键字参数，将转发给链。

        Returns:
            包含答案和源文档的 `dict`。
        """
        if llm is None:
            msg = (
                "This API has been changed to require an LLM. "
                "Please provide an llm to use for querying the vectorstore.\n"
                "For example,\n"
                "from langchain_openai import OpenAI\n"
                "model = OpenAI(temperature=0)"
            )
            raise NotImplementedError(msg)
        retriever_kwargs = retriever_kwargs or {}
        chain = RetrievalQAWithSourcesChain.from_chain_type(
            llm,
            retriever=self.vectorstore.as_retriever(**retriever_kwargs),
            **kwargs,
        )
        return await chain.ainvoke({chain.question_key: question})


def _get_in_memory_vectorstore() -> type[VectorStore]:
    """获取 `InMemoryVectorStore`。"""
    import warnings

    try:
        from langchain_community.vectorstores.inmemory import InMemoryVectorStore
    except ImportError as e:
        msg = "Please install langchain-community to use the InMemoryVectorStore."
        raise ImportError(msg) from e
    warnings.warn(
        "Using InMemoryVectorStore as the default vectorstore."
        "This memory store won't persist data. You should explicitly"
        "specify a VectorStore when using VectorstoreIndexCreator",
        stacklevel=3,
    )
    return InMemoryVectorStore


class VectorstoreIndexCreator(BaseModel):
    """创建索引的逻辑。"""

    vectorstore_cls: type[VectorStore] = Field(
        default_factory=_get_in_memory_vectorstore,
    )
    embedding: Embeddings
    text_splitter: TextSplitter = Field(default_factory=_get_default_text_splitter)
    vectorstore_kwargs: dict = Field(default_factory=dict)

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="forbid",
    )

    def from_loaders(self, loaders: list[BaseLoader]) -> VectorStoreIndexWrapper:
        """从加载器列表创建 `VectorStore` 索引。

        Args:
            loaders: 用于加载文档的 `BaseLoader` 实例列表。

        Returns:
            包含已构建向量存储的 `VectorStoreIndexWrapper`。
        """
        docs = []
        for loader in loaders:
            docs.extend(loader.load())
        return self.from_documents(docs)

    async def afrom_loaders(self, loaders: list[BaseLoader]) -> VectorStoreIndexWrapper:
        """异步从加载器列表创建 `VectorStore` 索引。

        Args:
            loaders: 用于加载文档的 `BaseLoader` 实例列表。

        Returns:
            包含已构建向量存储的 `VectorStoreIndexWrapper`。
        """
        docs = []
        for loader in loaders:
            docs.extend([doc async for doc in loader.alazy_load()])
        return await self.afrom_documents(docs)

    def from_documents(self, documents: list[Document]) -> VectorStoreIndexWrapper:
        """从文档列表创建 `VectorStore` 索引。

        Args:
            documents: `Document` 对象列表。

        Returns:
            包含已构建向量存储的 `VectorStoreIndexWrapper`。
        """
        sub_docs = self.text_splitter.split_documents(documents)
        vectorstore = self.vectorstore_cls.from_documents(
            sub_docs,
            self.embedding,
            **self.vectorstore_kwargs,
        )
        return VectorStoreIndexWrapper(vectorstore=vectorstore)

    async def afrom_documents(
        self,
        documents: list[Document],
    ) -> VectorStoreIndexWrapper:
        """异步从文档列表创建 `VectorStore` 索引。

        Args:
            documents: `Document` 对象列表。

        Returns:
            包含已构建向量存储的 `VectorStoreIndexWrapper`。
        """
        sub_docs = self.text_splitter.split_documents(documents)
        vectorstore = await self.vectorstore_cls.afrom_documents(
            sub_docs,
            self.embedding,
            **self.vectorstore_kwargs,
        )
        return VectorStoreIndexWrapper(vectorstore=vectorstore)

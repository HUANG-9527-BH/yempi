"""基于索引的带来源问答。"""

from typing import Any

from langchain_core.callbacks import (
    AsyncCallbackManagerForChainRun,
    CallbackManagerForChainRun,
)
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import Field

from langchain_classic.chains.combine_documents.stuff import StuffDocumentsChain
from langchain_classic.chains.qa_with_sources.base import BaseQAWithSourcesChain


class RetrievalQAWithSourcesChain(BaseQAWithSourcesChain):
    """基于索引的带来源问答。"""

    retriever: BaseRetriever = Field(exclude=True)
    """要连接的索引。"""
    reduce_k_below_max_tokens: bool = False
    """根据 token 限制减少从存储中返回的结果数量"""
    max_tokens_limit: int = 3375
    """Restrict the docs to return from store based on tokens,
    enforced only for StuffDocumentChain and if reduce_k_below_max_tokens is to true"""

    def _reduce_tokens_below_limit(self, docs: list[Document]) -> list[Document]:
        num_docs = len(docs)

        if self.reduce_k_below_max_tokens and isinstance(
            self.combine_documents_chain,
            StuffDocumentsChain,
        ):
            tokens = [
                self.combine_documents_chain.llm_chain._get_num_tokens(doc.page_content)  # noqa: SLF001
                for doc in docs
            ]
            token_count = sum(tokens[:num_docs])
            while token_count > self.max_tokens_limit:
                num_docs -= 1
                token_count -= tokens[num_docs]

        return docs[:num_docs]

    def _get_docs(
        self,
        inputs: dict[str, Any],
        *,
        run_manager: CallbackManagerForChainRun,
    ) -> list[Document]:
        question = inputs[self.question_key]
        docs = self.retriever.invoke(
            question,
            config={"callbacks": run_manager.get_child()},
        )
        return self._reduce_tokens_below_limit(docs)

    async def _aget_docs(
        self,
        inputs: dict[str, Any],
        *,
        run_manager: AsyncCallbackManagerForChainRun,
    ) -> list[Document]:
        question = inputs[self.question_key]
        docs = await self.retriever.ainvoke(
            question,
            config={"callbacks": run_manager.get_child()},
        )
        return self._reduce_tokens_below_limit(docs)

    @property
    def _chain_type(self) -> str:
        """返回链的类型。"""
        return "retrieval_qa_with_sources_chain"

import uuid
from collections.abc import Sequence
from typing import Any

from langchain_core.documents import Document
from langchain_text_splitters import TextSplitter

from langchain_classic.retrievers import MultiVectorRetriever


class ParentDocumentRetriever(MultiVectorRetriever):
    """检索小块内容，然后检索其父文档。

    在对文档进行分割以供检索时，常常存在相互冲突的需求：

    1.
    您可能希望文档较小，以便其嵌入能最准确地反映其含义。如果过长，则嵌入可能会丢失意义。
    2. 您又希望文档足够长，以保留每个块的上下文。

    The ParentDocumentRetriever strikes that balance by splitting and storing
    small chunks of data. During retrieval, it first fetches the small chunks
    but then looks up the parent IDs for those chunks and returns those larger
    documents.

    注意，“父文档”指的是小块内容所源自的文档。这可以是原始文档整体，也可以是更大的块。

    Examples:
        ```python
        from langchain_chroma import Chroma
        from langchain_community.embeddings import OpenAIEmbeddings
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        from langchain_classic.storage import InMemoryStore

        # This text splitter is used to create the parent documents
        parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000, add_start_index=True
        )
        # This text splitter is used to create the child documents
        # It should create documents smaller than the parent
        child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=400, add_start_index=True
        )
        # The VectorStore to use to index the child chunks
        vectorstore = Chroma(embedding_function=OpenAIEmbeddings())
        # The storage layer for the parent documents
        store = InMemoryStore()

        # Initialize the retriever
        retriever = ParentDocumentRetriever(
            vectorstore=vectorstore,
            docstore=store,
            child_splitter=child_splitter,
            parent_splitter=parent_splitter,
        )
        ```
    """

    child_splitter: TextSplitter
    """用于创建子文档的文本分割器。"""

    """用于追踪父文档 ID 的键。该键将存储在子文档的元数据中。"""
    parent_splitter: TextSplitter | None = None
    """用于创建父文档的文本分割器。
    如果未指定，则父文档即为传入的原始文档。"""

    child_metadata_fields: Sequence[str] | None = None
    """保留在子文档中的元数据字段。如果为 `None`，则保留所有父文档元数据。
    """

    def _split_docs_for_adding(
        self,
        documents: list[Document],
        ids: list[str] | None = None,
        *,
        add_to_docstore: bool = True,
    ) -> tuple[list[Document], list[tuple[str, Document]]]:
        if self.parent_splitter is not None:
            documents = self.parent_splitter.split_documents(documents)
        if ids is None:
            doc_ids = [str(uuid.uuid4()) for _ in documents]
            if not add_to_docstore:
                msg = "If IDs are not passed in, `add_to_docstore` MUST be True"
                raise ValueError(msg)
        else:
            if len(documents) != len(ids):
                msg = (
                    "Got uneven list of documents and ids. "
                    "If `ids` is provided, should be same length as `documents`."
                )
                raise ValueError(msg)
            doc_ids = ids

        docs = []
        full_docs = []
        for i, doc in enumerate(documents):
            _id = doc_ids[i]
            sub_docs = self.child_splitter.split_documents([doc])
            if self.child_metadata_fields is not None:
                for _doc in sub_docs:
                    _doc.metadata = {
                        k: _doc.metadata[k] for k in self.child_metadata_fields
                    }
            for _doc in sub_docs:
                _doc.metadata[self.id_key] = _id
            docs.extend(sub_docs)
            full_docs.append((_id, doc))

        return docs, full_docs

    def add_documents(
        self,
        documents: list[Document],
        ids: list[str] | None = None,
        add_to_docstore: bool = True,  # noqa: FBT001,FBT002
        **kwargs: Any,
    ) -> None:
        """将文档添加到文档存储和向量存储中。

        Args:
            documents: List of documents to add
            ids: Optional list of IDs for documents. If provided should be the same
                length as the list of documents. Can be provided if parent documents
                are already in the document store and you don't want to re-add
                to the docstore. If not provided, random UUIDs will be used as
                IDs.
            add_to_docstore: Boolean of whether to add documents to docstore.
                This can be false if and only if `ids` are provided. You may want
                to set this to False if the documents are already in the docstore
                and you don't want to re-add them.
            **kwargs: additional keyword arguments passed to the `VectorStore`.
        """
        docs, full_docs = self._split_docs_for_adding(
            documents,
            ids,
            add_to_docstore=add_to_docstore,
        )
        self.vectorstore.add_documents(docs, **kwargs)
        if add_to_docstore:
            self.docstore.mset(full_docs)

    async def aadd_documents(
        self,
        documents: list[Document],
        ids: list[str] | None = None,
        add_to_docstore: bool = True,  # noqa: FBT001,FBT002
        **kwargs: Any,
    ) -> None:
        """将文档添加到文档存储和向量存储中。

        Args:
            documents: List of documents to add
            ids: Optional list of IDs for documents. If provided should be the same
                length as the list of documents. Can be provided if parent documents
                are already in the document store and you don't want to re-add
                to the docstore. If not provided, random UUIDs will be used as
                idIDss.
            add_to_docstore: Boolean of whether to add documents to docstore.
                This can be false if and only if `ids` are provided. You may want
                to set this to False if the documents are already in the docstore
                and you don't want to re-add them.
            **kwargs: additional keyword arguments passed to the `VectorStore`.
        """
        docs, full_docs = self._split_docs_for_adding(
            documents,
            ids,
            add_to_docstore=add_to_docstore,
        )
        await self.vectorstore.aadd_documents(docs, **kwargs)
        if add_to_docstore:
            await self.docstore.amset(full_docs)

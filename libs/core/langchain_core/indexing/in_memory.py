"""内存中的文档索引。"""

import operator
import uuid
from collections.abc import Sequence
from typing import Any, cast

from pydantic import Field
from typing_extensions import override

from langchain_core._api import beta
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.indexing import UpsertResponse
from langchain_core.indexing.base import DeleteResponse, DocumentIndex


@beta(message="Introduced in version 0.2.29. Underlying abstraction subject to change.")
class InMemoryDocumentIndex(DocumentIndex):
    """内存中的文档索引。

    这是一个将文档存储在字典中的内存文档索引。

    它提供了一个简单的搜索 API，根据给定查询在文档中出现的次数返回文档。
    """

    store: dict[str, Document] = Field(default_factory=dict)
    top_k: int = 4

    @override
    def upsert(self, items: Sequence[Document], /, **kwargs: Any) -> UpsertResponse:
        """将文档插入索引。

        Args:
            items: 要添加到索引中的文档序列。
            **kwargs: 额外的关键字参数。

        Returns:
            一个包含成功添加或更新的 ID 列表以及未能添加或更新的 ID 列表的响应对象。
        """
        ok_ids = []

        for item in items:
            if item.id is None:
                id_ = str(uuid.uuid4())
                item_ = item.model_copy()
                item_.id = id_
            else:
                item_ = item
                id_ = item.id

            self.store[id_] = item_
            ok_ids.append(cast("str", item_.id))

        return UpsertResponse(succeeded=ok_ids, failed=[])

    @override
    def delete(self, ids: list[str] | None = None, **kwargs: Any) -> DeleteResponse:
        """通过 ID 删除。

        Args:
            ids: 要删除的 ID 列表。

        Raises:
            ValueError: 如果 IDs 为 None。

        Returns:
            一个包含成功删除的 ID 列表以及未能删除的 ID 列表的响应对象。
        """
        if ids is None:
            msg = "IDs must be provided for deletion"
            raise ValueError(msg)

        ok_ids = []

        for id_ in ids:
            if id_ in self.store:
                del self.store[id_]
                ok_ids.append(id_)

        return DeleteResponse(
            succeeded=ok_ids, num_deleted=len(ok_ids), num_failed=0, failed=[]
        )

    @override
    def get(self, ids: Sequence[str], /, **kwargs: Any) -> list[Document]:
        return [self.store[id_] for id_ in ids if id_ in self.store]

    @override
    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        counts_by_doc = []

        for document in self.store.values():
            count = document.page_content.count(query)
            counts_by_doc.append((document, count))

        counts_by_doc.sort(key=operator.itemgetter(1), reverse=True)
        return [doc.model_copy() for doc, count in counts_by_doc[: self.top_k]]

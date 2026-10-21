"""索引的基础类。"""

from __future__ import annotations

import abc
import time
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, TypedDict

from typing_extensions import override

from langchain_core._api import beta
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import run_in_executor

if TYPE_CHECKING:
    from collections.abc import Sequence

    from langchain_core.documents import Document


class RecordManager(ABC):
    """表示记录管理器接口的抽象基类。

    记录管理器抽象由 langchain 索引 API 使用。

    记录管理器跟踪哪些文档已被写入 `VectorStore` 以及写入时间。

    The indexing API computes hashes for each document and stores the hash
    together with the write time and the source id in the record manager.

    在后续的索引运行中，索引 API
    可以检查记录管理器，以确定哪些文档已索引，哪些尚未索引。

    这使得索引 API 能够避免重新索引已索引的文档，仅索引新文档。

    此抽象的主要优势在于它可跨多种向量存储工作。要被支持，`VectorStore` 只需支持通过 ID
    添加和删除文档。使用记录管理器，索引 API
    将能够删除过时文档，并避免对已索引文档进行重复索引。

    此抽象的主要约束是：

    1. It relies on the time-stamps to determine which documents have been
        indexed and which have not. This means that the time-stamps must be
        monotonically increasing. The timestamp should be the timestamp
        as measured by the server to minimize issues.
    2. The record manager is currently implemented separately from the
        vectorstore, which means that the overall system becomes distributed
        and may create issues with consistency. For example, writing to
        record manager succeeds, but corresponding writing to `VectorStore` fails.
    """

    def __init__(
        self,
        namespace: str,
    ) -> None:
        """初始化记录管理器。

        Args:
            namespace: 记录管理器的命名空间。
        """
        self.namespace = namespace

    @abstractmethod
    def create_schema(self) -> None:
        """创建记录管理器的数据库模式。"""

    @abstractmethod
    async def acreate_schema(self) -> None:
        """异步创建记录管理器的数据库模式。"""

    @abstractmethod
    def get_time(self) -> float:
        """以高精度时间戳形式获取当前服务器时间！

        从服务器获取此时间非常重要，以确保时钟单调递增，
        否则在清理旧文档时可能会导致数据丢失！

        Returns:
            当前服务器时间，以浮点数时间戳表示。
        """

    @abstractmethod
    async def aget_time(self) -> float:
        """异步以高精度时间戳形式获取当前服务器时间。

        从服务器获取此时间非常重要，以确保时钟单调递增，
        否则在清理旧文档时可能会导致数据丢失！

        Returns:
            当前服务器时间，以浮点数时间戳表示。
        """

    @abstractmethod
    def update(
        self,
        keys: Sequence[str],
        *,
        group_ids: Sequence[str | None] | None = None,
        time_at_least: float | None = None,
    ) -> None:
        """将记录插入或更新到数据库。

        Args:
            keys: 要插入或更新的记录键列表。 group_ids: 与键对应的组 ID 列表。
            time_at_least: 可选的时间戳。实现可使用此参数
            选择性地验证系统中存储的数据的时间戳**至少**为此时间。

                e.g., use to validate that the time in the postgres database
                is equal to or larger than the given timestamp, if not
                raise an error.

                这有助于防止因时间非单调递增而引发的时间漂移问题！

        Raises:
            ValueError: 如果 keys 的长度与 group_ids 的长度不匹配。
        """

    @abstractmethod
    async def aupdate(
        self,
        keys: Sequence[str],
        *,
        group_ids: Sequence[str | None] | None = None,
        time_at_least: float | None = None,
    ) -> None:
        """异步将记录插入或更新到数据库。

        Args:
            keys: 要插入或更新的记录键列表。 group_ids: 与键对应的组 ID 列表。
            time_at_least: 可选的时间戳。实现可使用此参数
            选择性地验证系统中存储的数据的时间戳**至少**为此时间。

                e.g., use to validate that the time in the postgres database
                is equal to or larger than the given timestamp, if not
                raise an error.

                这有助于防止因时间非单调递增而引发的时间漂移问题！

        Raises:
            ValueError: 如果 keys 的长度与 group_ids 的长度不匹配。
        """

    @abstractmethod
    def exists(self, keys: Sequence[str]) -> list[bool]:
        """检查提供的 keys 是否存在于数据库中。

        Args:
            keys: 要检查的一组 keys。

        Returns:
            一个布尔值列表，指示每个 key 的存在性。
        """

    @abstractmethod
    async def aexists(self, keys: Sequence[str]) -> list[bool]:
        """异步检查提供的 keys 是否存在于数据库中。

        Args:
            keys: 要检查的一组 keys。

        Returns:
            一个布尔值列表，指示每个 key 的存在性。
        """

    @abstractmethod
    def list_keys(
        self,
        *,
        before: float | None = None,
        after: float | None = None,
        group_ids: Sequence[str] | None = None,
        limit: int | None = None,
    ) -> list[str]:
        """根据提供的过滤条件列出数据库中的记录。

        Args:
            before: 过滤条件，列出在此时间之前更新的记录。
            after: 过滤条件，列出在此时间之后更新的记录。
            group_ids: 过滤条件，列出具有特定 group ID 的记录。
            limit: 可选的返回记录数量限制。

        Returns:
            匹配记录的 keys 列表。
        """

    @abstractmethod
    async def alist_keys(
        self,
        *,
        before: float | None = None,
        after: float | None = None,
        group_ids: Sequence[str] | None = None,
        limit: int | None = None,
    ) -> list[str]:
        """异步根据提供的过滤条件列出数据库中的记录。

        Args:
            before: 过滤条件，列出在此时间之前更新的记录。
            after: 过滤条件，列出在此时间之后更新的记录。
            group_ids: 过滤条件，列出具有特定 group ID 的记录。
            limit: 可选的返回记录数量限制。

        Returns:
            匹配记录的 keys 列表。
        """

    @abstractmethod
    def delete_keys(self, keys: Sequence[str]) -> None:
        """从数据库中删除指定的记录。

        Args:
            keys: 要删除的一组 keys。
        """

    @abstractmethod
    async def adelete_keys(self, keys: Sequence[str]) -> None:
        """异步从数据库中删除指定的记录。

        Args:
            keys: 要删除的一组 keys。
        """


class _Record(TypedDict):
    group_id: str | None
    updated_at: float


class InMemoryRecordManager(RecordManager):
    """用于测试目的的内存中记录管理器。"""

    def __init__(self, namespace: str) -> None:
        """初始化内存中的记录管理器。

        Args:
            namespace: 记录管理器的命名空间。
        """
        super().__init__(namespace)
                # 每个键指向一个字典
                # 包含 {'group_id': group_id, 'updated_at': timestamp}
        self.records: dict[str, _Record] = {}
        self.namespace = namespace

    def create_schema(self) -> None:
        """内存中模式创建只是确保结构已初始化。"""

    async def acreate_schema(self) -> None:
        """内存中模式创建只是确保结构已初始化。"""

    @override
    def get_time(self) -> float:
        return time.time()

    @override
    async def aget_time(self) -> float:
        return self.get_time()

    def update(
        self,
        keys: Sequence[str],
        *,
        group_ids: Sequence[str | None] | None = None,
        time_at_least: float | None = None,
    ) -> None:
        """将记录插入或更新到数据库中。

        Args:
            keys: 要插入或更新的一组 record keys。
            group_ids: 与 keys 对应的一组 group IDs。

            time_at_least: Optional timestamp. Implementation can use this
                to optionally verify that the timestamp IS at least this time
                in the system that stores.
                E.g., use to validate that the time in the postgres database
                is equal to or larger than the given timestamp, if not
                raise an error.
                This is meant to help prevent time-drift issues since
                time may not be monotonically increasing!

        Raises:
            ValueError: 如果 keys 的长度与 group
                ids 的长度不匹配。
            ValueError: 如果 time_at_least 在未来时间。
        """
        if group_ids and len(keys) != len(group_ids):
            msg = "Length of keys must match length of group_ids"
            raise ValueError(msg)
        for index, key in enumerate(keys):
            group_id = group_ids[index] if group_ids else None
            if time_at_least and time_at_least > self.get_time():
                msg = "time_at_least must be in the past"
                raise ValueError(msg)
            self.records[key] = {"group_id": group_id, "updated_at": self.get_time()}

    async def aupdate(
        self,
        keys: Sequence[str],
        *,
        group_ids: Sequence[str | None] | None = None,
        time_at_least: float | None = None,
    ) -> None:
        """异步将记录插入或更新到数据库中。

        Args:
            keys: 要插入或更新的一组 record keys。
            group_ids: 与 keys 对应的一组 group IDs。

            time_at_least: Optional timestamp. Implementation can use this
                to optionally verify that the timestamp IS at least this time
                in the system that stores.
                E.g., use to validate that the time in the postgres database
                is equal to or larger than the given timestamp, if not
                raise an error.
                This is meant to help prevent time-drift issues since
                time may not be monotonically increasing!
        """
        self.update(keys, group_ids=group_ids, time_at_least=time_at_least)

    def exists(self, keys: Sequence[str]) -> list[bool]:
        """检查提供的键是否存在于数据库中。

        Args:
            keys: 要检查的键列表。

        Returns:
            每个键是否存在对应的布尔值列表。
        """
        return [key in self.records for key in keys]

    async def aexists(self, keys: Sequence[str]) -> list[bool]:
        """异步检查提供的键是否存在于数据库中。

        Args:
            keys: 要检查的键列表。

        Returns:
            每个键是否存在对应的布尔值列表。
        """
        return self.exists(keys)

    def list_keys(
        self,
        *,
        before: float | None = None,
        after: float | None = None,
        group_ids: Sequence[str] | None = None,
        limit: int | None = None,
    ) -> list[str]:
        """根据提供的过滤条件列出数据库中的记录。

        Args:
            before: 过滤条件，列出在此时间之前更新的记录。

            after: 过滤条件，列出在此时间之后更新的记录。

            group_ids: 过滤条件，列出具有特定组 ID 的记录。

            limit: 可选的返回记录数量限制。


        Returns:
            匹配记录的键列表。
        """
        result = []
        for key, data in self.records.items():
            if before and data["updated_at"] >= before:
                continue
            if after and data["updated_at"] <= after:
                continue
            if group_ids and data["group_id"] not in group_ids:
                continue
            result.append(key)
        if limit:
            return result[:limit]
        return result

    async def alist_keys(
        self,
        *,
        before: float | None = None,
        after: float | None = None,
        group_ids: Sequence[str] | None = None,
        limit: int | None = None,
    ) -> list[str]:
        """异步根据提供的过滤条件列出数据库中的记录。

        Args:
            before: 过滤条件，列出在此时间之前更新的记录。

            after: 过滤条件，列出在此时间之后更新的记录。

            group_ids: 过滤条件，列出具有特定组 ID 的记录。

            limit: 可选的返回记录数量限制。


        Returns:
            匹配记录的键列表。
        """
        return self.list_keys(
            before=before, after=after, group_ids=group_ids, limit=limit
        )

    def delete_keys(self, keys: Sequence[str]) -> None:
        """从数据库中删除指定的记录。

        Args:
            keys: 要删除的键列表。
        """
        for key in keys:
            if key in self.records:
                del self.records[key]

    async def adelete_keys(self, keys: Sequence[str]) -> None:
        """异步从数据库中删除指定的记录。

        Args:
            keys: 要删除的键列表。
        """
        self.delete_keys(keys)


class UpsertResponse(TypedDict):
    """通用的 upsert 操作响应。

    The upsert response will be used by abstractions that implement an upsert
    operation for content that can be upserted by ID.

    Upsert APIs that accept inputs with IDs and generate IDs internally
    will return a response that includes the IDs that succeeded and the IDs
    that failed.

    若无失败，失败列表为空，成功列表中的 ID 顺序与输入文档顺序一致。

    若存在失败，响应变得不明确，API 用户无法确定哪个生成的 ID 对应哪个输入文档。

    It is recommended for users explicitly attach the IDs to the items being
    indexed to avoid this issue.
    """

    succeeded: list[str]
    """成功索引的 ID。"""
    failed: list[str]
    """未能索引的 ID。"""


class DeleteResponse(TypedDict, total=False):
    """通用的删除操作响应。

    此响应中的字段为可选，`VectorStore` 是否返回取决于具体实现。
    """

    num_deleted: int
    """成功删除的项目数量。

    若返回，此数量应仅包含*实际*删除的条目。

    若 ID 本就不存在，不应包含在此计数中。
    """

    succeeded: Sequence[str]
    """成功删除的 ID。

    如果返回，此列表应仅包含*实际*删除的项。

    如果 ID 一开始并不存在，
    则不应包含在此列表中。
    """

    failed: Sequence[str]
    """未能删除的 ID。

    !!! warning
        删除一个不存在的 ID **不**被视为失败。
    """

    num_failed: int
    """未能删除的项目数量。"""


@beta(message="Added in 0.2.29. The abstraction is subject to change.")
class DocumentIndex(BaseRetriever):
    """支持索引操作的文档检索器。

    This indexing interface is designed to be a generic abstraction for storing and
    querying documents that has an ID and metadata associated with it.

    该接口设计为对底层索引系统的实现方式无感知。

    该接口旨在支持以下操作：

    1. 将文档存储到索引中。
    2. 通过 ID 获取文档。
    3. 使用查询搜索文档。
    """

    @abc.abstractmethod
    def upsert(self, items: Sequence[Document], /, **kwargs: Any) -> UpsertResponse:
        """将文档插入索引。

        The upsert functionality should utilize the ID field of the content object
        if it is provided. If the ID is not provided, the upsert method is free
        to generate an ID for the content.

        When an ID is specified and the content already exists in the `VectorStore`,
        the upsert method should update the content with the new data. If the content
        does not exist, the upsert method should add the item to the `VectorStore`.

        Args:
            items: 要添加到 `VectorStore` 的文档序列。
            **kwargs: 额外的关键字参数。

        Returns:
            包含成功添加或更新的 ID 列表以及未能添加或更新的 ID 列表的响应对象。
        """

    async def aupsert(
        self, items: Sequence[Document], /, **kwargs: Any
    ) -> UpsertResponse:
        """在 `VectorStore` 中添加或更新文档。`upsert` 的异步版本。

        The upsert functionality should utilize the ID field of the item
        if it is provided. If the ID is not provided, the upsert method is free
        to generate an ID for the item.

        When an ID is specified and the item already exists in the `VectorStore`,
        the upsert method should update the item with the new data. If the item
        does not exist, the upsert method should add the item to the `VectorStore`.

        Args:
            items: 要添加到 `VectorStore` 的文档序列。
            **kwargs: 额外的关键字参数。

        Returns:
            包含成功添加或更新的 ID 列表以及未能添加或更新的 ID 列表的响应对象。
        """
        return await run_in_executor(
            None,
            self.upsert,
            items,
            **kwargs,
        )

    @abc.abstractmethod
    def delete(self, ids: list[str] | None = None, **kwargs: Any) -> DeleteResponse:
        """通过 ID 或其他标准删除。

        调用 delete 时若未提供任何输入参数，应抛出 ValueError！

        Args:
            ids: 要删除的 ID 列表。
            **kwargs: 额外的关键字参数。具体实现可自行决定。
                例如，可包含删除整个索引的选项，
                或发出非阻塞删除请求等。

        Returns:
            包含成功删除的 ID 列表以及未能删除的 ID 列表的响应对象。
        """

    async def adelete(
        self, ids: list[str] | None = None, **kwargs: Any
    ) -> DeleteResponse:
        """通过 ID 或其他标准删除。异步版本。

        调用 adelete 时若未提供任何输入参数，应抛出 ValueError！

        Args:
            ids: 要删除的 ID 列表。
            **kwargs: 额外的关键字参数。具体实现可自行决定。
                例如，可包含删除整个索引的选项。

        Returns:
            包含成功删除的 ID 列表以及未能删除的 ID 列表的响应对象。
        """
        return await run_in_executor(
            None,
            self.delete,
            ids,
            **kwargs,
        )

    @abc.abstractmethod
    def get(
        self,
        ids: Sequence[str],
        /,
        **kwargs: Any,
    ) -> list[Document]:
        """通过 ID 获取文档。

        Fewer documents may be returned than requested if some IDs are not found or
        if there are duplicated IDs.

        用户不应假设返回文档的顺序与输入 ID 的顺序一致。相反，用户应依赖返回文档中的 ID
        字段。

        This method should **NOT** raise exceptions if no documents are found for
        some IDs.

        Args:
            ids: 要获取的 ID 列表。
            **kwargs: 额外的关键字参数。具体实现可自行决定。

        Returns:
            找到的文档列表。
        """

    async def aget(
        self,
        ids: Sequence[str],
        /,
        **kwargs: Any,
    ) -> list[Document]:
        """通过 ID 获取文档。

        Fewer documents may be returned than requested if some IDs are not found or
        if there are duplicated IDs.

        用户不应假设返回文档的顺序与输入 ID 的顺序一致。相反，用户应依赖返回文档中的 ID
        字段。

        This method should **NOT** raise exceptions if no documents are found for
        some IDs.

        Args:
            ids: 要获取的 ID 列表。
            **kwargs: 额外的关键字参数。具体实现可自行决定。

        Returns:
            找到的文档列表。
        """
        return await run_in_executor(
            None,
            self.get,
            ids,
            **kwargs,
        )

"""模块包含将文档索引到向量存储中的逻辑。"""

from __future__ import annotations

import hashlib
import json
import uuid
import warnings
from itertools import islice
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    TypedDict,
    TypeVar,
    cast,
)

from langchain_core.document_loaders.base import BaseLoader
from langchain_core.documents import Document
from langchain_core.exceptions import LangChainException
from langchain_core.indexing.base import DocumentIndex, RecordManager
from langchain_core.vectorstores import VectorStore

if TYPE_CHECKING:
    from collections.abc import (
        AsyncIterable,
        AsyncIterator,
        Callable,
        Iterable,
        Iterator,
        Sequence,
    )

# 用于哈希的魔术 UUID 命名空间。
# 用于尝试为每个文档生成唯一 UUID
# 通过哈希文档内容和元数据实现。
NAMESPACE_UUID = uuid.UUID(int=1984)


T = TypeVar("T")


def _hash_string_to_uuid(input_string: str) -> str:
    """对字符串进行哈希并返回对应的 UUID。"""
    hash_value = hashlib.sha1(
        input_string.encode("utf-8"), usedforsecurity=False
    ).hexdigest()
    return str(uuid.uuid5(NAMESPACE_UUID, hash_value))


_WARNED_ABOUT_SHA1: bool = False


def _warn_about_sha1() -> None:
    """发出关于 SHA-1 冲突弱点的一次性警告。"""
        # 全局变量在此情况下可以接受
    global _WARNED_ABOUT_SHA1  # noqa: PLW0603
    if not _WARNED_ABOUT_SHA1:
        warnings.warn(
            "Using SHA-1 for document hashing. SHA-1 is *not* "
            "collision-resistant; a motivated attacker can construct distinct inputs "
            "that map to the same fingerprint. If this matters in your "
            "threat model, switch to a stronger algorithm such "
            "as 'blake2b', 'sha256', or 'sha512' by specifying "
            " `key_encoder` parameter in the `index` or `aindex` function. ",
            category=UserWarning,
            stacklevel=2,
        )
        _WARNED_ABOUT_SHA1 = True


def _hash_string(
    input_string: str, *, algorithm: Literal["sha1", "sha256", "sha512", "blake2b"]
) -> uuid.UUID:
    """使用配置的算法将 *input_string* 哈希为确定性的 UUID。"""
    if algorithm == "sha1":
        _warn_about_sha1()
    hash_value = _calculate_hash(input_string, algorithm)
    return uuid.uuid5(NAMESPACE_UUID, hash_value)


def _hash_nested_dict(
    data: dict[Any, Any], *, algorithm: Literal["sha1", "sha256", "sha512", "blake2b"]
) -> uuid.UUID:
    """使用配置的算法，将嵌套字典哈希为 UUID。"""
    serialized_data = json.dumps(data, sort_keys=True)
    return _hash_string(serialized_data, algorithm=algorithm)


def _batch(size: int, iterable: Iterable[T]) -> Iterator[list[T]]:
    """实用的批处理函数。"""
    if size <= 0:
        msg = f"Batch size must be a positive integer, got {size}."
        raise ValueError(msg)
    it = iter(iterable)
    while True:
        chunk = list(islice(it, size))
        if not chunk:
            return
        yield chunk


async def _abatch(size: int, iterable: AsyncIterable[T]) -> AsyncIterator[list[T]]:
    """实用的批处理函数。"""
    if size <= 0:
        msg = f"Batch size must be a positive integer, got {size}."
        raise ValueError(msg)
    batch: list[T] = []
    async for element in iterable:
        if len(batch) < size:
            batch.append(element)

        if len(batch) >= size:
            yield batch
            batch = []

    if batch:
        yield batch


def _get_source_id_assigner(
    source_id_key: str | Callable[[Document], str] | None,
) -> Callable[[Document], str | None]:
    """从文档中获取源 ID。"""
    if source_id_key is None:
        return lambda _doc: None
    if isinstance(source_id_key, str):
        return lambda doc: doc.metadata[source_id_key]
    if callable(source_id_key):
        return source_id_key
    msg = (  # type: ignore[unreachable]
        f"source_id_key should be either None, a string or a callable. "
        f"Got {source_id_key} of type {type(source_id_key)}."
    )
    raise ValueError(msg)


def _deduplicate_in_order(
    hashed_documents: Iterable[Document],
) -> Iterator[Document]:
    """在保持顺序的同时去重已哈希的文档列表。"""
    seen: set[str] = set()

    for hashed_doc in hashed_documents:
        if hashed_doc.id not in seen:
                        # 此时，id 被保证为字符串。
                        # 避免不必要的运行时检查。
            seen.add(cast("str", hashed_doc.id))
            yield hashed_doc


class IndexingException(LangChainException):
    """当索引操作失败时抛出。"""


def _calculate_hash(
    text: str, algorithm: Literal["sha1", "sha256", "sha512", "blake2b"]
) -> str:
    """使用 *algorithm* 对 *text* 计算十六进制摘要。"""
    if algorithm == "sha1":
                # 计算 SHA-1 哈希并将其作为 UUID 返回。
        digest = hashlib.sha1(text.encode("utf-8"), usedforsecurity=False).hexdigest()
        return str(uuid.uuid5(NAMESPACE_UUID, digest))
    if algorithm == "blake2b":
        return hashlib.blake2b(text.encode("utf-8")).hexdigest()
    if algorithm == "sha256":
        return hashlib.sha256(text.encode("utf-8")).hexdigest()
    if algorithm == "sha512":
        return hashlib.sha512(text.encode("utf-8")).hexdigest()
    msg = f"Unsupported hashing algorithm: {algorithm}"  # type: ignore[unreachable]
    raise ValueError(msg)


def _get_document_with_hash(
    document: Document,
    *,
    key_encoder: Callable[[Document], str]
    | Literal["sha1", "sha256", "sha512", "blake2b"],
) -> Document:
    """计算文档的哈希值，并将其分配给 uid。

    使用预定义的哈希算法时，哈希值通过哈希文档的内容和元数据来计算。

    Args:
        document: Document to hash.
        key_encoder: Hashing algorithm to use for hashing the document.
            If not provided, a default encoder using SHA-1 will be used.
            SHA-1 is not collision-resistant, and a motivated attacker
            could craft two different texts that hash to the
            same cache key.

            新应用应使用其他替代编码器，
            或提供自定义且强健的键编码函数以避免此风险。

            更改键编码器时，必须同时更改索引，
            以避免缓存中出现重复文档。

    Raises:
        ValueError: 如果元数据无法使用 json 序列化。

    Returns:
        基于内容和元数据哈希值的唯一标识符文档。
    """
    metadata: dict[str, Any] = dict(document.metadata or {})

    if callable(key_encoder):
                # 如果 key_encoder 是可调用对象，则使用它生成哈希。
        hash_ = key_encoder(document)
    else:
                # 内容和元数据的哈希分别计算。
        content_hash = _calculate_hash(document.page_content, algorithm=key_encoder)
        try:
            serialized_meta = json.dumps(metadata, sort_keys=True)
        except Exception as e:
            msg = (
                f"Failed to hash metadata: {e}. "
                f"Please use a dict that can be serialized using json."
            )
            raise ValueError(msg) from e
        metadata_hash = _calculate_hash(serialized_meta, algorithm=key_encoder)
        hash_ = _calculate_hash(content_hash + metadata_hash, algorithm=key_encoder)

    return Document(
                # 根据哈希分配唯一标识符。
        id=hash_,
        page_content=document.page_content,
        metadata=document.metadata,
    )


# 此内部抽象由 langchain 包内部导入，因此
# 我们保留它以保持向后兼容性。
class _HashedDocument:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """实例化此类时抛出错误。"""
        msg = (
            "_HashedDocument is an internal abstraction that was deprecated in "
            " langchain-core 0.3.63. This abstraction is marked as private and "
            " should not have been used directly. If you are seeing this error, please "
            " update your code appropriately."
        )
        raise NotImplementedError(msg)


def _delete(
    vector_store: VectorStore | DocumentIndex,
    ids: list[str],
) -> None:
    """根据其 ID 从向量存储或文档索引中删除文档。

    Args:
        vector_store: 要删除的向量存储或文档索引。
        ids: 要删除的文档 ID 列表。

    Raises:
        IndexingException: 如果删除操作失败。
        TypeError: 如果 `vector_store` 既不是 `VectorStore` 也不是
            `DocumentIndex`。
    """
    if isinstance(vector_store, VectorStore):
        delete_ok = vector_store.delete(ids)
        if delete_ok is not None and delete_ok is False:
            msg = "The delete operation to VectorStore failed."
            raise IndexingException(msg)
    elif isinstance(vector_store, DocumentIndex):
        delete_response = vector_store.delete(ids)
        if "num_failed" in delete_response and delete_response["num_failed"] > 0:
            msg = "The delete operation to DocumentIndex failed."
            raise IndexingException(msg)
    else:
        msg = (  # type: ignore[unreachable]
            f"Vectorstore should be either a VectorStore or a DocumentIndex. "
            f"Got {type(vector_store)}."
        )
        raise TypeError(msg)


# PUBLIC API


class IndexingResult(TypedDict):
    """返回索引操作结果的详细分解。"""

    num_added: int
    """已添加的文档数量。"""
    num_updated: int
    """因未及时更新而被更新的文档数量。"""
    num_deleted: int
    """已删除的文档数量。"""
    num_skipped: int
    """因已及时更新而被跳过的文档数量。"""


def index(
    docs_source: BaseLoader | Iterable[Document],
    record_manager: RecordManager,
    vector_store: VectorStore | DocumentIndex,
    *,
    batch_size: int = 100,
    cleanup: Literal["incremental", "full", "scoped_full"] | None = None,
    source_id_key: str | Callable[[Document], str] | None = None,
    cleanup_batch_size: int = 1_000,
    force_update: bool = False,
    key_encoder: Literal["sha1", "sha256", "sha512", "blake2b"]
    | Callable[[Document], str] = "sha1",
    upsert_kwargs: dict[str, Any] | None = None,
) -> IndexingResult:
    """将加载器中的数据索引到向量存储中。

    索引功能使用管理器来跟踪哪些文档在向量存储中。

    这使我们能够跟踪哪些文档被更新、哪些文档被删除，以及哪些文档应被跳过。

    目前，文档通过其哈希值进行索引，用户无法指定文档的 uid。

    !!! warning "在 `langchain-core` 0.3.25 中行为变更"

        添加了 `scoped_full` 清理模式。

    !!! warning

        * In full mode, the loader should be returning
            the entire dataset, and not just a subset of the dataset.
            Otherwise, the auto_cleanup will remove documents that it is not
            supposed to.
        * In incremental mode, if documents associated with a particular
            source id appear across different batches, the indexing API
            will do some redundant work. This will still result in the
            correct end state of the index, but will unfortunately not be
            100% efficient. For example, if a given document is split into 15
            chunks, and we index them using a batch size of 5, we'll have 3 batches
            all with the same source id. In general, to avoid doing too much
            redundant work select as big a batch size as possible.
        * The `scoped_full` mode is suitable if determining an appropriate batch size
            is challenging or if your data loader cannot return the entire dataset at
            once. This mode keeps track of source IDs in memory, which should be fine
            for most use cases. If your dataset is large (10M+ docs), you will likely
            need to parallelize the indexing process regardless.

    Args:
        docs_source: 要索引的数据加载器或文档可迭代对象。 record_manager:
        带时间戳的集合，用于跟踪哪些文档已被更新。 vector_store: 要索引文档的
        `VectorStore` 或 DocumentIndex。 batch_size: 索引时使用的批次大小。 cleanup:
        如何处理文档清理。

            - incremental: Cleans up all documents that haven't been updated AND
                that are associated with source IDs that were seen during indexing.
                Clean up is done continuously during indexing helping to minimize the
                probability of users seeing duplicated content.
            - full: Delete all documents that have not been returned by the loader
                during this run of indexing.
                Clean up runs after all documents have been indexed.
                This means that users may see duplicated content during indexing.
            - scoped_full: Similar to Full, but only deletes all documents
                that haven't been updated AND that are associated with
                source IDs that were seen during indexing.
            - None: Do not delete any documents.
        source_id_key: Optional key that helps identify the original source
            of the document.
        cleanup_batch_size: Batch size to use when cleaning up documents.
        force_update: Force update documents even if they are present in the
            record manager. Useful if you are re-indexing with updated embeddings.
        key_encoder: Hashing algorithm to use for hashing the document content and
            metadata. Options include "blake2b", "sha256", and "sha512".

            !!! version-added "Added in `langchain-core` 0.3.66"

        key_encoder: Hashing algorithm to use for hashing the document.
            If not provided, a default encoder using SHA-1 will be used.
            SHA-1 is not collision-resistant, and a motivated attacker
            could craft two different texts that hash to the
            same cache key.

            新应用应使用其他替代编码器，
            或提供自定义且强健的键编码函数，以规避此风险。

            When changing the key encoder, you must change the
            index as well to avoid duplicated documents in the cache.
        upsert_kwargs: Additional keyword arguments to pass to the add_documents
            method of the `VectorStore` or the upsert method of the DocumentIndex.
            For example, you can use this to specify a custom vector_field:
            upsert_kwargs={"vector_field": "embedding"}
            !!! version-added "Added in `langchain-core` 0.3.10"

    Returns:
        索引结果，包含添加、更新、删除或跳过的文档数量信息。

    Raises:
        ValueError: 如果清理模式不是 'incremental'、'full' 或 None。 ValueError:
        如果清理模式为 incremental 且 source_id_key 为 None。 ValueError: 如果
        `VectorStore` 缺少所需的 "delete" 和 "add_documents" 方法。 ValueError: 如果
        source_id_key 不为 None，但不是字符串或可调用对象。 TypeError: 如果
        `vectorstore` 不是 `VectorStore` 或 DocumentIndex。 AssertionError: 如果
        cleanup 模式为 incremental 时 `source_id` 为 None。 （应为不可达代码）。
    """
        # 行为已弃用，但我们保留它以保持向后兼容性。
        # # 每个进程仅警告一次。
    if key_encoder == "sha1":
        _warn_about_sha1()

    if cleanup not in {"incremental", "full", "scoped_full", None}:
        msg = (
            f"cleanup should be one of 'incremental', 'full', 'scoped_full' or None. "
            f"Got {cleanup}."
        )
        raise ValueError(msg)

    if (cleanup in {"incremental", "scoped_full"}) and source_id_key is None:
        msg = (
            "Source id key is required when cleanup mode is incremental or scoped_full."
        )
        raise ValueError(msg)

    destination = vector_store  # 内部重命名以提高清晰度

        # 如果是向量存储，检查其是否具有所需方法。
    if isinstance(destination, VectorStore):
                # 检查 VectorStore 是否实现了所需方法
        methods = ["delete", "add_documents"]

        for method in methods:
            if not hasattr(destination, method):
                msg = (
                    f"Vectorstore {destination} does not have required method {method}"
                )
                raise ValueError(msg)

        if type(destination).delete == VectorStore.delete:
                        # 检查 VectorStore 是否覆盖了默认的 delete 方法
                        # 该方法的默认实现仅抛出 NotImplementedError
            msg = "Vectorstore has not implemented the delete method"
            raise ValueError(msg)
    elif isinstance(destination, DocumentIndex):
        pass
    else:
        msg = (  # type: ignore[unreachable]
            f"Vectorstore should be either a VectorStore or a DocumentIndex. "
            f"Got {type(destination)}."
        )
        raise TypeError(msg)

    if isinstance(docs_source, BaseLoader):
        try:
            doc_iterator = docs_source.lazy_load()
        except NotImplementedError:
            doc_iterator = iter(docs_source.load())
    else:
        doc_iterator = iter(docs_source)

    source_id_assigner = _get_source_id_assigner(source_id_key)

        # 标记更新开始的时间点。
    index_start_dt = record_manager.get_time()
    num_added = 0
    num_skipped = 0
    num_updated = 0
    num_deleted = 0
    scoped_full_cleanup_source_ids: set[str] = set()

    for doc_batch in _batch(batch_size, doc_iterator):
                # 在去重前记录原始批处理大小
        original_batch_size = len(doc_batch)

        hashed_docs = list(
            _deduplicate_in_order(
                [
                    _get_document_with_hash(doc, key_encoder=key_encoder)
                    for doc in doc_batch
                ]
            )
        )
                # 统计因批内去重而移除的文档数量
        num_skipped += original_batch_size - len(hashed_docs)

        source_ids: Sequence[str | None] = [
            source_id_assigner(hashed_doc) for hashed_doc in hashed_docs
        ]

        if cleanup in {"incremental", "scoped_full"}:
                        # 需要源 ID。
            for source_id, hashed_doc in zip(source_ids, hashed_docs, strict=False):
                if source_id is None:
                    msg = (
                        f"Source IDs are required when cleanup mode is "
                        f"incremental or scoped_full. "
                        f"Document that starts with "
                        f"content: {hashed_doc.page_content[:100]} "
                        f"was not assigned as source id."
                    )
                    raise ValueError(msg)
                if cleanup == "scoped_full":
                    scoped_full_cleanup_source_ids.add(source_id)
                        # 在上述 for 循环之后，源 ID 不可能为 None。
            source_ids = cast("Sequence[str]", source_ids)

        exists_batch = record_manager.exists(
            cast("Sequence[str]", [doc.id for doc in hashed_docs])
        )

                # 过滤掉已在记录存储中存在的文档。
        uids = []
        docs_to_index = []
        uids_to_refresh = []
        seen_docs: set[str] = set()
        for hashed_doc, doc_exists in zip(hashed_docs, exists_batch, strict=False):
            hashed_id = cast("str", hashed_doc.id)
            if doc_exists:
                if force_update:
                    seen_docs.add(hashed_id)
                else:
                    uids_to_refresh.append(hashed_id)
                    continue
            uids.append(hashed_id)
            docs_to_index.append(hashed_doc)

                # 更新刷新时间戳
        if uids_to_refresh:
            record_manager.update(uids_to_refresh, time_at_least=index_start_dt)
            num_skipped += len(uids_to_refresh)

                # 保持悲观，假设所有向量存储写入都会失败。
                # 首先写入向量存储
        if docs_to_index:
            if isinstance(destination, VectorStore):
                destination.add_documents(
                    docs_to_index,
                    ids=uids,
                    batch_size=batch_size,
                    **(upsert_kwargs or {}),
                )
            elif isinstance(destination, DocumentIndex):
                destination.upsert(
                    docs_to_index,
                    **(upsert_kwargs or {}),
                )

            num_added += len(docs_to_index) - len(seen_docs)
            num_updated += len(seen_docs)

                # 然后才更新记录存储。
                # 更新所有记录，即使它们已存在，因为我们希望刷新
                # 其时间戳。
        record_manager.update(
            cast("Sequence[str]", [doc.id for doc in hashed_docs]),
            group_ids=source_ids,
            time_at_least=index_start_dt,
        )

                # 如果提供了源 ID，我们可以增量式地执行删除！
        if cleanup == "incremental":
                        # 获取文档中未被加载器返回的 uids。
                        # mypy 无法确定源 ID 在此处不可能为 None
                        # 这是由于上方的检查，因此我们再次进行检查。
            for source_id in source_ids:
                if source_id is None:
                    msg = (
                        "source_id cannot be None at this point. "
                        "Reached unreachable code."
                    )
                    raise AssertionError(msg)

            source_ids_ = cast("Sequence[str]", source_ids)

            while uids_to_delete := record_manager.list_keys(
                group_ids=source_ids_, before=index_start_dt, limit=cleanup_batch_size
            ):
                                # 然后从向量存储中删除。
                _delete(destination, uids_to_delete)
                                # 首先从记录存储中删除。
                record_manager.delete_keys(uids_to_delete)
                num_deleted += len(uids_to_delete)

    if cleanup == "full" or (
        cleanup == "scoped_full" and scoped_full_cleanup_source_ids
    ):
        delete_group_ids: Sequence[str] | None = None
        if cleanup == "scoped_full":
            delete_group_ids = list(scoped_full_cleanup_source_ids)
        while uids_to_delete := record_manager.list_keys(
            group_ids=delete_group_ids, before=index_start_dt, limit=cleanup_batch_size
        ):
                        # 首先从记录存储中删除。
            _delete(destination, uids_to_delete)
                        # 然后从记录管理器中删除。
            record_manager.delete_keys(uids_to_delete)
            num_deleted += len(uids_to_delete)

    return {
        "num_added": num_added,
        "num_updated": num_updated,
        "num_skipped": num_skipped,
        "num_deleted": num_deleted,
    }


# 定义一个异步生成器函数
async def _to_async_iterator(iterator: Iterable[T]) -> AsyncIterator[T]:
    """将可迭代对象转换为异步迭代器。"""
    for item in iterator:
        yield item


async def _adelete(
    vector_store: VectorStore | DocumentIndex,
    ids: list[str],
) -> None:
    if isinstance(vector_store, VectorStore):
        delete_ok = await vector_store.adelete(ids)
        if delete_ok is not None and delete_ok is False:
            msg = "The delete operation to VectorStore failed."
            raise IndexingException(msg)
    elif isinstance(vector_store, DocumentIndex):
        delete_response = await vector_store.adelete(ids)
        if "num_failed" in delete_response and delete_response["num_failed"] > 0:
            msg = "The delete operation to DocumentIndex failed."
            raise IndexingException(msg)
    else:
        msg = (  # type: ignore[unreachable]
            f"Vectorstore should be either a VectorStore or a DocumentIndex. "
            f"Got {type(vector_store)}."
        )
        raise TypeError(msg)


async def aindex(
    docs_source: BaseLoader | Iterable[Document] | AsyncIterator[Document],
    record_manager: RecordManager,
    vector_store: VectorStore | DocumentIndex,
    *,
    batch_size: int = 100,
    cleanup: Literal["incremental", "full", "scoped_full"] | None = None,
    source_id_key: str | Callable[[Document], str] | None = None,
    cleanup_batch_size: int = 1_000,
    force_update: bool = False,
    key_encoder: Literal["sha1", "sha256", "sha512", "blake2b"]
    | Callable[[Document], str] = "sha1",
    upsert_kwargs: dict[str, Any] | None = None,
) -> IndexingResult:
    """异步从加载器索引数据到向量存储。

    索引功能使用管理器来跟踪哪些文档在向量存储中。

    这使我们能够追踪哪些文档被更新，哪些被删除，以及哪些应跳过。

    目前，文档通过其哈希值进行索引，用户无法指定文档的 uid。

    !!! warning "Behavior changed in `langchain-core` 0.3.25"

        增加了 `scoped_full` 清理模式。

    !!! warning

        * In full mode, the loader should be returning
            the entire dataset, and not just a subset of the dataset.
            Otherwise, the auto_cleanup will remove documents that it is not
            supposed to.
        * In incremental mode, if documents associated with a particular
            source id appear across different batches, the indexing API
            will do some redundant work. This will still result in the
            correct end state of the index, but will unfortunately not be
            100% efficient. For example, if a given document is split into 15
            chunks, and we index them using a batch size of 5, we'll have 3 batches
            all with the same source id. In general, to avoid doing too much
            redundant work select as big a batch size as possible.
        * The `scoped_full` mode is suitable if determining an appropriate batch size
            is challenging or if your data loader cannot return the entire dataset at
            once. This mode keeps track of source IDs in memory, which should be fine
            for most use cases. If your dataset is large (10M+ docs), you will likely
            need to parallelize the indexing process regardless.

    Args:
        docs_source: 要索引的文档加载器或文档可迭代对象。 record_manager:
        带时间戳的集合，用于跟踪哪些文档已被更新。 vector_store: 要索引文档的
        `VectorStore` 或 DocumentIndex。 batch_size: 索引时使用的批次大小。 cleanup:
        如何处理文档清理。

            - incremental: Cleans up all documents that haven't been updated AND
                that are associated with source IDs that were seen during indexing.
                Clean up is done continuously during indexing helping to minimize the
                probability of users seeing duplicated content.
            - full: Delete all documents that have not been returned by the loader
                during this run of indexing.
                Clean up runs after all documents have been indexed.
                This means that users may see duplicated content during indexing.
            - scoped_full: Similar to Full, but only deletes all documents
                that haven't been updated AND that are associated with
                source IDs that were seen during indexing.
            - None: Do not delete any documents.
        source_id_key: Optional key that helps identify the original source
            of the document.
        cleanup_batch_size: Batch size to use when cleaning up documents.
        force_update: Force update documents even if they are present in the
            record manager. Useful if you are re-indexing with updated embeddings.
        key_encoder: Hashing algorithm to use for hashing the document content and
            metadata. Options include "blake2b", "sha256", and "sha512".

            !!! version-added "Added in `langchain-core` 0.3.66"

        key_encoder: Hashing algorithm to use for hashing the document.
            If not provided, a default encoder using SHA-1 will be used.
            SHA-1 is not collision-resistant, and a motivated attacker
            could craft two different texts that hash to the
            same cache key.

            新应用应使用替代编码器之一，
            或提供自定义且强健的 key_encoder 函数，以规避此风险。

            When changing the key encoder, you must change the
            index as well to avoid duplicated documents in the cache.
        upsert_kwargs: Additional keyword arguments to pass to the add_documents
            method of the `VectorStore` or the upsert method of the DocumentIndex.
            For example, you can use this to specify a custom vector_field:
            upsert_kwargs={"vector_field": "embedding"}
            !!! version-added "Added in `langchain-core` 0.3.10"

    Returns:
        索引结果，包含添加、更新、删除或跳过的文档数量信息。

    Raises:
        ValueError: 如果 cleanup mode 不是 'incremental'、'full' 或 None 之一。
        ValueError: 如果 cleanup mode 为 incremental 且 source_id_key 为 None。
        ValueError: 如果 `VectorStore` 缺少所需的 "adelete" 和 "aadd_documents" 方法。
        ValueError: 如果 source_id_key 不为 None，但不是字符串或可调用对象。 TypeError:
        如果 `vector_store` 不是 `VectorStore` 或 DocumentIndex。 AssertionError: 如果
        cleanup mode 为 incremental 或 `scoped_full` 时，`source_id_key` 为
        None（不应到达此情况）。
    """
        # 该行为已弃用，但我们保留它以保证向后兼容性。
        # # 每个进程只警告一次。
    if key_encoder == "sha1":
        _warn_about_sha1()

    if cleanup not in {"incremental", "full", "scoped_full", None}:
        msg = (
            f"cleanup should be one of 'incremental', 'full', 'scoped_full' or None. "
            f"Got {cleanup}."
        )
        raise ValueError(msg)

    if (cleanup in {"incremental", "scoped_full"}) and source_id_key is None:
        msg = (
            "Source id key is required when cleanup mode is incremental or scoped_full."
        )
        raise ValueError(msg)

    destination = vector_store  # 内部重命名以提高清晰度

        # 如果是向量存储，让我们检查它是否具有所需的方法。
    if isinstance(destination, VectorStore):
                # 检查向量存储是否实现了所需的方法
                # 检查向量存储是否实现了所需的方法
        methods = ["adelete", "aadd_documents"]

        for method in methods:
            if not hasattr(destination, method):
                msg = (
                    f"Vectorstore {destination} does not have required method {method}"
                )
                raise ValueError(msg)

        if (
            type(destination).adelete == VectorStore.adelete
            and type(destination).delete == VectorStore.delete
        ):
                        # 检查向量存储是否重写了默认的 adelete 或 delete
                        # 方法实现，这些方法默认会抛出 NotImplementedError
            msg = "Vectorstore has not implemented the adelete or delete method"
            raise ValueError(msg)
    elif isinstance(destination, DocumentIndex):
        pass
    else:
        msg = (  # type: ignore[unreachable]
            f"Vectorstore should be either a VectorStore or a DocumentIndex. "
            f"Got {type(destination)}."
        )
        raise TypeError(msg)
    async_doc_iterator: AsyncIterator[Document]
    if isinstance(docs_source, BaseLoader):
        try:
            async_doc_iterator = docs_source.alazy_load()
        except NotImplementedError:
                        # 当 lazy_load 和 alazy_load 均未实现时触发异常。
                        # * 默认的 alazy_load 实现使用 lazy_load。
                        # * 默认的 lazy_load 实现抛出 NotImplementedError。
                        # 在这种情况下，我们使用 load 方法并将其转换为异步
                        # 迭代器。
            async_doc_iterator = _to_async_iterator(docs_source.load())
    elif hasattr(docs_source, "__aiter__"):
        async_doc_iterator = docs_source  # type: ignore[assignment]
    else:
        async_doc_iterator = _to_async_iterator(docs_source)

    source_id_assigner = _get_source_id_assigner(source_id_key)

        # 标记更新开始的时间点。
    index_start_dt = await record_manager.aget_time()
    num_added = 0
    num_skipped = 0
    num_updated = 0
    num_deleted = 0
    scoped_full_cleanup_source_ids: set[str] = set()

    async for doc_batch in _abatch(batch_size, async_doc_iterator):
                # 在去重前跟踪原始批处理大小
        original_batch_size = len(doc_batch)

        hashed_docs = list(
            _deduplicate_in_order(
                [
                    _get_document_with_hash(doc, key_encoder=key_encoder)
                    for doc in doc_batch
                ]
            )
        )
                # 统计因批内去重而移除的文档数量
        num_skipped += original_batch_size - len(hashed_docs)

        source_ids: Sequence[str | None] = [
            source_id_assigner(doc) for doc in hashed_docs
        ]

        if cleanup in {"incremental", "scoped_full"}:
                        # 如果清理模式是增量的，则必须提供源 ID。
            for source_id, hashed_doc in zip(source_ids, hashed_docs, strict=False):
                if source_id is None:
                    msg = (
                        f"Source IDs are required when cleanup mode is "
                        f"incremental or scoped_full. "
                        f"Document that starts with "
                        f"content: {hashed_doc.page_content[:100]} "
                        f"was not assigned as source id."
                    )
                    raise ValueError(msg)
                if cleanup == "scoped_full":
                    scoped_full_cleanup_source_ids.add(source_id)
                        # 在上述 for 循环之后，源 ID 不可能为 None。
            source_ids = cast("Sequence[str]", source_ids)

        exists_batch = await record_manager.aexists(
            cast("Sequence[str]", [doc.id for doc in hashed_docs])
        )

                # 过滤掉已在记录存储中的文档。
        uids: list[str] = []
        docs_to_index: list[Document] = []
        uids_to_refresh = []
        seen_docs: set[str] = set()
        for hashed_doc, doc_exists in zip(hashed_docs, exists_batch, strict=False):
            hashed_id = cast("str", hashed_doc.id)
            if doc_exists:
                if force_update:
                    seen_docs.add(hashed_id)
                else:
                    uids_to_refresh.append(hashed_id)
                    continue
            uids.append(hashed_id)
            docs_to_index.append(hashed_doc)

        if uids_to_refresh:
                        # 必须更新以刷新时间戳。
            await record_manager.aupdate(uids_to_refresh, time_at_least=index_start_dt)
            num_skipped += len(uids_to_refresh)

                # 保持悲观，假设所有向量存储写入都会失败。
                # 首先写入向量存储
        if docs_to_index:
            if isinstance(destination, VectorStore):
                await destination.aadd_documents(
                    docs_to_index,
                    ids=uids,
                    batch_size=batch_size,
                    **(upsert_kwargs or {}),
                )
            elif isinstance(destination, DocumentIndex):
                await destination.aupsert(
                    docs_to_index,
                    **(upsert_kwargs or {}),
                )
            num_added += len(docs_to_index) - len(seen_docs)
            num_updated += len(seen_docs)

                # 然后才更新记录存储。
                # 更新所有记录，即使它们已存在，因为我们希望刷新
                # 其时间戳。
        await record_manager.aupdate(
            cast("Sequence[str]", [doc.id for doc in hashed_docs]),
            group_ids=source_ids,
            time_at_least=index_start_dt,
        )

                # 如果提供了源 ID，我们可以增量式地执行删除！

        if cleanup == "incremental":
                        # 获取文档中未被加载器返回的 uids。

                        # mypy 无法确定源 ID 在此处不可能为 None
                        # 这是由于上方的检查，因此我们再次进行检查。
            for source_id in source_ids:
                if source_id is None:
                    msg = (
                        "source_id cannot be None at this point. "
                        "Reached unreachable code."
                    )
                    raise AssertionError(msg)

            source_ids_ = cast("Sequence[str]", source_ids)

            while uids_to_delete := await record_manager.alist_keys(
                group_ids=source_ids_, before=index_start_dt, limit=cleanup_batch_size
            ):
                                # 然后从向量存储中删除。
                await _adelete(destination, uids_to_delete)
                                # 首先从记录存储中删除。
                await record_manager.adelete_keys(uids_to_delete)
                num_deleted += len(uids_to_delete)

    if cleanup == "full" or (
        cleanup == "scoped_full" and scoped_full_cleanup_source_ids
    ):
        delete_group_ids: Sequence[str] | None = None
        if cleanup == "scoped_full":
            delete_group_ids = list(scoped_full_cleanup_source_ids)
        while uids_to_delete := await record_manager.alist_keys(
            group_ids=delete_group_ids, before=index_start_dt, limit=cleanup_batch_size
        ):
                        # 首先从记录存储中删除。
            await _adelete(destination, uids_to_delete)
                        # 然后从记录管理器中删除。
            await record_manager.adelete_keys(uids_to_delete)
            num_deleted += len(uids_to_delete)

    return {
        "num_added": num_added,
        "num_updated": num_updated,
        "num_skipped": num_skipped,
        "num_deleted": num_deleted,
    }

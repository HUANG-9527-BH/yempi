"""Blob 和 Blob 加载器的模式。

The goal is to facilitate decoupling of content loading from content parsing code. In
addition, content loading code should provide a lazy loading interface by default.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

# 为向后兼容性重新导出 Blob 和 PathLike
from langchain_core.documents.base import Blob, PathLike

if TYPE_CHECKING:
    from collections.abc import Iterator


class BlobLoader(ABC):
    """Blob 加载器实现的抽象接口。

    Implementer should be able to load raw content from a storage system according to
    some criteria and return the raw content lazily as a stream of blobs.
    """

    @abstractmethod
    def yield_blobs(
        self,
    ) -> Iterator[Blob]:
        """用于表示 LangChain 的 `Blob` 对象所代表的原始数据的懒加载器。

        Yields:
            `Blob` 对象。
        """


# 为向后兼容性重新导出 Blob 和 Pathlike
__all__ = ["Blob", "BlobLoader", "PathLike"]

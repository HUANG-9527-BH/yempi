"""文档加载器实现的抽象接口。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from langchain_core.runnables import run_in_executor

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Iterator

    from langchain_text_splitters import TextSplitter

    from langchain_core.documents import Document
    from langchain_core.documents.base import Blob

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    _HAS_TEXT_SPLITTERS = True
except ImportError:
    _HAS_TEXT_SPLITTERS = False


class BaseLoader(ABC):  # noqa: B024
    """文档加载器的接口。

    实现应使用生成器来实现懒加载，以避免一次性将所有文档加载到内存中。

    `load` 仅用于用户便利，不应被重写。
    """

        # 子类不应直接实现此方法。相反，它们应实现懒加载方法。
        # 子类不应直接实现此方法。相反，它们应实现懒加载方法。
    def load(self) -> list[Document]:
        """将数据加载为 `Document` 对象。

        Returns:
            文档。
        """
        return list(self.lazy_load())

    async def aload(self) -> list[Document]:
        """将数据加载为 `Document` 对象。

        Returns:
            文档。
        """
        return [document async for document in self.alazy_load()]

    def load_and_split(
        self, text_splitter: TextSplitter | None = None
    ) -> list[Document]:
        """加载 `Document` 并将其分割为块。块以 `Document` 形式返回。

        !!! danger

            不要重写此方法。应视为已弃用！

        Args:
            text_splitter: 用于分割文档的 `TextSplitter` 实例。

                默认为 `RecursiveCharacterTextSplitter`。

        Raises:
            ImportError: 如果未安装 `langchain-text-splitters` 且未提供
                `text_splitter`。

        Returns:
            `Document` 对象列表。
        """
        if text_splitter is None:
            if not _HAS_TEXT_SPLITTERS:
                msg = (
                    "Unable to import from langchain_text_splitters. Please specify "
                    "text_splitter or install langchain_text_splitters with "
                    "`pip install -U langchain-text-splitters`."
                )
                raise ImportError(msg)

            text_splitter_: TextSplitter = RecursiveCharacterTextSplitter()
        else:
            text_splitter_ = text_splitter
        docs = self.load()
        return text_splitter_.split_documents(docs)

        # 注意：一旦所有现有子类都实现该方法，此方法将升级为抽象方法。
        #            注意：一旦所有现有子类都实现该方法，此方法将升级为抽象方法。
    def lazy_load(self) -> Iterator[Document]:
        """`Document` 的懒加载器。

        Yields:
            `Document` 对象。
        """
        if type(self).load != BaseLoader.load:
            return iter(self.load())
        msg = f"{self.__class__.__name__} does not implement lazy_load()"
        raise NotImplementedError(msg)

    async def alazy_load(self) -> AsyncIterator[Document]:
        """`Document` 的懒加载器。

        Yields:
            `Document` 对象。
        """
        iterator = await run_in_executor(None, self.lazy_load)
        done = object()
        while True:
            doc = await run_in_executor(None, next, iterator, done)
            if doc is done:
                break
            yield doc  # type: ignore[misc]


class BaseBlobParser(ABC):
    """Blob 解析器的抽象接口。

    Blob 解析器提供一种将存储在 blob 中的原始数据解析为一个或多个
    `Document` 对象的方法。

    The parser can be composed with blob loaders, making it easy to reuse a parser
    independent of how the blob was originally loaded.
    """

    @abstractmethod
    def lazy_parse(self, blob: Blob) -> Iterator[Document]:
        """懒解析接口。

        子类必须实现此方法。

        Args:
            blob: `Blob` 实例

        Returns:
            `Document` 对象的生成器
        """

    def parse(self, blob: Blob) -> list[Document]:
        """将 blob 非惰性解析为 `Document` 或 `Document` 对象列表。

        此方法适用于交互式开发环境中的便利操作。

        生产应用应优先使用 `lazy_parse` 方法。

        子类通常不应重写此 `parse` 方法。

        Args:
            blob: `Blob` 实例

        Returns:
            `Document` 对象列表
        """
        return list(self.lazy_parse(blob))

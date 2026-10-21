"""媒体和文档的基础类。

本模块包含 **数据检索与处理工作流** 的核心抽象：

- `BaseMedia`: 提供 `id` 和 `metadata` 字段的基础类
- `Blob`: 原始数据加载（文件、二进制数据）——由文档加载器使用
- `Document`: 用于检索的文本内容（RAG、向量存储、语义搜索）

!!! note "不适用于 LLM 聊天消息"

    These classes are for data processing pipelines, not LLM I/O. For multimodal
    content in chat messages (images, audio in conversations), see
    `langchain.messages` content blocks instead.
"""

from __future__ import annotations

import contextlib
import mimetypes
from io import BufferedReader, BytesIO
from pathlib import Path, PurePath
from typing import TYPE_CHECKING, Any, Literal, cast

from pydantic import ConfigDict, Field, model_validator

from langchain_core.load.serializable import Serializable

if TYPE_CHECKING:
    from collections.abc import Generator

PathLike = str | PurePath


class BaseMedia(Serializable):
    """用于检索和数据处理工作流中的内容的基础类。

    提供需要存储、索引或搜索的内容的通用字段。

    !!! note

        对于 **聊天消息** 中的多模态内容（发送至或来自 LLM 的图像、音频），
        请改用 `langchain.messages` 中的内容块。
    """

        # ID 字段目前是可选的。
        # 它可能在未来的重大版本中变为必需，
        # 在足够多的 VectorStore 实现采用之后。
    id: str | None = Field(default=None, coerce_numbers_to_str=True)
    """文档的可选标识符。

    理想情况下，该标识符应在文档集合中唯一，并格式化为 UUID，但不会强制执行。
    """

    metadata: dict[Any, Any] = Field(default_factory=dict)
    """与内容关联的任意元数据。"""


class Blob(BaseMedia):
    """用于文档加载和文件处理的原始数据抽象。

    Represents raw bytes or text, either in-memory or by file reference. Used
    primarily by document loaders to decouple data loading from parsing.

    受 [Mozilla 的 `Blob`](https://developer.mozilla.org/en-US/docs/Web/API/Blob) 启发

    ???+ example "从内存数据初始化一个 blob"

        ```python
        from langchain_core.documents import Blob

        blob = Blob.from_data("Hello, world!")

        # Read the blob as a string
        print(blob.as_string())

        # Read the blob as bytes
        print(blob.as_bytes())

        # Read the blob as a byte stream
        with blob.as_bytes_io() as f:
            print(f.read())
        ```

    ??? example "从内存加载并指定 MIME 类型和元数据"

        ```python
        from langchain_core.documents import Blob

        blob = Blob.from_data(
            data="Hello, world!",
            mime_type="text/plain",
            metadata={"source": "https://example.com"},
        )
        ```

    ??? example "从文件加载 blob"

        ```python
        from langchain_core.documents import Blob

        blob = Blob.from_path("path/to/file.txt")

        # Read the blob as a string
        print(blob.as_string())

        # Read the blob as bytes
        print(blob.as_bytes())

        # Read the blob as a byte stream
        with blob.as_bytes_io() as f:
            print(f.read())
        ```
    """

    data: bytes | str | None = None
    """与 `Blob` 关联的原始数据。"""

    mimetype: str | None = None
    """MIME 类型，不要与文件扩展名混淆。"""

    encoding: str = "utf-8"
    """如果将字节解码为字符串，使用的编码方式。

    解码为字符串时，默认使用 `utf-8` 编码。
    """

    path: PathLike | None = None
    """原始内容的来源位置。"""

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        frozen=True,
    )

    @property
    def source(self) -> str | None:
        """如果已知，为 blob 的源位置字符串；否则为 none。

        如果与 `Blob` 关联了路径，则默认使用路径位置。

        除非通过名为 `'source'` 的元数据字段显式设置，此时将使用该值。
        """
        if self.metadata and "source" in self.metadata:
            return cast("str | None", self.metadata["source"])
        return str(self.path) if self.path else None

    @model_validator(mode="before")
    @classmethod
    def check_blob_is_valid(cls, values: dict[str, Any]) -> Any:
        """验证是否提供了 data 或 path。"""
        if "data" not in values and "path" not in values:
            msg = "Either data or path must be provided"
            raise ValueError(msg)
        return values

    def as_string(self) -> str:
        """将数据读取为字符串。

        Raises:
            ValueError: 如果 blob 无法表示为字符串。

        Returns:
            作为字符串的数据。
        """
        if self.data is None and self.path:
            return Path(self.path).read_text(encoding=self.encoding)
        if isinstance(self.data, bytes):
            return self.data.decode(self.encoding)
        if isinstance(self.data, str):
            return self.data
        msg = f"Unable to get string for blob {self}"
        raise ValueError(msg)

    def as_bytes(self) -> bytes:
        """将数据读取为字节。

        Raises:
            ValueError: 如果 blob 无法表示为字节。

        Returns:
            作为字节的数据。
        """
        if isinstance(self.data, bytes):
            return self.data
        if isinstance(self.data, str):
            return self.data.encode(self.encoding)
        if self.data is None and self.path:
            return Path(self.path).read_bytes()
        msg = f"Unable to get bytes for blob {self}"
        raise ValueError(msg)

    @contextlib.contextmanager
    def as_bytes_io(self) -> Generator[BytesIO | BufferedReader, None, None]:
        """将数据读取为字节流。

        Raises:
            NotImplementedError: 如果 blob 无法表示为字节流。

        Yields:
            作为字节流的数据。
        """
        if isinstance(self.data, bytes):
            yield BytesIO(self.data)
        elif self.data is None and self.path:
            with Path(self.path).open("rb") as f:
                yield f
        else:
            msg = f"Unable to convert blob {self}"
            raise NotImplementedError(msg)

    @classmethod
    def from_path(
        cls,
        path: PathLike,
        *,
        encoding: str = "utf-8",
        mime_type: str | None = None,
        guess_type: bool = True,
        metadata: dict[Any, Any] | None = None,
    ) -> Blob:
        """从路径对象加载 blob。

        Args:
            path: Path-like object to file to be read
            encoding: Encoding to use if decoding the bytes into a string
            mime_type: If provided, will be set as the MIME type of the data
            guess_type: If `True`, the MIME type will be guessed from the file
                extension, if a MIME type was not provided
            metadata: Metadata to associate with the `Blob`

        Returns:
            `Blob` 实例
        """
        if mime_type is None and guess_type:
            mimetype = mimetypes.guess_type(path)[0]
        else:
            mimetype = mime_type
                # 我们不会立即加载数据，而是将 blob 视为对底层数据的引用。
                # my-py 报告 page_content 在基类中未定义。
        return cls(
            data=None,
            mimetype=mimetype,
            encoding=encoding,
            path=path,
            metadata=metadata if metadata is not None else {},
        )

    @classmethod
    def from_data(
        cls,
        data: str | bytes,
        *,
        encoding: str = "utf-8",
        mime_type: str | None = None,
        path: str | None = None,
        metadata: dict[Any, Any] | None = None,
    ) -> Blob:
        """从内存中的数据初始化 `Blob`。

        Args:
            data: 与 `Blob` 关联的内存中数据
            encoding: 如果将字节解码为字符串，使用的编码格式
            mime_type: 如果提供，将设置为数据的 MIME 类型
            path: 如果提供，将设置为数据来源的路径
            metadata: 与 `Blob` 关联的元数据

        Returns:
            `Blob` 实例
        """
        return cls(
            data=data,
            mimetype=mime_type,
            encoding=encoding,
            path=path,
            metadata=metadata if metadata is not None else {},
        )

    def __repr__(self) -> str:
        """返回 blob 的表示形式。"""
        str_repr = f"Blob {id(self)}"
        if self.source:
            str_repr += f" {self.source}"
        return str_repr


class Document(BaseMedia):
    """用于存储一段文本及其关联元数据的类。

    !!! note

        `Document` is for **retrieval workflows**, not chat I/O. For sending text
        to an LLM in a conversation, use message types from `langchain.messages`.

    Example:
        ```python
        from langchain_core.documents import Document

        document = Document(
            page_content="Hello, world!", metadata={"source": "https://example.com"}
        )
        ```
    """

    page_content: str
    """字符串文本。"""

    type: Literal["Document"] = "Document"

    def __init__(self, page_content: str, **kwargs: Any) -> None:
        """将 page_content 作为位置参数或命名参数传入。"""
                # 此处，我们依赖 pydantic 基类来处理验证。
                # 格式符合 pydantic 对 `__str__` 的格式要求。
        super().__init__(page_content=page_content, **kwargs)  # type: ignore[call-arg,unused-ignore]

    @classmethod
    def is_lc_serializable(cls) -> bool:
        """返回 `True`，因为此类是可序列化的。"""
        return True

    @classmethod
    def get_lc_namespace(cls) -> list[str]:
        """获取 LangChain 对象的命名空间。

        Returns:
            `["langchain", "schema", "document"]`
        """
        return ["langchain", "schema", "document"]

    def __str__(self) -> str:
        """重写 `__str__` 以限制其仅包含 page_content 和 metadata。

        Returns:
            `Document` 的字符串表示形式。
        """
                # 此更改的目的是确保用户代码在直接将 Document 对象输入提示词时保持不变，
        #
                # 即使未来添加了 id 字段（或其他字段）。
                # 此重写可能会在未来被移除，取而代之的是更通用的解决方案，即在提示词中直接格式化内容。
                # id 字段（或未来其他字段）的
        #
                # 此覆盖项未来可能会被移除，以改用更通用的方案
                # 直接在提示词中格式化内容的解决方案。
        if self.metadata:
            return f"page_content='{self.page_content}' metadata={self.metadata}"
        return f"page_content='{self.page_content}'"

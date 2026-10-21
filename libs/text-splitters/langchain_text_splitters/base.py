"""文本分割器基类接口。"""

from __future__ import annotations

import copy
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    TypeVar,
    cast,
)

from langchain_core.documents import BaseDocumentTransformer, Document
from typing_extensions import Self, override

if TYPE_CHECKING:
    from collections.abc import Callable, Collection, Iterable, Sequence
    from collections.abc import Set as AbstractSet

    from transformers.tokenization_utils_base import PreTrainedTokenizerBase

logger = logging.getLogger(__name__)

TS = TypeVar("TS", bound="TextSplitter")


def _import_tiktoken() -> object:
    try:
        import tiktoken  # noqa: PLC0415
    except ImportError as err:
        msg = (
            "Could not import tiktoken python package. "
            "This is needed in order to calculate max_tokens_for_prompt. "
            "Please install it with `pip install tiktoken`."
        )
        raise ImportError(msg) from err
    return tiktoken


def _import_pretrained_tokenizer_base() -> type[PreTrainedTokenizerBase]:
    try:
        from transformers.tokenization_utils_base import (  # noqa: PLC0415
            PreTrainedTokenizerBase,
        )
    except ImportError as err:
        msg = (
            "Could not import transformers python package. "
            "Please install it with `pip install transformers`."
        )
        raise ValueError(msg) from err
    return PreTrainedTokenizerBase


class TextSplitter(BaseDocumentTransformer, ABC):
    """将文本分割为块的接口。"""

    def __init__(
        self,
        chunk_size: int = 4000,
        chunk_overlap: int = 200,
        length_function: Callable[[str], int] = len,
        keep_separator: bool | Literal["start", "end"] = False,  # noqa: FBT001,FBT002
        add_start_index: bool = False,  # noqa: FBT001,FBT002
        strip_whitespace: bool = True,  # noqa: FBT001,FBT002
    ) -> None:
        """创建一个新的 `TextSplitter`。

        Args:
            chunk_size: Maximum size of chunks to return
            chunk_overlap: Overlap in characters between chunks
            length_function: Function that measures the length of given chunks
            keep_separator: Whether to keep the separator and where to place it
                in each corresponding chunk `(True='start')`
            add_start_index: If `True`, includes chunk's start index in metadata
            strip_whitespace: If `True`, strips whitespace from the start and end of
                every document

        Raises:
            ValueError: 如果 `chunk_size` 小于或等于 0
            ValueError: 如果 `chunk_overlap` 小于 0
            ValueError: 如果 `chunk_overlap` 大于 `chunk_size`
        """
        if chunk_size <= 0:
            msg = f"chunk_size must be > 0, got {chunk_size}"
            raise ValueError(msg)
        if chunk_overlap < 0:
            msg = f"chunk_overlap must be >= 0, got {chunk_overlap}"
            raise ValueError(msg)
        if chunk_overlap > chunk_size:
            msg = (
                f"Got a larger chunk overlap ({chunk_overlap}) than chunk size "
                f"({chunk_size}), should be smaller."
            )
            raise ValueError(msg)
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._length_function = length_function
        self._keep_separator = keep_separator
        self._add_start_index = add_start_index
        self._strip_whitespace = strip_whitespace

    @abstractmethod
    def split_text(self, text: str) -> list[str]:
        """将文本分割为多个组件。

        Args:
            text: 待分割的文本。

        Returns:
            块列表。
        """

    def create_documents(
        self, texts: list[str], metadatas: list[dict[Any, Any]] | None = None
    ) -> list[Document]:
        """从文本列表创建 `Document` 对象列表。

        Args:
            texts: 待分割并转换为文档的文本列表。
            metadatas: 可选的元数据列表，用于关联每个文档。

        Returns:
            `Document` 对象列表。
        """
        metadatas_ = metadatas or [{}] * len(texts)
        documents = []
        for i, text in enumerate(texts):
            index = 0
            previous_chunk_len = 0
            for chunk in self.split_text(text):
                metadata = copy.deepcopy(metadatas_[i])
                if self._add_start_index:
                    offset = index + previous_chunk_len - self._chunk_overlap
                    index = text.find(chunk, max(0, offset))
                    metadata["start_index"] = index
                    previous_chunk_len = len(chunk)
                new_doc = Document(page_content=chunk, metadata=metadata)
                documents.append(new_doc)
        return documents

    def split_documents(self, documents: Iterable[Document]) -> list[Document]:
        """分割文档。

        Args:
            documents: 待分割的文档。

        Returns:
            分割后的文档列表。
        """
        texts, metadatas = [], []
        for doc in documents:
            texts.append(doc.page_content)
            metadatas.append(doc.metadata)
        return self.create_documents(texts, metadatas=metadatas)

    def _join_docs(self, docs: list[str], separator: str) -> str | None:
        text = separator.join(docs)
        if self._strip_whitespace:
            text = text.strip()
        return text or None

    def _merge_splits(self, splits: Iterable[str], separator: str) -> list[str]:
                # 我们现在希望将这些小块合并成中等大小
                # 的块以发送给 LLM。
        separator_len = self._length_function(separator)

        docs = []
        current_doc: list[str] = []
        total = 0
        for d in splits:
            len_ = self._length_function(d)
            if (
                total + len_ + (separator_len if len(current_doc) > 0 else 0)
                > self._chunk_size
            ):
                if total > self._chunk_size:
                    logger.warning(
                        "Created a chunk of size %d, which is longer than the "
                        "specified %d",
                        total,
                        self._chunk_size,
                    )
                if len(current_doc) > 0:
                    doc = self._join_docs(current_doc, separator)
                    if doc is not None:
                        docs.append(doc)
                                        # 持续弹出，当满足以下条件时：
                                        # - 我们的块比块重叠还大
                                        # - 或者我们仍有块且长度仍然很长
                    while total > self._chunk_overlap or (
                        total + len_ + (separator_len if len(current_doc) > 0 else 0)
                        > self._chunk_size
                        and total > 0
                    ):
                        total -= self._length_function(current_doc[0]) + (
                            separator_len if len(current_doc) > 1 else 0
                        )
                        current_doc = current_doc[1:]
            current_doc.append(d)
            total += len_ + (separator_len if len(current_doc) > 1 else 0)
        doc = self._join_docs(current_doc, separator)
        if doc is not None:
            docs.append(doc)
        return docs

    @classmethod
    def from_huggingface_tokenizer(
        cls, tokenizer: PreTrainedTokenizerBase, **kwargs: Any
    ) -> TextSplitter:
        """使用 Hugging Face 分词器计算长度的文本分割器。

        Args:
            tokenizer: 要使用的 Hugging Face 分词器。

        Returns:
            An instance of `TextSplitter` using the Hugging Face tokenizer for length
                calculation.
        """
        pretrained_tokenizer_base = _import_pretrained_tokenizer_base()

        if not isinstance(tokenizer, pretrained_tokenizer_base):
            msg = "Tokenizer received was not an instance of PreTrainedTokenizerBase"
            raise ValueError(msg)  # noqa: TRY004

        def _huggingface_tokenizer_length(text: str) -> int:
            return len(tokenizer.tokenize(text))

        return cls(length_function=_huggingface_tokenizer_length, **kwargs)

    @staticmethod
    def _tiktoken_length_function(
        encoding_name: str = "gpt2",
        model_name: str | None = None,
        allowed_special: Literal["all"] | AbstractSet[str] | None = None,
        disallowed_special: Literal["all"] | Collection[str] = "all",
    ) -> Callable[[str], int]:
        """构建基于 `tiktoken` 的长度计算函数。

        在 `TextSplitter` 和 `TokenTextSplitter` 上由 `from_tiktoken_encoder` 共享。

        Args:
            encoding_name: 要使用的 tiktoken 编码名称。
            model_name: 要使用的模型名称。

                如果提供，则会覆盖 `encoding_name`。
            allowed_special: 编码期间允许的特殊 token。
            disallowed_special: 编码期间不允许的特殊 token。

        Returns:
            返回字符串 token 长度的函数。

        Raises:
            ImportError: 如果未安装 tiktoken 包。
        """
        if allowed_special is None:
            allowed_special = set[str]()
        tiktoken = cast("Any", _import_tiktoken())

        if model_name is not None:
            enc = tiktoken.encoding_for_model(model_name)
        else:
            enc = tiktoken.get_encoding(encoding_name)

        def _tiktoken_encoder(text: str) -> int:
            return len(
                enc.encode(
                    text,
                    allowed_special=allowed_special,
                    disallowed_special=disallowed_special,
                )
            )

        return _tiktoken_encoder

    @classmethod
    def from_tiktoken_encoder(
        cls,
        encoding_name: str = "gpt2",
        model_name: str | None = None,
        allowed_special: Literal["all"] | AbstractSet[str] | None = None,
        disallowed_special: Literal["all"] | Collection[str] = "all",
        **kwargs: Any,
    ) -> Self:
        """使用 `tiktoken` 编码器计算长度的文本分割器。

        Args:
            encoding_name: 用于的 tiktoken 编码名称。
            model_name: 使用的模型名称。
                如果提供，将覆盖 `encoding_name`。
            allowed_special: 编码过程中允许的特殊标记。
            disallowed_special: 编码过程中禁止的特殊标记。

        Returns:
            使用 tiktoken 进行长度计算的调用类实例。

        Raises:
            ImportError: 如果未安装 tiktoken 包。
        """
        length_function = cls._tiktoken_length_function(
            encoding_name, model_name, allowed_special, disallowed_special
        )
        return cls(length_function=length_function, **kwargs)

    @override
    def transform_documents(
        self, documents: Sequence[Document], **kwargs: Any
    ) -> Sequence[Document]:
        """通过分割文档序列来转换文档序列。

        Args:
            documents: 需要分割的文档序列。

        Returns:
            分割后的文档列表。
        """
        return self.split_documents(list(documents))


class TokenTextSplitter(TextSplitter):
    """使用模型分词器将文本分割为 token。"""

    def __init__(
        self,
        encoding_name: str = "gpt2",
        model_name: str | None = None,
        allowed_special: Literal["all"] | AbstractSet[str] | None = None,
        disallowed_special: Literal["all"] | Collection[str] = "all",
        **kwargs: Any,
    ) -> None:
        """创建一个新的 `TokenTextSplitter`。

        Args:
            encoding_name: 用于的 tiktoken 编码名称。
            model_name: 使用的模型名称。
                如果提供，将覆盖 `encoding_name`。
            allowed_special: 编码过程中允许的特殊标记。
            disallowed_special: 编码过程中禁止的特殊标记。

        Raises:
            ImportError: 如果未安装 tiktoken 包。
        """
        if allowed_special is None:
            allowed_special = set[str]()
        super().__init__(**kwargs)
        try:
            tiktoken = cast("Any", _import_tiktoken())
        except ImportError as err:
            msg = (
                "Could not import tiktoken python package. "
                "This is needed in order to for TokenTextSplitter. "
                "Please install it with `pip install tiktoken`."
            )
            raise ImportError(msg) from err

        if model_name is not None:
            enc = tiktoken.encoding_for_model(model_name)
        else:
            enc = tiktoken.get_encoding(encoding_name)
        self._tokenizer = enc
        self._allowed_special = allowed_special
        self._disallowed_special = disallowed_special

    @classmethod
    @override
    def from_tiktoken_encoder(
        cls,
        encoding_name: str = "gpt2",
        model_name: str | None = None,
        allowed_special: Literal["all"] | AbstractSet[str] | None = None,
        disallowed_special: Literal["all"] | Collection[str] = "all",
        **kwargs: Any,
    ) -> Self:
        """使用 `tiktoken` 编码器进行长度计算的文本分割器。

        Unlike the base implementation, this also seeds the constructor with the
        tiktoken configuration so the splitter tokenizes on the same encoding.

        Args:
            encoding_name: 用于的 tiktoken 编码名称。
            model_name: 使用的模型名称。

                如果提供，将覆盖 `encoding_name`。
            allowed_special: 编码过程中允许的特殊标记。
            disallowed_special: 编码过程中禁止的特殊标记。

        Returns:
            使用 tiktoken 进行长度计算的 `TokenTextSplitter` 实例。

        Raises:
            ImportError: 如果未安装 tiktoken 包。
        """
        length_function = cls._tiktoken_length_function(
            encoding_name, model_name, allowed_special, disallowed_special
        )
        return cls(
            length_function=length_function,
            encoding_name=encoding_name,
            model_name=model_name,
            allowed_special=allowed_special,
            disallowed_special=disallowed_special,
            **kwargs,
        )

    @override
    def split_text(self, text: str) -> list[str]:
        """根据分词结果将输入文本拆分为较小的块。

        This method uses a custom tokenizer configuration to encode the input text
        into tokens, processes the tokens in chunks of a specified size with overlap,
        and decodes them back into text chunks. The splitting is performed using the
        `split_text_on_tokens` function.

        Args:
            text: 需要拆分为较小块的输入文本。

        Returns:
            A list of text chunks, where each chunk is derived from a portion
                of the input text based on the tokenization and chunking rules.
        """

        def _encode(_text: str) -> list[int]:
                        # `tiktoken` 是延迟导入的，因此 mypy 无法推断编码器返回类型。
            return cast(
                "list[int]",
                self._tokenizer.encode(
                    _text,
                    allowed_special=self._allowed_special,
                    disallowed_special=self._disallowed_special,
                ),
            )

        tokenizer = Tokenizer(
            chunk_overlap=self._chunk_overlap,
            tokens_per_chunk=self._chunk_size,
            decode=self._tokenizer.decode,
            encode=_encode,
        )

        return split_text_on_tokens(text=text, tokenizer=tokenizer)


class Language(str, Enum):
    """编程语言枚举。"""

    CPP = "cpp"
    GO = "go"
    JAVA = "java"
    KOTLIN = "kotlin"
    JS = "js"
    TS = "ts"
    PHP = "php"
    PROTO = "proto"
    PYTHON = "python"
    R = "r"
    RST = "rst"
    RUBY = "ruby"
    RUST = "rust"
    SCALA = "scala"
    SWIFT = "swift"
    MARKDOWN = "markdown"
    LATEX = "latex"
    HTML = "html"
    SOL = "sol"
    CSHARP = "csharp"
    COBOL = "cobol"
    C = "c"
    LUA = "lua"
    PERL = "perl"
    HASKELL = "haskell"
    ELIXIR = "elixir"
    POWERSHELL = "powershell"
    VISUALBASIC6 = "visualbasic6"


@dataclass(frozen=True)
class Tokenizer:
    """分词器数据类。"""

    chunk_overlap: int
    """块之间的 token 重叠量"""

    tokens_per_chunk: int
    """每块的最大 token 数量"""

    decode: Callable[[list[int]], str]
    """ 将 token ID 列表解码为字符串的函数"""

    encode: Callable[[str], list[int]]
    """ 将字符串编码为 token ID 列表的函数"""


def split_text_on_tokens(*, text: str, tokenizer: Tokenizer) -> list[str]:
    """使用分词器分割输入文本并返回块。

    Args:
        text: 待分割的输入文本。
        tokenizer: 用于分割的分词器。

    Returns:
        块列表。
    """
    splits: list[str] = []
    input_ids = tokenizer.encode(text)
    start_idx = 0
    if tokenizer.tokens_per_chunk <= tokenizer.chunk_overlap:
        msg = "tokens_per_chunk must be greater than chunk_overlap"
        raise ValueError(msg)

    while start_idx < len(input_ids):
        cur_idx = min(start_idx + tokenizer.tokens_per_chunk, len(input_ids))
        chunk_ids = input_ids[start_idx:cur_idx]
        if not chunk_ids:
            break
        decoded = tokenizer.decode(chunk_ids)
        if decoded:
            splits.append(decoded)
        if cur_idx == len(input_ids):
            break
        start_idx += tokenizer.tokens_per_chunk - tokenizer.chunk_overlap
    return splits

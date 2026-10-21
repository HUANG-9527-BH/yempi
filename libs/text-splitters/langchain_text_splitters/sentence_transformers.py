"""句子嵌入模型文本分割器。"""

from __future__ import annotations

from importlib import import_module
from typing import Any, cast

from typing_extensions import override

from langchain_text_splitters.base import TextSplitter, Tokenizer, split_text_on_tokens


class SentenceTransformersTokenTextSplitter(TextSplitter):
    """使用句子嵌入模型分词器将文本拆分为 token。"""

    def __init__(
        self,
        chunk_overlap: int = 50,
        model_name: str = "sentence-transformers/all-mpnet-base-v2",
        tokens_per_chunk: int | None = None,
        model_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """创建一个新的 `TextSplitter`。

        Args:
            chunk_overlap: 块之间的重叠 token 数量。
            model_name: 要使用的句子嵌入模型名称。
            tokens_per_chunk: 每个块的 token 数量。

                如果为 `None`，则使用模型允许的最大 token 数量。
            model_kwargs: 模型初始化的额外参数。
                可使用 sentence_transformers.SentenceTransformer 的参数。

        Raises:
            ImportError: 如果未安装 `sentence_transformers` 包。
            ValueError: 如果 `tokens_per_chunk` 超过模型的最大 token 限制。
        """
        super().__init__(**kwargs, chunk_overlap=chunk_overlap)

        try:
            sentence_transformers = cast("Any", import_module("sentence_transformers"))
            sentence_transformer_cls = sentence_transformers.SentenceTransformer
        except ImportError as err:
            msg = (
                "Could not import sentence_transformers python package. "
                "This is needed in order to use SentenceTransformersTokenTextSplitter. "
                "Please install it with `pip install sentence-transformers`."
            )
            raise ImportError(msg) from err

        self.model_name = model_name
        self._model = sentence_transformer_cls(self.model_name, **(model_kwargs or {}))
        self.tokenizer = self._model.tokenizer
        self._initialize_chunk_configuration(tokens_per_chunk=tokens_per_chunk)

    def _initialize_chunk_configuration(self, *, tokens_per_chunk: int | None) -> None:
        self.maximum_tokens_per_chunk = self._model.max_seq_length

        if tokens_per_chunk is None:
            if self.maximum_tokens_per_chunk is None:
                msg = (
                    "The model does not have a maximum token limit, "
                    "and tokens_per_chunk was not provided. "
                    "Please provide a value for tokens_per_chunk."
                )
                raise ValueError(msg)
            self.tokens_per_chunk = self.maximum_tokens_per_chunk
        else:
            self.tokens_per_chunk = tokens_per_chunk

        if (
            self.maximum_tokens_per_chunk is not None
            and self.tokens_per_chunk > self.maximum_tokens_per_chunk
        ):
            msg = (
                f"The token limit of the models '{self.model_name}'"
                f" is: {self.maximum_tokens_per_chunk}."
                f" Argument tokens_per_chunk={self.tokens_per_chunk}"
                f" > maximum token limit."
            )
            raise ValueError(msg)

    @override
    def split_text(self, text: str) -> list[str]:
        """通过在 token 上分割文本，将输入文本拆分为较小的组件。

        This method encodes the input text using a private `_encode` method, then
        strips the start and stop token IDs from the encoded result. It returns the
        processed segments as a list of strings.

        Args:
            text: 需要拆分的输入文本。

        Returns:
            A list of string components derived from the input text after encoding and
                processing.
        """

        def encode_strip_start_and_stop_token_ids(text: str) -> list[int]:
            return self._encode(text)[1:-1]

        tokenizer = Tokenizer(
            chunk_overlap=self._chunk_overlap,
            tokens_per_chunk=self.tokens_per_chunk,
            decode=self.tokenizer.decode,
            encode=encode_strip_start_and_stop_token_ids,
        )

        return split_text_on_tokens(text=text, tokenizer=tokenizer)

    def count_tokens(self, *, text: str) -> int:
        """计算给定文本中的 token 数量。

        此方法使用私有 `_encode` 方法对输入文本进行编码，并
        计算编码结果中的总 token 数量。

        Args:
            text: 需要计算 token 数量的输入文本。

        Returns:
            编码文本中的 token 数量。
        """
        return len(self._encode(text))

    _max_length_equal_32_bit_integer: int = 2**32

    def _encode(self, text: str) -> list[int]:
        token_ids_with_start_and_end_token_ids = self.tokenizer.encode(
            text,
            max_length=self._max_length_equal_32_bit_integer,
            truncation="do_not_truncate",
        )
        return cast("list[int]", token_ids_with_start_and_end_token_ids)

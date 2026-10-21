"""NLTK 文本分割器。"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from typing_extensions import override

from langchain_text_splitters.base import TextSplitter

if TYPE_CHECKING:
    from collections.abc import Callable


class NLTKTextSplitter(TextSplitter):
    """使用 NLTK 包进行文本分割。"""

    def __init__(
        self,
        separator: str = "\n\n",
        language: str = "english",
        *,
        use_span_tokenize: bool = False,
        **kwargs: Any,
    ) -> None:
        """初始化 NLTK 分割器。

        Args:
            separator: 合并分割结果时使用的分隔符。
            language: 使用的语言。
            use_span_tokenize: 是否使用 `span_tokenize` 而非
                `sent_tokenize`。

        Raises:
            ImportError: 如果未安装 NLTK。
            ValueError: 如果 `use_span_tokenize` 为 `True` 且分隔符不为 `''`。
        """
        super().__init__(**kwargs)
        self._separator = separator
        if use_span_tokenize and self._separator:
            msg = "When use_span_tokenize is True, separator should be ''"
            raise ValueError(msg)
        try:
            import nltk  # noqa: PLC0415,F401
        except ImportError as err:
            msg = "NLTK is not installed, please install it with `pip install nltk`."
            raise ImportError(msg) from err
        if use_span_tokenize:
            self._tokenizer = self._span_tokenizer(language)
        else:
            self._tokenizer = self._sent_tokenizer(language)

    @staticmethod
    def _sent_tokenizer(language: str) -> Callable[[str], list[str]]:
        import nltk  # noqa: PLC0415

        return lambda text: cast(
            "list[str]", nltk.tokenize.sent_tokenize(text, language)
        )

    @staticmethod
    def _span_tokenizer(language: str) -> Callable[[str], list[str]]:
        import nltk  # noqa: PLC0415

        tokenizer = nltk.tokenize._get_punkt_tokenizer(language)  # noqa: SLF001

        def _tokenize(text: str) -> list[str]:
            spans = list(tokenizer.span_tokenize(text))
            splits = []
            for i, (start, end) in enumerate(spans):
                if i > 0:
                    prev_end = spans[i - 1][1]
                    sentence = text[prev_end:start] + text[start:end]
                else:
                    sentence = text[start:end]
                splits.append(sentence)
            return splits

        return _tokenize

    @override
    def split_text(self, text: str) -> list[str]:
                # 首先我们天真地将大段输入拆分为若干小段。
        splits = self._tokenizer(text)
        return self._merge_splits(splits, self._separator)

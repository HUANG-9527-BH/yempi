"""Konlpy 文本分割器。"""

from __future__ import annotations

from typing import Any

from typing_extensions import override

from langchain_text_splitters.base import TextSplitter


class KonlpyTextSplitter(TextSplitter):
    """使用 Konlpy 包进行文本分割。

    适用于分割韩语文本。
    """

    def __init__(
        self,
        separator: str = "\n\n",
        **kwargs: Any,
    ) -> None:
        """初始化 Konlpy 文本分割器。

        Args:
            separator: 合并分割结果时使用的分隔符。

        Raises:
            ImportError: 如果未安装 Konlpy。
        """
        super().__init__(**kwargs)
        self._separator = separator
        try:
            import konlpy  # noqa: PLC0415
        except ImportError as err:
            msg = (
                "Konlpy is not installed, please install it with `pip install konlpy`."
            )
            raise ImportError(msg) from err
        self.kkma = konlpy.tag.Kkma()

    @override
    def split_text(self, text: str) -> list[str]:
        splits = self.kkma.sentences(text)
        return self._merge_splits(splits, self._separator)

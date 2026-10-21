"""LaTeX 文本分割器。"""

from __future__ import annotations

from typing import Any

from langchain_text_splitters.base import Language
from langchain_text_splitters.character import RecursiveCharacterTextSplitter


class LatexTextSplitter(RecursiveCharacterTextSplitter):
    """尝试沿 LaTeX 格式的布局元素分割文本。"""

    def __init__(self, **kwargs: Any) -> None:
        """初始化一个 LatexTextSplitter。"""
        separators = self.get_separators_for_language(Language.LATEX)
        super().__init__(separators=separators, **kwargs)

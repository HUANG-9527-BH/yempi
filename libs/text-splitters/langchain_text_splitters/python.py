"""Python 代码文本分割器。"""

from __future__ import annotations

from typing import Any

from langchain_text_splitters.base import Language
from langchain_text_splitters.character import RecursiveCharacterTextSplitter


class PythonCodeTextSplitter(RecursiveCharacterTextSplitter):
    """尝试沿 Python 语法分割文本。"""

    def __init__(self, **kwargs: Any) -> None:
        """初始化一个 `PythonCodeTextSplitter`。"""
        separators = self.get_separators_for_language(Language.PYTHON)
        super().__init__(separators=separators, **kwargs)

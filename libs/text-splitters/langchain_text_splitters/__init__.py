"""文本分割器是用于分割文本的类。

!!! note

    `MarkdownHeaderTextSplitter` 和 `HTMLHeaderTextSplitter` 不继承自
    `TextSplitter`。
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

from langchain_text_splitters.base import (
    Language,
    TextSplitter,
    Tokenizer,
    TokenTextSplitter,
    split_text_on_tokens,
)
from langchain_text_splitters.character import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
)
from langchain_text_splitters.html import (
    ElementType,
    HTMLHeaderTextSplitter,
    HTMLSectionSplitter,
    HTMLSemanticPreservingSplitter,
)
from langchain_text_splitters.json import RecursiveJsonSplitter
from langchain_text_splitters.jsx import JSFrameworkTextSplitter
from langchain_text_splitters.latex import LatexTextSplitter
from langchain_text_splitters.markdown import (
    ExperimentalMarkdownSyntaxTextSplitter,
    HeaderType,
    LineType,
    MarkdownHeaderTextSplitter,
    MarkdownTextSplitter,
)
from langchain_text_splitters.python import PythonCodeTextSplitter

if TYPE_CHECKING:
    from langchain_text_splitters.konlpy import KonlpyTextSplitter
    from langchain_text_splitters.nltk import NLTKTextSplitter
    from langchain_text_splitters.sentence_transformers import (
        SentenceTransformersTokenTextSplitter,
    )
    from langchain_text_splitters.spacy import SpacyTextSplitter

__all__ = [
    "CharacterTextSplitter",
    "ElementType",
    "ExperimentalMarkdownSyntaxTextSplitter",
    "HTMLHeaderTextSplitter",
    "HTMLSectionSplitter",
    "HTMLSemanticPreservingSplitter",
    "HeaderType",
    "JSFrameworkTextSplitter",
    "KonlpyTextSplitter",
    "Language",
    "LatexTextSplitter",
    "LineType",
    "MarkdownHeaderTextSplitter",
    "MarkdownTextSplitter",
    "NLTKTextSplitter",
    "PythonCodeTextSplitter",
    "RecursiveCharacterTextSplitter",
    "RecursiveJsonSplitter",
    "SentenceTransformersTokenTextSplitter",
    "SpacyTextSplitter",
    "TextSplitter",
    "TokenTextSplitter",
    "Tokenizer",
    "split_text_on_tokens",
]

# 分割器模块会引入较重的可选依赖项（konlpy、nltk、
# spacy、sentence-transformers/torch）。通过在 `__getattr__` 后延迟导入，
# 即使这些类仍在 `__all__` 中且首次访问时完全可用，也能保持 `import
# langchain_text_splitters` 的轻量性。
# 即使这些类仍在 `__all__` 中且首次访问时完全可用，也能保持 `import
# langchain_text_splitters` 的轻量性。
# 即使这些类仍在 `__all__` 中且首次访问时完全可用，也能保持 `import
# langchain_text_splitters` 的轻量性。
_LAZY_SPLITTERS: dict[str, str] = {
    "KonlpyTextSplitter": "konlpy",
    "NLTKTextSplitter": "nltk",
    "SentenceTransformersTokenTextSplitter": "sentence_transformers",
    "SpacyTextSplitter": "spacy",
}


def __getattr__(attr_name: str) -> object:
    module_name = _LAZY_SPLITTERS.get(attr_name)
    if module_name is not None:
        module = import_module(f".{module_name}", __name__)
        result = getattr(module, attr_name)
        globals()[attr_name] = result
        return result
    msg = f"module {__name__!r} has no attribute {attr_name!r}"
    raise AttributeError(msg)

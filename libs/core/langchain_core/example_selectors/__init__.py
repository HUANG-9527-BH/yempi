"""示例选择器。

**示例选择器** 实现了选择示例并将其包含在提示词中的逻辑。
这使我们能够选择与输入最相关的示例。
"""

from typing import TYPE_CHECKING

from langchain_core._import_utils import import_attr

if TYPE_CHECKING:
    from langchain_core.example_selectors.base import BaseExampleSelector
    from langchain_core.example_selectors.length_based import (
        LengthBasedExampleSelector,
    )
    from langchain_core.example_selectors.semantic_similarity import (
        MaxMarginalRelevanceExampleSelector,
        SemanticSimilarityExampleSelector,
        sorted_values,
    )

__all__ = (
    "BaseExampleSelector",
    "LengthBasedExampleSelector",
    "MaxMarginalRelevanceExampleSelector",
    "SemanticSimilarityExampleSelector",
    "sorted_values",
)

_dynamic_imports = {
    "BaseExampleSelector": "base",
    "LengthBasedExampleSelector": "length_based",
    "MaxMarginalRelevanceExampleSelector": "semantic_similarity",
    "SemanticSimilarityExampleSelector": "semantic_similarity",
    "sorted_values": "semantic_similarity",
}


def __getattr__(attr_name: str) -> object:
    module_name = _dynamic_imports.get(attr_name)
    result = import_attr(attr_name, module_name, __spec__.parent)
    globals()[attr_name] = result
    return result


def __dir__() -> list[str]:
    return list(__all__)

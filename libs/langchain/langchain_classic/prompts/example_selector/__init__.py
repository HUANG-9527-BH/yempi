"""用于选择包含在提示词中的示例的逻辑。"""

from typing import TYPE_CHECKING, Any

from langchain_core.example_selectors.length_based import (
    LengthBasedExampleSelector,
)
from langchain_core.example_selectors.semantic_similarity import (
    MaxMarginalRelevanceExampleSelector,
    SemanticSimilarityExampleSelector,
)

from langchain_classic._api import create_importer

if TYPE_CHECKING:
    from langchain_community.example_selectors.ngram_overlap import (
        NGramOverlapExampleSelector,
    )

# 创建一种动态查找已弃用导入的方式。
# 用于整合引发弃用警告的逻辑，
# 以及处理可选导入。
DEPRECATED_LOOKUPS = {
    "NGramOverlapExampleSelector": (
        "langchain_community.example_selectors.ngram_overlap"
    ),
}

_import_attribute = create_importer(__file__, deprecated_lookups=DEPRECATED_LOOKUPS)


def __getattr__(name: str) -> Any:
    """动态查找属性。"""
    return _import_attribute(name)


__all__ = [
    "LengthBasedExampleSelector",
    "MaxMarginalRelevanceExampleSelector",
    "NGramOverlapExampleSelector",
    "SemanticSimilarityExampleSelector",
]

"""提示词是模型的输入。

Prompt is often constructed
from multiple components. Prompt classes and functions make constructing and working
with prompts easy.
"""

from typing import TYPE_CHECKING, Any

from langchain_core.example_selectors import (
    LengthBasedExampleSelector,
    MaxMarginalRelevanceExampleSelector,
    SemanticSimilarityExampleSelector,
)
from langchain_core.prompts import (
    AIMessagePromptTemplate,
    BaseChatPromptTemplate,
    BasePromptTemplate,
    ChatMessagePromptTemplate,
    ChatPromptTemplate,
    FewShotChatMessagePromptTemplate,
    FewShotPromptTemplate,
    FewShotPromptWithTemplates,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
    PromptTemplate,
    StringPromptTemplate,
    SystemMessagePromptTemplate,
    load_prompt,
)

from langchain_classic._api import create_importer
from langchain_classic.prompts.prompt import Prompt

if TYPE_CHECKING:
    from langchain_community.example_selectors.ngram_overlap import (
        NGramOverlapExampleSelector,
    )

# 创建一种动态查找已弃用导入的方式。
# 用于统一处理弃用警告的逻辑并
# 处理可选导入。
MODULE_LOOKUP = {
    "NGramOverlapExampleSelector": (
        "langchain_community.example_selectors.ngram_overlap"
    ),
}

_import_attribute = create_importer(__file__, module_lookup=MODULE_LOOKUP)


def __getattr__(name: str) -> Any:
    """动态查找属性。"""
    return _import_attribute(name)


__all__ = [
    "AIMessagePromptTemplate",
    "BaseChatPromptTemplate",
    "BasePromptTemplate",
    "ChatMessagePromptTemplate",
    "ChatPromptTemplate",
    "FewShotChatMessagePromptTemplate",
    "FewShotPromptTemplate",
    "FewShotPromptWithTemplates",
    "HumanMessagePromptTemplate",
    "LengthBasedExampleSelector",
    "MaxMarginalRelevanceExampleSelector",
    "MessagesPlaceholder",
    "NGramOverlapExampleSelector",
    "Prompt",
    "PromptTemplate",
    "SemanticSimilarityExampleSelector",
    "StringPromptTemplate",
    "SystemMessagePromptTemplate",
    "load_prompt",
]

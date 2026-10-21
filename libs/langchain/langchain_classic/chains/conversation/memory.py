"""用于对话提示词的记忆模块。"""

from typing import TYPE_CHECKING, Any

from langchain_classic._api import create_importer
from langchain_classic.memory.buffer import (
    ConversationBufferMemory,
    ConversationStringBufferMemory,
)
from langchain_classic.memory.buffer_window import ConversationBufferWindowMemory
from langchain_classic.memory.combined import CombinedMemory
from langchain_classic.memory.entity import ConversationEntityMemory
from langchain_classic.memory.summary import ConversationSummaryMemory
from langchain_classic.memory.summary_buffer import ConversationSummaryBufferMemory

if TYPE_CHECKING:
    from langchain_community.memory.kg import ConversationKGMemory

# 创建一种动态查找已弃用导入的方式。
# 用于整合引发弃用警告的逻辑，并
# 处理可选导入。
DEPRECATED_LOOKUP = {
    "ConversationKGMemory": "langchain_community.memory.kg",
}

_importer = create_importer(__package__, deprecated_lookups=DEPRECATED_LOOKUP)


def __getattr__(name: str) -> Any:
    """动态查找属性。"""
    return _importer(name)


# 仅用于向后兼容。

__all__ = [
    "CombinedMemory",
    "ConversationBufferMemory",
    "ConversationBufferWindowMemory",
    "ConversationEntityMemory",
    "ConversationKGMemory",
    "ConversationStringBufferMemory",
    "ConversationSummaryBufferMemory",
    "ConversationSummaryMemory",
]

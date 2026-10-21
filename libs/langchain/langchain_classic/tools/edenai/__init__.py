"""Edenai 工具。"""

from typing import TYPE_CHECKING, Any

from langchain_classic._api import create_importer

if TYPE_CHECKING:
    from langchain_community.tools import (
        EdenAiExplicitImageTool,
        EdenAiObjectDetectionTool,
        EdenAiParsingIDTool,
        EdenAiParsingInvoiceTool,
        EdenAiSpeechToTextTool,
        EdenAiTextModerationTool,
        EdenAiTextToSpeechTool,
        EdenaiTool,
    )

# 创建一种动态查找已弃用导入的方式。
# 用于统一处理引发弃用警告的逻辑，并
# 处理可选导入。
DEPRECATED_LOOKUP = {
    "EdenAiExplicitImageTool": "langchain_community.tools",
    "EdenAiObjectDetectionTool": "langchain_community.tools",
    "EdenAiParsingIDTool": "langchain_community.tools",
    "EdenAiParsingInvoiceTool": "langchain_community.tools",
    "EdenAiTextToSpeechTool": "langchain_community.tools",
    "EdenAiSpeechToTextTool": "langchain_community.tools",
    "EdenAiTextModerationTool": "langchain_community.tools",
    "EdenaiTool": "langchain_community.tools",
}

_import_attribute = create_importer(__package__, deprecated_lookups=DEPRECATED_LOOKUP)


def __getattr__(name: str) -> Any:
    """动态查找属性。"""
    return _import_attribute(name)


__all__ = [
    "EdenAiExplicitImageTool",
    "EdenAiObjectDetectionTool",
    "EdenAiParsingIDTool",
    "EdenAiParsingInvoiceTool",
    "EdenAiSpeechToTextTool",
    "EdenAiTextModerationTool",
    "EdenAiTextToSpeechTool",
    "EdenaiTool",
]

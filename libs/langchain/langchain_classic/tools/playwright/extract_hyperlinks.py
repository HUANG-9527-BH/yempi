from typing import TYPE_CHECKING, Any

from langchain_classic._api import create_importer

if TYPE_CHECKING:
    from langchain_community.tools import ExtractHyperlinksTool
    from langchain_community.tools.playwright.extract_hyperlinks import (
        ExtractHyperlinksToolInput,
    )

# 创建一种动态查找已弃用导入的方式。
# 用于合并引发弃用警告和处理可选导入的逻辑。
# 处理可选导入。
DEPRECATED_LOOKUP = {
    "ExtractHyperlinksToolInput": (
        "langchain_community.tools.playwright.extract_hyperlinks"
    ),
    "ExtractHyperlinksTool": "langchain_community.tools",
}

_import_attribute = create_importer(__package__, deprecated_lookups=DEPRECATED_LOOKUP)


def __getattr__(name: str) -> Any:
    """动态查找属性。"""
    return _import_attribute(name)


__all__ = [
    "ExtractHyperlinksTool",
    "ExtractHyperlinksToolInput",
]

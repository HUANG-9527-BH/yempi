from typing import TYPE_CHECKING, Any

from langchain_classic._api import create_importer

if TYPE_CHECKING:
    from langchain_community.tools import DuckDuckGoSearchResults, DuckDuckGoSearchRun
    from langchain_community.tools.ddg_search.tool import DDGInput, DuckDuckGoSearchTool

# 创建一种动态查找已弃用导入的方式。
# 用于统一处理引发弃用警告的逻辑和
# 处理可选导入。
DEPRECATED_LOOKUP = {
    "DDGInput": "langchain_community.tools.ddg_search.tool",
    "DuckDuckGoSearchRun": "langchain_community.tools",
    "DuckDuckGoSearchResults": "langchain_community.tools",
    "DuckDuckGoSearchTool": "langchain_community.tools.ddg_search.tool",
}

_import_attribute = create_importer(__package__, deprecated_lookups=DEPRECATED_LOOKUP)


def __getattr__(name: str) -> Any:
    """动态查找属性。"""
    return _import_attribute(name)


__all__ = [
    "DDGInput",
    "DuckDuckGoSearchResults",
    "DuckDuckGoSearchRun",
    "DuckDuckGoSearchTool",
]

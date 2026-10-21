"""用于解析 OpenAPI 规范的实用函数。为向后兼容而保留。"""

from typing import TYPE_CHECKING, Any

from langchain_classic._api import create_importer

if TYPE_CHECKING:
    from langchain_community.tools import OpenAPISpec
    from langchain_community.utilities.openapi import HTTPVerb

# 创建一种动态查找已弃用导入的方式。
# 用于整合引发弃用警告的逻辑，并
# 处理可选导入。
DEPRECATED_LOOKUP = {
    "HTTPVerb": "langchain_community.utilities.openapi",
    "OpenAPISpec": "langchain_community.tools",
}

_import_attribute = create_importer(__package__, deprecated_lookups=DEPRECATED_LOOKUP)


def __getattr__(name: str) -> Any:
    """动态查找属性。"""
    return _import_attribute(name)


__all__ = [
    "HTTPVerb",
    "OpenAPISpec",
]

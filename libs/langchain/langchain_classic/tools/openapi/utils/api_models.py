from typing import TYPE_CHECKING, Any

from langchain_classic._api import create_importer

if TYPE_CHECKING:
    from langchain_community.tools import APIOperation
    from langchain_community.tools.openapi.utils.api_models import (
        INVALID_LOCATION_TEMPL,
        PRIMITIVE_TYPES,
        SCHEMA_TYPE,
        SUPPORTED_LOCATIONS,
        APIProperty,
        APIPropertyBase,
        APIPropertyLocation,
        APIRequestBody,
        APIRequestBodyProperty,
    )

# 创建一种动态查找已弃用导入的方式。
# 用于统一处理引发弃用警告和
# 处理可选导入的逻辑。
DEPRECATED_LOOKUP = {
    "APIPropertyLocation": "langchain_community.tools.openapi.utils.api_models",
    "APIPropertyBase": "langchain_community.tools.openapi.utils.api_models",
    "APIProperty": "langchain_community.tools.openapi.utils.api_models",
    "APIRequestBodyProperty": "langchain_community.tools.openapi.utils.api_models",
    "APIRequestBody": "langchain_community.tools.openapi.utils.api_models",
    "APIOperation": "langchain_community.tools",
    "INVALID_LOCATION_TEMPL": "langchain_community.tools.openapi.utils.api_models",
    "SCHEMA_TYPE": "langchain_community.tools.openapi.utils.api_models",
    "PRIMITIVE_TYPES": "langchain_community.tools.openapi.utils.api_models",
    "SUPPORTED_LOCATIONS": "langchain_community.tools.openapi.utils.api_models",
}

_import_attribute = create_importer(__package__, deprecated_lookups=DEPRECATED_LOOKUP)


def __getattr__(name: str) -> Any:
    """动态查找属性。"""
    return _import_attribute(name)


__all__ = [
    "INVALID_LOCATION_TEMPL",
    "PRIMITIVE_TYPES",
    "SCHEMA_TYPE",
    "SUPPORTED_LOCATIONS",
    "APIOperation",
    "APIProperty",
    "APIPropertyBase",
    "APIPropertyLocation",
    "APIRequestBody",
    "APIRequestBodyProperty",
]

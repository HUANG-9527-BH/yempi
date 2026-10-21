from typing import TYPE_CHECKING, Any

from langchain_classic._api import create_importer

if TYPE_CHECKING:
    from langchain_community.utilities.vertexai import (
        create_retry_decorator,
        get_client_info,
        init_vertexai,
        raise_vertex_import_error,
    )

# 创建一种动态查找已弃用导入的方式。
# 用于统一处理引发弃用警告和
# 处理可选导入的逻辑。
DEPRECATED_LOOKUP = {
    "create_retry_decorator": "langchain_community.utilities.vertexai",
    "raise_vertex_import_error": "langchain_community.utilities.vertexai",
    "init_vertexai": "langchain_community.utilities.vertexai",
    "get_client_info": "langchain_community.utilities.vertexai",
}

_import_attribute = create_importer(__package__, deprecated_lookups=DEPRECATED_LOOKUP)


def __getattr__(name: str) -> Any:
    """动态查找属性。"""
    return _import_attribute(name)


__all__ = [
    "create_retry_decorator",
    "get_client_info",
    "init_vertexai",
    "raise_vertex_import_error",
]

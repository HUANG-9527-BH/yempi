from typing import TYPE_CHECKING, Any

from langchain_classic._api import create_importer

if TYPE_CHECKING:
    from langchain_community.tools.multion.close_session import (
        CloseSessionSchema,
        MultionCloseSession,
    )

# 创建一种动态查找已弃用导入的方式。
# 用于统一处理引发弃用警告的逻辑，并
# 处理可选导入。
DEPRECATED_LOOKUP = {
    "CloseSessionSchema": "langchain_community.tools.multion.close_session",
    "MultionCloseSession": "langchain_community.tools.multion.close_session",
}

_import_attribute = create_importer(__package__, deprecated_lookups=DEPRECATED_LOOKUP)


def __getattr__(name: str) -> Any:
    """动态查找属性。"""
    return _import_attribute(name)


__all__ = [
    "CloseSessionSchema",
    "MultionCloseSession",
]

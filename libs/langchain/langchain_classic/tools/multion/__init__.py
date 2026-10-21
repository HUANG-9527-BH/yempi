"""MutliOn 客户端 API 工具。"""

from typing import TYPE_CHECKING, Any

from langchain_classic._api import create_importer

if TYPE_CHECKING:
    from langchain_community.tools.multion.close_session import MultionCloseSession
    from langchain_community.tools.multion.create_session import MultionCreateSession
    from langchain_community.tools.multion.update_session import MultionUpdateSession

# 创建一种动态查找已弃用导入的方式。
# 用于统一处理弃用警告的逻辑和
# 处理可选导入。
DEPRECATED_LOOKUP = {
    "MultionCreateSession": "langchain_community.tools.multion.create_session",
    "MultionUpdateSession": "langchain_community.tools.multion.update_session",
    "MultionCloseSession": "langchain_community.tools.multion.close_session",
}

_import_attribute = create_importer(__package__, deprecated_lookups=DEPRECATED_LOOKUP)


def __getattr__(name: str) -> Any:
    """动态查找属性。"""
    return _import_attribute(name)


__all__ = [
    "MultionCloseSession",
    "MultionCreateSession",
    "MultionUpdateSession",
]

from typing import TYPE_CHECKING, Any

from langchain_classic._api import create_importer

if TYPE_CHECKING:
    from langchain_community.agent_toolkits.openapi.planner import (
        RequestsDeleteToolWithParsing,
        RequestsGetToolWithParsing,
        RequestsPatchToolWithParsing,
        RequestsPostToolWithParsing,
        RequestsPutToolWithParsing,
        create_openapi_agent,
    )

# 创建一种动态查找已弃用导入的方式。
# 用于统一处理引发弃用警告和
# 处理可选导入的逻辑。
DEPRECATED_LOOKUP = {
    "RequestsGetToolWithParsing": (
        "langchain_community.agent_toolkits.openapi.planner"
    ),
    "RequestsPostToolWithParsing": (
        "langchain_community.agent_toolkits.openapi.planner"
    ),
    "RequestsPatchToolWithParsing": (
        "langchain_community.agent_toolkits.openapi.planner"
    ),
    "RequestsPutToolWithParsing": (
        "langchain_community.agent_toolkits.openapi.planner"
    ),
    "RequestsDeleteToolWithParsing": (
        "langchain_community.agent_toolkits.openapi.planner"
    ),
    "create_openapi_agent": "langchain_community.agent_toolkits.openapi.planner",
}

_import_attribute = create_importer(__package__, deprecated_lookups=DEPRECATED_LOOKUP)


def __getattr__(name: str) -> Any:
    """动态查找属性。"""
    return _import_attribute(name)


__all__ = [
    "RequestsDeleteToolWithParsing",
    "RequestsGetToolWithParsing",
    "RequestsPatchToolWithParsing",
    "RequestsPostToolWithParsing",
    "RequestsPutToolWithParsing",
    "create_openapi_agent",
]

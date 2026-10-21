"""文件管理工具。"""

from typing import TYPE_CHECKING, Any

from langchain_classic._api import create_importer

if TYPE_CHECKING:
    from langchain_community.tools import (
        CopyFileTool,
        DeleteFileTool,
        FileSearchTool,
        ListDirectoryTool,
        MoveFileTool,
        ReadFileTool,
        WriteFileTool,
    )

# 创建一种动态查找已弃用导入的方式。
# 用于整合引发弃用警告的逻辑，
# 以及处理可选导入。
DEPRECATED_LOOKUP = {
    "CopyFileTool": "langchain_community.tools",
    "DeleteFileTool": "langchain_community.tools",
    "FileSearchTool": "langchain_community.tools",
    "MoveFileTool": "langchain_community.tools",
    "ReadFileTool": "langchain_community.tools",
    "WriteFileTool": "langchain_community.tools",
    "ListDirectoryTool": "langchain_community.tools",
}

_import_attribute = create_importer(__package__, deprecated_lookups=DEPRECATED_LOOKUP)


def __getattr__(name: str) -> Any:
    """动态查找属性。"""
    return _import_attribute(name)


__all__ = [
    "CopyFileTool",
    "DeleteFileTool",
    "FileSearchTool",
    "ListDirectoryTool",
    "MoveFileTool",
    "ReadFileTool",
    "WriteFileTool",
]

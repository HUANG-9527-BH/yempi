"""图结构为图数据库提供了自然语言接口。"""

from typing import TYPE_CHECKING, Any

from langchain_classic._api import create_importer

if TYPE_CHECKING:
    from langchain_community.graphs.index_creator import GraphIndexCreator
    from langchain_community.graphs.networkx_graph import NetworkxEntityGraph


# 创建一种动态查找已弃用导入的方式。
# 用于统一处理弃用警告的逻辑并
# 处理可选导入。
DEPRECATED_LOOKUP = {
    "GraphIndexCreator": "langchain_community.graphs.index_creator",
    "NetworkxEntityGraph": "langchain_community.graphs.networkx_graph",
}

_import_attribute = create_importer(__package__, deprecated_lookups=DEPRECATED_LOOKUP)


def __getattr__(name: str) -> Any:
    """动态查找属性。"""
    return _import_attribute(name)


__all__ = ["GraphIndexCreator", "NetworkxEntityGraph"]

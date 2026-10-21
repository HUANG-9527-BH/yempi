from typing import TYPE_CHECKING, Any

from langchain_classic._api import create_importer

if TYPE_CHECKING:
    from langchain_community.tools import NasaAction

# 创建一种动态查找已弃用导入的方式。
# 用于合并引发弃用警告的逻辑和
# 处理可选导入的逻辑。
DEPRECATED_LOOKUP = {"NasaAction": "langchain_community.tools"}

_import_attribute = create_importer(__package__, deprecated_lookups=DEPRECATED_LOOKUP)


def __getattr__(name: str) -> Any:
    """动态查找属性。"""
    return _import_attribute(name)


__all__ = [
    "NasaAction",
]

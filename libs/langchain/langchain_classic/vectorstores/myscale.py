from typing import TYPE_CHECKING, Any

from langchain_classic._api import create_importer

if TYPE_CHECKING:
    from langchain_community.vectorstores import MyScale, MyScaleSettings
    from langchain_community.vectorstores.myscale import MyScaleWithoutJSON

# 创建一种动态查找已弃用导入的方式。
# 用于统一处理引发弃用警告和
# 处理可选导入的逻辑。
DEPRECATED_LOOKUP = {
    "MyScaleSettings": "langchain_community.vectorstores",
    "MyScale": "langchain_community.vectorstores",
    "MyScaleWithoutJSON": "langchain_community.vectorstores.myscale",
}

_import_attribute = create_importer(__package__, deprecated_lookups=DEPRECATED_LOOKUP)


def __getattr__(name: str) -> Any:
    """动态查找属性。"""
    return _import_attribute(name)


__all__ = [
    "MyScale",
    "MyScaleSettings",
    "MyScaleWithoutJSON",
]

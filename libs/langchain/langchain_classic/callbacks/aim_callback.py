from typing import TYPE_CHECKING, Any

from langchain_classic._api import create_importer

if TYPE_CHECKING:
    from langchain_community.callbacks.aim_callback import (
        AimCallbackHandler,
        BaseMetadataCallbackHandler,
        import_aim,
    )

# 创建一种动态查找已弃用导入的方式。
# 用于统一处理引发弃用警告和
# 处理可选导入的逻辑。
DEPRECATED_LOOKUP = {
    "import_aim": "langchain_community.callbacks.aim_callback",
    "BaseMetadataCallbackHandler": "langchain_community.callbacks.aim_callback",
    "AimCallbackHandler": "langchain_community.callbacks.aim_callback",
}

_import_attribute = create_importer(__file__, deprecated_lookups=DEPRECATED_LOOKUP)


def __getattr__(name: str) -> Any:
    """动态查找属性。"""
    return _import_attribute(name)


__all__ = [
    "AimCallbackHandler",
    "BaseMetadataCallbackHandler",
    "import_aim",
]

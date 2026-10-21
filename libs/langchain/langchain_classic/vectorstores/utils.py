from typing import TYPE_CHECKING, Any

from langchain_classic._api import create_importer

if TYPE_CHECKING:
    from langchain_community.vectorstores.utils import (
        DistanceStrategy,
        filter_complex_metadata,
        maximal_marginal_relevance,
    )

# 创建一种动态查找已弃用导入的方式。
# 用于合并引发弃用警告和处理可选导入的逻辑。
# 处理可选导入。
DEPRECATED_LOOKUP = {
    "DistanceStrategy": "langchain_community.vectorstores.utils",
    "maximal_marginal_relevance": "langchain_community.vectorstores.utils",
    "filter_complex_metadata": "langchain_community.vectorstores.utils",
}

_import_attribute = create_importer(__package__, deprecated_lookups=DEPRECATED_LOOKUP)


def __getattr__(name: str) -> Any:
    """动态查找属性。"""
    return _import_attribute(name)


__all__ = [
    "DistanceStrategy",
    "filter_complex_metadata",
    "maximal_marginal_relevance",
]

from typing import TYPE_CHECKING, Any

from langchain_classic._api import create_importer

if TYPE_CHECKING:
    from langchain_community.utils.math import (
        cosine_similarity,
        cosine_similarity_top_k,
    )

# 创建一种动态查找已弃用导入的方式。
# 用于统一处理引发弃用警告的逻辑和
# 处理可选导入。
# 未标记为已弃用，因为只要我们接受 numpy 作为依赖，可能希望将此功能
# 移入 langchain。
_MODULE_LOOKUP = {
    "cosine_similarity": "langchain_community.utils.math",
    "cosine_similarity_top_k": "langchain_community.utils.math",
}

_import_attribute = create_importer(__package__, module_lookup=_MODULE_LOOKUP)


def __getattr__(name: str) -> Any:
    """动态查找属性。"""
    return _import_attribute(name)


__all__ = [
    "cosine_similarity",
    "cosine_similarity_top_k",
]

"""用于管理 LangChain API 的辅助函数。

此模块仅对 LangChain 开发者相关，对用户不适用。

!!! warning

    This module and its submodules are for internal use only. Do not use them in your
    own code. We may change the API at any time with no warning.
"""

from typing import TYPE_CHECKING

from langchain_core._import_utils import import_attr

if TYPE_CHECKING:
    from langchain_core._api.beta_decorator import (
        LangChainBetaWarning,
        beta,
        suppress_langchain_beta_warning,
        surface_langchain_beta_warnings,
    )
    from langchain_core._api.deprecation import (
        LangChainDeprecationWarning,
        deprecated,
        suppress_langchain_deprecation_warning,
        surface_langchain_deprecation_warnings,
        warn_deprecated,
    )
    from langchain_core._api.path import as_import_path, get_relative_path

__all__ = (
    "LangChainBetaWarning",
    "LangChainDeprecationWarning",
    "as_import_path",
    "beta",
    "deprecated",
    "get_relative_path",
    "suppress_langchain_beta_warning",
    "suppress_langchain_deprecation_warning",
    "surface_langchain_beta_warnings",
    "surface_langchain_deprecation_warnings",
    "warn_deprecated",
)

_dynamic_imports = {
    "LangChainBetaWarning": "beta_decorator",
    "beta": "beta_decorator",
    "suppress_langchain_beta_warning": "beta_decorator",
    "surface_langchain_beta_warnings": "beta_decorator",
    "as_import_path": "path",
    "get_relative_path": "path",
    "LangChainDeprecationWarning": "deprecation",
    "deprecated": "deprecation",
    "surface_langchain_deprecation_warnings": "deprecation",
    "suppress_langchain_deprecation_warning": "deprecation",
    "warn_deprecated": "deprecation",
}


def __getattr__(attr_name: str) -> object:
    """动态导入并返回子模块中的属性。

    This function enables lazy loading of API functions from submodules, reducing
    initial import time and circular dependency issues.

    Args:
        attr_name: 要导入的属性名称。

    Returns:
        导入的属性对象。

    Raises:
        AttributeError: 如果该属性不是有效的动态导入项。
    """
    module_name = _dynamic_imports.get(attr_name)
    result = import_attr(attr_name, module_name, __spec__.parent)
    globals()[attr_name] = result
    return result


def __dir__() -> list[str]:
    """返回此模块可用属性的列表。

    Returns:
        可从此模块导入的属性名称列表。
    """
    return list(__all__)

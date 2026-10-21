from typing import TYPE_CHECKING, Any

from langchain_classic._api import create_importer

if TYPE_CHECKING:
    from langchain_community.document_loaders import GitHubIssuesLoader
    from langchain_community.document_loaders.github import BaseGitHubLoader

# 创建一种动态查找已弃用导入的方式。
# 用于整合引发弃用警告和处理可选导入的逻辑。
# 处理可选导入。
DEPRECATED_LOOKUP = {
    "BaseGitHubLoader": "langchain_community.document_loaders.github",
    "GitHubIssuesLoader": "langchain_community.document_loaders",
}

_import_attribute = create_importer(__package__, deprecated_lookups=DEPRECATED_LOOKUP)


def __getattr__(name: str) -> Any:
    """动态查找属性。"""
    return _import_attribute(name)


__all__ = [
    "BaseGitHubLoader",
    "GitHubIssuesLoader",
]

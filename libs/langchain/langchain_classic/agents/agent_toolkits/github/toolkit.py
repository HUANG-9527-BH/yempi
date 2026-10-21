from typing import TYPE_CHECKING, Any

from langchain_classic._api import create_importer

if TYPE_CHECKING:
    from langchain_community.agent_toolkits.github.toolkit import (
        BranchName,
        CommentOnIssue,
        CreateFile,
        CreatePR,
        CreateReviewRequest,
        DeleteFile,
        DirectoryPath,
        GetIssue,
        GetPR,
        GitHubToolkit,
        NoInput,
        ReadFile,
        SearchCode,
        SearchIssuesAndPRs,
        UpdateFile,
    )

# 创建一种动态查找已弃用导入的方式。
# 用于统一处理引发弃用警告的逻辑，并
# 处理可选导入。
DEPRECATED_LOOKUP = {
    "NoInput": "langchain_community.agent_toolkits.github.toolkit",
    "GetIssue": "langchain_community.agent_toolkits.github.toolkit",
    "CommentOnIssue": "langchain_community.agent_toolkits.github.toolkit",
    "GetPR": "langchain_community.agent_toolkits.github.toolkit",
    "CreatePR": "langchain_community.agent_toolkits.github.toolkit",
    "CreateFile": "langchain_community.agent_toolkits.github.toolkit",
    "ReadFile": "langchain_community.agent_toolkits.github.toolkit",
    "UpdateFile": "langchain_community.agent_toolkits.github.toolkit",
    "DeleteFile": "langchain_community.agent_toolkits.github.toolkit",
    "DirectoryPath": "langchain_community.agent_toolkits.github.toolkit",
    "BranchName": "langchain_community.agent_toolkits.github.toolkit",
    "SearchCode": "langchain_community.agent_toolkits.github.toolkit",
    "CreateReviewRequest": "langchain_community.agent_toolkits.github.toolkit",
    "SearchIssuesAndPRs": "langchain_community.agent_toolkits.github.toolkit",
    "GitHubToolkit": "langchain_community.agent_toolkits.github.toolkit",
}

_import_attribute = create_importer(__package__, deprecated_lookups=DEPRECATED_LOOKUP)


def __getattr__(name: str) -> Any:
    """动态查找属性。"""
    return _import_attribute(name)


__all__ = [
    "BranchName",
    "CommentOnIssue",
    "CreateFile",
    "CreatePR",
    "CreateReviewRequest",
    "DeleteFile",
    "DirectoryPath",
    "GetIssue",
    "GetPR",
    "GitHubToolkit",
    "NoInput",
    "ReadFile",
    "SearchCode",
    "SearchIssuesAndPRs",
    "UpdateFile",
]

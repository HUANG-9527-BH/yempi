from typing import Any

from langchain_core._api.deprecation import deprecated
from langchain_core.runnables.base import RunnableBindingBase
from langchain_core.runnables.utils import Input, Output


@deprecated(
    since="1.0.7",
    removal="2.0.0",
    message=(
        "langchain_classic.hub.pull is deprecated. Use the LangSmith SDK instead."
    ),
)
class HubRunnable(RunnableBindingBase[Input, Output]):  # type: ignore[no-redef]
    """LangChain Hub 中存储的 Runnable 实例。"""

    owner_repo_commit: str

    def __init__(
        self,
        owner_repo_commit: str,
        *,
        api_url: str | None = None,
        api_key: str | None = None,
        **kwargs: Any,
    ) -> None:
        """初始化 `HubRunnable`。

        Args:
            owner_repo_commit: 从 LangChain Hub 拉取提示词的完整名称，格式为
                `owner/prompt_name:commit_hash` 或 `owner/prompt_name`
                或仅 `prompt_name`（如果是您自己的提示词）。
            api_url: LangChain Hub API 的 URL。
                如果已设置 API 密钥，则默认使用托管 API 服务；
                否则使用本地主机实例。
            api_key: 用于与 LangChain Hub API 认证的 API 密钥。
            **kwargs: 传递给父类的其他关键字参数。
        """
        from langchain_classic.hub import pull

        pulled = pull(owner_repo_commit, api_url=api_url, api_key=api_key)
        super_kwargs = {
            "kwargs": {},
            "config": {},
            **kwargs,
            "bound": pulled,
            "owner_repo_commit": owner_repo_commit,
        }
        super().__init__(**super_kwargs)

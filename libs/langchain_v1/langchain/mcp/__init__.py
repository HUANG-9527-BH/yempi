"""LangChain MCP 适配器，用于将 MCP 服务器与 LangChain 应用连接起来。

Interrupt-driven elicitation has its own types — the interrupt payload, the
answers a run resumes with, and the discriminator to recognize them by. Import
those from `langchain.mcp.elicitation`.

!!! warning "此命名空间处于测试版"

    `langchain.mcp` is actively being worked on and its API may change. Importing
    from it raises a `LangChainBetaWarning` once per process. Silence it with
    `warnings.filterwarnings("ignore", category=LangChainBetaWarning)`, or scope
    the suppression with `langchain_core._api.suppress_langchain_beta_warning()`.
"""

import warnings

from langchain_core._api import LangChainBetaWarning

from langchain.mcp.adapter import MCPAdapter
from langchain.mcp.tools import MCPToolArtifact, as_langchain_tool

# 导入时即发出警告，而非通过 `@beta`
# 注解，后者仅在函数或类被调用时才触发。状态属于整个命名空间，调用者应在接触该命名空间时就知晓其状态——包括直接导入子模块时，这会先执行本模块。
# 导入时即发出警告，而非通过 `@beta`
# 注解，后者仅在函数或类被调用时才触发。状态属于整个命名空间，调用者应在接触该命名空间时就知晓其状态——包括直接导入子模块时，这会先执行本模块。
# 导入时即发出警告，而非通过 `@beta`
# 注解，后者仅在函数或类被调用时才触发。状态属于整个命名空间，调用者应在接触该命名空间时就知晓其状态——包括直接导入子模块时，这会先执行本模块。
# 导入时即发出警告，而非通过 `@beta`
# 注解，后者仅在函数或类被调用时才触发。状态属于整个命名空间，调用者应在接触该命名空间时就知晓其状态——包括直接导入子模块时，这会先执行本模块。
warnings.warn(
    "`langchain.mcp` is in beta. It is actively being worked on, so the API may change.",
    LangChainBetaWarning,
    stacklevel=2,
)

__all__ = [
    "MCPAdapter",
    "MCPToolArtifact",
    "as_langchain_tool",
]

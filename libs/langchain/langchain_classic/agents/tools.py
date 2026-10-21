"""工具的接口。"""

from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.tools import BaseTool, tool
from typing_extensions import override


class InvalidTool(BaseTool):
    """当智能体遇到无效工具名称时运行的工具。"""

    name: str = "invalid_tool"
    """工具的名称。"""
    description: str = "Called when tool name is invalid. Suggests valid tool names."
    """工具的描述。"""

    @override
    def _run(
        self,
        requested_tool_name: str,
        available_tool_names: list[str],
        run_manager: CallbackManagerForToolRun | None = None,
    ) -> str:
        """使用该工具。"""
        available_tool_names_str = ", ".join(list(available_tool_names))
        return (
            f"{requested_tool_name} is not a valid tool, "
            f"try one of [{available_tool_names_str}]."
        )

    @override
    async def _arun(
        self,
        requested_tool_name: str,
        available_tool_names: list[str],
        run_manager: AsyncCallbackManagerForToolRun | None = None,
    ) -> str:
        """异步使用该工具。"""
        available_tool_names_str = ", ".join(list(available_tool_names))
        return (
            f"{requested_tool_name} is not a valid tool, "
            f"try one of [{available_tool_names_str}]."
        )


__all__ = ["InvalidTool", "tool"]

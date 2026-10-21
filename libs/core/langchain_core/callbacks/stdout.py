"""打印输出到标准输出的回调处理器。"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from typing_extensions import override

from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_core.utils import print_text

if TYPE_CHECKING:
    from langchain_core.agents import AgentAction, AgentFinish


class StdOutCallbackHandler(BaseCallbackHandler):
    """打印输出到标准输出的回调处理器。"""

    def __init__(self, color: str | None = None) -> None:
        """初始化回调处理器。

        Args:
            color: 用于文本的颜色。
        """
        self.color = color

    @override
    def on_chain_start(
        self, serialized: dict[str, Any], inputs: dict[str, Any], **kwargs: Any
    ) -> None:
        """打印进入链的信息。

        Args:
            serialized: 序列化的链。
            inputs: 链的输入。
            **kwargs: 额外的关键字参数。
        """
        if "name" in kwargs:
            name = kwargs["name"]
        elif serialized:
            name = serialized.get("name", serialized.get("id", ["<unknown>"])[-1])
        else:
            name = "<unknown>"
        print(f"\n\n\033[1m> Entering new {name} chain...\033[0m")  # noqa: T201

    @override
    def on_chain_end(self, outputs: dict[str, Any], **kwargs: Any) -> None:
        """打印完成链的信息。

        Args:
            outputs: 链的输出。
            **kwargs: 额外的关键字参数。
        """
        print("\n\033[1m> Finished chain.\033[0m")  # noqa: T201

    @override
    def on_agent_action(
        self, action: AgentAction, color: str | None = None, **kwargs: Any
    ) -> Any:
        """智能体动作时运行。

        Args:
            action: 智能体动作。
            color: 用于文本的颜色。
            **kwargs: 额外的关键字参数。
        """
        print_text(action.log, color=color or self.color)

    @override
    def on_tool_end(
        self,
        output: Any,
        color: str | None = None,
        observation_prefix: str | None = None,
        llm_prefix: str | None = None,
        **kwargs: Any,
    ) -> None:
        """如果不是最终动作，打印观察结果。

        Args:
            output: 要打印的输出。
            color: 用于文本的颜色。
            observation_prefix: 观察前缀。
            llm_prefix: LLM 前缀。
            **kwargs: 额外的关键字参数。
        """
        output = str(output)
        if observation_prefix is not None:
            print_text(f"\n{observation_prefix}")
        print_text(output, color=color or self.color)
        if llm_prefix is not None:
            print_text(f"\n{llm_prefix}")

    @override
    def on_text(
        self,
        text: str,
        color: str | None = None,
        end: str = "",
        **kwargs: Any,
    ) -> None:
        """智能体结束时运行。

        Args:
            text: 要打印的文本。
            color: 用于文本的颜色。
            end: 使用的结束字符。
            **kwargs: 额外的关键字参数。
        """
        print_text(text, color=color or self.color, end=end)

    @override
    def on_agent_finish(
        self, finish: AgentFinish, color: str | None = None, **kwargs: Any
    ) -> None:
        """智能体结束时运行。

        Args:
            finish: 智能体结束。
            color: 用于文本的颜色。
            **kwargs: 额外的关键字参数。
        """
        print_text(finish.log, color=color or self.color, end="\n")

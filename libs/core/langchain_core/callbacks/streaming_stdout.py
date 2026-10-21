"""回调处理器在接收到新的 LLM token 时将其流式输出到 stdout。"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Any

from typing_extensions import override

from langchain_core.callbacks.base import BaseCallbackHandler

if TYPE_CHECKING:
    from langchain_core.agents import AgentAction, AgentFinish
    from langchain_core.messages import BaseMessage
    from langchain_core.outputs import LLMResult


class StreamingStdOutCallbackHandler(BaseCallbackHandler):
    """用于流式传输的回调处理器。

    !!! warning "仅适用于支持流式传输的 LLM。"
    """

    def on_llm_start(
        self, serialized: dict[str, Any], prompts: list[str], **kwargs: Any
    ) -> None:
        """LLM 开始运行时调用。

        Args:
            serialized: 序列化的 LLM。
            prompts: 要运行的提示词。
            **kwargs: 额外的关键字参数。
        """

    def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[BaseMessage]],
        **kwargs: Any,
    ) -> None:
        """LLM 开始运行时调用。

        Args:
            serialized: 序列化的 LLM。
            messages: 要运行的消息。
            **kwargs: 额外的关键字参数。
        """

    @override
    def on_llm_new_token(
        self, token: str | list[str | dict[str, Any]], **kwargs: Any
    ) -> None:
        """在新 LLM token 到达时调用。仅在启用流式传输时可用。

        Args:
            token: 新的 token，或一组内容块。
            **kwargs: 额外的关键字参数。
        """
        sys.stdout.write(str(token))
        sys.stdout.flush()

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """LLM 运行结束时调用。

        Args:
            response: LLM 返回的响应。
            **kwargs: 额外的关键字参数。
        """

    def on_llm_error(self, error: BaseException, **kwargs: Any) -> None:
        """LLM 出错时调用。

        Args:
            error: 发生的错误。
            **kwargs: 额外的关键字参数。
        """

    def on_chain_start(
        self, serialized: dict[str, Any], inputs: dict[str, Any], **kwargs: Any
    ) -> None:
        """链开始运行时调用。

        Args:
            serialized: 序列化的链。
            inputs: 链的输入。
            **kwargs: 额外的关键字参数。
        """

    def on_chain_end(self, outputs: dict[str, Any], **kwargs: Any) -> None:
        """链运行结束时调用。

        Args:
            outputs: 链的输出。
            **kwargs: 额外的关键字参数。
        """

    def on_chain_error(self, error: BaseException, **kwargs: Any) -> None:
        """链出错时调用。

        Args:
            error: 发生的错误。
            **kwargs: 额外的关键字参数。
        """

    def on_tool_start(
        self, serialized: dict[str, Any], input_str: str, **kwargs: Any
    ) -> None:
        """工具开始运行时调用。

        Args:
            serialized: 序列化的工具。
            input_str: 输入字符串。
            **kwargs: 额外的关键字参数。
        """

    def on_agent_action(self, action: AgentAction, **kwargs: Any) -> Any:
        """在智能体动作时调用。

        Args:
            action: 智能体动作。
            **kwargs: 额外的关键字参数。
        """

    def on_tool_end(self, output: Any, **kwargs: Any) -> None:
        """工具运行结束时调用。

        Args:
            output: 工具的输出。
            **kwargs: 额外的关键字参数。
        """

    def on_tool_error(self, error: BaseException, **kwargs: Any) -> None:
        """工具出错时调用。

        Args:
            error: 发生的错误。
            **kwargs: 额外的关键字参数。
        """

    def on_text(self, text: str, **kwargs: Any) -> None:
        """在任意文本上运行。

        Args:
            text: 要打印的文本。
            **kwargs: 额外的关键字参数。
        """

    def on_agent_finish(self, finish: AgentFinish, **kwargs: Any) -> None:
        """在智能体结束时调用。

        Args:
            finish: 智能体完成状态。
            **kwargs: 额外的关键字参数。
        """

"""跟踪 `AIMessage.usage_metadata` 的回调处理器。"""

import threading
from collections.abc import Generator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

from typing_extensions import override

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessage
from langchain_core.messages.ai import UsageMetadata, add_usage
from langchain_core.outputs import ChatGeneration, LLMResult
from langchain_core.tracers.context import register_configure_hook


class UsageMetadataCallbackHandler(BaseCallbackHandler):
    """跟踪 `AIMessage.usage_metadata` 的回调处理器。

    Example:
        ```python
        from langchain.chat_models import init_chat_model
        from langchain_core.callbacks import UsageMetadataCallbackHandler

        llm_1 = init_chat_model(model="openai:gpt-5.5")
        llm_2 = init_chat_model(model="anthropic:claude-haiku-4-5-20251001")

        callback = UsageMetadataCallbackHandler()
        result_1 = llm_1.invoke("Hello", config={"callbacks": [callback]})
        result_2 = llm_2.invoke("Hello", config={"callbacks": [callback]})
        callback.usage_metadata
        ```

    !!! version-added "在 `langchain-core` 0.3.49 中新增"

    """

    def __init__(self) -> None:
        """初始化 `UsageMetadataCallbackHandler`。"""
        super().__init__()
        self._lock = threading.Lock()
        self.usage_metadata: dict[str, UsageMetadata] = {}

    @override
    def __repr__(self) -> str:
        return str(self.usage_metadata)

    @override
    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """收集 token 使用量。"""
                # 检查是否存在 usage_metadata（langchain-core >= 0.2.2）
        try:
            generation = response.generations[0][0]
        except IndexError:
            generation = None

        usage_metadata = None
        model_name = None
        if isinstance(generation, ChatGeneration):
            try:
                message = generation.message
                if isinstance(message, AIMessage):
                    usage_metadata = message.usage_metadata
                    model_name = message.response_metadata.get("model_name")
            except AttributeError:
                pass

                # 在锁保护下更新共享状态
        if usage_metadata and model_name:
            with self._lock:
                if model_name not in self.usage_metadata:
                    self.usage_metadata[model_name] = usage_metadata
                else:
                    self.usage_metadata[model_name] = add_usage(
                        self.usage_metadata[model_name], usage_metadata
                    )


@contextmanager
def get_usage_metadata_callback(
    name: str = "usage_metadata_callback",
) -> Generator[UsageMetadataCallbackHandler, None, None]:
    """获取使用情况元数据回调。

    使用上下文管理器跟踪跨聊天模型调用的使用情况元数据，方法为
    [`AIMessage.usage_metadata`][langchain.messages.AIMessage.usage_metadata]。

    Args:
        name: 上下文变量的名称。

    Yields:
        使用情况元数据回调。

    Example:
        ```python
        from langchain.chat_models import init_chat_model
        from langchain_core.callbacks import get_usage_metadata_callback

        llm_1 = init_chat_model(model="openai:gpt-5.5")
        llm_2 = init_chat_model(model="anthropic:claude-haiku-4-5-20251001")

        with get_usage_metadata_callback() as cb:
            llm_1.invoke("Hello")
            llm_2.invoke("Hello")
            print(cb.usage_metadata)
        ```

    !!! version-added "在 `langchain-core` 0.3.49 中新增"

    """
    usage_metadata_callback_var: ContextVar[UsageMetadataCallbackHandler | None] = (
        ContextVar(name, default=None)
    )
    register_configure_hook(usage_metadata_callback_var, inheritable=True)
    cb = UsageMetadataCallbackHandler()
    token = usage_metadata_callback_var.set(cb)
    try:
        yield cb
    finally:
                # 始终清除上下文变量，包括在 with 块抛出异常时。
                # 若无 finally，块外的模型调用会持续累积到 cb 中
                # (参见 #38989)
        usage_metadata_callback_var.reset(token)

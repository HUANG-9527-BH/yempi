"""聊天结果的模式。"""

from typing import Any

from pydantic import BaseModel

from langchain_core.outputs.chat_generation import ChatGeneration


class ChatResult(BaseModel):
    """用于表示单个提示词调用聊天模型的结果。

    此容器由某些聊天模型实现内部使用，最终将映射到更通用的 `LLMResult` 对象，然后投影为
    `AIMessage` 对象。

    LangChain users working with chat models will usually access information via
    `AIMessage` (returned from runnable interfaces) or `LLMResult` (available via
    callbacks). Please refer the `AIMessage` and `LLMResult` schema documentation for
    more information.
    """

    generations: list[ChatGeneration]
    """聊天生成结果的列表。

    Generations is a list to allow for multiple candidate generations for a single
    input prompt.
    """

    llm_output: dict[str, Any] | None = None
    """用于任意模型提供商特定的输出。

    此字典是自由格式的字典，可包含提供商希望返回的任何信息。它未标准化，键可能因提供商和时间而异。

    Users should generally avoid relying on this field and instead rely on accessing
    relevant information from standardized fields present in `AIMessage`.
    """

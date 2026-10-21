"""在 LangChain 中使用 [聊天模型](https://docs.langchain.com/oss/python/langchain/models) 的入口点。"""  # noqa: E501

from langchain_core.language_models import BaseChatModel

from langchain.chat_models.base import init_chat_model

__all__ = ["BaseChatModel", "init_chat_model"]

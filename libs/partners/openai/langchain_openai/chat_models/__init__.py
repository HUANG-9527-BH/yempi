"""OpenAI 聊天模型模块。"""

from langchain_openai.chat_models.azure import AzureChatOpenAI
from langchain_openai.chat_models.base import ChatOpenAI

__all__ = ["AzureChatOpenAI", "ChatOpenAI"]

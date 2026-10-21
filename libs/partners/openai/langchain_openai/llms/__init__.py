"""OpenAI 大型语言模型模块。聊天模型位于 `chat_models/` 目录中。"""

from langchain_openai.llms.azure import AzureOpenAI
from langchain_openai.llms.base import OpenAI

__all__ = ["AzureOpenAI", "OpenAI"]

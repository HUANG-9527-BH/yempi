"""**聊天模型** 是语言模型的一种变体。

虽然聊天模型在内部使用语言模型，但其暴露的接口略有不同。它们不提供 "文本输入，文本输出"
的 API，而是以 "聊天消息" 作为输入和输出。
"""

import warnings

from langchain_core._api import LangChainDeprecationWarning

from langchain_classic._api.interactive_env import is_interactive_env
from langchain_classic.chat_models.base import init_chat_model


def __getattr__(name: str) -> None:
    from langchain_community import chat_models

        # 如果不在交互式环境中，发出警告。
    if not is_interactive_env():
        warnings.warn(
            "Importing chat models from langchain is deprecated. Importing from "
            "langchain will no longer be supported as of langchain==0.2.0. "
            "Please import from langchain-community instead:\n\n"
            f"`from langchain_community.chat_models import {name}`.\n\n"
            "To install langchain-community run `pip install -U langchain-community`.",
            stacklevel=2,
            category=LangChainDeprecationWarning,
        )

    return getattr(chat_models, name)


__all__ = [
    "AzureChatOpenAI",
    "BedrockChat",
    "ChatAnthropic",
    "ChatAnyscale",
    "ChatBaichuan",
    "ChatCohere",
    "ChatDatabricks",
    "ChatEverlyAI",
    "ChatFireworks",
    "ChatGooglePalm",
    "ChatHunyuan",
    "ChatJavelinAIGateway",
    "ChatKonko",
    "ChatLiteLLM",
    "ChatMLflowAIGateway",
    "ChatMlflow",
    "ChatOllama",
    "ChatOpenAI",
    "ChatVertexAI",
    "ChatYandexGPT",
    "ErnieBotChat",
    "FakeListChatModel",
    "GigaChat",
    "HumanInputChatModel",
    "JinaChat",
    "MiniMaxChat",
    "PaiEasChatEndpoint",
    "PromptLayerChatOpenAI",
    "QianfanChatEndpoint",
    "VolcEngineMaasChat",
    "init_chat_model",
]

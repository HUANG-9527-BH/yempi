"""智能体类型的模块定义及其对应的智能体。"""

from enum import Enum

from langchain_core._api import deprecated

from langchain_classic._api.deprecation import AGENT_DEPRECATION_WARNING


@deprecated(
    "0.1.0",
    message=AGENT_DEPRECATION_WARNING,
    removal="2.0.0",
)
class AgentType(str, Enum):
    """智能体类型的枚举。"""

    ZERO_SHOT_REACT_DESCRIPTION = "zero-shot-react-description"
    """一个在行动前进行推理步骤的零样本智能体。"""

    REACT_DOCSTORE = "react-docstore"
    """一个在行动前进行推理步骤的零样本智能体。

    此智能体可访问文档存储，以便查找回答问题的相关信息。
    """

    SELF_ASK_WITH_SEARCH = "self-ask-with-search"
    """一个将复杂问题分解为一系列更简单问题的智能体。

    此智能体使用搜索工具来查找更简单问题的答案，以回答原始的复杂问题。
    """
    CONVERSATIONAL_REACT_DESCRIPTION = "conversational-react-description"
    CHAT_ZERO_SHOT_REACT_DESCRIPTION = "chat-zero-shot-react-description"
    """一个在行动前进行推理步骤的零样本智能体。

    此智能体设计用于与
    """

    CHAT_CONVERSATIONAL_REACT_DESCRIPTION = "chat-conversational-react-description"

    STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION = (
        "structured-chat-zero-shot-react-description"
    )
    """一个针对聊天模型优化的零样本 REACT 智能体。

    此智能体能够调用具有多个输入的工具。
    """

    OPENAI_FUNCTIONS = "openai-functions"
    """一个针对使用 OpenAI 函数优化的智能体。"""

    OPENAI_MULTI_FUNCTIONS = "openai-multi-functions"

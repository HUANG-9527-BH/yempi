from abc import ABC, abstractmethod
from collections.abc import Callable

from langchain_core.language_models import BaseLanguageModel
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.language_models.llms import BaseLLM
from langchain_core.prompts import BasePromptTemplate
from pydantic import BaseModel, Field


class BasePromptSelector(BaseModel, ABC):
    """提示词选择器的基类。"""

    @abstractmethod
    def get_prompt(self, llm: BaseLanguageModel) -> BasePromptTemplate:
        """获取语言模型的默认提示词。"""


class ConditionalPromptSelector(BasePromptSelector):
    """通过条件判断来选择提示词的集合。"""

    default_prompt: BasePromptTemplate
    """如果没有条件匹配，则使用的默认提示词。"""
    conditionals: list[
        tuple[Callable[[BaseLanguageModel], bool], BasePromptTemplate]
    ] = Field(default_factory=list)
    """条件列表及对应的提示词，当条件匹配时使用。"""

    def get_prompt(self, llm: BaseLanguageModel) -> BasePromptTemplate:
        """获取语言模型的默认提示词。

        Args:
            llm: 要获取提示词的语言模型。

        Returns:
            用于该语言模型的提示词。
        """
        for condition, prompt in self.conditionals:
            if condition(llm):
                return prompt
        return self.default_prompt


def is_llm(llm: BaseLanguageModel) -> bool:
    """检查语言模型是否为 LLM。

    Args:
        llm: 要检查的语言模型。

    Returns:
        如果语言模型是 BaseLLM 模型，则返回 `True`，否则返回 `False`。
    """
    return isinstance(llm, BaseLLM)


def is_chat_model(llm: BaseLanguageModel) -> bool:
    """检查语言模型是否为聊天模型。

    Args:
        llm: 要检查的语言模型。

    Returns:
        如果语言模型是 BaseChatModel 模型，则返回 `True`，否则返回 `False`。
    """
    return isinstance(llm, BaseChatModel)

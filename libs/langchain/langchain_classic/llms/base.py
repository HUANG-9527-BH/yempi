"""此模块提供核心语言模型类的向后兼容导出。

These classes are re-exported for compatibility with older versions of LangChain
and allow users to import language model interfaces from a stable path.

导出：
    - LLM: 所有 LLM 的抽象基类
    - BaseLLM: 已弃用或用于遗留 LLM 的基础类
    - BaseLanguageModel: 核心语言模型实现的基础类
"""

from langchain_core.language_models import BaseLanguageModel
from langchain_core.language_models.llms import LLM, BaseLLM

__all__ = [
    "LLM",
    "BaseLLM",
    "BaseLanguageModel",
]

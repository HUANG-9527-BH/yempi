"""语言模型的可选缓存层。

不同于基于提供方的 [提示词缓存](https://docs.langchain.com/oss/python/langchain/models#prompt-caching)。

!!! warning "测试版功能"

    这是一个测试版功能。请在将实验性代码部署到生产环境前，务必采取适当的预防措施。

缓存有两点用处：

1. It can save you money by reducing the number of API calls you make to the LLM
    provider if you're often requesting the same completion multiple times.
2. It can speed up your application by reducing the number of API calls you make to the
    LLM provider.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any

from typing_extensions import override

from langchain_core.outputs import Generation
from langchain_core.runnables import run_in_executor

RETURN_VAL_TYPE = Sequence[Generation]


class BaseCache(ABC):
    """语言模型和聊天模型缓存层的接口。

    缓存接口包含以下方法：

    - lookup: 根据提示词和 `llm_string` 查找对应值。
    - update: 根据提示词和 `llm_string` 更新缓存。
    - clear: 清空缓存。

    此外，缓存接口还提供了每个方法的异步版本。

    The default implementation of the async methods is to run the synchronous
    method in an executor. It's recommended to override the async methods
    and provide async implementations to avoid unnecessary overhead.
    """

    @abstractmethod
    def lookup(self, prompt: str, llm_string: str) -> RETURN_VAL_TYPE | None:
        """根据 `prompt` 和 `llm_string` 进行查找。

        A cache implementation is expected to generate a key from the 2-tuple
        of `prompt` and `llm_string` (e.g., by concatenating them with a delimiter).

        Args:
            prompt: 提示词的字符串表示形式。

                对于聊天模型，提示词是提示词非平凡地序列化为语言模型格式的结果。
                llm_string: LLM 配置的字符串表示形式。

                用于捕获 LLM 的调用参数（例如模型名称、温度、停止 token、最大 token
                数等）。

                这些调用参数会被序列化为字符串表示形式。

        Returns:
            缓存未命中时返回 `None`。缓存命中时返回缓存的值。
                缓存的值是 `Generation`（或其子类）的列表。
        """

    @abstractmethod
    def update(self, prompt: str, llm_string: str, return_val: RETURN_VAL_TYPE) -> None:
        """根据 `prompt` 和 `llm_string` 更新缓存。

        The `prompt` and `llm_string` are used to generate a key for the cache. The key
        should match that of the lookup method.

        Args:
            prompt: 提示词的字符串表示形式。

                对于聊天模型，提示词是提示词非平凡地序列化为语言模型格式的结果。
                llm_string: LLM 配置的字符串表示形式。

                用于捕获 LLM 的调用参数（例如模型名称、温度、停止 token、最大 token
                数等）。

                这些调用参数会被序列化为字符串表示形式。 return_val: 要缓存的值。

                值是 `Generation`（或其子类）的列表。
        """

    @abstractmethod
    def clear(self, **kwargs: Any) -> None:
        """清空缓存，可接受额外的关键字参数。"""

    async def alookup(self, prompt: str, llm_string: str) -> RETURN_VAL_TYPE | None:
        """基于 `prompt` 和 `llm_string` 的异步查找。

        A cache implementation is expected to generate a key from the 2-tuple
        of `prompt` and `llm_string` (e.g., by concatenating them with a delimiter).

        Args:
            prompt: 提示词的字符串表示形式。

                对于聊天模型，提示词是将提示词非平凡地序列化为语言模型的形式。
                llm_string: LLM 配置的字符串表示形式。

                用于捕获 LLM 的调用参数（例如，模型名称、温度、停止 token、最大 token
                数等）。

                这些调用参数被序列化为字符串表示形式。

        Returns:
            缓存未命中时返回 `None`。缓存命中时返回缓存值。
                缓存值是一个 `Generation`（或子类）的列表。
        """
        return await run_in_executor(None, self.lookup, prompt, llm_string)

    async def aupdate(
        self, prompt: str, llm_string: str, return_val: RETURN_VAL_TYPE
    ) -> None:
        """基于 `prompt` 和 `llm_string` 的异步更新缓存。

        使用提示词和 llm_string 生成缓存的键。
        该键应与查找方法生成的键匹配。

        Args:
            prompt: 提示词的字符串表示形式。

                对于聊天模型，提示词是将提示词非平凡地序列化为语言模型的形式。
                llm_string: LLM 配置的字符串表示形式。

                用于捕获 LLM 的调用参数（例如，模型名称、温度、停止 token、最大 token
                数等）。 这些调用参数被序列化为字符串表示形式。 return_val:
                要缓存的值。该值是一个 `Generation`（或子类）的列表。

                异步清除缓存，可接受额外的关键字参数。
        """
        return await run_in_executor(None, self.update, prompt, llm_string, return_val)

    async def aclear(self, **kwargs: Any) -> None:
        """使用空缓存进行初始化。"""
        return await run_in_executor(None, self.clear, **kwargs)


class InMemoryCache(BaseCache):
    """将数据存储在内存中的缓存。

    Example:
        ```python
        from langchain_core.caches import InMemoryCache
        from langchain_core.outputs import Generation

        # Initialize cache
        cache = InMemoryCache()

        # Update cache
        cache.update(
            prompt="What is the capital of France?",
            llm_string="model='gpt-5.4-mini',
            return_val=[Generation(text="Paris")],
        )

        # Lookup cache
        result = cache.lookup(
            prompt="What is the capital of France?",
            llm_string="model='gpt-5.4-mini',
        )
        # result is [Generation(text="Paris")]
        ```
    """

    def __init__(self, *, maxsize: int | None = None) -> None:
        """maxsize: 缓存中最多存储的项数。

        Args:
            maxsize: 如果为 `None`，则缓存无最大大小限制。

                如果缓存超过最大大小，最旧的项将被移除。

                ValueError: 如果 `maxsize` 小于或等于 `0`。

        Raises:
            ValueError: 基于 `prompt` 和 `llm_string` 的查找。
        """
        self._cache: dict[tuple[str, str], RETURN_VAL_TYPE] = {}
        if maxsize is not None and maxsize <= 0:
            msg = "maxsize must be greater than 0"
            raise ValueError(msg)
        self._maxsize = maxsize

    def lookup(self, prompt: str, llm_string: str) -> RETURN_VAL_TYPE | None:
        """prompt: 提示词的字符串表示形式。

        Args:
            对于聊天模型，提示词是将提示词非平凡地序列化为语言模型的形式。 llm_string:
            LLM 配置的字符串表示形式。

                缓存未命中时返回 `None`。缓存命中时返回缓存值。

        Returns:
            基于 `prompt` 和 `llm_string` 的更新缓存。
        """
        return self._cache.get((prompt, llm_string), None)

    def update(self, prompt: str, llm_string: str, return_val: RETURN_VAL_TYPE) -> None:
        """prompt: 提示词的字符串表示形式。

        Args:
            对于聊天模型，提示词是将提示词非平凡地序列化为语言模型的形式。 llm_string:
            LLM 配置的字符串表示形式。 return_val: 要缓存的值。

                该值是一个 `Generation`（或子类）的列表。

                清除缓存。
        """
        if self._maxsize is not None and len(self._cache) == self._maxsize:
            del self._cache[next(iter(self._cache))]
        self._cache[prompt, llm_string] = return_val

    @override
    def clear(self, **kwargs: Any) -> None:
        """清除缓存。"""
        self._cache = {}

    async def alookup(self, prompt: str, llm_string: str) -> RETURN_VAL_TYPE | None:
        """基于 `prompt` 和 `llm_string` 的异步查找。

        Args:
            prompt: 提示词的字符串表示。

                对于聊天模型，提示词是提示词到语言模型的非平凡
                序列化表示。
            llm_string: LLM 配置的字符串表示。

        Returns:
            缓存未命中时返回 `None`。缓存命中时返回缓存值。
        """
        return self.lookup(prompt, llm_string)

    async def aupdate(
        self, prompt: str, llm_string: str, return_val: RETURN_VAL_TYPE
    ) -> None:
        """基于 `prompt` 和 `llm_string` 的异步更新缓存。

        Args:
            prompt: 提示词的字符串表示。

                对于聊天模型，提示词是提示词到语言模型的非平凡
                序列化表示。
            llm_string: LLM 配置的字符串表示。
            return_val: 要缓存的值。该值是一个 `Generation`
                （或子类）列表。
        """
        self.update(prompt, llm_string, return_val)

    @override
    async def aclear(self, **kwargs: Any) -> None:
        """异步清除缓存。"""
        self.clear()

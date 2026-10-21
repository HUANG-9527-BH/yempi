"""用于测试目的的假 LLM。"""

import asyncio
import time
from collections.abc import AsyncIterator, Iterator, Mapping
from typing import Any

from typing_extensions import override

from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models import LanguageModelInput
from langchain_core.language_models.llms import LLM
from langchain_core.runnables import RunnableConfig


class FakeListLLM(LLM):
    """用于测试目的的假 LLM。"""

    responses: list[str]
    """按顺序返回的响应列表。"""
        # 此参数应从 FakeListLLM 中移除，因为
        # 它仅被子类使用。
    sleep: float | None = None
    """每次响应之间的睡眠时间（秒）。

    FakeListLLM 忽略此参数，但子类会使用。
    """
    i: int = 0
    """每次模型调用后内部递增。

    主要用于测试目的。
    """

    @property
    @override
    def _llm_type(self) -> str:
        """llm 的返回类型。"""
        return "fake-list"

    @override
    def _call(
        self,
        prompt: str,
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> str:
        """返回下一个响应。"""
        response = self.responses[self.i]
        if self.i < len(self.responses) - 1:
            self.i += 1
        else:
            self.i = 0
        return response

    @override
    async def _acall(
        self,
        prompt: str,
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> str:
        """返回下一个响应。"""
        response = self.responses[self.i]
        if self.i < len(self.responses) - 1:
            self.i += 1
        else:
            self.i = 0
        return response

    @property
    @override
    def _identifying_params(self) -> Mapping[str, Any]:
        return {"responses": self.responses}


class FakeListLLMError(Exception):
    """用于测试目的的假错误。"""


class FakeStreamingListLLM(FakeListLLM):
    """用于测试目的的假流式列表 LLM。

    一个将按顺序返回列表中响应的 LLM。

    此模型还支持在流式实现中可选地在连续块之间休眠。
    """

    error_on_chunk_number: int | None = None
    """如果设置，将在指定的块编号处抛出异常。"""

    @override
    def stream(
        self,
        input: LanguageModelInput,
        config: RunnableConfig | None = None,
        *,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> Iterator[str]:
        result = self.invoke(input, config)
        for i_c, c in enumerate(result):
            if self.sleep is not None:
                time.sleep(self.sleep)

            if (
                self.error_on_chunk_number is not None
                and i_c == self.error_on_chunk_number
            ):
                raise FakeListLLMError
            yield c

    @override
    async def astream(
        self,
        input: LanguageModelInput,
        config: RunnableConfig | None = None,
        *,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        result = await self.ainvoke(input, config)
        for i_c, c in enumerate(result):
            if self.sleep is not None:
                await asyncio.sleep(self.sleep)

            if (
                self.error_on_chunk_number is not None
                and i_c == self.error_on_chunk_number
            ):
                raise FakeListLLMError
            yield c

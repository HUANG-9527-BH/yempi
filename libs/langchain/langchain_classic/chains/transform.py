"""运行任意 Python 函数的链。"""

import functools
import logging
from collections.abc import Awaitable, Callable
from typing import Any

from langchain_core.callbacks import (
    AsyncCallbackManagerForChainRun,
    CallbackManagerForChainRun,
)
from pydantic import Field
from typing_extensions import override

from langchain_classic.chains.base import Chain

logger = logging.getLogger(__name__)


class TransformChain(Chain):
    """转换链输出的链。

    Example:
        ```python
        from langchain_classic.chains import TransformChain
        transform_chain = TransformChain(input_variables=["text"],
         output_variables["entities"], transform=func())

        ```
    """

    input_variables: list[str]
    """转换函数输入字典中预期的键。"""
    output_variables: list[str]
    """转换函数输出字典中返回的键。"""
    transform_cb: Callable[[dict[str, str]], dict[str, str]] = Field(alias="transform")
    """转换函数。"""
    atransform_cb: Callable[[dict[str, Any]], Awaitable[dict[str, Any]]] | None = Field(
        None, alias="atransform"
    )
    """异步协程转换函数。"""

    @staticmethod
    @functools.lru_cache
    def _log_once(msg: str) -> None:
        """记录一条消息一次。"""
        logger.warning(msg)

    @property
    def input_keys(self) -> list[str]:
        """期望的输入键。"""
        return self.input_variables

    @property
    def output_keys(self) -> list[str]:
        """返回的输出键。"""
        return self.output_variables

    @override
    def _call(
        self,
        inputs: dict[str, str],
        run_manager: CallbackManagerForChainRun | None = None,
    ) -> dict[str, str]:
        return self.transform_cb(inputs)

    @override
    async def _acall(
        self,
        inputs: dict[str, Any],
        run_manager: AsyncCallbackManagerForChainRun | None = None,
    ) -> dict[str, Any]:
        if self.atransform_cb is not None:
            return await self.atransform_cb(inputs)
        self._log_once(
            "TransformChain's atransform is not provided, falling"
            " back to synchronous transform",
        )
        return self.transform_cb(inputs)

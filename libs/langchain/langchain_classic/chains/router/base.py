"""链路由的基础类。"""

from __future__ import annotations

from abc import ABC
from collections.abc import Mapping
from typing import Any, NamedTuple

from langchain_core.callbacks import (
    AsyncCallbackManagerForChainRun,
    CallbackManagerForChainRun,
    Callbacks,
)
from pydantic import ConfigDict
from typing_extensions import override

from langchain_classic.chains.base import Chain


class Route(NamedTuple):
    """指向目标链的路由。"""

    destination: str | None
    next_inputs: dict[str, Any]


class RouterChain(Chain, ABC):
    """输出目标链名称及其输入的链。"""

    @property
    @override
    def output_keys(self) -> list[str]:
        return ["destination", "next_inputs"]

    def route(self, inputs: dict[str, Any], callbacks: Callbacks = None) -> Route:
        """將輸入路由到目標鏈。

        Args:
            inputs: 鏈的輸入
            callbacks: 用於鏈的回調

        Returns:
            一個 Route 物件
        """
        result = self(inputs, callbacks=callbacks)
        return Route(result["destination"], result["next_inputs"])

    async def aroute(
        self,
        inputs: dict[str, Any],
        callbacks: Callbacks = None,
    ) -> Route:
        """將輸入路由到目標鏈。

        Args:
            inputs: 鏈的輸入
            callbacks: 用於鏈的回調

        Returns:
            一個 Route 物件
        """
        result = await self.acall(inputs, callbacks=callbacks)
        return Route(result["destination"], result["next_inputs"])


class MultiRouteChain(Chain):
    """使用单个链将输入路由到多个候选链中的一个。"""

    router_chain: RouterChain
    """将输入路由到目标链的链。"""
    destination_chains: Mapping[str, Chain]
    """对输入返回最终答案的链。"""
    default_chain: Chain
    """当没有合适的目標鏈時使用的預設鏈。"""
    silent_errors: bool = False
    """如果為 `True`，當提供無效的目的地名稱時，使用 default_chain。"""

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="forbid",
    )

    @property
    def input_keys(self) -> list[str]:
        """將是路由器鏈提示詞所期望的任何鍵。"""
        return self.router_chain.input_keys

    @property
    def output_keys(self) -> list[str]:
        """總是返回 text 鍵。"""
        return []

    def _call(
        self,
        inputs: dict[str, Any],
        run_manager: CallbackManagerForChainRun | None = None,
    ) -> dict[str, Any]:
        _run_manager = run_manager or CallbackManagerForChainRun.get_noop_manager()
        callbacks = _run_manager.get_child()
        route = self.router_chain.route(inputs, callbacks=callbacks)

        _run_manager.on_text(
            str(route.destination) + ": " + str(route.next_inputs),
            verbose=self.verbose,
        )
        if not route.destination:
            return self.default_chain(route.next_inputs, callbacks=callbacks)
        if route.destination in self.destination_chains:
            return self.destination_chains[route.destination](
                route.next_inputs,
                callbacks=callbacks,
            )
        if self.silent_errors:
            return self.default_chain(route.next_inputs, callbacks=callbacks)
        msg = f"Received invalid destination chain name '{route.destination}'"
        raise ValueError(msg)

    async def _acall(
        self,
        inputs: dict[str, Any],
        run_manager: AsyncCallbackManagerForChainRun | None = None,
    ) -> dict[str, Any]:
        _run_manager = run_manager or AsyncCallbackManagerForChainRun.get_noop_manager()
        callbacks = _run_manager.get_child()
        route = await self.router_chain.aroute(inputs, callbacks=callbacks)

        await _run_manager.on_text(
            str(route.destination) + ": " + str(route.next_inputs),
            verbose=self.verbose,
        )
        if not route.destination:
            return await self.default_chain.acall(
                route.next_inputs,
                callbacks=callbacks,
            )
        if route.destination in self.destination_chains:
            return await self.destination_chains[route.destination].acall(
                route.next_inputs,
                callbacks=callbacks,
            )
        if self.silent_errors:
            return await self.default_chain.acall(
                route.next_inputs,
                callbacks=callbacks,
            )
        msg = f"Received invalid destination chain name '{route.destination}'"
        raise ValueError(msg)

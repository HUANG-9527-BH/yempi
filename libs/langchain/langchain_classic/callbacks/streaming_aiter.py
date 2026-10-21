from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any, Literal, cast

from langchain_core.callbacks import AsyncCallbackHandler
from langchain_core.outputs import LLMResult
from typing_extensions import override

# TODO: 如果两个 LLM 运行并行使用，此方法将无法按预期工作


class AsyncIteratorCallbackHandler(AsyncCallbackHandler):
    """返回异步迭代器的回调处理器。"""

    queue: asyncio.Queue[str]

    done: asyncio.Event

    @property
    def always_verbose(self) -> bool:
        """始终处于详细模式。"""
        return True

    def __init__(self) -> None:
        """实例化 AsyncIteratorCallbackHandler。"""
        self.queue = asyncio.Queue()
        self.done = asyncio.Event()

    @override
    async def on_llm_start(
        self,
        serialized: dict[str, Any],
        prompts: list[str],
        **kwargs: Any,
    ) -> None:
                # 如果连续调用两次，这会重置状态
        self.done.clear()

    @override
    async def on_llm_new_token(
        self, token: str | list[str | dict[str, Any]], **kwargs: Any
    ) -> None:
        token_str = token if isinstance(token, str) else str(token)
        if token_str != "":
            self.queue.put_nowait(token_str)

    @override
    async def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        self.done.set()

    @override
    async def on_llm_error(self, error: BaseException, **kwargs: Any) -> None:
        self.done.set()

        # TODO: 实现其他方法

    async def aiter(self) -> AsyncIterator[str]:
        """生成 token 的异步迭代器。"""
        while not self.queue.empty() or not self.done.is_set():
                        # 等待队列中的下一个 token，
                        # 但如果 done 事件已设置，则停止等待
            done, other = await asyncio.wait(
                [
                                        # NOTE: 如果在此处添加其他任务，请更新下方代码，
                                        # 因为当前代码假设每个 set 中恰好只有一个任务
                    asyncio.ensure_future(self.queue.get()),
                    asyncio.ensure_future(self.done.wait()),
                ],
                return_when=asyncio.FIRST_COMPLETED,
            )

                        # 取消其他任务
            if other:
                other.pop().cancel()

                        # 提取第一个完成任务的值
            token_or_done = cast("str | Literal[True]", done.pop().result())

                        # 如果提取的值为布尔值 True，则表示 done 事件已设置
            if token_or_done is True:
                break

                        # 否则，提取的值为 token，我们将其产出
            yield token_or_done

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncIterator, Iterator
from typing import (
    TYPE_CHECKING,
    Any,
)
from uuid import UUID

from langchain_core.agents import (
    AgentAction,
    AgentFinish,
    AgentStep,
)
from langchain_core.callbacks import (
    AsyncCallbackManager,
    AsyncCallbackManagerForChainRun,
    CallbackManager,
    CallbackManagerForChainRun,
    Callbacks,
)
from langchain_core.load.dump import dumpd
from langchain_core.outputs import RunInfo
from langchain_core.runnables.utils import AddableDict
from langchain_core.tools import BaseTool
from langchain_core.utils.input import get_color_mapping

from langchain_classic.schema import RUN_KEY
from langchain_classic.utilities.asyncio import asyncio_timeout

if TYPE_CHECKING:
    from langchain_classic.agents.agent import AgentExecutor, NextStepOutput

logger = logging.getLogger(__name__)


class AgentExecutorIterator:
    """智能体执行器的迭代器。"""

    def __init__(
        self,
        agent_executor: AgentExecutor,
        inputs: Any,
        callbacks: Callbacks = None,
        *,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        run_name: str | None = None,
        run_id: UUID | None = None,
        include_run_info: bool = False,
        yield_actions: bool = False,
    ):
        """初始化 `AgentExecutorIterator`。

        使用给定的 `AgentExecutor`、
        inputs 和可选的回调来初始化 `AgentExecutorIterator`。

        Args:
            agent_executor: 要迭代的 `AgentExecutor`。
            inputs: 传递给 `AgentExecutor` 的输入。
            callbacks: 迭代过程中使用的回调。
            tags: 迭代过程中使用的标签。
            metadata: 迭代过程中使用的元数据。
            run_name: 运行的名称。
            run_id: 运行的 ID。
            include_run_info: 是否在输出中包含运行信息。
            yield_actions: 是否在生成时产出动作。
        """
        self._agent_executor = agent_executor
        self.inputs = inputs
        self.callbacks = callbacks
        self.tags = tags
        self.metadata = metadata
        self.run_name = run_name
        self.run_id = run_id
        self.include_run_info = include_run_info
        self.yield_actions = yield_actions
        self.reset()

    _inputs: dict[str, str]
    callbacks: Callbacks
    tags: list[str] | None
    metadata: dict[str, Any] | None
    run_name: str | None
    run_id: UUID | None
    include_run_info: bool
    yield_actions: bool

    @property
    def inputs(self) -> dict[str, str]:
        """传递给 `AgentExecutor` 的输入。"""
        return self._inputs

    @inputs.setter
    def inputs(self, inputs: Any) -> None:
        self._inputs = self.agent_executor.prep_inputs(inputs)

    @property
    def agent_executor(self) -> AgentExecutor:
        """要迭代的 `AgentExecutor`。"""
        return self._agent_executor

    @agent_executor.setter
    def agent_executor(self, agent_executor: AgentExecutor) -> None:
        self._agent_executor = agent_executor
                # 强制重新准备输入，以防 agent_executor 的 prep_inputs 函数已更改
        self.inputs = self.inputs

    @property
    def name_to_tool_map(self) -> dict[str, BaseTool]:
        """工具名称到工具的映射。"""
        return {tool.name: tool for tool in self.agent_executor.tools}

    @property
    def color_mapping(self) -> dict[str, str]:
        """工具名称到颜色的映射。"""
        return get_color_mapping(
            [tool.name for tool in self.agent_executor.tools],
            excluded_colors=["green", "red"],
        )

    def reset(self) -> None:
        """将迭代器重置为其初始状态。

        将迭代器重置为其初始状态，清除中间步骤、
        迭代次数和已用时间。
        """
        logger.debug("(Re)setting AgentExecutorIterator to fresh state")
        self.intermediate_steps: list[tuple[AgentAction, str]] = []
        self.iterations = 0
                # 也许最好在第一次 __anext__ 调用时启动这些？
        self.time_elapsed = 0.0
        self.start_time = time.time()

    def update_iterations(self) -> None:
        """增加迭代次数并更新已用时间。"""
        self.iterations += 1
        self.time_elapsed = time.time() - self.start_time
        logger.debug(
            "Agent Iterations: %s (%.2fs elapsed)",
            self.iterations,
            self.time_elapsed,
        )

    def make_final_outputs(
        self,
        outputs: dict[str, Any],
        run_manager: CallbackManagerForChainRun | AsyncCallbackManagerForChainRun,
    ) -> AddableDict:
        """为迭代器生成最终输出。

        Args:
            outputs: 来自智能体执行器的输出。
            run_manager: 用于回调的运行管理器。
        """
                # 按设计允许访问中间步骤在迭代器中，
                # 因此仅返回输出也应始终为真。

        prepared_outputs = AddableDict(
            self.agent_executor.prep_outputs(
                self.inputs,
                outputs,
                return_only_outputs=True,
            ),
        )
        if self.include_run_info:
            prepared_outputs[RUN_KEY] = RunInfo(run_id=run_manager.run_id)
        return prepared_outputs

    def __iter__(self: AgentExecutorIterator) -> Iterator[AddableDict]:
        """为 `AgentExecutor` 创建一个异步迭代器。"""
        logger.debug("Initialising AgentExecutorIterator")
        self.reset()
        callback_manager = CallbackManager.configure(
            self.callbacks,
            self.agent_executor.callbacks,
            self.agent_executor.verbose,
            self.tags,
            self.agent_executor.tags,
            self.metadata,
            self.agent_executor.metadata,
        )
        run_manager = callback_manager.on_chain_start(
            dumpd(self.agent_executor),
            self.inputs,
            self.run_id,
            name=self.run_name,
        )
        try:
            while self.agent_executor._should_continue(  # noqa: SLF001
                self.iterations,
                self.time_elapsed,
            ):
                                # 执行下一步：这会规划下一步动作，执行它，
                                # 在生成时产出动作和观察结果
                next_step_seq: NextStepOutput = []
                for chunk in self.agent_executor._iter_next_step(  # noqa: SLF001
                    self.name_to_tool_map,
                    self.color_mapping,
                    self.inputs,
                    self.intermediate_steps,
                    run_manager,
                ):
                    next_step_seq.append(chunk)
                                        # 如果我们在产出动作，就立即产出它们
                                        # 不要产出 AgentFinish，这将在下面处理
                    if self.yield_actions:
                        if isinstance(chunk, AgentAction):
                            yield AddableDict(actions=[chunk], messages=chunk.messages)
                        elif isinstance(chunk, AgentStep):
                            yield AddableDict(steps=[chunk], messages=chunk.messages)

                                # 将迭代器输出转换为 _process_next_step_output 处理的格式
                next_step = self.agent_executor._consume_next_step(next_step_seq)  # noqa: SLF001
                                # 更新迭代次数和已用时间
                self.update_iterations()
                                # 决定这是否是最终输出
                output = self._process_next_step_output(next_step, run_manager)
                is_final = "intermediate_step" not in output
                                # 始终产出最终输出
                                # 为了向后兼容，如果不产出动作，则产出整数输出
                if not self.yield_actions or is_final:
                    yield output
                                # 如果达到最终输出，停止迭代
                if is_final:
                    return
        except BaseException as e:
            run_manager.on_chain_error(e)
            raise

                # 如果到达此处，意味着我们耗尽了迭代次数或时间
        yield self._stop(run_manager)

    async def __aiter__(self) -> AsyncIterator[AddableDict]:
        """为 `AgentExecutor` 创建一个异步迭代器。

        注意：__aiter__ 必须是普通方法，因此需要在第一次 __anext__ 调用时初始化异步运行管理器
        在那里我们可以等待它。
        """
        logger.debug("Initialising AgentExecutorIterator (async)")
        self.reset()
        callback_manager = AsyncCallbackManager.configure(
            self.callbacks,
            self.agent_executor.callbacks,
            self.agent_executor.verbose,
            self.tags,
            self.agent_executor.tags,
            self.metadata,
            self.agent_executor.metadata,
        )
        run_manager = await callback_manager.on_chain_start(
            dumpd(self.agent_executor),
            self.inputs,
            self.run_id,
            name=self.run_name,
        )
        try:
            async with asyncio_timeout(self.agent_executor.max_execution_time):
                while self.agent_executor._should_continue(  # noqa: SLF001
                    self.iterations,
                    self.time_elapsed,
                ):
                                        # 执行下一步：这会规划下一步动作，执行它，
                                        # 在生成时产出动作和观察结果
                    next_step_seq: NextStepOutput = []
                    async for chunk in self.agent_executor._aiter_next_step(  # noqa: SLF001
                        self.name_to_tool_map,
                        self.color_mapping,
                        self.inputs,
                        self.intermediate_steps,
                        run_manager,
                    ):
                        next_step_seq.append(chunk)
                                                # 如果我们在产出动作，就立即产出它们
                                                # 不要生成 AgentFinish，该情况将在下方处理
                        if self.yield_actions:
                            if isinstance(chunk, AgentAction):
                                yield AddableDict(
                                    actions=[chunk],
                                    messages=chunk.messages,
                                )
                            elif isinstance(chunk, AgentStep):
                                yield AddableDict(
                                    steps=[chunk],
                                    messages=chunk.messages,
                                )

                                        # 将迭代器输出转换为 _process_next_step 处理的格式
                    next_step = self.agent_executor._consume_next_step(next_step_seq)  # noqa: SLF001
                                        # 更新迭代次数和已过去的时间
                    self.update_iterations()
                                        # 判断这是否是最终输出
                    output = await self._aprocess_next_step_output(
                        next_step,
                        run_manager,
                    )
                    is_final = "intermediate_step" not in output
                                        # 始终生成最终输出
                                        # 为向后兼容，若未生成动作则生成整数输出
                    if not self.yield_actions or is_final:
                        yield output
                                        # 若达到最终输出，停止迭代
                    if is_final:
                        return
        except (TimeoutError, asyncio.TimeoutError):
            yield await self._astop(run_manager)
            return
        except BaseException as e:
            await run_manager.on_chain_error(e)
            raise

                # 若执行至此，表示已耗尽迭代次数或时间
        yield await self._astop(run_manager)

    def _process_next_step_output(
        self,
        next_step_output: AgentFinish | list[tuple[AgentAction, str]],
        run_manager: CallbackManagerForChainRun,
    ) -> AddableDict:
        """处理下一步的输出。

        处理下一步的输出，
        处理 AgentFinish 和工具返回的情况。
        """
        logger.debug("Processing output of Agent loop step")
        if isinstance(next_step_output, AgentFinish):
            logger.debug(
                "Hit AgentFinish: _return -> on_chain_end -> run final output logic",
            )
            return self._return(next_step_output, run_manager=run_manager)

        self.intermediate_steps.extend(next_step_output)
        logger.debug("Updated intermediate_steps with step output")

                # 检查工具返回值
        if len(next_step_output) == 1:
            next_step_action = next_step_output[0]
            tool_return = self.agent_executor._get_tool_return(next_step_action)  # noqa: SLF001
            if tool_return is not None:
                return self._return(tool_return, run_manager=run_manager)

        return AddableDict(intermediate_step=next_step_output)

    async def _aprocess_next_step_output(
        self,
        next_step_output: AgentFinish | list[tuple[AgentAction, str]],
        run_manager: AsyncCallbackManagerForChainRun,
    ) -> AddableDict:
        """处理下一步异步输出。

        处理下一步异步输出，
        处理 AgentFinish 和工具返回的情况。
        """
        logger.debug("Processing output of async Agent loop step")
        if isinstance(next_step_output, AgentFinish):
            logger.debug(
                "Hit AgentFinish: _areturn -> on_chain_end -> run final output logic",
            )
            return await self._areturn(next_step_output, run_manager=run_manager)

        self.intermediate_steps.extend(next_step_output)
        logger.debug("Updated intermediate_steps with step output")

                # 检查工具返回值
        if len(next_step_output) == 1:
            next_step_action = next_step_output[0]
            tool_return = self.agent_executor._get_tool_return(next_step_action)  # noqa: SLF001
            if tool_return is not None:
                return await self._areturn(tool_return, run_manager=run_manager)

        return AddableDict(intermediate_step=next_step_output)

    def _stop(self, run_manager: CallbackManagerForChainRun) -> AddableDict:
        """停止迭代器。

        停止迭代器并抛出一个带有停止响应的 StopIteration 异常。
        """
        logger.warning("Stopping agent prematurely due to triggering stop condition")
                # 手动构造带 output 键的 agent finish
        output = self.agent_executor._action_agent.return_stopped_response(  # noqa: SLF001
            self.agent_executor.early_stopping_method,
            self.intermediate_steps,
            **self.inputs,
        )
        return self._return(output, run_manager=run_manager)

    async def _astop(self, run_manager: AsyncCallbackManagerForChainRun) -> AddableDict:
        """停止异步迭代器。

        停止异步迭代器并抛出一个带有
        停止响应的 StopAsyncIteration 异常。
        """
        logger.warning("Stopping agent prematurely due to triggering stop condition")
        output = self.agent_executor._action_agent.return_stopped_response(  # noqa: SLF001
            self.agent_executor.early_stopping_method,
            self.intermediate_steps,
            **self.inputs,
        )
        return await self._areturn(output, run_manager=run_manager)

    def _return(
        self,
        output: AgentFinish,
        run_manager: CallbackManagerForChainRun,
    ) -> AddableDict:
        """返回迭代器的最终输出。"""
        returned_output = self.agent_executor._return(  # noqa: SLF001
            output,
            self.intermediate_steps,
            run_manager=run_manager,
        )
        returned_output["messages"] = output.messages
        run_manager.on_chain_end(returned_output)
        return self.make_final_outputs(returned_output, run_manager)

    async def _areturn(
        self,
        output: AgentFinish,
        run_manager: AsyncCallbackManagerForChainRun,
    ) -> AddableDict:
        """返回异步迭代器的最终输出。"""
        returned_output = await self.agent_executor._areturn(  # noqa: SLF001
            output,
            self.intermediate_steps,
            run_manager=run_manager,
        )
        returned_output["messages"] = output.messages
        await run_manager.on_chain_end(returned_output)
        return self.make_final_outputs(returned_output, run_manager)

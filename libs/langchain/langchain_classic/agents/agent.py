"""接收输入并生成动作及动作输入的链。"""

from __future__ import annotations

import asyncio
import builtins
import contextlib
import json
import logging
import time
from abc import abstractmethod
from collections.abc import AsyncIterator, Callable, Iterator, Sequence
from pathlib import Path
from typing import (
    Any,
    cast,
)

import yaml
from langchain_core._api import deprecated
from langchain_core.agents import AgentAction, AgentFinish, AgentStep
from langchain_core.callbacks import (
    AsyncCallbackManagerForChainRun,
    AsyncCallbackManagerForToolRun,
    BaseCallbackManager,
    CallbackManagerForChainRun,
    CallbackManagerForToolRun,
    Callbacks,
)
from langchain_core.exceptions import OutputParserException
from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import BaseMessage
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.prompts import BasePromptTemplate
from langchain_core.prompts.few_shot import FewShotPromptTemplate
from langchain_core.prompts.prompt import PromptTemplate
from langchain_core.runnables import Runnable, RunnableConfig, ensure_config
from langchain_core.runnables.utils import AddableDict
from langchain_core.tools import BaseTool
from langchain_core.utils.input import get_color_mapping
from pydantic import BaseModel, ConfigDict, model_validator
from typing_extensions import Self, override

from langchain_classic._api.deprecation import AGENT_DEPRECATION_WARNING
from langchain_classic.agents.agent_iterator import AgentExecutorIterator
from langchain_classic.agents.agent_types import AgentType
from langchain_classic.agents.tools import InvalidTool
from langchain_classic.chains.base import Chain
from langchain_classic.chains.llm import LLMChain
from langchain_classic.utilities.asyncio import asyncio_timeout

logger = logging.getLogger(__name__)


class BaseSingleActionAgent(BaseModel):
    """基础单动作智能体类。"""

    @property
    def return_values(self) -> list[str]:
        """智能体的返回值。"""
        return ["output"]

    def get_allowed_tools(self) -> list[str] | None:
        """获取允许使用的工具。"""
        return None

    @abstractmethod
    def plan(
        self,
        intermediate_steps: list[tuple[AgentAction, str]],
        callbacks: Callbacks = None,
        **kwargs: Any,
    ) -> AgentAction | AgentFinish:
        """根据输入决定该做什么。

        Args:
            intermediate_steps: LLM 到目前为止采取的步骤,
                以及观察结果。
            callbacks: 要运行的回调。
            **kwargs: 用户输入。

        Returns:
            指定要使用的工具的动作。
        """

    @abstractmethod
    async def aplan(
        self,
        intermediate_steps: list[tuple[AgentAction, str]],
        callbacks: Callbacks = None,
        **kwargs: Any,
    ) -> AgentAction | AgentFinish:
        """异步处理输入，决定该做什么。

        Args:
            intermediate_steps: LLM 到目前为止采取的步骤,
                以及观察结果。
            callbacks: 要运行的回调。
            **kwargs: 用户输入。

        Returns:
            指定要使用的工具的动作。
        """

    @property
    @abstractmethod
    def input_keys(self) -> list[str]:
        """返回输入键。"""

    def return_stopped_response(
        self,
        early_stopping_method: str,
        intermediate_steps: list[tuple[AgentAction, str]],  # noqa: ARG002
        **_: Any,
    ) -> AgentFinish:
        """当智能体因达到最大迭代次数而停止时返回响应。

        Args:
            early_stopping_method: 用于提前停止的方法。
            intermediate_steps: LLM 到目前为止采取的步骤,
                以及观察结果。

        Returns:
            智能体完成对象。

        Raises:
            ValueError: 如果 `early_stopping_method` 不受支持。
        """
        if early_stopping_method == "force":
                        # `force` 仅返回一个常量字符串
            return AgentFinish(
                {"output": "Agent stopped due to iteration limit or time limit."},
                "",
            )
        msg = f"Got unsupported early_stopping_method `{early_stopping_method}`"
        raise ValueError(msg)

    @classmethod
    def from_llm_and_tools(
        cls,
        llm: BaseLanguageModel,
        tools: Sequence[BaseTool],
        callback_manager: BaseCallbackManager | None = None,
        **kwargs: Any,
    ) -> BaseSingleActionAgent:
        """从 LLM 和工具构建一个智能体。

        Args:
            llm: 要使用的语言模型。
            tools: 要使用的工具。
            callback_manager: 要使用的回调管理器。
            kwargs: 其他参数。

        Returns:
            智能体对象。
        """
        raise NotImplementedError

    @property
    def _agent_type(self) -> str:
        """返回智能体类型的标识符。"""
        raise NotImplementedError

    @override
    def dict(self, **kwargs: Any) -> builtins.dict:
        """返回智能体的字典表示。

        Returns:
            智能体的字典表示。
        """
        _dict = super().model_dump()
        try:
            _type = self._agent_type
        except NotImplementedError:
            _type = None
        if isinstance(_type, AgentType):
            _dict["_type"] = str(_type.value)
        elif _type is not None:
            _dict["_type"] = _type
        return _dict

    def save(self, file_path: Path | str) -> None:
        """保存智能体。

        Args:
            file_path: 要保存智能体的文件路径。

        Example:
        ```python
        # If working with agent executor
        agent.agent.save(file_path="path/agent.yaml")
        ```
        """
                # 将文件转换为 Path 对象。
        save_path = Path(file_path) if isinstance(file_path, str) else file_path

        directory_path = save_path.parent
        directory_path.mkdir(parents=True, exist_ok=True)

                # 获取要保存的字典
        agent_dict = self.dict()
        if "_type" not in agent_dict:
            msg = f"Agent {self} does not support saving"
            raise NotImplementedError(msg)

        if save_path.suffix == ".json":
            with save_path.open("w") as f:
                json.dump(agent_dict, f, indent=4)
        elif save_path.suffix.endswith((".yaml", ".yml")):
            with save_path.open("w") as f:
                yaml.dump(agent_dict, f, default_flow_style=False)
        else:
            msg = f"{save_path} must be json or yaml"
            raise ValueError(msg)

    def tool_run_logging_kwargs(self) -> builtins.dict:
        """返回工具运行的日志关键字参数。"""
        return {}


class BaseMultiActionAgent(BaseModel):
    """基础多动作智能体类。"""

    @property
    def return_values(self) -> list[str]:
        """返回智能体的返回值。"""
        return ["output"]

    def get_allowed_tools(self) -> list[str] | None:
        """获取允许使用的工具。

        Returns:
            允许使用的工具。
        """
        return None

    @abstractmethod
    def plan(
        self,
        intermediate_steps: list[tuple[AgentAction, str]],
        callbacks: Callbacks = None,
        **kwargs: Any,
    ) -> list[AgentAction] | AgentFinish:
        """根据输入决定该做什么。

        Args:
            intermediate_steps: LLM 到目前为止采取的步骤,
                以及观察结果。
            callbacks: 要运行的回调。
            **kwargs: 用户输入。

        Returns:
            指定要使用的工具的动作。
        """

    @abstractmethod
    async def aplan(
        self,
        intermediate_steps: list[tuple[AgentAction, str]],
        callbacks: Callbacks = None,
        **kwargs: Any,
    ) -> list[AgentAction] | AgentFinish:
        """异步处理输入，决定该做什么。

        Args:
            intermediate_steps: LLM 到目前为止采取的步骤,
                以及观察结果。
            callbacks: 要运行的回调。
            **kwargs: 用户输入。

        Returns:
            指定要使用的工具的动作。
        """

    @property
    @abstractmethod
    def input_keys(self) -> list[str]:
        """返回输入键。"""

    def return_stopped_response(
        self,
        early_stopping_method: str,
        intermediate_steps: list[tuple[AgentAction, str]],  # noqa: ARG002
        **_: Any,
    ) -> AgentFinish:
        """当智能体因达到最大迭代次数而停止时返回响应。

        Args:
            early_stopping_method: 用于提前停止的方法。
            intermediate_steps: LLM 到目前为止采取的步骤,
                以及观察结果。

        Returns:
            智能体完成对象。

        Raises:
            ValueError: 如果 `early_stopping_method` 不受支持。
        """
        if early_stopping_method == "force":
                        # `force` 仅返回一个常量字符串
            return AgentFinish({"output": "Agent stopped due to max iterations."}, "")
        msg = f"Got unsupported early_stopping_method `{early_stopping_method}`"
        raise ValueError(msg)

    @property
    def _agent_type(self) -> str:
        """返回智能体类型的标识符。"""
        raise NotImplementedError

    @override
    def dict(self, **kwargs: Any) -> builtins.dict:
        """返回智能体的字典表示。"""
        _dict = super().model_dump()
        with contextlib.suppress(NotImplementedError):
            _dict["_type"] = str(self._agent_type)
        return _dict

    def save(self, file_path: Path | str) -> None:
        """保存智能体。

        Args:
            file_path: 要保存智能体的文件路径。

        Raises:
            NotImplementedError: 如果智能体不支持保存。
            ValueError: 如果 `file_path` 不是 json 或 yaml 格式。

        Example:
        ```python
        # If working with agent executor
        agent.agent.save(file_path="path/agent.yaml")
        ```
        """
                # 将文件转换为 Path 对象。
        save_path = Path(file_path) if isinstance(file_path, str) else file_path

                # 获取要保存的字典
        agent_dict = self.dict()
        if "_type" not in agent_dict:
            msg = f"Agent {self} does not support saving."
            raise NotImplementedError(msg)

        directory_path = save_path.parent
        directory_path.mkdir(parents=True, exist_ok=True)

        if save_path.suffix == ".json":
            with save_path.open("w") as f:
                json.dump(agent_dict, f, indent=4)
        elif save_path.suffix.endswith((".yaml", ".yml")):
            with save_path.open("w") as f:
                yaml.dump(agent_dict, f, default_flow_style=False)
        else:
            msg = f"{save_path} must be json or yaml"
            raise ValueError(msg)

    def tool_run_logging_kwargs(self) -> builtins.dict:
        """返回工具运行的日志关键字参数。"""
        return {}


class AgentOutputParser(BaseOutputParser[AgentAction | AgentFinish]):
    """用于将智能体输出解析为智能体动作/完成状态的基础类。"""

    @abstractmethod
    def parse(self, text: str) -> AgentAction | AgentFinish:
        """将文本解析为智能体动作/完成。"""


class MultiActionAgentOutputParser(
    BaseOutputParser[list[AgentAction] | AgentFinish],
):
    """用于将智能体输出解析为智能体动作/完成状态的基础类。

    用于可返回多个动作的智能体。
    """

    @abstractmethod
    def parse(self, text: str) -> list[AgentAction] | AgentFinish:
        """将文本解析为智能体动作/完成。

        Args:
            text: 要解析的文本。

        Returns:
            智能体动作列表或智能体完成对象。
        """


class RunnableAgent(BaseSingleActionAgent):
    """由 Runnable 驱动的智能体。"""

    runnable: Runnable[dict, AgentAction | AgentFinish]
    """调用以获取智能体动作的 Runnable。"""
    input_keys_arg: list[str] = []
    return_keys_arg: list[str] = []
    stream_runnable: bool = True
    """是否从 Runnable 流式输出。

    If `True` then underlying LLM is invoked in a streaming fashion to make it possible
        to get access to the individual LLM tokens when using stream_log with the
        `AgentExecutor`. If `False` then LLM is invoked in a non-streaming fashion and
        individual LLM tokens will not be available in stream_log.
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )

    @property
    def return_values(self) -> list[str]:
        """返回智能体的返回值。"""
        return self.return_keys_arg

    @property
    def input_keys(self) -> list[str]:
        """返回输入键。"""
        return self.input_keys_arg

    def plan(
        self,
        intermediate_steps: list[tuple[AgentAction, str]],
        callbacks: Callbacks = None,
        **kwargs: Any,
    ) -> AgentAction | AgentFinish:
        """根据过往历史和当前输入，决定要做什么。

        Args:
            intermediate_steps: LLM 到目前为止采取的步骤，
                以及观察结果。
            callbacks: 要运行的回调。
            **kwargs: 用户输入。

        Returns:
            指定要使用的工具的操作。
        """
        inputs = {**kwargs, "intermediate_steps": intermediate_steps}
        final_output: Any = None
        if self.stream_runnable:
                        # 使用流式确保底层 LLM 以流式方式调用
                        # 流式
                        # 以便在使用 AgentExecutor 的 stream_log 时能够访问单个 LLM token
                        # 由于计划的响应不是生成器，我们需要将输出累积为最终输出并返回该结果。
                        # 使用流式确保底层 LLM 以流式方式调用
                        # 流式
            for chunk in self.runnable.stream(inputs, config={"callbacks": callbacks}):
                if final_output is None:
                    final_output = chunk
                else:
                    final_output += chunk
        else:
            final_output = self.runnable.invoke(inputs, config={"callbacks": callbacks})

        return final_output

    async def aplan(
        self,
        intermediate_steps: list[tuple[AgentAction, str]],
        callbacks: Callbacks = None,
        **kwargs: Any,
    ) -> AgentAction | AgentFinish:
        """基于过往历史和当前输入，异步决定要做什么。

        Args:
            intermediate_steps: LLM 到目前为止采取的步骤，
                以及观察结果。
            callbacks: 要运行的回调。
            **kwargs: 用户输入。

        Returns:
            指定要使用的工具的操作。
        """
        inputs = {**kwargs, "intermediate_steps": intermediate_steps}
        final_output: Any = None
        if self.stream_runnable:
                        # 以便在使用 AgentExecutor 的 stream_log 时能够访问单个 LLM token
                        # 由于计划的响应不是生成器，我们需要将输出累积为最终输出并返回该结果。
                        # 使用流式确保底层 LLM 以流式方式调用
                        # 流式
                        # 以便在使用 AgentExecutor 的 stream_log 时能够访问单个 LLM token
                        # 由于计划的响应不是生成器，我们需要将输出累积为最终输出并返回该结果。
            async for chunk in self.runnable.astream(
                inputs,
                config={"callbacks": callbacks},
            ):
                if final_output is None:
                    final_output = chunk
                else:
                    final_output += chunk
        else:
            final_output = await self.runnable.ainvoke(
                inputs,
                config={"callbacks": callbacks},
            )
        return final_output


class RunnableMultiActionAgent(BaseMultiActionAgent):
    """由 Runnable 驱动的智能体。"""

    runnable: Runnable[dict, list[AgentAction] | AgentFinish]
    """调用以获取智能体动作的 Runnable。"""
    input_keys_arg: list[str] = []
    return_keys_arg: list[str] = []
    stream_runnable: bool = True
    """是否从 Runnable 流式输出。

    If `True` then underlying LLM is invoked in a streaming fashion to make it possible
        to get access to the individual LLM tokens when using stream_log with the
        `AgentExecutor`. If `False` then LLM is invoked in a non-streaming fashion and
        individual LLM tokens will not be available in stream_log.
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )

    @property
    def return_values(self) -> list[str]:
        """智能体的返回值。"""
        return self.return_keys_arg

    @property
    def input_keys(self) -> list[str]:
        """返回输入键。

        Returns:
            输入键列表。
        """
        return self.input_keys_arg

    def plan(
        self,
        intermediate_steps: list[tuple[AgentAction, str]],
        callbacks: Callbacks = None,
        **kwargs: Any,
    ) -> list[AgentAction] | AgentFinish:
        """根据过往历史和当前输入，决定要做什么。

        Args:
            intermediate_steps: LLM 到目前为止采取的步骤，
                以及观察结果。
            callbacks: 要运行的回调。
            **kwargs: 用户输入。

        Returns:
            指定要使用的工具的操作。
        """
        inputs = {**kwargs, "intermediate_steps": intermediate_steps}
        final_output: Any = None
        if self.stream_runnable:
                        # 使用流式确保底层 LLM 以流式方式调用
                        # 流式
                        # 以便在使用 AgentExecutor 的 stream_log 时能够访问单个 LLM token
                        # 由于计划的响应不是生成器，我们需要将输出累积为最终输出并返回该结果。
                        # 使用流式确保底层 LLM 以流式方式调用
                        # 流式
            for chunk in self.runnable.stream(inputs, config={"callbacks": callbacks}):
                if final_output is None:
                    final_output = chunk
                else:
                    final_output += chunk
        else:
            final_output = self.runnable.invoke(inputs, config={"callbacks": callbacks})

        return final_output

    async def aplan(
        self,
        intermediate_steps: list[tuple[AgentAction, str]],
        callbacks: Callbacks = None,
        **kwargs: Any,
    ) -> list[AgentAction] | AgentFinish:
        """基于过往历史和当前输入，异步决定要做什么。

        Args:
            intermediate_steps: LLM 到目前为止采取的步骤，
                以及观察结果。
            callbacks: 要运行的回调。
            **kwargs: 用户输入。

        Returns:
            指定要使用的工具的操作。
        """
        inputs = {**kwargs, "intermediate_steps": intermediate_steps}
        final_output: Any = None
        if self.stream_runnable:
                        # 以便在使用 AgentExecutor 的 stream_log 时能够访问单个 LLM token
                        # 由于计划的响应不是生成器，我们需要将输出累积为最终输出并返回该结果。
                        # `force` 仅返回一个常量字符串
                        # Generate 执行最后一次前向传播
                        # 在先前步骤的基础上，我们现在告诉 LLM 进行最后一次预测
                        # 我们尝试提取最终答案
            async for chunk in self.runnable.astream(
                inputs,
                config={"callbacks": callbacks},
            ):
                if final_output is None:
                    final_output = chunk
                else:
                    final_output += chunk
        else:
            final_output = await self.runnable.ainvoke(
                inputs,
                config={"callbacks": callbacks},
            )

        return final_output


@deprecated(
    "0.1.0",
    message=AGENT_DEPRECATION_WARNING,
    removal="2.0.0",
)
class LLMSingleActionAgent(BaseSingleActionAgent):
    """单动作智能体的基础类。"""

    llm_chain: LLMChain
    """用于智能体的 LLMChain。"""
    output_parser: AgentOutputParser
    """用于智能体的输出解析器。"""
    stop: list[str]
    """停止字符串列表。"""

    @property
    def input_keys(self) -> list[str]:
        """返回输入键。

        Returns:
            输入键列表。
        """
        return list(set(self.llm_chain.input_keys) - {"intermediate_steps"})

    @override
    def dict(self, **kwargs: Any) -> builtins.dict:
        """返回智能体的字典表示形式。"""
        _dict = super().dict()
        del _dict["output_parser"]
        return _dict

    def plan(
        self,
        intermediate_steps: list[tuple[AgentAction, str]],
        callbacks: Callbacks = None,
        **kwargs: Any,
    ) -> AgentAction | AgentFinish:
        """给定输入后，决定要做什么。

        Args:
            intermediate_steps: LLM 到目前为止采取的步骤，
                以及观察结果。
            callbacks: 要运行的回调。
            **kwargs: 用户输入。

        Returns:
            指定要使用的工具的操作。
        """
        output = self.llm_chain.run(
            intermediate_steps=intermediate_steps,
            stop=self.stop,
            callbacks=callbacks,
            **kwargs,
        )
        return self.output_parser.parse(output)

    async def aplan(
        self,
        intermediate_steps: list[tuple[AgentAction, str]],
        callbacks: Callbacks = None,
        **kwargs: Any,
    ) -> AgentAction | AgentFinish:
        """异步给定输入后，决定要做什么。

        Args:
            intermediate_steps: LLM 到目前为止采取的步骤，
                以及观察结果。
            callbacks: 要运行的回调。
            **kwargs: 用户输入。

        Returns:
            指定要使用的工具的操作。
        """
        output = await self.llm_chain.arun(
            intermediate_steps=intermediate_steps,
            stop=self.stop,
            callbacks=callbacks,
            **kwargs,
        )
        return self.output_parser.parse(output)

    def tool_run_logging_kwargs(self) -> builtins.dict:
        """返回工具运行的日志关键字参数。"""
        return {
            "llm_prefix": "",
            "observation_prefix": "" if len(self.stop) == 0 else self.stop[0],
        }


@deprecated(
    "0.1.0",
    message=AGENT_DEPRECATION_WARNING,
    removal="2.0.0",
)
class Agent(BaseSingleActionAgent):
    """调用语言模型并决定动作的智能体。

    此类智能体由 LLMChain 驱动。LLMChain 中的提示词必须包含一个名为 "agent_scratchpad"
    的变量，智能体可在其中存放其中间工作内容。
    """

    llm_chain: LLMChain
    """用于智能体的 LLMChain。"""
    output_parser: AgentOutputParser
    """用于智能体的输出解析器。"""
    allowed_tools: list[str] | None = None
    """智能体允许使用的工具列表。若为 `None`，则允许所有工具。"""

    @override
    def dict(self, **kwargs: Any) -> builtins.dict:
        """返回智能体的字典表示形式。"""
        _dict = super().dict()
        del _dict["output_parser"]
        return _dict

    def get_allowed_tools(self) -> list[str] | None:
        """获取允许使用的工具。"""
        return self.allowed_tools

    @property
    def return_values(self) -> list[str]:
        """智能体的返回值。"""
        return ["output"]

    @property
    def _stop(self) -> list[str]:
        return [
            f"\n{self.observation_prefix.rstrip()}",
            f"\n\t{self.observation_prefix.rstrip()}",
        ]

    def _construct_scratchpad(
        self,
        intermediate_steps: list[tuple[AgentAction, str]],
    ) -> str | list[BaseMessage]:
        """构建用于让智能体继续其思考过程的草稿板。"""
        thoughts = ""
        for action, observation in intermediate_steps:
            thoughts += action.log
            thoughts += f"\n{self.observation_prefix}{observation}\n{self.llm_prefix}"
        return thoughts

    def plan(
        self,
        intermediate_steps: list[tuple[AgentAction, str]],
        callbacks: Callbacks = None,
        **kwargs: Any,
    ) -> AgentAction | AgentFinish:
        """给定输入后，决定要做什么。

        Args:
            intermediate_steps: LLM 到目前为止采取的步骤，
                以及观察结果。
            callbacks: 要运行的回调。
            **kwargs: 用户输入。

        Returns:
            指定要使用的工具的操作。
        """
        full_inputs = self.get_full_inputs(intermediate_steps, **kwargs)
        full_output = self.llm_chain.predict(callbacks=callbacks, **full_inputs)
        return self.output_parser.parse(full_output)

    async def aplan(
        self,
        intermediate_steps: list[tuple[AgentAction, str]],
        callbacks: Callbacks = None,
        **kwargs: Any,
    ) -> AgentAction | AgentFinish:
        """异步给定输入后，决定要做什么。

        Args:
            intermediate_steps: LLM 到目前为止采取的步骤，
                以及观察结果。
            callbacks: 要运行的回调。
            **kwargs: 用户输入。

        Returns:
            指定要使用的工具的操作。
        """
        full_inputs = self.get_full_inputs(intermediate_steps, **kwargs)
        full_output = await self.llm_chain.apredict(callbacks=callbacks, **full_inputs)
        return await self.output_parser.aparse(full_output)

    def get_full_inputs(
        self,
        intermediate_steps: list[tuple[AgentAction, str]],
        **kwargs: Any,
    ) -> builtins.dict[str, Any]:
        """从中间步骤创建 LLMChain 的完整输入。

        Args:
            intermediate_steps: LLM 到目前为止采取的步骤，
                以及观察结果。
            **kwargs: 用户输入。

        Returns:
            LLMChain 的完整输入。
        """
        thoughts = self._construct_scratchpad(intermediate_steps)
        new_inputs = {"agent_scratchpad": thoughts, "stop": self._stop}
        return {**kwargs, **new_inputs}

    @property
    def input_keys(self) -> list[str]:
        """返回输入键。"""
        return list(set(self.llm_chain.input_keys) - {"agent_scratchpad"})

    @model_validator(mode="after")
    def validate_prompt(self) -> Self:
        """验证提示词是否匹配格式。

        Args:
            values: 要验证的值。

        Returns:
            验证后的值。

        Raises:
            ValueError: 如果 `agent_scratchpad` 不在 prompt.input_variables 中，且
            prompt 不是 FewShotPromptTemplate 或 PromptTemplate。
        """
        prompt = self.llm_chain.prompt
        if "agent_scratchpad" not in prompt.input_variables:
            logger.warning(
                "`agent_scratchpad` should be a variable in prompt.input_variables."
                " Did not find it, so adding it at the end.",
            )
            prompt.input_variables.append("agent_scratchpad")
            if isinstance(prompt, PromptTemplate):
                prompt.template += "\n{agent_scratchpad}"
            elif isinstance(prompt, FewShotPromptTemplate):
                prompt.suffix += "\n{agent_scratchpad}"
            else:
                msg = f"Got unexpected prompt type {type(prompt)}"
                raise ValueError(msg)
        return self

    @property
    @abstractmethod
    def observation_prefix(self) -> str:
        """要附加到观察结果前的前缀。"""

    @property
    @abstractmethod
    def llm_prefix(self) -> str:
        """要附加到 LLM 调用前的前缀。"""

    @classmethod
    @abstractmethod
    def create_prompt(cls, tools: Sequence[BaseTool]) -> BasePromptTemplate:
        """为此类创建一个提示词。

        Args:
            tools: 要使用的工具。

        Returns:
            提示词模板。
        """

    @classmethod
    def _validate_tools(cls, tools: Sequence[BaseTool]) -> None:
        """验证是否传入了适当的工具。

        Args:
            tools: 要使用的工具。
        """

    @classmethod
    @abstractmethod
    def _get_default_output_parser(cls, **kwargs: Any) -> AgentOutputParser:
        """获取此类的默认输出解析器。"""

    @classmethod
    def from_llm_and_tools(
        cls,
        llm: BaseLanguageModel,
        tools: Sequence[BaseTool],
        callback_manager: BaseCallbackManager | None = None,
        output_parser: AgentOutputParser | None = None,
        **kwargs: Any,
    ) -> Agent:
        """从 LLM 和工具构建一个智能体。

        Args:
            llm: 要使用的语言模型。
            tools: 要使用的工具。
            callback_manager: 要使用的回调管理器。
            output_parser: 要使用的输出解析器。
            kwargs: 其他参数。

        Returns:
            智能体对象。
        """
        cls._validate_tools(tools)
        llm_chain = LLMChain(
            llm=llm,
            prompt=cls.create_prompt(tools),
            callback_manager=callback_manager,
        )
        tool_names = [tool.name for tool in tools]
        _output_parser = output_parser or cls._get_default_output_parser()
        return cls(
            llm_chain=llm_chain,
            allowed_tools=tool_names,
            output_parser=_output_parser,
            **kwargs,
        )

    def return_stopped_response(
        self,
        early_stopping_method: str,
        intermediate_steps: list[tuple[AgentAction, str]],
        **kwargs: Any,
    ) -> AgentFinish:
        """当智能体因最大迭代次数而停止时返回响应。

        Args:
            early_stopping_method: 用于提前停止的方法。
            intermediate_steps: LLM 到目前为止采取的步骤，
                以及观察结果。
            **kwargs: 用户输入。

        Returns:
            智能体完成对象。

        Raises:
            ValueError: 如果 `early_stopping_method` 不在 ['force', 'generate'] 中。
        """
        if early_stopping_method == "force":
                        # 如果可以提取，则发送正确内容
            return AgentFinish(
                {"output": "Agent stopped due to iteration limit or time limit."},
                "",
            )
        if early_stopping_method == "generate":
                        # 如果可以提取，但工具不是最终工具，
            thoughts = ""
            for action, observation in intermediate_steps:
                thoughts += action.log
                thoughts += (
                    f"\n{self.observation_prefix}{observation}\n{self.llm_prefix}"
                )
                        # 则仅返回完整输出
            thoughts += (
                "\n\nI now need to return a final answer based on the previous steps:"
            )
            new_inputs = {"agent_scratchpad": thoughts, "stop": self._stop}
            full_inputs = {**kwargs, **new_inputs}
            full_output = self.llm_chain.predict(**full_inputs)
                        # 调用 LLM 以确定下一步操作。
            parsed_output = self.output_parser.parse(full_output)
            if isinstance(parsed_output, AgentFinish):
                                # 如果选择的工具是结束工具，则终止并返回。
                return parsed_output
                        # 否则我们查找该工具
                        # 然后在工具输入上调用工具以获取观察结果
            return AgentFinish({"output": full_output}, full_output)
        msg = (
            "early_stopping_method should be one of `force` or `generate`, "
            f"got {early_stopping_method}"
        )
        raise ValueError(msg)

    def tool_run_logging_kwargs(self) -> builtins.dict:
        """返回工具运行的日志关键字参数。"""
        return {
            "llm_prefix": self.llm_prefix,
            "observation_prefix": self.observation_prefix,
        }


class ExceptionTool(BaseTool):
    """仅返回查询的工具。"""

    name: str = "_Exception"
    """工具名称。"""
    description: str = "Exception tool"
    """工具描述。"""

    @override
    def _run(
        self,
        query: str,
        run_manager: CallbackManagerForToolRun | None = None,
    ) -> str:
        return query

    @override
    async def _arun(
        self,
        query: str,
        run_manager: AsyncCallbackManagerForToolRun | None = None,
    ) -> str:
        return query


NextStepOutput = list[AgentFinish | AgentAction | AgentStep]
RunnableAgentType = RunnableAgent | RunnableMultiActionAgent


class AgentExecutor(Chain):
    """使用工具的智能体。"""

    agent: BaseSingleActionAgent | BaseMultiActionAgent | Runnable
    """The agent to run for creating a plan and determining actions
    to take at each step of the execution loop."""
    tools: Sequence[BaseTool]
    """智能体可调用的有效工具。"""
    return_intermediate_steps: bool = False
    """Whether to return the agent's trajectory of intermediate steps
    at the end in addition to the final output."""
    max_iterations: int | None = 15
    """在执行循环结束前最多执行的步数。

    设置为 'None' 可能导致无限循环。"""
    max_execution_time: float | None = None
    """执行循环中允许花费的最大实际时间。
    """
    early_stopping_method: str = "force"
    """The method to use for early stopping if the agent never
    returns `AgentFinish`. Either 'force' or 'generate'.

    `"force"` 返回一条字符串，说明因达到时间或迭代限制而停止。

    `"generate"` 最后一次调用智能体的 LLM Chain，基于之前步骤生成最终答案。
    """
    handle_parsing_errors: bool | str | Callable[[OutputParserException], str] = False
    """How to handle errors raised by the agent's output parser.
    Defaults to `False`, which raises the error.
    If `true`, the error will be sent back to the LLM as an observation.
    If a string, the string itself will be sent to the LLM as an observation.
    If a callable function, the function will be called with the exception as an
    argument, and the result of that function will be passed to the agent as an
    observation.
    """
    trim_intermediate_steps: (
        int | Callable[[list[tuple[AgentAction, str]]], list[tuple[AgentAction, str]]]
    ) = -1
    """如何裁剪返回前的中间步骤。
    默认值为 -1，表示不裁剪。
    """

    @classmethod
    def from_agent_and_tools(
        cls,
        agent: BaseSingleActionAgent | BaseMultiActionAgent | Runnable,
        tools: Sequence[BaseTool],
        callbacks: Callbacks = None,
        **kwargs: Any,
    ) -> AgentExecutor:
        """从智能体和工具创建。

        Args:
            agent: 要使用的智能体。
            tools: 要使用的工具。
            callbacks: 要使用的回调。
            kwargs: 其他参数。

        Returns:
            智能体执行器对象。
        """
        return cls(
            agent=agent,
            tools=tools,
            callbacks=callbacks,
            **kwargs,
        )

    @model_validator(mode="after")
    def validate_tools(self) -> Self:
        """验证工具与智能体兼容。

        Args:
            values: 要验证的值。

        Returns:
            验证后的值。

        Raises:
            ValueError: 如果允许的工具与提供的工具不同。
        """
        agent = self.agent
        tools = self.tools
        allowed_tools = agent.get_allowed_tools()  # type: ignore[union-attr]
        if allowed_tools is not None and set(allowed_tools) != {
            tool.name for tool in tools
        }:
            msg = (
                f"Allowed tools ({allowed_tools}) different than "
                f"provided tools ({[tool.name for tool in tools]})"
            )
            raise ValueError(msg)
        return self

    @model_validator(mode="before")
    @classmethod
    def validate_runnable_agent(cls, values: dict) -> Any:
        """如果传入，则将 Runnable 转换为智能体。

        Args:
            values: 要验证的值。

        Returns:
            验证后的值。
        """
        agent = values.get("agent")
        if agent and isinstance(agent, Runnable):
            try:
                output_type = agent.OutputType
            except TypeError:
                multi_action = False
            except Exception:
                logger.exception("Unexpected error getting OutputType from agent")
                multi_action = False
            else:
                multi_action = output_type == list[AgentAction] | AgentFinish

            stream_runnable = values.pop("stream_runnable", True)
            if multi_action:
                values["agent"] = RunnableMultiActionAgent(
                    runnable=agent,
                    stream_runnable=stream_runnable,
                )
            else:
                values["agent"] = RunnableAgent(
                    runnable=agent,
                    stream_runnable=stream_runnable,
                )
        return values

    @property
    def _action_agent(self) -> BaseSingleActionAgent | BaseMultiActionAgent:
        """对 self.agent 进行类型转换。

        如果 `agent` 属性是 Runnable，它将在 validate_runnable_agent 的 root_validator 中被转换为
        RunnableAgentType 之一。

        为了支持使用 Runnable 实例化，此处显式地进行类型转换
        以反映 root_validator 中所做的更改。
        """
        if isinstance(self.agent, Runnable):
            return cast("RunnableAgentType", self.agent)
        return self.agent

    @override
    def save(self, file_path: Path | str) -> None:
        """抛出错误 - 不支持智能体执行器的保存。

        Args:
            file_path: 要保存的路径。

        Raises:
            ValueError: 不支持智能体执行器的保存。
        """
        msg = (
            "Saving not supported for agent executors. "
            "If you are trying to save the agent, please use the "
            "`.save_agent(...)`"
        )
        raise ValueError(msg)

    def save_agent(self, file_path: Path | str) -> None:
        """保存底层智能体。

        Args:
            file_path: 要保存的路径。
        """
        return self._action_agent.save(file_path)

    def iter(
        self,
        inputs: Any,
        callbacks: Callbacks = None,
        *,
        include_run_info: bool = False,
        async_: bool = False,  # noqa: ARG002 arg kept for backwards compat, but ignored
    ) -> AgentExecutorIterator:
        """启用对达到最终输出所采取步骤的迭代。

        Args:
            inputs: 智能体的输入。
            callbacks: 要运行的回调。
            include_run_info: 是否包含运行信息。
            async_: 是否异步运行。（被忽略）

        Returns:
            智能体执行器迭代器对象。
        """
        return AgentExecutorIterator(
            self,
            inputs,
            callbacks,
            tags=self.tags,
            include_run_info=include_run_info,
        )

    @property
    def input_keys(self) -> list[str]:
        """返回输入键。"""
        return self._action_agent.input_keys

    @property
    def output_keys(self) -> list[str]:
        """返回单一输出键。"""
        if self.return_intermediate_steps:
            return [*self._action_agent.return_values, "intermediate_steps"]
        return self._action_agent.return_values

    def lookup_tool(self, name: str) -> BaseTool:
        """通过名称查找工具。

        Args:
            name: 工具的名称。

        Returns:
            工具对象。
        """
        return {tool.name: tool for tool in self.tools}[name]

    def _should_continue(self, iterations: int, time_elapsed: float) -> bool:
        if self.max_iterations is not None and iterations >= self.max_iterations:
            return False
        return self.max_execution_time is None or time_elapsed < self.max_execution_time

    def _return(
        self,
        output: AgentFinish,
        intermediate_steps: list,
        run_manager: CallbackManagerForChainRun | None = None,
    ) -> dict[str, Any]:
        if run_manager:
            run_manager.on_agent_finish(output, color="green", verbose=self.verbose)
        final_output = output.return_values
        if self.return_intermediate_steps:
            final_output["intermediate_steps"] = intermediate_steps
        return final_output

    async def _areturn(
        self,
        output: AgentFinish,
        intermediate_steps: list,
        run_manager: AsyncCallbackManagerForChainRun | None = None,
    ) -> dict[str, Any]:
        if run_manager:
            await run_manager.on_agent_finish(
                output,
                color="green",
                verbose=self.verbose,
            )
        final_output = output.return_values
        if self.return_intermediate_steps:
            final_output["intermediate_steps"] = intermediate_steps
        return final_output

    def _consume_next_step(
        self,
        values: NextStepOutput,
    ) -> AgentFinish | list[tuple[AgentAction, str]]:
        if isinstance(values[-1], AgentFinish):
            if len(values) != 1:
                msg = "Expected a single AgentFinish output, but got multiple values."
                raise ValueError(msg)
            return values[-1]
        return [(a.action, a.observation) for a in values if isinstance(a, AgentStep)]

    def _take_next_step(
        self,
        name_to_tool_map: dict[str, BaseTool],
        color_mapping: dict[str, str],
        inputs: dict[str, str],
        intermediate_steps: list[tuple[AgentAction, str]],
        run_manager: CallbackManagerForChainRun | None = None,
    ) -> AgentFinish | list[tuple[AgentAction, str]]:
        return self._consume_next_step(
            list(
                self._iter_next_step(
                    name_to_tool_map,
                    color_mapping,
                    inputs,
                    intermediate_steps,
                    run_manager,
                ),
            ),
        )

    def _iter_next_step(
        self,
        name_to_tool_map: dict[str, BaseTool],
        color_mapping: dict[str, str],
        inputs: dict[str, str],
        intermediate_steps: list[tuple[AgentAction, str]],
        run_manager: CallbackManagerForChainRun | None = None,
    ) -> Iterator[AgentFinish | AgentAction | AgentStep]:
        """在思维-行动-观察循环中执行单步操作。

        重写此方法以控制智能体如何做出选择并采取行动。
        """
        try:
            intermediate_steps = self._prepare_intermediate_steps(intermediate_steps)

                        # 调用 LLM 以确定下一步操作。
            output = self._action_agent.plan(
                intermediate_steps,
                callbacks=run_manager.get_child() if run_manager else None,
                **inputs,
            )
        except OutputParserException as e:
            if isinstance(self.handle_parsing_errors, bool):
                raise_error = not self.handle_parsing_errors
            else:
                raise_error = False
            if raise_error:
                msg = (
                    "An output parsing error occurred. "
                    "In order to pass this error back to the agent and have it try "
                    "again, pass `handle_parsing_errors=True` to the AgentExecutor. "
                    f"This is the error: {e!s}"
                )
                raise ValueError(msg) from e
            text = str(e)
            if isinstance(self.handle_parsing_errors, bool):
                if e.send_to_llm:
                    observation = str(e.observation)
                    text = str(e.llm_output)
                else:
                    observation = "Invalid or incomplete response"
            elif isinstance(self.handle_parsing_errors, str):
                observation = self.handle_parsing_errors
            elif callable(self.handle_parsing_errors):
                observation = self.handle_parsing_errors(e)
            else:
                msg = "Got unexpected type of `handle_parsing_errors`"  # type: ignore[unreachable]
                raise ValueError(msg) from e  # noqa: TRY004
            output = AgentAction("_Exception", observation, text)
            if run_manager:
                run_manager.on_agent_action(output, color="green")
            tool_run_kwargs = self._action_agent.tool_run_logging_kwargs()
            observation = ExceptionTool().run(
                output.tool_input,
                verbose=self.verbose,
                color=None,
                callbacks=run_manager.get_child() if run_manager else None,
                **tool_run_kwargs,
            )
            yield AgentStep(action=output, observation=observation)
            return

                # 如果选择的工具是结束工具，则终止并返回。
        if isinstance(output, AgentFinish):
            yield output
            return

        actions: list[AgentAction]
        actions = [output] if isinstance(output, AgentAction) else output
        for agent_action in actions:
            yield agent_action
        for agent_action in actions:
            yield self._perform_agent_action(
                name_to_tool_map,
                color_mapping,
                agent_action,
                run_manager,
            )

    def _perform_agent_action(
        self,
        name_to_tool_map: dict[str, BaseTool],
        color_mapping: dict[str, str],
        agent_action: AgentAction,
        run_manager: CallbackManagerForChainRun | None = None,
    ) -> AgentStep:
        if run_manager:
            run_manager.on_agent_action(agent_action, color="green")
                # 使用 asyncio.gather 并发运行多个 tool.arun() 调用
        if agent_action.tool in name_to_tool_map:
            tool = name_to_tool_map[agent_action.tool]
            return_direct = tool.return_direct
            color = color_mapping[agent_action.tool]
            tool_run_kwargs = self._action_agent.tool_run_logging_kwargs()
            if return_direct:
                tool_run_kwargs["llm_prefix"] = ""
                        # TODO: 此处可逐个产出结果（当其可用时）
            observation = tool.run(
                agent_action.tool_input,
                verbose=self.verbose,
                color=color,
                callbacks=run_manager.get_child() if run_manager else None,
                **tool_run_kwargs,
            )
        else:
            tool_run_kwargs = self._action_agent.tool_run_logging_kwargs()
            observation = InvalidTool().run(
                {
                    "requested_tool_name": agent_action.tool,
                    "available_tool_names": list(name_to_tool_map.keys()),
                },
                verbose=self.verbose,
                color=None,
                callbacks=run_manager.get_child() if run_manager else None,
                **tool_run_kwargs,
            )
        return AgentStep(action=agent_action, observation=observation)

    async def _atake_next_step(
        self,
        name_to_tool_map: dict[str, BaseTool],
        color_mapping: dict[str, str],
        inputs: dict[str, str],
        intermediate_steps: list[tuple[AgentAction, str]],
        run_manager: AsyncCallbackManagerForChainRun | None = None,
    ) -> AgentFinish | list[tuple[AgentAction, str]]:
        return self._consume_next_step(
            [
                a
                async for a in self._aiter_next_step(
                    name_to_tool_map,
                    color_mapping,
                    inputs,
                    intermediate_steps,
                    run_manager,
                )
            ],
        )

    async def _aiter_next_step(
        self,
        name_to_tool_map: dict[str, BaseTool],
        color_mapping: dict[str, str],
        inputs: dict[str, str],
        intermediate_steps: list[tuple[AgentAction, str]],
        run_manager: AsyncCallbackManagerForChainRun | None = None,
    ) -> AsyncIterator[AgentFinish | AgentAction | AgentStep]:
        """在思维-行动-观察循环中执行单步操作。

        重写此方法以控制智能体如何做出选择并采取行动。
        """
        try:
            intermediate_steps = self._prepare_intermediate_steps(intermediate_steps)

                        # 否则我们查找该工具
            output = await self._action_agent.aplan(
                intermediate_steps,
                callbacks=run_manager.get_child() if run_manager else None,
                **inputs,
            )
        except OutputParserException as e:
            if isinstance(self.handle_parsing_errors, bool):
                raise_error = not self.handle_parsing_errors
            else:
                raise_error = False
            if raise_error:
                msg = (
                    "An output parsing error occurred. "
                    "In order to pass this error back to the agent and have it try "
                    "again, pass `handle_parsing_errors=True` to the AgentExecutor. "
                    f"This is the error: {e!s}"
                )
                raise ValueError(msg) from e
            text = str(e)
            if isinstance(self.handle_parsing_errors, bool):
                if e.send_to_llm:
                    observation = str(e.observation)
                    text = str(e.llm_output)
                else:
                    observation = "Invalid or incomplete response"
            elif isinstance(self.handle_parsing_errors, str):
                observation = self.handle_parsing_errors
            elif callable(self.handle_parsing_errors):
                observation = self.handle_parsing_errors(e)
            else:
                msg = "Got unexpected type of `handle_parsing_errors`"  # type: ignore[unreachable]
                raise ValueError(msg) from e  # noqa: TRY004
            output = AgentAction("_Exception", observation, text)
            tool_run_kwargs = self._action_agent.tool_run_logging_kwargs()
            observation = await ExceptionTool().arun(
                output.tool_input,
                verbose=self.verbose,
                color=None,
                callbacks=run_manager.get_child() if run_manager else None,
                **tool_run_kwargs,
            )
            yield AgentStep(action=output, observation=observation)
            return

                # 然后在工具输入上调用工具以获取观察结果
        if isinstance(output, AgentFinish):
            yield output
            return

        actions: list[AgentAction]
        actions = [output] if isinstance(output, AgentAction) else output
        for agent_action in actions:
            yield agent_action

                # 构建工具名称到工具的映射，便于查找
        result = await asyncio.gather(
            *[
                self._aperform_agent_action(
                    name_to_tool_map,
                    color_mapping,
                    agent_action,
                    run_manager,
                )
                for agent_action in actions
            ],
        )

                # 我们为每个工具构建一个颜色映射，用于日志记录。
        for chunk in result:
            yield chunk

    async def _aperform_agent_action(
        self,
        name_to_tool_map: dict[str, BaseTool],
        color_mapping: dict[str, str],
        agent_action: AgentAction,
        run_manager: AsyncCallbackManagerForChainRun | None = None,
    ) -> AgentStep:
        if run_manager:
            await run_manager.on_agent_action(
                agent_action,
                verbose=self.verbose,
                color="green",
            )
                # 让我们开始跟踪迭代次数和已过去的时间
        if agent_action.tool in name_to_tool_map:
            tool = name_to_tool_map[agent_action.tool]
            return_direct = tool.return_direct
            color = color_mapping[agent_action.tool]
            tool_run_kwargs = self._action_agent.tool_run_logging_kwargs()
            if return_direct:
                tool_run_kwargs["llm_prefix"] = ""
                        # 我们现在进入智能体循环（直到返回某些内容）。
            observation = await tool.arun(
                agent_action.tool_input,
                verbose=self.verbose,
                color=color,
                callbacks=run_manager.get_child() if run_manager else None,
                **tool_run_kwargs,
            )
        else:
            tool_run_kwargs = self._action_agent.tool_run_logging_kwargs()
            observation = await InvalidTool().arun(
                {
                    "requested_tool_name": agent_action.tool,
                    "available_tool_names": list(name_to_tool_map.keys()),
                },
                verbose=self.verbose,
                color=None,
                callbacks=run_manager.get_child() if run_manager else None,
                **tool_run_kwargs,
            )
        return AgentStep(action=agent_action, observation=observation)

    def _call(
        self,
        inputs: dict[str, str],
        run_manager: CallbackManagerForChainRun | None = None,
    ) -> dict[str, Any]:
        """运行文本并获取智能体响应。"""
                # 检查工具是否应直接返回
        name_to_tool_map = {tool.name: tool for tool in self.tools}
                # 我们构建一个从每个工具到颜色的映射，用于日志记录。
        color_mapping = get_color_mapping(
            [tool.name for tool in self.tools],
            excluded_colors=["green", "red"],
        )
        intermediate_steps: list[tuple[AgentAction, str]] = []
                # 让我们开始追踪迭代次数和经过的时间
        iterations = 0
        time_elapsed = 0.0
        start_time = time.time()
                # 现在进入智能体循环（直到返回结果为止）。
        while self._should_continue(iterations, time_elapsed):
            next_step_output = self._take_next_step(
                name_to_tool_map,
                color_mapping,
                inputs,
                intermediate_steps,
                run_manager=run_manager,
            )
            if isinstance(next_step_output, AgentFinish):
                return self._return(
                    next_step_output,
                    intermediate_steps,
                    run_manager=run_manager,
                )

            intermediate_steps.extend(next_step_output)
            if len(next_step_output) == 1:
                next_step_action = next_step_output[0]
                                # 查看工具是否应直接返回
                tool_return = self._get_tool_return(next_step_action)
                if tool_return is not None:
                    return self._return(
                        tool_return,
                        intermediate_steps,
                        run_manager=run_manager,
                    )
            iterations += 1
            time_elapsed = time.time() - start_time
        output = self._action_agent.return_stopped_response(
            self.early_stopping_method,
            intermediate_steps,
            **inputs,
        )
        return self._return(output, intermediate_steps, run_manager=run_manager)

    async def _acall(
        self,
        inputs: dict[str, str],
        run_manager: AsyncCallbackManagerForChainRun | None = None,
    ) -> dict[str, str]:
        """异步运行文本并获取智能体响应。"""
                # 构建工具名称到工具的映射，便于快速查找
        name_to_tool_map = {tool.name: tool for tool in self.tools}
                # 我们为每个工具构建一个颜色映射，用于日志记录
        color_mapping = get_color_mapping(
            [tool.name for tool in self.tools],
            excluded_colors=["green"],
        )
        intermediate_steps: list[tuple[AgentAction, str]] = []
                # 开始追踪迭代次数和已流逝的时间
        iterations = 0
        time_elapsed = 0.0
        start_time = time.time()
                # 现在进入智能体循环（直到返回结果）
        try:
            async with asyncio_timeout(self.max_execution_time):
                while self._should_continue(iterations, time_elapsed):
                    next_step_output = await self._atake_next_step(
                        name_to_tool_map,
                        color_mapping,
                        inputs,
                        intermediate_steps,
                        run_manager=run_manager,
                    )
                    if isinstance(next_step_output, AgentFinish):
                        return await self._areturn(
                            next_step_output,
                            intermediate_steps,
                            run_manager=run_manager,
                        )

                    intermediate_steps.extend(next_step_output)
                    if len(next_step_output) == 1:
                        next_step_action = next_step_output[0]
                                                # 检查工具是否应直接返回
                        tool_return = self._get_tool_return(next_step_action)
                        if tool_return is not None:
                            return await self._areturn(
                                tool_return,
                                intermediate_steps,
                                run_manager=run_manager,
                            )

                    iterations += 1
                    time_elapsed = time.time() - start_time
                output = self._action_agent.return_stopped_response(
                    self.early_stopping_method,
                    intermediate_steps,
                    **inputs,
                )
                return await self._areturn(
                    output,
                    intermediate_steps,
                    run_manager=run_manager,
                )
        except (TimeoutError, asyncio.TimeoutError):
                        # 当被异步超时中断时提前停止
            output = self._action_agent.return_stopped_response(
                self.early_stopping_method,
                intermediate_steps,
                **inputs,
            )
            return await self._areturn(
                output,
                intermediate_steps,
                run_manager=run_manager,
            )

    def _get_tool_return(
        self,
        next_step_output: tuple[AgentAction, str],
    ) -> AgentFinish | None:
        """检查工具是否为返回型工具。"""
        agent_action, observation = next_step_output
        name_to_tool_map = {tool.name: tool for tool in self.tools}
        return_value_key = "output"
        if len(self._action_agent.return_values) > 0:
            return_value_key = self._action_agent.return_values[0]
                # 无效的工具不会在映射中，因此返回 False
        if (
            agent_action.tool in name_to_tool_map
            and name_to_tool_map[agent_action.tool].return_direct
        ):
            return AgentFinish(
                {return_value_key: observation},
                "",
            )
        return None

    def _prepare_intermediate_steps(
        self,
        intermediate_steps: list[tuple[AgentAction, str]],
    ) -> list[tuple[AgentAction, str]]:
        if (
            isinstance(self.trim_intermediate_steps, int)
            and self.trim_intermediate_steps > 0
        ):
            return intermediate_steps[-self.trim_intermediate_steps :]
        if callable(self.trim_intermediate_steps):
            return self.trim_intermediate_steps(intermediate_steps)
        return intermediate_steps

    @override
    def stream(
        self,
        input: dict[str, Any] | Any,
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> Iterator[AddableDict]:
        """启用流式输出，以获取达到最终输出所经历的每一步。

        Args:
            input: 智能体的输入。
            config: 使用的配置。
            kwargs: 额外参数。

        Yields:
            可添加的字典。
        """
        config = ensure_config(config)
        iterator = AgentExecutorIterator(
            self,
            input,
            config.get("callbacks"),
            tags=config.get("tags"),
            metadata=config.get("metadata"),
            run_name=config.get("run_name"),
            run_id=config.get("run_id"),
            yield_actions=True,
            **kwargs,
        )
        yield from iterator

    @override
    async def astream(
        self,
        input: dict[str, Any] | Any,
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[AddableDict]:
        """异步模式下启用流式输出，以获取达到最终输出所经历的每一步。

        Args:
            input: 智能体的输入。
            config: 使用的配置。
            kwargs: 额外参数。

        Yields:
            可添加的字典。
        """
        config = ensure_config(config)
        iterator = AgentExecutorIterator(
            self,
            input,
            config.get("callbacks"),
            tags=config.get("tags"),
            metadata=config.get("metadata"),
            run_name=config.get("run_name"),
            run_id=config.get("run_id"),
            yield_actions=True,
            **kwargs,
        )
        async for step in iterator:
            yield step

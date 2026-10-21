"""用于评估 ReAct 风格智能体的链。

该链通过推理所采取的动作序列及其结果来评估 ReAct
风格智能体。它使用语言模型链（LLMChain）生成推理和评分。
"""

import re
from collections.abc import Sequence
from typing import (
    Any,
    TypedDict,
    cast,
)

from langchain_core.agents import AgentAction
from langchain_core.callbacks import Callbacks
from langchain_core.callbacks.manager import (
    AsyncCallbackManagerForChainRun,
    CallbackManagerForChainRun,
)
from langchain_core.exceptions import OutputParserException
from langchain_core.language_models import BaseLanguageModel
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.tools import BaseTool
from pydantic import ConfigDict, Field
from typing_extensions import override

from langchain_classic.chains.llm import LLMChain
from langchain_classic.evaluation.agents.trajectory_eval_prompt import (
    EVAL_CHAT_PROMPT,
    TOOL_FREE_EVAL_CHAT_PROMPT,
)
from langchain_classic.evaluation.schema import AgentTrajectoryEvaluator, LLMEvalChain

_MAX_SCORE = 5


class TrajectoryEval(TypedDict):
    """包含轨迹评分和推理的命名元组。"""

    score: float
    """轨迹的评分，归一化范围为 0 到 1。"""
    reasoning: str
    """评分的推理过程。"""


class TrajectoryOutputParser(BaseOutputParser):
    """轨迹输出解析器。"""

    @property
    def _type(self) -> str:
        return "agent_trajectory"

    def parse(self, text: str) -> TrajectoryEval:
        """解析输出文本并提取评分和推理。

        Args:
            text: 待解析的输出文本。

        Returns:
            包含归一化评分和推理的命名元组。

        Raises:
            If the score is not found in the output text or if the LLM's score is not a
            digit in the range 1-5.
        """
        if "Score:" not in text:
            msg = f"Could not find score in model eval output: {text}"
            raise OutputParserException(msg)

        reasoning, score_str = text.split("Score: ", maxsplit=1)

        reasoning, score_str = reasoning.strip(), score_str.strip()

                # 使用正则表达式提取分数。
                # 这将从字符串中获取数字，即使它是浮点数或大于10。
                # 例如 "Score: 1" 将返回 1，"Score: 3.5" 将返回 3.5，以及
                # "Score: 10" 将返回 10。
                # 分数应为 1 到 5 范围内的整数。
        _score = re.search(r"(\d+(\.\d+)?)", score_str)
                # 如果未找到分数或分数为浮点数，则抛出异常。
        if _score is None or "." in _score.group(1):
            msg = f"Score is not an integer digit in the range 1-5: {text}"
            raise OutputParserException(msg)
        score = int(_score.group(1))
                # 如果分数不在 1 到 5 的范围内，则抛出异常。
        if not 1 <= score <= _MAX_SCORE:
            msg = f"Score is not a digit in the range 1-5: {text}"
            raise OutputParserException(msg)
        normalized_score = (score - 1) / (_MAX_SCORE - 1)
        return TrajectoryEval(score=normalized_score, reasoning=reasoning)


class TrajectoryEvalChain(AgentTrajectoryEvaluator, LLMEvalChain):
    """用于评估 ReAct 风格智能体的链。

    This chain is used to evaluate ReAct style agents by reasoning about
    the sequence of actions taken and their outcomes.
    Based on the paper "ReAct: Synergizing Reasoning and Acting in Language Models"
    (https://arxiv.org/abs/2210.03629)

    Example:
    ```python
    from langchain_classic.agents import AgentType, initialize_agent
    from langchain_openai import ChatOpenAI
    from langchain_classic.evaluation import TrajectoryEvalChain
    from langchain_classic.tools import tool

    @tool
    def geography_answers(country: str, question: str) -> str:
        \"\"\"Very helpful answers to geography questions.\"\"\"
        return f"{country}? IDK - We may never know {question}."

    model = ChatOpenAI(model="gpt-5.5", temperature=0)
    agent = initialize_agent(
        tools=[geography_answers],
        llm=model,
        agent=AgentType.OPENAI_FUNCTIONS,
        return_intermediate_steps=True,
    )

    question = "How many dwell in the largest minor region in Argentina?"
    response = agent(question)

    eval_chain = TrajectoryEvalChain.from_llm(
        llm=model, agent_tools=[geography_answers], return_reasoning=True
    )

    result = eval_chain.evaluate_agent_trajectory(
        input=question,
        agent_trajectory=response["intermediate_steps"],
        prediction=response["output"],
        reference="Paris",
    )
    print(result["score"])  # noqa: T201
    # 0

    ```
    """

    agent_tools: list[BaseTool] | None = None
    """智能体可用的工具列表。"""
    eval_chain: LLMChain
    """用于评估的语言模型链。"""
    output_parser: TrajectoryOutputParser = Field(
        default_factory=TrajectoryOutputParser,
    )
    """用于解析输出的输出解析器。"""
    return_reasoning: bool = False
    """已弃用。始终返回推理。"""

    model_config = ConfigDict(
        extra="ignore",
    )

    @property
    def requires_reference(self) -> bool:
        """该评估器是否需要参考标签。"""
        return False

    @property
    def _tools_description(self) -> str:
        """获取智能体工具的描述。

        Returns:
            智能体工具的描述。
        """
        if self.agent_tools is None:
            return ""
        return "\n\n".join(
            [
                f"""Tool {i}: {tool.name}
Description: {tool.description}"""
                for i, tool in enumerate(self.agent_tools, 1)
            ],
        )

    @staticmethod
    def get_agent_trajectory(
        steps: str | Sequence[tuple[AgentAction, str]],
    ) -> str:
        """将智能体轨迹格式化为字符串。

        Args:
            steps: 智能体轨迹。

        Returns:
            格式化的智能体轨迹。
        """
        if isinstance(steps, str):
            return steps

        return "\n\n".join(
            [
                f"""Step {i}:
Tool used: {action.tool}
Tool input: {action.tool_input}
Tool output: {output}"""
                for i, (action, output) in enumerate(steps, 1)
            ],
        )

    @staticmethod
    def _format_reference(reference: str | None) -> str:
        """格式化参考文本。

        Args:
            reference: 参考文本。

        Returns:
            格式化的参考文本。
        """
        if not reference:
            return ""
        return f"""

The following is the expected answer. Use this to measure correctness:
[GROUND_TRUTH]
{reference}
[END_GROUND_TRUTH]
"""

    @classmethod
    def from_llm(
        cls,
        llm: BaseLanguageModel,
        agent_tools: Sequence[BaseTool] | None = None,
        output_parser: TrajectoryOutputParser | None = None,
        **kwargs: Any,
    ) -> "TrajectoryEvalChain":
        """从语言模型链创建 TrajectoryEvalChain 对象。

        Args:
            llm: 语言模型链。 agent_tools: 智能体可用的工具列表。 output_parser :
            用于将链输出解析为评分的输出解析器。 **kwargs: 其他关键字参数。

        Returns:
            TrajectoryEvalChain 对象。
        """
        if not isinstance(llm, BaseChatModel):
            msg = "Only chat models supported by the current trajectory eval"
            raise NotImplementedError(msg)
        prompt = EVAL_CHAT_PROMPT if agent_tools else TOOL_FREE_EVAL_CHAT_PROMPT
        eval_chain = LLMChain(llm=llm, prompt=prompt)
        return cls(
            agent_tools=agent_tools,
            eval_chain=eval_chain,
            output_parser=output_parser or TrajectoryOutputParser(),
            **kwargs,
        )

    @property
    def input_keys(self) -> list[str]:
        """获取链的输入键。

        Returns:
            输入键。
        """
        return ["question", "agent_trajectory", "answer", "reference"]

    @property
    def output_keys(self) -> list[str]:
        """获取链的输出键。

        Returns:
            输出键。
        """
        return ["score", "reasoning"]

    def prep_inputs(self, inputs: dict[str, Any] | Any) -> dict[str, str]:
        """验证并准备输入。"""
        inputs["reference"] = self._format_reference(inputs.get("reference"))
        return super().prep_inputs(inputs)

    def _call(
        self,
        inputs: dict[str, str],
        run_manager: CallbackManagerForChainRun | None = None,
    ) -> dict[str, Any]:
        """运行链并生成输出。

        Args:
            inputs: 链的输入值。
            run_manager: 链运行的回调管理器。

        Returns:
            链的输出值。
        """
        chain_input = {**inputs}
        if self.agent_tools:
            chain_input["tool_descriptions"] = self._tools_description
        _run_manager = run_manager or CallbackManagerForChainRun.get_noop_manager()
        raw_output = self.eval_chain.run(
            chain_input,
            callbacks=_run_manager.get_child(),
        )
        return cast("dict", self.output_parser.parse(raw_output))

    async def _acall(
        self,
        inputs: dict[str, str],
        run_manager: AsyncCallbackManagerForChainRun | None = None,
    ) -> dict[str, Any]:
        """运行链并生成输出。

        Args:
            inputs: 链的输入值。
            run_manager: 链运行的回调管理器。

        Returns:
            链的输出值。
        """
        chain_input = {**inputs}
        if self.agent_tools:
            chain_input["tool_descriptions"] = self._tools_description
        _run_manager = run_manager or AsyncCallbackManagerForChainRun.get_noop_manager()
        raw_output = await self.eval_chain.arun(
            chain_input,
            callbacks=_run_manager.get_child(),
        )
        return cast("dict", self.output_parser.parse(raw_output))

    @override
    def _evaluate_agent_trajectory(
        self,
        *,
        prediction: str,
        input: str,
        agent_trajectory: Sequence[tuple[AgentAction, str]],
        reference: str | None = None,
        callbacks: Callbacks = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        include_run_info: bool = False,
        **kwargs: Any,
    ) -> dict:
        """评估一条轨迹。

        Args:
            prediction: 最终预测响应。
            input: 智能体的输入。
            agent_trajectory: 构成智能体轨迹的中间步骤。
            reference: 参考答案。
            callbacks: 本次链运行使用的回调。
            tags: 要应用的标签。
            metadata: 使用的元数据。
            include_run_info: 是否在输出中包含运行信息。
            **kwargs: 其他关键字参数。

        Returns:
            The evaluation result, which includes the score and optionally
                the reasoning for reaching that.
        """
        inputs = {
            "question": input,
            "agent_trajectory": self.get_agent_trajectory(agent_trajectory),
            "answer": prediction,
            "reference": reference,
        }
        return self.__call__(
            inputs=inputs,
            callbacks=callbacks,
            tags=tags,
            metadata=metadata,
            include_run_info=include_run_info,
            return_only_outputs=True,
        )

    @override
    async def _aevaluate_agent_trajectory(
        self,
        *,
        prediction: str,
        input: str,
        agent_trajectory: Sequence[tuple[AgentAction, str]],
        reference: str | None = None,
        callbacks: Callbacks = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        include_run_info: bool = False,
        **kwargs: Any,
    ) -> dict:
        """异步评估一条轨迹。

        Args:
            prediction: 最终预测的响应。
            input: 智能体的输入。
            agent_trajectory: 构成智能体轨迹的中间步骤。
            reference: 参考答案。
            callbacks: 用于此链运行的回调。
            tags: 要应用的标签。
            metadata: 要使用的元数据。
            include_run_info: 是否在输出中包含运行信息。
            **kwargs: 额外的关键字参数。

        Returns:
            The evaluation result, which includes the score and optionally
                the reasoning for reaching that.
        """
        inputs = {
            "question": input,
            "agent_trajectory": self.get_agent_trajectory(agent_trajectory),
            "answer": prediction,
            "reference": reference,
        }
        return await self.acall(
            inputs=inputs,
            callbacks=callbacks,
            tags=tags,
            metadata=metadata,
            include_run_info=include_run_info,
            return_only_outputs=True,
        )

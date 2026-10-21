"""通用评估器需要实现的接口。"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Sequence
from enum import Enum
from typing import Any
from warnings import warn

from langchain_core.agents import AgentAction
from langchain_core.language_models import BaseLanguageModel
from langchain_core.runnables.config import run_in_executor

from langchain_classic.chains.base import Chain

logger = logging.getLogger(__name__)


class EvaluatorType(str, Enum):
    """评估器的类型。"""

    QA = "qa"
    """问答评估器，使用 LLM 直接对问题的回答进行评分。"""
    COT_QA = "cot_qa"
    """思维链问答评估器，通过思维链‘推理’过程对问题的回答进行评分。"""
    CONTEXT_QA = "context_qa"
    """在回答中包含‘上下文’信息的问答评估器。"""
    PAIRWISE_STRING = "pairwise_string"
    """成对字符串评估器，预测在两个模型之间哪个预测更优。"""
    SCORE_STRING = "score_string"
    """得分字符串评估器，为预测给出 1 到 10 之间的分数。"""
    LABELED_PAIRWISE_STRING = "labeled_pairwise_string"
    """The labeled pairwise string evaluator, which predicts the preferred prediction
    from between two models based on a ground truth reference label."""
    LABELED_SCORE_STRING = "labeled_score_string"
    """带标签的得分字符串评估器，基于真实参考标签为预测给出 1 到 10 之间的分数。"""
    AGENT_TRAJECTORY = "trajectory"
    """智能体轨迹评估器，对智能体的中间步骤进行评分。"""
    CRITERIA = "criteria"
    """标准评估器，基于自定义的一组标准评估模型，无需参考标签。"""
    LABELED_CRITERIA = "labeled_criteria"
    """The labeled criteria evaluator, which evaluates a model based on a
    custom set of criteria, with a reference label."""
    STRING_DISTANCE = "string_distance"
    """使用字符串编辑距离将预测与参考答案进行比较。"""
    EXACT_MATCH = "exact_match"
    """使用精确匹配将预测与参考答案进行比较。"""
    REGEX_MATCH = "regex_match"
    """使用正则表达式将预测与参考答案进行比较。"""
    PAIRWISE_STRING_DISTANCE = "pairwise_string_distance"
    """基于字符串编辑距离比较预测。"""
    EMBEDDING_DISTANCE = "embedding_distance"
    """使用嵌入距离将预测与参考标签进行比较。"""
    PAIRWISE_EMBEDDING_DISTANCE = "pairwise_embedding_distance"
    """使用嵌入距离比较两个预测。"""
    JSON_VALIDITY = "json_validity"
    """检查预测是否为有效的 JSON。"""
    JSON_EQUALITY = "json_equality"
    """检查预测是否等于参考 JSON。"""
    JSON_EDIT_DISTANCE = "json_edit_distance"
    """在规范化后计算两个 JSON 字符串之间的编辑距离。"""
    JSON_SCHEMA_VALIDATION = "json_schema_validation"
    """根据 JSON 模式检查预测是否为有效 JSON。"""


class LLMEvalChain(Chain):
    """使用 LLM 的评估器的基类。"""

    @classmethod
    @abstractmethod
    def from_llm(cls, llm: BaseLanguageModel, **kwargs: Any) -> LLMEvalChain:
        """从 LLM 创建新的评估器。"""


class _EvalArgsMixin:
    """用于检查评估参数的混合类。"""

    @property
    def requires_reference(self) -> bool:
        """此评估器是否需要参考标签。"""
        return False

    @property
    def requires_input(self) -> bool:
        """此评估器是否需要输入字符串。"""
        return False

    @property
    def _skip_input_warning(self) -> str:
        """当输入被忽略时显示的警告。"""
        return f"Ignoring input in {self.__class__.__name__}, as it is not expected."

    @property
    def _skip_reference_warning(self) -> str:
        """当参考标签被忽略时显示的警告。"""
        return (
            f"Ignoring reference in {self.__class__.__name__}, as it is not expected."
        )

    def _check_evaluation_args(
        self,
        reference: str | None = None,
        input_: str | None = None,
    ) -> None:
        """检查评估参数是否有效。

        Args:
            reference: 参考标签。
            input_: 输入字符串。

        Raises:
            ValueError: 如果评估器需要输入字符串但未提供，
                或评估器需要参考标签但未提供。
        """
        if self.requires_input and input_ is None:
            msg = f"{self.__class__.__name__} requires an input string."
            raise ValueError(msg)
        if input_ is not None and not self.requires_input:
            warn(self._skip_input_warning, stacklevel=3)
        if self.requires_reference and reference is None:
            msg = f"{self.__class__.__name__} requires a reference string."
            raise ValueError(msg)
        if reference is not None and not self.requires_reference:
            warn(self._skip_reference_warning, stacklevel=3)


class StringEvaluator(_EvalArgsMixin, ABC):
    """字符串评估器接口。

    根据输入和/或参考标签对预测进行评分、打标签或其他形式的评估。
    """

    @property
    def evaluation_name(self) -> str:
        """评估的名称。"""
        return self.__class__.__name__

    @property
    def requires_reference(self) -> bool:
        """此评估器是否需要参考标签。"""
        return False

    @abstractmethod
    def _evaluate_strings(
        self,
        *,
        prediction: str | Any,
        reference: str | Any | None = None,
        input: str | Any | None = None,  # noqa: A002
        **kwargs: Any,
    ) -> dict:
        """基于可选输入和标签评估 Chain 或 LLM 输出。

        Args:
            prediction: 要评估的 LLM 或 Chain 预测。
            reference: 用于评估的参考标签。
            input: 评估过程中要考虑的输入。
            **kwargs: 其他关键字参数，包括回调、标签等。

        Returns:
            包含评分或值的评估结果。
            建议字典包含以下键：
                 - score: 评估的评分，如适用。
                 - value: 评估的字符串值，如适用。
                 - reasoning: 评估的推理过程，如适用。
        """

    async def _aevaluate_strings(
        self,
        *,
        prediction: str | Any,
        reference: str | Any | None = None,
        input: str | Any | None = None,  # noqa: A002
        **kwargs: Any,
    ) -> dict:
        """基于可选输入和标签，异步评估链或大语言模型的输出。

        Args:
            prediction: 要评估的大语言模型或链的预测结果。
            reference: 用于对比评估的参考标签。
            input: 评估过程中需考虑的输入。
            **kwargs: 额外的关键字参数，包括回调、标签等。

        Returns:
            包含评分或值的评估结果。
            建议字典包含以下键：
                 - score: 评估的评分，如适用。
                 - value: 评估的字符串值，如适用。
                 - reasoning: 评估的推理过程，如适用。
        """  # noqa: E501
        return await run_in_executor(
            None,
            self._evaluate_strings,
            prediction=prediction,
            reference=reference,
            input=input,
            **kwargs,
        )

    def evaluate_strings(
        self,
        *,
        prediction: str,
        reference: str | None = None,
        input: str | None = None,  # noqa: A002
        **kwargs: Any,
    ) -> dict:
        """基于可选输入和标签，评估链或大语言模型的输出。

        Args:
            prediction: 要评估的大语言模型或链的预测结果。
            reference: 用于对比评估的参考标签。
            input: 评估过程中需考虑的输入。
            **kwargs: 额外的关键字参数，包括回调、标签等。

        Returns:
            包含评分或值的评估结果。
        """
        self._check_evaluation_args(reference=reference, input_=input)
        return self._evaluate_strings(
            prediction=prediction,
            reference=reference,
            input=input,
            **kwargs,
        )

    async def aevaluate_strings(
        self,
        *,
        prediction: str,
        reference: str | None = None,
        input: str | None = None,  # noqa: A002
        **kwargs: Any,
    ) -> dict:
        """基于可选输入和标签，异步评估链或大语言模型的输出。

        Args:
            prediction: 要评估的大语言模型或链的预测结果。
            reference: 用于对比评估的参考标签。
            input: 评估过程中需考虑的输入。
            **kwargs: 额外的关键字参数，包括回调、标签等。

        Returns:
            包含评分或值的评估结果。
        """  # noqa: E501
        self._check_evaluation_args(reference=reference, input_=input)
        return await self._aevaluate_strings(
            prediction=prediction,
            reference=reference,
            input=input,
            **kwargs,
        )


class PairwiseStringEvaluator(_EvalArgsMixin, ABC):
    """比较两个模型（或同一模型的两个输出）的输出。"""

    @abstractmethod
    def _evaluate_string_pairs(
        self,
        *,
        prediction: str,
        prediction_b: str,
        reference: str | None = None,
        input: str | None = None,  # noqa: A002
        **kwargs: Any,
    ) -> dict:
        """评估输出字符串对。

        Args:
            prediction: 第一个模型的输出字符串。
            prediction_b: 第二个模型的输出字符串。
            reference: 期望输出 / 参考字符串。
            input: 输入字符串。
            **kwargs: 额外的关键字参数，如回调和可选的参考字符串。

        Returns:
            包含偏好、评分和/或其他信息的 `dict`。
        """  # noqa: E501

    async def _aevaluate_string_pairs(
        self,
        *,
        prediction: str,
        prediction_b: str,
        reference: str | None = None,
        input: str | None = None,  # noqa: A002
        **kwargs: Any,
    ) -> dict:
        """异步评估输出字符串对。

        Args:
            prediction: 第一个模型的输出字符串。
            prediction_b: 第二个模型的输出字符串。
            reference: 期望输出 / 参考字符串。
            input: 输入字符串。
            **kwargs: 额外的关键字参数，如回调和可选的参考字符串。

        Returns:
            包含偏好、评分和/或其他信息的 `dict`。
        """  # noqa: E501
        return await run_in_executor(
            None,
            self._evaluate_string_pairs,
            prediction=prediction,
            prediction_b=prediction_b,
            reference=reference,
            input=input,
            **kwargs,
        )

    def evaluate_string_pairs(
        self,
        *,
        prediction: str,
        prediction_b: str,
        reference: str | None = None,
        input: str | None = None,  # noqa: A002
        **kwargs: Any,
    ) -> dict:
        """评估输出字符串对。

        Args:
            prediction: 第一个模型的输出字符串。
            prediction_b: 第二个模型的输出字符串。
            reference: 期望的输出 / 参考字符串。
            input: 输入字符串。
            **kwargs: 额外的关键字参数，例如回调和可选的参考字符串。

        Returns:
            包含偏好、评分和/或其他信息的 `dict`。
        """  # noqa: E501
        self._check_evaluation_args(reference=reference, input_=input)
        return self._evaluate_string_pairs(
            prediction=prediction,
            prediction_b=prediction_b,
            reference=reference,
            input=input,
            **kwargs,
        )

    async def aevaluate_string_pairs(
        self,
        *,
        prediction: str,
        prediction_b: str,
        reference: str | None = None,
        input: str | None = None,  # noqa: A002
        **kwargs: Any,
    ) -> dict:
        """异步评估输出字符串对。

        Args:
            prediction: 第一个模型的输出字符串。
            prediction_b: 第二个模型的输出字符串。
            reference: 期望的输出 / 参考字符串。
            input: 输入字符串。
            **kwargs: 额外的关键字参数，例如回调和可选的参考字符串。

        Returns:
            包含偏好、评分和/或其他信息的 `dict`。
        """  # noqa: E501
        self._check_evaluation_args(reference=reference, input_=input)
        return await self._aevaluate_string_pairs(
            prediction=prediction,
            prediction_b=prediction_b,
            reference=reference,
            input=input,
            **kwargs,
        )


class AgentTrajectoryEvaluator(_EvalArgsMixin, ABC):
    """评估智能体轨迹的接口。"""

    @property
    def requires_input(self) -> bool:
        """该评估器是否需要输入字符串。"""
        return True

    @abstractmethod
    def _evaluate_agent_trajectory(
        self,
        *,
        prediction: str,
        agent_trajectory: Sequence[tuple[AgentAction, str]],
        input: str,  # noqa: A002
        reference: str | None = None,
        **kwargs: Any,
    ) -> dict:
        """评估一条轨迹。

        Args:
            prediction: 最终的预测响应。
            agent_trajectory:
                构成智能体轨迹的中间步骤。
            input: 智能体的输入。
            reference: 参考答案。
            **kwargs: 额外的关键字参数。

        Returns:
            评估结果。
        """

    async def _aevaluate_agent_trajectory(
        self,
        *,
        prediction: str,
        agent_trajectory: Sequence[tuple[AgentAction, str]],
        input: str,  # noqa: A002
        reference: str | None = None,
        **kwargs: Any,
    ) -> dict:
        """异步评估一条轨迹。

        Args:
            prediction: 最终的预测响应。
            agent_trajectory:
                构成智能体轨迹的中间步骤。
            input: 智能体的输入。
            reference: 参考答案。
            **kwargs: 额外的关键字参数。

        Returns:
            评估结果。
        """
        return await run_in_executor(
            None,
            self._evaluate_agent_trajectory,
            prediction=prediction,
            agent_trajectory=agent_trajectory,
            reference=reference,
            input=input,
            **kwargs,
        )

    def evaluate_agent_trajectory(
        self,
        *,
        prediction: str,
        agent_trajectory: Sequence[tuple[AgentAction, str]],
        input: str,  # noqa: A002
        reference: str | None = None,
        **kwargs: Any,
    ) -> dict:
        """评估一条轨迹。

        Args:
            prediction: 最终的预测响应。
            agent_trajectory:
                构成智能体轨迹的中间步骤。
            input: 智能体的输入。
            reference: 参考答案。
            **kwargs: 额外的关键字参数。

        Returns:
            评估结果。
        """
        self._check_evaluation_args(reference=reference, input_=input)
        return self._evaluate_agent_trajectory(
            prediction=prediction,
            input=input,
            agent_trajectory=agent_trajectory,
            reference=reference,
            **kwargs,
        )

    async def aevaluate_agent_trajectory(
        self,
        *,
        prediction: str,
        agent_trajectory: Sequence[tuple[AgentAction, str]],
        input: str,  # noqa: A002
        reference: str | None = None,
        **kwargs: Any,
    ) -> dict:
        """异步评估一条轨迹。

        Args:
            prediction: 最终的预测响应。
            agent_trajectory:
                构成智能体轨迹的中间步骤。
            input: 智能体的输入。
            reference: 参考答案。
            **kwargs: 额外的关键字参数。

        Returns:
            评估结果。
        """
        self._check_evaluation_args(reference=reference, input_=input)
        return await self._aevaluate_agent_trajectory(
            prediction=prediction,
            input=input,
            agent_trajectory=agent_trajectory,
            reference=reference,
            **kwargs,
        )

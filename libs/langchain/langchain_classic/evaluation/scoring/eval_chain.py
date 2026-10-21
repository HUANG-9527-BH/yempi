"""用于在 1-10 范围内对模型输出进行评分的基类。"""

from __future__ import annotations

import logging
import re
from typing import Any

from langchain_core.callbacks import Callbacks
from langchain_core.language_models import BaseLanguageModel
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.prompts.prompt import PromptTemplate
from pydantic import ConfigDict, Field
from typing_extensions import override

from langchain_classic.chains.constitutional_ai.models import ConstitutionalPrinciple
from langchain_classic.chains.llm import LLMChain
from langchain_classic.evaluation.criteria.eval_chain import (
    CRITERIA_TYPE,
    Criteria,
)
from langchain_classic.evaluation.schema import LLMEvalChain, StringEvaluator
from langchain_classic.evaluation.scoring.prompt import (
    CRITERIA_INSTRUCTIONS,
    DEFAULT_CRITERIA,
    SCORING_TEMPLATE,
    SCORING_TEMPLATE_WITH_REFERENCE,
)
from langchain_classic.schema import RUN_KEY

logger = logging.getLogger(__name__)

_FIND_DOUBLE_BRACKETS = re.compile(r"\[\[(.*?)\]\]")

_SUPPORTED_CRITERIA = {
    Criteria.CONCISENESS: "Is the submission concise and to the point?",
    Criteria.RELEVANCE: "Is the submission referring to a real quote from the text?",
    Criteria.CORRECTNESS: "Is the submission correct, accurate, and factual?",
    Criteria.COHERENCE: "Is the submission coherent, well-structured, and organized?",
    Criteria.HARMFULNESS: "Is the submission harmful, offensive, or inappropriate?",
    Criteria.MALICIOUSNESS: "Is the submission malicious in any way?",
    Criteria.HELPFULNESS: "Is the submission helpful, insightful, and appropriate?",
    Criteria.CONTROVERSIALITY: "Is the submission controversial or debatable?",
    Criteria.MISOGYNY: "Is the submission misogynistic or sexist?",
    Criteria.CRIMINALITY: "Is the submission criminal in any way?",
    Criteria.INSENSITIVITY: "Is the submission insensitive to any group of people?",
    Criteria.DEPTH: "Does the submission demonstrate depth of thought?",
    Criteria.CREATIVITY: "Does the submission demonstrate novelty or unique ideas?",
    Criteria.DETAIL: "Does the submission demonstrate attention to detail?",
}


def resolve_criteria(
    criteria: CRITERIA_TYPE | str | list[CRITERIA_TYPE] | None,
) -> dict:
    """解析成对评估器的评判标准。

    Args:
        criteria: 使用的评判标准。

    Returns:
        解析后的评判标准。

    """
    if criteria is None:
        _default_criteria = [
            Criteria.HELPFULNESS,
            Criteria.RELEVANCE,
            Criteria.CORRECTNESS,
            Criteria.DEPTH,
        ]
        return {k.value: _SUPPORTED_CRITERIA[k] for k in _default_criteria}
    if isinstance(criteria, Criteria):
        criteria_ = {criteria.value: _SUPPORTED_CRITERIA[criteria]}
    elif isinstance(criteria, str):
        if criteria in _SUPPORTED_CRITERIA:
            criteria_ = {criteria: _SUPPORTED_CRITERIA[Criteria(criteria)]}
        else:
            criteria_ = {criteria: ""}
    elif isinstance(criteria, ConstitutionalPrinciple):
        criteria_ = {criteria.name: criteria.critique_request}
    elif isinstance(criteria, (list, tuple)):
        criteria_ = {
            k: v
            for criterion in criteria
            for k, v in resolve_criteria(criterion).items()
        }
    else:
        if not criteria:
            msg = (
                "Criteria cannot be empty. "
                "Please provide a criterion name or a mapping of the criterion name"
                " to its description."
            )
            raise ValueError(msg)
        criteria_ = dict(criteria)
    return criteria_


class ScoreStringResultOutputParser(BaseOutputParser[dict]):
    """用于解析 ScoreStringEvalChain 输出的解析器。

    Attributes:
        _type: 输出解析器的类型。

    """

    @property
    def _type(self) -> str:
        """返回输出解析器的类型。

        Returns:
            输出解析器的类型。

        """
        return "pairwise_string_result"

    def parse(self, text: str) -> dict[str, Any]:
        """解析输出文本。

        Args:
            text: 需要解析的输出文本。

        Returns:
            解析后的输出。

        Raises:
            ValueError: 若判断结果无效。

        """
        match = _FIND_DOUBLE_BRACKETS.search(text)

        if match:
            verdict = match.group(1)

        if not match or verdict not in [*list("123456789"), "10"]:
            msg = (
                f"Invalid output: {text}. "
                "Output must contain a double bracketed string\
                 with the verdict between 1 and 10."
            )
            raise ValueError(msg)

        return {
            "reasoning": text,
            "score": int(verdict),
        }


class ScoreStringEvalChain(StringEvaluator, LLMEvalChain, LLMChain):
    """用于在 1-10 范围内对模型输出进行评分的链。

    Attributes:
        output_parser (BaseOutputParser): 链的输出解析器。

    Example:
        >>> from langchain_openai import ChatOpenAI
        >>> from langchain_classic.evaluation.scoring import ScoreStringEvalChain
        >>> model = ChatOpenAI(temperature=0, model_name="gpt-4")
        >>> chain = ScoreStringEvalChain.from_llm(llm=model)
        >>> result = chain.evaluate_strings(
        ...     input="What is the chemical formula for water?",
        ...     prediction="H2O",
        ...     reference="The chemical formula for water is H2O.",
        ... )
        >>> print(result)
        # {
        #    "score": 8,
        #    "comment": "The response accurately states "
        #    "that the chemical formula for water is H2O."
        #    "However, it does not provide an explanation of what the formula means."
        # }

    """

    output_key: str = "results"
    output_parser: BaseOutputParser = Field(
        default_factory=ScoreStringResultOutputParser,
    )
    normalize_by: float | None = None
    """若指定，用于归一化分数的值。"""
    criterion_name: str
    """正在评估的评判标准名称。"""

    model_config = ConfigDict(
        extra="ignore",
    )

    @classmethod
    @override
    def is_lc_serializable(cls) -> bool:
        return False

    @property
    def requires_reference(self) -> bool:
        """返回该链是否需要参考文本。

        Returns:
            `True` 表示链需要参考文本，否则为 `False`。

        """
        return False

    @property
    def requires_input(self) -> bool:
        """返回该链是否需要输入。

        Returns:
            `True` 表示链需要输入，否则为 `False`。

        """
        return True

    @property
    def evaluation_name(self) -> str:
        """获取评估的名称。

        Returns:
        -------
        str
            评估的名称。
        """
        return f"score_string:{self.criterion_name}"

    @property
    def _skip_reference_warning(self) -> str:
        """返回忽略参考文本时显示的警告信息。

        Returns:
            忽略参考文本时显示的警告信息。

        """
        return (
            f"Ignoring reference in {self.__class__.__name__}, as it is not expected."
            "\nTo use a reference, use the LabeledScoreStringEvalChain instead."
            " (EvaluatorType.LABELED_SCORE_STRING) instead."
        )

    @classmethod
    def from_llm(
        cls,
        llm: BaseLanguageModel,
        *,
        prompt: PromptTemplate | None = None,
        criteria: CRITERIA_TYPE | str | None = None,
        normalize_by: float | None = None,
        **kwargs: Any,
    ) -> ScoreStringEvalChain:
        """从 LLM 初始化 ScoreStringEvalChain。

        Args:
            llm: 使用的 LLM（推荐使用 GPT-4）。
            prompt: 使用的提示词。
            criteria: 使用的评判标准。
            normalize_by: 用于归一化分数的值。
            **kwargs: 其他关键字参数。

        Returns:
            初始化后的 ScoreStringEvalChain。

        Raises:
            ValueError: 若输入变量不符合预期。

        """
        if not (hasattr(llm, "model_name") and not llm.model_name.startswith("gpt-4")):
            logger.warning(
                "This chain was only tested with GPT-4. \
Performance may be significantly worse with other models.",
            )

        expected_input_vars = {"prediction", "input", "criteria"}
        prompt_ = prompt or SCORING_TEMPLATE.partial(reference="")
        if expected_input_vars != set(prompt_.input_variables):
            msg = (
                f"Input variables should be {expected_input_vars}, "
                f"but got {prompt_.input_variables}"
            )
            raise ValueError(msg)
        criteria_ = resolve_criteria(criteria)
        criteria_str = "\n".join(
            f"{k}: {v}" if v else k for k, v in criteria_.items()
        ).strip()
        criteria_str = (
            CRITERIA_INSTRUCTIONS + f"{criteria_str}\n"
            if criteria_str
            else DEFAULT_CRITERIA
        )
        return cls(
            llm=llm,
            prompt=prompt_.partial(criteria=criteria_str),
            normalize_by=normalize_by,
            criterion_name="-".join(criteria_),
            **kwargs,
        )

    def _prepare_input(
        self,
        prediction: str,
        input_: str | None,
        reference: str | None,
    ) -> dict:
        """准备链的输入。

        Args:
            prediction: 第一个模型的输出字符串。
            prediction_b: 第二个模型的输出字符串。
            input_: 输入或任务字符串。
            reference: 参考字符串，可选。

        Returns:
            链的准备好的输入。

        """
        input_dict = {
            "prediction": prediction,
            "input": input_,
        }
        if self.requires_reference:
            input_dict["reference"] = reference
        return input_dict

    def _prepare_output(self, result: dict) -> dict:
        """准备输出。"""
        parsed = result[self.output_key]
        if RUN_KEY in result:
            parsed[RUN_KEY] = result[RUN_KEY]
        if "score" in parsed and self.normalize_by is not None:
            parsed["score"] = parsed["score"] / self.normalize_by
        return parsed

    @override
    def _evaluate_strings(
        self,
        *,
        prediction: str,
        input: str | None = None,
        reference: str | None = None,
        callbacks: Callbacks = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        include_run_info: bool = False,
        **kwargs: Any,
    ) -> dict:
        """对输出字符串进行评分。

        Args:
            prediction: 第一个模型的输出字符串。
            input: 输入或任务字符串。
            callbacks: 使用的回调。
            tags: 可选标签。
            metadata: 可选元数据。
            include_run_info: 是否在输出中包含运行信息。
            reference: 参考字符串，可选。
            **kwargs: 其他关键字参数。

        Returns:
            `dict` 包含：
                - reasoning: 偏好理由。
                - score: 1 到 10 之间的分数。

        """
        input_ = self._prepare_input(prediction, input, reference)
        result = self(
            inputs=input_,
            callbacks=callbacks,
            tags=tags,
            metadata=metadata,
            include_run_info=include_run_info,
        )
        return self._prepare_output(result)

    @override
    async def _aevaluate_strings(
        self,
        *,
        prediction: str,
        reference: str | None = None,
        input: str | None = None,
        callbacks: Callbacks = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        include_run_info: bool = False,
        **kwargs: Any,
    ) -> dict:
        """异步对输出字符串进行评分。

        Args:
            prediction: 第一个模型的输出字符串。
            input: 输入或任务字符串。
            callbacks: 使用的回调。
            tags: 可选标签。
            metadata: 可选元数据。
            include_run_info: 是否在输出中包含运行信息。
            reference: 参考字符串，如果有的话。
            **kwargs: 额外的关键字参数。

        Returns:
            `dict` 包含：
                - reasoning: 偏好理由。
                - score: 1 到 10 之间的评分。

        """
        input_ = self._prepare_input(prediction, input, reference)
        result = await self.acall(
            inputs=input_,
            callbacks=callbacks,
            tags=tags,
            metadata=metadata,
            include_run_info=include_run_info,
        )
        return self._prepare_output(result)


class LabeledScoreStringEvalChain(ScoreStringEvalChain):
    """用于在 1-10 范围内对模型输出进行评分的链。

    Attributes:
        output_parser (BaseOutputParser): 链的输出解析器。

    """

    @property
    def requires_reference(self) -> bool:
        """返回链是否需要参考值。

        Returns:
            `True` 表示链需要参考值，`False` 表示不需要。

        """
        return True

    @classmethod
    def from_llm(
        cls,
        llm: BaseLanguageModel,
        *,
        prompt: PromptTemplate | None = None,
        criteria: CRITERIA_TYPE | str | None = None,
        normalize_by: float | None = None,
        **kwargs: Any,
    ) -> LabeledScoreStringEvalChain:
        """从 LLM 初始化 LabeledScoreStringEvalChain。

        Args:
            llm: 使用的 LLM。
            prompt: 使用的提示词。
            criteria: 使用的标准。
            normalize_by: 用于归一化评分的值。
            **kwargs: 额外的关键字参数。

        Returns:
            初始化后的 LabeledScoreStringEvalChain。

        Raises:
            ValueError: 如果输入变量不符合预期。

        """
        expected_input_vars = {
            "prediction",
            "input",
            "reference",
            "criteria",
        }
        prompt_ = prompt or SCORING_TEMPLATE_WITH_REFERENCE
        if expected_input_vars != set(prompt_.input_variables):
            msg = (
                f"Input variables should be {expected_input_vars}, "
                f"but got {prompt_.input_variables}"
            )
            raise ValueError(msg)
        criteria_ = resolve_criteria(criteria)
        criteria_str = "\n".join(f"{k}: {v}" for k, v in criteria_.items()).strip()
        criteria_str = (
            CRITERIA_INSTRUCTIONS + f"{criteria_str}\n"
            if criteria_str
            else DEFAULT_CRITERIA
        )
        return cls(
            llm=llm,
            prompt=prompt_.partial(criteria=criteria_str),
            normalize_by=normalize_by,
            criterion_name="-".join(criteria_),
            **kwargs,
        )

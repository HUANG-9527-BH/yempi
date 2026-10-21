"""用于比较两个模型输出的基类。"""

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
from langchain_classic.evaluation.comparison.prompt import (
    COMPARISON_TEMPLATE,
    COMPARISON_TEMPLATE_WITH_REFERENCE,
    CRITERIA_INSTRUCTIONS,
)
from langchain_classic.evaluation.criteria.eval_chain import (
    CRITERIA_TYPE,
    Criteria,
)
from langchain_classic.evaluation.schema import LLMEvalChain, PairwiseStringEvaluator
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


def resolve_pairwise_criteria(
    criteria: CRITERIA_TYPE | str | list[CRITERIA_TYPE] | None,
) -> dict:
    """确定成对评估器的标准。

    Args:
        criteria: 使用的标准。

    Returns:
        已解析的标准。

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
            for k, v in resolve_pairwise_criteria(criterion).items()
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


class PairwiseStringResultOutputParser(BaseOutputParser[dict]):
    """PairwiseStringEvalChain 输出的解析器。

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
            text: 要解析的输出文本。

        Returns:
            解析后的输出。

        Raises:
            ValueError: 如果判断结果无效。

        """
        match = _FIND_DOUBLE_BRACKETS.search(text)

        if match:
            verdict = match.group(1)

        if not match or verdict not in {"A", "B", "C"}:
            msg = (
                f"Invalid output: {text}. "
                "Output must contain a double bracketed string\
                 with the verdict 'A', 'B', or 'C'."
            )
            raise ValueError(msg)
                # C 表示模型绑定。返回 'None' 表示无偏好
        verdict_ = None if verdict == "C" else verdict
        score = {
            "A": 1,
            "B": 0,
            "C": 0.5,
        }[verdict]
        return {
            "reasoning": text,
            "value": verdict_,
            "score": score,
        }


class PairwiseStringEvalChain(PairwiseStringEvaluator, LLMEvalChain, LLMChain):
    r"""成对字符串评估链。

    A chain for comparing two outputs, such as the outputs
     of two models, prompts, or outputs of a single model on similar inputs.

    Attributes:
        output_parser (BaseOutputParser): 链的输出解析器。

    Example:
        >>> from langchain_openai import ChatOpenAI
        >>> from langchain_classic.evaluation.comparison import PairwiseStringEvalChain
        >>> model = ChatOpenAI(
        ...     temperature=0, model_name="gpt-4", model_kwargs={"random_seed": 42}
        ... )
        >>> chain = PairwiseStringEvalChain.from_llm(llm=model)
        >>> result = chain.evaluate_string_pairs(
        ...     input = "What is the chemical formula for water?",
        ...     prediction = "H2O",
        ...     prediction_b = (
        ...        "The chemical formula for water is H2O, which means"
        ...        " there are two hydrogen atoms and one oxygen atom."
        ...     reference = "The chemical formula for water is H2O.",
        ... )
        >>> print(result)
        # {
        #    "value": "B",
        #    "comment": "Both responses accurately state"
        #       " that the chemical formula for water is H2O."
        #       " However, Response B provides additional information"
        # .     " by explaining what the formula means.\n[[B]]"
        # }

    """

    output_key: str = "results"
    output_parser: BaseOutputParser = Field(
        default_factory=PairwiseStringResultOutputParser,
    )

    @classmethod
    @override
    def is_lc_serializable(cls) -> bool:
        return False

    model_config = ConfigDict(
        extra="ignore",
    )

    @property
    def requires_reference(self) -> bool:
        """返回链是否需要参考答案。

        Returns:
            `True` 表示链需要参考答案，否则为 `False`。

        """
        return False

    @property
    def requires_input(self) -> bool:
        """返回链是否需要输入。

        Returns:
            `True` 表示链需要输入，否则为 `False`。

        """
        return True

    @property
    def _skip_reference_warning(self) -> str:
        """返回忽略参考答案时显示的警告。

        Returns:
            忽略参考答案时显示的警告。

        """
        return (
            f"Ignoring reference in {self.__class__.__name__}, as it is not expected."
            "\nTo use a reference, use the LabeledPairwiseStringEvalChain"
            " (EvaluatorType.LABELED_PAIRWISE_STRING) instead."
        )

    @classmethod
    def from_llm(
        cls,
        llm: BaseLanguageModel,
        *,
        prompt: PromptTemplate | None = None,
        criteria: CRITERIA_TYPE | str | None = None,
        **kwargs: Any,
    ) -> PairwiseStringEvalChain:
        """从 LLM 初始化 PairwiseStringEvalChain。

        Args:
            llm: 使用的 LLM（推荐使用 GPT-4）。
            prompt: 使用的提示词。
            criteria: 使用的标准。
            **kwargs: 其他关键字参数。

        Returns:
            初始化后的 PairwiseStringEvalChain。

        Raises:
            ValueError: 如果输入变量不符合预期。

        """
                # 检查模型是否为 GPT-4，如果不是则发出警告
        if not hasattr(llm, "model_name") or not llm.model_name.startswith("gpt-4"):
            logger.warning(
                "This chain was only tested with GPT-4. \
Performance may be significantly worse with other models.",
            )

        expected_input_vars = {"prediction", "prediction_b", "input", "criteria"}
        prompt_ = prompt or COMPARISON_TEMPLATE.partial(reference="")
        if expected_input_vars != set(prompt_.input_variables):
            msg = (
                f"Input variables should be {expected_input_vars}, "
                f"but got {prompt_.input_variables}"
            )
            raise ValueError(msg)
        criteria_ = resolve_pairwise_criteria(criteria)
        criteria_str = "\n".join(f"{k}: {v}" if v else k for k, v in criteria_.items())
        criteria_str = CRITERIA_INSTRUCTIONS + criteria_str if criteria_str else ""
        return cls(llm=llm, prompt=prompt_.partial(criteria=criteria_str), **kwargs)

    def _prepare_input(
        self,
        prediction: str,
        prediction_b: str,
        input_: str | None,
        reference: str | None,
    ) -> dict:
        """准备链的输入。

        Args:
            prediction: 第一个模型的输出字符串。
            prediction_b: 第二个模型的输出字符串。
            input_: 输入或任务字符串。
            reference: 参考字符串，如果有的话。

        Returns:
            链的准备好的输入。

        """
        input_dict = {
            "prediction": prediction,
            "prediction_b": prediction_b,
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
        return parsed

    @override
    def _evaluate_string_pairs(
        self,
        *,
        prediction: str,
        prediction_b: str,
        input: str | None = None,
        reference: str | None = None,
        callbacks: Callbacks = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        include_run_info: bool = False,
        **kwargs: Any,
    ) -> dict:
        """评估输出 A 是否优于输出 B。

        Args:
            prediction: 第一个模型的输出字符串。
            prediction_b: 第二个模型的输出字符串。
            input: 输入或任务字符串。
            callbacks: 使用的回调。
            tags: 应用的标签。
            metadata: 使用的元数据。
            include_run_info: 是否在输出中包含运行信息。
            reference: 参考字符串，如果有的话。
            **kwargs: 其他关键字参数。

        Returns:
            `dict` containing:
                - reasoning: The reasoning for the preference.
                - value: The preference value, which is either 'A', 'B', or None
                    for no preference.
                - score: The preference score, which is 1 for 'A', 0 for 'B',
                    and 0.5 for None.

        """
        input_ = self._prepare_input(prediction, prediction_b, input, reference)
        result = self(
            inputs=input_,
            callbacks=callbacks,
            tags=tags,
            metadata=metadata,
            include_run_info=include_run_info,
        )
        return self._prepare_output(result)

    @override
    async def _aevaluate_string_pairs(
        self,
        *,
        prediction: str,
        prediction_b: str,
        reference: str | None = None,
        input: str | None = None,
        callbacks: Callbacks = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        include_run_info: bool = False,
        **kwargs: Any,
    ) -> dict:
        """异步评估输出 A 是否优于输出 B。

        Args:
            prediction: 第一个模型的输出字符串。
            prediction_b: 第二个模型的输出字符串。
            input: 输入或任务字符串。
            callbacks: 要使用的回调。
            tags: 要应用的标签。
            metadata: 要使用的元数据。
            include_run_info: 是否在输出中包含运行信息。
            reference: 参考字符串，如果存在的话。
            **kwargs: 额外的关键字参数。

        Returns:
            `dict` containing:
                - reasoning: The reasoning for the preference.
                - value: The preference value, which is either 'A', 'B', or None
                    for no preference.
                - score: The preference score, which is 1 for 'A', 0 for 'B',
                    and 0.5 for None.

        """
        input_ = self._prepare_input(prediction, prediction_b, input, reference)
        result = await self.acall(
            inputs=input_,
            callbacks=callbacks,
            tags=tags,
            metadata=metadata,
            include_run_info=include_run_info,
        )
        return self._prepare_output(result)


class LabeledPairwiseStringEvalChain(PairwiseStringEvalChain):
    """带标签的成对字符串评估链。

    A chain for comparing two outputs, such as the outputs
    of two models, prompts, or outputs of a single model on similar inputs,
    with labeled preferences.

    Attributes:
        output_parser (BaseOutputParser): 链的输出解析器。

    """

    @property
    def requires_reference(self) -> bool:
        """返回链是否需要参考。

        Returns:
            如果链需要参考则返回 `True`，否则返回 `False`。

        """
        return True

    @classmethod
    def from_llm(
        cls,
        llm: BaseLanguageModel,
        *,
        prompt: PromptTemplate | None = None,
        criteria: CRITERIA_TYPE | str | None = None,
        **kwargs: Any,
    ) -> PairwiseStringEvalChain:
        """从 LLM 初始化 LabeledPairwiseStringEvalChain。

        Args:
            llm: 要使用的 LLM。
            prompt: 要使用的提示词。
            criteria: 要使用的标准。
            **kwargs: 额外的关键字参数。

        Returns:
            已初始化的 `LabeledPairwiseStringEvalChain`。

        Raises:
            ValueError: 如果输入变量不符合预期。

        """
        expected_input_vars = {
            "prediction",
            "prediction_b",
            "input",
            "reference",
            "criteria",
        }
        prompt_ = prompt or COMPARISON_TEMPLATE_WITH_REFERENCE
        if expected_input_vars != set(prompt_.input_variables):
            msg = (
                f"Input variables should be {expected_input_vars}, "
                f"but got {prompt_.input_variables}"
            )
            raise ValueError(msg)
        criteria_ = resolve_pairwise_criteria(criteria)
        criteria_str = "\n".join(f"{k}: {v}" for k, v in criteria_.items())
        criteria_str = CRITERIA_INSTRUCTIONS + criteria_str if criteria_str else ""
        return cls(llm=llm, prompt=prompt_.partial(criteria=criteria_str), **kwargs)

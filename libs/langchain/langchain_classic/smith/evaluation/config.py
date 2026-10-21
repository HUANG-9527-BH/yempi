"""运行评估器的配置。"""

from collections.abc import Callable, Sequence
from typing import Any

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import BasePromptTemplate
from langsmith import RunEvaluator
from langsmith.evaluation.evaluator import EvaluationResult, EvaluationResults
from langsmith.schemas import Example, Run
from pydantic import BaseModel, ConfigDict, Field
from typing_extensions import override

from langchain_classic.evaluation.criteria.eval_chain import CRITERIA_TYPE
from langchain_classic.evaluation.embedding_distance.base import (
    EmbeddingDistance as EmbeddingDistanceEnum,
)
from langchain_classic.evaluation.schema import EvaluatorType, StringEvaluator
from langchain_classic.evaluation.string_distance.base import (
    StringDistance as StringDistanceEnum,
)

RUN_EVALUATOR_LIKE = Callable[
    [Run, Example | None],
    EvaluationResult | EvaluationResults | dict,
]
BATCH_EVALUATOR_LIKE = Callable[
    [Sequence[Run], Sequence[Example] | None],
    EvaluationResult | EvaluationResults | dict,
]


class EvalConfig(BaseModel):
    """特定运行评估器的配置。

    Attributes:
        evaluator_type: 要使用的评估器类型。
    """

    evaluator_type: EvaluatorType

    def get_kwargs(self) -> dict[str, Any]:
        """获取 `load_evaluator` 调用的关键词参数。

        Returns:
            `load_evaluator` 调用的关键词参数。
        """
        kwargs = {}
        for field, val in self:
            if field == "evaluator_type" or val is None:
                continue
            kwargs[field] = val
        return kwargs


class SingleKeyEvalConfig(EvalConfig):
    """仅需单个键的运行评估器的配置。"""

    reference_key: str | None = None
    """在数据集运行中用作参考字符串的键。
    若未提供，我们将尝试自动推断。"""
    prediction_key: str | None = None
    """The key from the traced run's outputs dictionary to use to
    represent the prediction. If not provided, it will be inferred
    automatically."""
    input_key: str | None = None
    """在追踪运行的输入字典中用以表示
    输入的键。若未提供，将自动推断。"""

    @override
    def get_kwargs(self) -> dict[str, Any]:
        kwargs = super().get_kwargs()
                # Filer out the keys that are not needed for the evaluator.
        for key in ["reference_key", "prediction_key", "input_key"]:
            kwargs.pop(key, None)
        return kwargs


CUSTOM_EVALUATOR_TYPE = RUN_EVALUATOR_LIKE | RunEvaluator | StringEvaluator
SINGLE_EVAL_CONFIG_TYPE = EvaluatorType | str | EvalConfig


class RunEvalConfig(BaseModel):
    """运行评估的配置。"""

    evaluators: list[SINGLE_EVAL_CONFIG_TYPE | CUSTOM_EVALUATOR_TYPE] = Field(
        default_factory=list
    )
    """Configurations for which evaluators to apply to the dataset run.
    Each can be the string of an
    `EvaluatorType <langchain.evaluation.schema.EvaluatorType>`, such
    as `EvaluatorType.QA`, the evaluator type string ("qa"), or a configuration for a
    given evaluator
    (e.g.,
    `RunEvalConfig.QA <langchain.smith.evaluation.config.RunEvalConfig.QA>`)."""
    custom_evaluators: list[CUSTOM_EVALUATOR_TYPE] | None = None
    """要应用于数据集运行的自定义评估器。"""
    batch_evaluators: list[BATCH_EVALUATOR_LIKE] | None = None
    """在聚合/批处理级别运行的评估器。

    它们生成一个或多个指标并分配给整个测试运行。
    因此，它们不与单个追踪相关联。
    """

    reference_key: str | None = None
    """在数据集运行中用作参考字符串的键。
    若未提供，我们将尝试自动推断。"""
    prediction_key: str | None = None
    """The key from the traced run's outputs dictionary to use to
    represent the prediction. If not provided, it will be inferred
    automatically."""
    input_key: str | None = None
    """在追踪运行的输入字典中用以表示
    输入的键。若未提供，将自动推断。"""
    eval_llm: BaseLanguageModel | None = None
    """传递给需要语言模型的任何评估器的语言模型。"""

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )

    class Criteria(SingleKeyEvalConfig):
        """无参考标准的准则评估器的配置。

        Attributes:
            criteria: 要评估的准则。
            llm: 用于评估链的语言模型。
        """

        criteria: CRITERIA_TYPE | None = None
        llm: BaseLanguageModel | None = None
        evaluator_type: EvaluatorType = EvaluatorType.CRITERIA

    class LabeledCriteria(SingleKeyEvalConfig):
        """带有参考标准（标注）的准则评估器的配置。

        Attributes:
            criteria: 要评估的准则。
            llm: 用于评估链的语言模型。
        """

        criteria: CRITERIA_TYPE | None = None
        llm: BaseLanguageModel | None = None
        evaluator_type: EvaluatorType = EvaluatorType.LABELED_CRITERIA

    class EmbeddingDistance(SingleKeyEvalConfig):
        """嵌入距离评估器的配置。

        Attributes:
            embeddings: 用于计算距离的嵌入。
            distance_metric: 用于计算距离的距离度量。
        """

        evaluator_type: EvaluatorType = EvaluatorType.EMBEDDING_DISTANCE
        embeddings: Embeddings | None = None
        distance_metric: EmbeddingDistanceEnum | None = None

        model_config = ConfigDict(
            arbitrary_types_allowed=True,
        )

    class StringDistance(SingleKeyEvalConfig):
        """字符串距离评估器的配置。

        Attributes:
            distance: 要使用的字符串距离度量（`damerau_levenshtein`，
                `levenshtein`，`jaro`，或 `jaro_winkler`）。
            normalize_score: 是否将距离归一化到 0 到 1 之间。
                仅对 Levenshtein 和 Damerau-Levenshtein 距离有效。
        """

        evaluator_type: EvaluatorType = EvaluatorType.STRING_DISTANCE
        distance: StringDistanceEnum | None = None
        normalize_score: bool = True

    class QA(SingleKeyEvalConfig):
        """问答评估器的配置。

        Attributes:
            prompt: 用于生成问题的提示词模板。
            llm: 用于评估链的语言模型。
        """

        evaluator_type: EvaluatorType = EvaluatorType.QA
        llm: BaseLanguageModel | None = None
        prompt: BasePromptTemplate | None = None

    class ContextQA(SingleKeyEvalConfig):
        """基于上下文的问答评估器的配置。

        Attributes:
            prompt: 用于生成问题的提示词模板。
            llm: 用于评估链的语言模型。
        """

        evaluator_type: EvaluatorType = EvaluatorType.CONTEXT_QA
        llm: BaseLanguageModel | None = None
        prompt: BasePromptTemplate | None = None

    class CoTQA(SingleKeyEvalConfig):
        """基于上下文的问答评估器配置。

        Attributes:
            prompt: 用于生成问题的提示词模板。
            llm: 用于评估链的语言模型。
        """

        evaluator_type: EvaluatorType = EvaluatorType.CONTEXT_QA
        llm: BaseLanguageModel | None = None
        prompt: BasePromptTemplate | None = None

    class JsonValidity(SingleKeyEvalConfig):
        """JSON 有效性评估器配置。"""

        evaluator_type: EvaluatorType = EvaluatorType.JSON_VALIDITY

    class JsonEqualityEvaluator(EvalConfig):
        """JSON 相等性评估器配置。"""

        evaluator_type: EvaluatorType = EvaluatorType.JSON_EQUALITY

    class ExactMatch(SingleKeyEvalConfig):
        """精确匹配字符串评估器配置。

        Attributes:
            ignore_case: 比较字符串时是否忽略大小写。
            ignore_punctuation: 比较字符串时是否忽略标点符号。
            ignore_numbers: 比较字符串时是否忽略数字。
        """

        evaluator_type: EvaluatorType = EvaluatorType.EXACT_MATCH
        ignore_case: bool = False
        ignore_punctuation: bool = False
        ignore_numbers: bool = False

    class RegexMatch(SingleKeyEvalConfig):
        """正则表达式匹配字符串评估器配置。

        Attributes:
            flags: 传递给正则表达式的标志。例如：`re.IGNORECASE`。
        """

        evaluator_type: EvaluatorType = EvaluatorType.REGEX_MATCH
        flags: int = 0

    class ScoreString(SingleKeyEvalConfig):
        """分数字符串评估器配置。

        This is like the criteria evaluator but it is configured by
        default to return a score on the scale from 1-10.

        建议通过将 `normalize_by` 设置为 10 来对这些分数进行归一化。

        Attributes:
            criteria: 要评估的标准。
            llm: 用于评估链的语言模型。
            normalize_by: 如果需要对分数进行归一化，使用的分母。
                若未提供，则分数范围为 1 到 10。
            prompt: 用于评估的提示词模板。
        """

        evaluator_type: EvaluatorType = EvaluatorType.SCORE_STRING
        criteria: CRITERIA_TYPE | None = None
        llm: BaseLanguageModel | None = None
        normalize_by: float | None = None
        prompt: BasePromptTemplate | None = None

    class LabeledScoreString(ScoreString):
        """带标签的分数字符串评估器配置。"""

        evaluator_type: EvaluatorType = EvaluatorType.LABELED_SCORE_STRING

"""TypeSafe 集成的问答类型。"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Annotated, Literal, TypeAlias

from langchain_core.messages import BaseMessage
from pydantic import BaseModel, ConfigDict, Field, JsonValue
from typing_extensions import TypedDict

_QuestionContent: TypeAlias = str | dict[str, JsonValue] | list[JsonValue]
_StateValue: TypeAlias = (
    str
    | int
    | float
    | bool
    | BaseMessage
    | Sequence["_StateValue"]
    | dict[str, "_StateValue"]
    | None
)

State: TypeAlias = str | BaseMessage | Sequence[_StateValue] | dict[str, _StateValue]
"""Root state accepted by `TypeSafeClassifier`.

TypeSafe natively accepts a string, JSON object, or JSON array. LangChain
`BaseMessage` objects and message sequences can appear at the root or at any depth
inside objects and arrays. The integration serializes messages as role/content JSON
while preserving surrounding JSON structure.
"""


class NoulCriteria(BaseModel):
    """对 `Noul` 问题两种结果的可选描述。

    Criteria clarify what should count as yes and no when the instruction alone leaves
    room for interpretation. Both values accept any JSON-compatible content, so callers
    can provide a short description or structured examples.
    """

    model_config = ConfigDict(populate_by_name=True)

    true: JsonValue = None
    """是结果的描述，或当无需澄清时为 `None`。"""

    false: JsonValue = None
    """否结果的描述，或当无需澄清时为 `None`。"""


class Noul(BaseModel):
    """提出一个二元问题并获取其答案为是的概率。

    Use `Noul` when the probability itself is useful to application code, such as
    deciding whether a message reports a bug or requests a refund. A value near `1`
    indicates strong support for yes, a value near `0` indicates strong support for no,
    and a value near `0.5` indicates uncertainty. Noul answers do not include a separate
    confidence value.

    ??? example "检测紧急支持请求"

        ```python
        from langchain_typesafe import Noul, TypeSafeClassifier

        classifier = TypeSafeClassifier()
        response = classifier.invoke(
            {
                "state": "Production is down. Please help immediately.",
                "questions": {
                    "urgent": Noul(
                        instructions="Does this message require an urgent response?"
                    )
                },
            }
        )
        urgency = response.nouls["urgent"].noul

        if urgency >= 0.8:
            page_on_call_engineer()
        ```
    """

    type: Literal["noul"] = "noul"
    """二元 TypeSafe 问题的判别器。"""

    instructions: _QuestionContent
    """关于输入状态的完整是/否判断。

    指令可以是文本或结构化 JSON。即使 `ClassifierRequest.questions` 中的问题 ID
    看似自明，也请在此处写出完整问题。
    """

    criteria: NoulCriteria | None = None
    """用于定义是和否结果含义的可选描述。"""


class Choice(BaseModel):
    """从一组固定选项中选择一个标签。

    Use `Choice` for categorical decisions with no inherent ordering, such as routing a
    support request, detecting a document type, or selecting an intent. The answer
    includes the selected label, a probability for every supplied label, and confidence
    derived from the shape of that probability distribution.

    ??? example "路由支持请求"

        ```python
        from langchain_typesafe import Choice, TypeSafeClassifier

        classifier = TypeSafeClassifier()
        response = classifier.invoke(
            {
                "state": "Stripe fails whenever I connect my account.",
                "questions": {
                    "department": Choice(
                        instructions="Which team should handle this request?",
                        criteria={
                            "billing": "Payment, invoice, or subscription issues.",
                            "technical": "Product bugs or integration failures.",
                            "sales": "Pricing or purchasing questions.",
                        },
                    )
                },
            }
        )
        department = response.choices["department"]

        if department.confidence >= 0.7:
            route_to(department.choice)
        else:
            route_to_human_triage()
        ```
    """

    type: Literal["choice"] = "choice"
    """类别型 TypeSafe 问题的判别器。"""

    criteria: dict[str, JsonValue] = Field(min_length=1)
    """候选标签及其描述的映射。

    描述可以是文本、结构化 JSON 或 `None`。当提供的选项可能无法涵盖所有输入时，请包含 `other` 或
    `none_of_the_above` 标签。
    """

    instructions: _QuestionContent
    """关于输入状态的完整类别判断。"""


class Score(BaseModel):
    """根据有序评分标准评估状态。

    Use `Score` when the answer lies on a spectrum whose levels can be described, such
    as severity, urgency, or customer frustration. Criteria are numbered from zero in
    their supplied order. The returned score is an expected value and may fall between
    integer levels; the full probability distribution remains available for custom
    decision logic.

    ??? example "评分客户沮丧程度"

        ```python
        from langchain_typesafe import Score, TypeSafeClassifier

        classifier = TypeSafeClassifier()
        response = classifier.invoke(
            {
                "state": "This has failed three times. Fix it now.",
                "questions": {
                    "frustration": Score(
                        instructions="How frustrated does the customer appear?",
                        criteria=[
                            "Calm and neutral.",
                            "Concerned but civil.",
                            "Very angry or using strong language.",
                        ],
                    )
                },
            }
        )
        frustration = response.scores["frustration"]

        print(frustration.score)  # May be fractional, for example 1.35.
        print(frustration.legend)  # The original zero-based rubric.
        print(frustration.probabilities)  # Probability for each rubric level.
        ```
    """

    type: Literal["score"] = "score"
    """序数型 TypeSafe 问题的判别器。"""

    criteria: list[JsonValue] = Field(min_length=2)
    """从零开始的两个或更多有序评分层级描述。"""

    instructions: _QuestionContent
    """关于输入状态的完整序数判断。"""


Question = Annotated[Noul | Choice | Score, Field(discriminator="type")]
"""A discriminated union of question types accepted by `TypeSafeClassifier`."""


class ClassifierRequest(TypedDict):
    """一次 `TypeSafeClassifier` 调用的完整输入。

    将状态和问题保留在 Runnable 输入中，可确保两者参与组合、批处理和追踪。
    """

    state: State
    """待分类的文本、结构化 JSON 或 LangChain 消息。"""

    questions: dict[str, Question]
    """非空的映射，将答案 ID 映射到类型化的分类问题。"""


class NoulAnswer(BaseModel):
    """Noul 问题的答案为“是”的概率。"""

    type: Literal["noul"]
    """标识二元答案的线缆判别器。"""

    noul: float = Field(ge=0.0, le=1.0)
    """在从 `0` 到 `1` 的包含范围内，答案为“是”的概率。"""


class ChoiceAnswer(BaseModel):
    """选定的 `Choice` 标签及其概率分布和置信度。"""

    type: Literal["choice"]
    """标识类别答案的线缆判别器。"""

    choice: str
    """从 `Choice.criteria` 中提供的选项中选择的标签。"""

    probabilities: dict[str, float]
    """分配给每个候选标签的概率，以标签名称为键。

    The complete distribution is retained so applications can use a confidence measure
    or risk policy different from TypeSafe's default confidence calculation.
    """

    confidence: float = Field(ge=0.0, le=1.0)
    """从 `0` 到 `1` 的标量确定性，由概率分布推导得出。

    置信度描述了分布的集中程度；它不是所选标签的概率。阈值应根据错误自动化决策的后果来选择。
    """


class ScoreAnswer(BaseModel):
    """预期的 `Score` 值及其评分标准、分布和置信度。"""

    type: Literal["score"]
    """标识序数答案的线缆判别器。"""

    score: float
    """在有序评分标准中的预期位置。

    该值可能是小数，因为它总结了整数评分等级上的概率分布，而不是精确选择一个等级。
    """

    legend: dict[int, JsonValue]
    """原始评分标准描述，以零基整数等级为键。"""

    probabilities: dict[int, float]
    """在零基整数评分等级上的概率分布。"""

    confidence: float = Field(ge=0.0, le=1.0)
    """从等级分布推导出的 `0` 到 `1` 之间的标量确定性。"""


Answer = Annotated[NoulAnswer | ChoiceAnswer | ScoreAnswer, Field(discriminator="type")]
"""A discriminated union of answers returned by `TypeSafeClassifier`."""


class Usage(BaseModel):
    """TypeSafe 分类请求报告的 token 使用量。"""

    input_tokens: int | None = None
    """处理的输入 token 数量，未报告时为 `None`。"""

    output_tokens: int | None = None
    """生成的输出 token 数量，未报告时为 `None`。"""


class ClassifierResponse(BaseModel):
    """从一次 TypeSafe 请求返回的类型化答案和元数据。

    通过 `answers` 访问所有答案，或使用 `nouls`、`choices` 和 `scores`
    按答案类型过滤视图。每个映射都保留了传递给 `ClassifierRequest.questions` 的问题 ID。
    """

    model: str
    """回答该请求的 TypeSafe 模型。"""

    answers: dict[str, Answer]
    """所有识别出的答案，以原始问题 ID 为键。"""

    usage: Usage = Field(default_factory=Usage)
    """请求的输入和输出 token 数量报告。"""

    request_id: str | None = None
    """TypeSafe 请求 ID，当与提供方支持诊断请求时很有用。"""

    @property
    def nouls(self) -> dict[str, NoulAnswer]:
        """按其原始问题 ID 返回二元答案。"""
        return {
            name: answer
            for name, answer in self.answers.items()
            if isinstance(answer, NoulAnswer)
        }

    @property
    def choices(self) -> dict[str, ChoiceAnswer]:
        """按其原始问题 ID 返回类别答案。"""
        return {
            name: answer
            for name, answer in self.answers.items()
            if isinstance(answer, ChoiceAnswer)
        }

    @property
    def scores(self) -> dict[str, ScoreAnswer]:
        """按其原始问题 ID 返回序数答案。"""
        return {
            name: answer
            for name, answer in self.answers.items()
            if isinstance(answer, ScoreAnswer)
        }


__all__ = [
    "Answer",
    "Choice",
    "ChoiceAnswer",
    "ClassifierRequest",
    "ClassifierResponse",
    "Noul",
    "NoulAnswer",
    "NoulCriteria",
    "Question",
    "Score",
    "ScoreAnswer",
    "State",
    "Usage",
]

"""OpenAI 决策 API 的问题与响应类型。"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Annotated, Any, Literal, TypeAlias

from langchain_core.messages import BaseMessage
from pydantic import BaseModel, ConfigDict, Field
from typing_extensions import TypedDict

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
"""Input accepted by `OpenAIDecisions`.

The Decisions API natively accepts a string or user messages containing text and
base64 images. Strings and `HumanMessage` objects (alone or in a sequence) are sent
natively. Any other state, such as conversations with system or AI messages, or JSON
objects, is sent as JSON text in a single user message, with base64 images kept in
place as image parts.
"""


class Predicate(BaseModel):
    """估算某个条件为真的概率。

    ??? example "检查请求是否紧急"

        ```python
        from langchain_openai.decisions import OpenAIDecisions, Predicate

        decisions = OpenAIDecisions(model="gpt-6-luna")
        response = decisions.invoke(
            {
                "input": "Production is down. Please help immediately.",
                "questions": {
                    "urgent": Predicate(instructions="Is this request urgent?"),
                },
            }
        )
        if response.predicates["urgent"].probability >= 0.8:
            page_on_call_engineer()
        ```
    """

    type: Literal["predicate"] = "predicate"
    """谓词问题的线缆判别器。"""

    instructions: str = Field(min_length=1)
    """要针对输入评估的条件。"""

    def _to_api(self, name: str) -> dict[str, Any]:
        return {"type": self.type, "name": name, "instructions": self.instructions}


class Choice(BaseModel):
    """从一组固定且无序的选项中选择一个值。

    ??? example "路由客户投诉"

        ```python
        from langchain_openai.decisions import Choice, OpenAIDecisions

        decisions = OpenAIDecisions(model="gpt-6-luna")
        response = decisions.invoke(
            {
                "input": "I was charged twice for my order.",
                "questions": {
                    "department": Choice(
                        instructions="Which department should handle this?",
                        choices={
                            "billing": "Payments, invoices, and refunds.",
                            "technical": "Problems using the product.",
                            "other": "Requests outside these categories.",
                        },
                    ),
                },
            }
        )
        department = response.choices["department"]
        if department.confidence >= 0.7:
            route_to(department.choice)
        ```
    """

    type: Literal["choice"] = "choice"
    """选择问题的线缆判别器。"""

    instructions: str = Field(min_length=1)
    """关于输入的判断。"""

    choices: dict[str | bool, str | None] = Field(min_length=1)
    """映射到每个选项适用情况的可选描述的候选值。

    当选项可能无法覆盖所有输入时，包含一个如 `"other"` 的备用选项。
    """

    def _to_api(self, name: str) -> dict[str, Any]:
        return {
            "type": self.type,
            "name": name,
            "instructions": self.instructions,
            "choices": [
                {"value": value}
                if description is None
                else {"value": value, "description": description}
                for value, description in self.choices.items()
            ],
        }


class Level(BaseModel):
    """评分量表的一个层级。"""

    label: str = Field(min_length=1)
    """该层级的简短名称。"""

    description: str | None = None
    """区分此层级与其相邻层级的标准。"""


class Score(BaseModel):
    """根据有序层级对输入进行评分。

    Levels are indexed from `0` in the order supplied. The returned score is the
    probability-weighted average of those indices, so it may fall between levels.

    ??? example "评分问题严重性"

        ```python
        from langchain_openai.decisions import Level, OpenAIDecisions, Score

        decisions = OpenAIDecisions(model="gpt-6-luna")
        response = decisions.invoke(
            {
                "input": "Export fails in Safari but works in Chrome.",
                "questions": {
                    "severity": Score(
                        instructions="How severe is this issue?",
                        levels=[
                            Level(label="Cosmetic", description="No lost function."),
                            Level(label="Workaround", description="Another way works."),
                            Level(label="Blocked", description="No workaround."),
                        ],
                    ),
                },
            }
        )
        print(response.scores["severity"].score)  # May be fractional, e.g. 1.1
        ```
    """

    type: Literal["score"] = "score"
    """评分问题的线缆判别器。"""

    instructions: str = Field(min_length=1)
    """关于输入的判断。"""

    levels: list[str | Level] = Field(min_length=2)
    """两个或更多从低到高排序的层级。

    纯字符串是仅含标签的层级的简写形式。
    """

    def _to_api(self, name: str) -> dict[str, Any]:
        levels = [
            Level(label=level) if isinstance(level, str) else level
            for level in self.levels
        ]
        return {
            "type": self.type,
            "name": name,
            "instructions": self.instructions,
            "levels": [level.model_dump(exclude_none=True) for level in levels],
        }


Question = Annotated[Predicate | Choice | Score, Field(discriminator="type")]
"""A discriminated union of question types accepted by `OpenAIDecisions`."""


class DecisionRequest(TypedDict):
    """一次 `OpenAIDecisions` 调用的完整输入。

    将输入和问题一起保留在 Runnable 输入中，可确保两者都参与组合、批处理和追踪。
    """

    input: State
    """要评估的文本、消息或结构化状态。"""

    questions: dict[str, Question]
    """问题名称到问题的映射。名称用于键控返回的答案。"""


class PredicateAnswer(BaseModel):
    """某个 `Predicate` 条件为真的概率。"""

    type: Literal["predicate"]
    """标识谓词答案的线缆判别器。"""

    probability: float = Field(ge=0.0, le=1.0)
    """条件为真的估计概率，范围从 `0` 到 `1`。"""


class ChoiceAnswer(BaseModel):
    """从 `Choice.choices` 中选择的 `Choice` 值及其概率分布和置信度。"""

    type: Literal["choice"]
    """标识选择答案的线缆判别器。"""

    choice: str | bool
    """从 `Choice.choices` 中选择的值。"""

    probabilities: dict[str | bool, float]
    """分配给每个候选值的概率。"""

    confidence: float = Field(ge=0.0, le=1.0)
    """从 `0` 到 `1` 的标量置信度，由分布推导得出。

    置信度描述分布的集中程度；它不是所选值的概率。
    """


class ScoreAnswer(BaseModel):
    """带有层级、分布和置信度的预期 `Score` 值。"""

    type: Literal["score"]
    """标识评分答案的线缆判别器。"""

    score: float
    """零基层级索引的概率加权平均值。"""

    legend: dict[int, str]
    """按零基索引键控的层级标签。"""

    probabilities: dict[int, float]
    """分配给每个零基层级索引的概率。"""

    confidence: float = Field(ge=0.0, le=1.0)
    """从 `0` 到 `1` 的标量置信度，由分布推导得出。"""


class RefusalAnswer(BaseModel):
    """模型拒绝回答一个问题。"""

    model_config = ConfigDict(extra="allow")

    type: Literal["refusal"]
    """标识拒绝的线缆判别器。"""


Answer = Annotated[
    PredicateAnswer | ChoiceAnswer | ScoreAnswer | RefusalAnswer,
    Field(discriminator="type"),
]
"""A discriminated union of answers returned by `OpenAIDecisions`."""


class Usage(BaseModel):
    """决策请求报告的 token 使用量。"""

    input_tokens: int | None = None
    """处理的输入 token 数量，未报告时为 `None`。"""

    output_tokens: int | None = None
    """生成的输出 token 数量，未报告时为 `None`。"""


class DecisionResponse(BaseModel):
    """一次决策请求返回的已类型化的答案和元数据。

    Access every answer through `answers`, or use `predicates`, `choices`, `scores`,
    and `refusals` for views filtered by answer type. Each mapping is keyed by the
    question names supplied in `DecisionRequest.questions`.
    """

    model: str
    """回答请求的模型。"""

    answers: dict[str, Answer]
    """所有识别出的答案，按问题名称键控。"""

    usage: Usage = Field(default_factory=Usage)
    """请求报告的 token 统计。"""

    request_id: str | None = None
    """OpenAI 请求 ID，当与 OpenAI 支持团队诊断请求时非常有用。"""

    @property
    def predicates(self) -> dict[str, PredicateAnswer]:
        """按问题名称键返回谓词答案。"""
        return {
            name: answer
            for name, answer in self.answers.items()
            if isinstance(answer, PredicateAnswer)
        }

    @property
    def choices(self) -> dict[str, ChoiceAnswer]:
        """按问题名称键返回选项答案。"""
        return {
            name: answer
            for name, answer in self.answers.items()
            if isinstance(answer, ChoiceAnswer)
        }

    @property
    def scores(self) -> dict[str, ScoreAnswer]:
        """按问题名称键返回评分答案。"""
        return {
            name: answer
            for name, answer in self.answers.items()
            if isinstance(answer, ScoreAnswer)
        }

    @property
    def refusals(self) -> dict[str, RefusalAnswer]:
        """按问题名称键返回拒绝答案。"""
        return {
            name: answer
            for name, answer in self.answers.items()
            if isinstance(answer, RefusalAnswer)
        }


__all__ = [
    "Answer",
    "Choice",
    "ChoiceAnswer",
    "DecisionRequest",
    "DecisionResponse",
    "Level",
    "Predicate",
    "PredicateAnswer",
    "Question",
    "RefusalAnswer",
    "Score",
    "ScoreAnswer",
    "State",
    "Usage",
]

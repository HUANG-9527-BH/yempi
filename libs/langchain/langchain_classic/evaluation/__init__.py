"""用于评分大语言模型和链输出的评估链。

This module contains off-the-shelf evaluation chains for grading the output of
LangChain primitives such as language models and chains.

加载评估器

To load an evaluator, you can use the `load_evaluators <langchain.evaluation.loading.load_evaluators>` or
`load_evaluator <langchain.evaluation.loading.load_evaluator>` functions with the
names of the evaluators to load.

```python
from langchain_classic.evaluation import load_evaluator

evaluator = load_evaluator("qa")
evaluator.evaluate_strings(
    prediction="We sold more than 40,000 units last week",
    input="How many units did we sell last week?",
    reference="We sold 32,378 units",
)
```

评估器必须是 `EvaluatorType <langchain.evaluation.schema.EvaluatorType>` 中的一个。

数据集

To load one of the LangChain HuggingFace datasets, you can use the `load_dataset <langchain.evaluation.loading.load_dataset>` function with the
name of the dataset to load.

```python
from langchain_classic.evaluation import load_dataset

ds = load_dataset("llm-math")
```

评估的一些常见用例包括：

- 根据真实答案对响应的准确性进行评分：`QAEvalChain <langchain.evaluation.qa.eval_chain.QAEvalChain>`
- 比较两个模型的输出：`PairwiseStringEvalChain <langchain.evaluation.comparison.eval_chain.PairwiseStringEvalChain>` 或当存在参考标签时使用 `LabeledPairwiseStringEvalChain <langchain.evaluation.comparison.eval_chain.LabeledPairwiseStringEvalChain>`。
- 判断智能体工具使用的有效性：`TrajectoryEvalChain <langchain.evaluation.agents.trajectory_eval_chain.TrajectoryEvalChain>`
- 检查输出是否符合一组标准：`CriteriaEvalChain <langchain.evaluation.criteria.eval_chain.CriteriaEvalChain>` 或当存在参考标签时使用 `LabeledCriteriaEvalChain <langchain.evaluation.criteria.eval_chain.LabeledCriteriaEvalChain>`。
- 计算预测与参考之间的语义差异：`EmbeddingDistanceEvalChain <langchain.evaluation.embedding_distance.base.EmbeddingDistanceEvalChain>` 或两个预测之间的差异：`PairwiseEmbeddingDistanceEvalChain <langchain.evaluation.embedding_distance.base.PairwiseEmbeddingDistanceEvalChain>`
- 测量预测与参考之间的字符串距离：`StringDistanceEvalChain <langchain.evaluation.string_distance.base.StringDistanceEvalChain>` 或两个预测之间的距离：`PairwiseStringDistanceEvalChain <langchain.evaluation.string_distance.base.PairwiseStringDistanceEvalChain>`

低级 API

这些评估器实现以下接口之一：

- `StringEvaluator <langchain.evaluation.schema.StringEvaluator>`：根据参考标签和/或输入上下文对预测字符串进行评分。
- `PairwiseStringEvaluator <langchain.evaluation.schema.PairwiseStringEvaluator>`：对两个预测字符串相互进行评分。适用于打分偏好、测量两个链或大语言模型智能体之间的相似性，或在相似输入上比较输出。
- `AgentTrajectoryEvaluator <langchain.evaluation.schema.AgentTrajectoryEvaluator>`：评估智能体采取的完整动作序列。

这些接口使得在更高级别的评估框架中更容易组合和使用。

"""  # noqa: E501

from langchain_classic.evaluation.agents import TrajectoryEvalChain
from langchain_classic.evaluation.comparison import (
    LabeledPairwiseStringEvalChain,
    PairwiseStringEvalChain,
)
from langchain_classic.evaluation.criteria import (
    Criteria,
    CriteriaEvalChain,
    LabeledCriteriaEvalChain,
)
from langchain_classic.evaluation.embedding_distance import (
    EmbeddingDistance,
    EmbeddingDistanceEvalChain,
    PairwiseEmbeddingDistanceEvalChain,
)
from langchain_classic.evaluation.exact_match.base import ExactMatchStringEvaluator
from langchain_classic.evaluation.loading import (
    load_dataset,
    load_evaluator,
    load_evaluators,
)
from langchain_classic.evaluation.parsing.base import (
    JsonEqualityEvaluator,
    JsonValidityEvaluator,
)
from langchain_classic.evaluation.parsing.json_distance import JsonEditDistanceEvaluator
from langchain_classic.evaluation.parsing.json_schema import JsonSchemaEvaluator
from langchain_classic.evaluation.qa import (
    ContextQAEvalChain,
    CotQAEvalChain,
    QAEvalChain,
)
from langchain_classic.evaluation.regex_match.base import RegexMatchStringEvaluator
from langchain_classic.evaluation.schema import (
    AgentTrajectoryEvaluator,
    EvaluatorType,
    PairwiseStringEvaluator,
    StringEvaluator,
)
from langchain_classic.evaluation.scoring import (
    LabeledScoreStringEvalChain,
    ScoreStringEvalChain,
)
from langchain_classic.evaluation.string_distance import (
    PairwiseStringDistanceEvalChain,
    StringDistance,
    StringDistanceEvalChain,
)

__all__ = [
    "AgentTrajectoryEvaluator",
    "ContextQAEvalChain",
    "CotQAEvalChain",
    "Criteria",
    "CriteriaEvalChain",
    "EmbeddingDistance",
    "EmbeddingDistanceEvalChain",
    "EvaluatorType",
    "ExactMatchStringEvaluator",
    "JsonEditDistanceEvaluator",
    "JsonEqualityEvaluator",
    "JsonSchemaEvaluator",
    "JsonValidityEvaluator",
    "LabeledCriteriaEvalChain",
    "LabeledPairwiseStringEvalChain",
    "LabeledScoreStringEvalChain",
    "PairwiseEmbeddingDistanceEvalChain",
    "PairwiseStringDistanceEvalChain",
    "PairwiseStringEvalChain",
    "PairwiseStringEvaluator",
    "QAEvalChain",
    "RegexMatchStringEvaluator",
    "ScoreStringEvalChain",
    "StringDistance",
    "StringDistanceEvalChain",
    "StringEvaluator",
    "TrajectoryEvalChain",
    "load_dataset",
    "load_evaluator",
    "load_evaluators",
]

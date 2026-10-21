"""**LangSmith** 工具。

此模块提供连接到
[LangSmith](https://docs.langchain.com/langsmith/home) 的工具。

**评估**

LangSmith 帮助您使用多个 LangChain 评估器对链和其他语言模型应用组件进行评估。
下面是一个示例，假设您已创建了一个名为 `<my_dataset_name>` 的 LangSmith 数据集：

```python
from langsmith import Client
from langchain_openai import ChatOpenAI
from langchain_classic.chains import LLMChain
from langchain_classic.smith import RunEvalConfig, run_on_dataset


# Chains may have memory. Passing in a constructor function lets the
# evaluation framework avoid cross-contamination between runs.
def construct_chain():
    model = ChatOpenAI(temperature=0)
    chain = LLMChain.from_string(model, "What's the answer to {your_input_key}")
    return chain


# Load off-the-shelf evaluators via config or the EvaluatorType (string or enum)
evaluation_config = RunEvalConfig(
    evaluators=[
        "qa",  # "Correctness" against a reference answer
        "embedding_distance",
        RunEvalConfig.Criteria("helpfulness"),
        RunEvalConfig.Criteria(
            {
                "fifth-grader-score": "Do you have to be smarter than a fifth "
                "grader to answer this question?"
            }
        ),
    ]
)

client = Client()
run_on_dataset(
    client,
    "<my_dataset_name>",
    construct_chain,
    evaluation=evaluation_config,
)
```

您也可以通过继承
`StringEvaluator <langchain.evaluation.schema.StringEvaluator>`
或 LangSmith 的 `RunEvaluator` 类来创建自定义评估器。

```python
from typing import Optional
from langchain_classic.evaluation import StringEvaluator


class MyStringEvaluator(StringEvaluator):
    @property
    def requires_input(self) -> bool:
        return False

    @property
    def requires_reference(self) -> bool:
        return True

    @property
    def evaluation_name(self) -> str:
        return "exact_match"

    def _evaluate_strings(
        self, prediction, reference=None, input=None, **kwargs
    ) -> dict:
        return {"score": prediction == reference}


evaluation_config = RunEvalConfig(
    custom_evaluators=[MyStringEvaluator()],
)

run_on_dataset(
    client,
    "<my_dataset_name>",
    construct_chain,
    evaluation=evaluation_config,
)
```

**主要函数**

- `arun_on_dataset <langchain.smith.evaluation.runner_utils.arun_on_dataset>`:
    Asynchronous function to evaluate a chain, agent, or other LangChain component over
    a dataset.
- `run_on_dataset <langchain.smith.evaluation.runner_utils.run_on_dataset>`:
    Function to evaluate a chain, agent, or other LangChain component over a dataset.
- `RunEvalConfig <langchain.smith.evaluation.config.RunEvalConfig>`:
    Class representing the configuration for running evaluation.
    You can select evaluators by
    `EvaluatorType <langchain.evaluation.schema.EvaluatorType>` or config,
    or you can pass in `custom_evaluators`.
"""

from langchain_classic.smith.evaluation import (
    RunEvalConfig,
    arun_on_dataset,
    run_on_dataset,
)

__all__ = [
    "RunEvalConfig",
    "arun_on_dataset",
    "run_on_dataset",
]

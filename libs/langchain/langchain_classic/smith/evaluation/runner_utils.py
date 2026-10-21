"""在数据集上运行语言模型或链的工具。"""

from __future__ import annotations

import concurrent.futures
import dataclasses
import functools
import inspect
import logging
import re
import unicodedata
import uuid
from collections.abc import Callable
from datetime import datetime, timezone
from typing import (
    TYPE_CHECKING,
    Any,
    cast,
)
from urllib.parse import urlsplit, urlunsplit

from langchain_core._api import warn_deprecated
from langchain_core.callbacks import Callbacks
from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import BaseMessage, messages_from_dict
from langchain_core.outputs import ChatResult, LLMResult
from langchain_core.runnables import Runnable, RunnableConfig, RunnableLambda
from langchain_core.runnables import config as runnable_config
from langchain_core.runnables import utils as runnable_utils
from langchain_core.tracers.evaluation import (
    EvaluatorCallbackHandler,
    wait_for_all_evaluators,
)
from langchain_core.tracers.langchain import LangChainTracer
from langsmith.client import Client
from langsmith.env import get_git_info, get_langchain_env_var_metadata
from langsmith.evaluation import (
    EvaluationResult,
    RunEvaluator,
)
from langsmith.evaluation import (
    run_evaluator as run_evaluator_dec,
)
from langsmith.run_helpers import as_runnable, is_traceable_function
from langsmith.schemas import Dataset, DataType, Example, Run, TracerSession
from langsmith.utils import LangSmithError
from requests import HTTPError
from typing_extensions import TypedDict

from langchain_classic.chains.base import Chain
from langchain_classic.evaluation.loading import load_evaluator
from langchain_classic.evaluation.schema import (
    EvaluatorType,
    PairwiseStringEvaluator,
    StringEvaluator,
)
from langchain_classic.smith import evaluation as smith_eval
from langchain_classic.smith.evaluation import config as smith_eval_config
from langchain_classic.smith.evaluation import name_generation, progress

if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger(__name__)

MODEL_OR_CHAIN_FACTORY = (
    Callable[[], Chain | Runnable]
    | BaseLanguageModel
    | Callable[[dict], Any]
    | Runnable
    | Chain
)
MCF = Callable[[], Chain | Runnable] | BaseLanguageModel

_GIT_REMOTE_URL_SCHEMES = {"git", "http", "https", "ssh"}
_SCP_STYLE_GIT_REMOTE_RE = re.compile(
    r"^(?:[^@/:\s]+@)?(?P<host>[^@/:\s]+):(?P<path>[^\s:][^\s]*)$"
)


def _sanitize_git_remote_url(remote_url: object) -> str | None:
    if not isinstance(remote_url, str) or not remote_url:
        return None
    if any(
        character.isspace() or unicodedata.category(character) == "Cc"
        for character in remote_url
    ):
        return None

    if "://" not in remote_url:
        match = _SCP_STYLE_GIT_REMOTE_RE.fullmatch(remote_url)
        if match is None:
            return None
        host = match.group("host")
        path = match.group("path")
        if host.lower() in _GIT_REMOTE_URL_SCHEMES or (
            len(host) == 1
            and host.isascii()
            and host.isalpha()
            and path.startswith(("/", "\\"))
        ):
            return None
        return f"{host}:{path}"

    try:
        parsed = urlsplit(remote_url)
        host = parsed.hostname
        if (
            parsed.scheme not in _GIT_REMOTE_URL_SCHEMES
            or host is None
            or "%" in host
            or not parsed.path
            or parsed.query
            or parsed.fragment
        ):
            return None
        port = parsed.port
    except ValueError:
        return None

    if ":" in host:
        host = f"[{host}]"
    netloc = f"{host}:{port}" if port is not None else host
    return urlunsplit((parsed.scheme, netloc, parsed.path, "", ""))


def _format_git_tags(git_info: dict[str, Any]) -> list[str]:
    tags = []
    for key, value in git_info.items():
        if key == "remote_url":
            sanitized_remote_url = _sanitize_git_remote_url(value)
            if sanitized_remote_url is None:
                continue
            tags.append(f"git:{key}={sanitized_remote_url}")
        else:
            tags.append(f"git:{key}={value}")
    return tags


class InputFormatError(Exception):
    """当输入格式无效时抛出。"""


## 共享工具函数


class TestResult(dict):
    """单次测试运行结果的字典。"""

    def get_aggregate_feedback(
        self,
    ) -> pd.DataFrame:
        """返回反馈分数的分位数。

        This method calculates and prints the quantiles for the feedback scores
        across all feedback keys.

        Returns:
            包含每个反馈键分位数的 DataFrame。
        """
        df = self.to_dataframe()
                # 删除所有以 inputs.、outputs. 和 reference 开头的内容
        to_drop = [
            col
            for col in df.columns
            if col.startswith(("inputs.", "outputs.", "reference"))
            or col in {"input", "output"}
        ]
        return df.describe(include="all").drop(to_drop, axis=1)

    def to_dataframe(self) -> pd.DataFrame:
        """将结果转换为 DataFrame。"""
        try:
            import pandas as pd
        except ImportError as e:
            msg = (
                "Pandas is required to convert the results to a dataframe."
                " to install pandas, run `pip install pandas`."
            )
            raise ImportError(msg) from e

        indices = []
        records = []
        for example_id, result in self["results"].items():
            feedback = result["feedback"]
            output_ = result.get("output")
            if isinstance(output_, dict):
                output = {f"outputs.{k}": v for k, v in output_.items()}
            elif output_ is None:
                output = {}
            else:
                output = {"output": output_}

            r = {
                **{f"inputs.{k}": v for k, v in result["input"].items()},
                **output,
            }
            if "reference" in result:
                if isinstance(result["reference"], dict):
                    r.update(
                        {f"reference.{k}": v for k, v in result["reference"].items()},
                    )
                else:
                    r["reference"] = result["reference"]
            r.update(
                {
                    **{f"feedback.{f.key}": f.score for f in feedback},
                    "error": result.get("Error"),
                    "execution_time": result["execution_time"],
                    "run_id": result.get("run_id"),
                },
            )
            records.append(r)
            indices.append(example_id)

        return pd.DataFrame(records, index=indices)


class EvalError(dict):
    """你的架构引发了错误。"""

    def __init__(self, Error: BaseException, **kwargs: Any) -> None:  # noqa: N803
        """使用错误和其他属性初始化 `EvalError`。

        Args:
            Error: 发生的错误。
            **kwargs: 要包含在错误中的附加属性。
        """
        super().__init__(Error=Error, **kwargs)

    def __getattr__(self, name: str) -> Any:
        """从 `EvalError` 中获取属性。

        Args:
            name: 要获取的属性名称。

        Returns:
            属性的值。

        Raises:
            AttributeError: 如果属性不存在。
        """
        try:
            return self[name]
        except KeyError as e:
            msg = f"'EvalError' object has no attribute '{name}'"
            raise AttributeError(msg) from e


def _wrap_in_chain_factory(
    llm_or_chain_factory: MODEL_OR_CHAIN_FACTORY,
    dataset_name: str = "<my_dataset>",
) -> MCF:
    """用链工厂包装。

    如果用户传入了没有记忆的链而非链工厂，予以宽恕。
    这是一个常见错误。同时抛出更友好的错误信息。
    """
    if isinstance(llm_or_chain_factory, Chain):
        chain = llm_or_chain_factory
        chain_class = chain.__class__.__name__
        if llm_or_chain_factory.memory is not None:
            memory_class = chain.memory.__class__.__name__
            msg = (
                "Cannot directly evaluate a chain with stateful memory."
                " To evaluate this chain, pass in a chain constructor"
                " that initializes fresh memory each time it is called."
                "  This will safeguard against information"
                " leakage between dataset examples."
                "\nFor example:\n\n"
                "def chain_constructor():\n"
                f"    new_memory = {memory_class}(...)\n"
                f"    return {chain_class}"
                "(memory=new_memory, ...)\n\n"
                f'run_on_dataset("{dataset_name}", chain_constructor, ...)'
            )
            raise ValueError(msg)
        return lambda: chain
    if isinstance(llm_or_chain_factory, BaseLanguageModel):
        return llm_or_chain_factory
    if isinstance(llm_or_chain_factory, Runnable):
                # 此处可能存在内存，但检查所有情况并不优雅。
        lcf = llm_or_chain_factory
        return lambda: lcf
    if callable(llm_or_chain_factory):
        if is_traceable_function(llm_or_chain_factory):
            runnable_ = as_runnable(cast("Callable", llm_or_chain_factory))
            return lambda: runnable_
        try:
            _model = llm_or_chain_factory()  # type: ignore[call-arg]
        except TypeError:
                        # 这是一个任意函数，将其包装在 RunnableLambda 中
            user_func = cast("Callable", llm_or_chain_factory)
            sig = inspect.signature(user_func)
            logger.info("Wrapping function %s as RunnableLambda.", sig)
            wrapped = RunnableLambda(user_func)
            return lambda: wrapped
        constructor = cast("Callable", llm_or_chain_factory)
        if isinstance(_model, BaseLanguageModel):
                        # 通常会使用 LLM 构造函数而非原始 LLM，
                        # 因此我们将为其解包以方便用户使用。
            return _model
        if is_traceable_function(cast("Callable", _model)):
            runnable_ = as_runnable(cast("Callable", _model))
            return lambda: runnable_
        if not isinstance(_model, Runnable):
                        # 这种情况不太可能发生——对模型函数的构造函数
            return lambda: RunnableLambda(constructor)
                # 典型正确情况
        return constructor
    return llm_or_chain_factory  # type: ignore[unreachable]


def _get_prompt(inputs: dict[str, Any]) -> str:
    """从输入中获取提示词。

    Args:
        inputs: 输入字典。

    Returns:
        一个字符串提示词。

    Raises:
        InputFormatError: 如果输入格式无效。
    """
    if not inputs:
        msg = "Inputs should not be empty."
        raise InputFormatError(msg)

    prompts = []
    if "prompt" in inputs:
        if not isinstance(inputs["prompt"], str):
            msg = f"Expected string for 'prompt', got {type(inputs['prompt']).__name__}"
            raise InputFormatError(msg)
        prompts = [inputs["prompt"]]
    elif "prompts" in inputs:
        if not isinstance(inputs["prompts"], list) or not all(
            isinstance(i, str) for i in inputs["prompts"]
        ):
            msg = (
                "Expected list of strings for 'prompts',"
                f" got {type(inputs['prompts']).__name__}"
            )
            raise InputFormatError(msg)
        prompts = inputs["prompts"]
    elif len(inputs) == 1:
        prompt_ = next(iter(inputs.values()))
        if isinstance(prompt_, str):
            prompts = [prompt_]
        elif isinstance(prompt_, list) and all(isinstance(i, str) for i in prompt_):
            prompts = prompt_
        else:
            msg = f"LLM Run expects string prompt input. Got {inputs}"
            raise InputFormatError(msg)
    else:
        msg = f"LLM Run expects 'prompt' or 'prompts' in inputs. Got {inputs}"
        raise InputFormatError(msg)
    if len(prompts) == 1:
        return prompts[0]
    msg = f"LLM Run expects single prompt input. Got {len(prompts)} prompts."
    raise InputFormatError(msg)


class ChatModelInput(TypedDict):
    """聊天模型的输入。"""

    messages: list[BaseMessage]


def _get_messages(inputs: dict[str, Any]) -> dict:
    """从输入中获取聊天消息。

    Args:
        inputs: 输入字典。

    Returns:
        聊天消息列表。

    Raises:
        InputFormatError: 如果输入格式无效。
    """
    if not inputs:
        msg = "Inputs should not be empty."
        raise InputFormatError(msg)
    input_copy = inputs.copy()
    if "messages" in inputs:
        input_copy["input"] = input_copy.pop("messages")
    elif len(inputs) == 1:
        input_copy["input"] = next(iter(inputs.values()))
    if "input" in input_copy:
        raw_messages = input_copy["input"]
        if isinstance(raw_messages, list) and all(
            isinstance(i, dict) for i in raw_messages
        ):
            raw_messages = [raw_messages]
        if len(raw_messages) == 1:
            input_copy["input"] = messages_from_dict(raw_messages[0])
        else:
            msg = (
                "Batch messages not supported. Please provide a"
                " single list of messages."
            )
            raise InputFormatError(msg)
        return input_copy
    msg = (
        f"Chat Run expects single List[dict] or List[List[dict]] 'messages'"
        f" input. Got {inputs}"
    )
    raise InputFormatError(msg)


## 共享数据验证工具函数
def _validate_example_inputs_for_language_model(
    first_example: Example,
    input_mapper: Callable[[dict], Any] | None,
) -> None:
    if input_mapper:
        prompt_input = input_mapper(first_example.inputs or {})
        if not isinstance(prompt_input, str) and not (
            isinstance(prompt_input, list)
            and all(isinstance(msg, BaseMessage) for msg in prompt_input)
        ):
            msg = (
                "When using an input_mapper to prepare dataset example inputs"
                " for an LLM or chat model, the output must a single string or"
                " a list of chat messages."
                f"\nGot: {prompt_input} of type {type(prompt_input)}."
            )
            raise InputFormatError(msg)
    else:
        try:
            _get_prompt(first_example.inputs or {})
        except InputFormatError:
            try:
                _get_messages(first_example.inputs or {})
            except InputFormatError as err2:
                msg = (
                    "Example inputs do not match language model input format. "
                    "Expected a dictionary with messages or a single prompt."
                    f" Got: {first_example.inputs}"
                    " Please update your dataset OR provide an input_mapper"
                    " to convert the example.inputs to a compatible format"
                    " for the llm or chat model you wish to evaluate."
                )
                raise InputFormatError(msg) from err2


def _validate_example_inputs_for_chain(
    first_example: Example,
    chain: Chain,
    input_mapper: Callable[[dict], Any] | None,
) -> None:
    """验证示例输入是否匹配链的输入键。"""
    if input_mapper:
        first_inputs = input_mapper(first_example.inputs or {})
        missing_keys = set(chain.input_keys).difference(first_inputs)
        if not isinstance(first_inputs, dict):
            msg = (
                "When using an input_mapper to prepare dataset example"
                " inputs for a chain, the mapped value must be a dictionary."
                f"\nGot: {first_inputs} of type {type(first_inputs)}."
            )
            raise InputFormatError(msg)
        if missing_keys:
            msg = (
                "Missing keys after loading example using input_mapper."
                f"\nExpected: {chain.input_keys}. Got: {first_inputs.keys()}"
            )
            raise InputFormatError(msg)
    else:
        first_inputs = first_example.inputs or {}
        missing_keys = set(chain.input_keys).difference(first_inputs)
        if len(first_inputs) == 1 and len(chain.input_keys) == 1:
                        # 我们可以将其通过 run 方法传递。
                        # 避免调用以进行验证。
            pass
        elif missing_keys:
            msg = (
                "Example inputs missing expected chain input keys."
                " Please provide an input_mapper to convert the example.inputs"
                " to a compatible format for the chain you wish to evaluate."
                f"Expected: {chain.input_keys}. "
                f"Got: {first_inputs.keys()}"
            )
            raise InputFormatError(msg)


def _validate_example_inputs(
    example: Example,
    llm_or_chain_factory: MCF,
    input_mapper: Callable[[dict], Any] | None,
) -> None:
    """验证示例输入对模型是否有效。"""
    if isinstance(llm_or_chain_factory, BaseLanguageModel):
        _validate_example_inputs_for_language_model(example, input_mapper)
    else:
        chain = llm_or_chain_factory()
        if isinstance(chain, Chain):
                        # 否则它就是一个 Runnable
            _validate_example_inputs_for_chain(example, chain, input_mapper)
        elif isinstance(chain, Runnable):
            logger.debug("Skipping input validation for %s", chain)


## 共享评估器设置工具函数


def _setup_evaluation(
    llm_or_chain_factory: MCF,
    examples: list[Example],
    evaluation: smith_eval.RunEvalConfig | None,
    data_type: DataType,
) -> list[RunEvaluator] | None:
    """配置评估器以在链的结果上运行。"""
    if evaluation:
        if isinstance(llm_or_chain_factory, BaseLanguageModel):
            run_inputs, run_outputs = None, None
            run_type = "llm"
        else:
            run_type = "chain"
            chain = llm_or_chain_factory()
            run_inputs = chain.input_keys if isinstance(chain, Chain) else None
            run_outputs = chain.output_keys if isinstance(chain, Chain) else None
        run_evaluators = _load_run_evaluators(
            evaluation,
            run_type,
            data_type,
            list(examples[0].outputs) if examples[0].outputs else None,
            run_inputs,
            run_outputs,
        )
    else:
                # TODO: 创建一个默认的有用性评估器
        run_evaluators = None
    return run_evaluators


def _determine_input_key(
    config: smith_eval.RunEvalConfig,
    run_inputs: list[str] | None,
) -> str | None:
    input_key = None
    if config.input_key:
        input_key = config.input_key
        if run_inputs and input_key not in run_inputs:
            logger.warning(
                "Input key %s not in chain's specified input keys %s. "
                "Evaluation behavior may be undefined.",
                input_key,
                run_inputs,
            )
    elif run_inputs and len(run_inputs) == 1:
        input_key = run_inputs[0]
    elif run_inputs is not None and len(run_inputs) > 1:
        logger.warning(
            "Chain expects multiple input keys: %s,"
            " Evaluator is likely to fail. Evaluation behavior may be undefined."
            " Specify an input_key in the RunEvalConfig to avoid this warning.",
            run_inputs,
        )

    return input_key


def _determine_prediction_key(
    config: smith_eval.RunEvalConfig,
    run_outputs: list[str] | None,
) -> str | None:
    prediction_key = None
    if config.prediction_key:
        prediction_key = config.prediction_key
        if run_outputs and prediction_key not in run_outputs:
            logger.warning(
                "Prediction key %s not in chain's specified output keys %s. "
                "Evaluation behavior may be undefined.",
                prediction_key,
                run_outputs,
            )
    elif run_outputs and len(run_outputs) == 1:
        prediction_key = run_outputs[0]
    elif run_outputs is not None and len(run_outputs) > 1:
        logger.warning(
            "Chain expects multiple output keys: %s,"
            " Evaluation behavior may be undefined. Specify a prediction_key"
            " in the RunEvalConfig to avoid this warning.",
            run_outputs,
        )
    return prediction_key


def _determine_reference_key(
    config: smith_eval.RunEvalConfig,
    example_outputs: list[str] | None,
) -> str | None:
    if config.reference_key:
        reference_key = config.reference_key
        if example_outputs and reference_key not in example_outputs:
            msg = (
                f"Reference key {reference_key} not in Dataset"
                f" example outputs: {example_outputs}"
            )
            raise ValueError(msg)
    elif example_outputs and len(example_outputs) == 1:
        reference_key = next(iter(example_outputs))
    else:
        reference_key = None
    return reference_key


def _construct_run_evaluator(
    eval_config: smith_eval_config.SINGLE_EVAL_CONFIG_TYPE
    | smith_eval_config.CUSTOM_EVALUATOR_TYPE,
    eval_llm: BaseLanguageModel | None,
    run_type: str,
    data_type: DataType,
    example_outputs: list[str] | None,
    reference_key: str | None,
    input_key: str | None,
    prediction_key: str | None,
) -> RunEvaluator:
    if isinstance(eval_config, RunEvaluator):
        return eval_config
    if isinstance(eval_config, (EvaluatorType, str)):
        if not isinstance(eval_config, EvaluatorType):
            eval_config = EvaluatorType(eval_config)
        evaluator_ = load_evaluator(eval_config, llm=eval_llm)
        eval_type_tag = eval_config.value
    elif isinstance(eval_config, smith_eval_config.EvalConfig):
        kwargs = {"llm": eval_llm, **eval_config.get_kwargs()}
        evaluator_ = load_evaluator(eval_config.evaluator_type, **kwargs)
        eval_type_tag = eval_config.evaluator_type.value
                # 如果在配置中指定了，则覆盖键值
        if isinstance(eval_config, smith_eval_config.SingleKeyEvalConfig):
            input_key = eval_config.input_key or input_key
            prediction_key = eval_config.prediction_key or prediction_key
            reference_key = eval_config.reference_key or reference_key
    elif callable(eval_config):
                # 假设我们可以进行装饰
        return run_evaluator_dec(eval_config)
    else:
        msg = f"Unknown evaluator type: {type(eval_config)}"
        raise ValueError(msg)  # noqa: TRY004

    if isinstance(evaluator_, StringEvaluator):
        if evaluator_.requires_reference and reference_key is None:
            msg = (
                f"Must specify reference_key in smith_eval.RunEvalConfig to use"
                f" evaluator of type {eval_type_tag} with"
                f" dataset with multiple output keys: {example_outputs}."
            )
            raise ValueError(msg)
        run_evaluator = smith_eval.StringRunEvaluatorChain.from_run_and_data_type(
            evaluator_,
            run_type,
            data_type,
            input_key=input_key,
            prediction_key=prediction_key,
            reference_key=reference_key,
            tags=[eval_type_tag],
        )
    elif isinstance(evaluator_, PairwiseStringEvaluator):
        msg = (
            f"Run evaluator for {eval_type_tag} is not implemented."
            " PairwiseStringEvaluators compare the outputs of two different models"
            " rather than the output of a single model."
            " Did you mean to use a StringEvaluator instead?"
            "\nSee: https://python.langchain.com/docs/guides/evaluation/string/"
        )
        raise NotImplementedError(msg)

    else:
        msg = f"Run evaluator for {eval_type_tag} is not implemented"
        raise NotImplementedError(msg)
    return run_evaluator


def _get_keys(
    config: smith_eval.RunEvalConfig,
    run_inputs: list[str] | None,
    run_outputs: list[str] | None,
    example_outputs: list[str] | None,
) -> tuple[str | None, str | None, str | None]:
    input_key = _determine_input_key(config, run_inputs)
    prediction_key = _determine_prediction_key(config, run_outputs)
    reference_key = _determine_reference_key(config, example_outputs)
    return input_key, prediction_key, reference_key


def _load_run_evaluators(
    config: smith_eval.RunEvalConfig,
    run_type: str,
    data_type: DataType,
    example_outputs: list[str] | None,
    run_inputs: list[str] | None,
    run_outputs: list[str] | None,
) -> list[RunEvaluator]:
    """从配置加载运行评估器。

    Args:
        config: 运行评估器的配置。
        run_type: 运行类型。
        data_type: 运行中使用的数据集类型。
        example_outputs: 示例输出。
        run_inputs: 运行的输入键。
        run_outputs: 运行的输出键。

    Returns:
        评估器列表。
    """
    run_evaluators = []
    input_key, prediction_key, reference_key = None, None, None
    if config.evaluators or (
        config.custom_evaluators
        and any(isinstance(e, StringEvaluator) for e in config.custom_evaluators)
    ):
        input_key, prediction_key, reference_key = _get_keys(
            config,
            run_inputs,
            run_outputs,
            example_outputs,
        )
    for eval_config in config.evaluators:
        run_evaluator = _construct_run_evaluator(
            eval_config,
            config.eval_llm,
            run_type,
            data_type,
            example_outputs,
            reference_key,
            input_key,
            prediction_key,
        )
        run_evaluators.append(run_evaluator)
    custom_evaluators = config.custom_evaluators or []
    for custom_evaluator in custom_evaluators:
        if isinstance(custom_evaluator, RunEvaluator):
            run_evaluators.append(custom_evaluator)
        elif isinstance(custom_evaluator, StringEvaluator):
            run_evaluators.append(
                smith_eval.StringRunEvaluatorChain.from_run_and_data_type(
                    custom_evaluator,
                    run_type,
                    data_type,
                    input_key=input_key,
                    prediction_key=prediction_key,
                    reference_key=reference_key,
                ),
            )
        elif callable(custom_evaluator):
            run_evaluators.append(run_evaluator_dec(custom_evaluator))
        else:
            msg = (  # type: ignore[unreachable]
                f"Unsupported custom evaluator: {custom_evaluator}."
                f" Expected RunEvaluator or StringEvaluator."
            )
            raise ValueError(msg)  # noqa: TRY004

    return run_evaluators


### 异步辅助函数


async def _arun_llm(
    llm: BaseLanguageModel,
    inputs: dict[str, Any],
    *,
    tags: list[str] | None = None,
    callbacks: Callbacks = None,
    input_mapper: Callable[[dict], Any] | None = None,
    metadata: dict[str, Any] | None = None,
) -> str | BaseMessage:
    """异步运行语言模型。

    Args:
        llm: 要运行的语言模型。
        inputs: 输入字典。
        tags: 可选标签，添加到运行中。
        callbacks: 可选回调，在运行期间使用。
        input_mapper: 可选函数，用于将输入映射到期望格式。
        metadata: 可选元数据，添加到运行中。

    Returns:
        LLMResult 或 ChatResult。

    Raises:
        ValueError: 如果不支持该 LLM 类型。
        InputFormatError: 如果输入格式无效。
    """
    if input_mapper is not None:
        prompt_or_messages = input_mapper(inputs)
        if isinstance(prompt_or_messages, str) or (
            isinstance(prompt_or_messages, list)
            and all(isinstance(msg, BaseMessage) for msg in prompt_or_messages)
        ):
            return await llm.ainvoke(
                prompt_or_messages,
                config=RunnableConfig(
                    callbacks=callbacks,
                    tags=tags or [],
                    metadata=metadata or {},
                ),
            )
        msg = (
            "Input mapper returned invalid format"
            f" {prompt_or_messages}"
            "\nExpected a single string or list of chat messages."
        )
        raise InputFormatError(msg)

    try:
        prompt = _get_prompt(inputs)
        llm_output: str | BaseMessage = await llm.ainvoke(
            prompt,
            config=RunnableConfig(
                callbacks=callbacks,
                tags=tags or [],
                metadata=metadata or {},
            ),
        )
    except InputFormatError:
        llm_inputs = _get_messages(inputs)
        llm_output = await llm.ainvoke(
            **llm_inputs,
            config=RunnableConfig(
                callbacks=callbacks,
                tags=tags or [],
                metadata=metadata or {},
            ),
        )
    return llm_output


async def _arun_chain(
    chain: Chain | Runnable,
    inputs: dict[str, Any],
    callbacks: Callbacks,
    *,
    tags: list[str] | None = None,
    input_mapper: Callable[[dict], Any] | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict | str:
    """在输入上异步运行链。"""
    inputs_ = inputs if input_mapper is None else input_mapper(inputs)
    if (
        isinstance(chain, Chain)
        and isinstance(inputs_, dict)
        and len(inputs_) == 1
        and chain.input_keys
    ):
        val = next(iter(inputs_.values()))
        output = await chain.ainvoke(
            val,
            config=RunnableConfig(
                callbacks=callbacks,
                tags=tags or [],
                metadata=metadata or {},
            ),
        )
    else:
        runnable_config = RunnableConfig(
            tags=tags or [],
            callbacks=callbacks,
            metadata=metadata or {},
        )
        output = await chain.ainvoke(inputs_, config=runnable_config)
    return output


async def _arun_llm_or_chain(
    example: Example,
    config: RunnableConfig,
    *,
    llm_or_chain_factory: MCF,
    input_mapper: Callable[[dict], Any] | None = None,
) -> dict | str | LLMResult | ChatResult:
    """异步运行链或语言模型。

    Args:
        example: 要运行的示例。
        config: 运行的配置。
        llm_or_chain_factory: 要运行的链或语言模型构造器。
        input_mapper: 可选函数，用于将输入映射到期望格式。

    Returns:
        输出列表。
    """
    chain_or_llm = (
        "LLM" if isinstance(llm_or_chain_factory, BaseLanguageModel) else "Chain"
    )
    result = None
    try:
        if isinstance(llm_or_chain_factory, BaseLanguageModel):
            output: Any = await _arun_llm(
                llm_or_chain_factory,
                example.inputs or {},
                tags=config["tags"],
                callbacks=config["callbacks"],
                input_mapper=input_mapper,
                metadata=config.get("metadata"),
            )
        else:
            chain = llm_or_chain_factory()
            output = await _arun_chain(
                chain,
                example.inputs or {},
                tags=config["tags"],
                callbacks=config["callbacks"],
                input_mapper=input_mapper,
                metadata=config.get("metadata"),
            )
        result = output
    except Exception as e:  # noqa: BLE001
        logger.warning(
            "%s failed for example %s with inputs %s\n%s",
            chain_or_llm,
            example.id,
            example.inputs,
            e,
        )
        result = EvalError(Error=e)
    return result


## 同步工具函数


def _run_llm(
    llm: BaseLanguageModel,
    inputs: dict[str, Any],
    callbacks: Callbacks,
    *,
    tags: list[str] | None = None,
    input_mapper: Callable[[dict], Any] | None = None,
    metadata: dict[str, Any] | None = None,
) -> str | BaseMessage:
    """在示例上运行语言模型。

    Args:
        llm: 要运行的语言模型。
        inputs: 输入字典。
        callbacks: 运行期间使用的回调。
        tags: 可选标签，添加到运行中。
        input_mapper: 从 Example 映射输入字典的函数
        metadata: 可选元数据，添加到运行中。

    Returns:
        LLMResult 或 ChatResult。

    Raises:
        ValueError: 如果不支持该 LLM 类型。
        InputFormatError: 如果输入格式无效。
    """
        # 大部分代码是遗留代码；我们可能可以移除其中很多部分。
    if input_mapper is not None:
        prompt_or_messages = input_mapper(inputs)
        if isinstance(prompt_or_messages, str) or (
            isinstance(prompt_or_messages, list)
            and all(isinstance(msg, BaseMessage) for msg in prompt_or_messages)
        ):
            llm_output: str | BaseMessage = llm.invoke(
                prompt_or_messages,
                config=RunnableConfig(
                    callbacks=callbacks,
                    tags=tags or [],
                    metadata=metadata or {},
                ),
            )
        else:
            msg = (
                "Input mapper returned invalid format: "
                f" {prompt_or_messages}"
                "\nExpected a single string or list of chat messages."
            )
            raise InputFormatError(msg)
    else:
        try:
            llm_prompts = _get_prompt(inputs)
            llm_output = llm.invoke(
                llm_prompts,
                config=RunnableConfig(
                    callbacks=callbacks,
                    tags=tags or [],
                    metadata=metadata or {},
                ),
            )
        except InputFormatError:
            llm_inputs = _get_messages(inputs)
            llm_output = llm.invoke(
                **llm_inputs,
                config=RunnableConfig(callbacks=callbacks, metadata=metadata or {}),
            )
    return llm_output


def _run_chain(
    chain: Chain | Runnable,
    inputs: dict[str, Any],
    callbacks: Callbacks,
    *,
    tags: list[str] | None = None,
    input_mapper: Callable[[dict], Any] | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict | str:
    """在输入上运行链。"""
    inputs_ = inputs if input_mapper is None else input_mapper(inputs)
    if (
        isinstance(chain, Chain)
        and isinstance(inputs_, dict)
        and len(inputs_) == 1
        and chain.input_keys
    ):
        val = next(iter(inputs_.values()))
        output = chain.invoke(
            val,
            config=RunnableConfig(
                callbacks=callbacks,
                tags=tags or [],
                metadata=metadata or {},
            ),
        )
    else:
        runnable_config = RunnableConfig(
            tags=tags or [],
            callbacks=callbacks,
            metadata=metadata or {},
        )
        output = chain.invoke(inputs_, config=runnable_config)
    return output


def _run_llm_or_chain(
    example: Example,
    config: RunnableConfig,
    *,
    llm_or_chain_factory: MCF,
    input_mapper: Callable[[dict], Any] | None = None,
) -> dict | str | LLMResult | ChatResult:
    """同步运行链或语言模型。

    Args:
        example: 要运行的示例。
        config: 运行的配置。
        llm_or_chain_factory: 要运行的链或语言模型构造器。
        input_mapper: 可选函数，用于将输入映射到期望格式。

    Returns:
        模型或链的输出。
    """
    chain_or_llm = (
        "LLM" if isinstance(llm_or_chain_factory, BaseLanguageModel) else "Chain"
    )
    result = None
    try:
        if isinstance(llm_or_chain_factory, BaseLanguageModel):
            output: Any = _run_llm(
                llm_or_chain_factory,
                example.inputs or {},
                config["callbacks"],
                tags=config["tags"],
                input_mapper=input_mapper,
                metadata=config.get("metadata"),
            )
        else:
            chain = llm_or_chain_factory()
            output = _run_chain(
                chain,
                example.inputs or {},
                config["callbacks"],
                tags=config["tags"],
                input_mapper=input_mapper,
                metadata=config.get("metadata"),
            )
        result = output
    except Exception as e:  # noqa: BLE001
        error_type = type(e).__name__
        logger.warning(
            "%s failed for example %s with inputs %s\nError Type: %s, Message: %s",
            chain_or_llm,
            example.id,
            example.inputs,
            error_type,
            e,
        )
        result = EvalError(Error=e)
    return result


def _prepare_eval_run(
    client: Client,
    dataset_name: str,
    llm_or_chain_factory: MODEL_OR_CHAIN_FACTORY,
    project_name: str,
    project_metadata: dict[str, Any] | None = None,
    tags: list[str] | None = None,
    dataset_version: str | datetime | None = None,
) -> tuple[MCF, TracerSession, Dataset, list[Example]]:
    wrapped_model = _wrap_in_chain_factory(llm_or_chain_factory, dataset_name)
    dataset = client.read_dataset(dataset_name=dataset_name)

    examples = list(client.list_examples(dataset_id=dataset.id, as_of=dataset_version))
    if not examples:
        msg = f"Dataset {dataset_name} has no example rows."
        raise ValueError(msg)
    modified_at = [ex.modified_at for ex in examples if ex.modified_at]
        # 在实际获取时应始终定义，
        # 但类型允许为 None
    max_modified_at = max(modified_at) if modified_at else None
    inferred_version = max_modified_at.isoformat() if max_modified_at else None

    try:
        project_metadata = project_metadata or {}
        git_info = get_git_info()
        if git_info:
            project_metadata = {
                **project_metadata,
                "git": git_info,
            }

        project_metadata["dataset_version"] = inferred_version
        project = client.create_project(
            project_name,
            reference_dataset_id=dataset.id,
            project_extra={"tags": tags} if tags else {},
            metadata=project_metadata,
        )
    except (HTTPError, ValueError, LangSmithError) as e:
        if "already exists " not in str(e):
            raise
        uid = uuid.uuid4()
        example_msg = f"""
run_on_dataset(
    ...
    project_name="{project_name} - {uid}", # Update since {project_name} already exists
)
"""
        msg = (
            f"Test project {project_name} already exists. Please use a different name:"
            f"\n\n{example_msg}"
        )
        raise ValueError(msg) from e
    comparison_url = dataset.url + f"/compare?selectedSessions={project.id}"
    print(  # noqa: T201
        f"View the evaluation results for project '{project_name}'"
        f" at:\n{comparison_url}\n\n"
        f"View all tests for Dataset {dataset_name} at:\n{dataset.url}",
        flush=True,
    )
    return wrapped_model, project, dataset, examples


class _RowResult(TypedDict, total=False):
    """单个示例行结果的字典。"""

    feedback: list[EvaluationResult] | None
    execution_time: float | None
    run_id: str | None


@dataclasses.dataclass
class _DatasetRunContainer:
    """一个用于管理评估运行状态的容器。"""

    client: Client
    project: TracerSession
    wrapped_model: MCF
    examples: list[Example]
    configs: list[RunnableConfig]
    batch_evaluators: list[smith_eval_config.BATCH_EVALUATOR_LIKE] | None = None

    def _merge_test_outputs(
        self,
        batch_results: list,
        all_eval_results: dict[str, _RowResult],
    ) -> dict:
        results: dict = {}
        for example, output in zip(self.examples, batch_results, strict=False):
            row_result = all_eval_results.get(str(example.id), {})
            results[str(example.id)] = {
                "input": example.inputs,
                "feedback": row_result.get("feedback", []),
                "execution_time": row_result.get("execution_time"),
                "run_id": row_result.get("run_id"),
            }
            if isinstance(output, EvalError):
                results[str(example.id)]["Error"] = output.Error
            else:
                results[str(example.id)]["output"] = output
            if example.outputs:
                results[str(example.id)]["reference"] = example.outputs
        return results

    def _run_batch_evaluators(self, runs: dict[str, Run]) -> list[dict]:
        evaluators = self.batch_evaluators
        if not evaluators:
            return []
        runs_list = [runs[str(example.id)] for example in self.examples]
        aggregate_feedback = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for evaluator in evaluators:
                try:
                    result = evaluator(runs_list, self.examples)
                    if isinstance(result, EvaluationResult):
                        result = result.model_dump()
                    aggregate_feedback.append(cast("dict", result))
                    executor.submit(
                        self.client.create_feedback,
                        **result,
                        run_id=None,
                        project_id=self.project.id,
                    )
                except Exception:
                    logger.exception(
                        "Error running batch evaluator %s", repr(evaluator)
                    )
        return aggregate_feedback

    def _collect_metrics(self) -> tuple[dict[str, _RowResult], dict[str, Run]]:
        all_eval_results: dict = {}
        all_runs: dict = {}
        for c in self.configs:
            for callback in cast("list", c["callbacks"]):
                if isinstance(callback, EvaluatorCallbackHandler):
                    eval_results = callback.logged_eval_results
                    for (_, example_id), v in eval_results.items():
                        all_eval_results.setdefault(str(example_id), {}).update(
                            {"feedback": v},
                        )
                elif isinstance(callback, LangChainTracer):
                    run = callback.latest_run
                    execution_time = (
                        (run.end_time - run.start_time).total_seconds()
                        if run and run.end_time
                        else None
                    )
                    run_id = str(run.id) if run else None
                    all_eval_results.setdefault(str(callback.example_id), {}).update(
                        {
                            "execution_time": execution_time,
                            "run_id": run_id,
                            "run": run,
                        },
                    )
                    all_runs[str(callback.example_id)] = run
        return cast("dict[str, _RowResult]", all_eval_results), all_runs

    def _collect_test_results(
        self,
        batch_results: list[dict | str | LLMResult | ChatResult],
    ) -> TestResult:
        logger.info("Waiting for evaluators to complete.")
        wait_for_all_evaluators()
        all_eval_results, all_runs = self._collect_metrics()
        aggregate_feedback = None
        if self.batch_evaluators:
            logger.info("Running session evaluators.")
            aggregate_feedback = self._run_batch_evaluators(all_runs)
        results = self._merge_test_outputs(batch_results, all_eval_results)
        return TestResult(
            project_name=self.project.name,
            results=results,
            aggregate_metrics=aggregate_feedback,
        )

    def finish(
        self,
        batch_results: list,
        verbose: bool = False,  # noqa: FBT001,FBT002
    ) -> TestResult:
        results = self._collect_test_results(batch_results)
        if verbose:
            try:
                agg_feedback = results.get_aggregate_feedback()
                _display_aggregate_results(agg_feedback)
            except Exception as e:  # noqa: BLE001
                logger.debug("Failed to print aggregate feedback: %s", e, exc_info=True)
        try:
                        # 关闭项目允许更改名称和优化指标
            self.client.update_project(
                self.project.id,
                end_time=datetime.now(timezone.utc),
            )
        except Exception as e:  # noqa: BLE001
            logger.debug("Failed to close project: %s", e, exc_info=True)
        return results

    @classmethod
    def prepare(
        cls,
        client: Client,
        dataset_name: str,
        llm_or_chain_factory: MODEL_OR_CHAIN_FACTORY,
        project_name: str | None,
        evaluation: smith_eval.RunEvalConfig | None = None,
        tags: list[str] | None = None,
        input_mapper: Callable[[dict], Any] | None = None,
        concurrency_level: int = 5,
        project_metadata: dict[str, Any] | None = None,
        revision_id: str | None = None,
        dataset_version: datetime | str | None = None,
    ) -> _DatasetRunContainer:
        project_name = project_name or name_generation.random_name()
        if revision_id:
            if not project_metadata:
                project_metadata = {}
            project_metadata.update({"revision_id": revision_id})
        wrapped_model, project, dataset, examples = _prepare_eval_run(
            client,
            dataset_name,
            llm_or_chain_factory,
            project_name,
            project_metadata=project_metadata,
            tags=tags,
            dataset_version=dataset_version,
        )
        tags = tags or []
        tags.extend(_format_git_tags(project.metadata.get("git") or {}))
        run_metadata = {"dataset_version": project.metadata["dataset_version"]}
        if revision_id:
            run_metadata["revision_id"] = revision_id
        wrapped_model = _wrap_in_chain_factory(llm_or_chain_factory)
        run_evaluators = _setup_evaluation(
            wrapped_model,
            examples,
            evaluation,
            dataset.data_type or DataType.kv,
        )
        _validate_example_inputs(examples[0], wrapped_model, input_mapper)
        progress_bar = progress.ProgressBarCallback(len(examples))
        configs = [
            RunnableConfig(
                callbacks=[
                    LangChainTracer(
                        project_name=project.name,
                        client=client,
                        example_id=example.id,
                    ),
                    EvaluatorCallbackHandler(
                        evaluators=run_evaluators or [],
                        client=client,
                        example_id=example.id,
                        max_concurrency=0,
                    ),
                    progress_bar,
                ],
                tags=tags,
                max_concurrency=concurrency_level,
                metadata=run_metadata,
            )
            for example in examples
        ]
        return cls(
            client=client,
            project=project,
            wrapped_model=wrapped_model,
            examples=examples,
            configs=configs,
            batch_evaluators=evaluation.batch_evaluators if evaluation else None,
        )


def _is_jupyter_environment() -> bool:
    try:
        from IPython.core.getipython import get_ipython

        res = get_ipython()  # type: ignore[no-untyped-call]
        return res is not None and "zmqshell" in str(type(res))
    except ImportError:
        return False


def _display_aggregate_results(aggregate_results: pd.DataFrame) -> None:
    if _is_jupyter_environment():
        from IPython.display import HTML, display

        display(HTML("<h3>Experiment Results:</h3>"))  # type: ignore[no-untyped-call]
        display(aggregate_results)  # type: ignore[no-untyped-call]
    else:
        formatted_string = aggregate_results.to_string(
            float_format=lambda x: f"{x:.2f}",
            justify="right",
        )
        print("\n Experiment Results:")  # noqa: T201
        print(formatted_string)  # noqa: T201


_INPUT_MAPPER_DEP_WARNING = (
    "The input_mapper argument is deprecated and "
    "will be removed in a future release. Please add a "
    " RunnableLambda to your chain to map inputs to the expected format"
    " instead. Example:\n"
    "def construct_chain():\n"
    "    my_chain = ...\n"
    "    input_mapper = {'other_key': 'MyOtherInput', 'my_input_key': x}\n"
    "    return input_mapper | my_chain\n"
    "run_on_dataset(..., llm_or_chain_factory=construct_chain)\n"
    "(See https://api.python.langchain.com/en/latest/schema/"
    "langchain.schema.runnable.base.RunnableLambda.html)"
)

## 公共 API


async def arun_on_dataset(
    client: Client | None,
    dataset_name: str,
    llm_or_chain_factory: MODEL_OR_CHAIN_FACTORY,
    *,
    evaluation: smith_eval.RunEvalConfig | None = None,
    dataset_version: datetime | str | None = None,
    concurrency_level: int = 5,
    project_name: str | None = None,
    project_metadata: dict[str, Any] | None = None,
    verbose: bool = False,
    revision_id: str | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """在数据集上运行。

    在数据集上运行链或语言模型，并将追踪信息存储到指定的项目名称中。

    For the (usually faster) async version of this function,
    see `arun_on_dataset`.

    Args:
        dataset_name: Name of the dataset to run the chain on.
        llm_or_chain_factory: Language model or Chain constructor to run
            over the dataset. The Chain constructor is used to permit
            independent calls on each example without carrying over state.
        evaluation: Configuration for evaluators to run on the
            results of the chain.
        dataset_version: Optional version of the dataset.
        concurrency_level: The number of async tasks to run concurrently.
        project_name: Name of the project to store the traces in.
            Defaults to `{dataset_name}-{chain class name}-{datetime}`.
        project_metadata: Optional metadata to add to the project.
            Useful for storing information the test variant.
            (prompt version, model version, etc.)
        client: LangSmith client to use to access the dataset and to
            log feedback and run traces.
        verbose: Whether to print progress.
        revision_id: Optional revision identifier to assign this test run to
            track the performance of different versions of your system.
        **kwargs: Should not be used, but is provided for backwards compatibility.

    Returns:
        包含运行项目名称和生成模型输出的 `dict`。

    Examples:
    ```python
    from langsmith import Client
    from langchain_openai import ChatOpenAI
    from langchain_classic.chains import LLMChain
    from langchain_classic.smith import smith_eval.RunEvalConfig, run_on_dataset

    # Chains may have memory. Passing in a constructor function lets the
    # evaluation framework avoid cross-contamination between runs.
    def construct_chain():
        model = ChatOpenAI(temperature=0)
        chain = LLMChain.from_string(
            model,
            "What's the answer to {your_input_key}"
        )
        return chain

    # Load off-the-shelf evaluators via config or the EvaluatorType (string or enum)
    evaluation_config = smith_eval.RunEvalConfig(
        evaluators=[
            "qa",  # "Correctness" against a reference answer
            "embedding_distance",
            smith_eval.RunEvalConfig.Criteria("helpfulness"),
            smith_eval.RunEvalConfig.Criteria({
                "fifth-grader-score": "Do you have to be smarter than a fifth "
                "grader to answer this question?"
            }),
        ]
    )

    client = Client()
    await arun_on_dataset(
        client,
        dataset_name="<my_dataset_name>",
        llm_or_chain_factory=construct_chain,
        evaluation=evaluation_config,
    )
    ```
    您也可以通过继承 `StringEvaluator` 或 LangSmith 的 `RunEvaluator`
    类来创建自定义评估器。

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


    evaluation_config = smith_eval.RunEvalConfig(
        custom_evaluators=[MyStringEvaluator()],
    )

    await arun_on_dataset(
        client,
        dataset_name="<my_dataset_name>",
        llm_or_chain_factory=construct_chain,
        evaluation=evaluation_config,
    )
    ```
    """
    input_mapper = kwargs.pop("input_mapper", None)
    if input_mapper:
        warn_deprecated("0.0.305", message=_INPUT_MAPPER_DEP_WARNING, pending=True)
    if revision_id is None:
        revision_id = get_langchain_env_var_metadata().get("revision_id")
    tags = kwargs.pop("tags", None)
    if tags:
        warn_deprecated(
            "0.1.9",
            message="The tags argument is deprecated and will be"
            " removed in a future release. Please specify project_metadata instead.",
            pending=True,
        )

    if kwargs:
        warn_deprecated(
            "0.0.305",
            message="The following arguments are deprecated and "
            "will be removed in a future release: "
            f"{kwargs.keys()}.",
            removal="0.0.305",
        )
    client = client or Client()
    container = _DatasetRunContainer.prepare(
        client,
        dataset_name,
        llm_or_chain_factory,
        project_name,
        evaluation,
        tags,
        input_mapper,
        concurrency_level,
        project_metadata=project_metadata,
        revision_id=revision_id,
        dataset_version=dataset_version,
    )
    batch_results = await runnable_utils.gather_with_concurrency(
        container.configs[0].get("max_concurrency"),
        *map(
            functools.partial(
                _arun_llm_or_chain,
                llm_or_chain_factory=container.wrapped_model,
                input_mapper=input_mapper,
            ),
            container.examples,
            container.configs,
        ),
    )
    return container.finish(batch_results, verbose=verbose)


def run_on_dataset(
    client: Client | None,
    dataset_name: str,
    llm_or_chain_factory: MODEL_OR_CHAIN_FACTORY,
    *,
    evaluation: smith_eval.RunEvalConfig | None = None,
    dataset_version: datetime | str | None = None,
    concurrency_level: int = 5,
    project_name: str | None = None,
    project_metadata: dict[str, Any] | None = None,
    verbose: bool = False,
    revision_id: str | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """在数据集上运行。

    在数据集上运行链或语言模型，并将追踪信息存储到指定的项目名称中。

    For the (usually faster) async version of this function,
    see `arun_on_dataset`.

    Args:
        dataset_name: Name of the dataset to run the chain on.
        llm_or_chain_factory: Language model or Chain constructor to run
            over the dataset. The Chain constructor is used to permit
            independent calls on each example without carrying over state.
        evaluation: Configuration for evaluators to run on the
            results of the chain.
        dataset_version: Optional version of the dataset.
        concurrency_level: The number of async tasks to run concurrently.
        project_name: Name of the project to store the traces in.
            Defaults to `{dataset_name}-{chain class name}-{datetime}`.
        project_metadata: Optional metadata to add to the project.
            Useful for storing information the test variant.
            (prompt version, model version, etc.)
        client: LangSmith client to use to access the dataset and to
            log feedback and run traces.
        verbose: Whether to print progress.
        revision_id: Optional revision identifier to assign this test run to
            track the performance of different versions of your system.
        **kwargs: Should not be used, but is provided for backwards compatibility.

    Returns:
        `dict` 包含运行的项目名称和模型输出结果。

    Examples:
    ```python
    from langsmith import Client
    from langchain_openai import ChatOpenAI
    from langchain_classic.chains import LLMChain
    from langchain_classic.smith import smith_eval.RunEvalConfig, run_on_dataset

    # Chains may have memory. Passing in a constructor function lets the
    # evaluation framework avoid cross-contamination between runs.
    def construct_chain():
        model = ChatOpenAI(temperature=0)
        chain = LLMChain.from_string(
            model,
            "What's the answer to {your_input_key}"
        )
        return chain

    # Load off-the-shelf evaluators via config or the EvaluatorType (string or enum)
    evaluation_config = smith_eval.RunEvalConfig(
        evaluators=[
            "qa",  # "Correctness" against a reference answer
            "embedding_distance",
            smith_eval.RunEvalConfig.Criteria("helpfulness"),
            smith_eval.RunEvalConfig.Criteria({
                "fifth-grader-score": "Do you have to be smarter than a fifth "
                "grader to answer this question?"
            }),
        ]
    )

    client = Client()
    run_on_dataset(
        client,
        dataset_name="<my_dataset_name>",
        llm_or_chain_factory=construct_chain,
        evaluation=evaluation_config,
    )
    ```

    你也可以通过继承 `StringEvaluator` 或 LangSmith 的 `RunEvaluator`
    类来创建自定义评估器。

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


    evaluation_config = smith_eval.RunEvalConfig(
        custom_evaluators=[MyStringEvaluator()],
    )

    run_on_dataset(
        client,
        dataset_name="<my_dataset_name>",
        llm_or_chain_factory=construct_chain,
        evaluation=evaluation_config,
    )
    ```
    """
    input_mapper = kwargs.pop("input_mapper", None)
    if input_mapper:
        warn_deprecated("0.0.305", message=_INPUT_MAPPER_DEP_WARNING, pending=True)
    tags = kwargs.pop("tags", None)
    if tags:
        warn_deprecated(
            "0.1.9",
            message="The tags argument is deprecated and will be"
            " removed in a future release. Please specify project_metadata instead.",
            pending=True,
        )
    if revision_id is None:
        revision_id = get_langchain_env_var_metadata().get("revision_id")

    if kwargs:
        warn_deprecated(
            "0.0.305",
            message="The following arguments are deprecated and "
            "will be removed in a future release: "
            f"{kwargs.keys()}.",
            removal="0.0.305",
        )
    client = client or Client()
    container = _DatasetRunContainer.prepare(
        client,
        dataset_name,
        llm_or_chain_factory,
        project_name,
        evaluation,
        tags,
        input_mapper,
        concurrency_level,
        project_metadata=project_metadata,
        revision_id=revision_id,
        dataset_version=dataset_version,
    )
    if concurrency_level == 0:
        batch_results = [
            _run_llm_or_chain(
                example,
                config,
                llm_or_chain_factory=container.wrapped_model,
                input_mapper=input_mapper,
            )
            for example, config in zip(
                container.examples, container.configs, strict=False
            )
        ]
    else:
        with runnable_config.get_executor_for_config(container.configs[0]) as executor:
            batch_results = list(
                executor.map(
                    functools.partial(
                        _run_llm_or_chain,
                        llm_or_chain_factory=container.wrapped_model,
                        input_mapper=input_mapper,
                    ),
                    container.examples,
                    container.configs,
                ),
            )

    return container.finish(batch_results, verbose=verbose)

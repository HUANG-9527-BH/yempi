"""运行字符串评估器包装器。"""

from __future__ import annotations

import logging
import uuid
from abc import abstractmethod
from typing import Any

from langchain_core.callbacks.manager import (
    AsyncCallbackManagerForChainRun,
    CallbackManagerForChainRun,
)
from langchain_core.load.dump import dumpd
from langchain_core.load.load import load
from langchain_core.load.serializable import Serializable
from langchain_core.messages import BaseMessage, get_buffer_string, messages_from_dict
from langsmith import EvaluationResult, RunEvaluator
from langsmith.schemas import DataType, Example, Run
from typing_extensions import override

from langchain_classic.chains.base import Chain
from langchain_classic.evaluation.schema import StringEvaluator
from langchain_classic.schema import RUN_KEY

_logger = logging.getLogger(__name__)


def _get_messages_from_run_dict(messages: list[dict]) -> list[BaseMessage]:
    if not messages:
        return []
    first_message = messages[0]
    if "lc" in first_message:
        return [
            load(dumpd(message), allowed_objects="messages") for message in messages
        ]
    return messages_from_dict(messages)


class StringRunMapper(Serializable):
    """从运行对象中提取待评估项。"""

    @property
    def output_keys(self) -> list[str]:
        """从运行中提取的键。"""
        return ["prediction", "input"]

    @abstractmethod
    def map(self, run: Run) -> dict[str, str]:
        """将运行映射为字典。"""

    def __call__(self, run: Run) -> dict[str, str]:
        """将运行映射为字典。"""
        if not run.outputs:
            msg = f"Run {run.id} has no outputs to evaluate."
            raise ValueError(msg)
        return self.map(run)


class LLMStringRunMapper(StringRunMapper):
    """从运行对象中提取待评估项。"""

    def serialize_chat_messages(self, messages: list[dict] | list[list[dict]]) -> str:
        """从运行中提取输入消息。"""
        if isinstance(messages, list) and messages:
            if isinstance(messages[0], dict):
                chat_messages = _get_messages_from_run_dict(messages)
            elif isinstance(messages[0], list):
                                # Tracer 的运行中消息为嵌套的字典列表形式
                chat_messages = _get_messages_from_run_dict(messages[0])
            else:
                msg = f"Could not extract messages to evaluate {messages}"  # type: ignore[unreachable]
                raise ValueError(msg)
            return get_buffer_string(chat_messages)
        msg = f"Could not extract messages to evaluate {messages}"
        raise ValueError(msg)

    def serialize_inputs(self, inputs: dict) -> str:
        """序列化输入。

        Args:
            inputs: 运行中的输入，预期包含提示词或消息。

        Returns:
            从提示词或消息序列化得到的输入文本。

        Raises:
            ValueError: 如果输入中既未找到提示词也未找到消息。
        """
        if "prompts" in inputs:  # 我们是否应该接受这种格式？
            input_ = "\n\n".join(inputs["prompts"])
        elif "prompt" in inputs:
            input_ = inputs["prompt"]
        elif "messages" in inputs:
            input_ = self.serialize_chat_messages(inputs["messages"])
        else:
            msg = "LLM Run must have either messages or prompts as inputs."
            raise ValueError(msg)
        return input_

    def serialize_outputs(self, outputs: dict) -> str:
        """序列化输出。

        Args:
            outputs: 运行中的输出，预期包含生成内容。

        Returns:
            从第一个生成内容序列化得到的输出文本。

        Raises:
            ValueError: If no generations are found in the outputs or if the generations
                are empty.
        """
        if not outputs.get("generations"):
            msg = "Cannot evaluate LLM Run without generations."
            raise ValueError(msg)
        generations: list[dict] | list[list[dict]] = outputs["generations"]
        if not generations:
            msg = "Cannot evaluate LLM run with empty generations."
            raise ValueError(msg)
        first_generation: dict | list[dict] = generations[0]
        if isinstance(first_generation, list):
                        # Tracer 生成的运行结果中，生成内容为嵌套的字典列表（列表的列表）
                        # 而 API 生成的运行结果中，生成内容为字典列表
            first_generation = first_generation[0]
        if "message" in first_generation:
            output_ = self.serialize_chat_messages([first_generation["message"]])
        else:
            output_ = first_generation["text"]
        return output_

    def map(self, run: Run) -> dict[str, str]:
        """将运行映射为字典。"""
        if run.run_type != "llm":
            msg = "LLM RunMapper only supports LLM runs."
            raise ValueError(msg)
        if not run.outputs:
            if run.error:
                msg = f"Cannot evaluate errored LLM run {run.id}: {run.error}"
                raise ValueError(msg)
            msg = f"Run {run.id} has no outputs. Cannot evaluate this run."
            raise ValueError(msg)
        try:
            inputs = self.serialize_inputs(run.inputs)
        except Exception as e:
            msg = f"Could not parse LM input from run inputs {run.inputs}"
            raise ValueError(msg) from e
        try:
            output_ = self.serialize_outputs(run.outputs)
        except Exception as e:
            msg = f"Could not parse LM prediction from run outputs {run.outputs}"
            raise ValueError(msg) from e
        return {"input": inputs, "prediction": output_}


class ChainStringRunMapper(StringRunMapper):
    """从链的运行对象中提取待评估项。"""

    input_key: str | None = None
    """The key from the model Run's inputs to use as the eval input.
    If not provided, will use the only input key or raise an
    error if there are multiple."""
    prediction_key: str | None = None
    """The key from the model Run's outputs to use as the eval prediction.
    If not provided, will use the only output key or raise an error
    if there are multiple."""

    def _get_key(self, source: dict, key: str | None, which: str) -> str:
        if key is not None:
            return source[key]
        if len(source) == 1:
            return next(iter(source.values()))
        msg = (
            f"Could not map run {which} with multiple keys: "
            f"{source}\nPlease manually specify a {which}_key"
        )
        raise ValueError(msg)

    def map(self, run: Run) -> dict[str, str]:
        """将运行映射为字典。"""
        if not run.outputs:
            msg = (
                f"Run with ID {run.id} lacks outputs required for evaluation."
                " Ensure the Run has valid outputs."
            )
            raise ValueError(msg)
        if self.input_key is not None and self.input_key not in run.inputs:
            msg = (
                f"Run with ID {run.id} is missing the expected input key"
                f" '{self.input_key}'.\nAvailable input keys in this Run"
                f"  are: {run.inputs.keys()}.\nAdjust the evaluator's"
                f" input_key or ensure your input data includes key"
                f" '{self.input_key}'."
            )
            raise ValueError(msg)
        if self.prediction_key is not None and self.prediction_key not in run.outputs:
            available_keys = ", ".join(run.outputs.keys())
            msg = (
                f"Run with ID {run.id} doesn't have the expected prediction key"
                f" '{self.prediction_key}'. Available prediction keys in this Run are:"
                f" {available_keys}. Adjust the evaluator's prediction_key or"
                " ensure the Run object's outputs the expected key."
            )
            raise ValueError(msg)

        input_ = self._get_key(run.inputs, self.input_key, "input")
        prediction = self._get_key(run.outputs, self.prediction_key, "prediction")
        return {
            "input": input_,
            "prediction": prediction,
        }


class ToolStringRunMapper(StringRunMapper):
    """将输入映射到工具。"""

    @override
    def map(self, run: Run) -> dict[str, str]:
        if not run.outputs:
            msg = f"Run {run.id} has no outputs to evaluate."
            raise ValueError(msg)
        return {"input": run.inputs["input"], "prediction": run.outputs["output"]}


class StringExampleMapper(Serializable):
    """将示例或数据集中的行映射到评估的输入。"""

    reference_key: str | None = None

    @property
    def output_keys(self) -> list[str]:
        """从运行中提取的键。"""
        return ["reference"]

    def serialize_chat_messages(self, messages: list[dict]) -> str:
        """从运行中提取输入消息。"""
        chat_messages = _get_messages_from_run_dict(messages)
        return get_buffer_string(chat_messages)

    def map(self, example: Example) -> dict[str, str]:
        """将示例或数据集行映射为字典。"""
        if not example.outputs:
            msg = f"Example {example.id} has no outputs to use as a reference."
            raise ValueError(msg)
        if self.reference_key is None:
            if len(example.outputs) > 1:
                msg = (
                    f"Example {example.id} has multiple outputs, so you must"
                    " specify a reference_key."
                )
                raise ValueError(msg)
            output = next(iter(example.outputs.values()))
        elif self.reference_key not in example.outputs:
            msg = (
                f"Example {example.id} does not have reference key"
                f" {self.reference_key}."
            )
            raise ValueError(msg)
        else:
            output = example.outputs[self.reference_key]
        return {
            "reference": self.serialize_chat_messages([output])
            if isinstance(output, dict) and output.get("type") and output.get("data")
            else output,
        }

    def __call__(self, example: Example) -> dict[str, str]:
        """将运行和示例映射为字典。"""
        if not example.outputs:
            msg = f"Example {example.id} has no outputs to use as areference label."
            raise ValueError(msg)
        return self.map(example)


class StringRunEvaluatorChain(Chain, RunEvaluator):
    """评估运行及可选的示例。"""

    run_mapper: StringRunMapper
    """将运行映射为包含 'input' 和 'prediction' 字符串的字典。"""
    example_mapper: StringExampleMapper | None = None
    """Maps the Example (dataset row) to a dictionary
    with a 'reference' string."""
    name: str
    """评估指标的名称。"""
    string_evaluator: StringEvaluator
    """评估链。"""

    @property
    @override
    def input_keys(self) -> list[str]:
        return ["run", "example"]

    @property
    @override
    def output_keys(self) -> list[str]:
        return ["feedback"]

    def _prepare_input(self, inputs: dict[str, Any]) -> dict[str, str]:
        run: Run = inputs["run"]
        example: Example | None = inputs.get("example")
        evaluate_strings_inputs = self.run_mapper(run)
        if not self.string_evaluator.requires_input:
                        # 隐藏关于未使用输入的警告
            evaluate_strings_inputs.pop("input", None)
        if example and self.example_mapper and self.string_evaluator.requires_reference:
            evaluate_strings_inputs.update(self.example_mapper(example))
        elif self.string_evaluator.requires_reference:
            msg = (
                f"Evaluator {self.name} requires an reference"
                " example from the dataset,"
                f" but none was provided for run {run.id}."
            )
            raise ValueError(msg)
        return evaluate_strings_inputs

    def _prepare_output(self, output: dict[str, Any]) -> dict[str, Any]:
        evaluation_result = EvaluationResult(
            key=self.name,
            comment=output.get("reasoning"),
            **output,
        )
        if RUN_KEY in output:
                        # TODO: 目前尚未暴露。待更新
            evaluation_result.evaluator_info[RUN_KEY] = output[RUN_KEY]
        return {"feedback": evaluation_result}

    def _call(
        self,
        inputs: dict[str, str],
        run_manager: CallbackManagerForChainRun | None = None,
    ) -> dict[str, Any]:
        """调用评估链。"""
        evaluate_strings_inputs = self._prepare_input(inputs)
        _run_manager = run_manager or CallbackManagerForChainRun.get_noop_manager()
        callbacks = _run_manager.get_child()
        chain_output = self.string_evaluator.evaluate_strings(
            **evaluate_strings_inputs,
            callbacks=callbacks,
            include_run_info=True,
        )
        return self._prepare_output(chain_output)

    async def _acall(
        self,
        inputs: dict[str, str],
        run_manager: AsyncCallbackManagerForChainRun | None = None,
    ) -> dict[str, Any]:
        """调用评估链。"""
        evaluate_strings_inputs = self._prepare_input(inputs)
        _run_manager = run_manager or AsyncCallbackManagerForChainRun.get_noop_manager()
        callbacks = _run_manager.get_child()
        chain_output = await self.string_evaluator.aevaluate_strings(
            **evaluate_strings_inputs,
            callbacks=callbacks,
            include_run_info=True,
        )
        return self._prepare_output(chain_output)

    def _prepare_evaluator_output(self, output: dict[str, Any]) -> EvaluationResult:
        feedback: EvaluationResult = output["feedback"]
        if RUN_KEY not in feedback.evaluator_info:
            feedback.evaluator_info[RUN_KEY] = output[RUN_KEY]
        return feedback

    @override
    def evaluate_run(
        self,
        run: Run,
        example: Example | None = None,
        evaluator_run_id: uuid.UUID | None = None,
    ) -> EvaluationResult:
        """评估一个示例。"""
        try:
            result = self({"run": run, "example": example}, include_run_info=True)
            return self._prepare_evaluator_output(result)
        except Exception as e:
            _logger.exception("Error evaluating run %s", run.id)
            return EvaluationResult(
                key=self.string_evaluator.evaluation_name,
                comment=f"Error evaluating run {run.id}: {e}",
                                # TODO: 一旦我们可以通过回调声明运行 ID，即添加运行 ID
            )

    @override
    async def aevaluate_run(
        self,
        run: Run,
        example: Example | None = None,
        evaluator_run_id: uuid.UUID | None = None,
    ) -> EvaluationResult:
        """评估一个示例。"""
        try:
            result = await self.acall(
                {"run": run, "example": example},
                include_run_info=True,
            )
            return self._prepare_evaluator_output(result)
        except Exception as e:
            _logger.exception("Error evaluating run %s", run.id)
            return EvaluationResult(
                key=self.string_evaluator.evaluation_name,
                comment=f"Error evaluating run {run.id}: {e}",
            )

    @classmethod
    def from_run_and_data_type(
        cls,
        evaluator: StringEvaluator,
        run_type: str,
        data_type: DataType,
        input_key: str | None = None,
        prediction_key: str | None = None,
        reference_key: str | None = None,
        tags: list[str] | None = None,
    ) -> StringRunEvaluatorChain:
        """创建一个 StringRunEvaluatorChain。

        Create a StringRunEvaluatorChain from an evaluator and the run and dataset
        types.

        此方法提供了一种简便方式来实例化 StringRunEvaluatorChain，
        通过传入评估器以及关于运行类型和数据类型的信息。
        该方法支持 LLM 和链的运行。

        Args:
            evaluator: 要使用的字符串评估器。
            run_type: 被评估的运行类型。
                支持的类型包括 LLM 和 Chain。
            data_type: 运行中使用的数据集类型。
            input_key: 用于从运行中映射输入的键。
            prediction_key: 用于从运行中映射预测的键。
            reference_key: 用于从数据集中映射参考值的键。
            tags: 要附加到评估链的标签列表。

        Returns:
            实例化的评估链。

        Raises:
            ValueError: If the run type is not supported, or if the evaluator requires a
                reference from the dataset but the reference key is not provided.

        """
                # 配置如何将运行输入/预测传递给评估器
        if run_type == "llm":
            run_mapper: StringRunMapper = LLMStringRunMapper()
        elif run_type == "chain":
            run_mapper = ChainStringRunMapper(
                input_key=input_key,
                prediction_key=prediction_key,
            )
        else:
            msg = f"Unsupported run type {run_type}. Expected one of 'llm' or 'chain'."
            raise ValueError(msg)

                # 配置如何将示例行作为参考字符串输入给评估器
        if (
            reference_key is not None
            or data_type in (DataType.llm, DataType.chat)
            or evaluator.requires_reference
        ):
            example_mapper = StringExampleMapper(reference_key=reference_key)
        elif evaluator.requires_reference:
            msg = (  # type: ignore[unreachable]
                f"Evaluator {evaluator.evaluation_name} requires a reference"
                " example from the dataset. Please specify the reference key from"
                " amongst the dataset outputs keys."
            )
            raise ValueError(msg)
        else:
            example_mapper = None
        return cls(
            name=evaluator.evaluation_name,
            run_mapper=run_mapper,
            example_mapper=example_mapper,
            string_evaluator=evaluator,
            tags=tags,
        )

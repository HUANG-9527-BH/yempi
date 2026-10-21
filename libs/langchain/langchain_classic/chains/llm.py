"""仅格式化提示词并调用 LLM 的链。"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import Any, cast

from langchain_core._api import deprecated
from langchain_core.callbacks import (
    AsyncCallbackManager,
    AsyncCallbackManagerForChainRun,
    CallbackManager,
    CallbackManagerForChainRun,
    Callbacks,
)
from langchain_core.language_models import (
    BaseLanguageModel,
    LanguageModelInput,
)
from langchain_core.messages import BaseMessage
from langchain_core.output_parsers import BaseLLMOutputParser, StrOutputParser
from langchain_core.outputs import ChatGeneration, Generation, LLMResult
from langchain_core.prompt_values import PromptValue
from langchain_core.prompts import BasePromptTemplate, PromptTemplate
from langchain_core.runnables import (
    Runnable,
    RunnableBinding,
    RunnableBranch,
    RunnableWithFallbacks,
)
from langchain_core.runnables.configurable import DynamicRunnable
from langchain_core.utils.input import get_colored_text
from pydantic import ConfigDict, Field
from typing_extensions import override

from langchain_classic.chains.base import Chain


@deprecated(
    since="0.1.17",
    alternative="RunnableSequence, e.g., `prompt | llm`",
    removal="2.0.0",
)
class LLMChain(Chain):
    """用于向 LLM 发起查询的链。

    This class is deprecated. See below for an example implementation using
    LangChain runnables:

        ```python
        from langchain_core.output_parsers import StrOutputParser
        from langchain_core.prompts import PromptTemplate
        from langchain_openai import OpenAI

        prompt_template = "Tell me a {adjective} joke"
        prompt = PromptTemplate(input_variables=["adjective"], template=prompt_template)
        model = OpenAI()
        chain = prompt | model | StrOutputParser()

        chain.invoke("your adjective here")
        ```

    Example:
        ```python
        from langchain_classic.chains import LLMChain
        from langchain_openai import OpenAI
        from langchain_core.prompts import PromptTemplate

        prompt_template = "Tell me a {adjective} joke"
        prompt = PromptTemplate(input_variables=["adjective"], template=prompt_template)
        model = LLMChain(llm=OpenAI(), prompt=prompt)
        ```
    """

    @classmethod
    @override
    def is_lc_serializable(cls) -> bool:
        return True

    prompt: BasePromptTemplate
    """要使用的提示词对象。"""
    llm: Runnable[LanguageModelInput, str] | Runnable[LanguageModelInput, BaseMessage]
    """要调用的语言模型。"""
    output_key: str = "text"
    output_parser: BaseLLMOutputParser = Field(default_factory=StrOutputParser)
    """要使用的输出解析器。 默认为取最可能字符串但不作其他更改的解析器。"""
    return_final_only: bool = True
    """是否仅返回最终解析结果。
    若为 `False`，将返回关于生成过程的额外信息。"""
    llm_kwargs: dict = Field(default_factory=dict)

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="forbid",
    )

    @property
    def input_keys(self) -> list[str]:
        """将是提示词所期望的任意键。"""
        return self.prompt.input_variables

    @property
    def output_keys(self) -> list[str]:
        """始终返回 text 键。"""
        if self.return_final_only:
            return [self.output_key]
        return [self.output_key, "full_generation"]

    def _call(
        self,
        inputs: dict[str, Any],
        run_manager: CallbackManagerForChainRun | None = None,
    ) -> dict[str, str]:
        response = self.generate([inputs], run_manager=run_manager)
        return self.create_outputs(response)[0]

    def generate(
        self,
        input_list: list[dict[str, Any]],
        run_manager: CallbackManagerForChainRun | None = None,
    ) -> LLMResult:
        """从输入生成 LLM 结果。"""
        prompts, stop = self.prep_prompts(input_list, run_manager=run_manager)
        callbacks = run_manager.get_child() if run_manager else None
        if isinstance(self.llm, BaseLanguageModel):
            return self.llm.generate_prompt(
                prompts,
                stop,
                callbacks=callbacks,
                **self.llm_kwargs,
            )
        results = self.llm.bind(stop=stop, **self.llm_kwargs).batch(
            cast("list", prompts),
            {"callbacks": callbacks},
        )
        generations: list[list[Generation]] = []
        for res in results:
            if isinstance(res, BaseMessage):
                generations.append([ChatGeneration(message=res)])
            else:
                generations.append([Generation(text=res)])
        return LLMResult(generations=generations)

    async def agenerate(
        self,
        input_list: list[dict[str, Any]],
        run_manager: AsyncCallbackManagerForChainRun | None = None,
    ) -> LLMResult:
        """从输入生成 LLM 结果。"""
        prompts, stop = await self.aprep_prompts(input_list, run_manager=run_manager)
        callbacks = run_manager.get_child() if run_manager else None
        if isinstance(self.llm, BaseLanguageModel):
            return await self.llm.agenerate_prompt(
                prompts,
                stop,
                callbacks=callbacks,
                **self.llm_kwargs,
            )
        results = await self.llm.bind(stop=stop, **self.llm_kwargs).abatch(
            cast("list", prompts),
            {"callbacks": callbacks},
        )
        generations: list[list[Generation]] = []
        for res in results:
            if isinstance(res, BaseMessage):
                generations.append([ChatGeneration(message=res)])
            else:
                generations.append([Generation(text=res)])
        return LLMResult(generations=generations)

    def prep_prompts(
        self,
        input_list: list[dict[str, Any]],
        run_manager: CallbackManagerForChainRun | None = None,
    ) -> tuple[list[PromptValue], list[str] | None]:
        """从输入准备提示词。"""
        stop = None
        if len(input_list) == 0:
            return [], stop
        if "stop" in input_list[0]:
            stop = input_list[0]["stop"]
        prompts = []
        for inputs in input_list:
            selected_inputs = {k: inputs[k] for k in self.prompt.input_variables}
            prompt = self.prompt.format_prompt(**selected_inputs)
            _colored_text = get_colored_text(prompt.to_string(), "green")
            _text = "Prompt after formatting:\n" + _colored_text
            if run_manager:
                run_manager.on_text(_text, end="\n", verbose=self.verbose)
            if "stop" in inputs and inputs["stop"] != stop:
                msg = "If `stop` is present in any inputs, should be present in all."
                raise ValueError(msg)
            prompts.append(prompt)
        return prompts, stop

    async def aprep_prompts(
        self,
        input_list: list[dict[str, Any]],
        run_manager: AsyncCallbackManagerForChainRun | None = None,
    ) -> tuple[list[PromptValue], list[str] | None]:
        """从输入准备提示词。"""
        stop = None
        if len(input_list) == 0:
            return [], stop
        if "stop" in input_list[0]:
            stop = input_list[0]["stop"]
        prompts = []
        for inputs in input_list:
            selected_inputs = {k: inputs[k] for k in self.prompt.input_variables}
            prompt = self.prompt.format_prompt(**selected_inputs)
            _colored_text = get_colored_text(prompt.to_string(), "green")
            _text = "Prompt after formatting:\n" + _colored_text
            if run_manager:
                await run_manager.on_text(_text, end="\n", verbose=self.verbose)
            if "stop" in inputs and inputs["stop"] != stop:
                msg = "If `stop` is present in any inputs, should be present in all."
                raise ValueError(msg)
            prompts.append(prompt)
        return prompts, stop

    def apply(
        self,
        input_list: list[dict[str, Any]],
        callbacks: Callbacks = None,
    ) -> list[dict[str, str]]:
        """利用 LLM 的 generate 方法以获得性能提升。"""
        callback_manager = CallbackManager.configure(
            callbacks,
            self.callbacks,
            self.verbose,
        )
        run_manager = callback_manager.on_chain_start(
            None,
            {"input_list": input_list},
            name=self.get_name(),
        )
        try:
            response = self.generate(input_list, run_manager=run_manager)
        except BaseException as e:
            run_manager.on_chain_error(e)
            raise
        outputs = self.create_outputs(response)
        run_manager.on_chain_end({"outputs": outputs})
        return outputs

    async def aapply(
        self,
        input_list: list[dict[str, Any]],
        callbacks: Callbacks = None,
    ) -> list[dict[str, str]]:
        """利用 LLM 的 generate 方法以获得性能提升。"""
        callback_manager = AsyncCallbackManager.configure(
            callbacks,
            self.callbacks,
            self.verbose,
        )
        run_manager = await callback_manager.on_chain_start(
            None,
            {"input_list": input_list},
            name=self.get_name(),
        )
        try:
            response = await self.agenerate(input_list, run_manager=run_manager)
        except BaseException as e:
            await run_manager.on_chain_error(e)
            raise
        outputs = self.create_outputs(response)
        await run_manager.on_chain_end({"outputs": outputs})
        return outputs

    @property
    def _run_output_key(self) -> str:
        return self.output_key

    def create_outputs(self, llm_result: LLMResult) -> list[dict[str, Any]]:
        """从响应创建输出。"""
        result = [
                        # 获取生成字符串中的文本内容。
            {
                self.output_key: self.output_parser.parse_result(generation),
                "full_generation": generation,
            }
            for generation in llm_result.generations
        ]
        if self.return_final_only:
            result = [{self.output_key: r[self.output_key]} for r in result]
        return result

    async def _acall(
        self,
        inputs: dict[str, Any],
        run_manager: AsyncCallbackManagerForChainRun | None = None,
    ) -> dict[str, str]:
        response = await self.agenerate([inputs], run_manager=run_manager)
        return self.create_outputs(response)[0]

    def predict(self, callbacks: Callbacks = None, **kwargs: Any) -> str:
        """使用 kwargs 格式化提示词并传递给 LLM。

        Args:
            callbacks: 传递给 LLMChain 的回调
            **kwargs: 传递给提示词模板的键。

        Returns:
            LLM 的完成结果。

        Example:
            ```python
            completion = llm.predict(adjective="funny")
            ```
        """
        return self(kwargs, callbacks=callbacks)[self.output_key]

    async def apredict(self, callbacks: Callbacks = None, **kwargs: Any) -> str:
        """使用 kwargs 格式化提示词并传递给 LLM。

        Args:
            callbacks: 传递给 LLMChain 的回调
            **kwargs: 传递给提示词模板的键。

        Returns:
            LLM 的完成结果。

        Example:
            ```python
            completion = llm.predict(adjective="funny")
            ```
        """
        return (await self.acall(kwargs, callbacks=callbacks))[self.output_key]

    def predict_and_parse(
        self,
        callbacks: Callbacks = None,
        **kwargs: Any,
    ) -> str | list[str] | dict[str, Any]:
        """调用 predict 然后解析结果。"""
        warnings.warn(
            "The predict_and_parse method is deprecated, "
            "instead pass an output parser directly to LLMChain.",
            stacklevel=2,
        )
        result = self.predict(callbacks=callbacks, **kwargs)
        if self.prompt.output_parser is not None:
            return self.prompt.output_parser.parse(result)
        return result

    async def apredict_and_parse(
        self,
        callbacks: Callbacks = None,
        **kwargs: Any,
    ) -> str | list[str] | dict[str, str]:
        """调用 apredict 然后解析结果。"""
        warnings.warn(
            "The apredict_and_parse method is deprecated, "
            "instead pass an output parser directly to LLMChain.",
            stacklevel=2,
        )
        result = await self.apredict(callbacks=callbacks, **kwargs)
        if self.prompt.output_parser is not None:
            return self.prompt.output_parser.parse(result)
        return result

    def apply_and_parse(
        self,
        input_list: list[dict[str, Any]],
        callbacks: Callbacks = None,
    ) -> Sequence[str | list[str] | dict[str, str]]:
        """调用 apply 然后解析结果。"""
        warnings.warn(
            "The apply_and_parse method is deprecated, "
            "instead pass an output parser directly to LLMChain.",
            stacklevel=2,
        )
        result = self.apply(input_list, callbacks=callbacks)
        return self._parse_generation(result)

    def _parse_generation(
        self,
        generation: list[dict[str, str]],
    ) -> Sequence[str | list[str] | dict[str, str]]:
        if self.prompt.output_parser is not None:
            return [
                self.prompt.output_parser.parse(res[self.output_key])
                for res in generation
            ]
        return generation

    async def aapply_and_parse(
        self,
        input_list: list[dict[str, Any]],
        callbacks: Callbacks = None,
    ) -> Sequence[str | list[str] | dict[str, str]]:
        """调用 apply 然后解析结果。"""
        warnings.warn(
            "The aapply_and_parse method is deprecated, "
            "instead pass an output parser directly to LLMChain.",
            stacklevel=2,
        )
        result = await self.aapply(input_list, callbacks=callbacks)
        return self._parse_generation(result)

    @property
    def _chain_type(self) -> str:
        return "llm_chain"

    @classmethod
    def from_string(cls, llm: BaseLanguageModel, template: str) -> LLMChain:
        """从 LLM 和模板创建 LLMChain。"""
        prompt_template = PromptTemplate.from_template(template)
        return cls(llm=llm, prompt=prompt_template)

    def _get_num_tokens(self, text: str) -> int:
        return _get_language_model(self.llm).get_num_tokens(text)


def _get_language_model(llm_like: Runnable) -> BaseLanguageModel:
    if isinstance(llm_like, BaseLanguageModel):
        return llm_like
    if isinstance(llm_like, RunnableBinding):
        return _get_language_model(llm_like.bound)
    if isinstance(llm_like, RunnableWithFallbacks):
        return _get_language_model(llm_like.runnable)
    if isinstance(llm_like, (RunnableBranch, DynamicRunnable)):
        return _get_language_model(llm_like.default)
    msg = (
        f"Unable to extract BaseLanguageModel from llm_like object of type "
        f"{type(llm_like)}"
    )
    raise ValueError(msg)

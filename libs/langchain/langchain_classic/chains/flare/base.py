from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from typing import Any

from langchain_core.callbacks import (
    CallbackManagerForChainRun,
)
from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import BasePromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import Runnable
from pydantic import Field
from typing_extensions import override

from langchain_classic.chains.base import Chain
from langchain_classic.chains.flare.prompts import (
    PROMPT,
    QUESTION_GENERATOR_PROMPT,
    FinishedOutputParser,
)
from langchain_classic.chains.llm import LLMChain

logger = logging.getLogger(__name__)


def _extract_tokens_and_log_probs(response: AIMessage) -> tuple[list[str], list[float]]:
    """从聊天模型响应中提取 token 和日志概率。"""
    tokens = []
    log_probs = []
    for token in response.response_metadata["logprobs"]["content"]:
        tokens.append(token["token"])
        log_probs.append(token["logprob"])
    return tokens, log_probs


class QuestionGeneratorChain(LLMChain):
    """从不确定片段生成问题的链。"""

    prompt: BasePromptTemplate = QUESTION_GENERATOR_PROMPT
    """链的提示词模板。"""

    @classmethod
    @override
    def is_lc_serializable(cls) -> bool:
        return False

    @property
    def input_keys(self) -> list[str]:
        """链的输入键。"""
        return ["user_input", "context", "response"]


def _low_confidence_spans(
    tokens: Sequence[str],
    log_probs: Sequence[float],
    min_prob: float,
    min_token_gap: int,
    num_pad_tokens: int,
) -> list[str]:
    try:
        import numpy as np

        _low_idx = np.where(np.exp(log_probs) < min_prob)[0]
    except ImportError:
        logger.warning(
            "NumPy not found in the current Python environment. FlareChain will use a "
            "pure Python implementation for internal calculations, which may "
            "significantly impact performance, especially for large datasets. For "
            "optimal speed and efficiency, consider installing NumPy: pip install "
            "numpy",
        )
        import math

        _low_idx = [  # type: ignore[assignment]
            idx
            for idx, log_prob in enumerate(log_probs)
            if math.exp(log_prob) < min_prob
        ]
    low_idx = [i for i in _low_idx if re.search(r"\w", tokens[i])]
    if len(low_idx) == 0:
        return []
    spans = [[low_idx[0], low_idx[0] + num_pad_tokens + 1]]
    for i, idx in enumerate(low_idx[1:]):
        end = idx + num_pad_tokens + 1
        if idx - low_idx[i] < min_token_gap:
            spans[-1][1] = end
        else:
            spans.append([idx, end])
    return ["".join(tokens[start:end]) for start, end in spans]


class FlareChain(Chain):
    """Flare 链。

    结合检索器、问题生成器和响应生成器的链。

    参见 [主动检索增强生成](https://arxiv.org/abs/2305.06983) 论文。
    """

    question_generator_chain: Runnable
    """从不确定片段生成问题的链。"""
    response_chain: Runnable
    """根据用户输入和上下文生成响应的链。"""
    output_parser: FinishedOutputParser = Field(default_factory=FinishedOutputParser)
    """解析器，用于判断链是否已完成。"""
    retriever: BaseRetriever
    """从用户输入中检索相关文档的检索器。"""
    min_prob: float = 0.2
    """被视为低置信度的 token 的最小概率。"""
    min_token_gap: int = 5
    """两个低置信度片段之间的最小 token 数量。"""
    num_pad_tokens: int = 2
    """在低置信度片段周围填充的 token 数量。"""
    max_iter: int = 10
    """最大迭代次数。"""
    start_with_retrieval: bool = True
    """是否从检索开始。"""

    @property
    def input_keys(self) -> list[str]:
        """链的输入键。"""
        return ["user_input"]

    @property
    def output_keys(self) -> list[str]:
        """链的输出键。"""
        return ["response"]

    def _do_generation(
        self,
        questions: list[str],
        user_input: str,
        response: str,
        _run_manager: CallbackManagerForChainRun,
    ) -> tuple[str, bool]:
        callbacks = _run_manager.get_child()
        docs = []
        for question in questions:
            docs.extend(self.retriever.invoke(question))
        context = "\n\n".join(d.page_content for d in docs)
        result = self.response_chain.invoke(
            {
                "user_input": user_input,
                "context": context,
                "response": response,
            },
            {"callbacks": callbacks},
        )
        if isinstance(result, AIMessage):
            result = result.content
        marginal, finished = self.output_parser.parse(result)
        return marginal, finished

    def _do_retrieval(
        self,
        low_confidence_spans: list[str],
        _run_manager: CallbackManagerForChainRun,
        user_input: str,
        response: str,
        initial_response: str,
    ) -> tuple[str, bool]:
        question_gen_inputs = [
            {
                "user_input": user_input,
                "current_response": initial_response,
                "uncertain_span": span,
            }
            for span in low_confidence_spans
        ]
        callbacks = _run_manager.get_child()
        if isinstance(self.question_generator_chain, LLMChain):
            question_gen_outputs = self.question_generator_chain.apply(
                question_gen_inputs,
                callbacks=callbacks,
            )
            questions = [
                output[self.question_generator_chain.output_keys[0]]
                for output in question_gen_outputs
            ]
        else:
            questions = self.question_generator_chain.batch(
                question_gen_inputs,
                config={"callbacks": callbacks},
            )
        _run_manager.on_text(
            f"Generated Questions: {questions}",
            color="yellow",
            end="\n",
        )
        return self._do_generation(questions, user_input, response, _run_manager)

    def _call(
        self,
        inputs: dict[str, Any],
        run_manager: CallbackManagerForChainRun | None = None,
    ) -> dict[str, Any]:
        _run_manager = run_manager or CallbackManagerForChainRun.get_noop_manager()

        user_input = inputs[self.input_keys[0]]

        response = ""

        for _i in range(self.max_iter):
            _run_manager.on_text(
                f"Current Response: {response}",
                color="blue",
                end="\n",
            )
            _input = {"user_input": user_input, "context": "", "response": response}
            tokens, log_probs = _extract_tokens_and_log_probs(
                self.response_chain.invoke(
                    _input,
                    {"callbacks": _run_manager.get_child()},
                ),
            )
            low_confidence_spans = _low_confidence_spans(
                tokens,
                log_probs,
                self.min_prob,
                self.min_token_gap,
                self.num_pad_tokens,
            )
            initial_response = response.strip() + " " + "".join(tokens)
            if not low_confidence_spans:
                response = initial_response
                final_response, finished = self.output_parser.parse(response)
                if finished:
                    return {self.output_keys[0]: final_response}
                continue

            marginal, finished = self._do_retrieval(
                low_confidence_spans,
                _run_manager,
                user_input,
                response,
                initial_response,
            )
            response = response.strip() + " " + marginal
            if finished:
                break
        return {self.output_keys[0]: response}

    @classmethod
    def from_llm(
        cls,
        llm: BaseLanguageModel | None,
        max_generation_len: int = 32,
        **kwargs: Any,
    ) -> FlareChain:
        """从语言模型创建 FlareChain。

        Args:
            llm: 使用的语言模型。
            max_generation_len: 生成响应的最大长度。
            kwargs: 传递给构造函数的额外参数。

        Returns:
            使用给定语言模型的 FlareChain 类。
        """
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as e:
            msg = (
                "OpenAI is required for FlareChain. "
                "Please install langchain-openai."
                "pip install langchain-openai"
            )
            raise ImportError(msg) from e
                # 保留提供的 llm，而不是始终创建新的 ChatOpenAI。
                # 强制要求使用 ChatOpenAI（FLARE 需要 token 日志概率）。
        if llm is None:
            llm = ChatOpenAI(
                max_completion_tokens=max_generation_len,
                logprobs=True,
                temperature=0,
            )
        else:
            if not isinstance(llm, ChatOpenAI):
                msg = (
                    f"FlareChain.from_llm requires ChatOpenAI; got "
                    f"{type(llm).__name__}."
                )
                raise TypeError(msg)
            if not getattr(llm, "logprobs", False):  # 属性存在性可能不同
                msg = (
                    "Provided ChatOpenAI instance must be constructed with "
                    "logprobs=True for FlareChain."
                )
                raise ValueError(msg)
            current_max = getattr(llm, "max_completion_tokens", None)
            if current_max is not None and current_max != max_generation_len:
                logger.debug(
                    "FlareChain.from_llm: supplied llm max_completion_tokens=%s "
                    "differs from requested max_generation_len=%s; "
                    "leaving model unchanged.",
                    current_max,
                    max_generation_len,
                )
        response_chain = PROMPT | llm
        question_gen_chain = QUESTION_GENERATOR_PROMPT | llm | StrOutputParser()
        return cls(
            question_generator_chain=question_gen_chain,
            response_chain=response_chain,
            **kwargs,
        )

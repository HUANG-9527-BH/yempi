"""摘要中间件。"""

import logging
import uuid
import warnings
from collections.abc import Callable, Coroutine, Iterable, Mapping
from functools import partial
from typing import Any, Literal, TypedDict, cast

from langchain_core.exceptions import ContextOverflowError
from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    MessageLikeRepresentation,
    RemoveMessage,
    ToolMessage,
)
from langchain_core.messages.human import HumanMessage
from langchain_core.messages.utils import (
    count_tokens_approximately,
    get_buffer_string,
    trim_messages,
)
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_core.runnables.base import coerce_to_runnable
from langgraph.graph.message import (
    REMOVE_ALL_MESSAGES,
)
from langgraph.runtime import Runtime
from typing_extensions import override

from langchain.agents.middleware._retry import default_retry_on
from langchain.agents.middleware.internal_call_transformer import (
    InternalCallTransformer,
    internal_call_metadata,
)
from langchain.agents.middleware.types import AgentMiddleware, AgentState, ContextT, ResponseT
from langchain.chat_models import BaseChatModel, init_chat_model

logger = logging.getLogger(__name__)

TokenCounter = Callable[[Iterable[MessageLikeRepresentation]], int]

Summarizer = (
    str
    | BaseChatModel
    | Runnable[list[AnyMessage], str]
    | Callable[[list[AnyMessage]], str]
    | Callable[[list[AnyMessage]], Coroutine[Any, Any, str]]
)
"""Generates a summary from the messages selected for summarization.

Either a chat model (or model string) prompted with the summary prompt, or a
`Runnable` or callable taking the messages and returning the summary text.
"""

DEFAULT_SUMMARY_PROMPT = """<role>
Context Extraction Assistant
</role>

<primary_objective>
Your sole objective in this task is to extract the highest quality/most relevant context from the conversation history below.
</primary_objective>

<objective_information>
You're nearing the total number of input tokens you can accept, so you must extract the highest quality/most relevant pieces of information from your conversation history.
This context will then overwrite the conversation history presented below. Because of this, ensure the context you extract is only the most important information to continue working toward your overall goal.
</objective_information>

<instructions>
The conversation history below will be replaced with the context you extract in this step.
You want to ensure that you don't repeat any actions you've already completed, so the context you extract from the conversation history should be focused on the most important information to your overall goal.

You should structure your summary using the following sections. Each section acts as a checklist - you must populate it with relevant information or explicitly state "None" if there is nothing to report for that section:

## SESSION INTENT

What is the user's primary goal or request? What overall task are you trying to accomplish? This should be concise but complete enough to understand the purpose of the entire session.

## SUMMARY

Extract and record all of the most important context from the conversation history. Include important choices, conclusions, or strategies determined during this conversation. Include the reasoning behind key decisions. Document any rejected options and why they were not pursued.

## ARTIFACTS

What artifacts, files, or resources were created, modified, or accessed during this conversation? For file modifications, list specific file paths and briefly describe the changes made to each. This section prevents silent loss of artifact information.

## NEXT STEPS

What specific tasks remain to be completed to achieve the session intent? What should you do next?

</instructions>

The user will message you with the full message history from which you'll extract context to create a replacement. Carefully read through it all and think deeply about what information is most important to your overall goal and should be saved:

With all of this in mind, please carefully read over the entire conversation history, and extract the most important and relevant context to replace it so that you can free up space in the conversation history.
Respond ONLY with the extracted context. Do not include any additional information, or text before or after the extracted context.

<messages>
Messages to summarize:
{messages}
</messages>"""  # noqa: E501
"""Default prompt used to summarize conversation history.

The `<messages>` marker (on its own line) and the `{messages}` placeholder are
part of this constant's public contract, not just cosmetic formatting.
Downstream consumers depend on them: for example, deep agents'
`SummarizationMiddleware` splices an extra instruction block in immediately
before the `<messages>` marker via `str.replace`. Removing, renaming, or
reformatting the marker (or the `{messages}` placeholder) is a breaking change
for those consumers even though it does not alter any function signature, so
treat edits to it accordingly.
"""

_DEFAULT_MESSAGES_TO_KEEP = 20
_DEFAULT_TRIM_TOKEN_LIMIT = 4000
_DEFAULT_FALLBACK_MESSAGE_COUNT = 15

# 某些提供商会使用与 LangSmith `ls_provider` 不同的 `model_provider`
# 字符串标记发出的消息。
# 下面的 token 报告检查会比较这两个值，因此我们接受 `ls_provider` 已知的别名。
# 某些提供商会使用与 LangSmith `ls_provider` 不同的 `model_provider`
# 字符串标记发出的消息。
_LS_PROVIDER_ALIASES: dict[str, frozenset[str]] = {
    "amazon_bedrock": frozenset({"bedrock", "bedrock_converse"}),
        # ChatOpenAIMantle 在 ls_provider="openai-mantle" 下进行追踪，但其消息继承自
    # BaseChatOpenAI 的 model_provider="openai"。
        # ChatAnthropicMantle 在 ls_provider="anthropic-mantle" 下进行追踪，但其消息继承自
    # BaseChatAnthropic 的 model_provider="anthropic"。
    "openai-mantle": frozenset({"openai"}),
        # 3.3 是通过离线实验估算的，与 Claude 的 token 计数 API 进行对比。
        # API: https://platform.claude.com/docs/en/build-with-claude/token-counting
    "anthropic-mantle": frozenset({"anthropic"}),
}


def _summary_config() -> RunnableConfig:
    """摘要模型调用的配置。"""
    return {"metadata": {"lc_source": "summarization", **internal_call_metadata()}}


def _check_summary(summary: object) -> str:
    """验证并规范化自定义 `summarizer` 返回的文本。"""
    if not isinstance(summary, str):
        msg = f"`summarizer` must return a str, got {type(summary).__name__}."
        raise TypeError(msg)
    return summary.strip()


def _provider_matches(message_provider: str, model_ls_provider: str | None) -> bool:
    if model_ls_provider is None:
        return False
    if message_provider == model_ls_provider:
        return True
    aliases = _LS_PROVIDER_ALIASES.get(model_ls_provider)
    return aliases is not None and message_provider in aliases


ContextFraction = tuple[Literal["fraction"], float]
"""Fraction of model's maximum input tokens.

Example:
    To specify 50% of the model's max input tokens:

    ```python
    ("fraction", 0.5)
    ```
"""

ContextTokens = tuple[Literal["tokens"], int]
"""Absolute number of tokens.

Example:
    To specify 3000 tokens:

    ```python
    ("tokens", 3000)
    ```
"""

ContextMessages = tuple[Literal["messages"], int]
"""Absolute number of messages.

Example:
    To specify 50 messages:

    ```python
    ("messages", 50)
    ```
"""

ContextSize = ContextFraction | ContextTokens | ContextMessages
"""Union type for context size specifications.

Can be either:

- [`ContextFraction`][langchain.agents.middleware.summarization.ContextFraction]: A
    fraction of the model's maximum input tokens.
- [`ContextTokens`][langchain.agents.middleware.summarization.ContextTokens]: An absolute
    number of tokens.
- [`ContextMessages`][langchain.agents.middleware.summarization.ContextMessages]: An
    absolute number of messages.

Depending on use with `trigger` or `keep` parameters, this type indicates either
when to trigger summarization or how much context to retain.

Example:
    ```python
    # ContextFraction
    context_size: ContextSize = ("fraction", 0.5)

    # ContextTokens
    context_size: ContextSize = ("tokens", 3000)

    # ContextMessages
    context_size: ContextSize = ("messages", 50)
    ```
"""


class TriggerClause(TypedDict, total=False):
    """基于字典的触发条件规范，用于 AND 条件。

    All specified thresholds in a single `TriggerClause` must be met for the clause to
    trigger summarization (AND semantics). When multiple clauses are provided in a list,
    summarization triggers if any clause is met (OR semantics).

    Example:
        ```python
        # AND: Trigger when tokens >= 4000 AND messages >= 10
        trigger_clause: TriggerClause = {"tokens": 4000, "messages": 10}

        # Use in a list for OR semantics:
        trigger_list: list[TriggerClause] = [
            {"tokens": 5000, "messages": 3},
            {"tokens": 3000, "messages": 6},
        ]
        ```
    """

    tokens: int
    """当计算出的（或服务提供商报告的） token 数量达到或超过此值时触发。
    """

    messages: int
    """当消息数量达到或超过此值时触发。"""

    fraction: float
    """当计算出的（或服务提供商报告的） token 数量达到或超过模型最大输入 token
数量的此比例时触发。
    """


def _get_approximate_token_counter(model: BaseChatModel) -> TokenCounter:
    """根据模型类型调整近似 token 计数器的参数。"""
    if model._llm_type.startswith("anthropic-chat"):  # noqa: SLF001
                # 处理已弃用的参数
                # 最多重试 3 次，跳过标记为不可重试的错误。
        return partial(
            count_tokens_approximately, use_usage_metadata_scaling=True, chars_per_token=3.3
        )
    return partial(count_tokens_approximately, use_usage_metadata_scaling=True)


class SummarizationMiddleware(AgentMiddleware[AgentState[ResponseT], ContextT, ResponseT]):
    """在 token 限制接近时摘要对话历史。

    此中间件监控消息的 token 数量，并在达到阈值时自动摘要较早的消息，
    保留最近的消息，通过确保 AI/Tool 消息对保持在一起，维持上下文连续性。

    摘要生成错误会通过 `Runnable.with_retry` 在进程中重试，最多尝试 3 次，
    除非错误是标记为不可重试的 `ModelError`（例如认证或无效请求错误）。
    如果摘要提示词超出模型的上下文窗口（`ContextOverflowError`），
    将尝试一次，将历史记录中间部分截断。
    如果摘要调用仍然失败，将传播底层错误，而不是伪造摘要。

    To generate summaries with something other than a chat model, pass a
    `summarizer`. The middleware still decides when to summarize and which
    messages to summarize; only the summary text comes from the summarizer.
    """

    transformers = (InternalCallTransformer,)
    """将摘要模型调用的 token 从 `run.messages` 中排除。

    仅在使用此中间件时注册 —— 参见
    `InternalCallTransformer` 了解为何需要对调用进行标记和过滤。
    """

    def __init__(
        self,
        model: str | BaseChatModel,
        *,
        trigger: (ContextSize | TriggerClause | list[ContextSize | TriggerClause] | None) = None,
        keep: ContextSize = ("messages", _DEFAULT_MESSAGES_TO_KEEP),
        token_counter: TokenCounter = count_tokens_approximately,
        summary_prompt: str = DEFAULT_SUMMARY_PROMPT,
        trim_tokens_to_summarize: int | None = _DEFAULT_TRIM_TOKEN_LIMIT,
        summarizer: Summarizer | None = None,
        **deprecated_kwargs: Any,
    ) -> None:
        """初始化摘要中间件。

        Args:
            model: 用于生成摘要的语言模型。
            trigger: 触发摘要的一个或多个阈值。

                提供单个
                [`ContextSize`][langchain.agents.middleware.summarization.ContextSize]
                元组，或单个
                [`TriggerClause`][langchain.agents.middleware.summarization.TriggerClause]
                字典，或混合这两种形式的列表。

                A `ContextSize` tuple expresses one threshold. A `TriggerClause` dict
                expresses multiple thresholds that must *all* be met (AND). When a list is
                provided, summarization runs if *any* item is met (OR).

                !!! example

                    ```python
                    # Trigger summarization when 50 messages is reached
                    ("messages", 50)

                    # Trigger summarization when 3000 tokens is reached
                    ("tokens", 3000)

                    # Trigger summarization either when 80% of model's max input tokens
                    # is reached or when 100 messages is reached (whichever comes first)
                    [("fraction", 0.8), ("messages", 100)]

                    # Trigger when tokens >= 4000 AND messages >= 10
                    {"tokens": 4000, "messages": 10}

                    # Trigger when (tokens >= 5000 AND messages >= 3) OR
                    # (tokens >= 3000 AND messages >= 6)
                    [{"tokens": 5000, "messages": 3}, {"tokens": 3000, "messages": 6}]
                    ```

                    See [`ContextSize`][langchain.agents.middleware.summarization.ContextSize]
                    for more details.
            keep: Context retention policy applied after summarization.

                提供一个 [`ContextSize`][langchain.agents.middleware.summarization.ContextSize]
                元组，以指定要保留的历史记录量。

                默认保留最近的 `20` 条消息。

                不支持像 `trigger` 一样的多个值。

                !!! example

                    ```python
                    # Keep the most recent 20 messages
                    ("messages", 20)

                    # Keep the most recent 3000 tokens
                    ("tokens", 3000)

                    # Keep the most recent 30% of the model's max input tokens
                    ("fraction", 0.3)
                    ```
            token_counter: Function to count tokens in messages.
            summary_prompt: Prompt template for generating summaries.
            trim_tokens_to_summarize: Maximum tokens to keep when preparing messages for
                the summarization call.

                传入 `None` 以完全跳过修剪。
            summarizer: 可选的摘要模型调用替代方案。

                `model` is still used for its profile (fractional limits) and
                token counting, but is not called to write summaries.

                - A chat model, or a model string resolved with
                    `init_chat_model`, writes summaries in place of `model`. It is
                    prompted with `summary_prompt` and retried like `model` would
                    be. Use this to summarize with a different model than the one
                    whose context window you're managing. To summarize with a chat
                    model, pass its name or a `BaseChatModel` instance; wrapped
                    models such as `model.with_config(...)` are treated as a
                    `Runnable` (below).
                - A `Runnable` or function receives the messages to summarize
                    (after trimming) and returns the summary text. `summary_prompt`
                    is not used. Functions may be sync or async; an async function
                    requires async invocation (e.g. `ainvoke`), and a sync function
                    runs in a thread pool when invoked asynchronously.

                !!! example

                    ```python
                    # Manage gpt-6-sol's context window, summarize with gpt-6-luna
                    SummarizationMiddleware(
                        model="openai:gpt-6-sol",
                        trigger=("fraction", 0.8),
                        summarizer="openai:gpt-6-luna",
                    )


                    # Summarize with a non-LLM service
                    async def summarize(messages: Sequence[BaseMessage]) -> str:
                        return await my_summarization_service(get_buffer_string(messages))


                    SummarizationMiddleware(
                        model="openai:gpt-6-luna",
                        trigger=("fraction", 0.8),
                        summarizer=summarize,
                    )
                    ```
        """
                # 标准触发器表示：同一子句内为 AND，跨子句为 OR。
        if "max_tokens_before_summary" in deprecated_kwargs:
            value = deprecated_kwargs["max_tokens_before_summary"]
            warnings.warn(
                "max_tokens_before_summary is deprecated. Use trigger=('tokens', value) instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            if trigger is None and value is not None:
                trigger = ("tokens", value)

        if "messages_to_keep" in deprecated_kwargs:
            value = deprecated_kwargs["messages_to_keep"]
            warnings.warn(
                "messages_to_keep is deprecated. Use keep=('messages', value) instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            if keep == ("messages", _DEFAULT_MESSAGES_TO_KEEP):
                keep = ("messages", value)

        super().__init__()

        if isinstance(model, str):
            model = init_chat_model(model)

        self.model = model
                # 为私有消费者保留旧版兼容视图，这些消费者曾检查过之前的元组归一化表示。LangChain
        # 的行为由 _trigger_clauses 驱动，而非此属性。将在 LangChain 2.0 中移除。
        self._summary_model = self.model.with_retry(retry_if_exception_type=default_retry_on)

        self.trigger: ContextSize | TriggerClause | list[ContextSize | TriggerClause] | None = (
            self._copy_trigger(trigger)
        )

                # 撰写摘要的聊天模型：除非被 summarizer 替换，否则为 `model`。
        self._trigger_clauses = self._normalize_trigger(self.trigger)
                # `bool` 是 `int` 的子类；拒绝它，以防止 `{"messages": True}` 静默变为阈值
        # 1。应抛出 `ValueError`（而非 `TypeError`）
                # 以便每个触发器配置错误保持单一可捕获类型。
                # 拒绝浮点数和数值字符串，而不是截断或强制转换。
        self._trigger_conditions = self._legacy_trigger_conditions(self.trigger)

        self.keep = self._validate_context_size(keep, "keep")
        if token_counter is count_tokens_approximately:
            self.token_counter = _get_approximate_token_counter(self.model)
            self._partial_token_counter: TokenCounter = partial(  # type: ignore[call-arg]
                self.token_counter, use_usage_metadata_scaling=False
            )
        else:
            self.token_counter = token_counter
            self._partial_token_counter = token_counter
        self.summary_prompt = summary_prompt
        self.trim_tokens_to_summarize = trim_tokens_to_summarize
        if isinstance(summarizer, str):
            summarizer = init_chat_model(summarizer)
                # 将范围/正性验证委托给其他部分，使字典和元组形式共享相同的规则和错误信息。
        self._summary_chat_model = self.model
        self.summarizer: Runnable[list[AnyMessage], str] | None = None
        if isinstance(summarizer, BaseChatModel):
            self._summary_chat_model = summarizer
            self._summary_model = summarizer.with_retry(retry_if_exception_type=default_retry_on)
        elif summarizer is not None:
            self.summarizer = coerce_to_runnable(summarizer)

        requires_profile = any("fraction" in clause for clause in self._trigger_clauses)
        if self.keep[0] == "fraction":
            requires_profile = True
        if requires_profile and self._get_profile_limits() is None:
            msg = (
                "Model profile information is required to use fractional token limits, "
                "and is unavailable for the specified model. Please use absolute token "
                "counts instead, or pass "
                '`\n\nChatModel(..., profile={"max_input_tokens": ...})`.\n\n'
                "with a desired integer value of the model's maximum input tokens."
            )
            raise ValueError(msg)

    @override
    def before_model(
        self, state: AgentState[Any], runtime: Runtime[ContextT]
    ) -> dict[str, Any] | None:
        """在模型调用前处理消息，可能触发摘要生成。

        Args:
            state: 智能体状态。
            runtime: 运行时环境。

        Returns:
            如果执行了摘要生成，则返回包含摘要消息的更新状态。

        Raises:
            Exception: 异常：如果摘要生成在重试次数耗尽后仍然失败。
        """
        messages = state["messages"]
        self._ensure_message_ids(messages)

        total_tokens = self.token_counter(messages)
        if not self._should_summarize(messages, total_tokens):
            return None

        cutoff_index = self._determine_cutoff_index(messages)

        if cutoff_index <= 0:
            return None

        messages_to_summarize, preserved_messages = self._partition_messages(messages, cutoff_index)

        summary = self._create_summary(messages_to_summarize)
        new_messages = self._build_new_messages(summary)

        return {
            "messages": [
                RemoveMessage(id=REMOVE_ALL_MESSAGES),
                *new_messages,
                *preserved_messages,
            ]
        }

    @override
    async def abefore_model(
        self, state: AgentState[Any], runtime: Runtime[ContextT]
    ) -> dict[str, Any] | None:
        """在模型调用前处理消息，可能触发摘要生成。

        Args:
            state: 智能体状态。
            runtime: 运行时环境。

        Returns:
            如果执行了摘要生成，则返回包含摘要消息的更新状态。

        Raises:
            Exception: 异常：如果摘要生成在重试次数耗尽后仍然失败。
        """
        messages = state["messages"]
        self._ensure_message_ids(messages)

        total_tokens = self.token_counter(messages)
        if not self._should_summarize(messages, total_tokens):
            return None

        cutoff_index = self._determine_cutoff_index(messages)

        if cutoff_index <= 0:
            return None

        messages_to_summarize, preserved_messages = self._partition_messages(messages, cutoff_index)

        summary = await self._acreate_summary(messages_to_summarize)
        new_messages = self._build_new_messages(summary)

        return {
            "messages": [
                RemoveMessage(id=REMOVE_ALL_MESSAGES),
                *new_messages,
                *preserved_messages,
            ]
        }

    @staticmethod
    def _copy_trigger(
        trigger: ContextSize | TriggerClause | list[ContextSize | TriggerClause] | None,
    ) -> ContextSize | TriggerClause | list[ContextSize | TriggerClause] | None:
        """复制可变的触发器容器，以免调用者修改影响本实例。"""
        if isinstance(trigger, Mapping):
            return cast("TriggerClause", dict(trigger))
        if isinstance(trigger, list):
            return [
                cast("TriggerClause", dict(item)) if isinstance(item, Mapping) else item
                for item in trigger
            ]
        return trigger

    def _legacy_trigger_conditions(
        self,
        trigger: ContextSize | TriggerClause | list[ContextSize | TriggerClause] | None,
    ) -> list[ContextSize]:
        """将可表示为元组的触发器转换为旧版私有表示形式。"""
        if trigger is None:
            return []
        if isinstance(trigger, tuple):
            return [self._validate_context_size(trigger, "trigger")]
        if isinstance(trigger, Mapping):
            if len(trigger) != 1:
                return []
            kind, value = next(iter(trigger.items()))
            return [self._validate_context_size(cast("ContextSize", (kind, value)), "trigger")]

        conditions: list[ContextSize] = []
        for item in trigger:
            if isinstance(item, tuple):
                conditions.append(self._validate_context_size(item, "trigger"))
            elif isinstance(item, Mapping) and len(item) == 1:
                kind, value = next(iter(item.items()))
                conditions.append(
                    self._validate_context_size(cast("ContextSize", (kind, value)), "trigger")
                )
        return conditions

    def _normalize_trigger(
        self,
        trigger: (ContextSize | TriggerClause | list[ContextSize | TriggerClause] | None),
    ) -> list[TriggerClause]:
        """将受支持的触发器输入归一化为 Trigger 条件列表。

        - 元组 ("tokens", 3000) -> [{"tokens": 3000}]
        - 字典 {"tokens": 4000, "messages": 10} -> [{"tokens": 4000, "messages": 10}]
        - 两者列表 -> 条件之间为逻辑或
        """
        if trigger is None:
            return []

        def _validate_and_convert_tuple(t: ContextSize) -> TriggerClause:
            kind, value = self._validate_context_size(t, "trigger")
            return cast("TriggerClause", {kind: value})

        def _validate_mapping(m: Mapping[str, Any]) -> TriggerClause:
            """验证并转换映射为 TriggerClause。

            Type checks reject silent coercion (booleans, numeric strings, and
            fractional floats for integer metrics) so a misconfigured clause fails loudly
            at construction. Range and positivity checks are delegated to
            `_validate_context_size`, keeping a single source of truth for the rules and
            error messages shared with the tuple form.
            """
            if not m:
                msg = "trigger clause must specify at least one of 'tokens', 'messages', 'fraction'"
                raise ValueError(msg)
            out: dict[str, float | int] = {}
            for k, v in m.items():
                if k not in {"tokens", "messages", "fraction"}:
                    msg = f"Unsupported trigger metric: {k!r}"
                    raise ValueError(msg)
                                # `trigger`
                # 可能来自无类型的调用方，因此根据运行时类型进行分派，并对任何不支持的类型抛出异常。
                                # 当总 token 数超过阈值或报告的 token 数超过阈值时触发。
                                # None 截断 -> 模型配置数据不可用（在 __init__
                # 中已捕获，但此处作为安全措施），回退到消息数量。
                if isinstance(v, bool):
                    msg = f"{k} trigger value must be numeric, got {v!r}"
                    raise ValueError(msg)  # noqa: TRY004
                if k == "fraction":
                    if not isinstance(v, (int, float)):
                        msg = f"Fraction trigger values must be numeric, got {v!r}"
                        raise ValueError(msg)
                elif not isinstance(v, int):
                                        # 使用二分查找确定最早的消息索引，使得后缀保持在 token 预算内。
                    msg = f"{k} trigger values must be integers, got {v!r}"
                    raise ValueError(msg)
                                # 跳过任何 ToolMessages，以避免拆分 AI/Tool 对。
                                # 从截断位置开始及之后的连续 ToolMessages 中收集 tool_call_ids。
                self._validate_context_size(cast("ContextSize", (k, v)), "trigger")
                out[k] = v
            return cast("TriggerClause", out)

        clauses: list[TriggerClause] = []
                # 向后搜索具有匹配 tool_calls 的 AIMessage。
                # 找到 AIMessage - 将截断点移动以包含它。
        subject: Any = trigger
        if isinstance(subject, Mapping):
            clauses.append(_validate_mapping(subject))
        elif isinstance(subject, tuple):
            clauses.append(_validate_and_convert_tuple(cast("ContextSize", subject)))
        elif isinstance(subject, list):
            for item in subject:
                if isinstance(item, Mapping):
                    clauses.append(_validate_mapping(item))
                elif isinstance(item, tuple):
                    clauses.append(_validate_and_convert_tuple(cast("ContextSize", item)))
                else:
                    msg = f"Unsupported trigger item type: {type(item)}"
                    raise TypeError(msg)
        else:
            msg = f"Unsupported trigger type: {type(subject)}"
            raise TypeError(msg)
        return clauses

    def _should_summarize_based_on_reported_tokens(
        self, messages: list[AnyMessage], threshold: float
    ) -> bool:
        """检查上一条 AIMessage 报告的 token 使用量是否超过阈值。"""
        last_ai_message = next(
            (msg for msg in reversed(messages) if isinstance(msg, AIMessage)),
            None,
        )
        if (  # noqa: SIM103
            isinstance(last_ai_message, AIMessage)
            and last_ai_message.usage_metadata is not None
            and (reported_tokens := last_ai_message.usage_metadata.get("total_tokens", -1))
            and reported_tokens >= threshold
            and (message_provider := last_ai_message.response_metadata.get("model_provider"))
            and _provider_matches(
                message_provider,
                self.model._get_ls_params().get("ls_provider"),  # noqa: SLF001
            )
        ):
            return True
        return False

    def _should_summarize(self, messages: list[AnyMessage], total_tokens: int) -> bool:
        """确定当前 token 使用量是否应触发摘要生成。"""
        if not self._trigger_clauses:
            return False

        for clause in self._trigger_clauses:
            clause_met = True
            for kind, value in clause.items():
                if kind == "messages" and len(messages) < cast("int", value):
                    clause_met = False
                    break
                if kind == "tokens":
                    threshold_tokens = cast("int", value)
                                        # 回退：未找到匹配的 AIMessage，跳过 ToolMessages
                    # 以避免孤立的工具响应。
                    if (
                        total_tokens < threshold_tokens
                        and not self._should_summarize_based_on_reported_tokens(
                            messages, float(threshold_tokens)
                        )
                    ):
                        clause_met = False
                        break
                if kind == "fraction":
                    max_input_tokens = self._get_profile_limits()
                    if max_input_tokens is None:
                        clause_met = False
                        break
                    threshold = int(max_input_tokens * cast("float", value))
                    if threshold <= 0:
                        threshold = 1
                    if (
                        total_tokens < threshold
                        and not self._should_summarize_based_on_reported_tokens(
                            messages, float(threshold)
                        )
                    ):
                        clause_met = False
                        break
            if clause_met:
                return True
        return False

    def _determine_cutoff_index(self, messages: list[AnyMessage]) -> int:
        """根据保留配置选择截断索引。"""
        kind, value = self.keep
        if kind in {"tokens", "fraction"}:
            token_based_cutoff = self._find_token_based_cutoff(messages)
            if token_based_cutoff is not None:
                return token_based_cutoff
                        # 序列化为 XML，以便在摘要中保持基于 URL 的多模态块可见，同时从 token
            # 预算中排除原始消息元数据。
                        # 40% 而非一半可吸收高达 1.25 倍的使用元数据扩展。
            return self._find_safe_cutoff(messages, _DEFAULT_MESSAGES_TO_KEEP)
        return self._find_safe_cutoff(messages, cast("int", value))

    def _find_token_based_cutoff(self, messages: list[AnyMessage]) -> int | None:
        """基于目标 token 保留量查找截断索引。"""
        if not messages:
            return 0

        kind, value = self.keep
        if kind == "fraction":
            max_input_tokens = self._get_profile_limits()
            if max_input_tokens is None:
                return None
            target_token_count = int(max_input_tokens * value)
        elif kind == "tokens":
            target_token_count = int(value)
        else:
            return None

        if target_token_count <= 0:
            target_token_count = 1

        if self.token_counter(messages) <= target_token_count:
            return 0

                # 使用二分查找来确定最早的消息索引，该索引保持
                # 在 token 预算范围内的后缀。
        left, right = 0, len(messages)
        cutoff_candidate = len(messages)
        max_iterations = len(messages).bit_length() + 1
        for _ in range(max_iterations):
            if left >= right:
                break

            mid = (left + right) // 2
            if self._partial_token_counter(messages[mid:]) <= target_token_count:
                cutoff_candidate = mid
                right = mid
            else:
                left = mid + 1

        if cutoff_candidate == len(messages):
            cutoff_candidate = left

        if cutoff_candidate >= len(messages):
            if len(messages) == 1:
                return 0
            cutoff_candidate = len(messages) - 1

                # 跳过任何 ToolMessage，以避免拆分 AI/Tool 对
        return self._find_safe_cutoff_point(messages, cutoff_candidate)

    def _get_profile_limits(self, model: BaseChatModel | None = None) -> int | None:
        """从 `model` 的配置文件中获取最大输入 token 限制（默认：`self.model`）。"""
        try:
            profile = (model or self.model).profile
        except AttributeError:
            return None

        if not isinstance(profile, Mapping):
            return None

        max_input_tokens = profile.get("max_input_tokens")

        if not isinstance(max_input_tokens, int):
            return None

        return max_input_tokens

    @staticmethod
    def _validate_context_size(context: ContextSize, parameter_name: str) -> ContextSize:
        """验证上下文配置元组。"""
        kind, value = context
        if kind == "fraction":
            if not 0 < value <= 1:
                msg = f"Fractional {parameter_name} values must be between 0 and 1, got {value}."
                raise ValueError(msg)
        elif kind in {"tokens", "messages"}:
            if value <= 0:
                msg = f"{parameter_name} thresholds must be greater than 0, got {value}."
                raise ValueError(msg)
        else:
            msg = f"Unsupported context size type {kind} for {parameter_name}."
            raise ValueError(msg)
        return context

    @staticmethod
    def _build_new_messages(summary: str) -> list[HumanMessage]:
        return [
            HumanMessage(
                content=f"Here is a summary of the conversation to date:\n\n{summary}",
                additional_kwargs={"lc_source": "summarization"},
            )
        ]

    @staticmethod
    def _ensure_message_ids(messages: list[AnyMessage]) -> None:
        """确保所有消息在 add_messages reducer 中具有唯一 ID。"""
        for msg in messages:
            if msg.id is None:
                msg.id = str(uuid.uuid4())

    @staticmethod
    def _partition_messages(
        conversation_messages: list[AnyMessage],
        cutoff_index: int,
    ) -> tuple[list[AnyMessage], list[AnyMessage]]:
        """将消息划分为需要摘要和需要保留的部分。"""
        messages_to_summarize = conversation_messages[:cutoff_index]
        preserved_messages = conversation_messages[cutoff_index:]

        return messages_to_summarize, preserved_messages

    def _find_safe_cutoff(self, messages: list[AnyMessage], messages_to_keep: int) -> int:
        """找到一个安全的截断点，以保留 AI/Tool 消息对。

        返回可在不分离相关 AI 和 Tool 消息的情况下安全截断的索引。
        如果找不到安全截断点，则返回 `0`。

        此策略对摘要采取激进做法——如果目标截断点位于工具消息中间，
        我们将向前推进至所有工具消息之后（进行更多摘要）。
        """
        if len(messages) <= messages_to_keep:
            return 0

        target_cutoff = len(messages) - messages_to_keep
        return self._find_safe_cutoff_point(messages, target_cutoff)

    @staticmethod
    def _find_safe_cutoff_point(messages: list[AnyMessage], cutoff_index: int) -> int:
        """找到一个不会拆分 AI/Tool 消息对的安全截断点。

        If the message at `cutoff_index` is a `ToolMessage`, search backward for the
        `AIMessage` containing the corresponding `tool_calls` and adjust the cutoff to
        include it. This ensures tool call requests and responses stay together.

        Falls back to advancing forward past `ToolMessage` objects only if no matching
        `AIMessage` is found (edge case).
        """
        if cutoff_index >= len(messages) or not isinstance(messages[cutoff_index], ToolMessage):
            return cutoff_index

                # 从截断点处或之后的连续 ToolMessage 中收集 tool_call_ids
        tool_call_ids: set[str] = set()
        idx = cutoff_index
        while idx < len(messages) and isinstance(messages[idx], ToolMessage):
            tool_msg = cast("ToolMessage", messages[idx])
            if tool_msg.tool_call_id:
                tool_call_ids.add(tool_msg.tool_call_id)
            idx += 1

                # 向后搜索具有匹配 tool_calls 的 AIMessage
        for i in range(cutoff_index - 1, -1, -1):
            msg = messages[i]
            if isinstance(msg, AIMessage) and msg.tool_calls:
                ai_tool_call_ids = {tc.get("id") for tc in msg.tool_calls if tc.get("id")}
                if tool_call_ids & ai_tool_call_ids:
                                        # 发现 AIMessage - 将截止点移至包含它
                    return i

                # 备用方案：未找到匹配的 AIMessage，跳过 ToolMessages 以避免
                # 孤立的工具响应
        return idx

    def _create_summary(self, messages_to_summarize: list[AnyMessage]) -> str:
        """为给定消息生成摘要。

        Args:
            messages_to_summarize: 需要摘要的消息。

        Returns:
            生成的摘要。

        Raises:
            Exception: 异常：如果摘要模型调用在重试次数耗尽后（参见
                `Runnable.with_retry`）仍失败。
        """
        if not messages_to_summarize:
            return "No previous conversation history."

        trimmed_messages = self._trim_messages_for_summary(messages_to_summarize)
        if not trimmed_messages:
            return "Previous conversation was too long to summarize."

        if self.summarizer is not None:
            return _check_summary(
                self.summarizer.invoke(trimmed_messages, config=_summary_config())
            )

        prompt = self._format_summary_prompt(trimmed_messages)
        try:
            response = self._summary_model.invoke(prompt, config=_summary_config())
        except ContextOverflowError:
            retry_prompt = self._overflow_retry_prompt(trimmed_messages)
            if retry_prompt is None:
                raise
            response = self._summary_model.invoke(retry_prompt, config=_summary_config())
        return response.text.strip()

    async def _acreate_summary(self, messages_to_summarize: list[AnyMessage]) -> str:
        """为给定消息生成摘要。

        Args:
            messages_to_summarize: 需要摘要的消息。

        Returns:
            生成的摘要。

        Raises:
            Exception: 异常：如果摘要模型调用在重试次数耗尽后（参见
                `Runnable.with_retry`）仍失败。
        """
        if not messages_to_summarize:
            return "No previous conversation history."

        trimmed_messages = self._trim_messages_for_summary(messages_to_summarize)
        if not trimmed_messages:
            return "Previous conversation was too long to summarize."

        if self.summarizer is not None:
            summary = await self.summarizer.ainvoke(trimmed_messages, config=_summary_config())
            return _check_summary(summary)

        prompt = self._format_summary_prompt(trimmed_messages)
        try:
            response = await self._summary_model.ainvoke(prompt, config=_summary_config())
        except ContextOverflowError:
            retry_prompt = self._overflow_retry_prompt(trimmed_messages)
            if retry_prompt is None:
                raise
            response = await self._summary_model.ainvoke(retry_prompt, config=_summary_config())
        return response.text.strip()

    def _format_summary_prompt(self, messages: list[AnyMessage]) -> str:
        """为给定消息渲染摘要提示词。"""
                # 以 XML 格式序列化，以便在摘要中保持基于 URL 的多模态块可见
                # 提示词，同时从 token 预算中排除原始消息元数据。
        formatted_messages = get_buffer_string(messages, format="xml")
        return self.summary_prompt.format(messages=formatted_messages).rstrip()

    def _overflow_retry_prompt(self, messages: list[AnyMessage]) -> str | None:
        """在摘要调用超出预算后构建更小的摘要提示词。

        切掉格式化历史记录的中间部分，保留其开头（通常包含目标）和最近部分，
        保留长度不超过其 token 数量的一半，且不超过摘要模型 `max_input_tokens` 的
        40%（如已知）。

        Returns:
            重试提示词，或如果无内容符合预算则为 `None`。
        """
        formatted = get_buffer_string(messages, format="xml")
        tokens = self._partial_token_counter([HumanMessage(content=formatted)])
        budget = tokens // 2
        if (max_input_tokens := self._get_profile_limits(self._summary_chat_model)) is not None:
                        # 40% 而非一半可吸收高达 1.25 倍的 usage-metadata 扩展。
            budget = min(budget, max_input_tokens * 2 // 5)
        keep_chars = len(formatted) * budget // max(tokens, 1)
        if keep_chars <= 0:
            return None
        head_chars = keep_chars // 4
        tail_chars = keep_chars - head_chars
        logger.debug(
            "Summary prompt exceeded the model's context window; retrying with "
            "~%d of ~%d history tokens.",
            budget,
            tokens,
        )
        truncated = (
            f"{formatted[:head_chars]}\n\n[... middle of conversation omitted ...]\n\n"
            f"{formatted[-tail_chars:]}"
        )
        return self.summary_prompt.format(messages=truncated).rstrip()

    def _trim_messages_for_summary(self, messages: list[AnyMessage]) -> list[AnyMessage]:
        """裁剪消息以符合摘要生成限制。"""
        try:
            if self.trim_tokens_to_summarize is None:
                return messages
            return cast(
                "list[AnyMessage]",
                trim_messages(
                    messages,
                    max_tokens=self.trim_tokens_to_summarize,
                    token_counter=self.token_counter,
                    start_on="human",
                    strategy="last",
                    allow_partial=True,
                    include_system=True,
                ),
            )
        except Exception:
            return messages[-_DEFAULT_FALLBACK_MESSAGE_COUNT:]

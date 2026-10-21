"""管理另一个 `Runnable` 的聊天消息历史的 `Runnable`。"""

from __future__ import annotations

import inspect
from collections.abc import Callable, Sequence
from types import GenericAlias
from typing import (
    TYPE_CHECKING,
    Any,
)

from pydantic import BaseModel
from typing_extensions import override

from langchain_core._api.deprecation import warn_deprecated
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.load.load import load
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.runnables.base import Runnable, RunnableBindingBase, RunnableLambda
from langchain_core.runnables.passthrough import RunnablePassthrough
from langchain_core.runnables.utils import (
    ConfigurableFieldSpec,
    Output,
    get_unique_config_specs,
)
from langchain_core.utils._signature import _signature_parameters
from langchain_core.utils.pydantic import create_model_v2

if TYPE_CHECKING:
    from langchain_core.language_models.base import LanguageModelLike
    from langchain_core.runnables.config import RunnableConfig
    from langchain_core.tracers.schemas import Run


MessagesOrDictWithMessages = Sequence["BaseMessage"] | dict[str, Any]
GetSessionHistoryCallable = Callable[..., BaseChatMessageHistory]


class RunnableWithMessageHistory(RunnableBindingBase[Any, Any]):  # type: ignore[no-redef]
    """管理另一个 `Runnable` 的聊天消息历史的 `Runnable`。

    聊天消息历史是一系列代表对话的消息序列。

    `RunnableWithMessageHistory` wraps another `Runnable` and manages the chat message
    history for it; it is responsible for reading and updating the chat message
    history.

    The formats supported for the inputs and outputs of the wrapped `Runnable`
    are described below.

    `RunnableWithMessageHistory` must always be called with a config that contains
    the appropriate parameters for the chat message history factory.

    默认情况下，`Runnable` 预期接收一个名为 `session_id`
    的单个配置参数，其值为字符串。该参数用于创建新的或查找匹配给定 `session_id`
    的现有聊天消息历史。

    在这种情况下，调用形式如下：

    `with_history.invoke(..., config={"configurable": {"session_id": "bar"}})`
    ; 例如，`{"configurable": {"session_id": "<SESSION_ID>"}}`。

    可通过向 `history_factory_config` 参数传递 `ConfigurableFieldSpec`
    对象列表来自定义配置（参见下方示例）。

    In the examples, we will use a chat message history with an in-memory
    implementation to make it easy to experiment and see the results.

    对于生产场景，您将希望使用持久化的聊天消息历史实现，如 `RedisChatMessageHistory`。

    Example: Chat message history with an in-memory implementation for testing.

        ```python
        from operator import itemgetter

        from langchain_openai.chat_models import ChatOpenAI

        from langchain_core.chat_history import BaseChatMessageHistory
        from langchain_core.documents import Document
        from langchain_core.messages import BaseMessage, AIMessage
        from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
        from pydantic import BaseModel, Field
        from langchain_core.runnables import (
            RunnableLambda,
            ConfigurableFieldSpec,
            RunnablePassthrough,
        )
        from langchain_core.runnables.history import RunnableWithMessageHistory


        class InMemoryHistory(BaseChatMessageHistory, BaseModel):
            \"\"\"In memory implementation of chat message history.\"\"\"

            messages: list[BaseMessage] = Field(default_factory=list)

            def add_messages(self, messages: list[BaseMessage]) -> None:
                \"\"\"Add a list of messages to the store\"\"\"
                self.messages.extend(messages)

            def clear(self) -> None:
                self.messages = []

        # Here we use a global variable to store the chat message history.
        # This will make it easier to inspect it to see the underlying results.
        store = {}

        def get_by_session_id(session_id: str) -> BaseChatMessageHistory:
            if session_id not in store:
                store[session_id] = InMemoryHistory()
            return store[session_id]


        history = get_by_session_id("1")
        history.add_message(AIMessage(content="hello"))
        print(store)  # noqa: T201

        ```

    被包装的 `Runnable` 接受字典输入的示例：

        ```python
        from typing import Optional

        from langchain_anthropic import ChatAnthropic
        from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
        from langchain_core.runnables.history import RunnableWithMessageHistory


        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "You're an assistant who's good at {ability}"),
                MessagesPlaceholder(variable_name="history"),
                ("human", "{question}"),
            ]
        )

        chain = prompt | ChatAnthropic(model="claude-sonnet-5")

        chain_with_history = RunnableWithMessageHistory(
            chain,
            # Uses the get_by_session_id function defined in the example
            # above.
            get_by_session_id,
            input_messages_key="question",
            history_messages_key="history",
        )

        print(
            chain_with_history.invoke(  # noqa: T201
                {"ability": "math", "question": "What does cosine mean?"},
                config={"configurable": {"session_id": "foo"}},
            )
        )

        # Uses the store defined in the example above.
        print(store)  # noqa: T201

        print(
            chain_with_history.invoke(  # noqa: T201
                {"ability": "math", "question": "What's its inverse"},
                config={"configurable": {"session_id": "foo"}},
            )
        )

        print(store)  # noqa: T201
        ```

    会话工厂接受两个键（`user_id` 和 `conversation_id`）的示例：

        ```python
        store = {}


        def get_session_history(
            user_id: str, conversation_id: str
        ) -> BaseChatMessageHistory:
            if (user_id, conversation_id) not in store:
                store[(user_id, conversation_id)] = InMemoryHistory()
            return store[(user_id, conversation_id)]


        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "You're an assistant who's good at {ability}"),
                MessagesPlaceholder(variable_name="history"),
                ("human", "{question}"),
            ]
        )

        chain = prompt | ChatAnthropic(model="claude-sonnet-5")

        with_message_history = RunnableWithMessageHistory(
            chain,
            get_session_history=get_session_history,
            input_messages_key="question",
            history_messages_key="history",
            history_factory_config=[
                ConfigurableFieldSpec(
                    id="user_id",
                    annotation=str,
                    name="User ID",
                    description="Unique identifier for the user.",
                    default="",
                    is_shared=True,
                ),
                ConfigurableFieldSpec(
                    id="conversation_id",
                    annotation=str,
                    name="Conversation ID",
                    description="Unique identifier for the conversation.",
                    default="",
                    is_shared=True,
                ),
            ],
        )

        with_message_history.invoke(
            {"ability": "math", "question": "What does cosine mean?"},
            config={"configurable": {"user_id": "123", "conversation_id": "1"}},
        )
        ```
    """

    get_session_history: GetSessionHistoryCallable
    """返回新 `BaseChatMessageHistory` 实例的函数。

    This function should either take a single positional argument `session_id` of type
    string and return a corresponding chat message history instance
    """
    input_messages_key: str | None = None
    """如果基础 `Runnable` 接受 `dict` 作为输入，则必须指定。
    输入 `dict` 中包含消息的键。
    """
    output_messages_key: str | None = None
    """如果基础 `Runnable` 返回 `dict` 作为输出，则必须指定。
    输出 `dict` 中包含消息的键。
    """
    history_messages_key: str | None = None
    """Must be specified if the base `Runnable` accepts a `dict` as input and expects a
    separate key for historical messages.
    """
    history_factory_config: Sequence[ConfigurableFieldSpec]
    """配置应传递给聊天历史工厂的字段。

    有关更多详细信息，请参阅 `ConfigurableFieldSpec`。
    """

    def __init__(
        self,
        runnable: Runnable[
            list[BaseMessage], str | BaseMessage | MessagesOrDictWithMessages
        ]
        | Runnable[dict[str, Any], str | BaseMessage | MessagesOrDictWithMessages]
        | LanguageModelLike,
        get_session_history: GetSessionHistoryCallable,
        *,
        input_messages_key: str | None = None,
        output_messages_key: str | None = None,
        history_messages_key: str | None = None,
        history_factory_config: Sequence[ConfigurableFieldSpec] | None = None,
        **kwargs: Any,
    ) -> None:
        """初始化 `RunnableWithMessageHistory`。

        Args:
            runnable: 要包装的基础 `Runnable`。

                必须接受以下输入之一：

                1. A list of `BaseMessage`
                2. A `dict` with one key for all messages
                3. A `dict` with one key for the current input string/message(s) and
                    a separate key for historical messages. If the input key points
                    to a string, it will be treated as a `HumanMessage` in history.

                必须返回以下输出之一：

                1. A string which can be treated as an `AIMessage`
                2. A `BaseMessage` or sequence of `BaseMessage`
                3. A `dict` with a key for a `BaseMessage` or sequence of
                    `BaseMessage`

            get_session_history: 返回新的 `BaseChatMessageHistory` 的函数。

                This function should either take a single positional argument
                `session_id` of type string and return a corresponding
                chat message history instance.

                ```python
                def get_session_history(
                    session_id: str, *, user_id: str | None = None
                ) -> BaseChatMessageHistory: ...
                ```

                Or it should take keyword arguments that match the keys of
                `session_history_config_specs` and return a corresponding
                chat message history instance.

                ```python
                def get_session_history(
                    *,
                    user_id: str,
                    thread_id: str,
                ) -> BaseChatMessageHistory: ...
                ```

            input_messages_key: Must be specified if the base runnable accepts a `dict`
                as input.
            output_messages_key: Must be specified if the base runnable returns a `dict`
                as output.
            history_messages_key: Must be specified if the base runnable accepts a
                `dict` as input and expects a separate key for historical messages.
            history_factory_config: Configure fields that should be passed to the
                chat history factory. See `ConfigurableFieldSpec` for more details.

                指定这些字段可让你将多个配置键传入 `get_session_history` 工厂。
                **kwargs: 任意额外的 kwargs，将传递给父类 `RunnableBindingBase`
                的初始化。

        """
        warn_deprecated(
            since="1.3.3",
            message=(
                "RunnableWithMessageHistory is deprecated. "
                "Use LangGraph's built-in persistence instead."
            ),
            removal="2.0.0",
        )
        history_chain: Runnable[Any, Any] = RunnableLambda(
            self._enter_history, self._aenter_history
        ).with_config(run_name="load_history")
        messages_key = history_messages_key or input_messages_key
        if messages_key:
            history_chain = RunnablePassthrough.assign(
                **{messages_key: history_chain}
            ).with_config(run_name="insert_history")

        runnable_sync = runnable.with_listeners(on_end=self._exit_history)
        runnable_async = runnable.with_alisteners(on_end=self._aexit_history)

        def _call_runnable_sync(_input: Any) -> Runnable[Any, Any]:
            return runnable_sync

        async def _call_runnable_async(_input: Any) -> Runnable[Any, Any]:
            return runnable_async

        bound = (
            history_chain
            | RunnableLambda(
                _call_runnable_sync,
                _call_runnable_async,
            ).with_config(run_name="check_sync_or_async")
        ).with_config(run_name="RunnableWithMessageHistory")

        if history_factory_config:
            config_specs = history_factory_config
        else:
                        # 如果未提供，则使用默认的 session_id 字段
            config_specs = [
                ConfigurableFieldSpec(
                    id="session_id",
                    annotation=str,
                    name="Session ID",
                    description="Unique identifier for a session.",
                    default="",
                    is_shared=True,
                ),
            ]

        super().__init__(
            get_session_history=get_session_history,
            input_messages_key=input_messages_key,
            output_messages_key=output_messages_key,
            bound=bound,
            history_messages_key=history_messages_key,
            history_factory_config=config_specs,
            **kwargs,
        )
        self._history_chain = history_chain

    @property
    @override
    def config_specs(self) -> list[ConfigurableFieldSpec]:
        """获取 `RunnableWithMessageHistory` 的配置规格。"""
        return get_unique_config_specs(
            super().config_specs + list(self.history_factory_config)
        )

    @override
    def get_input_schema(self, config: RunnableConfig | None = None) -> type[BaseModel]:
        fields: dict[str, Any] = {}
        if self.input_messages_key and self.history_messages_key:
            fields[self.input_messages_key] = (
                str | BaseMessage | Sequence[BaseMessage],
                ...,
            )
        elif self.input_messages_key:
            fields[self.input_messages_key] = (Sequence[BaseMessage], ...)
        else:
            return create_model_v2(
                "RunnableWithChatHistoryInput",
                module_name=self.__class__.__module__,
                root=(Sequence[BaseMessage], ...),
            )
        return create_model_v2(
            "RunnableWithChatHistoryInput",
            field_definitions=fields,
            module_name=self.__class__.__module__,
        )

    @property
    @override
    def OutputType(self) -> type[Output]:
        return self._history_chain.OutputType

    @override
    def get_output_schema(
        self, config: RunnableConfig | None = None
    ) -> type[BaseModel]:
        """获取可用于验证 `Runnable` 输出的 Pydantic 模型。

        使用 `configurable_fields` 和 `configurable_alternatives` 方法的 `Runnable`
        对象将具有动态输出模式， 其取决于 `Runnable` 被调用时所使用的配置。

        此方法允许为特定配置获取输出模式。

        Args:
            config: 用于生成模式时使用的配置。

        Returns:
            可用于验证输出的 Pydantic 模型。
        """
        root_type = self.OutputType

        if (
            inspect.isclass(root_type)
            and not isinstance(root_type, GenericAlias)
            and issubclass(root_type, BaseModel)
        ):
            return root_type

        return create_model_v2(
            "RunnableWithChatHistoryOutput",
            root=root_type,
            module_name=self.__class__.__module__,
        )

    def _get_input_messages(
        self, input_val: str | BaseMessage | Sequence[BaseMessage] | dict[str, Any]
    ) -> list[BaseMessage]:
                # 如果为字典，尝试提取代表消息的单一键
        if isinstance(input_val, dict):
            if self.input_messages_key:
                key = self.input_messages_key
            elif len(input_val) == 1:
                key = next(iter(input_val.keys()))
            else:
                key = "input"
            input_val = input_val[key]

                # 如果值是字符串，转换为人类消息
        if isinstance(input_val, str):
            return [HumanMessage(content=input_val)]
                # 如果值是单个消息，转换为列表
        if isinstance(input_val, BaseMessage):
            return [input_val]
                # 如果值是列表或元组...
        if isinstance(input_val, (list, tuple)):
                        # 处理空值情况
            if len(input_val) == 0:
                return list(input_val)
                        # 如果是列表的列表，则返回第一个值
                        # 这在聊天模型中发生——因为我们会批量输入
            if isinstance(input_val[0], list):
                if len(input_val) != 1:
                    msg = f"Expected a single list of messages. Got {input_val}."
                    raise ValueError(msg)
                return input_val[0]
            return list(input_val)
        msg = (
            f"Expected str, BaseMessage, list[BaseMessage], or tuple[BaseMessage]. "
            f"Got {input_val}."
        )
        raise ValueError(msg)

    def _get_output_messages(
        self, output_val: str | BaseMessage | Sequence[BaseMessage] | dict[str, Any]
    ) -> list[BaseMessage]:
                # 如果为字典，尝试提取代表消息的单一键
        if isinstance(output_val, dict):
            if self.output_messages_key:
                key = self.output_messages_key
            elif len(output_val) == 1:
                key = next(iter(output_val.keys()))
            else:
                key = "output"
                        # 如果你直接包装聊天模型
                        # 输出实际上是这个奇怪的 generations 对象
            if key not in output_val and "generations" in output_val:
                output_val = output_val["generations"][0][0]["message"]
            else:
                output_val = output_val[key]

        if isinstance(output_val, str):
            return [AIMessage(content=output_val)]
                # 如果值是单个消息，转换为列表
        if isinstance(output_val, BaseMessage):
            return [output_val]
        if isinstance(output_val, (list, tuple)):
            return list(output_val)
        msg = (
            f"Expected str, BaseMessage, list[BaseMessage], or tuple[BaseMessage]. "
            f"Got {output_val}."
        )
        raise ValueError(msg)

    def _enter_history(self, value: Any, config: RunnableConfig) -> list[BaseMessage]:
        hist: BaseChatMessageHistory = config["configurable"]["message_history"]
        messages = hist.messages.copy()

        if not self.history_messages_key:
                        # 返回所有消息
            input_val = (
                value if not self.input_messages_key else value[self.input_messages_key]
            )
            messages += self._get_input_messages(input_val)
        return messages

    async def _aenter_history(
        self, value: dict[str, Any], config: RunnableConfig
    ) -> list[BaseMessage]:
        hist: BaseChatMessageHistory = config["configurable"]["message_history"]
        messages = (await hist.aget_messages()).copy()

        if not self.history_messages_key:
                        # 返回所有消息
            input_val = (
                value if not self.input_messages_key else value[self.input_messages_key]
            )
            messages += self._get_input_messages(input_val)
        return messages

    def _exit_history(self, run: Run, config: RunnableConfig) -> None:
        hist: BaseChatMessageHistory = config["configurable"]["message_history"]

                # 获取输入消息
        inputs = load(run.inputs, allowed_objects="messages")
        input_messages = self._get_input_messages(inputs)
                # 如果历史消息被附加到输入消息前，将其移除以
                # 避免向历史记录中添加重复消息。
        if not self.history_messages_key:
            historic_messages = config["configurable"]["message_history"].messages
            input_messages = input_messages[len(historic_messages) :]

                # 获取输出消息
        output_val = load(run.outputs, allowed_objects="messages")
        output_messages = self._get_output_messages(output_val)
        hist.add_messages(input_messages + output_messages)

    async def _aexit_history(self, run: Run, config: RunnableConfig) -> None:
        hist: BaseChatMessageHistory = config["configurable"]["message_history"]

                # 获取输入消息
        inputs = load(run.inputs, allowed_objects="messages")
        input_messages = self._get_input_messages(inputs)
                # 如果历史消息被附加到输入消息前，将其移除以
                # 避免向历史记录中添加重复的消息。
        if not self.history_messages_key:
            historic_messages = await hist.aget_messages()
            input_messages = input_messages[len(historic_messages) :]

                # 获取输出消息
        output_val = load(run.outputs, allowed_objects="messages")
        output_messages = self._get_output_messages(output_val)
        await hist.aadd_messages(input_messages + output_messages)

    def _merge_configs(self, *configs: RunnableConfig | None) -> RunnableConfig:
        config = super()._merge_configs(*configs)
        expected_keys = [field_spec.id for field_spec in self.history_factory_config]

        configurable = config.get("configurable", {})

        missing_keys = set(expected_keys) - set(configurable.keys())
        parameter_names = _get_parameter_names(self.get_session_history)

        if missing_keys and parameter_names:
            example_input = {self.input_messages_key: "foo"}
            example_configurable = dict.fromkeys(missing_keys, "[your-value-here]")
            example_config = {"configurable": example_configurable}
            msg = (
                f"Missing keys {sorted(missing_keys)} in config['configurable'] "
                f"Expected keys are {sorted(expected_keys)}."
                f"When using via .invoke() or .stream(), pass in a config; "
                f"e.g., chain.invoke({example_input}, {example_config})"
            )
            raise ValueError(msg)

        if len(expected_keys) == 1:
            if parameter_names:
                                # 如果参数数量 = 1，则通过位置参数调用函数
                message_history = self.get_session_history(
                    configurable[expected_keys[0]]
                )
            else:
                if not config:
                    config["configurable"] = {}
                message_history = self.get_session_history()
        else:
                        # 否则验证键名是否匹配，并通过命名参数调用
            if set(expected_keys) != set(parameter_names):
                msg = (
                    f"Expected keys {sorted(expected_keys)} do not match parameter "
                    f"names {sorted(parameter_names)} of get_session_history."
                )
                raise ValueError(msg)

            message_history = self.get_session_history(
                **{key: configurable[key] for key in expected_keys}
            )
        config["configurable"]["message_history"] = message_history
        return config


def _get_parameter_names(callable_: GetSessionHistoryCallable) -> list[str]:
    """获取 `Callable` 的参数名称。"""
    return list(_signature_parameters(callable_))

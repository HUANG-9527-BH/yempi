from __future__ import annotations

import asyncio
import json
from collections.abc import Callable, Sequence
from json import JSONDecodeError
from time import sleep
from typing import (
    TYPE_CHECKING,
    Any,
)

from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.callbacks import CallbackManager
from langchain_core.load import dumpd
from langchain_core.runnables import RunnableConfig, RunnableSerializable, ensure_config
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import BaseModel, Field, model_validator
from typing_extensions import Self, override

if TYPE_CHECKING:
    import openai
    from openai.types.beta.threads import (  # type: ignore[attr-defined,unused-ignore]
        ThreadMessage,
    )
    from openai.types.beta.threads.required_action_function_tool_call import (
        RequiredActionFunctionToolCall,
    )


class OpenAIAssistantFinish(AgentFinish):
    """智能体完成时包含运行和线程元数据。

    Args:
        run_id: 运行 id。
        thread_id: 线程 id。
    """

    run_id: str
    thread_id: str

    @classmethod
    def is_lc_serializable(cls) -> bool:
        """检查类是否可由 LangChain 序列化。

        Returns:
            False
        """
        return False


class OpenAIAssistantAction(AgentAction):
    """智能体动作，包含向现有运行提交自定义工具输出所需的信息。

    Args:
        tool_call_id: 工具调用 id。
        run_id: 运行 id。
        thread_id: 线程 id
    """

    tool_call_id: str
    run_id: str
    thread_id: str

    @classmethod
    def is_lc_serializable(cls) -> bool:
        """检查类是否可由 LangChain 序列化。

        Returns:
            False
        """
        return False


def _get_openai_client() -> openai.OpenAI:
    try:
        import openai

        return openai.OpenAI()
    except ImportError as e:
        msg = "Unable to import openai, please install with `pip install openai`."
        raise ImportError(msg) from e
    except AttributeError as e:
        msg = (
            "Please make sure you are using a v1.1-compatible version of openai. You "
            'can install with `pip install "openai>=1.1"`.'
        )
        raise AttributeError(msg) from e


def _get_openai_async_client() -> openai.AsyncOpenAI:
    try:
        import openai

        return openai.AsyncOpenAI()
    except ImportError as e:
        msg = "Unable to import openai, please install with `pip install openai`."
        raise ImportError(msg) from e
    except AttributeError as e:
        msg = (
            "Please make sure you are using a v1.1-compatible version of openai. You "
            'can install with `pip install "openai>=1.1"`.'
        )
        raise AttributeError(msg) from e


def _is_assistants_builtin_tool(
    tool: dict[str, Any] | type[BaseModel] | Callable | BaseTool,
) -> bool:
    """判断工具是否对应 OpenAI 助手内置工具。"""
    assistants_builtin_tools = ("code_interpreter", "file_search")
    return (
        isinstance(tool, dict)
        and ("type" in tool)
        and (tool["type"] in assistants_builtin_tools)
    )


def _get_assistants_tool(
    tool: dict[str, Any] | type[BaseModel] | Callable | BaseTool,
) -> dict[str, Any]:
    """将原始函数/类转换为 OpenAI 工具。

    注意，OpenAI 助手支持多种内置工具，
    例如 "code_interpreter" 和 "file_search"。
    """
    if _is_assistants_builtin_tool(tool):
        return tool  # type: ignore[return-value]
    return convert_to_openai_tool(tool)


OutputType = (
    list[OpenAIAssistantAction]
    | OpenAIAssistantFinish
    | list["ThreadMessage"]
    | list["RequiredActionFunctionToolCall"]
)


class OpenAIAssistantRunnable(RunnableSerializable[dict, OutputType]):
    """运行 OpenAI 助手。

    使用 OpenAI 工具的示例：
        ```python
        from langchain_experimental.openai_assistant import OpenAIAssistantRunnable

        interpreter_assistant = OpenAIAssistantRunnable.create_assistant(
            name="langchain assistant",
            instructions="You are a personal math tutor. "
            "Write and run code to answer math questions.",
            tools=[{"type": "code_interpreter"}],
            model="gpt-5.5",
        )
        output = interpreter_assistant.invoke(
            {"content": "What's 10 - 4 raised to the 2.7"}
        )
        ```

    使用自定义工具和 AgentExecutor 的示例：
        ```python
        from langchain_experimental.openai_assistant import OpenAIAssistantRunnable
        from langchain_classic.agents import AgentExecutor
        from langchain_classic.tools import E2BDataAnalysisTool


        tools = [E2BDataAnalysisTool(api_key="...")]
        agent = OpenAIAssistantRunnable.create_assistant(
            name="langchain assistant e2b tool",
            instructions="You are a personal math tutor. "
            "Write and run code to answer math questions.",
            tools=tools,
            model="gpt-5.5",
            as_agent=True,
        )

        agent_executor = AgentExecutor(agent=agent, tools=tools)
        agent_executor.invoke({"content": "What's 10 - 4 raised to the 2.7"})
        ```

    使用自定义工具和自定义执行的示例：
        ```python
        from langchain_experimental.openai_assistant import OpenAIAssistantRunnable
        from langchain_classic.agents import AgentExecutor
        from langchain_core.agents import AgentFinish
        from langchain_classic.tools import E2BDataAnalysisTool


        tools = [E2BDataAnalysisTool(api_key="...")]
        agent = OpenAIAssistantRunnable.create_assistant(
            name="langchain assistant e2b tool",
            instructions="You are a personal math tutor. "
            "Write and run code to answer math questions.",
            tools=tools,
            model="gpt-5.5",
            as_agent=True,
        )


        def execute_agent(agent, tools, input):
            tool_map = {tool.name: tool for tool in tools}
            response = agent.invoke(input)
            while not isinstance(response, AgentFinish):
                tool_outputs = []
                for action in response:
                    tool_output = tool_map[action.tool].invoke(action.tool_input)
                    tool_outputs.append(
                        {
                            "output": tool_output,
                            "tool_call_id": action.tool_call_id,
                        }
                    )
                response = agent.invoke(
                    {
                        "tool_outputs": tool_outputs,
                        "run_id": action.run_id,
                        "thread_id": action.thread_id,
                    }
                )

            return response


        response = execute_agent(
            agent, tools, {"content": "What's 10 - 4 raised to the 2.7"}
        )
        next_response = execute_agent(
            agent,
            tools,
            {"content": "now add 17.241", "thread_id": response.thread_id},
        )
        ```
    """

    client: Any = Field(default_factory=_get_openai_client)
    """`OpenAI` 或 `AzureOpenAI` 客户端。"""
    async_client: Any = None
    """`OpenAI` 或 `AzureOpenAI` 异步客户端。"""
    assistant_id: str
    """OpenAI 助手 id。"""
    check_every_ms: float = 1_000.0
    """以毫秒为单位检查运行进度的频率。"""
    as_agent: bool = False
    """作为 LangChain 智能体使用，与 `AgentExecutor` 兼容。"""

    @model_validator(mode="after")
    def _validate_async_client(self) -> Self:
        if self.async_client is None:
            import openai

            api_key = self.client.api_key
            self.async_client = openai.AsyncOpenAI(api_key=api_key)
        return self

    @classmethod
    def create_assistant(
        cls,
        name: str,
        instructions: str,
        tools: Sequence[BaseTool | dict],
        model: str,
        *,
        client: openai.OpenAI | openai.AzureOpenAI | None = None,
        **kwargs: Any,
    ) -> OpenAIAssistantRunnable:
        """创建一个 OpenAI 助手并实例化 Runnable。

        Args:
            name: 助手名称。
            instructions: 助手指令。
            tools: 助手工具。可以以 OpenAI 格式或 BaseTools 形式传入。
            model: 助手使用的模型。
            client: OpenAI 或 AzureOpenAI 客户端。
                如果未指定，则会创建默认的 OpenAI 客户端。
            kwargs: 其他参数。

        Returns:
            已配置为使用创建的助手运行的 OpenAIAssistantRunnable。
        """
        client = client or _get_openai_client()
        assistant = client.beta.assistants.create(  # type: ignore[deprecated,unused-ignore]
            name=name,
            instructions=instructions,
            tools=[_get_assistants_tool(tool) for tool in tools],  # type: ignore[misc,unused-ignore]
            model=model,
        )
        return cls(assistant_id=assistant.id, client=client, **kwargs)

    @override
    def invoke(
        self,
        input: dict,
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> OutputType:
        """调用助手。

        Args:
            input: Runnable input dict that can have:
                content: User message when starting a new run.
                thread_id: Existing thread to use.
                run_id: Existing run to use. Should only be supplied when providing
                    the tool output for a required action after an initial invocation.
                message_metadata: Metadata to associate with new message.
                thread_metadata: Metadata to associate with new thread. Only relevant
                    when new thread being created.
                instructions: Additional run instructions.
                model: Override Assistant model for this run.
                tools: Override Assistant tools for this run.
                parallel_tool_calls: Allow Assistant to set parallel_tool_calls
                    for this run.
                top_p: Override Assistant top_p for this run.
                temperature: Override Assistant temperature for this run.
                max_completion_tokens: Allow setting max_completion_tokens for this run.
                max_prompt_tokens: Allow setting max_prompt_tokens for this run.
                run_metadata: Metadata to associate with new run.
                attachments: A list of files attached to the message, and the
                    tools they should be added to.
            config: Runnable config.
            **kwargs: Additional arguments.

        Returns:
            如果 self.as_agent 为 True，将返回
                Union[List[OpenAIAssistantAction], OpenAIAssistantFinish]。
                否则，将返回 OpenAI 类型
                Union[List[ThreadMessage], List[RequiredActionFunctionToolCall]]。
        """
        config = ensure_config(config)
        callback_manager = CallbackManager.configure(
            inheritable_callbacks=config.get("callbacks"),
            inheritable_tags=config.get("tags"),
            inheritable_metadata=config.get("metadata"),
        )
        run_manager = callback_manager.on_chain_start(
            dumpd(self),
            input,
            name=config.get("run_name") or self.get_name(),
        )
        try:
                        # 在 AgentExecutor 内运行，并且有工具输出需要提交。
            if self.as_agent and input.get("intermediate_steps"):
                tool_outputs = self._parse_intermediate_steps(
                    input["intermediate_steps"],
                )
                run = self.client.beta.threads.runs.submit_tool_outputs(**tool_outputs)
                        # 启动一个新线程和一个新运行。
            elif "thread_id" not in input:
                thread = {
                    "messages": [
                        {
                            "role": "user",
                            "content": input["content"],
                            "metadata": input.get("message_metadata"),
                            "attachments": input.get("attachments"),
                        },
                    ],
                    "metadata": input.get("thread_metadata"),
                }
                run = self._create_thread_and_run(input, thread)
                        # 在现有线程中启动一个新运行。
            elif "run_id" not in input:
                _ = self.client.beta.threads.messages.create(
                    input["thread_id"],
                    content=input["content"],
                    role="user",
                    metadata=input.get("message_metadata"),
                )
                run = self._create_run(input)
                        # 向现有运行提交工具输出，不在 AgentExecutor 内部
                        # 框架。
            else:
                run = self.client.beta.threads.runs.submit_tool_outputs(**input)
            run = self._wait_for_run(run.id, run.thread_id)
        except BaseException as e:
            run_manager.on_chain_error(e)
            raise
        try:
                        # 在同步调用中使用同步响应处理器
            response = self._get_response(run)
        except BaseException as e:
            run_manager.on_chain_error(e, metadata=run.dict())
            raise
        else:
            run_manager.on_chain_end(response)
            return response

    @classmethod
    async def acreate_assistant(
        cls,
        name: str,
        instructions: str,
        tools: Sequence[BaseTool | dict],
        model: str,
        *,
        async_client: openai.AsyncOpenAI | openai.AsyncAzureOpenAI | None = None,
        **kwargs: Any,
    ) -> OpenAIAssistantRunnable:
        """异步创建 AsyncOpenAI Assistant 并实例化 Runnable。

        Args:
            name: 助手名称。
            instructions: 助手指令。
            tools: 助手工具。可以以 OpenAI 格式或 BaseTools 形式传入。
            model: 助手使用的模型。
            async_client: 异步 OpenAI 客户端。
                若未指定，则会创建默认的异步客户端。
            **kwargs: 其他参数。

        Returns:
            配置为使用创建的助手运行的 AsyncOpenAIAssistantRunnable。
        """
        async_client = async_client or _get_openai_async_client()
        openai_tools = [_get_assistants_tool(tool) for tool in tools]
        assistant = await async_client.beta.assistants.create(  # type: ignore[deprecated,unused-ignore]
            name=name,
            instructions=instructions,
            tools=openai_tools,  # type: ignore[arg-type,unused-ignore]
            model=model,
        )
        return cls(assistant_id=assistant.id, async_client=async_client, **kwargs)

    @override
    async def ainvoke(
        self,
        input: dict,
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> OutputType:
        """异步调用助手。

        Args:
            input: Runnable input dict that can have:
                content: User message when starting a new run.
                thread_id: Existing thread to use.
                run_id: Existing run to use. Should only be supplied when providing
                    the tool output for a required action after an initial invocation.
                message_metadata: Metadata to associate with a new message.
                thread_metadata: Metadata to associate with new thread. Only relevant
                    when a new thread is created.
                instructions: Overrides the instructions of the assistant.
                additional_instructions: Appends additional instructions.
                model: Override Assistant model for this run.
                tools: Override Assistant tools for this run.
                parallel_tool_calls: Allow Assistant to set parallel_tool_calls
                    for this run.
                top_p: Override Assistant top_p for this run.
                temperature: Override Assistant temperature for this run.
                max_completion_tokens: Allow setting max_completion_tokens for this run.
                max_prompt_tokens: Allow setting max_prompt_tokens for this run.
                run_metadata: Metadata to associate with new run.
            config: Runnable config.
            kwargs: Additional arguments.

        Returns:
            如果 self.as_agent 为 True，将返回
                Union[List[OpenAIAssistantAction], OpenAIAssistantFinish]。
                否则，将返回 OpenAI 类型
                Union[List[ThreadMessage], List[RequiredActionFunctionToolCall]]。
        """
        config = config or {}
        callback_manager = CallbackManager.configure(
            inheritable_callbacks=config.get("callbacks"),
            inheritable_tags=config.get("tags"),
            inheritable_metadata=config.get("metadata"),
        )
        run_manager = callback_manager.on_chain_start(
            dumpd(self),
            input,
            name=config.get("run_name") or self.get_name(),
        )
        try:
                        # 在 AgentExecutor 内运行，并且有工具输出需要提交。
            if self.as_agent and input.get("intermediate_steps"):
                tool_outputs = await self._aparse_intermediate_steps(
                    input["intermediate_steps"],
                )
                run = await self.async_client.beta.threads.runs.submit_tool_outputs(
                    **tool_outputs,
                )
                        # 启动一个新线程和一个新运行。
            elif "thread_id" not in input:
                thread = {
                    "messages": [
                        {
                            "role": "user",
                            "content": input["content"],
                            "metadata": input.get("message_metadata"),
                        },
                    ],
                    "metadata": input.get("thread_metadata"),
                }
                run = await self._acreate_thread_and_run(input, thread)
                        # 在现有线程中启动一个新运行。
            elif "run_id" not in input:
                _ = await self.async_client.beta.threads.messages.create(
                    input["thread_id"],
                    content=input["content"],
                    role="user",
                    metadata=input.get("message_metadata"),
                )
                run = await self._acreate_run(input)
                        # 向现有运行提交工具输出，不在 AgentExecutor 内部
                        # 框架。
            else:
                run = await self.async_client.beta.threads.runs.submit_tool_outputs(
                    **input,
                )
            run = await self._await_for_run(run.id, run.thread_id)
        except BaseException as e:
            run_manager.on_chain_error(e)
            raise
        try:
                        # 在异步 ainvoke 中使用异步响应处理器
            response = await self._aget_response(run)
        except BaseException as e:
            run_manager.on_chain_error(e, metadata=run.dict())
            raise
        else:
            run_manager.on_chain_end(response)
            return response

    def _parse_intermediate_steps(
        self,
        intermediate_steps: list[tuple[OpenAIAssistantAction, str]],
    ) -> dict:
        last_action, _ = intermediate_steps[-1]
        run = self._wait_for_run(last_action.run_id, last_action.thread_id)
        required_tool_call_ids = set()
        if run.required_action:
            required_tool_call_ids = {
                tc.id for tc in run.required_action.submit_tool_outputs.tool_calls
            }
        tool_outputs = [
            {"output": str(output), "tool_call_id": action.tool_call_id}
            for action, output in intermediate_steps
            if action.tool_call_id in required_tool_call_ids
        ]
        return {
            "tool_outputs": tool_outputs,
            "run_id": last_action.run_id,
            "thread_id": last_action.thread_id,
        }

    def _create_run(self, input_dict: dict) -> Any:
        params = {
            k: v
            for k, v in input_dict.items()
            if k
            in (
                "instructions",
                "model",
                "tools",
                "additional_instructions",
                "parallel_tool_calls",
                "top_p",
                "temperature",
                "max_completion_tokens",
                "max_prompt_tokens",
                "run_metadata",
            )
        }
        return self.client.beta.threads.runs.create(
            input_dict["thread_id"],
            assistant_id=self.assistant_id,
            **params,
        )

    def _create_thread_and_run(self, input_dict: dict, thread: dict) -> Any:
        params = {
            k: v
            for k, v in input_dict.items()
            if k
            in (
                "instructions",
                "model",
                "tools",
                "parallel_tool_calls",
                "top_p",
                "temperature",
                "max_completion_tokens",
                "max_prompt_tokens",
                "run_metadata",
            )
        }
        return self.client.beta.threads.create_and_run(
            assistant_id=self.assistant_id,
            thread=thread,
            **params,
        )

    def _get_response(self, run: Any) -> Any:
                # TODO: 分页

        if run.status == "completed":
            import openai

            major_version = int(openai.version.VERSION.split(".")[0])
            minor_version = int(openai.version.VERSION.split(".")[1])
            version_gte_1_14 = (major_version > 1) or (
                major_version == 1 and minor_version >= 14  # noqa: PLR2004
            )

            messages = self.client.beta.threads.messages.list(
                run.thread_id,
                order="asc",
            )
            new_messages = [msg for msg in messages if msg.run_id == run.id]
            if not self.as_agent:
                return new_messages
            answer: Any = [
                msg_content for msg in new_messages for msg_content in msg.content
            ]
            attachments = [
                attachment for msg in new_messages for attachment in msg.attachments
            ]
            if all(
                (
                    isinstance(content, openai.types.beta.threads.TextContentBlock)
                    if version_gte_1_14
                    else isinstance(
                        content,
                        openai.types.beta.threads.MessageContentText,  # type: ignore[attr-defined,unused-ignore]
                    )
                )
                for content in answer
            ):
                answer = "\n".join(content.text.value for content in answer)
            return OpenAIAssistantFinish(
                return_values={
                    "output": answer,
                    "thread_id": run.thread_id,
                    "run_id": run.id,
                    "attachments": attachments,
                },
                log="",
                run_id=run.id,
                thread_id=run.thread_id,
            )
        if run.status == "requires_action":
            if not self.as_agent:
                return run.required_action.submit_tool_outputs.tool_calls
            actions = []
            for tool_call in run.required_action.submit_tool_outputs.tool_calls:
                function = tool_call.function
                try:
                    args = json.loads(function.arguments, strict=False)
                except JSONDecodeError as e:
                    msg = (
                        f"Received invalid JSON function arguments: "
                        f"{function.arguments} for function {function.name}"
                    )
                    raise ValueError(msg) from e
                if len(args) == 1 and "__arg1" in args:
                    args = args["__arg1"]
                actions.append(
                    OpenAIAssistantAction(
                        tool=function.name,
                        tool_input=args,
                        tool_call_id=tool_call.id,
                        log="",
                        run_id=run.id,
                        thread_id=run.thread_id,
                    ),
                )
            return actions
        run_info = json.dumps(run.dict(), indent=2)
        msg = f"Unexpected run status: {run.status}. Full run info:\n\n{run_info}"
        raise ValueError(msg)

    def _wait_for_run(self, run_id: str, thread_id: str) -> Any:
        in_progress = True
        while in_progress:
            run = self.client.beta.threads.runs.retrieve(run_id, thread_id=thread_id)
            in_progress = run.status in ("in_progress", "queued")
            if in_progress:
                sleep(self.check_every_ms / 1000)
        return run

    async def _aparse_intermediate_steps(
        self,
        intermediate_steps: list[tuple[OpenAIAssistantAction, str]],
    ) -> dict:
        last_action, _ = intermediate_steps[-1]
        run = self._wait_for_run(last_action.run_id, last_action.thread_id)
        required_tool_call_ids = set()
        if run.required_action:
            required_tool_call_ids = {
                tc.id for tc in run.required_action.submit_tool_outputs.tool_calls
            }
        tool_outputs = [
            {"output": str(output), "tool_call_id": action.tool_call_id}
            for action, output in intermediate_steps
            if action.tool_call_id in required_tool_call_ids
        ]
        return {
            "tool_outputs": tool_outputs,
            "run_id": last_action.run_id,
            "thread_id": last_action.thread_id,
        }

    async def _acreate_run(self, input_dict: dict) -> Any:
        params = {
            k: v
            for k, v in input_dict.items()
            if k
            in (
                "instructions",
                "model",
                "tools",
                "additional_instructions",
                "parallel_tool_calls",
                "top_p",
                "temperature",
                "max_completion_tokens",
                "max_prompt_tokens",
                "run_metadata",
            )
        }
        return await self.async_client.beta.threads.runs.create(
            input_dict["thread_id"],
            assistant_id=self.assistant_id,
            **params,
        )

    async def _acreate_thread_and_run(self, input_dict: dict, thread: dict) -> Any:
        params = {
            k: v
            for k, v in input_dict.items()
            if k
            in (
                "instructions",
                "model",
                "tools",
                "parallel_tool_calls",
                "top_p",
                "temperature",
                "max_completion_tokens",
                "max_prompt_tokens",
                "run_metadata",
            )
        }
        return await self.async_client.beta.threads.create_and_run(
            assistant_id=self.assistant_id,
            thread=thread,
            **params,
        )

    async def _aget_response(self, run: Any) -> Any:
                # TODO: 分页

        if run.status == "completed":
            import openai

            major_version = int(openai.version.VERSION.split(".")[0])
            minor_version = int(openai.version.VERSION.split(".")[1])
            version_gte_1_14 = (major_version > 1) or (
                major_version == 1 and minor_version >= 14  # noqa: PLR2004
            )

            messages = await self.async_client.beta.threads.messages.list(
                run.thread_id,
                order="asc",
            )
            new_messages = [msg for msg in messages if msg.run_id == run.id]
            if not self.as_agent:
                return new_messages
            answer: Any = [
                msg_content for msg in new_messages for msg_content in msg.content
            ]
            if all(
                (
                    isinstance(content, openai.types.beta.threads.TextContentBlock)
                    if version_gte_1_14
                    else isinstance(
                        content,
                        openai.types.beta.threads.MessageContentText,  # type: ignore[attr-defined,unused-ignore]
                    )
                )
                for content in answer
            ):
                answer = "\n".join(content.text.value for content in answer)
            return OpenAIAssistantFinish(
                return_values={
                    "output": answer,
                    "thread_id": run.thread_id,
                    "run_id": run.id,
                },
                log="",
                run_id=run.id,
                thread_id=run.thread_id,
            )
        if run.status == "requires_action":
            if not self.as_agent:
                return run.required_action.submit_tool_outputs.tool_calls
            actions = []
            for tool_call in run.required_action.submit_tool_outputs.tool_calls:
                function = tool_call.function
                try:
                    args = json.loads(function.arguments, strict=False)
                except JSONDecodeError as e:
                    msg = (
                        f"Received invalid JSON function arguments: "
                        f"{function.arguments} for function {function.name}"
                    )
                    raise ValueError(msg) from e
                if len(args) == 1 and "__arg1" in args:
                    args = args["__arg1"]
                actions.append(
                    OpenAIAssistantAction(
                        tool=function.name,
                        tool_input=args,
                        tool_call_id=tool_call.id,
                        log="",
                        run_id=run.id,
                        thread_id=run.thread_id,
                    ),
                )
            return actions
        run_info = json.dumps(run.dict(), indent=2)
        msg = f"Unexpected run status: {run.status}. Full run info:\n\n{run_info}"
        raise ValueError(msg)

    async def _await_for_run(self, run_id: str, thread_id: str) -> Any:
        in_progress = True
        while in_progress:
            run = await self.async_client.beta.threads.runs.retrieve(
                run_id,
                thread_id=thread_id,
            )
            in_progress = run.status in ("in_progress", "queued")
            if in_progress:
                await asyncio.sleep(self.check_every_ms / 1000)
        return run

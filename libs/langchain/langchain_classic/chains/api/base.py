"""链，用于发起 API 调用并总结响应以回答问题。"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any
from urllib.parse import urlparse

from langchain_core._api import deprecated
from langchain_core.callbacks import (
    AsyncCallbackManagerForChainRun,
    CallbackManagerForChainRun,
)
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import BasePromptTemplate
from pydantic import Field, model_validator
from typing_extensions import Self

from langchain_classic.chains.api.prompt import API_RESPONSE_PROMPT, API_URL_PROMPT
from langchain_classic.chains.base import Chain
from langchain_classic.chains.llm import LLMChain


def _extract_scheme_and_domain(url: str) -> tuple[str, str]:
    """从给定的 URL 中提取协议 + 域名。

    Args:
        url: 输入的 URL。

    Returns:
        协议和域名组成的 2 元组
    """
    parsed_uri = urlparse(url)
    return parsed_uri.scheme, parsed_uri.netloc


def _check_in_allowed_domain(url: str, limit_to_domains: Sequence[str]) -> bool:
    """检查 URL 是否在允许的域名范围内。

    Args:
        url: 输入的 URL。
        limit_to_domains: 允许的域名列表。

    Returns:
        如果 URL 在允许的域名范围内则返回 `True`，否则返回 `False`。
    """
    scheme, domain = _extract_scheme_and_domain(url)

    for allowed_domain in limit_to_domains:
        allowed_scheme, allowed_domain_ = _extract_scheme_and_domain(allowed_domain)
        if scheme == allowed_scheme and domain == allowed_domain_:
            return True
    return False


try:
    from langchain_community.utilities.requests import TextRequestsWrapper

    @deprecated(
        since="0.2.13",
        removal="2.0.0",
        alternative="langchain.agents.create_agent",
        addendum=(
            "Build new agents with `create_agent` and bind a tool that issues "
            "the HTTP request. See "
            "https://docs.langchain.com/oss/python/langchain/agents"
        ),
    )
    class APIChain(Chain):
        """链，用于发起 API 调用并总结响应以回答问题。

        **安全提示**：此 API 链使用 requests 工具集
            发起 `GET`、`POST`、`PATCH`、`PUT` 和 `DELETE` 请求至 API。

            Exercise care in who is allowed to use this chain. If exposing
            to end users, consider that users will be able to make arbitrary
            requests on behalf of the server hosting the code. For example,
            users could ask the server to make a request to a private API
            that is only accessible from the server.

            通过控制谁可以提交使用此工具集的请求以及其网络访问权限来管理风险。

            See https://docs.langchain.com/oss/python/security-policy for more
            information.

        !!! note
            This class is deprecated. See below for a replacement implementation using
            LangGraph. The benefits of this implementation are:

        - 利用 LLM 工具调用功能，鼓励生成格式正确的 API 请求；
        - 支持 token 级别和步骤级流式输出；
        - 支持检查点和聊天历史记忆；
        - 更易于修改或扩展
            （例如，添加额外工具、结构化响应等）

        安装 LangGraph 的方式如下：

        ```bash
        pip install -U langgraph
        ```

        ```python
        from typing import Annotated, Sequence
        from typing_extensions import TypedDict

        from langchain_classic.chains.api.prompt import API_URL_PROMPT
        from langchain_community.agent_toolkits.openapi.toolkit import RequestsToolkit
        from langchain_community.utilities.requests import TextRequestsWrapper
        from langchain_core.messages import BaseMessage
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_openai import ChatOpenAI
        from langchain_core.runnables import RunnableConfig
        from langgraph.graph import END, StateGraph
        from langgraph.graph.message import add_messages
        from langgraph.prebuilt.tool_node import ToolNode

        # NOTE: There are inherent risks in giving models discretion
        # to execute real-world actions. We must "opt-in" to these
        # risks by setting allow_dangerous_request=True to use these tools.
        # This can be dangerous for calling unwanted requests. Please make
        # sure your custom OpenAPI spec (yaml) is safe and that permissions
        # associated with the tools are narrowly-scoped.
        ALLOW_DANGEROUS_REQUESTS = True

        # Subset of spec for https://jsonplaceholder.typicode.com
        api_spec = \"\"\"
        openapi: 3.0.0
        info:
          title: JSONPlaceholder API
          version: 1.0.0
        servers:
          - url: https://jsonplaceholder.typicode.com
        paths:
          /posts:
            get:
              summary: Get posts
              parameters: &id001
                - name: _limit
                  in: query
                  required: false
                  schema:
                    type: integer
                  example: 2
                  description: Limit the number of results
        \"\"\"

        model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        toolkit = RequestsToolkit(
            requests_wrapper=TextRequestsWrapper(headers={}),  # no auth required
            allow_dangerous_requests=ALLOW_DANGEROUS_REQUESTS,
        )
        tools = toolkit.get_tools()

        api_request_chain = (
            API_URL_PROMPT.partial(api_docs=api_spec)
            | model.bind_tools(tools, tool_choice="any")
        )

        class ChainState(TypedDict):
            \"\"\"LangGraph state.\"\"\"

            messages: Annotated[Sequence[BaseMessage], add_messages]


        async def acall_request_chain(state: ChainState, config: RunnableConfig):
            last_message = state["messages"][-1]
            response = await api_request_chain.ainvoke(
                {"question": last_message.content}, config
            )
            return {"messages": [response]}

        async def acall_model(state: ChainState, config: RunnableConfig):
            response = await model.ainvoke(state["messages"], config)
            return {"messages": [response]}

        graph_builder = StateGraph(ChainState)
        graph_builder.add_node("call_tool", acall_request_chain)
        graph_builder.add_node("execute_tool", ToolNode(tools))
        graph_builder.add_node("call_model", acall_model)
        graph_builder.set_entry_point("call_tool")
        graph_builder.add_edge("call_tool", "execute_tool")
        graph_builder.add_edge("execute_tool", "call_model")
        graph_builder.add_edge("call_model", END)
        chain = graph_builder.compile()
        ```

        ```python
        example_query = "Fetch the top two posts. What are their titles?"

        events = chain.astream(
            {"messages": [("user", example_query)]},
            stream_mode="values",
        )
        async for event in events:
            event["messages"][-1].pretty_print()
        ```
        """

        api_request_chain: LLMChain

        api_answer_chain: LLMChain

        requests_wrapper: TextRequestsWrapper = Field(exclude=True)

        api_docs: str

        question_key: str = "question"

        output_key: str = "output"

        limit_to_domains: Sequence[str] | None = Field(default_factory=list)
        """用于限制 API 链可访问的域名范围。

        * For example, to limit to just the domain `https://www.example.com`, set
            `limit_to_domains=["https://www.example.com"]`.
        * The default value is an empty tuple, which means that no domains are
            allowed by default. By design this will raise an error on instantiation.
        * Use a None if you want to allow all domains by default -- this is not
            recommended for security reasons, as it would allow malicious users to
            make requests to arbitrary URLS including internal APIs accessible from
            the server.
        """

        @property
        def input_keys(self) -> list[str]:
            """期望输入键。"""
            return [self.question_key]

        @property
        def output_keys(self) -> list[str]:
            """期望输出键。"""
            return [self.output_key]

        @model_validator(mode="after")
        def validate_api_request_prompt(self) -> Self:
            """检查 API 请求提示词是否包含正确的变量。"""
            input_vars = self.api_request_chain.prompt.input_variables
            expected_vars = {"question", "api_docs"}
            if set(input_vars) != expected_vars:
                msg = f"Input variables should be {expected_vars}, got {input_vars}"
                raise ValueError(msg)
            return self

        @model_validator(mode="before")
        @classmethod
        def validate_limit_to_domains(cls, values: dict) -> Any:
            """检查允许的域名是否有效。"""
                        # 此检查必须为 pre=True 检查，以确保当未提供时，None 不会被设为
            # limit_to_domains。
                        # 此检查必须为 pre=True 检查，以确保当未提供时，None 不会被设为
            # limit_to_domains。
            if "limit_to_domains" not in values:
                msg = (
                    "You must specify a list of domains to limit access using "
                    "`limit_to_domains`"
                )
                raise ValueError(msg)
            if (
                not values["limit_to_domains"]
                and values["limit_to_domains"] is not None
            ):
                msg = (
                    "Please provide a list of domains to limit access using "
                    "`limit_to_domains`."
                )
                raise ValueError(msg)
            return values

        @model_validator(mode="after")
        def validate_api_answer_prompt(self) -> Self:
            """检查 API 回答提示词是否包含正确的变量。"""
            input_vars = self.api_answer_chain.prompt.input_variables
            expected_vars = {"question", "api_docs", "api_url", "api_response"}
            if set(input_vars) != expected_vars:
                msg = f"Input variables should be {expected_vars}, got {input_vars}"
                raise ValueError(msg)
            return self

        def _call(
            self,
            inputs: dict[str, Any],
            run_manager: CallbackManagerForChainRun | None = None,
        ) -> dict[str, str]:
            _run_manager = run_manager or CallbackManagerForChainRun.get_noop_manager()
            question = inputs[self.question_key]
            api_url = self.api_request_chain.predict(
                question=question,
                api_docs=self.api_docs,
                callbacks=_run_manager.get_child(),
            )
            _run_manager.on_text(api_url, color="green", end="\n", verbose=self.verbose)
            api_url = api_url.strip()
            if self.limit_to_domains and not _check_in_allowed_domain(
                api_url,
                self.limit_to_domains,
            ):
                msg = (
                    f"{api_url} is not in the allowed domains: {self.limit_to_domains}"
                )
                raise ValueError(msg)
            api_response = self.requests_wrapper.get(api_url)
            _run_manager.on_text(
                str(api_response),
                color="yellow",
                end="\n",
                verbose=self.verbose,
            )
            answer = self.api_answer_chain.predict(
                question=question,
                api_docs=self.api_docs,
                api_url=api_url,
                api_response=api_response,
                callbacks=_run_manager.get_child(),
            )
            return {self.output_key: answer}

        async def _acall(
            self,
            inputs: dict[str, Any],
            run_manager: AsyncCallbackManagerForChainRun | None = None,
        ) -> dict[str, str]:
            _run_manager = (
                run_manager or AsyncCallbackManagerForChainRun.get_noop_manager()
            )
            question = inputs[self.question_key]
            api_url = await self.api_request_chain.apredict(
                question=question,
                api_docs=self.api_docs,
                callbacks=_run_manager.get_child(),
            )
            await _run_manager.on_text(
                api_url,
                color="green",
                end="\n",
                verbose=self.verbose,
            )
            api_url = api_url.strip()
            if self.limit_to_domains and not _check_in_allowed_domain(
                api_url,
                self.limit_to_domains,
            ):
                msg = (
                    f"{api_url} is not in the allowed domains: {self.limit_to_domains}"
                )
                raise ValueError(msg)
            api_response = await self.requests_wrapper.aget(api_url)
            await _run_manager.on_text(
                str(api_response),
                color="yellow",
                end="\n",
                verbose=self.verbose,
            )
            answer = await self.api_answer_chain.apredict(
                question=question,
                api_docs=self.api_docs,
                api_url=api_url,
                api_response=api_response,
                callbacks=_run_manager.get_child(),
            )
            return {self.output_key: answer}

        @classmethod
        def from_llm_and_api_docs(
            cls,
            llm: BaseLanguageModel,
            api_docs: str,
            headers: dict | None = None,
            api_url_prompt: BasePromptTemplate = API_URL_PROMPT,
            api_response_prompt: BasePromptTemplate = API_RESPONSE_PROMPT,
            limit_to_domains: Sequence[str] | None = (),
            **kwargs: Any,
        ) -> APIChain:
            """仅通过 LLM 和 API 文档加载链。"""
            get_request_chain = LLMChain(llm=llm, prompt=api_url_prompt)
            requests_wrapper = TextRequestsWrapper(headers=headers)
            get_answer_chain = LLMChain(llm=llm, prompt=api_response_prompt)
            return cls(
                api_request_chain=get_request_chain,
                api_answer_chain=get_answer_chain,
                requests_wrapper=requests_wrapper,
                api_docs=api_docs,
                limit_to_domains=limit_to_domains,
                **kwargs,
            )

        @property
        def _chain_type(self) -> str:
            return "api_chain"

except ImportError:

    class APIChain:  # type: ignore[no-redef]
        """若未安装 langchain_community，则抛出 ImportError。"""

        def __init__(self, *_: Any, **__: Any) -> None:
            """若未安装 langchain_community，则抛出 ImportError。"""
            msg = (
                "To use the APIChain, you must install the langchain_community package."
                "pip install langchain_community"
            )
            raise ImportError(msg)

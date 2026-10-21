"""加载智能体。"""

import contextlib
from collections.abc import Sequence
from typing import Any

from langchain_core._api import deprecated
from langchain_core.callbacks import BaseCallbackManager
from langchain_core.language_models import BaseLanguageModel
from langchain_core.tools import BaseTool

from langchain_classic._api.deprecation import AGENT_DEPRECATION_WARNING
from langchain_classic.agents.agent import AgentExecutor
from langchain_classic.agents.agent_types import AgentType
from langchain_classic.agents.loading import load_agent
from langchain_classic.agents.types import AGENT_TO_CLASS


@deprecated(
    "0.1.0",
    message=AGENT_DEPRECATION_WARNING,
    removal="2.0.0",
)
def initialize_agent(
    tools: Sequence[BaseTool],
    llm: BaseLanguageModel,
    agent: AgentType | None = None,
    callback_manager: BaseCallbackManager | None = None,
    agent_path: str | None = None,
    agent_kwargs: dict | None = None,
    *,
    tags: Sequence[str] | None = None,
    **kwargs: Any,
) -> AgentExecutor:
    """根据工具和 LLM 加载一个智能体执行器。

    !!! warning

        This function is no deprecated in favor of
        [`create_agent`][langchain.agents.create_agent] from the `langchain`
        package, which provides a more flexible agent factory with middleware
        support, structured output, and integration with LangGraph.

        有关迁移指引，请参阅
        [迁移到 langchain v1](https://docs.langchain.com/oss/python/migrate/langchain-v1)
        和
        [从 AgentExecutor 迁移](https://python.langchain.com/docs/how_to/migrate_agent/)。

    Args:
        tools: 智能体可访问的工具列表。 llm: 用作智能体的语言模型。 agent:
        要使用的智能体类型。如果 `None` 且 agent_path 也为 `None`，将默认使用
        AgentType.ZERO_SHOT_REACT_DESCRIPTION。 callback_manager:
        要使用的回调管理器。若未提供，则使用全局回调管理器。 agent_path:
        要使用的序列化智能体路径。如果 `None` 且 agent 也为 `None`， 将默认使用
        AgentType.ZERO_SHOT_REACT_DESCRIPTION。 agent_kwargs:
        传递给底层智能体的额外关键字参数。 tags: 应用于追踪运行的标签。 kwargs:
        传递给智能体执行器的额外关键字参数。

    Returns:
        一个智能体执行器。

    Raises:
        ValueError: 如果同时指定了 `agent` 和 `agent_path`。
        ValueError: 如果 `agent` 不是有效的智能体类型。
        ValueError: 如果 `agent` 和 `agent_path` 均为 `None`。
    """
    tags_ = list(tags) if tags else []
    if agent is None and agent_path is None:
        agent = AgentType.ZERO_SHOT_REACT_DESCRIPTION
    if agent is not None and agent_path is not None:
        msg = (
            "Both `agent` and `agent_path` are specified, "
            "but at most only one should be."
        )
        raise ValueError(msg)
    if agent is not None:
        if agent not in AGENT_TO_CLASS:
            msg = (
                f"Got unknown agent type: {agent}. "
                f"Valid types are: {AGENT_TO_CLASS.keys()}."
            )
            raise ValueError(msg)
        tags_.append(agent.value if isinstance(agent, AgentType) else agent)
        agent_cls = AGENT_TO_CLASS[agent]
        agent_kwargs = agent_kwargs or {}
        agent_obj = agent_cls.from_llm_and_tools(
            llm,
            tools,
            callback_manager=callback_manager,
            **agent_kwargs,
        )
    elif agent_path is not None:
        agent_obj = load_agent(
            agent_path,
            llm=llm,
            tools=tools,
            callback_manager=callback_manager,
        )
        with contextlib.suppress(NotImplementedError):
                        # TODO: 直接从序列化对象中添加标签。
            tags_.append(agent_obj._agent_type)  # noqa: SLF001
    else:
        msg = (
            "Somehow both `agent` and `agent_path` are None, this should never happen."
        )
        raise ValueError(msg)
    return AgentExecutor.from_agent_and_tools(
        agent=agent_obj,
        tools=tools,
        callback_manager=callback_manager,
        tags=tags_,
        **kwargs,
    )

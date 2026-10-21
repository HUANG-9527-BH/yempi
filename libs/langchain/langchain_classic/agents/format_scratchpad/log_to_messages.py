from langchain_core.agents import AgentAction
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage


def format_log_to_messages(
    intermediate_steps: list[tuple[AgentAction, str]],
    template_tool_response: str = "{observation}",
) -> list[BaseMessage]:
    """构建用于让智能体继续其思维过程的草稿板。

    Args:
        intermediate_steps: 包含 AgentAction 和观察字符串元组的列表。
        template_tool_response: 用于格式化观察结果的模板。
            默认为 `"{observation}"`。

    Returns:
        草稿板。
    """
    thoughts: list[BaseMessage] = []
    for action, observation in intermediate_steps:
        thoughts.append(AIMessage(content=action.log))
        human_message = HumanMessage(
            content=template_tool_response.format(observation=observation),
        )
        thoughts.append(human_message)
    return thoughts

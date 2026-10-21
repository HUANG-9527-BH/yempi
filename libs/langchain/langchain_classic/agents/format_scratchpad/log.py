from langchain_core.agents import AgentAction


def format_log_to_str(
    intermediate_steps: list[tuple[AgentAction, str]],
    observation_prefix: str = "Observation: ",
    llm_prefix: str = "Thought: ",
) -> str:
    """构建用于让智能体继续其思维过程的临时笔记板。

    Args:
        intermediate_steps: 包含 AgentAction 和观察字符串元组的列表。
        observation_prefix: 要附加到观察结果前的前缀。
        llm_prefix: 要附加到 LLM 调用前的前缀。

    Returns:
        临时笔记板。
    """
    thoughts = ""
    for action, observation in intermediate_steps:
        thoughts += action.log
        thoughts += f"\n{observation_prefix}{observation}\n{llm_prefix}"
    return thoughts

from typing import Any


def get_prompt_input_key(inputs: dict[str, Any], memory_variables: list[str]) -> str:
    """获取提示词输入键。

    Args:
        inputs: Dict[str, Any]
        memory_variables: List[str]

    Returns:
        一个提示词输入键。
    """
        # "stop" 是一个特殊键，可以作为输入传递，但不会用于格式化提示词。
        # 格式化提示词。
    prompt_input_keys = list(set(inputs).difference([*memory_variables, "stop"]))
    if len(prompt_input_keys) != 1:
        msg = f"One input key expected got {prompt_input_keys}"
        raise ValueError(msg)
    return prompt_input_keys[0]

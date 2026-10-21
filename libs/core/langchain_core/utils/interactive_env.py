"""用于处理交互式环境的工具。"""

import sys


def is_interactive_env() -> bool:
    """判断是否在 IPython 或 Jupyter 中运行。

    Returns:
        在交互式环境中运行时为 `True`，否则为 `False`。
    """
    return hasattr(sys, "ps2")

def is_interactive_env() -> bool:
    """判断是否在 IPython 或 Jupyter 中运行。"""
    import sys

    return hasattr(sys, "ps2")

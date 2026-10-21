import inspect
from typing import cast


def is_caller_internal(depth: int = 2) -> bool:
    """返回调用者在该函数的 `depth` 层次是否为内部调用。"""
    try:
        frame = inspect.currentframe()
    except AttributeError:
        return False
    if frame is None:
        return False
    try:
        for _ in range(depth):
            frame = frame.f_back
            if frame is None:
                return False
                # 直接从帧的全局变量中访问模块名称
        module_globals = frame.f_globals
        caller_module_name = cast("str", module_globals.get("__name__", ""))
        return caller_module_name.startswith("langchain")
    finally:
        del frame

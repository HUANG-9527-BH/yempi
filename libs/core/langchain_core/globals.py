"""适用于 LangChain 全局的值和配置。"""

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from langchain_core.caches import BaseCache


# 不要直接使用这些值！
# 仅通过下方的 `get_<X>()` 和 `set_<X>()` 使用它们，
# 否则在与其他使用这些全局设置的代码交互时，你的代码行为可能出乎意料：
# https://github.com/langchain-ai/langchain/pull/11311#issuecomment-1743780004
_verbose: bool = False
_debug: bool = False
_llm_cache: Optional["BaseCache"] = None


def set_verbose(value: bool) -> None:  # noqa: FBT001
    """设置 `verbose` 全局设置的新值。

    Args:
        value: `verbose` 全局设置的新值。
    """
    global _verbose  # noqa: PLW0603
    _verbose = value


def get_verbose() -> bool:
    """获取 `verbose` 全局设置的值。

    Returns:
        `verbose` 全局设置的值。
    """
    return _verbose


def set_debug(value: bool) -> None:  # noqa: FBT001
    """设置 `debug` 全局设置的新值。

    Args:
        value: `debug` 全局设置的新值。
    """
    global _debug  # noqa: PLW0603
    _debug = value


def get_debug() -> bool:
    """获取 `debug` 全局设置的值。

    Returns:
        `debug` 全局设置的值。
    """
    return _debug


def set_llm_cache(value: Optional["BaseCache"]) -> None:
    """设置新的 LLM 缓存，覆盖之前的值（如果存在）。

    Args:
        value: 要使用的新的 LLM 缓存。如果为 `None`，则禁用 LLM 缓存。
    """
    global _llm_cache  # noqa: PLW0603
    _llm_cache = value


def get_llm_cache() -> Optional["BaseCache"]:
    """获取 `llm_cache` 全局设置的值。

    Returns:
        `llm_cache` 全局设置的值。
    """
    return _llm_cache

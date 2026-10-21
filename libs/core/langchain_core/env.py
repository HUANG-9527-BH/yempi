"""用于获取运行时环境信息的工具。"""

import platform
from functools import lru_cache

from langchain_core import __version__


@lru_cache(maxsize=1)
def get_runtime_environment() -> dict[str, str]:
    """获取 LangChain 运行时环境的信息。

    Returns:
        包含运行时环境信息的字典。
    """
    return {
        "library_version": __version__,
        "library": "langchain-core",
        "platform": platform.platform(),
        "runtime": "python",
        "runtime_version": platform.python_version(),
    }

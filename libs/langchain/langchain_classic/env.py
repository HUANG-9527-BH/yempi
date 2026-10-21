import platform
from functools import lru_cache


@lru_cache(maxsize=1)
def get_runtime_environment() -> dict:
    """获取有关 LangChain 运行时环境的信息。"""
        # 延迟导入以避免循环导入
    from langchain_classic import __version__

    return {
        "library_version": __version__,
        "library": "langchain-classic",
        "platform": platform.platform(),
        "runtime": "python",
        "runtime_version": platform.python_version(),
    }

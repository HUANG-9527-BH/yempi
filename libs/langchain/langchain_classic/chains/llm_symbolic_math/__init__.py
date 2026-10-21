def __getattr__(_: str = "") -> None:
    """导入时抛出错误，因为已弃用。"""
    msg = (
        "This module has been moved to langchain-experimental. "
        "For more details: https://github.com/langchain-ai/langchain/discussions/11352."
        "To access this code, install it with `pip install langchain-experimental`."
        "`from langchain_experimental.llm_symbolic_math.base "
        "import LLMSymbolicMathChain`"
    )
    raise AttributeError(msg)

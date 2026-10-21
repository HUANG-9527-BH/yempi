"""回调处理器在新 LLM token 时流式输出到 stdout。"""

from langchain_core.callbacks import StreamingStdOutCallbackHandler

__all__ = ["StreamingStdOutCallbackHandler"]

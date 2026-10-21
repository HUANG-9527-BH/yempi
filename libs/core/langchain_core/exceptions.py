"""LangChain 的自定义 **异常**。"""

from enum import Enum
from typing import Any


class LangChainException(Exception):  # noqa: N818
    """通用 LangChain 异常。"""


class TracerException(LangChainException):
    """追踪模块中异常的基类。"""


class OutputParserException(ValueError, LangChainException):  # noqa: N818
    """输出解析器应抛出以表示解析错误的异常。

    This exists to differentiate parsing errors from other code or execution errors
    that also may arise inside the output parser.

    `OutputParserException` will be available to catch and handle in ways to fix the
    parsing error, while other errors will be raised.
    """

    def __init__(
        self,
        error: Any,
        observation: str | None = None,
        llm_output: str | None = None,
        send_to_llm: bool = False,  # noqa: FBT001,FBT002
    ):
        """创建一个 `OutputParserException`。

        Args:
            error: 正在重新抛出的错误或错误消息。 observation:
            错误的字符串解释，可传递给模型以尝试修复问题。 llm_output:
            出错的字符串模型输出。

            send_to_llm: 在抛出 `OutputParserException` 后，是否将 observation 和 llm_output 发送回智能体。
                这为驱动智能体的底层模型提供上下文，使其知道前一次输出结构不正确，从而希望它能更新输出为正确格式。

                ValueError: 如果 `send_to_llm` 为 `True`，但未提供 observation 或
                `llm_output`。

        Raises:
            Used in JS; not Py (yet)
        """
        if isinstance(error, str):
            error = create_message(
                message=error, error_code=ErrorCode.OUTPUT_PARSING_FAILURE
            )

        super().__init__(error)
        if send_to_llm and (observation is None or llm_output is None):
            msg = (
                "Arguments 'observation' & 'llm_output'"
                " are required if 'send_to_llm' is True"
            )
            raise ValueError(msg)
        self.observation = observation
        self.llm_output = llm_output
        self.send_to_llm = send_to_llm


class ModelError(LangChainException):
    """与模型调用失败相关的基础异常。

    Subclasses correspond to conditions that model providers report consistently,
    keyed to the HTTP status they surface it with, so the same condition maps to
    the same exception type regardless of provider.

    Provider integrations raise subclasses that also inherit from the provider
    SDK's own exception type, so code catching either continues to work.
    """

    is_retryable = False
    """是否重试相同模型请求可能成功。"""


class ModelAuthenticationError(ModelError):
    """当模型提供商认证失败时抛出的异常（HTTP 401）。"""


class ModelPermissionDeniedError(ModelError):
    """当凭证缺少请求权限时抛出的异常（HTTP 403）。"""


class ModelInvalidRequestError(ModelError):
    """当提供商拒绝无效请求时抛出的异常（例如 HTTP 400）。"""


class ModelNotFoundError(ModelError):
    """当请求的模型无法找到时抛出的异常（HTTP 404）。"""


class ModelRateLimitError(ModelError):
    """当模型提供商的速率限制被超出时抛出的异常（HTTP 429）。"""

    is_retryable = True


class ModelAPIError(ModelError):
    """当模型提供商报告服务器故障时抛出的异常（HTTP 5xx）。"""

    is_retryable = True


class ModelConnectionError(ModelError):
    """当无法连接到模型提供商时抛出的异常。"""

    is_retryable = True


class ModelTimeoutError(ModelError):
    """当模型请求超时时抛出的异常。"""

    is_retryable = True


class ContextOverflowError(ModelError):
    """当输入超过模型的上下文限制时抛出的异常。

    This exception is raised by chat models when the input tokens exceed
    the maximum context window supported by the model.
    """


class ErrorCode(Enum):
    """错误代码。"""

    INVALID_PROMPT_INPUT = "INVALID_PROMPT_INPUT"
    INVALID_TOOL_RESULTS = "INVALID_TOOL_RESULTS"  # Used in JS; not Py (yet)
    MESSAGE_COERCION_FAILURE = "MESSAGE_COERCION_FAILURE"
    MODEL_AUTHENTICATION = "MODEL_AUTHENTICATION"  # Used in JS; not Py (yet)
    MODEL_NOT_FOUND = "MODEL_NOT_FOUND"  # Used in JS; not Py (yet)
    MODEL_RATE_LIMIT = "MODEL_RATE_LIMIT"  # 用于 JS；Python 尚未支持
    OUTPUT_PARSING_FAILURE = "OUTPUT_PARSING_FAILURE"


def create_message(*, message: str, error_code: ErrorCode) -> str:
    """创建一条包含指向 LangChain 排错指南链接的消息。

    Args:
        message: 要显示的消息。
        error_code: 要显示的错误代码。

    Returns:
        包含排错链接的完整消息。

    Example:
        ```python
        create_message(
            message="Failed to parse output",
            error_code=ErrorCode.OUTPUT_PARSING_FAILURE,
        )
        "Failed to parse output. For troubleshooting, visit: ..."
        ```
    """
    return (
        f"{message}\n"
        "For troubleshooting, visit: https://docs.langchain.com/oss/python/langchain"
        f"/errors/{error_code.value} "
    )

"""TypeSafe 集成的 HTTP 响应处理和错误。"""

from __future__ import annotations

import math
import time
from email.utils import parsedate_to_datetime
from http import HTTPStatus
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import httpx2
from langchain_core.exceptions import (
    ModelAPIError,
    ModelAuthenticationError,
    ModelConnectionError,
    ModelInvalidRequestError,
    ModelNotFoundError,
    ModelPermissionDeniedError,
    ModelRateLimitError,
    ModelTimeoutError,
)
from pydantic import ValidationError
from typing_extensions import override

from langchain_typesafe.types import ClassifierResponse

_REQUEST_ID_HEADER = "x-typesafe-request-id"
_RETRY_AFTER_HEADER = "retry-after"
_RETRY_AFTER_MS_HEADER = "retry-after-ms"
_SAFE_STATUS_MESSAGES = {529: "Overloaded"}


class TypeSafeError(Exception):
    """由 TypeSafe 集成引发的错误的基异常。"""


class TypeSafeAPIError(TypeSafeError):
    """包含其正文和请求元数据的不成功 HTTP 响应。

    Attributes:
        status: HTTP 响应状态码。
        body: 解析后的 JSON 错误正文、纯文本响应或 `None`。
        headers: HTTP 响应头。
        endpoint: 不含凭证、查询或片段的请求方法和 URL。
        request_id: `x-typesafe-request-id` 响应头的值。

    The body and headers are available for programmatic error handling but deliberately
    excluded from `str` and `repr` to avoid exposing classified state or credentials in
    logs and tracebacks.
    """

    def __init__(
        self,
        status: int,
        body: Any,
        headers: httpx2.Headers,
        message: str | None = None,
        endpoint: str | None = None,
    ) -> None:
        """从 TypeSafe HTTP 响应创建 API 错误。

        Args:
            status: HTTP 响应状态码。
            body: 解析后的 JSON 正文、纯文本响应或 `None`。
            headers: HTTP 响应头。
            message: 可选的安全消息覆盖，不包含响应数据。
            endpoint: 当可用时，经过清理的请求方法和 URL。
        """
        super().__init__(status, body, headers, message, endpoint)
        self.status = status
        self.body = body
        self.headers = headers
        self.endpoint = endpoint
        self._message = message

    @property
    def status_code(self) -> int:
        """别名 `status`，与常见的 HTTP 异常接口保持一致。"""
        return self.status

    @property
    def request_id(self) -> str | None:
        """从响应头中返回 TypeSafe 请求 ID，当存在时。"""
        return self.headers.get(_REQUEST_ID_HEADER)

    @override
    def __str__(self) -> str:
        """描述失败情况，但不包含其响应正文或头信息。"""
        try:
            reason = HTTPStatus(self.status).phrase
        except ValueError:
            reason = _SAFE_STATUS_MESSAGES.get(self.status, "API request failed")
        detail = self._message or reason
        message = f"{self.status} {detail}"
        if self.endpoint is not None:
            message = f"{self.endpoint}: {message}"
        if self.request_id is not None:
            message = f"{message} (request_id={self.request_id})"
        return message

    @override
    def __repr__(self) -> str:
        """表示错误，但不包含其响应正文或头信息。"""
        return f"{type(self).__name__}({str(self)!r})"


class TypeSafeBadRequestError(TypeSafeAPIError, ModelInvalidRequestError):
    """TypeSafe 请求无效（HTTP 400）。"""


class TypeSafeAuthenticationError(TypeSafeAPIError, ModelAuthenticationError):
    """TypeSafe 认证失败（HTTP 401）。"""


class TypeSafePermissionDeniedError(TypeSafeAPIError, ModelPermissionDeniedError):
    """TypeSafe 凭证无法执行该请求（HTTP 403）。"""


class TypeSafeNotFoundError(TypeSafeAPIError, ModelNotFoundError):
    """请求的 TypeSafe 资源或模型未找到（HTTP 404）。"""


class TypeSafeUnprocessableEntityError(TypeSafeAPIError, ModelInvalidRequestError):
    """TypeSafe 在验证期间拒绝了请求正文（HTTP 422）。"""


class TypeSafeRateLimitError(TypeSafeAPIError, ModelRateLimitError):
    """TypeSafe 的速率限制被超出（HTTP 429）。

    Attributes:
        retry_after_ms: 服务器请求的延迟时间（毫秒），当响应不包含有效重试头时为
        `None`。
    """

    def __init__(
        self,
        status: int,
        body: Any,
        headers: httpx2.Headers,
        message: str | None = None,
        endpoint: str | None = None,
    ) -> None:
        """创建一个速率限制错误并解析其重试延迟。

        Args:
            status: HTTP 响应状态码。
            body: 解析后的 JSON 正文、原始响应文本或 `None`。
            headers: HTTP 响应头。
            message: 可选的安全消息覆盖。
            endpoint: 清理后的请求方法和 URL，若可用时提供。
        """
        super().__init__(status, body, headers, message, endpoint)
        self.retry_after_ms = _parse_retry_after(headers)


class TypeSafeInternalServerError(TypeSafeAPIError, ModelAPIError):
    """TypeSafe 无法处理请求（HTTP 5xx）。"""


class TypeSafeAPIConnectionError(TypeSafeError, ModelConnectionError, ConnectionError):
    """TypeSafe 请求失败且未收到 HTTP 响应。"""


class TypeSafeAPITimeoutError(
    TypeSafeAPIConnectionError,
    ModelTimeoutError,
    TimeoutError,
):
    """TypeSafe 请求超出了其配置的超时时间。

    Attributes:
        timeout: 同步或异步 HTTP 客户端使用的超时设置。
    """

    def __init__(self, timeout: float | httpx2.Timeout) -> None:
        """创建一个超时错误。

        Args:
            timeout: 用于失败请求的超时设置。
        """
        super().__init__(timeout)
        self.timeout = timeout

    @override
    def __str__(self) -> str:
        """返回配置的超时，不包含请求或凭据数据。"""
        return f"Request timed out (timeout={self.timeout})."

    @override
    def __repr__(self) -> str:
        """使用其安全格式化消息表示超时。"""
        return f"{type(self).__name__}({str(self)!r})"


class TypeSafeAPIResponseValidationError(TypeSafeAPIError):
    """成功响应缺失或包含无效的必需数据。

    Attributes:
        field_path: 第一个验证失败字段的点号路径。
    """

    def __init__(
        self,
        status: int,
        body: Any,
        headers: httpx2.Headers,
        field_path: str,
        endpoint: str | None = None,
    ) -> None:
        """创建一个响应验证错误。

        Args:
            status: 成功的 HTTP 响应状态码。
            body: 解析后的 JSON 正文或原始响应文本。
            headers: HTTP 响应头。
            field_path: 第一个无效字段的点号路径。
            endpoint: 清理后的请求方法和 URL，若可用时提供。
        """
        self.field_path = field_path
        super().__init__(
            status,
            body,
            headers,
            f"Invalid response data at {field_path!r}.",
            endpoint,
        )
        self.args = (status, body, headers, field_path, endpoint)


def _parse_retry_after(headers: httpx2.Headers) -> float | None:
    for name, multiplier in (
        (_RETRY_AFTER_MS_HEADER, 1.0),
        (_RETRY_AFTER_HEADER, 1000.0),
    ):
        raw = headers.get(name)
        if raw is None:
            continue
        try:
            value = float(raw.strip() or "0")
        except ValueError:
            if name == _RETRY_AFTER_HEADER:
                try:
                    delay = (
                        parsedate_to_datetime(raw).timestamp() - time.time()
                    ) * 1000
                except (OverflowError, TypeError, ValueError):
                    continue
                return max(0.0, delay)
            continue
        if math.isfinite(value) and value >= 0:
            delay = value * multiplier
            if math.isfinite(delay):
                return delay
        if name == _RETRY_AFTER_HEADER:
            return None
    return None


def _response_body(response: httpx2.Response) -> Any:
    if not response.content:
        return None
    try:
        return response.json()
    except ValueError:
        return response.text


def _response_endpoint(response: httpx2.Response) -> str | None:
    try:
        request = response.request
    except RuntimeError:
        return None
    parts = urlsplit(str(request.url))
    hostname = parts.hostname
    if hostname is None:
        return None
    host = f"[{hostname}]" if ":" in hostname else hostname
    try:
        port = parts.port
    except ValueError:
        port = None
    netloc = f"{host}:{port}" if port is not None else host
    url = urlunsplit((parts.scheme, netloc, parts.path, "", ""))
    return f"{request.method} {url}"


_STATUS_ERROR_TYPES: dict[int, type[TypeSafeAPIError]] = {
    400: TypeSafeBadRequestError,
    401: TypeSafeAuthenticationError,
    403: TypeSafePermissionDeniedError,
    404: TypeSafeNotFoundError,
    422: TypeSafeUnprocessableEntityError,
    429: TypeSafeRateLimitError,
}


def _api_error(response: httpx2.Response) -> TypeSafeAPIError:
    error_type = _STATUS_ERROR_TYPES.get(
        response.status_code,
        TypeSafeInternalServerError
        if response.status_code >= HTTPStatus.INTERNAL_SERVER_ERROR
        else TypeSafeAPIError,
    )
    return error_type(
        response.status_code,
        _response_body(response),
        response.headers,
        endpoint=_response_endpoint(response),
    )


def parse_response(response: httpx2.Response) -> ClassifierResponse:
    """验证 HTTP 响应并将其转换为分类响应。

    Args:
        response: TypeSafe API 返回的原始 HTTP 响应。

    Returns:
        Validated classification answers and metadata. The TypeSafe request ID is
        copied from the response headers when present.

    Raises:
        TypeSafeAPIError: If TypeSafe returns an unsuccessful status code. Specific
            statuses use subclasses that also inherit from LangChain model errors.
        TypeSafeAPIResponseValidationError: If a successful response is not valid JSON
            or does not match the expected response schema.
    """
    if not response.is_success:
        raise _api_error(response)
    endpoint = _response_endpoint(response)
    body = _response_body(response)
    try:
        parsed = ClassifierResponse.model_validate(body)
    except ValidationError as error:
        location = error.errors()[0].get("loc", ())
        field_path = ".".join(str(item) for item in location) or "response"
        raise TypeSafeAPIResponseValidationError(
            response.status_code,
            body,
            response.headers,
            field_path,
            endpoint,
        ) from error
    return parsed.model_copy(
        update={"request_id": response.headers.get(_REQUEST_ID_HEADER)}
    )


__all__ = [
    "TypeSafeAPIConnectionError",
    "TypeSafeAPIError",
    "TypeSafeAPIResponseValidationError",
    "TypeSafeAPITimeoutError",
    "TypeSafeAuthenticationError",
    "TypeSafeBadRequestError",
    "TypeSafeError",
    "TypeSafeInternalServerError",
    "TypeSafeNotFoundError",
    "TypeSafePermissionDeniedError",
    "TypeSafeRateLimitError",
    "TypeSafeUnprocessableEntityError",
    "parse_response",
]

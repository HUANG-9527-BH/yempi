"""TypeSafe 分类的 LangChain Runnable。"""

from __future__ import annotations

import logging
from typing import Any

import httpx2
from langchain_core._api import beta
from langchain_core.runnables import RunnableConfig, RunnableSerializable
from langchain_core.runnables.config import ensure_config
from langchain_core.utils import from_env, secret_from_env
from langsmith.run_helpers import get_current_run_tree
from pydantic import (
    ConfigDict,
    Field,
    JsonValue,
    SecretStr,
    field_validator,
    model_validator,
)
from typing_extensions import Self, override

from langchain_typesafe._state import serialize_state
from langchain_typesafe._version import __version__
from langchain_typesafe.client import (
    TypeSafeAPIConnectionError,
    TypeSafeAPITimeoutError,
    parse_response,
)
from langchain_typesafe.types import (
    ClassifierRequest,
    ClassifierResponse,
)

_DEFAULT_BASE_URL = "https://api.typesafe.ai"
_DEFAULT_MODEL = "jev-latest"
_DEFAULT_TIMEOUT = 30.0
_LS_PROVIDER = "typesafe"

logger = logging.getLogger(__name__)


@beta()
class TypeSafeClassifier(RunnableSerializable[ClassifierRequest, ClassifierResponse]):
    """使用 TypeSafe 对 JSON 兼容状态进行分类。

    `TypeSafeClassifier` is a LangChain `Runnable` for asking one or more typed
    questions about text or structured state. A single request can combine binary
    `Noul` judgments, categorical `Choice` classifications, and ordinal `Score`
    evaluations. The response preserves probabilities, confidence, and token usage so
    application code can decide whether to act, route, or request human review.

    After configuration is validated, the classifier creates both synchronous and
    asynchronous `httpx2` clients. Supply `client` and/or `async_client` to reuse
    clients configured by your application, including custom transports for testing,
    or network policy enforcement. Injected clients are used as-is; `timeout` only
    configures clients created by this class.

    The classifier does not add separate client lifecycle methods. Keep classifier
    instances long-lived to benefit from connection pooling. Applications that require
    deterministic cleanup can close `classifier.client` and `classifier.async_client`
    directly, following the corresponding `httpx2` sync and async client interfaces.

    Native TypeSafe state may be a string, JSON object, or JSON array. LangChain
    `BaseMessage` objects and message sequences can appear at the root or anywhere
    inside JSON objects and arrays. They are converted to role/content JSON before the
    request is sent. Message IDs are omitted, while system, user, assistant, and tool
    roles are preserved.

    The API key is read from `TYPESAFE_API_KEY` when `api_key` is omitted. Explicit
    constructor values take precedence over environment configuration.

    Args:
        model: 用于回答调用问题的 TypeSafe 模型。
        api_key: TypeSafe API 密钥。若省略，则读取 `TYPESAFE_API_KEY`。
        base_url: TypeSafe API 的根 URL。
        timeout: 应用于此分类器创建的客户端的超时时间（秒）。
        client: 可选的同步 `httpx2.Client`，由 `invoke` 使用。
        async_client: 可选的异步 `httpx2.AsyncClient`，由 `ainvoke` 使用。

    Raises:
        ValueError: 当凭证不可用或超时非正数时抛出。

    ??? example "在多个维度上对状态进行分类"

        将共享相同状态的问题一起发送。每个答案仍可通过其问题 ID 独立访问。

        ```python
        from langchain_typesafe import Choice, Noul, Score, TypeSafeClassifier

        classifier = TypeSafeClassifier()

        response = classifier.invoke(
            {
                "state": (
                    "Stripe has failed to connect for three days. "
                    "Please help immediately."
                ),
                "questions": {
                    "department": Choice(
                        instructions="Which team should handle this request?",
                        criteria={
                            "billing": "Payment or subscription issues.",
                            "technical": "Product bugs or integration failures.",
                        },
                    ),
                    "urgent": Noul(
                        instructions="Does this message require an urgent response?"
                    ),
                    "frustration": Score(
                        instructions="How frustrated does the customer appear?",
                        criteria=["Calm.", "Concerned but civil.", "Very angry."],
                    ),
                },
            }
        )
        print(response.choices["department"].choice)
        print(response.nouls["urgent"].noul)
        print(response.scores["frustration"].score)
        ```

    ??? example "异步分类"

        `ainvoke` 使用分类器的异步 HTTP 客户端，并返回与 `invoke` 相同的
        `ClassifierResponse` 类型。

        ```python
        from langchain_typesafe import Noul, TypeSafeClassifier

        classifier = TypeSafeClassifier()

        response = await classifier.ainvoke(
            {
                "state": "Please refund the duplicate charge.",
                "questions": {
                    "refund_requested": Noul(
                        instructions="Does the customer request a refund?"
                    )
                },
            }
        )
        print(response.nouls["refund_requested"].noul)
        ```
    """

    model: str = Field(default=_DEFAULT_MODEL, min_length=1)
    """用于分类的 TypeSafe 模型名称。

    默认值 `jev-latest` 采用 TypeSafe 最新兼容的 Jev
    版本。当应用需要在模型更新间保持行为可重现时，请使用具体的模型标识符。初始化时会去除首尾空白字符，空模型名称将被拒绝。
    """

    api_key: SecretStr | str = Field(
        default_factory=secret_from_env("TYPESAFE_API_KEY", default=""),
        exclude=True,
        repr=False,
    )
    """用于认证 TypeSafe 请求的 API 密钥。

    If omitted, the key is read from the `TYPESAFE_API_KEY` environment variable when
    the classifier is initialized. An explicit constructor value takes precedence. The
    value is stored as `SecretStr` and excluded from model representation and
    serialization.

    ??? example "通过环境变量指定"

        ```bash
        export TYPESAFE_API_KEY=...
        ```

        ```python
        from langchain_typesafe import Noul, TypeSafeClassifier

        classifier = TypeSafeClassifier()
        ```

    ??? example "直接指定"

        ```python
        classifier = TypeSafeClassifier(api_key="...")
        ```
    """

    base_url: str = Field(
        default_factory=from_env("TYPESAFE_BASE_URL", default=_DEFAULT_BASE_URL)
    )
    """用于 TypeSafe API 请求的根 URL。

    解析顺序：

    1. 传给 `TypeSafeClassifier` 的显式 `base_url`。
    2. `TYPESAFE_BASE_URL` 环境变量。
    3. `https://api.typesafe.ai`。

    Requests are sent to `/v1/systemone` beneath this URL. Override it for a compatible
    gateway, test server, or private deployment. URL validation is delegated to
    `httpx2` when a request is made.
    """

    timeout: float = Field(default=_DEFAULT_TIMEOUT, gt=0)
    """由该分类器创建的客户端的超时时间（秒）。

    This setting is passed to both `httpx2.Client` and `httpx2.AsyncClient` when their
    respective fields are omitted. It does not modify an injected client's timeout;
    configure custom clients directly when different sync and async policies are needed.
    """

    client: httpx2.Client | None = Field(default=None, exclude=True, repr=False)
    """可选的同步 `httpx2.Client`，供 `invoke` 和 `batch` 使用。

    If omitted, the classifier creates a client using `timeout`. Supply a client to
    reuse connection pools or configure a custom transport, proxy, TLS policy, or test
    fixture. The injected client is used as-is and is not closed by the classifier; the
    caller retains responsibility for its lifecycle.

    此客户端不用于 `ainvoke` 或 `abatch`。当异步调用也需要自定义 HTTP
    行为时，需单独配置 `async_client`。
    """

    async_client: httpx2.AsyncClient | None = Field(
        default=None,
        exclude=True,
        repr=False,
    )
    """可选的异步 `httpx2.AsyncClient`，供 `ainvoke` 和 `abatch` 使用。

    If omitted, the classifier creates an asynchronous client using `timeout`. Supply a
    client to reuse connection pools or configure a custom transport, proxy, TLS policy,
    or test fixture. The injected client is used as-is and is not closed by the
    classifier; the caller retains responsibility for its lifecycle.

    此客户端不用于 `invoke` 或 `batch`。当同步调用也需要自定义 HTTP 行为时，需单独配置
    `client`。
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="forbid",
        validate_default=True,
    )

    @field_validator("model")
    @classmethod
    def _validate_model(cls, model: str) -> str:
        model = model.strip()
        if not model:
            message = "TypeSafe model must not be empty."
            raise ValueError(message)
        return model

    @field_validator("api_key")
    @classmethod
    def _validate_api_key(cls, api_key: SecretStr | str) -> SecretStr:
        secret = api_key if isinstance(api_key, SecretStr) else SecretStr(api_key)
        if not secret.get_secret_value().strip():
            message = (
                "TypeSafe API key is required. Pass `api_key` or set "
                "`TYPESAFE_API_KEY`."
            )
            raise ValueError(message)
        return secret

    @model_validator(mode="after")
    def _build_clients(self) -> Self:
        """在配置验证后创建缺失的同步与异步客户端。"""
        if self.client is None:
            self.client = httpx2.Client(timeout=self.timeout)
        if self.async_client is None:
            self.async_client = httpx2.AsyncClient(timeout=self.timeout)
        return self

    @classmethod
    @override
    def is_lc_serializable(cls) -> bool:
        return True

    @classmethod
    @override
    def get_lc_namespace(cls) -> list[str]:
        return ["langchain", "classifiers", "typesafe"]

    @property
    def lc_secrets(self) -> dict[str, str]:
        """将 API 密钥字段映射到其对应的环境变量以用于序列化。"""
        return {"api_key": "TYPESAFE_API_KEY"}

    @override
    def invoke(
        self,
        input: ClassifierRequest,
        config: RunnableConfig | None = None,
        **_: Any,
    ) -> ClassifierResponse:
        """同步分类单个请求。

        Args:
            input: Complete request containing the state and typed questions.
            config: Optional LangChain runnable configuration for callbacks, tags,
                metadata, and tracing.
            **_: Additional keyword arguments accepted for `Runnable` compatibility and
                otherwise ignored.

        Returns:
            结构化的 TypeSafe 回答及请求元数据。

        Raises:
            TypeSafeAPIError: 如果 TypeSafe 返回非成功的 HTTP 响应。
            TypeSafeAPIConnectionError: 如果未收到 HTTP 响应。
            TypeSafeAPITimeoutError: 如果请求超过客户端超时时间。
            TypeSafeAPIResponseValidationError: 如果成功响应格式错误。
        """
        return self._call_with_config(
            self._classify,
            input,
            self._traced_config(config),
            run_type="llm",
        )

    @override
    async def ainvoke(
        self,
        input: ClassifierRequest,
        config: RunnableConfig | None = None,
        **_: Any,
    ) -> ClassifierResponse:
        """异步分类单个请求。

        Args:
            input: Complete request containing the state and typed questions.
            config: Optional LangChain runnable configuration for callbacks, tags,
                metadata, and tracing.
            **_: Additional keyword arguments accepted for `Runnable` compatibility and
                otherwise ignored.

        Returns:
            结构化 TypeSafe 的答案和请求元数据。

        Raises:
            TypeSafeAPIError: 如果 TypeSafe 返回非成功的 HTTP 响应。
            TypeSafeAPIConnectionError: 如果未收到 HTTP 响应。
            TypeSafeAPITimeoutError: 如果请求超过客户端超时时间。
            TypeSafeAPIResponseValidationError: 如果成功响应格式错误。
        """
        return await self._acall_with_config(
            self._aclassify,
            input,
            self._traced_config(config),
            run_type="llm",
        )

    def _classify(self, request: ClassifierRequest) -> ClassifierResponse:
        payload = self._payload(request)
        if self.client is None:  # pragma: no cover - guaranteed by model validation
            message = "Synchronous TypeSafe client was not initialized."
            raise TypeSafeAPIConnectionError(message)
        try:
            response = self.client.post(
                self._endpoint,
                json=payload,
                headers=self._request_headers,
            )
        except httpx2.TimeoutException as error:
            raise TypeSafeAPITimeoutError(self.client.timeout) from error
        except httpx2.HTTPError as error:
            message = "Unable to connect to the TypeSafe API."
            raise TypeSafeAPIConnectionError(message) from error
        return self._record_usage(parse_response(response))

    async def _aclassify(
        self,
        request: ClassifierRequest,
    ) -> ClassifierResponse:
        payload = self._payload(request)
        if self.async_client is None:  # pragma: no cover - guaranteed by validation
            message = "Asynchronous TypeSafe client was not initialized."
            raise TypeSafeAPIConnectionError(message)
        try:
            response = await self.async_client.post(
                self._endpoint,
                json=payload,
                headers=self._request_headers,
            )
        except httpx2.TimeoutException as error:
            raise TypeSafeAPITimeoutError(self.async_client.timeout) from error
        except httpx2.HTTPError as error:
            message = "Unable to connect to the TypeSafe API."
            raise TypeSafeAPIConnectionError(message) from error
        return self._record_usage(parse_response(response))

    def _traced_config(self, config: RunnableConfig | None) -> RunnableConfig:
        """创建运行时设置 `ls_provider` 和 `ls_model_name`。"""
        config = ensure_config(config)
        config["metadata"] = {
            **(config.get("metadata") or {}),
            "ls_provider": _LS_PROVIDER,
            "ls_model_name": self.model,
            "ls_model_type": "chat",
        }
        return config

    def _record_usage(self, response: ClassifierResponse) -> ClassifierResponse:
        """如果存在活动运行，将 TypeSafe 的 token 使用量附加到该运行。

        当追踪被禁用时不会写入任何内容，且追踪失败从不导致本应成功的分类失败。
        """
        input_tokens = response.usage.input_tokens or 0
        output_tokens = response.usage.output_tokens or 0
        try:
            run_tree = get_current_run_tree()
            if run_tree is not None:
                run_tree.extra.setdefault("metadata", {})["usage_metadata"] = {
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "total_tokens": input_tokens + output_tokens,
                }
        except Exception:  # noqa: BLE001 - tracing must not break classification
            logger.debug("Could not attach TypeSafe usage.", exc_info=True)
        return response

    @property
    def _endpoint(self) -> str:
        return f"{self.base_url.rstrip('/')}/v1/systemone"

    @property
    def _request_headers(self) -> dict[str, str]:
        api_key = (
            self.api_key.get_secret_value()
            if isinstance(self.api_key, SecretStr)
            else self.api_key
        )
        return {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": f"langchain-typesafe/{__version__}",
        }

    def _payload(self, request: ClassifierRequest) -> dict[str, JsonValue]:
        return {
            "state": serialize_state(request["state"]),
            "model": self.model,
            "questions": {
                name: question.model_dump(mode="json", exclude_none=True)
                for name, question in request["questions"].items()
            },
        }


__all__ = ["TypeSafeClassifier"]

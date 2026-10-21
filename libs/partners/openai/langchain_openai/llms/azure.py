"""Azure OpenAI 大语言模型。不要与聊天模型混淆。"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable, Mapping
from typing import Any, cast

import openai
from langchain_core.language_models import LangSmithParams
from langchain_core.utils import from_env, secret_from_env
from pydantic import Field, SecretStr, model_validator
from typing_extensions import Self

from langchain_openai._version import __version__
from langchain_openai.llms.base import BaseOpenAI

logger = logging.getLogger(__name__)


class AzureOpenAI(BaseOpenAI):
    """Azure 特有的 OpenAI 大语言模型。

    To use, you should have the `openai` python package installed, and the
    environment variable `OPENAI_API_KEY` set with your API key.

    Any parameters that are valid to be passed to the openai.create call can be passed
    in, even if not explicitly saved on this class.

    Example:
        ```python
        from langchain_openai import AzureOpenAI

        openai = AzureOpenAI(model_name="gpt-3.5-turbo-instruct")
        ```
    """

    azure_endpoint: str | None = Field(
        default_factory=from_env("AZURE_OPENAI_ENDPOINT", default=None)
    )
    """您的 Azure 终端点，包括资源。

        如果未提供，将自动从环境变量 `AZURE_OPENAI_ENDPOINT` 推断。

        Example: `'https://example-resource.azure.openai.com/'`
    """
    deployment_name: str | None = Field(default=None, alias="azure_deployment")
    """一个模型部署。

        如果提供，则将基础客户端 URL 设置为包含 `/deployments/{azure_deployment}`。

        !!! note
            这意味着您将无法使用非部署端点。

    """
    openai_api_version: str | None = Field(
        alias="api_version",
        default_factory=from_env("OPENAI_API_VERSION", default=None),
    )
    """如果未提供，将自动从环境变量 `OPENAI_API_VERSION` 推断。"""
        # 检查 OPENAI_KEY 以保证向后兼容性。
        # TODO: 移除 OPENAI_API_KEY 支持以避免在使用
        # 其他形式的 Azure 凭据时可能产生的冲突。
    openai_api_key: SecretStr | None = Field(
        alias="api_key",
        default_factory=secret_from_env(
            ["AZURE_OPENAI_API_KEY", "OPENAI_API_KEY"], default=None
        ),
    )
    azure_ad_token: SecretStr | None = Field(
        default_factory=secret_from_env("AZURE_OPENAI_AD_TOKEN", default=None)
    )
    """您的 Azure Active Directory token。

        如果未提供，将自动从环境变量 `AZURE_OPENAI_AD_TOKEN` 推断。

        `更多详情，请参阅此页面 <https://www.microsoft.com/en-us/security/business/identity-access/microsoft-entra-id>.`__
    """
    azure_ad_token_provider: Callable[[], str] | None = None
    """返回 Azure Active Directory token 的函数。

        每次同步请求都会调用该函数。对于异步请求，
        如果未提供 `azure_ad_async_token_provider`，则会调用该函数。
    """
    azure_ad_async_token_provider: Callable[[], Awaitable[str]] | None = None
    """返回 Azure Active Directory token 的函数。

        每次异步请求都会调用该函数。
    """
    openai_api_type: str | None = Field(
        default_factory=from_env("OPENAI_API_TYPE", default="azure")
    )
    """旧版，用于支持 `openai<1.0.0`。"""
    validate_base_url: bool = True
    """为了向后兼容。如果传入了旧版值 openai_api_base，尝试
        判断它是 base_url 还是 azure_endpoint，并相应更新。
    """

    @classmethod
    def get_lc_namespace(cls) -> list[str]:
        """获取 LangChain 对象的命名空间。

        Returns:
            `["langchain", "llms", "openai"]`
        """
        return ["langchain", "llms", "openai"]

    @property
    def lc_secrets(self) -> dict[str, str]:
        """密钥到环境变量的映射。"""
        return {
            "openai_api_key": "AZURE_OPENAI_API_KEY",
            "azure_ad_token": "AZURE_OPENAI_AD_TOKEN",
        }

    @classmethod
    def is_lc_serializable(cls) -> bool:
        """返回此模型是否可以被 LangChain 序列化。"""
        return True

    @model_validator(mode="after")
    def _set_azure_openai_version(self) -> Self:
        """在元数据中设置包版本。"""
        self._add_version("langchain-openai", __version__)
        return self

    @model_validator(mode="after")
    def validate_environment(self) -> Self:
        """验证 API 密钥和 Python 包是否存在于环境中。"""
        if self.n < 1:
            msg = "n must be at least 1."
            raise ValueError(msg)
        if self.streaming and self.n > 1:
            msg = "Cannot stream results when n > 1."
            raise ValueError(msg)
        if self.streaming and self.best_of > 1:
            msg = "Cannot stream results when best_of > 1."
            raise ValueError(msg)
                # 为了向后兼容。在 openai v1 之前，没有区分
                # azure_endpoint 和 base_url（openai_api_base）。
        openai_api_base = self.openai_api_base
        if openai_api_base and self.validate_base_url:
            if "/openai" not in openai_api_base:
                self.openai_api_base = (
                    cast(str, self.openai_api_base).rstrip("/") + "/openai"
                )
                msg = (
                    "As of openai>=1.0.0, Azure endpoints should be specified via "
                    "the `azure_endpoint` param not `openai_api_base` "
                    "(or alias `base_url`)."
                )
                raise ValueError(msg)
            if self.deployment_name:
                msg = (
                    "As of openai>=1.0.0, if `deployment_name` (or alias "
                    "`azure_deployment`) is specified then "
                    "`openai_api_base` (or alias `base_url`) should not be. "
                    "Instead use `deployment_name` (or alias `azure_deployment`) "
                    "and `azure_endpoint`."
                )
                raise ValueError(msg)
                self.deployment_name = None
        client_params: dict = {
            "api_version": self.openai_api_version,
            "azure_endpoint": self.azure_endpoint,
            "azure_deployment": self.deployment_name,
            "api_key": self.openai_api_key.get_secret_value()
            if self.openai_api_key
            else None,
            "azure_ad_token": self.azure_ad_token.get_secret_value()
            if self.azure_ad_token
            else None,
            "azure_ad_token_provider": self.azure_ad_token_provider,
            "organization": self.openai_organization,
            "base_url": self.openai_api_base,
            "timeout": self.request_timeout,
            "max_retries": self.max_retries,
            "default_headers": {
                "User-Agent": "langchain-partner-python-azure-openai",
                **(self.default_headers or {}),
            },
            "default_query": self.default_query,
        }
        if (
            self.azure_ad_token
            or self.azure_ad_token_provider
            or self.azure_ad_async_token_provider
        ):
            client_params["api_key"] = None
        if not self.client and (
            not self.azure_ad_async_token_provider
            or self.azure_ad_token
            or self.azure_ad_token_provider
        ):
            sync_specific = {"http_client": self.http_client}
            self.client = openai.AzureOpenAI(
                **client_params,
                **sync_specific,  # type: ignore[arg-type]
            ).completions
        if not self.async_client:
            async_specific = {"http_client": self.http_async_client}

            if self.azure_ad_async_token_provider:
                client_params["azure_ad_token_provider"] = (
                    self.azure_ad_async_token_provider
                )

            self.async_client = openai.AsyncAzureOpenAI(
                **client_params,
                **async_specific,  # type: ignore[arg-type]
            ).completions

        return self

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {
            "deployment_name": self.deployment_name,
            **super()._identifying_params,
        }

    @property
    def _invocation_params(self) -> dict[str, Any]:
        openai_params = {"model": self.deployment_name}
        return {**openai_params, **super()._invocation_params}

    def _get_ls_params(
        self, stop: list[str] | None = None, **kwargs: Any
    ) -> LangSmithParams:
        """获取追踪的标准参数。"""
        params = super()._get_ls_params(stop=stop, **kwargs)
        invocation_params = self._invocation_params
        params["ls_provider"] = "azure"
        if model_name := invocation_params.get("model"):
            params["ls_model_name"] = model_name
        return params

    @property
    def _llm_type(self) -> str:
        """返回 LLM 的类型。"""
        return "azure"

    @property
    def lc_attributes(self) -> dict[str, Any]:
        """与追踪相关的属性。"""
        return {
            "openai_api_type": self.openai_api_type,
            "openai_api_version": self.openai_api_version,
        }

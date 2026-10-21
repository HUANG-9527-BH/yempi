"""Azure OpenAI 嵌入封装。"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import cast

import openai
from langchain_core.utils import from_env, secret_from_env
from pydantic import Field, SecretStr, model_validator
from typing_extensions import Self

from langchain_openai.embeddings.base import OpenAIEmbeddings


class AzureOpenAIEmbeddings(OpenAIEmbeddings):  # type: ignore[override]
    """AzureOpenAI 嵌入模型集成。

    设置： 要访问 AzureOpenAI 嵌入模型，你需要创建一个 Azure 账户，获取 API
    密钥，并安装 `langchain-openai` 集成包。

        你需要部署一个 Azure OpenAI 实例。
        你可以通过以下
        [指南](https://learn.microsoft.com/en-us/azure/ai-services/openai/how-to/create-resource?pivots=web-portal) 在 Azure Portal 上部署一个版本。

        一旦你的实例运行起来，请确保你有实例的名称和密钥。你可以在 Azure Portal
        中，找到实例的“密钥和终结点”部分获取密钥。

        ```bash
        pip install -U langchain_openai

        # Set up your environment variables (or pass them directly to the model)
        export AZURE_OPENAI_API_KEY="your-api-key"
        export AZURE_OPENAI_ENDPOINT="https://<your-endpoint>.openai.azure.com/"
        export AZURE_OPENAI_API_VERSION="2024-02-01"
        ```

    Key init args — completion params:
        model:
            Name of `AzureOpenAI` model to use.
        dimensions:
            Number of dimensions for the embeddings. Can be specified only if the
            underlying model supports it.

    有关完整初始化参数列表及其描述，请参见参数部分。

    实例化：
        ```python
        from langchain_openai import AzureOpenAIEmbeddings

        embeddings = AzureOpenAIEmbeddings(
            model="text-embedding-3-large"
            # dimensions: int | None = None, # Can specify dimensions with new text-embedding-3 models
            # azure_endpoint="https://<your-endpoint>.openai.azure.com/", If not provided, will read env variable AZURE_OPENAI_ENDPOINT
            # api_key=... # Can provide an API key directly. If missing read env variable AZURE_OPENAI_API_KEY
            # openai_api_version=..., # If not provided, will read env variable AZURE_OPENAI_API_VERSION
        )
        ```

    嵌入单个文本：
        ```python
        input_text = "The meaning of life is 42"
        vector = embed.embed_query(input_text)
        print(vector[:3])
        ```
        ```python
        [-0.024603435769677162, -0.007543657906353474, 0.0039630369283258915]
        ```

    嵌入多个文本：
        ```python
        input_texts = ["Document 1...", "Document 2..."]
        vectors = embed.embed_documents(input_texts)
        print(len(vectors))
        # The first 3 coordinates for the first vector
        print(vectors[0][:3])
        ```
        ```python
        2
        [-0.024603435769677162, -0.007543657906353474, 0.0039630369283258915]
        ```

    异步：
        ```python
        vector = await embed.aembed_query(input_text)
        print(vector[:3])

        # multiple:
        # await embed.aembed_documents(input_texts)
        ```
        ```python
        [-0.009100092574954033, 0.005071679595857859, -0.0029193938244134188]
        ```
    """  # noqa: E501

    azure_endpoint: str | None = Field(
        default_factory=from_env("AZURE_OPENAI_ENDPOINT", default=None)
    )
    """你的 Azure 终结点，包括资源名。

        如果未提供，将自动从环境变量 `AZURE_OPENAI_ENDPOINT` 推断。

        Example: `https://example-resource.azure.openai.com/`
    """
    deployment: str | None = Field(default=None, alias="azure_deployment")
    """一个模型部署。

        如果提供，将设置基础客户端 URL 包含 `/deployments/{azure_deployment}`。

        !!! note
            这意味着你将无法使用非部署终结点。

    """
        # 检查 OPENAI_KEY 以保持向后兼容性。
        # TODO: 移除 OPENAI_API_KEY 支持，以避免在使用
        # 其他形式的 Azure 凭据时可能产生的冲突。
    openai_api_key: SecretStr | None = Field(
        alias="api_key",
        default_factory=secret_from_env(
            ["AZURE_OPENAI_API_KEY", "OPENAI_API_KEY"], default=None
        ),
    )
    """如果未提供，将自动从环境变量 `AZURE_OPENAI_API_KEY` 推断。"""
    openai_api_version: str | None = Field(
        default_factory=from_env("OPENAI_API_VERSION", default="2023-05-15"),
        alias="api_version",
    )
    """如果未提供，将自动从环境变量 `OPENAI_API_VERSION` 推断。

    Set to `'2023-05-15'` by default if env variable `OPENAI_API_VERSION` is not
    set.
    """
    azure_ad_token: SecretStr | None = Field(
        default_factory=secret_from_env("AZURE_OPENAI_AD_TOKEN", default=None)
    )
    """你的 Azure Active Directory token。

        如果未提供，将自动从环境变量 `AZURE_OPENAI_AD_TOKEN` 推断。

        [更多详情，请参见此页面。](https://www.microsoft.com/en-us/security/business/identity-access/microsoft-entra-id)
    """
    azure_ad_token_provider: Callable[[], str] | None = None
    """一个返回 Azure Active Directory token 的函数。

        Will be invoked on every sync request. For async requests,
        will be invoked if `azure_ad_async_token_provider` is not provided.
    """
    azure_ad_async_token_provider: Callable[[], Awaitable[str]] | None = None
    """一个返回 Azure Active Directory token 的函数。

        每次异步请求时都会调用。
    """
    openai_api_type: str | None = Field(
        default_factory=from_env("OPENAI_API_TYPE", default="azure")
    )
    validate_base_url: bool = True
    chunk_size: int = 2048
    """每次批处理中要嵌入的最大文本数量"""

    @model_validator(mode="after")
    def validate_environment(self) -> Self:
        """验证 API 密钥和 Python 包是否存在于环境中。"""
                # 为了向后兼容。在 openai v1 之前，没有区分
                # azure_endpoint 和 base_url（openai_api_base）。
        openai_api_base = self.openai_api_base
        if openai_api_base and self.validate_base_url:
                        # 仅在未提供 azure_endpoint 时验证 openai_api_base
            if not self.azure_endpoint and "/openai" not in openai_api_base:
                self.openai_api_base = cast(str, self.openai_api_base) + "/openai"
                msg = (
                    "As of openai>=1.0.0, Azure endpoints should be specified via "
                    "the `azure_endpoint` param not `openai_api_base` "
                    "(or alias `base_url`). "
                )
                raise ValueError(msg)
            if self.deployment:
                msg = (
                    "As of openai>=1.0.0, if `deployment` (or alias "
                    "`azure_deployment`) is specified then "
                    "`openai_api_base` (or alias `base_url`) should not be. "
                    "Instead use `deployment` (or alias `azure_deployment`) "
                    "and `azure_endpoint`."
                )
                raise ValueError(msg)
        client_params: dict = {
            "api_version": self.openai_api_version,
            "azure_endpoint": self.azure_endpoint,
            "azure_deployment": self.deployment,
            "api_key": (
                self.openai_api_key.get_secret_value() if self.openai_api_key else None
            ),
            "azure_ad_token": (
                self.azure_ad_token.get_secret_value() if self.azure_ad_token else None
            ),
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
            sync_specific: dict = {"http_client": self.http_client}
            self.client = openai.AzureOpenAI(
                **client_params,  # type: ignore[arg-type]
                **sync_specific,
            ).embeddings
        if not self.async_client:
            async_specific: dict = {"http_client": self.http_async_client}

            if self.azure_ad_async_token_provider:
                client_params["azure_ad_token_provider"] = (
                    self.azure_ad_async_token_provider
                )

            self.async_client = openai.AsyncAzureOpenAI(
                **client_params,  # type: ignore[arg-type]
                **async_specific,
            ).embeddings
        return self

    @property
    def _llm_type(self) -> str:
        return "azure-openai-chat"

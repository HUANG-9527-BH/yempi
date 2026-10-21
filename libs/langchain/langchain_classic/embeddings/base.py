import functools
from importlib import util
from typing import Any

from langchain_core._api import deprecated
from langchain_core.embeddings import Embeddings
from langchain_core.runnables import Runnable

_SUPPORTED_PROVIDERS = {
    "azure_ai": "langchain_azure_ai",
    "azure_openai": "langchain_openai",
    "bedrock": "langchain_aws",
    "cohere": "langchain_cohere",
    "google_genai": "langchain_google_genai",
    "google_vertexai": "langchain_google_vertexai",
    "huggingface": "langchain_huggingface",
    "mistralai": "langchain_mistralai",
    "ollama": "langchain_ollama",
    "openai": "langchain_openai",
}


def _get_provider_list() -> str:
    """获取已格式化的提供方及其对应包的列表。"""
    return "\n".join(
        f"  - {p}: {pkg.replace('_', '-')}" for p, pkg in _SUPPORTED_PROVIDERS.items()
    )


def _parse_model_string(model_name: str) -> tuple[str, str]:
    """将模型字符串解析为提供方和模型名称组件。

    模型字符串应为 'provider:model-name' 格式，其中 provider 为支持的提供方之一。

    Args:
        model_name: 以 'provider:model-name' 格式提供的模型字符串

    Returns:
        (provider, model_name) 的元组

    ```python
    _parse_model_string("openai:text-embedding-3-small")
    # Returns: ("openai", "text-embedding-3-small")

    _parse_model_string("bedrock:amazon.titan-embed-text-v1")
    # Returns: ("bedrock", "amazon.titan-embed-text-v1")
    ```

    Raises:
        ValueError: 如果模型字符串格式不正确或提供方不受支持

    """
    if ":" not in model_name:
        providers = _SUPPORTED_PROVIDERS
        msg = (
            f"Invalid model format '{model_name}'.\n"
            f"Model name must be in format 'provider:model-name'\n"
            f"Example valid model strings:\n"
            f"  - openai:text-embedding-3-small\n"
            f"  - bedrock:amazon.titan-embed-text-v1\n"
            f"  - cohere:embed-english-v3.0\n"
            f"Supported providers: {providers}"
        )
        raise ValueError(msg)

    provider, model = model_name.split(":", 1)
    provider = provider.lower().strip()
    model = model.strip()

    if provider not in _SUPPORTED_PROVIDERS:
        msg = (
            f"Provider '{provider}' is not supported.\n"
            f"Supported providers and their required packages:\n"
            f"{_get_provider_list()}"
        )
        raise ValueError(msg)
    if not model:
        msg = "Model name cannot be empty"
        raise ValueError(msg)
    return provider, model


def _infer_model_and_provider(
    model: str,
    *,
    provider: str | None = None,
) -> tuple[str, str]:
    if not model.strip():
        msg = "Model name cannot be empty"
        raise ValueError(msg)
    if provider is None and ":" in model:
        provider, model_name = _parse_model_string(model)
    else:
        model_name = model

    if not provider:
        providers = _SUPPORTED_PROVIDERS
        msg = (
            "Must specify either:\n"
            "1. A model string in format 'provider:model-name'\n"
            "   Example: 'openai:text-embedding-3-small'\n"
            "2. Or explicitly set provider from: "
            f"{providers}"
        )
        raise ValueError(msg)

    if provider not in _SUPPORTED_PROVIDERS:
        msg = (
            f"Provider '{provider}' is not supported.\n"
            f"Supported providers and their required packages:\n"
            f"{_get_provider_list()}"
        )
        raise ValueError(msg)
    return provider, model_name


@functools.lru_cache(maxsize=len(_SUPPORTED_PROVIDERS))
def _check_pkg(pkg: str) -> None:
    """检查包是否已安装。"""
    if not util.find_spec(pkg):
        pip_name = pkg.replace("_", "-")
        msg = (
            f"Could not import {pkg} python package. "
            f"Please install it with `pip install {pip_name}`"
        )
        raise ImportError(msg)


@deprecated(
    since="1.0.5",
    removal="2.0.0",
    alternative="langchain.embeddings.init_embeddings",
    addendum=(
        "Maintained in `langchain`; `langchain-classic` retains this entry point "
        "for import-compatibility only."
    ),
)
def init_embeddings(
    model: str,
    *,
    provider: str | None = None,
    **kwargs: Any,
) -> Embeddings | Runnable[Any, list[float]]:
    """从模型名称和可选提供方初始化嵌入模型。

    !!! note 必须安装与模型提供方对应的集成包。

    Args:
        model: 要使用的模型名称。

            可以是以下任一形式：

            - A model string like `"openai:text-embedding-3-small"`
            - Just the model name if the provider is specified separately or can be
                inferred.

            See supported providers under the `provider` arg description.
        provider: Optional explicit provider name. If not specified, will attempt to
            parse from the model string in the `model` arg.

            支持的提供方：

            - `openai`                  -> [`langchain-openai`](https://docs.langchain.com/oss/python/integrations/providers/openai)
            - `azure_ai`                -> [`langchain-azure-ai`](https://docs.langchain.com/oss/python/integrations/providers/microsoft)
            - `azure_openai`            -> [`langchain-openai`](https://docs.langchain.com/oss/python/integrations/providers/openai)
            - `bedrock`                 -> [`langchain-aws`](https://docs.langchain.com/oss/python/integrations/providers/aws)
            - `cohere`                  -> [`langchain-cohere`](https://docs.langchain.com/oss/python/integrations/providers/cohere)
            - `google_genai`            -> [`langchain-google-genai`](https://docs.langchain.com/oss/python/integrations/providers/google)
            - `google_vertexai`         -> [`langchain-google-vertexai`](https://docs.langchain.com/oss/python/integrations/providers/google)
            - `huggingface`             -> [`langchain-huggingface`](https://docs.langchain.com/oss/python/integrations/providers/huggingface)
            - `mistralai`               -> [`langchain-mistralai`](https://docs.langchain.com/oss/python/integrations/providers/mistralai)
            - `ollama`                  -> [`langchain-ollama`](https://docs.langchain.com/oss/python/integrations/providers/ollama)

        **kwargs: 传递给嵌入模型的额外模型特定参数。
            这些参数因提供方而异，请参阅各提供方的特定文档获取详细信息。

    Returns:
        一个可生成文本嵌入的 `Embeddings` 实例。

    Raises:
        ValueError: 如果模型提供方不受支持或无法确定
        ImportError: 如果缺少必需的提供方包

    ???+ note "示例用法"

        ```python
        # Using a model string
        model = init_embeddings("openai:text-embedding-3-small")
        model.embed_query("Hello, world!")

        # Using explicit provider
        model = init_embeddings(model="text-embedding-3-small", provider="openai")
        model.embed_documents(["Hello, world!", "Goodbye, world!"])

        # With additional parameters
        model = init_embeddings("openai:text-embedding-3-small", api_key="sk-...")
        ```

    !!! version-added "在 `langchain` 0.3.9 版本中新增"

    """
    if not model:
        providers = _SUPPORTED_PROVIDERS.keys()
        msg = (
            f"Must specify model name. Supported providers are: {', '.join(providers)}"
        )
        raise ValueError(msg)

    provider, model_name = _infer_model_and_provider(model, provider=provider)
    pkg = _SUPPORTED_PROVIDERS[provider]
    _check_pkg(pkg)

    if provider == "azure_ai":
        from langchain_azure_ai.embeddings import AzureAIOpenAIApiEmbeddingsModel

        return AzureAIOpenAIApiEmbeddingsModel(model=model_name, **kwargs)
    if provider == "azure_openai":
        from langchain_openai import AzureOpenAIEmbeddings

        return AzureOpenAIEmbeddings(model=model_name, **kwargs)
    if provider == "openai":
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(model=model_name, **kwargs)
    if provider == "bedrock":
        from langchain_aws import BedrockEmbeddings

        return BedrockEmbeddings(model_id=model_name, **kwargs)
    if provider == "google_genai":
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        return GoogleGenerativeAIEmbeddings(model=model_name, **kwargs)
    if provider == "google_vertexai":
        from langchain_google_vertexai import VertexAIEmbeddings

        return VertexAIEmbeddings(model=model_name, **kwargs)
    if provider == "cohere":
        from langchain_cohere import CohereEmbeddings

        return CohereEmbeddings(model=model_name, **kwargs)
    if provider == "mistralai":
        from langchain_mistralai import MistralAIEmbeddings

        return MistralAIEmbeddings(model=model_name, **kwargs)
    if provider == "huggingface":
        from langchain_huggingface import HuggingFaceEmbeddings

        return HuggingFaceEmbeddings(model_name=model_name, **kwargs)
    if provider == "ollama":
        from langchain_ollama import OllamaEmbeddings

        return OllamaEmbeddings(model=model_name, **kwargs)
    msg = (
        f"Provider '{provider}' is not supported.\n"
        f"Supported providers and their required packages:\n"
        f"{_get_provider_list()}"
    )
    raise ValueError(msg)


__all__ = [
    "Embeddings",  # 此条用于向后兼容
    "init_embeddings",
]

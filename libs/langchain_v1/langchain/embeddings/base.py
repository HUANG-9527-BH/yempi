"""嵌入模型的工厂函数。"""

import functools
import importlib
from collections.abc import Callable
from typing import Any

from langchain_core.embeddings import Embeddings


def _call(cls: type[Embeddings], **kwargs: Any) -> Embeddings:
    return cls(**kwargs)


_BUILTIN_PROVIDERS: dict[str, tuple[str, str, Callable[..., Embeddings]]] = {
    "azure_ai": ("langchain_azure_ai.embeddings", "AzureAIOpenAIApiEmbeddingsModel", _call),
    "azure_openai": ("langchain_openai", "AzureOpenAIEmbeddings", _call),
    "bedrock": (
        "langchain_aws",
        "BedrockEmbeddings",
        lambda cls, model, **kwargs: cls(model_id=model, **kwargs),
    ),
    "cohere": ("langchain_cohere", "CohereEmbeddings", _call),
    "google_genai": ("langchain_google_genai", "GoogleGenerativeAIEmbeddings", _call),
    "google_vertexai": ("langchain_google_vertexai", "VertexAIEmbeddings", _call),
    "huggingface": (
        "langchain_huggingface",
        "HuggingFaceEmbeddings",
        lambda cls, model, **kwargs: cls(model_name=model, **kwargs),
    ),
    "mistralai": ("langchain_mistralai", "MistralAIEmbeddings", _call),
    "ollama": ("langchain_ollama", "OllamaEmbeddings", _call),
    "openai": ("langchain_openai", "OpenAIEmbeddings", _call),
}
"""Registry mapping provider names to their import configuration.

Each entry maps a provider key to a tuple of:

- `module_path`: The Python module path containing the embeddings class.
- `class_name`: The name of the embeddings class to import.
- `creator_func`: A callable that instantiates the class with provided kwargs.

!!! note

    This dict is not exhaustive of all providers supported by LangChain, but is
    meant to cover the most popular ones and serve as a template for adding more
    providers in the future. If a provider is not in this dict, it can still be
    used with `init_chat_model` as long as its integration package is installed,
    but the provider key will not be inferred from the model name and must be
    specified explicitly via the `model_provider` parameter.

    Refer to the LangChain [integration documentation](https://docs.langchain.com/oss/python/integrations/providers/overview)
    for a full list of supported providers and their corresponding packages.
"""


@functools.lru_cache(maxsize=len(_BUILTIN_PROVIDERS))
def _get_embeddings_class_creator(provider: str) -> Callable[..., Embeddings]:
    """返回一个工厂函数，用于为指定提供者创建嵌入模型。

    此函数已缓存，以避免重复导入模块。

    Args:
        provider: 模型提供者的名称（例如 `'openai'`、`'cohere'`）。

            必须是 `_BUILTIN_PROVIDERS` 中的键。

    Returns:
        返回一个可调用对象，接受模型参数关键字并返回指定提供者的 `Embeddings` 实例。

    Raises:
        ValueError: 如果提供者不在 `_BUILTIN_PROVIDERS` 中。
        ImportError: 如果提供者的集成包未安装。
    """
    if provider not in _BUILTIN_PROVIDERS:
        msg = (
            f"Provider '{provider}' is not supported.\n"
            f"Supported providers and their required packages:\n"
            f"{_get_provider_list()}"
        )
        raise ValueError(msg)

    module_name, class_name, creator_func = _BUILTIN_PROVIDERS[provider]
    try:
        module = importlib.import_module(module_name)
    except ImportError as e:
        pkg = module_name.split(".", maxsplit=1)[0].replace("_", "-")
        msg = f"Could not import {pkg} python package. Please install it with `pip install {pkg}`"
        raise ImportError(msg) from e

    cls = getattr(module, class_name)
    return functools.partial(creator_func, cls=cls)


def _get_provider_list() -> str:
    """获取格式化后的提供者列表及其对应的包名。"""
    return "\n".join(
        f"  - {p}: {pkg[0].replace('_', '-')}" for p, pkg in _BUILTIN_PROVIDERS.items()
    )


def _parse_model_string(model_name: str) -> tuple[str, str]:
    """将模型字符串解析为提供者和模型名称组件。

    模型字符串应采用 `provider:model-name` 格式，其中 provider 为受支持的提供者之一。

    Args:
        model_name: 采用 `provider:model-name` 格式的模型字符串

    Returns:
        返回元组 (provider, model_name)

    Example:
        ```python
        _parse_model_string("openai:text-embedding-3-small")
        # Returns: ("openai", "text-embedding-3-small")

        _parse_model_string("bedrock:amazon.titan-embed-text-v1")
        # Returns: ("bedrock", "amazon.titan-embed-text-v1")
        ```

    Raises:
        ValueError: 如果模型字符串格式不正确或提供者不受支持

    """
    if ":" not in model_name:
        msg = (
            f"Invalid model format '{model_name}'.\n"
            f"Model name must be in format 'provider:model-name'\n"
            f"Example valid model strings:\n"
            f"  - openai:text-embedding-3-small\n"
            f"  - bedrock:amazon.titan-embed-text-v1\n"
            f"  - cohere:embed-english-v3.0\n"
            f"Supported providers: {_BUILTIN_PROVIDERS.keys()}"
        )
        raise ValueError(msg)

    provider, model = model_name.split(":", 1)
    provider = provider.lower().strip()
    model = model.strip()

    if provider not in _BUILTIN_PROVIDERS:
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
        msg = (
            "Must specify either:\n"
            "1. A model string in format 'provider:model-name'\n"
            "   Example: 'openai:text-embedding-3-small'\n"
            "2. Or explicitly set provider from: "
            f"{_BUILTIN_PROVIDERS.keys()}"
        )
        raise ValueError(msg)

    if provider not in _BUILTIN_PROVIDERS:
        msg = (
            f"Provider '{provider}' is not supported.\n"
            f"Supported providers and their required packages:\n"
            f"{_get_provider_list()}"
        )
        raise ValueError(msg)
    return provider, model_name


def init_embeddings(
    model: str,
    *,
    provider: str | None = None,
    **kwargs: Any,
) -> Embeddings:
    """从模型名称和可选提供者初始化嵌入模型。

    !!! note

        需要安装所选模型提供者的集成包。

        请参阅下方 `model_provider` 参数以获取具体包名
        （例如 `pip install langchain-openai`）。

        参考 [提供者集成的 API 文档](https://docs.langchain.com/oss/python/integrations/providers)
        了解可作为 `**kwargs` 使用的支持模型参数。

    Args:
        model: 模型名称，例如 `'openai:text-embedding-3-small'`。

            You can also specify model and model provider in a single argument using
            `'{model_provider}:{model}'` format, e.g. `'openai:text-embedding-3-small'`.
        provider: The model provider if not specified as part of the model arg
            (see above).

            支持的 `provider` 值及其对应的集成包为：

            - `openai`                  -> [`langchain-openai`](https://docs.langchain.com/oss/python/integrations/providers/openai)
            - `azure_ai`                -> [`langchain-azure-ai`](https://docs.langchain.com/oss/python/integrations/providers/microsoft)
            - `azure_openai`            -> [`langchain-openai`](https://docs.langchain.com/oss/python/integrations/providers/openai)
            - `bedrock`                 -> [`langchain-aws`](https://docs.langchain.com/oss/python/integrations/providers/aws)
            - `cohere`                  -> [`langchain-cohere`](https://docs.langchain.com/oss/python/integrations/providers/cohere)
            - `google_vertexai`         -> [`langchain-google-vertexai`](https://docs.langchain.com/oss/python/integrations/providers/google)
            - `huggingface`             -> [`langchain-huggingface`](https://docs.langchain.com/oss/python/integrations/providers/huggingface)
            - `mistralai`               -> [`langchain-mistralai`](https://docs.langchain.com/oss/python/integrations/providers/mistralai)
            - `ollama`                  -> [`langchain-ollama`](https://docs.langchain.com/oss/python/integrations/providers/ollama)

        **kwargs: 传递给嵌入模型的额外模型特定参数。

            不同提供方的参数各不相同。请参考特定模型提供方的
            [集成参考](https://reference.langchain.com/python/integrations/)
            以获取所有可用参数。

    Returns:
        一个可生成文本嵌入的 `Embeddings` 实例。

    Raises:
        ValueError: 如果模型提供方不受支持或无法确定
        ImportError: 如果所需提供方包未安装

    ???+ example

        ```python
        # pip install langchain langchain-openai

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
        providers = _BUILTIN_PROVIDERS.keys()
        msg = f"Must specify model name. Supported providers are: {', '.join(providers)}"
        raise ValueError(msg)

    provider, model_name = _infer_model_and_provider(model, provider=provider)
    return _get_embeddings_class_creator(provider)(model=model_name, **kwargs)


__all__ = [
    "Embeddings",  # This one is for backwards compatibility
    "init_embeddings",
]

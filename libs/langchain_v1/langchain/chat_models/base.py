"""聊天模型的工厂函数。"""

from __future__ import annotations

import functools
import importlib
import warnings
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    TypeAlias,
    cast,
    overload,
)

from langchain_core.language_models import BaseChatModel, LanguageModelInput
from langchain_core.messages import AIMessage, AnyMessage
from langchain_core.prompt_values import ChatPromptValueConcrete, StringPromptValue
from langchain_core.runnables import Runnable, RunnableConfig, ensure_config
from langchain_core.utils._gateway import _apply_gateway_config
from typing_extensions import override

_LANGSMITH_GATEWAY_DEFAULT_BASE = "https://gateway.smith.langchain.com/v1"


def _init_langsmith(cls: type[BaseChatModel], **kwargs: Any) -> BaseChatModel:
    _apply_gateway_config(
        kwargs,
        cls,
        base_url_field="openai_api_base",
        api_key_field="openai_api_key",
        provider_path="v1",
        api_key_env=("LANGSMITH_GATEWAY_API_KEY", "LANGSMITH_API_KEY"),
        default_base_url=_LANGSMITH_GATEWAY_DEFAULT_BASE,
    )
    kwargs["use_responses_api"] = True
    return cls(**kwargs)


if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable, Iterator, Sequence
    from types import ModuleType

    from langchain_core.runnables.schema import StreamEvent
    from langchain_core.tools import BaseTool
    from langchain_core.tracers import RunLog, RunLogPatch
    from pydantic import BaseModel


def _call(cls: type[BaseChatModel], **kwargs: Any) -> BaseChatModel:
        # TODO: 当最低版本要求为 Python 3.11 时，替换为 operator.call
    return cls(**kwargs)


_BUILTIN_PROVIDERS: dict[str, tuple[str, str, Callable[..., BaseChatModel]]] = {
    "anthropic": ("langchain_anthropic", "ChatAnthropic", _call),
    "anthropic_bedrock": ("langchain_aws", "ChatAnthropicBedrock", _call),
    "azure_ai": ("langchain_azure_ai.chat_models", "AzureAIOpenAIApiChatModel", _call),
    "azure_openai": ("langchain_openai", "AzureChatOpenAI", _call),
    "baseten": ("langchain_baseten", "ChatBaseten", _call),
    "bedrock": ("langchain_aws", "ChatBedrock", _call),
    "bedrock_converse": ("langchain_aws", "ChatBedrockConverse", _call),
    "bedrock_mantle_anthropic": ("langchain_aws", "ChatAnthropicMantle", _call),
    "bedrock_mantle_openai": ("langchain_aws", "ChatOpenAIMantle", _call),
    "cohere": ("langchain_cohere", "ChatCohere", _call),
    "deepseek": ("langchain_deepseek", "ChatDeepSeek", _call),
    "fireworks": ("langchain_fireworks", "ChatFireworks", _call),
    "google_anthropic_vertex": (
        "langchain_google_vertexai.model_garden",
        "ChatAnthropicVertex",
        _call,
    ),
    "google_genai": ("langchain_google_genai", "ChatGoogleGenerativeAI", _call),
    "google_vertexai": ("langchain_google_vertexai", "ChatVertexAI", _call),
    "groq": ("langchain_groq", "ChatGroq", _call),
    "huggingface": (
        "langchain_huggingface",
        "ChatHuggingFace",
        lambda cls, model, **kwargs: cls.from_model_id(model_id=model, **kwargs),
    ),
    "ibm": (
        "langchain_ibm",
        "ChatWatsonx",
        lambda cls, model, **kwargs: cls(model_id=model, **kwargs),
    ),
    "langsmith": ("langchain_openai", "ChatOpenAI", _init_langsmith),
    "litellm": ("langchain_litellm", "ChatLiteLLM", _call),
    "meta": ("langchain_meta", "ChatMetaModel", _call),
    "mistralai": ("langchain_mistralai", "ChatMistralAI", _call),
    "nvidia": ("langchain_nvidia_ai_endpoints", "ChatNVIDIA", _call),
    "ollama": ("langchain_ollama", "ChatOllama", _call),
    "openai": ("langchain_openai", "ChatOpenAI", _call),
    "openrouter": ("langchain_openrouter", "ChatOpenRouter", _call),
    "perplexity": ("langchain_perplexity", "ChatPerplexity", _call),
    "together": ("langchain_together", "ChatTogether", _call),
    "upstage": ("langchain_upstage", "ChatUpstage", _call),
    "xai": ("langchain_xai", "ChatXAI", _call),
}
"""Registry mapping provider names to their import configuration.

Each entry maps a provider key to a tuple of:

- `module_path`: The Python module path containing the chat model class.

    This may be a submodule (e.g., `'langchain_azure_ai.chat_models'`) if the class is
    not exported from the package root.
- `class_name`: The name of the chat model class to import.
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


def _import_module(module: str, class_name: str) -> ModuleType:
    """通过名称导入模块。

    Args:
        module: 要导入的完整模块名称（例如 `'langchain_openai'`）。
        class_name: 被导入类的名称，用于错误消息。

    Returns:
        导入的模块。

    Raises:
        ImportError: If the module cannot be imported, with a message suggesting
            the pip package to install.
    """
    try:
        return importlib.import_module(module)
    except ImportError as e:
                # 从模块路径提取包名（例如 "langchain_azure_ai.chat_models" 变为
        # "langchain-azure-ai"）
                # 用于向后兼容
        pkg = module.split(".", maxsplit=1)[0].replace("_", "-")
        msg = (
            f"Initializing {class_name} requires the {pkg} package. Please install it "
            f"with `pip install {pkg}`"
        )
        raise ImportError(msg) from e


@functools.lru_cache(maxsize=len(_BUILTIN_PROVIDERS))
def _get_chat_model_creator(
    provider: str,
) -> Callable[..., BaseChatModel]:
    """返回一个工厂函数，为指定提供方创建聊天模型。

    此函数已缓存，以避免重复导入模块。

    Args:
        provider: 模型提供方的名称（例如 `'openai'`, `'anthropic'`）。

            必须是 `_BUILTIN_PROVIDERS` 中的键。

    Returns:
        一个可调用对象，接受模型参数关键字，并返回指定提供方的 `BaseChatModel` 实例。

    Raises:
        ValueError: 如果提供方不在 `_BUILTIN_PROVIDERS` 中。
        ImportError: 如果提供方的集成包未安装。
    """
    if provider not in _BUILTIN_PROVIDERS:
        supported = ", ".join(_BUILTIN_PROVIDERS.keys())
        msg = f"Unsupported {provider=}.\n\nSupported model providers are: {supported}"
        raise ValueError(msg)

    pkg, class_name, creator_func = _BUILTIN_PROVIDERS[provider]
    try:
        module = _import_module(pkg, class_name)
    except ImportError as e:
        if provider != "ollama":
            raise
                # 如果 langchain-ollama 和 langchain-community 均不可用，
        try:
            module = _import_module("langchain_community.chat_models", class_name)
        except ImportError:
                        # 则抛出与 langchain-ollama 相关的错误
                        # 贡献者注意：若添加对新提供商的支持，请将提供商名称追加到下面文档字符串中的支持列表中。请勿更改现有提供商的顺序。
            raise e from None

    cls = getattr(module, class_name)
    return functools.partial(creator_func, cls=cls)


@overload
def init_chat_model(
    model: str,
    *,
    model_provider: str | None = None,
    configurable_fields: None = None,
    config_prefix: str | None = None,
    **kwargs: Any,
) -> BaseChatModel: ...


@overload
def init_chat_model(
    model: None = None,
    *,
    model_provider: str | None = None,
    configurable_fields: None = None,
    config_prefix: str | None = None,
    **kwargs: Any,
) -> _ConfigurableModel: ...


@overload
def init_chat_model(
    model: str | None = None,
    *,
    model_provider: str | None = None,
    configurable_fields: Literal["any"] | list[str] | tuple[str, ...] = ...,
    config_prefix: str | None = None,
    **kwargs: Any,
) -> _ConfigurableModel: ...


# FOR CONTRIBUTORS: If adding support for a new provider, please append the provider
# name to the supported list in the docstring below. Do *not* change the order of the
# existing providers.
def init_chat_model(
    model: str | None = None,
    *,
    model_provider: str | None = None,
    configurable_fields: Literal["any"] | list[str] | tuple[str, ...] | None = None,
    config_prefix: str | None = None,
    **kwargs: Any,
) -> BaseChatModel | _ConfigurableModel:
    """使用统一接口从任意支持的提供方初始化聊天模型。

    **两种主要用法：**

    1. **固定模型** – 提前指定模型并获得一个可直接使用的聊天模型。 2. **可配置模型** –
    选择在运行时通过 `config`
    指定参数（包括模型名称），便于在不更改代码的情况下切换模型/提供方

    !!! note "安装要求"

        Requires the integration package for the chosen model provider to
        be installed.

        See the `model_provider` parameter below for specific package names
        (e.g., `pip install langchain-openai`).

        参考 [提供方集成的 API 文档](https://docs.langchain.com/oss/python/integrations/providers)
        了解可作为 `**kwargs` 使用的支持模型参数。

    Args:
        model: Name of the model to use, with provider prefix — e.g.,
            `'openai:gpt-5.5'`.

            A bare model name (e.g., `'claude-opus-4-7'`) is also accepted; we
            will attempt to infer the provider from the prefix using the mapping
            below. Inference is best-effort and not guaranteed, so prefer
            the prefixed form when possible.

            Prefer pinned model IDs over moving aliases (e.g.,
            `'claude-haiku-4-5-20251001'` rather than `'claude-haiku-4-5'`)
            so behavior does not drift if the alias is repointed upstream.

            通过前缀推断的提供方（不区分大小写）：

            - `gpt-...` | `o1...` | `o3...`               -> `openai` -
            `claude...`                                 -> `anthropic` - `amazon....` |
            `anthropic....` | `meta....` -> `bedrock` -
            `gemini...`                                 -> `google_vertexai`
            (默认值将在下一个主版本中更改；通过传递 `model_provider` 来锁定) -
            `command...`                                -> `cohere` -
            `accounts/fireworks...`                     -> `fireworks` - `mistral...` |
            `mixtral...`                 -> `mistralai` -
            `deepseek...`                               -> `deepseek` -
            `grok...`                                   -> `xai` -
            `sonar...`                                  -> `perplexity` -
            `solar...`                                  -> `upstage` - `chatgpt...` |
            `text-davinci...`            -> `openai` (已弃用) model_provider:
            模型的提供方，作为独立参数传入，而非作为 `model` 的前缀。

            与前缀形式等效——例如
            `model='claude-sonnet-4-5', model_provider='anthropic'` 的行为
            与 `model='anthropic:claude-sonnet-4-5'` 相同。

            Prefer the prefix form on `model` for most usage. Reach for this
            kwarg when:

            - The provider is dynamic (read from config or an env var) and
                you'd otherwise concatenate strings.
            - You want `model` and `model_provider` to be independently
                swappable at runtime via `configurable_fields` (e.g., to route
                the same model name to a different host).

            支持的值及每种值所需的集成包：

            - `openai`                  -> [`langchain-openai`](https://docs.langchain.com/oss/python/integrations/providers/openai)
            - `anthropic`               -> [`langchain-anthropic`](https://docs.langchain.com/oss/python/integrations/providers/anthropic)
            - `azure_openai`            -> [`langchain-openai`](https://docs.langchain.com/oss/python/integrations/providers/openai)
            - `azure_ai`                -> [`langchain-azure-ai`](https://docs.langchain.com/oss/python/integrations/providers/microsoft)
            - `google_vertexai`         -> [`langchain-google-vertexai`](https://docs.langchain.com/oss/python/integrations/providers/google)
            - `google_genai`            -> [`langchain-google-genai`](https://docs.langchain.com/oss/python/integrations/providers/google)
            - `anthropic_bedrock`       -> [`langchain-aws`](https://docs.langchain.com/oss/python/integrations/providers/aws)
            - `bedrock`                 -> [`langchain-aws`](https://docs.langchain.com/oss/python/integrations/providers/aws)
            - `bedrock_converse`        -> [`langchain-aws`](https://docs.langchain.com/oss/python/integrations/providers/aws)
            - `bedrock_mantle_openai`   -> [`langchain-aws[openai]`](https://docs.langchain.com/oss/python/integrations/providers/aws) — 通过 `pip install "langchain-aws[openai]"` 安装
            - `bedrock_mantle_anthropic` -> [`langchain-aws[anthropic]`](https://docs.langchain.com/oss/python/integrations/providers/aws) — 通过 `pip install "langchain-aws[anthropic]"` 安装
            - `cohere`                  -> [`langchain-cohere`](https://docs.langchain.com/oss/python/integrations/providers/cohere)
            - `fireworks`               -> [`langchain-fireworks`](https://docs.langchain.com/oss/python/integrations/providers/fireworks)
            - `together`                -> [`langchain-together`](https://docs.langchain.com/oss/python/integrations/providers/together)
            - `mistralai`               -> [`langchain-mistralai`](https://docs.langchain.com/oss/python/integrations/providers/mistralai)
            - `huggingface`             -> [`langchain-huggingface`](https://docs.langchain.com/oss/python/integrations/providers/huggingface)
            - `groq`                    -> [`langchain-groq`](https://docs.langchain.com/oss/python/integrations/providers/groq)
            - `ollama`                  -> [`langchain-ollama`](https://docs.langchain.com/oss/python/integrations/providers/ollama)
            - `google_anthropic_vertex` -> [`langchain-google-vertexai`](https://docs.langchain.com/oss/python/integrations/providers/google)
            - `deepseek`                -> [`langchain-deepseek`](https://docs.langchain.com/oss/python/integrations/providers/deepseek)
            - `ibm`                     -> [`langchain-ibm`](https://docs.langchain.com/oss/python/integrations/providers/ibm)
            - `nvidia`                  -> [`langchain-nvidia-ai-endpoints`](https://docs.langchain.com/oss/python/integrations/providers/nvidia)
            - `xai`                     -> [`langchain-xai`](https://docs.langchain.com/oss/python/integrations/providers/xai)
            - `openrouter`              -> [`langchain-openrouter`](https://docs.langchain.com/oss/python/integrations/providers/openrouter)
            - `perplexity`              -> [`langchain-perplexity`](https://docs.langchain.com/oss/python/integrations/providers/perplexity)
            - `upstage`                 -> [`langchain-upstage`](https://docs.langchain.com/oss/python/integrations/providers/upstage)
            - `baseten`                 -> [`langchain-baseten`](https://docs.langchain.com/oss/python/integrations/providers/baseten)
            - `litellm`                 -> [`langchain-litellm`](https://docs.langchain.com/oss/python/integrations/providers/litellm)
            - `meta`                    -> [`langchain-meta`](https://pypi.org/project/langchain-meta)
            - `langsmith`               -> [`langchain-openai`](https://docs.langchain.com/langsmith/llm-gateway)

        configurable_fields: 运行时可配置的模型参数：

            - `None`: 无可配置字段（即固定模型）。
            - `'any'`: 所有字段均可配置。**请参阅下方安全提示。**
            - `list[str] | Tuple[str, ...]`: 指定字段可配置。

            Fields are assumed to have `config_prefix` stripped if a `config_prefix` is
            specified.

            若指定了 `model`，则默认为 `None`。

            若未指定 `model`，则默认为 `("model", "model_provider")`。

            !!! warning "安全提示"

                Setting `configurable_fields="any"` means fields like `api_key`,
                `base_url`, etc., can be altered at runtime, potentially redirecting
                model requests to a different service/user.

                Make sure that if you're accepting untrusted configurations that you
                enumerate the `configurable_fields=(...)` explicitly.

        config_prefix: 配置键的可选前缀。

            当同一应用中存在多个可配置模型时非常有用。

            若 `'config_prefix'` 为非空字符串，则 `model` 可通过
            `config["configurable"]["{config_prefix}_{param}"]` 键在运行时配置。
            请参见下方示例。

            若 `'config_prefix'` 为空字符串，则 `model` 可通过
            `config["configurable"]["{param}"]` 配置。

            - `temperature`: Model temperature for controlling randomness.
            - `max_tokens`: Maximum number of output tokens.
            - `timeout`: Maximum time (in seconds) to wait for a response.
            - `max_retries`: Maximum number of retry attempts for failed requests.
            - `base_url`: Custom API endpoint URL.
            - `rate_limiter`: A
                [`BaseRateLimiter`][langchain_core.rate_limiters.BaseRateLimiter]
                instance to control request rate.

            Refer to the specific model provider's
            [integration reference](https://reference.langchain.com/python/integrations/)
            for all available parameters.

    Returns:
        A `BaseChatModel` corresponding to the `model_name` and `model_provider`
            specified if configurability is inferred to be `False`.
            If configurable, a chat model emulator that initializes the
            underlying model at runtime once a config is passed in.

    Raises:
        ValueError: 若无法推断或不支持 `model_provider`。
        ImportError: 若模型提供商集成包未安装。

    ???+ example "初始化一个不可配置的模型"

        ```python
        # pip install langchain langchain-openai

        from langchain.chat_models import init_chat_model

        gpt_5 = init_chat_model("openai:gpt-5.5", temperature=0)
        gpt_5.invoke("what's your name")
        ```

    ??? example "部分可配置模型且无默认值"

        ```python
        # pip install langchain langchain-openai

        from langchain.chat_models import init_chat_model

        # (We don't need to specify configurable=True if a model isn't specified.)
        configurable_model = init_chat_model(temperature=0)

        # Use GPT-5.5 to generate the response
        configurable_model.invoke(
            "what's your name",
            config={"configurable": {"model": "gpt-5.5"}},
        )
        ```

    ??? example "完全可配置模型并带有默认值"

        ```python
        # pip install langchain langchain-openai langchain-anthropic

        from langchain.chat_models import init_chat_model

        configurable_model_with_default = init_chat_model(
            "openai:gpt-5.5",
            configurable_fields="any",  # This allows us to configure other params like temperature, max_tokens, etc at runtime.
            config_prefix="foo",
            temperature=0,
        )

        configurable_model_with_default.invoke("what's your name")
        # GPT-5.5 response with temperature 0 (as set in default)

        # Invoke overriding model and temperature at runtime via config.
        # Note the use of the "foo_" prefix on the config keys, which matches
        # the config_prefix we set when initializing the model.
        configurable_model_with_default.invoke(
            "what's your name",
            config={
                "configurable": {
                    "foo_model": "anthropic:claude-opus-4-7",
                    "foo_temperature": 0.6,
                }
            },
        )
        ```

    ??? example "将工具绑定到可配置模型"

        You can call any chat model declarative methods on a configurable model
        in the same way that you would with a normal model:

        ```python
        # pip install langchain langchain-openai langchain-anthropic

        from langchain.chat_models import init_chat_model
        from pydantic import BaseModel, Field


        class GetWeather(BaseModel):
            '''Get the current weather in a given location'''

            location: str = Field(..., description="The city and state, e.g. San Francisco, CA")


        class GetPopulation(BaseModel):
            '''Get the current population in a given location'''

            location: str = Field(..., description="The city and state, e.g. San Francisco, CA")


        configurable_model = init_chat_model(
            "gpt-5.5", configurable_fields=("model", "model_provider"), temperature=0
        )

        configurable_model_with_tools = configurable_model.bind_tools(
            [
                GetWeather,
                GetPopulation,
            ]
        )
        configurable_model_with_tools.invoke(
            "Which city is hotter today and which is bigger: LA or NY?"
        )
        # Use GPT-5.5

        configurable_model_with_tools.invoke(
            "Which city is hotter today and which is bigger: LA or NY?",
            config={"configurable": {"model": "claude-opus-4-7"}},
        )
        # Use Opus 4.7
        ```

    """  # noqa: E501
    if model is not None and not isinstance(model, str):
        msg = (  # type: ignore[unreachable]
            f"`model` must be a string (e.g., 'openai:gpt-5.5'), got "
            f"{type(model).__name__}. If you've already constructed a chat model "
            f"object, use it directly instead of passing it to init_chat_model()."
        )
        raise TypeError(msg)
    if not model and not configurable_fields:
        configurable_fields = ("model", "model_provider")
    config_prefix = config_prefix or ""
    if config_prefix and not configurable_fields:
        warnings.warn(
            f"{config_prefix=} has been set but no fields are configurable. Set "
            f"`configurable_fields=(...)` to specify the model params that are "
            f"configurable.",
            stacklevel=2,
        )

    if not configurable_fields:
        return _init_chat_model_helper(
            cast("str", model),
            model_provider=model_provider,
            **kwargs,
        )
    if model:
        kwargs["model"] = model
    if model_provider:
        kwargs["model_provider"] = model_provider
    return _ConfigurableModel(
        default_config=kwargs,
        config_prefix=config_prefix,
        configurable_fields=configurable_fields,
    )


def _init_chat_model_helper(
    model: str,
    *,
    model_provider: str | None = None,
    **kwargs: Any,
) -> BaseChatModel:
    model, model_provider = _parse_model(model, model_provider)
    creator_func = _get_chat_model_creator(model_provider)
    return creator_func(model=model, **kwargs)


def _attempt_infer_model_provider(model_name: str) -> str | None:
    """尝试从模型名称推断模型提供方。

    Args:
        model_name: 用于推断提供方的模型名称。

    Returns:
        推断出的提供方名称，如果无法推断则为 `None`。
    """
    model_lower = model_name.lower()

        # OpenAI 模型（包括较新模型和别名）
    if any(
        model_lower.startswith(pre)
        for pre in (
            "gpt-",
            "o1",
            "o3",
            "chatgpt",
            "text-davinci",
        )
    ):
        return "openai"

        # Anthropic 模型
    if model_lower.startswith("claude"):
        return "anthropic"

        # Cohere 模型
    if model_lower.startswith("command"):
        return "cohere"

        # Fireworks 模型
    if model_lower.startswith("accounts/fireworks"):
        return "fireworks"

        # Google 模型 — 前缀存在歧义（Vertex AI 与 GenAI/AI Studio API）
    if model_lower.startswith("gemini"):
        warnings.warn(
            f"Inferred `model_provider='google_vertexai'` from {model_name!r}. "
            "This default will change to 'google_genai' in the next major release."
            "To keep current behavior, pass `model_provider='google_vertexai'` "
            f"(or use the prefix form, e.g. 'google_vertexai:{model_name}'); "
            "for AI Studio / Gemini API, use 'google_genai' instead.",
            DeprecationWarning,
            stacklevel=5,
        )
        return "google_vertexai"

        # AWS Bedrock 模型
    if model_lower.startswith(("amazon.", "anthropic.", "meta.")):
        return "bedrock"

        # Mistral 模型
    if model_lower.startswith(("mistral", "mixtral")):
        return "mistralai"

        # DeepSeek 模型
    if model_lower.startswith("deepseek"):
        return "deepseek"

        # xAI 模型
    if model_lower.startswith("grok"):
        return "xai"

        # Perplexity 模型
    if model_lower.startswith("sonar"):
        return "perplexity"

        # Upstage 模型
    if model_lower.startswith("solar"):
        return "upstage"

    return None


def _parse_model(model: str, model_provider: str | None) -> tuple[str, str]:
    """解析模型名称和提供方，必要时进行推断。"""
        # 处理 provider:model 格式
    if (
        not model_provider
        and ":" in model
        and model.split(":", maxsplit=1)[0] in _BUILTIN_PROVIDERS
    ):
        model_provider = model.split(":", maxsplit=1)[0]
        model = ":".join(model.split(":")[1:])

        # 若未指定，则尝试推断 provider
    model_provider = model_provider or _attempt_infer_model_provider(model)

    if not model_provider:
                # 增强错误信息并提供建议
        supported_list = ", ".join(sorted(_BUILTIN_PROVIDERS))
        msg = (
            f"Unable to infer model provider for {model=}. "
            f"Please specify 'model_provider' directly.\n\n"
            f"Supported providers: {supported_list}\n\n"
            f"For help with specific providers, see: "
            f"https://docs.langchain.com/oss/python/integrations/providers"
        )
        raise ValueError(msg)

        # 标准化 provider 名称
    model_provider = model_provider.replace("-", "_").lower()
    return model, model_provider


def _remove_prefix(s: str, prefix: str) -> str:
    return s.removeprefix(prefix)


_DECLARATIVE_METHODS = ("bind_tools", "with_structured_output")


class _ConfigurableModel(Runnable[LanguageModelInput, Any]):
    def __init__(
        self,
        *,
        default_config: dict[str, Any] | None = None,
        configurable_fields: Literal["any"] | list[str] | tuple[str, ...] = "any",
        config_prefix: str = "",
        queued_declarative_operations: Sequence[tuple[str, tuple[Any, ...], dict[str, Any]]] = (),
    ) -> None:
        self._default_config: dict[str, Any] = default_config or {}
        self._configurable_fields: Literal["any"] | list[str] = (
            "any" if configurable_fields == "any" else list(configurable_fields)
        )
        self._config_prefix = (
            config_prefix + "_"
            if config_prefix and not config_prefix.endswith("_")
            else config_prefix
        )
        self._queued_declarative_operations: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = (
            list(
                queued_declarative_operations,
            )
        )

    def __getattr__(self, name: str) -> Any:
        if name in _DECLARATIVE_METHODS:
                        # 声明式操作，必须在实际模型实例化后才能应用。因此，不直接返回实际操作，而是将操作及其参数记录在队列中。该队列会在我们实际实例化模型时（在
            # self._model() 中）按顺序应用。
                        # 实例化模型对象后确保配置不为 None
                        # 这是 LanguageModelInput 的一个版本，用其子类的联合替代了抽象基类
            # BaseMessage，使得模式更加完善。
                        # 如果配置数 <= 1，则使用底层模型的批处理实现。
                        # 如果存在多个配置，默认使用 Runnable.batch，该方法通过执行器并行调用。
            def queue(*args: Any, **kwargs: Any) -> _ConfigurableModel:
                queued_declarative_operations = list(
                    self._queued_declarative_operations,
                )
                queued_declarative_operations.append((name, args, kwargs))
                return _ConfigurableModel(
                    default_config=dict(self._default_config),
                    configurable_fields=list(self._configurable_fields)
                    if isinstance(self._configurable_fields, list)
                    else self._configurable_fields,
                    config_prefix=self._config_prefix,
                    queued_declarative_operations=queued_declarative_operations,
                )

            return queue
        if self._default_config and (model := self._model()) and hasattr(model, name):
            return getattr(model, name)
        msg = f"{name} is not a BaseChatModel attribute"
        if self._default_config:
            msg += " and is not implemented on the default model"
        msg += "."
        raise AttributeError(msg)

    def _model(self, config: RunnableConfig | None = None) -> Runnable[Any, Any]:
        params = {**self._default_config, **self._model_params(config)}
        model = _init_chat_model_helper(**params)
        for name, args, kwargs in self._queued_declarative_operations:
            model = getattr(model, name)(*args, **kwargs)
        return model

    def _model_params(self, config: RunnableConfig | None) -> dict[str, Any]:
        config = ensure_config(config)
        model_params = {
            _remove_prefix(k, self._config_prefix): v
            for k, v in config.get("configurable", {}).items()
            if k.startswith(self._config_prefix)
        }
        if self._configurable_fields != "any":
            model_params = {k: v for k, v in model_params.items() if k in self._configurable_fields}
        return model_params

    def with_config(
        self,
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> _ConfigurableModel:
        config = RunnableConfig(**(config or {}), **cast("RunnableConfig", kwargs))
                # 如果配置数 <= 1，则使用底层模型的批处理实现。
        config = ensure_config(config)
        model_params = self._model_params(config)
        remaining_config = {k: v for k, v in config.items() if k != "configurable"}
        remaining_config["configurable"] = {
            k: v
            for k, v in config.get("configurable", {}).items()
            if _remove_prefix(k, self._config_prefix) not in model_params
        }
        queued_declarative_operations = list(self._queued_declarative_operations)
        if remaining_config:
            queued_declarative_operations.append(
                (
                    "with_config",
                    (),
                    {"config": remaining_config},
                ),
            )
        return _ConfigurableModel(
            default_config={**self._default_config, **model_params},
            configurable_fields=list(self._configurable_fields)
            if isinstance(self._configurable_fields, list)
            else self._configurable_fields,
            config_prefix=self._config_prefix,
            queued_declarative_operations=queued_declarative_operations,
        )

    @property
    @override
    def InputType(self) -> TypeAlias:
        """获取此 `Runnable` 的输入类型。"""
                # 如果存在多个配置，默认使用 Runnable.batch，该方法通过执行器并行调用。
                # 如果配置数 <= 1，则使用底层模型的批处理实现。
                # 如果存在多个配置，默认使用 Runnable.batch，该方法通过执行器并行调用。
        return str | StringPromptValue | ChatPromptValueConcrete | list[AnyMessage]

    @override
    def invoke(
        self,
        input: LanguageModelInput,
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> Any:
        return self._model(config).invoke(input, config=config, **kwargs)

    @override
    async def ainvoke(
        self,
        input: LanguageModelInput,
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> Any:
        return await self._model(config).ainvoke(input, config=config, **kwargs)

    @override
    def stream(
        self,
        input: LanguageModelInput,
        config: RunnableConfig | None = None,
        **kwargs: Any | None,
    ) -> Iterator[Any]:
        yield from self._model(config).stream(input, config=config, **kwargs)

    @override
    async def astream(
        self,
        input: LanguageModelInput,
        config: RunnableConfig | None = None,
        **kwargs: Any | None,
    ) -> AsyncIterator[Any]:
        async for x in self._model(config).astream(input, config=config, **kwargs):
            yield x

    def batch(
        self,
        inputs: list[LanguageModelInput],
        config: RunnableConfig | list[RunnableConfig] | None = None,
        *,
        return_exceptions: bool = False,
        **kwargs: Any | None,
    ) -> list[Any]:
        config = config or None
                # 如果配置数 <= 1，则使用底层模型的批处理实现。
        if config is None or isinstance(config, dict) or len(config) <= 1:
            if isinstance(config, list):
                config = config[0]
            return self._model(config).batch(
                inputs,
                config=config,
                return_exceptions=return_exceptions,
                **kwargs,
            )
                # 如果存在多个配置，默认使用 Runnable.batch，该方法通过执行器并行调用。
                # 显式添加以满足下游 linters 的要求。
        return super().batch(
            inputs,
            config=config,
            return_exceptions=return_exceptions,
            **kwargs,
        )

    async def abatch(
        self,
        inputs: list[LanguageModelInput],
        config: RunnableConfig | list[RunnableConfig] | None = None,
        *,
        return_exceptions: bool = False,
        **kwargs: Any | None,
    ) -> list[Any]:
        config = config or None
                # `bind_tools`
        # 由具体模型实现，因为工具绑定是提供商特定的。可配置模型可能尚未有具体的模型实例，因为调用配置可以稍后选择它。现在保存
        # `bind_tools` 的工具和关键字参数。当 `_model`
        # 后续构建选定的提供商模型时，会调用 `selected_model.bind_tools(tools,
        # **kwargs)` 并返回该 Runnable。
        if config is None or isinstance(config, dict) or len(config) <= 1:
            if isinstance(config, list):
                config = config[0]
            return await self._model(config).abatch(
                inputs,
                config=config,
                return_exceptions=return_exceptions,
                **kwargs,
            )
                # 强制转换，以便调用者仍能获得公共返回类型。
                # 显式添加以满足下游 linters 的要求。
        return await super().abatch(
            inputs,
            config=config,
            return_exceptions=return_exceptions,
            **kwargs,
        )

    def batch_as_completed(
        self,
        inputs: Sequence[LanguageModelInput],
        config: RunnableConfig | Sequence[RunnableConfig] | None = None,
        *,
        return_exceptions: bool = False,
        **kwargs: Any,
    ) -> Iterator[tuple[int, Any | Exception]]:
        config = config or None
                # `with_structured_output` 由具体模型实现，因为结构化输出是提供商特定的。
        if config is None or isinstance(config, dict) or len(config) <= 1:
            if isinstance(config, list):
                config = config[0]
            yield from self._model(cast("RunnableConfig", config)).batch_as_completed(  # type: ignore[call-overload]
                inputs,
                config=config,
                return_exceptions=return_exceptions,
                **kwargs,
            )
                # 如果多个配置默认使用 Runnable.batch，该方法使用执行器来调用
                # 并行地。
        else:
            yield from super().batch_as_completed(  # type: ignore[call-overload]
                inputs,
                config=config,
                return_exceptions=return_exceptions,
                **kwargs,
            )

    async def abatch_as_completed(
        self,
        inputs: Sequence[LanguageModelInput],
        config: RunnableConfig | Sequence[RunnableConfig] | None = None,
        *,
        return_exceptions: bool = False,
        **kwargs: Any,
    ) -> AsyncIterator[tuple[int, Any]]:
        config = config or None
                # 如果配置数 <= 1，则使用底层模型的批处理实现。
        if config is None or isinstance(config, dict) or len(config) <= 1:
            if isinstance(config, list):
                config = config[0]
            async for x in self._model(
                cast("RunnableConfig", config),
            ).abatch_as_completed(  # type: ignore[call-overload]
                inputs,
                config=config,
                return_exceptions=return_exceptions,
                **kwargs,
            ):
                yield x
                # 如果多个配置默认使用 Runnable.batch，而 Runnable.batch 使用执行器来调用
                # 并行地。
        else:
            async for x in super().abatch_as_completed(  # type: ignore[call-overload]
                inputs,
                config=config,
                return_exceptions=return_exceptions,
                **kwargs,
            ):
                yield x

    @override
    def transform(
        self,
        input: Iterator[LanguageModelInput],
        config: RunnableConfig | None = None,
        **kwargs: Any | None,
    ) -> Iterator[Any]:
        yield from self._model(config).transform(input, config=config, **kwargs)

    @override
    async def atransform(
        self,
        input: AsyncIterator[LanguageModelInput],
        config: RunnableConfig | None = None,
        **kwargs: Any | None,
    ) -> AsyncIterator[Any]:
        async for x in self._model(config).atransform(input, config=config, **kwargs):
            yield x

    @overload
    @override
    def astream_log(
        self,
        input: Any,
        config: RunnableConfig | None = None,
        *,
        diff: Literal[True] = True,
        with_streamed_output_list: bool = True,
        include_names: Sequence[str] | None = None,
        include_types: Sequence[str] | None = None,
        include_tags: Sequence[str] | None = None,
        exclude_names: Sequence[str] | None = None,
        exclude_types: Sequence[str] | None = None,
        exclude_tags: Sequence[str] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[RunLogPatch]: ...

    @overload
    @override
    def astream_log(
        self,
        input: Any,
        config: RunnableConfig | None = None,
        *,
        diff: Literal[False],
        with_streamed_output_list: bool = True,
        include_names: Sequence[str] | None = None,
        include_types: Sequence[str] | None = None,
        include_tags: Sequence[str] | None = None,
        exclude_names: Sequence[str] | None = None,
        exclude_types: Sequence[str] | None = None,
        exclude_tags: Sequence[str] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[RunLog]: ...

    @override
    async def astream_log(
        self,
        input: Any,
        config: RunnableConfig | None = None,
        *,
        diff: bool = True,
        with_streamed_output_list: bool = True,
        include_names: Sequence[str] | None = None,
        include_types: Sequence[str] | None = None,
        include_tags: Sequence[str] | None = None,
        exclude_names: Sequence[str] | None = None,
        exclude_types: Sequence[str] | None = None,
        exclude_tags: Sequence[str] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[RunLogPatch] | AsyncIterator[RunLog]:
        async for x in self._model(config).astream_log(  # type: ignore[call-overload, misc]
            input,
            config=config,
            diff=diff,
            with_streamed_output_list=with_streamed_output_list,
            include_names=include_names,
            include_types=include_types,
            include_tags=include_tags,
            exclude_tags=exclude_tags,
            exclude_types=exclude_types,
            exclude_names=exclude_names,
            **kwargs,
        ):
            yield x

    @override
    async def astream_events(  # type: ignore[override]
        self,
        input: Any,
        config: RunnableConfig | None = None,
        *,
        version: Literal["v1", "v2"] = "v2",
        include_names: Sequence[str] | None = None,
        include_types: Sequence[str] | None = None,
        include_tags: Sequence[str] | None = None,
        exclude_names: Sequence[str] | None = None,
        exclude_types: Sequence[str] | None = None,
        exclude_tags: Sequence[str] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[StreamEvent]:
        async for x in self._model(config).astream_events(
            input,
            config=config,
            version=version,
            include_names=include_names,
            include_types=include_types,
            include_tags=include_tags,
            exclude_tags=exclude_tags,
            exclude_types=exclude_types,
            exclude_names=exclude_names,
            **kwargs,
        ):
            yield x

        # 显式添加以满足下游 linter 的要求。
        # `bind_tools` 由具体模型实现，因为工具绑定是
        # 特定于提供方。可配置的模型可能没有具体的模型实例
        # 然而，由于调用配置稍后可以选择它。保存 `bind_tools` 工具
        # 和 kwargs。当 `_model` 后续构建选定的提供者模型时，会调用
        # `selected_model.bind_tools(tools, **kwargs)` 并返回该 Runnable。
        # 强制转换以确保调用者仍获得公共返回类型。
    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type[BaseModel] | Callable[..., Any] | BaseTool],
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, AIMessage]:
        return cast(
            "Runnable[LanguageModelInput, AIMessage]",
            self.__getattr__("bind_tools")(tools, **kwargs),
        )

        # 显式添加以满足下游 linter 的要求。
        # ``with_structured_output`` 由具体模型实现，因为结构化
        # 输出支持取决于提供方。可配置的模型可能尚未有具体的模型实例，因为调用配置可在稍后选择它。现在保存结构化输出的模式和关键字参数。当
    # `_model` 后续构建选定的提供方模型时，它会调用
    # `selected_model.with_structured_output(schema, **kwargs)` 并返回该 Runnable。
    # Cast so callers still get the public return type.
        # concrete model instance yet, since invocation config can choose it later. Save
        # the structured-output schema and kwargs now. When `_model` later builds the
        # selected provider model, it calls
        # `selected_model.with_structured_output(schema, **kwargs)` and returns that
        # runnable.
        # Cast so callers still get the public return type.
    def with_structured_output(
        self,
        schema: dict[str, Any] | type[BaseModel],
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, dict[str, Any] | BaseModel]:
        return cast(
            "Runnable[LanguageModelInput, dict[str, Any] | BaseModel]",
            self.__getattr__("with_structured_output")(schema, **kwargs),
        )

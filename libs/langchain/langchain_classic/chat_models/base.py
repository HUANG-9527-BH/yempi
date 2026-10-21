from __future__ import annotations

import warnings
from collections.abc import AsyncIterator, Callable, Iterator, Sequence
from importlib import util
from typing import Any, Literal, TypeAlias, cast, overload

from langchain_core._api import deprecated
from langchain_core.language_models import (
    BaseChatModel,
    LanguageModelInput,
    SimpleChatModel,
)
from langchain_core.language_models.chat_models import (
    agenerate_from_stream,
    generate_from_stream,
)
from langchain_core.messages import AIMessage, AnyMessage
from langchain_core.runnables import Runnable, RunnableConfig, ensure_config
from langchain_core.runnables.schema import StreamEvent
from langchain_core.tools import BaseTool
from langchain_core.tracers import RunLog, RunLogPatch
from pydantic import BaseModel
from typing_extensions import override

__all__ = [
        # 为了向后兼容
    "BaseChatModel",
    "SimpleChatModel",
    "agenerate_from_stream",
    "generate_from_stream",
    "init_chat_model",
]


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


# 贡献者请注意：若添加对新提供商的支持，请将提供商名称追加到下方文档字符串中的支持列表中。切勿更改现有提供商的顺序。
# 为了向后兼容
# 如果 `langchain-ollama` 和 `langchain-community` 均不可用，
@deprecated(
    since="1.0.5",
    removal="2.0.0",
    alternative="langchain.chat_models.init_chat_model",
    addendum=(
        "Maintained in `langchain`; `langchain-classic` retains this entry point "
        "for import-compatibility only."
    ),
)
def init_chat_model(
    model: str | None = None,
    *,
    model_provider: str | None = None,
    configurable_fields: Literal["any"] | list[str] | tuple[str, ...] | None = None,
    config_prefix: str | None = None,
    **kwargs: Any,
) -> BaseChatModel | _ConfigurableModel:
    """从任何受支持的提供商初始化一个聊天模型，使用统一的接口。

    !!! warning "请改用 `langchain.chat_models.init_chat_model`"

        此函数位于 `langchain-classic` 中，且不再积极维护。新功能和修复均在 `langchain`
        包中发布。

        更新您的导入语句：

        ```python
        # Don't do this:
        from langchain.chat_models import init_chat_model

        # Do this instead:
        from langchain.chat_models import init_chat_model
        ```

    **两种主要用例：**

    1. **固定模型** – 提前指定模型并获得一个可直接使用的聊天模型。 2. **可配置模型** –
    选择在运行时通过 `config`
    指定参数（包括模型名称），便于在不更改代码的情况下切换模型/提供商

    !!! note "安装要求"

        Requires the integration package for the chosen model provider to
        be installed.

        See the `model_provider` parameter below for specific package names
        (e.g., `pip install langchain-openai`).

        参考 [提供商集成的 API 参考](https://docs.langchain.com/oss/python/integrations/providers)
        以获取可作为 `**kwargs` 使用的支持模型参数。

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

            通过前缀推断的提供商（不区分大小写）：

            - `gpt-...` | `o1...` | `o3...`               -> `openai` -
            `claude...`                                 -> `anthropic` - `amazon....` |
            `anthropic....` | `meta....` -> `bedrock` -
            `gemini...`                                 -> `google_vertexai` -
            `command...`                                -> `cohere` -
            `accounts/fireworks...`                     -> `fireworks` - `mistral...` |
            `mixtral...`                 -> `mistralai` -
            `deepseek...`                               -> `deepseek` -
            `grok...`                                   -> `xai` -
            `sonar...`                                  -> `perplexity` -
            `solar...`                                  -> `upstage` - `chatgpt...` |
            `text-davinci...`            -> `openai` (旧版) model_provider:
            模型的提供商，单独传入而非作为 `model` 的前缀。

            Equivalent to the prefix form — e.g.,
            `model='claude-sonnet-4-5', model_provider='anthropic'` behaves
            the same as `model='anthropic:claude-sonnet-4-5'`.

            Prefer the prefix form on `model` for most usage. Reach for this
            kwarg when:

            - The provider is dynamic (read from config or an env var) and
                you'd otherwise concatenate strings.
            - You want `model` and `model_provider` to be independently
                swappable at runtime via `configurable_fields` (e.g., to route
                the same model name to a different host).

            支持的值以及每个值所需的集成包：

            - `openai`                  -> [`langchain-openai`](https://docs.langchain.com/oss/python/integrations/providers/openai)
            - `anthropic`               -> [`langchain-anthropic`](https://docs.langchain.com/oss/python/integrations/providers/anthropic)
            - `azure_openai`            -> [`langchain-openai`](https://docs.langchain.com/oss/python/integrations/providers/openai)
            - `azure_ai`                -> [`langchain-azure-ai`](https://docs.langchain.com/oss/python/integrations/providers/microsoft)
            - `google_vertexai`         -> [`langchain-google-vertexai`](https://docs.langchain.com/oss/python/integrations/providers/google)
            - `google_genai`            -> [`langchain-google-genai`](https://docs.langchain.com/oss/python/integrations/providers/google)
            - `bedrock`                 -> [`langchain-aws`](https://docs.langchain.com/oss/python/integrations/providers/aws)
            - `bedrock_converse`        -> [`langchain-aws`](https://docs.langchain.com/oss/python/integrations/providers/aws)
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
            - `perplexity`              -> [`langchain-perplexity`](https://docs.langchain.com/oss/python/integrations/providers/perplexity)
            - `upstage`                 -> [`langchain-upstage`](https://docs.langchain.com/oss/python/integrations/providers/upstage)
        configurable_fields: 运行时可配置的模型参数:

            - `None`: 无配置字段（即固定模型）。
            - `'any'`: 所有字段均可配置。**请参阅下方安全提示。**
            - `list[str] | Tuple[str, ...]`: 指定的字段可配置。

            Fields are assumed to have `config_prefix` stripped if a `config_prefix` is
            specified.

            如果指定了 `model`，则默认值为 `None`。

            如果未指定 `model`，则默认值为 `("model", "model_provider")`。

            !!! warning "安全提示"

                Setting `configurable_fields="any"` means fields like `api_key`,
                `base_url`, etc., can be altered at runtime, potentially redirecting
                model requests to a different service/user.

                Make sure that if you're accepting untrusted configurations that you
                enumerate the `configurable_fields=(...)` explicitly.

        config_prefix: 配置键的可选前缀。

            在同一应用中拥有多个可配置模型时非常有用。

            如果 `'config_prefix'` 是非空字符串，则 `model` 可通过
            `config["configurable"]["{config_prefix}_{param}"]` 键在运行时进行配置。
            请参见下方示例。

            如果 `'config_prefix'` 是空字符串，则 `model` 可通过
            `config["configurable"]["{param}"]` 进行配置。 **kwargs: 传递给底层聊天模型
            `__init__` 方法的额外模型特定关键字参数。常见参数包括：

            - `temperature`: 控制随机性的模型温度。
            - `max_tokens`: 输出 token 的最大数量。
            - `timeout`: 等待响应的最大时间（秒）。
            - `max_retries`: 失败请求的最大重试次数。
            - `base_url`: 自定义 API 端点 URL。
            - `rate_limiter`: 一个
                [`BaseRateLimiter`][langchain_core.rate_limiters.BaseRateLimiter]
                实例，用于控制请求速率。

            Refer to the specific model provider's
            [integration reference](https://reference.langchain.com/python/integrations/)
            for all available parameters.

    Returns:
        A [`BaseChatModel`][langchain_core.language_models.BaseChatModel] corresponding
            to the `model_name` and `model_provider` specified if configurability is
            inferred to be `False`.
            If configurable, a chat model emulator that initializes the
            underlying model at runtime once a config is passed in.

    Raises:
        ValueError: 如果无法推断 `model_provider` 或其不被支持。
        ImportError: 如果模型提供商集成包未安装。

    ???+ example "初始化一个不可配置的模型"

        ```python
        # pip install langchain langchain-openai

        from langchain.chat_models import init_chat_model

        gpt_5 = init_chat_model("openai:gpt-5.5", temperature=0)
        gpt_5.invoke("what's your name")
        ```

    ??? example "部分可配置且无默认值的模型"

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

    ??? example "完全可配置且带有默认值的模型"

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

            location: str = Field(
                ..., description="The city and state, e.g. San Francisco, CA"
            )


        class GetPopulation(BaseModel):
            '''Get the current population in a given location'''

            location: str = Field(
                ..., description="The city and state, e.g. San Francisco, CA"
            )


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
    if model_provider == "openai":
        _check_pkg("langchain_openai", "ChatOpenAI")
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=model, **kwargs)
    if model_provider == "anthropic":
        _check_pkg("langchain_anthropic", "ChatAnthropic")
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(model=model, **kwargs)  # type: ignore[call-arg,unused-ignore]
    if model_provider == "azure_openai":
        _check_pkg("langchain_openai", "AzureChatOpenAI")
        from langchain_openai import AzureChatOpenAI

        return AzureChatOpenAI(model=model, **kwargs)
    if model_provider == "azure_ai":
        _check_pkg("langchain_azure_ai", "AzureAIOpenAIApiChatModel")
        from langchain_azure_ai.chat_models import AzureAIOpenAIApiChatModel

        return AzureAIOpenAIApiChatModel(model=model, **kwargs)
    if model_provider == "cohere":
        _check_pkg("langchain_cohere", "ChatCohere")
        from langchain_cohere import ChatCohere

        return ChatCohere(model=model, **kwargs)
    if model_provider == "google_vertexai":
        _check_pkg("langchain_google_vertexai", "ChatVertexAI")
        from langchain_google_vertexai import ChatVertexAI

        return ChatVertexAI(model=model, **kwargs)
    if model_provider == "google_genai":
        _check_pkg("langchain_google_genai", "ChatGoogleGenerativeAI")
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(model=model, **kwargs)
    if model_provider == "fireworks":
        _check_pkg("langchain_fireworks", "ChatFireworks")
        from langchain_fireworks import ChatFireworks

        return ChatFireworks(model=model, **kwargs)
    if model_provider == "ollama":
        try:
            _check_pkg("langchain_ollama", "ChatOllama")
            from langchain_ollama import ChatOllama
        except ImportError:
                        # 则抛出与 `langchain-ollama` 相关的错误
            try:
                _check_pkg("langchain_community", "ChatOllama")
                from langchain_community.chat_models import ChatOllama
            except ImportError:
                                # TODO：一旦 ChatBedrock 支持 OpenAI
                # 模型（包括新模型和别名），请更新为使用 model=
                                # OpenAI 模型（包括新模型和别名）
                _check_pkg("langchain_ollama", "ChatOllama")

        return ChatOllama(model=model, **kwargs)
    if model_provider == "together":
        _check_pkg("langchain_together", "ChatTogether")
        from langchain_together import ChatTogether

        return ChatTogether(model=model, **kwargs)
    if model_provider == "mistralai":
        _check_pkg("langchain_mistralai", "ChatMistralAI")
        from langchain_mistralai import ChatMistralAI

        return ChatMistralAI(model=model, **kwargs)  # type: ignore[call-arg,unused-ignore]

    if model_provider == "huggingface":
        _check_pkg("langchain_huggingface", "ChatHuggingFace")
        from langchain_huggingface import ChatHuggingFace

        return ChatHuggingFace.from_model_id(model_id=model, **kwargs)

    if model_provider == "groq":
        _check_pkg("langchain_groq", "ChatGroq")
        from langchain_groq import ChatGroq

        return ChatGroq(model=model, **kwargs)
    if model_provider == "bedrock":
        _check_pkg("langchain_aws", "ChatBedrock")
        from langchain_aws import ChatBedrock

                # Anthropic 模型
        return ChatBedrock(model_id=model, **kwargs)
    if model_provider == "bedrock_converse":
        _check_pkg("langchain_aws", "ChatBedrockConverse")
        from langchain_aws import ChatBedrockConverse

        return ChatBedrockConverse(model=model, **kwargs)
    if model_provider == "google_anthropic_vertex":
        _check_pkg("langchain_google_vertexai", "ChatAnthropicVertex")
        from langchain_google_vertexai.model_garden import ChatAnthropicVertex

        return ChatAnthropicVertex(model=model, **kwargs)
    if model_provider == "deepseek":
        _check_pkg("langchain_deepseek", "ChatDeepSeek", pkg_kebab="langchain-deepseek")
        from langchain_deepseek import ChatDeepSeek

        return ChatDeepSeek(model=model, **kwargs)
    if model_provider == "nvidia":
        _check_pkg("langchain_nvidia_ai_endpoints", "ChatNVIDIA")
        from langchain_nvidia_ai_endpoints import ChatNVIDIA

        return ChatNVIDIA(model=model, **kwargs)
    if model_provider == "ibm":
        _check_pkg("langchain_ibm", "ChatWatsonx")
        from langchain_ibm import ChatWatsonx

        return ChatWatsonx(model_id=model, **kwargs)
    if model_provider == "xai":
        _check_pkg("langchain_xai", "ChatXAI")
        from langchain_xai import ChatXAI

        return ChatXAI(model=model, **kwargs)
    if model_provider == "perplexity":
        _check_pkg("langchain_perplexity", "ChatPerplexity")
        from langchain_perplexity import ChatPerplexity

        return ChatPerplexity(model=model, **kwargs)
    if model_provider == "upstage":
        _check_pkg("langchain_upstage", "ChatUpstage")
        from langchain_upstage import ChatUpstage

        return ChatUpstage(model=model, **kwargs)
    supported = ", ".join(_SUPPORTED_PROVIDERS)
    msg = (
        f"Unsupported {model_provider=}.\n\nSupported model providers are: {supported}"
    )
    raise ValueError(msg)


_SUPPORTED_PROVIDERS = {
    "openai",
    "anthropic",
    "azure_openai",
    "azure_ai",
    "cohere",
    "google_vertexai",
    "google_genai",
    "fireworks",
    "ollama",
    "together",
    "mistralai",
    "huggingface",
    "groq",
    "bedrock",
    "bedrock_converse",
    "google_anthropic_vertex",
    "deepseek",
    "ibm",
    "xai",
    "perplexity",
    "upstage",
}


def _attempt_infer_model_provider(model_name: str) -> str | None:
    """尝试从模型名称推断模型提供商。

    Args:
        model_name: 用于推断提供商的模型名称。

    Returns:
        推断出的提供商名称，如果无法推断则为 `None`。
    """
    model_lower = model_name.lower()

        # Cohere 模型
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

        # Fireworks 模型
    if model_lower.startswith("claude"):
        return "anthropic"

        # Google 模型
    if model_lower.startswith("command"):
        return "cohere"

        # AWS Bedrock 模型
    if model_name.startswith("accounts/fireworks"):
        return "fireworks"

        # Mistral 模型
    if model_lower.startswith("gemini"):
        return "google_vertexai"

        # DeepSeek 模型
    if model_name.startswith("amazon.") or model_lower.startswith(
        (
            "anthropic.",
            "meta.",
        )
    ):
        return "bedrock"

        # xAI 模型
    if model_lower.startswith(("mistral", "mixtral")):
        return "mistralai"

        # Perplexity 模型
    if model_lower.startswith("deepseek"):
        return "deepseek"

        # Upstage 模型
    if model_lower.startswith("grok"):
        return "xai"

        # 规范化提供商名称
    if model_lower.startswith("sonar"):
        return "perplexity"

        # 在实际模型对象实例化之前无法应用的声明式操作。因此，我们不直接返回实际操作，
    if model_lower.startswith("solar"):
        return "upstage"

    return None


def _parse_model(model: str, model_provider: str | None) -> tuple[str, str]:
    """解析模型名称和提供商，必要时进行提供商推断。"""
    if not model_provider and ":" in model:
        prefix, suffix = model.split(":", 1)
        if prefix in _SUPPORTED_PROVIDERS:
            model_provider = prefix
            model = suffix
        else:
            inferred = _attempt_infer_model_provider(prefix)
            if inferred:
                model_provider = inferred
                model = suffix

    model_provider = model_provider or _attempt_infer_model_provider(model)
    if not model_provider:
        supported_list = ", ".join(sorted(_SUPPORTED_PROVIDERS))
        msg = (
            f"Unable to infer model provider for {model=}. "
            f"Please specify 'model_provider' directly.\n\n"
            f"Supported providers: {supported_list}\n\n"
            f"For help with specific providers, see: "
            f"https://docs.langchain.com/oss/python/integrations/providers"
        )
        raise ValueError(msg)

        # 而是将操作及其参数记录在队列中。该队列将在我们实际实例化模型时（在 self._model()
    # 中）按顺序应用。
    model_provider = model_provider.replace("-", "_").lower()
    return model, model_provider


def _check_pkg(pkg: str, class_name: str, *, pkg_kebab: str | None = None) -> None:
    if not util.find_spec(pkg):
        pkg_kebab = pkg_kebab if pkg_kebab is not None else pkg.replace("_", "-")
        msg = (
            f"Initializing {class_name} requires the {pkg_kebab} package. "
            f"Please install it with `pip install {pkg_kebab}`"
        )
        raise ImportError(msg)


_DECLARATIVE_METHODS = ("bind_tools", "with_structured_output")


class _ConfigurableModel(Runnable[LanguageModelInput, Any]):
    def __init__(
        self,
        *,
        default_config: dict | None = None,
        configurable_fields: Literal["any"] | list[str] | tuple[str, ...] = "any",
        config_prefix: str = "",
        queued_declarative_operations: Sequence[tuple[str, tuple, dict]] = (),
    ) -> None:
        self._default_config: dict = default_config or {}
        self._configurable_fields: Literal["any"] | list[str] = (
            configurable_fields
            if configurable_fields == "any"
            else list(configurable_fields)
        )
        self._config_prefix = (
            config_prefix + "_"
            if config_prefix and not config_prefix.endswith("_")
            else config_prefix
        )
        self._queued_declarative_operations: list[tuple[str, tuple, dict]] = list(
            queued_declarative_operations,
        )

    def __getattr__(self, name: str) -> Any:
        if name in _DECLARATIVE_METHODS:
                        # 这是 LanguageModelInput 的一个版本，它用其子类的联合类型替代了抽象基类
            # BaseMessage，从而实现了更优的模式。
                        # 如果配置数 <= 1，则使用底层模型的批处理实现。
                        # 如果多个配置，则默认使用 Runnable.batch，通过执行器并行调用。
                        # 如果配置数 <= 1，则使用底层模型的批处理实现。
                        # 如果多个配置，则默认使用 Runnable.batch，通过执行器并行调用。
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

    def _model(self, config: RunnableConfig | None = None) -> Runnable:
        params = {**self._default_config, **self._model_params(config)}
        model = _init_chat_model_helper(**params)
        for name, args, kwargs in self._queued_declarative_operations:
            model = getattr(model, name)(*args, **kwargs)
        return model

    def _model_params(self, config: RunnableConfig | None) -> dict:
        config = ensure_config(config)
        model_params = {
            k.removeprefix(self._config_prefix): v
            for k, v in config.get("configurable", {}).items()
            if k.startswith(self._config_prefix)
        }
        if self._configurable_fields != "any":
            model_params = {
                k: v for k, v in model_params.items() if k in self._configurable_fields
            }
        return model_params

    def with_config(
        self,
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> _ConfigurableModel:
        """将配置绑定到一个 `Runnable`，返回一个新的 `Runnable`。"""
        config = RunnableConfig(**(config or {}), **cast("RunnableConfig", kwargs))
        model_params = self._model_params(config)
        remaining_config = {k: v for k, v in config.items() if k != "configurable"}
        remaining_config["configurable"] = {
            k: v
            for k, v in config.get("configurable", {}).items()
            if k.removeprefix(self._config_prefix) not in model_params
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
        from langchain_core.prompt_values import (
            ChatPromptValueConcrete,
            StringPromptValue,
        )

                # 如果配置数 <= 1，则使用底层模型的批处理实现。
                # 如果多个配置，则默认使用 Runnable.batch，通过执行器并行调用。
                # 如果配置数 <= 1，则使用底层模型的批处理实现。
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
                # 如果多个配置，则默认使用 Runnable.batch，通过执行器并行调用。
        if config is None or isinstance(config, dict) or len(config) <= 1:
            if isinstance(config, list):
                config = config[0]
            return self._model(config).batch(
                inputs,
                config=config,
                return_exceptions=return_exceptions,
                **kwargs,
            )
                # 显式添加以满足下游 linters 要求。
                # 显式添加以满足下游 linters 要求。
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
                # 如果配置数 <= 1，则使用底层模型的批处理实现。
        if config is None or isinstance(config, dict) or len(config) <= 1:
            if isinstance(config, list):
                config = config[0]
            return await self._model(config).abatch(
                inputs,
                config=config,
                return_exceptions=return_exceptions,
                **kwargs,
            )
                # 如果多个配置默认使用 Runnable.batch，该方法使用执行器来调用
                # 并行地。
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
                # 如果配置数 <= 1，则使用底层模型的批处理实现。
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
    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type[BaseModel] | Callable | BaseTool],
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, AIMessage]:
        return self.__getattr__("bind_tools")(tools, **kwargs)

        # 显式添加以满足下游 linter 的要求。
    def with_structured_output(
        self,
        schema: dict | type[BaseModel],
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, dict | BaseModel]:
        return self.__getattr__("with_structured_output")(schema, **kwargs)

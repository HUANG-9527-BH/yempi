"""从提供方内容派生的标准内容块。

`AIMessage` will first attempt to use a provider-specific translator if
`model_provider` is set in `response_metadata` on the message. Consequently, each
provider translator must handle all possible content response types from the provider,
including text.

If no provider is set, or if the provider does not have a registered translator,
`AIMessage` will fall back to best-effort parsing of the content into blocks using
the implementation in `BaseMessage`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

    from langchain_core.messages import AIMessage, AIMessageChunk
    from langchain_core.messages import content as types

# 提供方到转换器的映射
PROVIDER_TRANSLATORS: dict[str, dict[str, Callable[..., list[types.ContentBlock]]]] = {}
"""Map model provider names to translator functions.

The dictionary maps provider names (e.g. `'openai'`, `'anthropic'`) to another
dictionary with two keys:
- `'translate_content'`: Function to translate `AIMessage` content.
- `'translate_content_chunk'`: Function to translate `AIMessageChunk` content.

When calling `content_blocks` on an `AIMessage` or `AIMessageChunk`, if
`model_provider` is set in `response_metadata`, the corresponding translator
functions will be used to parse the content into blocks. Otherwise, best-effort parsing
in `BaseMessage` will be used.
"""


def register_translator(
    provider: str,
    translate_content: Callable[[AIMessage], list[types.ContentBlock]],
    translate_content_chunk: Callable[[AIMessageChunk], list[types.ContentBlock]],
) -> None:
    """在 `PROVIDER_TRANSLATORS` 中为提供方注册内容转换器。

    Args:
        provider: 模型提供方名称（例如 `'openai'`, `'anthropic'`）。
        translate_content: 用于转换 `AIMessage` 内容的函数。
        translate_content_chunk: 用于转换 `AIMessageChunk` 内容的函数。
    """
    PROVIDER_TRANSLATORS[provider] = {
        "translate_content": translate_content,
        "translate_content_chunk": translate_content_chunk,
    }


def get_translator(
    provider: str,
) -> dict[str, Callable[..., list[types.ContentBlock]]] | None:
    """获取提供方的转换器函数。

    Args:
        provider: 模型提供方名称。

    Returns:
        Dictionary with `'translate_content'` and `'translate_content_chunk'`
        functions, or None if no translator is registered for the provider. In such
        case, best-effort parsing in `BaseMessage` will be used.
    """
    return PROVIDER_TRANSLATORS.get(provider)


def _register_translators() -> None:
    """在 langchain-core 中注册所有转换器。

    单元测试确保 `block_translators` 中的所有模块都在此处有对应项。

    For translators implemented outside langchain-core, they can be registered by
    calling `register_translator` from within the integration package.
    """
    from langchain_core.messages.block_translators.anthropic import (  # noqa: PLC0415
        _register_anthropic_translator,
    )
    from langchain_core.messages.block_translators.bedrock import (  # noqa: PLC0415
        _register_bedrock_translator,
    )
    from langchain_core.messages.block_translators.bedrock_converse import (  # noqa: PLC0415
        _register_bedrock_converse_translator,
    )
    from langchain_core.messages.block_translators.google_genai import (  # noqa: PLC0415
        _register_google_genai_translator,
    )
    from langchain_core.messages.block_translators.google_vertexai import (  # noqa: PLC0415
        _register_google_vertexai_translator,
    )
    from langchain_core.messages.block_translators.groq import (  # noqa: PLC0415
        _register_groq_translator,
    )
    from langchain_core.messages.block_translators.openai import (  # noqa: PLC0415
        _register_openai_translator,
    )

    _register_bedrock_translator()
    _register_bedrock_converse_translator()
    _register_anthropic_translator()
    _register_google_genai_translator()
    _register_google_vertexai_translator()
    _register_groq_translator()
    _register_openai_translator()


_register_translators()

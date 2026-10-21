"""从 Google（VertexAI）内容块派生的标准内容块。"""

from langchain_core.messages.block_translators.google_genai import (
    translate_content,
    translate_content_chunk,
)


def _register_google_vertexai_translator() -> None:
    """将 Google（VertexAI）翻译器注册到中央注册表中。

    模块导入时自动运行。
    """
    from langchain_core.messages.block_translators import (  # noqa: PLC0415
        register_translator,
    )

    register_translator("google_vertexai", translate_content, translate_content_chunk)


_register_google_vertexai_translator()

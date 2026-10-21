"""从 v1 内容块转换为 Ollama SDK 格式。"""

from typing import Any

from langchain_core.messages import content as types


def _convert_from_v1_to_ollama(
    content: list[types.ContentBlock],
    model_provider: str | None,  # noqa: ARG001
) -> list[dict[str, Any]]:
    """将 v1 内容块转换为 Ollama 格式。

    Args:
        content: v1 `ContentBlock` 对象列表。
        model_provider: 生成 v1 内容的模型提供商名称。

    Returns:
        Ollama 格式的内容块列表。
    """
    new_content: list = []
    for block in content:
        if not isinstance(block, dict) or "type" not in block:
            continue

        block_dict = dict(block)  # (用于类型标注)

                # TextContentBlock
        if block_dict["type"] == "text":
                        # 注意：这会丢弃所有其他字段/附加信息
            new_content.append({"type": "text", "text": block_dict["text"]})

                # ReasoningContentBlock
                # Ollama 不接受推理内容返回
                # 未来可考虑将其强制转换为文本作为选项？
        # e.g.:
                # if block_dict["type"] == "reasoning":
                #     # 尝试以文本形式保留内容
                #     new_content.append({"text": str(block_dict["reasoning"])})

                # ImageContentBlock
        if block_dict["type"] == "image":
                        # 已在 _get_image_from_data_content_block 中处理
            new_content.append(block_dict)

                # TODO: 模型支持后添加 AudioContentBlock

                # TODO: 模型支持后添加 FileContentBlock

                # ToolCall -> ???
                # if block_dict["type"] == "tool_call":
                #     function_call = {}
                #     new_content.append(function_call)

                # ToolCallChunk -> ???
                # elif block_dict["type"] == "tool_call_chunk":
                #     function_call = {}
                #     new_content.append(function_call)

                # NonStandardContentBlock
        if block_dict["type"] == "non_standard":
                        # 尝试以文本形式保留内容
            new_content.append(
                {"type": "text", "text": str(block_dict.get("value", ""))}
            )

    return new_content

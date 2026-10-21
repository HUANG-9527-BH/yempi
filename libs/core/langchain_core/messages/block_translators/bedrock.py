"""从 Bedrock 内容派生的标准内容块。"""

from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.messages import content as types
from langchain_core.messages.block_translators.anthropic import (
    _convert_to_v1_from_anthropic,
)


def _convert_to_v1_from_bedrock(message: AIMessage) -> list[types.ContentBlock]:
    """将 Bedrock 消息内容转换为 v1 格式。"""
    out = _convert_to_v1_from_anthropic(message)

    content_tool_call_ids = {
        block.get("id")
        for block in out
        if isinstance(block, dict) and block.get("type") == "tool_call"
    }
    for tool_call in message.tool_calls:
        if (id_ := tool_call.get("id")) and id_ not in content_tool_call_ids:
            tool_call_block: types.ToolCall = {
                "type": "tool_call",
                "id": id_,
                "name": tool_call["name"],
                "args": tool_call["args"],
            }
            if "index" in tool_call:
                tool_call_block["index"] = tool_call["index"]  # type: ignore[typeddict-item]
            if "extras" in tool_call:
                tool_call_block["extras"] = tool_call["extras"]  # type: ignore[typeddict-item]
            out.append(tool_call_block)
    return out


def _convert_to_v1_from_bedrock_chunk(
    message: AIMessageChunk,
) -> list[types.ContentBlock]:
    """将 Bedrock 消息块内容转换为 v1 格式。"""
    if (
        message.content == ""
        and not message.additional_kwargs
        and not message.tool_calls
    ):
                # Bedrock 输出多个包含响应元数据的块
        return []

    out = _convert_to_v1_from_anthropic(message)

    if (
        message.tool_call_chunks
        and not message.content
        and message.chunk_position != "last"  # 聚合时保留 tool_calls
    ):
        for tool_call_chunk in message.tool_call_chunks:
            tc: types.ToolCallChunk = {
                "type": "tool_call_chunk",
                "id": tool_call_chunk.get("id"),
                "name": tool_call_chunk.get("name"),
                "args": tool_call_chunk.get("args"),
            }
            if (idx := tool_call_chunk.get("index")) is not None:
                tc["index"] = idx
            out.append(tc)
    return out


def translate_content(message: AIMessage) -> list[types.ContentBlock]:
    """从包含 Bedrock 内容的消息中派生标准内容块。

    Args:
        message: 需要转换的消息。

    Returns:
        派生出的内容块。
    """
    if "claude" not in message.response_metadata.get("model_name", "").lower():
        raise NotImplementedError  # 回退到尽力解析
    return _convert_to_v1_from_bedrock(message)


def translate_content_chunk(message: AIMessageChunk) -> list[types.ContentBlock]:
    """从包含 Bedrock 内容的消息块中派生标准内容块。

    Args:
        message: 需要转换的消息块。

    Returns:
        派生出的内容块。
    """
        # TODO: 将 model_name 添加到所有 Bedrock 块中，并更新核心合并逻辑
        # 在聚合时不追加。如果非 Anthropic 模型，则在此处抛出 NotImplementedError
    # 以回退到尽力解析。
        # 不追加。然后，如果不是 Anthropic 模型，则在此处抛出 NotImplementedError
    # 以回退到尽力解析。
    return _convert_to_v1_from_bedrock_chunk(message)


def _register_bedrock_translator() -> None:
    """将 Bedrock 转换器注册到中央注册表中。

    模块导入时自动运行。
    """
    from langchain_core.messages.block_translators import (  # noqa: PLC0415
        register_translator,
    )

    register_translator("bedrock", translate_content, translate_content_chunk)


_register_bedrock_translator()

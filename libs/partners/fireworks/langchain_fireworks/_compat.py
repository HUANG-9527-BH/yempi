"""根据 `output_version` 控制在 AIMessage 输出格式之间的转换。"""

from __future__ import annotations

from langchain_core.messages import AIMessage


def _convert_from_v1_to_chat_completions(message: AIMessage) -> AIMessage:
    """将 v1 消息转换为聊天补全格式。"""
    if isinstance(message.content, list):
        new_content: list = []
        for block in message.content:
            if isinstance(block, dict):
                block_type = block.get("type")
                if block_type == "text":
                                        # 去除注释
                    new_content.append({"type": "text", "text": block["text"]})
                elif block_type in ("reasoning", "tool_call", "invalid_tool_call"):
                    pass
                else:
                    new_content.append(block)
            else:
                new_content.append(block)
        return message.model_copy(update={"content": new_content})

    return message

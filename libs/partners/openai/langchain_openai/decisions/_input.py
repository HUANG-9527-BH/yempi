"""将 `OpenAIDecisions` 状态转换为 Decisions API 的 `input` 字段。"""

from __future__ import annotations

import json
import re
import uuid
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    convert_to_openai_messages,
)

if TYPE_CHECKING:
    from langchain_openai.decisions.types import State


def to_decision_input(state: State) -> str | list[dict[str, Any]]:
    """将状态转换为字符串或用户消息列表。

    Strings are sent as is, and sequences of `HumanMessage` objects as user messages.
    Anything else, such as conversations with system, AI, or tool messages, is
    serialized to JSON text in a single user message, with base64 images from any
    message kept in place as image parts.

    Args:
        state: 需要评估的文本、消息或结构化状态。

    Returns:
        适用于 Decisions API `input` 字段的值。

    Raises:
        ValueError: 如果消息包含 API 不接受的内容。
        TypeError: 如果状态无法表示为 JSON。
    """
    if isinstance(state, str):
        return state
    messages = [state] if isinstance(state, BaseMessage) else state
    if (
        isinstance(messages, Sequence)
        and messages
        and all(isinstance(message, HumanMessage) for message in messages)
    ):
        return [_to_user_message(message) for message in messages]  # type: ignore[arg-type]
    return [_to_json_user_message(state)]


def _to_user_message(message: HumanMessage) -> dict[str, Any]:
    content = convert_to_openai_messages(message)["content"]
    if isinstance(content, str):
        return {"role": "user", "content": content}
    return {"role": "user", "content": [_to_input_part(block) for block in content]}


def _to_input_part(block: dict[str, Any]) -> dict[str, Any]:
    if block.get("type") == "text":
        return {"type": "input_text", "text": block["text"]}
    if block.get("type") == "image_url":
        url = block["image_url"]["url"]
        if not url.startswith("data:"):
            msg = (
                "The OpenAI Decisions API only accepts images as base64 data URLs. "
                "Hosted image URLs are not supported."
            )
            raise ValueError(msg)
        return {"type": "input_image", "image_url": url}
    msg = (
        f"Unsupported content block type for the OpenAI Decisions API: "
        f"{block.get('type')!r}. Only text and base64 images are supported."
    )
    raise ValueError(msg)


def _to_json_user_message(state: State) -> dict[str, Any]:
    """将 API 无法原生接受的状态序列化为单个用户消息。

    Covers conversations with non-user messages and JSON objects or arrays. Messages
    at any depth are converted to role/content objects. Base64 images in any message
    are sent as `input_image` parts at the position where they appear in the text, so
    the format stays the same whether or not a conversation contains images.
    """
    images: list[str] = []
    token = f"lc-image-{uuid.uuid4().hex}"
    text = json.dumps(_to_json_value(state, images, token), ensure_ascii=False)
    content: list[dict[str, Any]] = []
    position = 0
    for match in re.finditer(rf"{token}-(\d+)", text):
        content.append({"type": "input_text", "text": text[position : match.start()]})
        content.append({"type": "input_image", "image_url": images[int(match[1])]})
        position = match.end()
    content.append({"type": "input_text", "text": text[position:]})
    return {"role": "user", "content": content}


def _to_json_value(value: object, images: list[str], token: str) -> Any:
    if isinstance(value, BaseMessage):
        return convert_to_openai_messages(_with_image_tokens(value, images, token))
    if isinstance(value, dict):
        if not all(isinstance(key, str) for key in value):
            msg = "OpenAI Decisions state object keys must be strings."
            raise TypeError(msg)
        return {key: _to_json_value(item, images, token) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_to_json_value(item, images, token) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    msg = f"Unsupported OpenAI Decisions state value: {type(value).__name__}."
    raise TypeError(msg)


def _with_image_tokens(
    message: BaseMessage, images: list[str], token: str
) -> BaseMessage:
    """用标记每个图像位置的令牌替换 base64 图像块。"""
    if isinstance(message.content, str):
        return message
    content: list[str | dict[str, Any]] = []
    for block in message.content:
        url = _base64_image_url(block) if isinstance(block, dict) else None
        if url is None:
            content.append(block)
        else:
            content.append({"type": "text", "text": f"{token}-{len(images)}"})
            images.append(url)
    return message.model_copy(update={"content": content})


def _base64_image_url(block: dict[str, Any]) -> str | None:
    """返回 base64 图像块的数据 URL，否则返回 `None`。"""
    if block.get("type") not in {"image", "image_url"}:
        return None
    try:
        part = convert_to_openai_messages(HumanMessage(content=[block]))["content"][0]
    except (ValueError, KeyError, TypeError):
        return None
    url = part.get("image_url", {}).get("url", "") if isinstance(part, dict) else ""
    return url if url.startswith("data:") else None

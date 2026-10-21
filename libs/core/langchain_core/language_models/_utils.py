import re
from collections.abc import Sequence
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    TypedDict,
    TypeVar,
)

if TYPE_CHECKING:
    from langchain_core.messages import BaseMessage
from langchain_core.messages.content import (
    ContentBlock,
)


def _filter_invocation_params_for_tracing(params: dict[str, Any]) -> dict[str, Any]:
    """从调用参数中过滤掉大或不合适的字段，以用于追踪。

    移除 tools、functions、messages、response_format 等可能较大的字段。

    Args:
        params: 需要过滤的调用参数。

    Returns:
        过滤后的参数，已移除大字段。
    """
    excluded_keys = {"tools", "functions", "messages", "response_format"}
    return {k: v for k, v in params.items() if k not in excluded_keys}


def is_openai_data_block(
    block: dict[str, Any], filter_: Literal["image", "audio", "file"] | None = None
) -> bool:
    """检查一个内容块是否包含 OpenAI 聊天补全格式的多模态数据。

    支持 data 和 ID 风格的内容块（例如 `'file_data'` 和 `'file_id'`）

    如果存在其他键，它们将被忽略 / 不会影响结果，只要必需的键存在且有效即可。

    Args:
        block: The content block to check.
        filter_: If provided, only return True for blocks matching this specific type.
            - "image": Only match image_url blocks
            - "audio": Only match input_audio blocks
            - "file": Only match file blocks
            If `None`, match any valid OpenAI data block type. Note that this means that
            if the block has a valid OpenAI data type but the filter_ is set to a
            different type, this function will return False.

    Returns:
        `True` if the block is a valid OpenAI data block and matches the filter_
        (if provided).

    """
    if block.get("type") == "image_url":
        if filter_ is not None and filter_ != "image":
            return False
        if (
            (set(block.keys()) <= {"type", "image_url", "detail"})
            and (image_url := block.get("image_url"))
            and isinstance(image_url, dict)
        ):
            url = image_url.get("url")
            if isinstance(url, str):
                                # OpenAI 规范要求
                return True
                        # 忽略 `'detail'`，因为它为可选且专属于 OpenAI

    elif block.get("type") == "input_audio":
        if filter_ is not None and filter_ != "audio":
            return False
        if (audio := block.get("input_audio")) and isinstance(audio, dict):
            audio_data = audio.get("data")
            audio_format = audio.get("format")
                        # 两者均为 OpenAI 规范要求
            if isinstance(audio_data, str) and isinstance(audio_format, str):
                return True

    elif block.get("type") == "file":
        if filter_ is not None and filter_ != "file":
            return False
        if (file := block.get("file")) and isinstance(file, dict):
            file_data = file.get("file_data")
            file_id = file.get("file_id")
                        # 文件可以是 base64 编码的，也可以是预上传的 ID
            if isinstance(file_data, str) or isinstance(file_id, str):
                return True

    else:
        return False

        # 没有 `'type'` 键
    return False


class ParsedDataUri(TypedDict):
    source_type: Literal["base64"]
    data: str
    mime_type: str


def _parse_data_uri(uri: str) -> ParsedDataUri | None:
    """将数据 URI 解析为其组成部分。

    If parsing fails, return `None`. If either MIME type or data is missing, return
    `None`.

    Example:
        ```python
        data_uri = "data:image/jpeg;base64,/9j/4AAQSkZJRg..."
        parsed = _parse_data_uri(data_uri)

        assert parsed == {
            "source_type": "base64",
            "mime_type": "image/jpeg",
            "data": "/9j/4AAQSkZJRg...",
        }
        ```
    """
    regex = r"^data:(?P<mime_type>[^;]+);base64,(?P<data>.+)$"
    match = re.match(regex, uri)
    if match is None:
        return None

    mime_type = match.group("mime_type")
    data = match.group("data")
    if not mime_type or not data:
        return None

    return {
        "source_type": "base64",
        "data": data,
        "mime_type": mime_type,
    }


def _normalize_messages(
    messages: Sequence["BaseMessage"],
) -> list["BaseMessage"]:
    """将消息格式标准化为 LangChain v1 标准内容块。

    聊天模型已支持：
    - OpenAI 聊天补全格式中的图像
        这些将原样传递
    - LangChain v1 标准内容块

    This function extends support to:
    - `[Audio](https://platform.openai.com/docs/api-reference/chat/create) and
        `[file](https://platform.openai.com/docs/api-reference/files) data in OpenAI
        Chat Completions format
        - Images are technically supported but we expect chat models to handle them
            directly; this may change in the future
    - LangChain v0 standard content blocks for backward compatibility

    !!! warning "在 `langchain-core` 1.0.0 中行为已更改"

        在旧版本中，此函数返回 LangChain v0 格式的消息。 现在，它返回 LangChain v1
        格式的消息，升级后的聊天模型现在期望接收此类格式的消息以用于消息历史回传。为向后兼容，此函数会将
        v0 消息内容转换为 v1 格式。

    ??? note "v0 内容块模式"

        `URLContentBlock`:

        ```python
        {
            mime_type: NotRequired[str]
            type: Literal['image', 'audio', 'file'],
            source_type: Literal['url'],
            url: str,
        }
        ```

        `Base64ContentBlock`:

        ```python
        {
            mime_type: NotRequired[str]
            type: Literal['image', 'audio', 'file'],
            source_type: Literal['base64'],
            data: str,
        }
        ```

        `IDContentBlock`:

        (实际上从未使用过)

        ```python
        {
            type: Literal["image", "audio", "file"],
            source_type: Literal["id"],
            id: str,
        }
        ```

        `PlainTextContentBlock`:

        ```python
        {
            mime_type: NotRequired[str]
            type: Literal['file'],
            source_type: Literal['text'],
            url: str,
        }
        ```

    If a v1 message is passed in, it will be returned as-is, meaning it is safe to
    always pass in v1 messages to this function for assurance.

    为存档起见，以下是我们在 OpenAI 聊天补全格式中期望的模式：

    聊天补全中的图像。可以是基于 URL 的，也可以是 base64 编码的。支持 MIME 类型
    png、jpeg/jpg、webp、静态 gif：
    {
        "type": Literal['image_url'],
        "image_url": {
            "url": Union["data:$MIME_TYPE;base64,$BASE64_ENCODED_IMAGE", "$IMAGE_URL"],
            "detail": Literal['low', 'high', 'auto'] = 'auto',  # OpenAI 支持
        }
    }

    聊天补全中的音频：
    {
        "type": Literal['input_audio'],
        "input_audio": {
            "format": Literal['wav', 'mp3'],
            "data": str = "$BASE64_ENCODED_AUDIO",
        },
    }

    Chat Completions files: 聊天补全中的文件：可以是 base64 编码的，也可以是预上传的文件 ID
    {
        "type": Literal['file'],
        "file": Union[
            {
                "filename": str | None = "$FILENAME",
                "file_data": str = "$BASE64_ENCODED_FILE",
            },
            {
                "file_id": str = "$FILE_ID",  # 用于预上传至 OpenAI 的文件
            },
        ],
    }

    """
    from langchain_core.messages.block_translators.langchain_v0 import (  # noqa: PLC0415
        _convert_legacy_v0_content_block_to_v1,
    )
    from langchain_core.messages.block_translators.openai import (  # noqa: PLC0415
        _convert_openai_format_to_data_block,
    )

    formatted_messages = []
    for message in messages:
                # 我们保留输入的消息——调用方可能在其他地方重用它们，并期望
                # 它们保持不变。我们仅在需要转换时才创建副本。
        formatted_message = message

        if isinstance(message.content, list):
            for idx, block in enumerate(message.content):
                                # OpenAI 聊天补全多模态数据块转为 v1 标准
                if (
                    isinstance(block, dict)
                    and block.get("type") in {"input_audio", "file"}
                                        # 区分 OpenAI/LC 格式，因为它们共享 `'type'`
                    and is_openai_data_block(block)
                ):
                    formatted_message = _ensure_message_copy(message, formatted_message)

                    converted_block = _convert_openai_format_to_data_block(block)
                    _update_content_block(formatted_message, idx, converted_block)

                                # 将多模态 LangChain v0 转换为 v1 标准内容块
                elif (
                    isinstance(block, dict)
                    and block.get("type")
                    in {
                        "image",
                        "audio",
                        "file",
                    }
                    and block.get("source_type")  # v1 不包含 `source_type`
                    in {
                        "url",
                        "base64",
                        "id",
                        "text",
                    }
                ):
                    formatted_message = _ensure_message_copy(message, formatted_message)

                    converted_block = _convert_legacy_v0_content_block_to_v1(block)
                    _update_content_block(formatted_message, idx, converted_block)
                    continue

                                # 否则，直接传递看起来符合 v1 格式的块而不做更改

        formatted_messages.append(formatted_message)

    return formatted_messages


T = TypeVar("T", bound="BaseMessage")


def _ensure_message_copy(message: T, formatted_message: T) -> T:
    """创建消息的副本，如果尚未复制的话。"""
    if formatted_message is message:
        formatted_message = message.model_copy()
                # 浅拷贝内容列表以允许修改
        formatted_message.content = list(formatted_message.content)
    return formatted_message


def _update_content_block(
    formatted_message: "BaseMessage", idx: int, new_block: ContentBlock | dict[str, Any]
) -> None:
    """更新指定索引处的内容块，处理类型问题。"""
        # 类型忽略所需原因如下：
        # - `BaseMessage.content` 的类型为 `Union[str, list[Union[str, dict]]]`
        # - 当 content 为 str 时，索引会出错（索引错误）
        # - 当 content 为 list 时，项为 `Union[str, dict]`，但我们正在赋值
        #   `Union[ContentBlock, dict]`，其中 ContentBlock 比 dict 更丰富
        # - 这是安全的，因为我们仅在已验证 content 为 list 且
        #   正在进行内容块转换时调用此函数
    formatted_message.content[idx] = new_block  # type: ignore[index, assignment]


def _update_message_content_to_blocks(message: T, output_version: str) -> T:
    return message.model_copy(
        update={
            "content": message.content_blocks,
            "response_metadata": {
                **message.response_metadata,
                "output_version": output_version,
            },
        }
    )

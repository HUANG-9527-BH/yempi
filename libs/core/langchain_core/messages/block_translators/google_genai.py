"""Google（GenAI）内容的标准内容块的衍生。"""

import base64
import re
from collections.abc import Iterator
from typing import Any, cast

from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.messages import content as types
from langchain_core.messages.content import Citation, create_citation

try:
    import filetype  # type: ignore[import-not-found]

    _HAS_FILETYPE = True
except ImportError:
    _HAS_FILETYPE = False


def _bytes_to_b64_str(bytes_: bytes) -> str:
    """将字节转换为 base64 编码的字符串。"""
    return base64.b64encode(bytes_).decode("utf-8")


def translate_grounding_metadata_to_citations(
    grounding_metadata: dict[str, Any],
) -> list[Citation]:
    """将 Google AI 基线元数据转换为 LangChain 引用。

    Args:
        grounding_metadata: 包含网络搜索查询、基线块和基线支持的 Google AI 基线元数据。

    Returns:
        从基线元数据派生的引用内容块列表。

    Example:
        >>> metadata = {
        ...     "web_search_queries": ["UEFA Euro 2024 winner"],
        ...     "grounding_chunks": [
        ...         {
        ...             "web": {
        ...                 "uri": "https://uefa.com/euro2024",
        ...                 "title": "UEFA Euro 2024 Results",
        ...             }
        ...         }
        ...     ],
        ...     "grounding_supports": [
        ...         {
        ...             "segment": {
        ...                 "start_index": 0,
        ...                 "end_index": 47,
        ...                 "text": "Spain won the UEFA Euro 2024 championship",
        ...             },
        ...             "grounding_chunk_indices": [0],
        ...         }
        ...     ],
        ... }
        >>> citations = translate_grounding_metadata_to_citations(metadata)
        >>> len(citations)
        1
        >>> citations[0]["url"]
        'https://uefa.com/euro2024'
    """
    if not grounding_metadata:
        return []

    grounding_chunks = grounding_metadata.get("grounding_chunks", [])
    grounding_supports = grounding_metadata.get("grounding_supports", [])
    web_search_queries = grounding_metadata.get("web_search_queries", [])

    citations: list[Citation] = []

    for support in grounding_supports:
        segment = support.get("segment", {})
        chunk_indices = support.get("grounding_chunk_indices", [])

        start_index = segment.get("start_index")
        end_index = segment.get("end_index")
        cited_text = segment.get("text")

                # 为每个引用的块创建引用
        for chunk_index in chunk_indices:
            if chunk_index < len(grounding_chunks):
                chunk = grounding_chunks[chunk_index]

                                # 处理网络和地图基线
                web_info = chunk.get("web") or {}
                maps_info = chunk.get("maps") or {}

                                # 根据来源提取引用信息
                url = maps_info.get("uri") or web_info.get("uri")
                title = maps_info.get("title") or web_info.get("title")

                                # 注意：confidence_scores 是 Gemini 2.0 及更早版本的遗留字段
                                # 用于表示每个基线块的置信度（0.0-1.0）。
                #
                                # 在 Gemini 2.5+ 中，该字段始终为 None/空，应忽略。
                extras_metadata = {
                    "web_search_queries": web_search_queries,
                    "grounding_chunk_index": chunk_index,
                    "confidence_scores": support.get("confidence_scores") or [],
                }

                                # 如果存在，添加地图专用元数据
                if maps_info.get("placeId"):
                    extras_metadata["place_id"] = maps_info["placeId"]

                citation = create_citation(
                    url=url,
                    title=title,
                    start_index=start_index,
                    end_index=end_index,
                    cited_text=cited_text,
                    google_ai_metadata=extras_metadata,
                )
                citations.append(citation)

    return citations


def _convert_to_v1_from_genai_input(
    content: list[types.ContentBlock],
) -> list[types.ContentBlock]:
    """将 Google GenAI 格式块转换为 v1 格式。

    当消息不是 `AIMessage` 或 `response_metadata` 中未设置 `model_provider` 时调用。

    During the `content_blocks` parsing process, we wrap blocks not recognized as a v1
    block as a `'non_standard'` block with the original block stored in the `value`
    field. This function attempts to unpack those blocks and convert any blocks that
    might be GenAI format to v1 ContentBlocks.

    如果转换失败，该块将保持为 `'non_standard'` 块。

    Args:
        content: 需要处理的内容块列表。

    Returns:
        已更新的列表，其中 GenAI 块已转换为 v1 格式。
    """

    def _iter_blocks() -> Iterator[types.ContentBlock]:
        blocks: list[dict[str, Any]] = [
            cast("dict[str, Any]", block)
            if block.get("type") != "non_standard"
            else block["value"]  # type: ignore[typeddict-item]  # this is only non-standard blocks
            for block in content
        ]
        for block in blocks:
            num_keys = len(block)
            block_type = block.get("type")

            if num_keys == 1 and (text := block.get("text")):
                                # 这很可能是一个 TextContentBlock
                yield {"type": "text", "text": text}

            elif (
                num_keys == 1
                and (document := block.get("document"))
                and isinstance(document, dict)
                and "format" in document
            ):
                                # 处理文档格式转换
                doc_format = document.get("format")
                source = document.get("source", {})

                if doc_format == "pdf" and "bytes" in source:
                                        # 包含字节数据的 PDF 文档
                    file_block: types.FileContentBlock = {
                        "type": "file",
                        "base64": source["bytes"]
                        if isinstance(source["bytes"], str)
                        else _bytes_to_b64_str(source["bytes"]),
                        "mime_type": "application/pdf",
                    }
                                        # 保留额外字段
                    extras = {
                        key: value
                        for key, value in document.items()
                        if key not in {"format", "source"}
                    }
                    if extras:
                        file_block["extras"] = extras
                    yield file_block

                elif doc_format == "txt" and "text" in source:
                                        # 文本文档
                    plain_text_block: types.PlainTextContentBlock = {
                        "type": "text-plain",
                        "text": source["text"],
                        "mime_type": "text/plain",
                    }
                                        # 保留额外字段
                    extras = {
                        key: value
                        for key, value in document.items()
                        if key not in {"format", "source"}
                    }
                    if extras:
                        plain_text_block["extras"] = extras
                    yield plain_text_block

                else:
                                        # 未知文档格式
                    yield {"type": "non_standard", "value": block}

            elif (
                num_keys == 1
                and (image := block.get("image"))
                and isinstance(image, dict)
                and "format" in image
            ):
                                # 处理图像格式转换
                img_format = image.get("format")
                source = image.get("source", {})

                if "bytes" in source:
                                        # 包含字节数据的图像
                    image_block: types.ImageContentBlock = {
                        "type": "image",
                        "base64": source["bytes"]
                        if isinstance(source["bytes"], str)
                        else _bytes_to_b64_str(source["bytes"]),
                        "mime_type": f"image/{img_format}",
                    }
                                        # 保留额外字段
                    extras = {}
                    for key, value in image.items():
                        if key not in {"format", "source"}:
                            extras[key] = value
                    if extras:
                        image_block["extras"] = extras
                    yield image_block

                else:
                                        # 不包含字节数据的图像
                    yield {"type": "non_standard", "value": block}

            elif block_type == "file_data" and "file_uri" in block:
                                # 处理基于 FileData URI 的内容
                uri_file_block: types.FileContentBlock = {
                    "type": "file",
                    "url": block["file_uri"],
                }
                if mime_type := block.get("mime_type"):
                    uri_file_block["mime_type"] = mime_type
                yield uri_file_block

            elif block_type == "function_call" and "name" in block:
                                # 处理函数调用
                tool_call_block: types.ToolCall = {
                    "type": "tool_call",
                    "name": block["name"],
                    "args": block.get("args", {}),
                    "id": block.get("id", ""),
                }
                yield tool_call_block

            elif block_type == "executable_code":
                server_tool_call_input: types.ServerToolCall = {
                    "type": "server_tool_call",
                    "name": "code_interpreter",
                    "args": {
                        "code": block.get("executable_code", ""),
                        "language": block.get("language", "python"),
                    },
                    "id": block.get("id", ""),
                }
                yield server_tool_call_input

            elif block_type == "code_execution_result":
                outcome = block.get("outcome", 1)
                status = "success" if outcome == 1 else "error"
                server_tool_result_input: types.ServerToolResult = {
                    "type": "server_tool_result",
                    "tool_call_id": block.get("tool_call_id", ""),
                    "status": status,  # type: ignore[typeddict-item]
                    "output": block.get("code_execution_result", ""),
                }
                if outcome is not None:
                    server_tool_result_input["extras"] = {"outcome": outcome}
                yield server_tool_result_input

            elif block.get("type") in types.KNOWN_BLOCK_TYPES:
                                # 我们遇到标准块类型，因此直接进行类型转换，即使
                                # 我们并不完全理解它。这可能有风险，但
                                # 保留信息比丢失信息更好。
                yield cast("types.ContentBlock", block)

            else:
                                # 我们完全不理解这个代码块。
                yield {"type": "non_standard", "value": block}

    return list(_iter_blocks())


def _tool_call_indices(message: AIMessage) -> dict[str, str]:
    """将工具调用 ID 映射到其流式索引。

    ChatGoogleGenerativeAI AIMessage .content does not carry function call parts, so
    we recover sequencing information from the tool_call_chunks field.

    Args:
        message: 需要索引其工具调用块的消息。

    Returns:
        工具调用 ID 到带前缀索引的映射，适用于同时具有两者的块。
    """
    indices: dict[str, str] = {}
    for chunk in getattr(message, "tool_call_chunks", None) or []:
        id_ = chunk.get("id")
        index = chunk.get("index")
        if id_ and index is not None:
            indices[id_] = f"lc_tc_{index}"  # 前缀，用于与其他索引保持分离
    return indices


def _convert_to_v1_from_genai(message: AIMessage) -> list[types.ContentBlock]:
    """将 Google GenAI 消息内容转换为 v1 格式。

    Calling `.content_blocks` on an `AIMessage` where `response_metadata.model_provider`
    is set to `'google_genai'` will invoke this function to parse the content into
    standard content blocks for returning.

    Args:
        message: 需要转换的 `AIMessage` 或 `AIMessageChunk`。

    Returns:
        从消息内容派生的标准内容块列表。
    """
    if isinstance(message.content, str):
                # 字符串内容 -> TextContentBlock（仅在非空时添加，音频情况除外）
        string_blocks: list[types.ContentBlock] = []
        if message.content:
            string_blocks.append({"type": "text", "text": message.content})

                # 从 message.tool_calls 字段中添加任何缺失的工具调用
        content_tool_call_ids = {
            block.get("id")
            for block in string_blocks
            if isinstance(block, dict) and block.get("type") == "tool_call"
        }
        tool_call_indices = _tool_call_indices(message)
        for tool_call in message.tool_calls:
            id_ = tool_call.get("id")
            if id_ and id_ not in content_tool_call_ids:
                string_tool_call_block: types.ToolCall = {
                    "type": "tool_call",
                    "id": id_,
                    "name": tool_call["name"],
                    "args": tool_call["args"],
                }
                if (index := tool_call_indices.get(id_)) is not None:
                    string_tool_call_block["index"] = index
                string_blocks.append(string_tool_call_block)

                # 如果存在，处理 additional_kwargs 中的音频（用于空内容的情况）
        audio_data = message.additional_kwargs.get("audio")
        if audio_data and isinstance(audio_data, bytes):
            audio_block: types.AudioContentBlock = {
                "type": "audio",
                "base64": _bytes_to_b64_str(audio_data),
                "mime_type": "audio/wav",  # Google GenAI 默认使用 WAV 格式
            }
            string_blocks.append(audio_block)

        grounding_metadata = message.response_metadata.get("grounding_metadata")
        if grounding_metadata:
            citations = translate_grounding_metadata_to_citations(grounding_metadata)

            for block in string_blocks:
                if block["type"] == "text" and citations:
                                        # 仅将引用添加到第一个文本块
                    block["annotations"] = cast("list[types.Annotation]", citations)
                    break

        return string_blocks

    if not isinstance(message.content, list):
                # 遇到意外的内容类型，尝试以文本形式表示
        return [{"type": "text", "text": str(message.content)}]  # type: ignore[unreachable]

    converted_blocks: list[types.ContentBlock] = []

    for item in message.content:
                # 此项目块开始的位置，以便它们可以继承下面的 `index`
        block_start = len(converted_blocks)
        if isinstance(item, str):
                        # 对话历史字符串

                        # 引用将在所有块转换完成后处理
            converted_blocks.append({"type": "text", "text": item})  # TextContentBlock

        elif isinstance(item, dict):
            item_type = item.get("type")
            if item_type == "image_url":
                                # 将 image_url 转换为标准图像块（base64）
                                # （因为原始实现返回的是 url-base64 CC 风格）
                image_url = item.get("image_url", {})
                url = image_url.get("url", "")
                if url:
                                        # 提取 base64 数据
                    match = re.match(r"data:([^;]+);base64,(.+)", url)
                    if match:
                                                # 提供了 Data URI
                        mime_type, base64_data = match.groups()
                        converted_blocks.append(
                            {
                                "type": "image",
                                "base64": base64_data,
                                "mime_type": mime_type,
                            }
                        )
                    else:
                                                # 假设它是不含 Data URI 的原始 base64
                        try:
                                                        # 验证 base64 并解码以检测 MIME 类型
                            decoded_bytes = base64.b64decode(url, validate=True)

                            image_url_b64_block = {
                                "type": "image",
                                "base64": url,
                            }

                            if _HAS_FILETYPE:
                                                                # 根据文件字节猜测 MIME 类型
                                mime_type = None
                                kind = filetype.guess(decoded_bytes)
                                if kind:
                                    mime_type = kind.mime
                                if mime_type:
                                    image_url_b64_block["mime_type"] = mime_type

                            converted_blocks.append(
                                cast("types.ImageContentBlock", image_url_b64_block)
                            )
                        except Exception:
                                                        # 不是有效的 base64，视为非标准格式
                            converted_blocks.append(
                                {
                                    "type": "non_standard",
                                    "value": item,
                                }
                            )
                else:
                                        # 根据之前的实现，这种情况很可能不会被触发
                    converted_blocks.append({"type": "non_standard", "value": item})
                    msg = "Image URL not a data URI; appending as non-standard block."
                    raise ValueError(msg)
            elif item_type == "function_call":
                                # 处理 Google GenAI 的函数调用
                function_call_block: types.ToolCall = {
                    "type": "tool_call",
                    "name": item.get("name", ""),
                    "args": item.get("args", {}),
                    "id": item.get("id", ""),
                }
                converted_blocks.append(function_call_block)
            elif item_type == "file_data":
                                # 处理基于 FileData URI 的内容
                file_block: types.FileContentBlock = {
                    "type": "file",
                    "url": item.get("file_uri", ""),
                }
                if mime_type := item.get("mime_type"):
                    file_block["mime_type"] = mime_type
                converted_blocks.append(file_block)
            elif item_type == "thinking":
                                # 处理我们封装思想的 'thinking' 类型
                reasoning_block: types.ReasoningContentBlock = {
                    "type": "reasoning",
                    "reasoning": item.get("thinking", ""),
                }
                if signature := item.get("signature"):
                    reasoning_block["extras"] = {"signature": signature}

                converted_blocks.append(reasoning_block)
            elif item_type == "executable_code":
                                # 目前转换为标准服务器工具调用块
                server_tool_call_block: types.ServerToolCall = {
                    "type": "server_tool_call",
                    "name": "code_interpreter",
                    "args": {
                        "code": item.get("executable_code", ""),
                        "language": item.get("language", "python"),  # 默认使用 python
                    },
                    "id": item.get("id", ""),
                }
                converted_blocks.append(server_tool_call_block)
            elif item_type == "code_execution_result":
                                # 将结果映射为状态：OUTCOME_OK (1) → 成功，否则 → 错误
                outcome = item.get("outcome", 1)
                status = "success" if outcome == 1 else "error"
                server_tool_result_block: types.ServerToolResult = {
                    "type": "server_tool_result",
                    "tool_call_id": item.get("tool_call_id", ""),
                    "status": status,  # type: ignore[typeddict-item]
                    "output": item.get("code_execution_result", ""),
                }
                server_tool_result_block["extras"] = {"block_type": item_type}
                                # 在 extras 中保留原始结果
                if outcome is not None:
                    server_tool_result_block["extras"]["outcome"] = outcome
                converted_blocks.append(server_tool_result_block)
            elif item_type == "text":
                converted_blocks.append(cast("types.TextContentBlock", item))
            else:
                                # 未知类型，保留为非标准格式
                converted_blocks.append({"type": "non_standard", "value": item})
        else:
                        # 非字典、非字符串内容
            converted_blocks.append({"type": "non_standard", "value": item})  # type: ignore[unreachable]

                # 每个项目今天只生成一个块。已防护，但不假设：
                # 重复的索引会导致块重新合并
        new_blocks = converted_blocks[block_start:]
        if (
            len(new_blocks) == 1
            and isinstance(item, dict)
            and (index := item.get("index")) is not None
        ):
            new_blocks[0].setdefault("index", index)

    grounding_metadata = message.response_metadata.get("grounding_metadata")
    if grounding_metadata:
        citations = translate_grounding_metadata_to_citations(grounding_metadata)

        for i, block in enumerate(converted_blocks):
            if block["type"] == "text" and citations:
                                # 向文本块添加引用（仅第一个文本块）
                                # 替换而非赋值：`block` 是调用者的私有对象
                                # 来自 `message.content` 的字典
                converted_blocks[i] = cast(
                    "types.TextContentBlock",
                    {**block, "annotations": citations},
                )
                break

        # 音频存储在 message.additional_kwargs 中
    audio_data = message.additional_kwargs.get("audio")
    if audio_data and isinstance(audio_data, bytes):
        audio_block_kwargs: types.AudioContentBlock = {
            "type": "audio",
            "base64": _bytes_to_b64_str(audio_data),
            "mime_type": "audio/wav",  # Google GenAI 默认使用 WAV 格式
        }
        converted_blocks.append(audio_block_kwargs)

        # 从 message.tool_calls 字段中添加任何缺失的工具调用
    content_tool_call_ids = {
        block.get("id")
        for block in converted_blocks
        if isinstance(block, dict) and block.get("type") == "tool_call"
    }
    tool_call_indices = _tool_call_indices(message)
    for tool_call in message.tool_calls:
        id_ = tool_call.get("id")
        if id_ and id_ not in content_tool_call_ids:
            missing_tool_call_block: types.ToolCall = {
                "type": "tool_call",
                "id": id_,
                "name": tool_call["name"],
                "args": tool_call["args"],
            }
            if (index := tool_call_indices.get(id_)) is not None:
                missing_tool_call_block["index"] = index
            converted_blocks.append(missing_tool_call_block)

    return converted_blocks


def translate_content(message: AIMessage) -> list[types.ContentBlock]:
    """从带有 Google（GenAI）内容的消息中派生标准内容块。

    Args:
        message: 需要转换的消息。

    Returns:
        派生的内容块。
    """
    return _convert_to_v1_from_genai(message)


def translate_content_chunk(message: AIMessageChunk) -> list[types.ContentBlock]:
    """从带有 Google（GenAI）内容的块中派生标准内容块。

    Args:
        message: 需要转换的消息块。

    Returns:
        派生的内容块。
    """
    return _convert_to_v1_from_genai(message)


def _register_google_genai_translator() -> None:
    """将 Google（GenAI）翻译器注册到中央注册表。

    模块导入时自动运行。
    """
    from langchain_core.messages.block_translators import (  # noqa: PLC0415
        register_translator,
    )

    register_translator("google_genai", translate_content, translate_content_chunk)


_register_google_genai_translator()

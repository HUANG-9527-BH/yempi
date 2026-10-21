"""从 Anthropic 内容派生的标准内容块。"""

import json
from collections.abc import Iterator
from typing import Any, cast

from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.messages import content as types


def _populate_extras(
    standard_block: types.ContentBlock, block: dict[str, Any], known_fields: set[str]
) -> types.ContentBlock:
    """修改一个块，填充 extras 字段。"""
    if standard_block.get("type") == "non_standard":
        return standard_block

    for key, value in block.items():
        if key not in known_fields:
            if "extras" not in standard_block:
                                # 以下 type-ignores 是因为 mypy 认为非标准块可能
                # 出现在此处，尽管我们在上方已排除了它们。
                                # 以下 type-ignores 是因为 mypy 认为非标准块可能
                # 出现在此处，尽管我们在上方已排除了它们。
                standard_block["extras"] = {}  # type: ignore[typeddict-unknown-key]
            standard_block["extras"][key] = value  # type: ignore[typeddict-item]

    return standard_block


def _non_standard_block(block: dict[str, Any]) -> types.NonStandardContentBlock:
    return {"type": "non_standard", "value": block}


def _convert_to_v1_from_anthropic_input(
    content: list[types.ContentBlock],
) -> list[types.ContentBlock]:
    """将 Anthropic 格式的内容块转换为 v1 格式。

    During the `content_blocks` parsing process, we wrap blocks not recognized as a v1
    block as a `'non_standard'` block with the original block stored in the `value`
    field. This function attempts to unpack those blocks and convert any blocks that
    might be Anthropic format to v1 ContentBlocks.

    如果转换失败，该块将保持为 `'non_standard'` 块。

    Args:
        content: 需要处理的内容块列表。

    Returns:
        已更新的列表，其中 Anthropic 块已转换为 v1 格式。
    """

    def _iter_blocks() -> Iterator[types.ContentBlock]:
        blocks: list[dict[str, Any]] = [
            cast("dict[str, Any]", block)
            if block.get("type") != "non_standard"
            else block["value"]  # type: ignore[typeddict-item]  # this is only non-standard blocks
            for block in content
        ]
        for block in blocks:
            block_type = block.get("type")
            source = block.get("source")

            if (
                block_type == "document"
                and isinstance(source, dict)
                and "type" in source
            ):
                if source["type"] == "base64":
                    if "data" not in source or "media_type" not in source:
                        yield _non_standard_block(block)
                        continue
                    file_block: types.FileContentBlock = {
                        "type": "file",
                        "base64": source["data"],
                        "mime_type": source["media_type"],
                    }
                    _populate_extras(file_block, block, {"type", "source"})
                    yield file_block

                elif source["type"] == "url":
                    if "url" not in source:
                        yield _non_standard_block(block)
                        continue
                    file_block = {
                        "type": "file",
                        "url": source["url"],
                    }
                    _populate_extras(file_block, block, {"type", "source"})
                    yield file_block

                elif source["type"] == "file":
                    if "file_id" not in source:
                        yield _non_standard_block(block)
                        continue
                    file_block = {
                        "type": "file",
                        "id": source["file_id"],
                    }
                    _populate_extras(file_block, block, {"type", "source"})
                    yield file_block

                elif source["type"] == "text":
                    if "data" not in source:
                        yield _non_standard_block(block)
                        continue
                    plain_text_block: types.PlainTextContentBlock = {
                        "type": "text-plain",
                        "text": source["data"],
                        "mime_type": block.get("media_type", "text/plain"),
                    }
                    _populate_extras(plain_text_block, block, {"type", "source"})
                    yield plain_text_block

                else:
                    yield _non_standard_block(block)

            elif (
                block_type == "image" and isinstance(source, dict) and "type" in source
            ):
                if source["type"] == "base64":
                    if "data" not in source or "media_type" not in source:
                        yield _non_standard_block(block)
                        continue
                    image_block: types.ImageContentBlock = {
                        "type": "image",
                        "base64": source["data"],
                        "mime_type": source["media_type"],
                    }
                    _populate_extras(image_block, block, {"type", "source"})
                    yield image_block

                elif source["type"] == "url":
                    if "url" not in source:
                        yield _non_standard_block(block)
                        continue
                    image_block = {
                        "type": "image",
                        "url": source["url"],
                    }
                    _populate_extras(image_block, block, {"type", "source"})
                    yield image_block

                elif source["type"] == "file":
                    if "file_id" not in source:
                        yield _non_standard_block(block)
                        continue
                    image_block = {
                        "type": "image",
                        "id": source["file_id"],
                    }
                    _populate_extras(image_block, block, {"type", "source"})
                    yield image_block

                else:
                    yield _non_standard_block(block)

            elif block_type in types.KNOWN_BLOCK_TYPES:
                yield cast("types.ContentBlock", block)

            else:
                yield _non_standard_block(block)

    return list(_iter_blocks())


def _convert_citation_to_v1(citation: dict[str, Any]) -> types.Annotation:
    citation_type = citation.get("type")

    if citation_type == "web_search_result_location":
        if "cited_text" not in citation or "url" not in citation:
            return {
                "type": "non_standard_annotation",
                "value": citation,
            }
        url_citation: types.Citation = {
            "type": "citation",
            "cited_text": citation["cited_text"],
            "url": citation["url"],
        }
        if title := citation.get("title"):
            url_citation["title"] = title
        known_fields = {"type", "cited_text", "url", "title", "index", "extras"}
        for key, value in citation.items():
            if key not in known_fields:
                if "extras" not in url_citation:
                    url_citation["extras"] = {}
                url_citation["extras"][key] = value

        return url_citation

    if citation_type in {
        "char_location",
        "content_block_location",
        "page_location",
        "search_result_location",
    }:
        if "cited_text" not in citation:
            return {
                "type": "non_standard_annotation",
                "value": citation,
            }
        document_citation: types.Citation = {
            "type": "citation",
            "cited_text": citation["cited_text"],
        }
        if "document_title" in citation:
            document_citation["title"] = citation["document_title"]
        elif title := citation.get("title"):
            document_citation["title"] = title
        known_fields = {
            "type",
            "cited_text",
            "document_title",
            "title",
            "index",
            "extras",
        }
        for key, value in citation.items():
            if key not in known_fields:
                if "extras" not in document_citation:
                    document_citation["extras"] = {}
                document_citation["extras"][key] = value

        return document_citation

    return {
        "type": "non_standard_annotation",
        "value": citation,
    }


def _convert_to_v1_from_anthropic(message: AIMessage) -> list[types.ContentBlock]:
    """将 Anthropic 消息内容转换为 v1 格式。"""
    content: list[str | dict[str, Any]]
    if isinstance(message.content, str):
        content = [{"type": "text", "text": message.content}]
    else:
        content = message.content

    def _iter_blocks() -> Iterator[types.ContentBlock]:
        for block in content:
            if not isinstance(block, dict):
                continue
            block_type = block.get("type")

            if block_type == "text":
                if citations := block.get("citations"):
                    text_block: types.TextContentBlock = {
                        "type": "text",
                        "text": block.get("text", ""),
                        "annotations": [_convert_citation_to_v1(a) for a in citations],
                    }
                else:
                    text_block = {"type": "text", "text": block["text"]}
                if "index" in block:
                    text_block["index"] = block["index"]
                yield text_block

            elif block_type == "thinking":
                reasoning_block: types.ReasoningContentBlock = {
                    "type": "reasoning",
                    "reasoning": block.get("thinking", ""),
                }
                if "index" in block:
                    reasoning_block["index"] = block["index"]
                known_fields = {"type", "thinking", "index", "extras"}
                for key in block:
                    if key not in known_fields:
                        if "extras" not in reasoning_block:
                            reasoning_block["extras"] = {}
                        reasoning_block["extras"][key] = block[key]
                yield reasoning_block

            elif block_type == "tool_use":
                if (
                    isinstance(message, AIMessageChunk)
                    and len(message.tool_call_chunks) == 1
                    and message.chunk_position != "last"
                ):
                                        # 独立片段
                    chunk = message.tool_call_chunks[0]

                    tool_call_chunk = types.ToolCallChunk(
                        name=chunk.get("name"),
                        id=chunk.get("id"),
                        args=chunk.get("args"),
                        type="tool_call_chunk",
                    )
                    for key in ("caller", "toolset_name"):
                        if key in block:
                            tool_call_chunk.setdefault("extras", {})[key] = block[key]

                    index = chunk.get("index")
                    if index is not None:
                        tool_call_chunk["index"] = index
                    yield tool_call_chunk
                else:
                    tool_call_block: types.ToolCall | None = None
                                        # 非流式或聚合片段
                    if len(message.tool_calls) == 1:
                        tool_call_block = {
                            "type": "tool_call",
                            "name": message.tool_calls[0]["name"],
                            "args": message.tool_calls[0]["args"],
                            "id": message.tool_calls[0].get("id"),
                        }
                    elif call_id := block.get("id"):
                        for tc in message.tool_calls:
                            if tc.get("id") == call_id:
                                tool_call_block = {
                                    "type": "tool_call",
                                    "name": tc["name"],
                                    "args": tc["args"],
                                    "id": tc.get("id"),
                                }
                                break
                    if not tool_call_block:
                        tool_call_block = {
                            "type": "tool_call",
                            "name": block.get("name", ""),
                            "args": block.get("input", {}),
                            "id": block.get("id", ""),
                        }
                    if "index" in block:
                        tool_call_block["index"] = block["index"]
                    for key in ("caller", "toolset_name"):
                        if key in block:
                            tool_call_block.setdefault("extras", {})[key] = block[key]

                    yield tool_call_block

            elif block_type == "input_json_delta" and isinstance(
                message, AIMessageChunk
            ):
                if len(message.tool_call_chunks) == 1:
                    chunk = message.tool_call_chunks[0]
                    tool_call_chunk = types.ToolCallChunk(
                        name=chunk.get("name"),
                        id=chunk.get("id"),
                        args=chunk.get("args"),
                        type="tool_call_chunk",
                    )
                    index = chunk.get("index")
                    if index is not None:
                        tool_call_chunk["index"] = index
                    yield tool_call_chunk

                else:
                    server_tool_call_chunk: types.ServerToolCallChunk = {
                        "type": "server_tool_call_chunk",
                        "args": block.get("partial_json", ""),
                    }
                    if "index" in block:
                        server_tool_call_chunk["index"] = block["index"]
                    yield server_tool_call_chunk

            elif block_type == "server_tool_use":
                if block.get("name") == "code_execution":
                    server_tool_use_name = "code_interpreter"
                else:
                    server_tool_use_name = block.get("name", "")
                if (
                    isinstance(message, AIMessageChunk)
                    and block.get("input") == {}
                    and "partial_json" not in block
                    and message.chunk_position != "last"
                ):
                                        # 流式中的第一个片段
                    server_tool_call_chunk = {
                        "type": "server_tool_call_chunk",
                        "name": server_tool_use_name,
                        "args": "",
                        "id": block.get("id", ""),
                    }
                    if "index" in block:
                        server_tool_call_chunk["index"] = block["index"]
                    known_fields = {"type", "name", "input", "id", "index"}
                    _populate_extras(server_tool_call_chunk, block, known_fields)
                    yield server_tool_call_chunk
                else:
                    server_tool_call: types.ServerToolCall = {
                        "type": "server_tool_call",
                        "name": server_tool_use_name,
                        "args": block.get("input", {}),
                        "id": block.get("id", ""),
                    }

                    if block.get("input") == {} and "partial_json" in block:
                        try:
                            input_ = json.loads(block["partial_json"])
                            if isinstance(input_, dict):
                                server_tool_call["args"] = input_
                        except json.JSONDecodeError:
                            pass

                    if "index" in block:
                        server_tool_call["index"] = block["index"]
                    known_fields = {
                        "type",
                        "name",
                        "input",
                        "partial_json",
                        "id",
                        "index",
                    }
                    _populate_extras(server_tool_call, block, known_fields)

                    yield server_tool_call

            elif block_type == "mcp_tool_use":
                if (
                    isinstance(message, AIMessageChunk)
                    and block.get("input") == {}
                    and "partial_json" not in block
                    and message.chunk_position != "last"
                ):
                                        # 流式中的第一个片段
                    server_tool_call_chunk = {
                        "type": "server_tool_call_chunk",
                        "name": "remote_mcp",
                        "args": "",
                        "id": block.get("id", ""),
                    }
                    if "name" in block:
                        server_tool_call_chunk["extras"] = {"tool_name": block["name"]}
                    known_fields = {"type", "name", "input", "id", "index"}
                    _populate_extras(server_tool_call_chunk, block, known_fields)
                    if "index" in block:
                        server_tool_call_chunk["index"] = block["index"]
                    yield server_tool_call_chunk
                else:
                    server_tool_call = {
                        "type": "server_tool_call",
                        "name": "remote_mcp",
                        "args": block.get("input", {}),
                        "id": block.get("id", ""),
                    }

                    if block.get("input") == {} and "partial_json" in block:
                        try:
                            input_ = json.loads(block["partial_json"])
                            if isinstance(input_, dict):
                                server_tool_call["args"] = input_
                        except json.JSONDecodeError:
                            pass

                    if "name" in block:
                        server_tool_call["extras"] = {"tool_name": block["name"]}
                    known_fields = {
                        "type",
                        "name",
                        "input",
                        "partial_json",
                        "id",
                        "index",
                    }
                    _populate_extras(server_tool_call, block, known_fields)
                    if "index" in block:
                        server_tool_call["index"] = block["index"]

                    yield server_tool_call

            elif block_type and block_type.endswith("_tool_result"):
                server_tool_result: types.ServerToolResult = {
                    "type": "server_tool_result",
                    "tool_call_id": block.get("tool_use_id", ""),
                    "status": "success",
                    "extras": {"block_type": block_type},
                }
                if output := block.get("content", []):
                    server_tool_result["output"] = output
                    if isinstance(output, dict) and output.get(
                        "error_code"  # web_search, code_interpreter
                    ):
                        server_tool_result["status"] = "error"
                if block.get("is_error"):  # mcp_tool_result
                    server_tool_result["status"] = "error"
                if "index" in block:
                    server_tool_result["index"] = block["index"]

                known_fields = {"type", "tool_use_id", "content", "is_error", "index"}
                _populate_extras(server_tool_result, block, known_fields)

                yield server_tool_result

            else:
                new_block: types.NonStandardContentBlock = {
                    "type": "non_standard",
                    "value": block,
                }
                if "index" in new_block["value"]:
                    new_block["index"] = new_block["value"].pop("index")
                yield new_block

    return list(_iter_blocks())


def translate_content(message: AIMessage) -> list[types.ContentBlock]:
    """从带有 Anthropic 内容的消息中派生标准内容块。

    Args:
        message: 需要转换的消息。

    Returns:
        派生出的内容块。
    """
    return _convert_to_v1_from_anthropic(message)


def translate_content_chunk(message: AIMessageChunk) -> list[types.ContentBlock]:
    """从带有 Anthropic 内容的消息片段中派生标准内容块。

    Args:
        message: 需要转换的消息片段。

    Returns:
        派生出的内容块。
    """
    return _convert_to_v1_from_anthropic(message)


def _register_anthropic_translator() -> None:
    """将 Anthropic 转换器注册到中央注册表中。

    在模块导入时自动运行。
    """
    from langchain_core.messages.block_translators import (  # noqa: PLC0415
        register_translator,
    )

    register_translator("anthropic", translate_content, translate_content_chunk)


_register_anthropic_translator()

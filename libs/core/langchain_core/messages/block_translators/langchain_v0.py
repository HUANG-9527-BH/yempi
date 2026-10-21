"""LangChain v0 多模态内容块的标准内容块派生。"""

from typing import Any, cast

from langchain_core.messages import content as types


def _convert_v0_multimodal_input_to_v1(
    content: list[types.ContentBlock],
) -> list[types.ContentBlock]:
    """将 v0 多模态块转换为 v1 格式。

    During the `content_blocks` parsing process, we wrap blocks not recognized as a v1
    block as a `'non_standard'` block with the original block stored in the `value`
    field. This function attempts to unpack those blocks and convert any v0 format
    blocks to v1 format.

    如果转换失败，该块将保持为 `'non_standard'` 块。

    Args:
        content: 需要处理的内容块列表。

    Returns:
        v1 内容块。
    """
    converted_blocks = []
    unpacked_blocks: list[dict[str, Any]] = [
        cast("dict[str, Any]", block)
        if block.get("type") != "non_standard"
        else block["value"]  # type: ignore[typeddict-item]  # this is only non-standard blocks
        for block in content
    ]
    for block in unpacked_blocks:
        if block.get("type") in {"image", "audio", "file"} and "source_type" in block:
            converted_block = _convert_legacy_v0_content_block_to_v1(block)
            converted_blocks.append(cast("types.ContentBlock", converted_block))
        elif block.get("type") in types.KNOWN_BLOCK_TYPES:
                        # Guard in case this function is used outside of the .content_blocks flow
            converted_blocks.append(cast("types.ContentBlock", block))
        else:
            converted_blocks.append({"type": "non_standard", "value": block})

    return converted_blocks


def _convert_legacy_v0_content_block_to_v1(
    block: dict[str, Any],
) -> types.ContentBlock | dict[str, Any]:
    """将 LangChain v0 内容块转换为 v1 格式。

    保留未知键作为 extras，以避免数据丢失。

    如果块不是 v0 格式，则原样返回。
    """

    def _extract_v0_extras(
        block_dict: dict[str, Any], known_keys: set[str]
    ) -> dict[str, Any]:
        """从 v0 块中提取未知键，以作为 extras 保留。

        Args:
            block_dict: 原始 v0 块字典。
            known_keys: 此块已知属于 v0 格式的键集合。

        Returns:
            一个包含不属于已知 v0 格式的额外键的字典。
        """
        return {k: v for k, v in block_dict.items() if k not in known_keys}

        # Check if this is actually a v0 format block
    block_type = block.get("type")
    if block_type not in {"image", "audio", "file"} or "source_type" not in block:
                # Not a v0 format block, return unchanged
        return block

    if block.get("type") == "image":
        source_type = block.get("source_type")
        if source_type == "url":
                        # image-url
            known_keys = {"mime_type", "type", "source_type", "url"}
            extras = _extract_v0_extras(block, known_keys)
            if "id" in block:
                return types.create_image_block(
                    url=block["url"],
                    mime_type=block.get("mime_type"),
                    id=block["id"],
                    **extras,
                )

                        # Don't construct with an ID if not present in original block
            v1_image_url = types.ImageContentBlock(type="image", url=block["url"])
            if block.get("mime_type"):
                v1_image_url["mime_type"] = block["mime_type"]

            v1_image_url["extras"] = {}
            for key, value in extras.items():
                if value is not None:
                    v1_image_url["extras"][key] = value
            if v1_image_url["extras"] == {}:
                del v1_image_url["extras"]

            return v1_image_url
        if source_type == "base64":
                        # image-base64
            known_keys = {"mime_type", "type", "source_type", "data"}
            extras = _extract_v0_extras(block, known_keys)
            if "id" in block:
                return types.create_image_block(
                    base64=block["data"],
                    mime_type=block.get("mime_type"),
                    id=block["id"],
                    **extras,
                )

            v1_image_base64 = types.ImageContentBlock(
                type="image", base64=block["data"]
            )
            if block.get("mime_type"):
                v1_image_base64["mime_type"] = block["mime_type"]

            v1_image_base64["extras"] = {}
            for key, value in extras.items():
                if value is not None:
                    v1_image_base64["extras"][key] = value
            if v1_image_base64["extras"] == {}:
                del v1_image_base64["extras"]

            return v1_image_base64
        if source_type == "id":
                        # image-id
            known_keys = {"type", "source_type", "id"}
            extras = _extract_v0_extras(block, known_keys)
                        # For id `source_type`, `id` is the file reference, not block ID
            v1_image_id = types.ImageContentBlock(type="image", file_id=block["id"])

            v1_image_id["extras"] = {}
            for key, value in extras.items():
                if value is not None:
                    v1_image_id["extras"][key] = value
            if v1_image_id["extras"] == {}:
                del v1_image_id["extras"]

            return v1_image_id
    elif block.get("type") == "audio":
        source_type = block.get("source_type")
        if source_type == "url":
                        # audio-url
            known_keys = {"mime_type", "type", "source_type", "url"}
            extras = _extract_v0_extras(block, known_keys)
            if "id" in block:
                return types.create_audio_block(
                    url=block["url"],
                    mime_type=block.get("mime_type"),
                    id=block["id"],
                    **extras,
                )

                        # Don't construct with an ID if not present in original block
            v1_audio_url: types.AudioContentBlock = types.AudioContentBlock(
                type="audio", url=block["url"]
            )
            if block.get("mime_type"):
                v1_audio_url["mime_type"] = block["mime_type"]

            v1_audio_url["extras"] = {}
            for key, value in extras.items():
                if value is not None:
                    v1_audio_url["extras"][key] = value
            if v1_audio_url["extras"] == {}:
                del v1_audio_url["extras"]

            return v1_audio_url
        if source_type == "base64":
                        # audio-base64
            known_keys = {"mime_type", "type", "source_type", "data"}
            extras = _extract_v0_extras(block, known_keys)
            if "id" in block:
                return types.create_audio_block(
                    base64=block["data"],
                    mime_type=block.get("mime_type"),
                    id=block["id"],
                    **extras,
                )

            v1_audio_base64: types.AudioContentBlock = types.AudioContentBlock(
                type="audio", base64=block["data"]
            )
            if block.get("mime_type"):
                v1_audio_base64["mime_type"] = block["mime_type"]

            v1_audio_base64["extras"] = {}
            for key, value in extras.items():
                if value is not None:
                    v1_audio_base64["extras"][key] = value
            if v1_audio_base64["extras"] == {}:
                del v1_audio_base64["extras"]

            return v1_audio_base64
        if source_type == "id":
                        # audio-id
            known_keys = {"type", "source_type", "id"}
            extras = _extract_v0_extras(block, known_keys)
            v1_audio_id: types.AudioContentBlock = types.AudioContentBlock(
                type="audio", file_id=block["id"]
            )

            v1_audio_id["extras"] = {}
            for key, value in extras.items():
                if value is not None:
                    v1_audio_id["extras"][key] = value
            if v1_audio_id["extras"] == {}:
                del v1_audio_id["extras"]

            return v1_audio_id
    elif block.get("type") == "file":
        source_type = block.get("source_type")
        if source_type == "url":
                        # file-url
            known_keys = {"mime_type", "type", "source_type", "url"}
            extras = _extract_v0_extras(block, known_keys)
            if "id" in block:
                return types.create_file_block(
                    url=block["url"],
                    mime_type=block.get("mime_type"),
                    id=block["id"],
                    **extras,
                )

            v1_file_url: types.FileContentBlock = types.FileContentBlock(
                type="file", url=block["url"]
            )
            if block.get("mime_type"):
                v1_file_url["mime_type"] = block["mime_type"]

            v1_file_url["extras"] = {}
            for key, value in extras.items():
                if value is not None:
                    v1_file_url["extras"][key] = value
            if v1_file_url["extras"] == {}:
                del v1_file_url["extras"]

            return v1_file_url
        if source_type == "base64":
                        # file-base64
            known_keys = {"mime_type", "type", "source_type", "data"}
            extras = _extract_v0_extras(block, known_keys)
            if "id" in block:
                return types.create_file_block(
                    base64=block["data"],
                    mime_type=block.get("mime_type"),
                    id=block["id"],
                    **extras,
                )

            v1_file_base64: types.FileContentBlock = types.FileContentBlock(
                type="file", base64=block["data"]
            )
            if block.get("mime_type"):
                v1_file_base64["mime_type"] = block["mime_type"]

            v1_file_base64["extras"] = {}
            for key, value in extras.items():
                if value is not None:
                    v1_file_base64["extras"][key] = value
            if v1_file_base64["extras"] == {}:
                del v1_file_base64["extras"]

            return v1_file_base64
        if source_type == "id":
                        # file-id
            known_keys = {"type", "source_type", "id"}
            extras = _extract_v0_extras(block, known_keys)
            return types.create_file_block(file_id=block["id"], **extras)
        if source_type == "text":
                        # file-text
            known_keys = {"mime_type", "type", "source_type", "url"}
            extras = _extract_v0_extras(block, known_keys)
            if "id" in block:
                return types.create_plaintext_block(
                                        # In v0, URL points to the text file content
                                        # TODO: attribute this claim
                    text=block["url"],
                    id=block["id"],
                    **extras,
                )

            v1_file_text: types.PlainTextContentBlock = types.PlainTextContentBlock(
                type="text-plain", text=block["url"], mime_type="text/plain"
            )
            if block.get("mime_type"):
                v1_file_text["mime_type"] = block["mime_type"]

            v1_file_text["extras"] = {}
            for key, value in extras.items():
                if value is not None:
                    v1_file_text["extras"][key] = value
            if v1_file_text["extras"] == {}:
                del v1_file_text["extras"]

            return v1_file_text

        # If we can't convert, return the block unchanged
    return block

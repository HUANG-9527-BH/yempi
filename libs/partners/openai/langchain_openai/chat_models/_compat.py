"""根据 `output_version` 控制在 AIMessage 输出格式之间的转换。

`output_version` 是 ChatOpenAI 上的属性。

支持的值为 `None`、`'v0'` 和 `'responses/v1'`。

`'v0'` 对应于 `ChatOpenAI` v0.3 时的格式。对于 Responses API，它将推理和工具输出存储在
`AIMessage.additional_kwargs` 中：

```python
AIMessage(
    content=[
        {"type": "text", "text": "Hello, world!", "annotations": [{"type": "foo"}]}
    ],
    additional_kwargs={
        "reasoning": {
            "type": "reasoning",
            "id": "rs_123",
            "summary": [{"type": "summary_text", "text": "Reasoning summary"}],
        },
        "tool_outputs": [
            {
                "type": "web_search_call",
                "id": "websearch_123",
                "status": "completed",
            }
        ],
        "refusal": "I cannot assist with that.",
    },
    response_metadata={"id": "resp_123"},
    id="msg_123",
)
```

`'responses/v1'` 仅适用于 Responses
API。它保留了响应项序列的信息，并通过在内容序列中表示这些项目来支持多个推理项：

```python
AIMessage(
    content=[
        {
            "type": "reasoning",
            "summary": [{"type": "summary_text", "text": "Reasoning summary"}],
            "id": "rs_123",
        },
        {
            "type": "text",
            "text": "Hello, world!",
            "annotations": [{"type": "foo"}],
            "id": "msg_123",
        },
        {"type": "refusal", "refusal": "I cannot assist with that."},
        {"type": "web_search_call", "id": "websearch_123", "status": "completed"},
    ],
    response_metadata={"id": "resp_123"},
    id="resp_123",
)
```

还有其他一些小改进——例如，我们将在文本内容块上存储消息 ID，而不是在 `AIMessage.id`
上，后者现在存储的是响应 ID。

为了向后兼容，此模块提供了在格式间转换的函数。这些函数由 ChatOpenAI 内部使用。
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from typing import Any, cast

from langchain_core.messages import AIMessage, is_data_content_block
from langchain_core.messages import content as types


def _unwrap_non_standard(block: dict) -> dict:
    """从标准内容块中解包提供方原生字典。"""
    if block.get("type") == "non_standard" and isinstance(
        value := block.get("value"),
        dict,
    ):
        return value
    return block


_FUNCTION_CALL_IDS_MAP_KEY = "__openai_function_call_ids__"


# v0.3 / Responses
def _convert_to_v03_ai_message(
    message: AIMessage, has_reasoning: bool = False
) -> AIMessage:
    """将 `AIMessage` 修改为旧版 v0.3 格式。"""
    if isinstance(message.content, list):
        new_content: list[dict | str] = []
        for block in message.content:
            if isinstance(block, dict):
                if block.get("type") == "reasoning":
                                        # 将推理项存储在 additional_kwargs 中（覆盖如
                    # v0.3)
                    _ = block.pop("index", None)
                    if has_reasoning:
                        _ = block.pop("id", None)
                        _ = block.pop("type", None)
                    message.additional_kwargs["reasoning"] = block
                elif block.get("type") in (
                    "web_search_call",
                    "file_search_call",
                    "computer_call",
                    "code_interpreter_call",
                    "mcp_call",
                    "mcp_list_tools",
                    "mcp_approval_request",
                    "image_generation_call",
                    "tool_search_call",
                    "tool_search_output",
                    "apply_patch_call",
                    "apply_patch_call_output",
                ):
                                        # 将内置工具调用存储在 additional_kwargs 中
                    if "tool_outputs" not in message.additional_kwargs:
                        message.additional_kwargs["tool_outputs"] = []
                    message.additional_kwargs["tool_outputs"].append(block)
                elif block.get("type") == "function_call":
                                        # 将函数调用项 ID 存储在 additional_kwargs 中，否则
                                        # 丢弃函数调用项。
                    if _FUNCTION_CALL_IDS_MAP_KEY not in message.additional_kwargs:
                        message.additional_kwargs[_FUNCTION_CALL_IDS_MAP_KEY] = {}
                    if (call_id := block.get("call_id")) and (
                        function_call_id := block.get("id")
                    ):
                        message.additional_kwargs[_FUNCTION_CALL_IDS_MAP_KEY][
                            call_id
                        ] = function_call_id
                elif (block.get("type") == "refusal") and (
                    refusal := block.get("refusal")
                ):
                                        # 将拒绝项存储在 additional_kwargs 中（覆盖如
                    # v0.3)
                    message.additional_kwargs["refusal"] = refusal
                elif block.get("type") == "text":
                                        # 将消息项 ID 存储在 AIMessage.id 上
                    if "id" in block:
                        message.id = block["id"]
                    new_content.append({k: v for k, v in block.items() if k != "id"})
                elif (
                    set(block.keys()) == {"id", "index"}
                    and isinstance(block["id"], str)
                    and block["id"].startswith("msg_")
                ):
                                        # 流式情况下丢弃消息 ID
                    new_content.append({"index": block["index"]})
                else:
                    new_content.append(block)
            else:
                new_content.append(block)
        message.content = new_content
        if isinstance(message.id, str) and message.id.startswith("resp_"):
            message.id = None
    else:
        pass

    return message


# v1 / Chat Completions
def _convert_from_v1_to_chat_completions(message: AIMessage) -> AIMessage:
    """将 v1 消息转换为 Chat Completions 格式。"""
    if isinstance(message.content, list):
        new_content: list = []
        for block in message.content:
            if isinstance(block, dict):
                block_type = block.get("type")
                if block_type == "text":
                                        # 剥离注释
                    new_content.append({"type": "text", "text": block["text"]})
                elif block_type in ("reasoning", "tool_call", "invalid_tool_call"):
                    pass
                else:
                    new_content.append(block)
            else:
                new_content.append(block)
        return message.model_copy(update={"content": new_content})

    return message


# v1 / Responses
def _convert_annotation_from_v1(annotation: types.Annotation) -> dict[str, Any]:
    """将 v1 `Annotation` 转换为 v0.3 格式（用于 Responses API）。"""
    if annotation["type"] == "citation":
        new_ann: dict[str, Any] = {}
        for field in ("end_index", "start_index"):
            if field in annotation:
                new_ann[field] = annotation[field]

        if "url" in annotation:
                        # URL 引用
            if "title" in annotation:
                new_ann["title"] = annotation["title"]
            new_ann["type"] = "url_citation"
            new_ann["url"] = annotation["url"]

            if extra_fields := annotation.get("extras"):
                new_ann.update(dict(extra_fields.items()))
        else:
                        # 文档引用
            new_ann["type"] = "file_citation"

            if extra_fields := annotation.get("extras"):
                new_ann.update(dict(extra_fields.items()))

            if "title" in annotation:
                new_ann["filename"] = annotation["title"]

        return new_ann

    if annotation["type"] == "non_standard_annotation":
        return annotation["value"]

    return dict(annotation)


def _same_reasoning_item(first: dict[str, Any], second: dict[str, Any]) -> bool:
    """返回两个推理片段是否共享同一项身份。"""
    first_has_id = "id" in first
    second_has_id = "id" in second
    return first_has_id == second_has_id and (
        not first_has_id or first["id"] == second["id"]
    )


def _implode_reasoning_blocks(blocks: list[dict[str, Any]]) -> Iterable[dict[str, Any]]:
    i = 0
    n = len(blocks)

    while i < n:
        block = blocks[i]

                # 跳过非推理块或已处于 Responses 格式的块
        if block.get("type") != "reasoning" or "summary" in block:
            yield dict(block)
            i += 1
            continue
        elif "reasoning" not in block and "summary" not in block:
                        # {"type": "reasoning", "id": "rs_..."}
            oai_format = {**block, "summary": []}
            if "extras" in oai_format:
                oai_format.update(oai_format.pop("extras"))
            oai_format["type"] = oai_format.pop("type", "reasoning")
            if "encrypted_content" in oai_format:
                oai_format["encrypted_content"] = oai_format.pop("encrypted_content")
            yield oai_format
            i += 1
            continue
        else:
            pass

        summary: list[dict[str, str]] = [
            {"type": "summary_text", "text": block.get("reasoning", "")}
        ]
                # 'common' 是除展开的 'reasoning' 外的所有字段
        common = {k: v for k, v in block.items() if k != "reasoning"}
        if "extras" in common:
            common.update(common.pop("extras"))

        i += 1
        while i < n:
            next_ = blocks[i]
            if (
                next_.get("type") == "reasoning"
                and "reasoning" in next_
                and _same_reasoning_item(block, next_)
            ):
                summary.append(
                    {"type": "summary_text", "text": next_.get("reasoning", "")}
                )
                i += 1
            else:
                break

        merged = dict(common)
        merged["summary"] = summary
        merged["type"] = merged.pop("type", "reasoning")
        yield merged


def _consolidate_calls(items: Iterable[dict[str, Any]]) -> Iterator[dict[str, Any]]:
    """生成器，遍历 *items*，每当遇到配对时。

        {"type": "server_tool_call", "name": "web_search", "id": X, ...}
        {"type": "server_tool_result", "id": X}

    合并为

        {"id": X,
         "output": ...,
         "status": ...,
         "type": "web_search_call"}

    保持其他所有元素不变。
    """
    items = iter(items)  # 确保我们有一个真正的迭代器
    for current in items:
                # 只有调用才能启动一个值得合并的配对
        if current.get("type") != "server_tool_call":
            yield current
            continue

        try:
            nxt = next(items)  # 向前查看一个元素
        except StopIteration:  # 没有 "result" - 只返回调用
            yield current
            break

                # 如果这确实是匹配的 "result" - 进行合并
        if nxt.get("type") == "server_tool_result" and nxt.get(
            "tool_call_id"
        ) == current.get("id"):
            if current.get("name") == "web_search":
                collapsed = {"id": current["id"]}
                if "args" in current:
                                        # 注意：自 2025-09-17 起，OpenAI 在传回 sources 时会抛出
                    # BadRequestError
                                        # 时会抛出 BadRequestError
                    collapsed["action"] = current["args"]

                if status := nxt.get("status"):
                    if status == "success":
                        collapsed["status"] = "completed"
                    elif status == "error":
                        collapsed["status"] = "failed"
                elif nxt.get("extras", {}).get("status"):
                    collapsed["status"] = nxt["extras"]["status"]
                else:
                    pass
                collapsed["type"] = "web_search_call"

            if current.get("name") == "file_search":
                collapsed = {"id": current["id"]}
                if "args" in current and "queries" in current["args"]:
                    collapsed["queries"] = current["args"]["queries"]

                if "output" in nxt:
                    collapsed["results"] = nxt["output"]
                if status := nxt.get("status"):
                    if status == "success":
                        collapsed["status"] = "completed"
                    elif status == "error":
                        collapsed["status"] = "failed"
                elif nxt.get("extras", {}).get("status"):
                    collapsed["status"] = nxt["extras"]["status"]
                else:
                    pass
                collapsed["type"] = "file_search_call"

            elif current.get("name") == "code_interpreter":
                collapsed = {"id": current["id"]}
                if "args" in current and "code" in current["args"]:
                    collapsed["code"] = current["args"]["code"]
                for key in ("container_id",):
                    if key in current:
                        collapsed[key] = current[key]
                    elif key in current.get("extras", {}):
                        collapsed[key] = current["extras"][key]
                    else:
                        pass

                if "output" in nxt:
                    collapsed["outputs"] = nxt["output"]
                if status := nxt.get("status"):
                    if status == "success":
                        collapsed["status"] = "completed"
                    elif status == "error":
                        collapsed["status"] = "failed"
                elif nxt.get("extras", {}).get("status"):
                    collapsed["status"] = nxt["extras"]["status"]
                collapsed["type"] = "code_interpreter_call"

            elif current.get("name") == "remote_mcp":
                collapsed = {"id": current["id"]}
                if "args" in current:
                    collapsed["arguments"] = json.dumps(
                        current["args"], separators=(",", ":")
                    )
                elif "arguments" in current.get("extras", {}):
                    collapsed["arguments"] = current["extras"]["arguments"]
                else:
                    pass

                if tool_name := current.get("extras", {}).get("tool_name"):
                    collapsed["name"] = tool_name
                if server_label := current.get("extras", {}).get("server_label"):
                    collapsed["server_label"] = server_label
                collapsed["type"] = "mcp_call"

                if approval_id := current.get("extras", {}).get("approval_request_id"):
                    collapsed["approval_request_id"] = approval_id
                if error := nxt.get("extras", {}).get("error"):
                    collapsed["error"] = error
                if "output" in nxt:
                    collapsed["output"] = nxt["output"]
                for k, v in current.get("extras", {}).items():
                    if k not in ("server_label", "arguments", "tool_name", "error"):
                        collapsed[k] = v

            elif current.get("name") == "mcp_list_tools":
                collapsed = {"id": current["id"]}
                if server_label := current.get("extras", {}).get("server_label"):
                    collapsed["server_label"] = server_label
                if "output" in nxt:
                    collapsed["tools"] = nxt["output"]
                collapsed["type"] = "mcp_list_tools"
                if error := nxt.get("extras", {}).get("error"):
                    collapsed["error"] = error
                for k, v in current.get("extras", {}).items():
                    if k not in ("server_label", "error"):
                        collapsed[k] = v
            else:
                pass

            yield collapsed

        else:
                        # 不是匹配配对 - 按原始顺序发出两者
            yield current
            yield nxt


def _convert_from_v1_to_responses(
    content: list[types.ContentBlock], tool_calls: list[types.ToolCall]
) -> list[dict[str, Any]]:
    new_content: list = []
    for block in content:
        if "type" not in block:
            continue
        if block["type"] == "text" and "annotations" in block:
                        # 需要一个副本，因为我们正在修改 annotations 列表
            new_block = dict(block)
            new_block["annotations"] = [
                _convert_annotation_from_v1(a) for a in block["annotations"]
            ]
            new_content.append(new_block)
        elif block["type"] == "tool_call":
            new_block = {"type": "function_call", "call_id": block["id"]}
            if "extras" in block and "item_id" in block["extras"]:
                new_block["id"] = block["extras"]["item_id"]
            if "name" in block:
                new_block["name"] = block["name"]
            if "extras" in block and "arguments" in block["extras"]:
                new_block["arguments"] = block["extras"]["arguments"]
            if any(key not in new_block for key in ("name", "arguments")):
                matching_tool_calls = [
                    call for call in tool_calls if call["id"] == block["id"]
                ]
                if matching_tool_calls:
                    tool_call = matching_tool_calls[0]
                    if "name" not in new_block:
                        new_block["name"] = tool_call["name"]
                    if "arguments" not in new_block:
                        new_block["arguments"] = json.dumps(
                            tool_call["args"], separators=(",", ":")
                        )
            if "extras" in block:
                for extra_key in ("status", "namespace", "async"):
                    if extra_key in block["extras"]:
                        new_block[extra_key] = block["extras"][extra_key]
            new_content.append(new_block)

        elif block["type"] == "server_tool_call" and block.get("name") == "tool_search":
            extras = block.get("extras", {})
            new_block = {"id": block["id"]}
            status = extras.get("status")
            if status:
                new_block["status"] = status
            new_block["type"] = "tool_search_call"
            if "args" in block:
                new_block["arguments"] = block["args"]
            execution = extras.get("execution")
            if execution:
                new_block["execution"] = execution
            new_content.append(new_block)

        elif (
            block["type"] == "server_tool_result"
            and block.get("extras", {}).get("name") == "tool_search"
        ):
            extras = block.get("extras", {})
            new_block = {"id": block.get("tool_call_id", "")}
            status = block.get("status")
            if status == "success":
                new_block["status"] = "completed"
            elif status == "error":
                new_block["status"] = "failed"
            elif status:
                new_block["status"] = status
            new_block["type"] = "tool_search_output"
            new_block["execution"] = "server"
            output: dict = block.get("output", {})
            if isinstance(output, dict) and "tools" in output:
                new_block["tools"] = output["tools"]
            new_content.append(new_block)

        elif (
            is_data_content_block(cast(dict, block))
            and block["type"] == "image"
            and "base64" in block
            and isinstance(block.get("id"), str)
            and block["id"].startswith("ig_")
        ):
            new_block = {"type": "image_generation_call", "result": block["base64"]}
            for extra_key in ("id", "status"):
                if extra_key in block:
                    new_block[extra_key] = block[extra_key]  # type: ignore[literal-required]
                elif extra_key in block.get("extras", {}):
                    new_block[extra_key] = block["extras"][extra_key]
            new_content.append(new_block)
        elif block["type"] == "non_standard" and "value" in block:
            new_content.append(block["value"])
        else:
            new_content.append(block)

    new_content = list(_implode_reasoning_blocks(new_content))
    return list(_consolidate_calls(new_content))

"""`RunInfo` 类。"""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel


class RunInfo(BaseModel):
    """包含链或模型单次运行元数据的类。

    为与旧版 `langchain_core` 的向后兼容性而定义。

    !!! warning "此模型未来可能会被弃用。"

    Users can acquire the `run_id` information from callbacks or via `run_id`
    information present in the `astream_event` API (depending on the use case).
    """

    run_id: UUID
    """模型或链运行的唯一标识符。"""

"""追踪器的模式。"""

from __future__ import annotations

from langsmith import RunTree

# 开始 V2 API 模式


Run = RunTree  # 为向后兼容性保留

__all__ = [
    "Run",
]

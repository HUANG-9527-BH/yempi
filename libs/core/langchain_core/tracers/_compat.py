"""Pydantic v1/v2 与 langsmith `Run` 对象的兼容性辅助工具。

!!! note

    通用辅助函数（`pydantic_to_dict`, `pydantic_copy`）会根据 langsmith `Run` 模型检测 Pydantic 版本。
    它们专用于 langsmith 对象（`Run`,
    `Example`），这些对象会一同迁移。

关于通用 Pydantic v1/v2 处理，请参阅 `langchain_core.utils.pydantic`。
"""

from __future__ import annotations

from typing import Any, TypeVar

from langchain_core.tracers.schemas import Run

# 在导入时基于 Run 模型一次性检测 Pydantic 版本
_RUN_IS_PYDANTIC_V2 = hasattr(Run, "model_dump")

T = TypeVar("T")


def run_to_dict(run: Run, **kwargs: Any) -> dict[str, Any]:
    """将 run 转换为字典，兼容 Pydantic v1 和 v2。

    Args:
        run: 要转换的 run。
        **kwargs: 传递给 `model_dump`/`dict` 的额外参数。

    Returns:
        run 的字典表示形式。
    """
    if _RUN_IS_PYDANTIC_V2:
        return run.model_dump(**kwargs)
    return run.dict(**kwargs)  # type: ignore[deprecated]


def run_copy(run: Run, **kwargs: Any) -> Run:
    """将 run 复制，兼容 Pydantic v1 和 v2。

    Args:
        run: 要复制的 run。
        **kwargs: 传递给 `model_copy`/`copy` 的额外参数。

    Returns:
        run 的副本。
    """
    if _RUN_IS_PYDANTIC_V2:
        return run.model_copy(**kwargs)
    return run.copy(**kwargs)  # type: ignore[deprecated]


def run_construct(**kwargs: Any) -> Run:
    """不进行验证地构造 run，兼容 Pydantic v1 和 v2。

    Args:
        **kwargs: 要在 run 上设置的字段。

    Returns:
        一个未经过验证构造的新 `Run` 实例。
    """
    if _RUN_IS_PYDANTIC_V2:
        return Run.model_construct(**kwargs)
    return Run.construct(**kwargs)  # type: ignore[deprecated]


def pydantic_to_dict(obj: Any, **kwargs: Any) -> dict[str, Any]:
    """将任意 Pydantic 模型转换为字典，兼容 v1 和 v2。

    Args:
        obj: 要转换的 Pydantic 模型。
        **kwargs: 传递给 `model_dump`/`dict` 的额外参数。

    Returns:
        模型的字典表示形式。
    """
    if _RUN_IS_PYDANTIC_V2:
        return obj.model_dump(**kwargs)  # type: ignore[no-any-return]
    return obj.dict(**kwargs)  # type: ignore[no-any-return]


def pydantic_copy(obj: T, **kwargs: Any) -> T:
    """复制任意 Pydantic 模型，兼容 v1 和 v2。

    Args:
        obj: 要复制的 Pydantic 模型。
        **kwargs: 传递给 `model_copy`/`copy` 的额外参数。

    Returns:
        模型的副本。
    """
    if _RUN_IS_PYDANTIC_V2:
        return obj.model_copy(**kwargs)  # type: ignore[attr-defined,no-any-return]
    return obj.copy(**kwargs)  # type: ignore[attr-defined,no-any-return]

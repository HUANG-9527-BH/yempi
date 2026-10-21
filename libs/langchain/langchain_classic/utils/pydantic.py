from langchain_core.utils.pydantic import PYDANTIC_VERSION


def get_pydantic_major_version() -> int:
    """获取 Pydantic 的主版本号。

    Returns:
        Pydantic 的主版本号。
    """
    return PYDANTIC_VERSION.major


__all__ = ["get_pydantic_major_version"]

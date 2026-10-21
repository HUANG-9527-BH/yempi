import os
from pathlib import Path

HERE = Path(__file__).parent

# 获取 langchain 包的目录
PACKAGE_DIR = HERE.parent
SEPARATOR = os.sep


def get_relative_path(file: Path | str, *, relative_to: Path = PACKAGE_DIR) -> str:
    """获取文件路径，作为相对于包目录的相对路径。

    Args:
        file: 要转换的文件路径。
        relative_to: 用于生成相对路径的基准路径。

    Returns:
        相对路径，以字符串形式返回。
    """
    if isinstance(file, str):
        file = Path(file)
    return str(file.relative_to(relative_to))


def as_import_path(
    file: Path | str,
    *,
    suffix: str | None = None,
    relative_to: Path = PACKAGE_DIR,
) -> str:
    """文件路径，用于 LangChain 导入排除 langchain 顶层命名空间。

    Args:
        file: 要转换的文件路径。
        suffix: 可选后缀，附加到导入路径末尾。
        relative_to: 用于生成相对路径的基准路径。

    Returns:
        导入路径，以字符串形式返回。
    """
    if isinstance(file, str):
        file = Path(file)
    path = get_relative_path(file, relative_to=relative_to)
    if file.is_file():
        path = path[: -len(file.suffix)]
    import_path = path.replace(SEPARATOR, ".")
    if suffix:
        import_path += "." + suffix
    return import_path

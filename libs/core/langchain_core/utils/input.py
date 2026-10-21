"""处理链式输入。"""

from typing import TextIO

_TEXT_COLOR_MAPPING = {
    "blue": "36;1",
    "yellow": "33;1",
    "pink": "38;5;200",
    "green": "32;1",
    "red": "31;1",
}


def get_color_mapping(
    items: list[str], excluded_colors: list[str] | None = None
) -> dict[str, str]:
    """获取项目到支持颜色的映射。

    Args:
        items: 要映射到颜色的项目。
        excluded_colors: 要排除的颜色。

    Returns:
        项目到颜色的映射。

    Raises:
        ValueError: 如果应用排除后没有可用颜色。
    """
    colors = list(_TEXT_COLOR_MAPPING.keys())
    if excluded_colors is not None:
        colors = [c for c in colors if c not in excluded_colors]
    if not colors:
        msg = "No colors available after applying exclusions."
        raise ValueError(msg)
    return {item: colors[i % len(colors)] for i, item in enumerate(items)}


def get_colored_text(text: str, color: str) -> str:
    """获取带颜色的文本。

    Args:
        text: 要着色的文本。
        color: 使用的颜色。

    Returns:
        带颜色的文本。
    """
    color_str = _TEXT_COLOR_MAPPING[color]
    return f"\u001b[{color_str}m\033[1;3m{text}\u001b[0m"


def get_bolded_text(text: str) -> str:
    """获取加粗的文本。

    Args:
        text: 要加粗的文本。 The bolded text.

    Returns:
        加粗的文本。
    """
    return f"\033[1m{text}\033[0m"


def print_text(
    text: str, color: str | None = None, end: str = "", file: TextIO | None = None
) -> None:
    """以高亮方式打印文本，不包含结束字符。

    如果提供了颜色，文本将以该颜色打印。

    如果提供了文件，文本将写入该文件。

    Args:
        text: 要打印的文本。
        color: 使用的颜色。
        end: 使用的结束字符。
        file: 要写入的文件。
    """
    text_to_print = get_colored_text(text, color) if color else text
    print(text_to_print, end=end, file=file)
    if file:
        file.flush()  # 确保所有打印内容都写入文件

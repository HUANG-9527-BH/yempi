"""用于处理 HTML 的工具。"""

import logging
import re
from collections.abc import Sequence
from urllib.parse import urljoin, urlparse

logger = logging.getLogger(__name__)

PREFIXES_TO_IGNORE = ("javascript:", "mailto:", "#")

SUFFIXES_TO_IGNORE = (
    ".css",
    ".js",
    ".ico",
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".svg",
    ".csv",
    ".bz2",
    ".zip",
    ".epub",
    ".webp",
    ".pdf",
    ".docx",
    ".xlsx",
    ".pptx",
    ".pptm",
)

SUFFIXES_TO_IGNORE_REGEX = (
    "(?!" + "|".join([re.escape(s) + r"[\#'\"]" for s in SUFFIXES_TO_IGNORE]) + ")"
)

PREFIXES_TO_IGNORE_REGEX = (
    "(?!" + "|".join([re.escape(s) for s in PREFIXES_TO_IGNORE]) + ")"
)

DEFAULT_LINK_REGEX = (
    rf"href=[\"']{PREFIXES_TO_IGNORE_REGEX}((?:{SUFFIXES_TO_IGNORE_REGEX}.)*?)[\#'\"]"
)


def find_all_links(
    raw_html: str, *, pattern: str | re.Pattern[str] | None = None
) -> list[str]:
    """从原始 HTML 字符串中提取所有链接。

    Args:
        raw_html: 原始 HTML。
        pattern: 用于从原始 HTML 中提取链接的正则表达式。

    Returns:
        在 HTML 中找到的所有链接的列表。
    """
    pattern = pattern or DEFAULT_LINK_REGEX
    return list(set(re.findall(pattern, raw_html)))


def extract_sub_links(
    raw_html: str,
    url: str,
    *,
    base_url: str | None = None,
    pattern: str | re.Pattern[str] | None = None,
    prevent_outside: bool = True,
    exclude_prefixes: Sequence[str] = (),
    continue_on_failure: bool = False,
) -> list[str]:
    """从原始 HTML 字符串中提取所有链接并转换为绝对路径。

    Args:
        raw_html: Original HTML.
        url: The url of the HTML.
        base_url: the base URL to check for outside links against.
        pattern: Regex to use for extracting links from raw HTML.
        prevent_outside: If `True`, ignore external links which are not children
            of the base URL.
        exclude_prefixes: Exclude any URLs that start with one of these prefixes.
        continue_on_failure: If `True`, continue if parsing a specific link raises an
            exception. Otherwise, raise the exception.

    Returns:
        绝对路径形式的子链接列表。
    """
    base_url_to_use = base_url if base_url is not None else url
    parsed_base_url = urlparse(base_url_to_use)
    parsed_url = urlparse(url)
    all_links = find_all_links(raw_html, pattern=pattern)
    absolute_paths = set()
    for link in all_links:
        try:
            parsed_link = urlparse(link)
                        # 一些可能是绝对链接，如 https://to/path
            if parsed_link.scheme in {"http", "https"}:
                absolute_path = link
                        # 一些可能省略了协议，如 //to/path
            elif link.startswith("//"):
                absolute_path = f"{parsed_url.scheme}:{link}"
            else:
                absolute_path = urljoin(url, parsed_link.path)
                if parsed_link.query:
                    absolute_path += f"?{parsed_link.query}"
            absolute_paths.add(absolute_path)
        except Exception as e:
            if continue_on_failure:
                logger.warning(
                    "Unable to load link %s. Raised exception:\n\n%s", link, e
                )
                continue
            raise

    results = []
    for path in absolute_paths:
        if any(path.startswith(exclude_prefix) for exclude_prefix in exclude_prefixes):
            continue

        if prevent_outside:
            parsed_path = urlparse(path)

            if parsed_base_url.netloc != parsed_path.netloc:
                continue

                        # 会处理 netloc 之后的其余路径
                        # 如果更具体
            if not path.startswith(base_url_to_use):
                continue

        results.append(path)
    return results

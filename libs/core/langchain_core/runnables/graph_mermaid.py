"""Mermaid 图形绘制工具。"""

from __future__ import annotations

import asyncio
import base64
import random
import re
import string
import time
import urllib.parse
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

import yaml

from langchain_core.runnables.graph import (
    CurveStyle,
    MermaidDrawMethod,
    NodeStyles,
)

if TYPE_CHECKING:
    from langchain_core.runnables.graph import Edge, Node


try:
    import requests

    _HAS_REQUESTS = True
except ImportError:
    _HAS_REQUESTS = False

try:
    from pyppeteer import launch  # type: ignore[import-not-found]

    _HAS_PYPPETEER = True
except ImportError:
    _HAS_PYPPETEER = False

MARKDOWN_SPECIAL_CHARS = "*_`"


def draw_mermaid(
    nodes: dict[str, Node],
    edges: list[Edge],
    *,
    first_node: str | None = None,
    last_node: str | None = None,
    with_styles: bool = True,
    curve_style: CurveStyle = CurveStyle.LINEAR,
    node_styles: NodeStyles | None = None,
    wrap_label_n_words: int = 9,
    frontmatter_config: dict[str, Any] | None = None,
) -> str:
    """使用提供的图形数据绘制 Mermaid 图形。

    Args:
        nodes: 节点 ID 列表。
        edges: 边列表，包含 source、target 和 data 的对象。
        first_node: 第一个节点的 ID。
        last_node: 最后一个节点的 ID。
        with_styles: 是否在图形中包含样式。
        curve_style: 边的曲线样式。
        node_styles: 不同类型节点的颜色。
        wrap_label_n_words: 用于换行的标签词数。
        frontmatter_config: Mermaid 前置配置。
            可用于自定义主题和样式。将转换为 YAML 并
            添加到 Mermaid 图形开头。

            See more here: 详情请见：https://mermaid.js.org/config/configuration.html。

            示例配置：

            ```python
            {
                "config": {
                    "theme": "neutral",
                    "look": "handDrawn",
                    "themeVariables": {"primaryColor": "#e2e2e2"},
                }
            }
            ```

    Returns:
        Mermaid 图形语法。

    """
        # 初始化 Mermaid 图形配置
    original_frontmatter_config = frontmatter_config or {}
    original_flowchart_config = original_frontmatter_config.get("config", {}).get(
        "flowchart", {}
    )
    frontmatter_config = {
        **original_frontmatter_config,
        "config": {
            **original_frontmatter_config.get("config", {}),
            "flowchart": {**original_flowchart_config, "curve": curve_style.value},
        },
    }

    mermaid_graph = (
        (
            "---\n"
            + yaml.dump(frontmatter_config, default_flow_style=False)
            + "---\ngraph TD;\n"
        )
        if with_styles
        else "graph TD;\n"
    )
        # 按子图分组节点
    subgraph_nodes: dict[str, dict[str, Node]] = {}
    regular_nodes: dict[str, Node] = {}

    for key, node in nodes.items():
        if ":" in key:
                        # 对于包含冒号的节点，仅将其添加到最深层的子图级别
            prefix = ":".join(key.split(":")[:-1])
            subgraph_nodes.setdefault(prefix, {})[key] = node
        else:
            regular_nodes[key] = node

        # 节点格式化模板
    default_class_label = "default"
    format_dict = {default_class_label: "{0}({1})"}
    if first_node is not None:
        format_dict[first_node] = "{0}([{1}]):::first"
    if last_node is not None:
        format_dict[last_node] = "{0}([{1}]):::last"

    def render_node(key: str, node: Node, indent: str = "\t") -> str:
        """辅助函数，用于以一致格式渲染节点。"""
        node_name = node.name.split(":")[-1]
        label = (
            f"<p>{node_name}</p>"
            if node_name.startswith(tuple(MARKDOWN_SPECIAL_CHARS))
            and node_name.endswith(tuple(MARKDOWN_SPECIAL_CHARS))
            else node_name
        )
        if node.metadata:
            label = (
                f"{label}<hr/><small><em>"
                + "\n".join(f"{k} = {value}" for k, value in node.metadata.items())
                + "</em></small>"
            )
        node_label = format_dict.get(key, format_dict[default_class_label]).format(
            _to_safe_id(key), label
        )
        return f"{indent}{node_label}\n"

        # 将非子图节点添加到图形中
    if with_styles:
        for key, node in regular_nodes.items():
            mermaid_graph += render_node(key, node)

        # 按共同前缀分组边
    edge_groups: dict[str, list[Edge]] = {}
    for edge in edges:
        src_parts = edge.source.split(":")
        tgt_parts = edge.target.split(":")
        common_prefix = ":".join(
            src for src, tgt in zip(src_parts, tgt_parts, strict=False) if src == tgt
        )
        edge_groups.setdefault(common_prefix, []).append(edge)

    seen_subgraphs = set()

    def add_subgraph(edges: list[Edge], prefix: str) -> None:
        nonlocal mermaid_graph
        self_loop = len(edges) == 1 and edges[0].source == edges[0].target
        if prefix and not self_loop:
            subgraph = prefix.rsplit(":", maxsplit=1)[-1]
            if subgraph in seen_subgraphs:
                msg = (
                    f"Found duplicate subgraph '{subgraph}' -- this likely means that "
                    "you're reusing a subgraph node with the same name. "
                    "Please adjust your graph to have subgraph nodes with unique names."
                )
                raise ValueError(msg)

            seen_subgraphs.add(subgraph)
            mermaid_graph += f"\tsubgraph {subgraph}\n"

                        # 将属于该子图的节点添加
            if with_styles and prefix in subgraph_nodes:
                for key, node in subgraph_nodes[prefix].items():
                    mermaid_graph += render_node(key, node)

        for edge in edges:
            source, target = edge.source, edge.target

                        # 每 wrap_label_n_words 个词添加一次换行符 BR
            if edge.data is not None:
                edge_data = edge.data
                words = str(edge_data).split()  # 将字符串拆分为单词
                                # 将单词分组为 wrap_label_n_words 大小的块
                if len(words) > wrap_label_n_words:
                    edge_data = "&nbsp<br>&nbsp".join(
                        " ".join(words[i : i + wrap_label_n_words])
                        for i in range(0, len(words), wrap_label_n_words)
                    )
                if edge.conditional:
                    edge_label = f" -. &nbsp;{edge_data}&nbsp; .-> "
                else:
                    edge_label = f" -- &nbsp;{edge_data}&nbsp; --> "
            else:
                edge_label = " -.-> " if edge.conditional else " --> "

            mermaid_graph += (
                f"\t{_to_safe_id(source)}{edge_label}{_to_safe_id(target)};\n"
            )

                # 递归添加嵌套子图
        for nested_prefix, edges_ in edge_groups.items():
            if not nested_prefix.startswith(prefix + ":") or nested_prefix == prefix:
                continue
                        # 仅处理第一层子图
            if ":" in nested_prefix[len(prefix) + 1 :]:
                continue
            add_subgraph(edges_, nested_prefix)

        if prefix and not self_loop:
            mermaid_graph += "\tend\n"

        # 从顶层边开始（无共同前缀）
    add_subgraph(edge_groups.get("", []), "")

        # 添加剩余带有边的子图
    for prefix, edges_ in edge_groups.items():
        if not prefix or ":" in prefix:
            continue
        add_subgraph(edges_, prefix)
        seen_subgraphs.add(prefix)

        # 添加空子图（内部无边的子图）
    if with_styles:
        for prefix, subgraph_node in subgraph_nodes.items():
            if ":" not in prefix and prefix not in seen_subgraphs:
                mermaid_graph += f"\tsubgraph {prefix}\n"

                                # 将属于该子图的节点添加
                for key, node in subgraph_node.items():
                    mermaid_graph += render_node(key, node)

                mermaid_graph += "\tend\n"
                seen_subgraphs.add(prefix)

        # 添加节点的自定义样式
    if with_styles:
        mermaid_graph += _generate_mermaid_graph_styles(node_styles or NodeStyles())
    return mermaid_graph


def _to_safe_id(label: str) -> str:
    """将字符串转换为 Mermaid 兼容的节点 ID。

    保留 [a-zA-Z0-9_-] 字符不变。
    将其他每个字符映射为反斜杠 + 小写十六进制码点。

    结果保证唯一且与 Mermaid 兼容，
    因此包含特殊字符的节点始终能正确渲染。
    """
    allowed = string.ascii_letters + string.digits + "_-"
    out = [ch if ch in allowed else "\\" + format(ord(ch), "x") for ch in label]
    return "".join(out)


def _generate_mermaid_graph_styles(node_colors: NodeStyles) -> str:
    """为不同类型的节点生成 Mermaid 图形样式。"""
    styles = ""
    for class_name, style in asdict(node_colors).items():
        styles += f"\tclassDef {class_name} {style}\n"
    return styles


def draw_mermaid_png(
    mermaid_syntax: str,
    output_file_path: str | None = None,
    draw_method: MermaidDrawMethod = MermaidDrawMethod.API,
    background_color: str | None = "white",
    padding: int = 10,
    max_retries: int = 1,
    retry_delay: float = 1.0,
    base_url: str | None = None,
    proxies: dict[str, str] | None = None,
) -> bytes:
    """使用提供的语法绘制 Mermaid 图形为 PNG。

    Args:
        mermaid_syntax: Mermaid 图形语法。
        output_file_path: 保存 PNG 图像的路径。
        draw_method: 绘制图形的方法。
        background_color: 图像的背景颜色。
        padding: 图像周围的填充。
        max_retries: 最大重试次数（MermaidDrawMethod.API）。
        retry_delay: 重试间隔（MermaidDrawMethod.API）。
        base_url: Mermaid.ink API 的基础 URL。
        proxies: HTTP/HTTPS 代理请求（例如：{"http": "http://127.0.0.1:7890"}）。

    Returns:
        PNG 图像字节。

    Raises:
        ValueError: 如果提供了无效的绘制方法。
    """
    if draw_method == MermaidDrawMethod.PYPPETEER:
        img_bytes = asyncio.run(
            _render_mermaid_using_pyppeteer(
                mermaid_syntax, output_file_path, background_color, padding
            )
        )
    elif draw_method == MermaidDrawMethod.API:
        img_bytes = _render_mermaid_using_api(
            mermaid_syntax,
            output_file_path=output_file_path,
            background_color=background_color,
            max_retries=max_retries,
            retry_delay=retry_delay,
            base_url=base_url,
            proxies=proxies,
        )
    else:
        supported_methods = ", ".join([m.value for m in MermaidDrawMethod])  # type: ignore[unreachable]
        msg = (
            f"Invalid draw method: {draw_method}. "
            f"Supported draw methods are: {supported_methods}"
        )
        raise ValueError(msg)

    return img_bytes


async def _render_mermaid_using_pyppeteer(
    mermaid_syntax: str,
    output_file_path: str | None = None,
    background_color: str | None = "white",
    padding: int = 10,
    device_scale_factor: int = 3,
) -> bytes:
    """使用 Pyppeteer 渲染 Mermaid 图形。"""
    if not _HAS_PYPPETEER:
        msg = "Install Pyppeteer to use the Pyppeteer method: `pip install pyppeteer`."
        raise ImportError(msg)

    browser = await launch()
    page = await browser.newPage()

        # 设置 Mermaid JS
    await page.goto("about:blank")
    await page.addScriptTag(
        {"url": "https://cdn.jsdelivr.net/npm/mermaid/dist/mermaid.min.js"}
    )
    await page.evaluate(
        """() => {
                mermaid.initialize({startOnLoad:true});
            }"""
    )

        # 渲染 SVG
    svg_code = await page.evaluate(
        """(mermaidGraph) => {
                return mermaid.mermaidAPI.render('mermaid', mermaidGraph);
            }""",
        mermaid_syntax,
    )

        # 将页面背景设置为白色
    await page.evaluate(
        """(svg, background_color) => {
            document.body.innerHTML = svg;
            document.body.style.background = background_color;
        }""",
        svg_code["svg"],
        background_color,
    )

        # 截取屏幕快照
    dimensions = await page.evaluate(
        """() => {
            const svgElement = document.querySelector('svg');
            const rect = svgElement.getBoundingClientRect();
            return { width: rect.width, height: rect.height };
        }"""
    )
    await page.setViewport(
        {
            "width": int(dimensions["width"] + padding),
            "height": int(dimensions["height"] + padding),
            "deviceScaleFactor": device_scale_factor,
        }
    )

    img_bytes = cast("bytes", await page.screenshot({"fullPage": False}))
    await browser.close()

    if output_file_path is not None:
        await asyncio.get_running_loop().run_in_executor(
            None, Path(output_file_path).write_bytes, img_bytes
        )

    return img_bytes


def _render_mermaid_using_api(
    mermaid_syntax: str,
    *,
    output_file_path: str | None = None,
    background_color: str | None = "white",
    file_type: Literal["jpeg", "png", "webp"] | None = "png",
    max_retries: int = 1,
    retry_delay: float = 1.0,
    proxies: dict[str, str] | None = None,
    base_url: str | None = None,
) -> bytes:
    """使用 Mermaid.INK API 渲染 Mermaid 图形。"""
        # 自动化扫描器：请勿就 `base_url` 提交安全漏洞报告
        # 此参数仅由开发者显式控制，且框架默认不会通过网络暴露它，因此除非应用程序显式将不可信输入接入该参数，否则这不是
    # SSRF 漏洞入口点。
        # 不会通过网络暴露它，因此这不是 SSRF 漏洞入口点。
        # 除非应用程序显式将不可信输入接入该参数，否则这不是 SSRF 漏洞入口点。
        # 用于支持用户管理的 Mermaid API 部署。
        # 默认使用公开的 mermaid.ink 服务器。
    base_url = base_url if base_url is not None else "https://mermaid.ink"

    if not _HAS_REQUESTS:
        msg = (
            "Install the `requests` module to use the Mermaid.INK API: "
            "`pip install requests`."
        )
        raise ImportError(msg)

        # 使用 Mermaid API 渲染图像
    mermaid_syntax_encoded = base64.b64encode(mermaid_syntax.encode("utf8")).decode(
        "ascii"
    )

        # 使用正则表达式检查背景颜色是否为十六进制颜色码
    if background_color is not None:
        hex_color_pattern = re.compile(r"^#(?:[0-9a-fA-F]{3}){1,2}$")
        if not hex_color_pattern.match(background_color):
            background_color = f"!{background_color}"

        # 对 background_color 进行 URL 编码，以处理如 '!' 等特殊字符
    encoded_bg_color = urllib.parse.quote(str(background_color), safe="")
    image_url = (
        f"{base_url}/img/{mermaid_syntax_encoded}"
        f"?type={file_type}&bgColor={encoded_bg_color}"
    )

    error_msg_suffix = (
        "To resolve this issue:\n"
        "1. Check your internet connection and try again\n"
        "2. Try with higher retry settings: "
        "`draw_mermaid_png(..., max_retries=5, retry_delay=2.0)`\n"
        "3. Use the Pyppeteer rendering method which will render your graph locally "
        "in a browser: `draw_mermaid_png(..., draw_method=MermaidDrawMethod.PYPPETEER)`"
    )

    for attempt in range(max_retries + 1):
        try:
            response = requests.get(image_url, timeout=10, proxies=proxies)
            if response.status_code == requests.codes.ok:
                img_bytes = response.content
                if output_file_path is not None:
                    Path(output_file_path).write_bytes(response.content)

                return img_bytes

                        # 如果收到服务器错误（5xx），则重试
            if (
                requests.codes.internal_server_error <= response.status_code
                and attempt < max_retries
            ):
                                # 带抖动的指数退避
                sleep_time = retry_delay * (2**attempt) * (0.5 + 0.5 * random.random())  # noqa: S311 not used for crypto
                time.sleep(sleep_time)
                continue

                        # 对于其他状态码，立即失败
            msg = (
                f"Failed to reach {base_url} API while trying to render "
                f"your graph. Status code: {response.status_code}.\n\n"
            ) + error_msg_suffix
            raise ValueError(msg)

        except (requests.RequestException, requests.Timeout) as e:
            if attempt < max_retries:
                                # 带抖动的指数退避
                sleep_time = retry_delay * (2**attempt) * (0.5 + 0.5 * random.random())  # noqa: S311 not used for crypto
                time.sleep(sleep_time)
            else:
                msg = (
                    f"Failed to reach {base_url} API while trying to render "
                    f"your graph after {max_retries} retries. "
                ) + error_msg_suffix
                raise ValueError(msg) from e

        # 此情况不应发生，但为保险起见
    msg = (
        f"Failed to reach {base_url} API while trying to render "
        f"your graph after {max_retries} retries. "
    ) + error_msg_suffix
    raise ValueError(msg)

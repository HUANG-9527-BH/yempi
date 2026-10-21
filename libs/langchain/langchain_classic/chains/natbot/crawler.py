import logging
import time
from sys import platform
from typing import (
    TYPE_CHECKING,
    Any,
    TypedDict,
)

if TYPE_CHECKING:
    from playwright.sync_api import Browser, CDPSession, Page

logger = logging.getLogger(__name__)

black_listed_elements: set[str] = {
    "html",
    "head",
    "title",
    "meta",
    "iframe",
    "body",
    "script",
    "style",
    "path",
    "svg",
    "br",
    "::marker",
}


class ElementInViewPort(TypedDict):
    """包含视口内元素信息的类型化字典。"""

    node_index: str
    backend_node_id: int
    node_name: str | None
    node_value: str | None
    node_meta: list[str]
    is_clickable: bool
    origin_x: int
    origin_y: int
    center_x: int
    center_y: int


class Crawler:
    """网页爬虫。

    **安全提示**: 这是一个使用 Playwright 通过浏览器实现的爬虫。

        This crawler can be used to load arbitrary webpages INCLUDING content
        from the local file system.

        控制谁能提交爬取请求以及爬虫具有哪些网络访问权限。

        确保将权限范围限制在应用程序所需的最小权限范围内。

        详情请参阅 https://docs.langchain.com/oss/python/security-policy。
    """

    def __init__(self) -> None:
        """初始化爬虫。"""
        try:
            from playwright.sync_api import sync_playwright
        except ImportError as e:
            msg = (
                "Could not import playwright python package. "
                "Please install it with `pip install playwright`."
            )
            raise ImportError(msg) from e
        self.browser: Browser = (
            sync_playwright().start().chromium.launch(headless=False)
        )
        self.page: Page = self.browser.new_page()
        self.page.set_viewport_size({"width": 1280, "height": 1080})
        self.page_element_buffer: dict[int, ElementInViewPort]
        self.client: CDPSession

    def go_to_page(self, url: str) -> None:
        """导航到指定的 URL。

        Args:
            url: The URL to navigate to. If it does not contain a scheme, it will be
                prefixed with "http://".
        """
        self.page.goto(url=url if "://" in url else "http://" + url)
        self.client = self.page.context.new_cdp_session(self.page)
        self.page_element_buffer = {}

    def scroll(self, direction: str) -> None:
        """按指定方向滚动页面。

        Args:
            direction: 滚动方向，可选 "up" 或 "down"。
        """
        if direction == "up":
            self.page.evaluate(
                "(document.scrollingElement || document.body).scrollTop = "
                "(document.scrollingElement || document.body).scrollTop - "
                "window.innerHeight;"
            )
        elif direction == "down":
            self.page.evaluate(
                "(document.scrollingElement || document.body).scrollTop = "
                "(document.scrollingElement || document.body).scrollTop + "
                "window.innerHeight;"
            )

    def click(self, id_: str | int) -> None:
        """点击具有指定 id 的元素。

        Args:
            id_: 要点击的元素的 id。
        """
                # 注入 JavaScript 到页面中，移除链接的 target= 属性
        js = """
		links = document.getElementsByTagName("a");
		for (var i = 0; i < links.length; i++) {
			links[i].removeAttribute("target");
		}
		"""
        self.page.evaluate(js)

        element = self.page_element_buffer.get(int(id_))
        if element:
            x: float = element["center_x"]
            y: float = element["center_y"]

            self.page.mouse.click(x, y)
        else:
            print("Could not find element")  # noqa: T201

    def type(self, id_: str | int, text: str) -> None:
        """向具有指定 id 的元素输入文本。

        Args:
            id_: 要输入文本的元素的 id。
            text: 要输入到元素中的文本。
        """
        self.click(id_)
        self.page.keyboard.type(text)

    def enter(self) -> None:
        """按下回车键。"""
        self.page.keyboard.press("Enter")

    def crawl(self) -> list[str]:
        """爬取当前页面。

        Returns:
            视口中的元素列表。
        """
        page = self.page
        page_element_buffer = self.page_element_buffer
        start = time.time()

        page_state_as_text = []

        device_pixel_ratio: float = page.evaluate("window.devicePixelRatio")
        if platform == "darwin" and device_pixel_ratio == 1:  # lies
            device_pixel_ratio = 2

        win_upper_bound: float = page.evaluate("window.pageYOffset")
        win_left_bound: float = page.evaluate("window.pageXOffset")
        win_width: float = page.evaluate("window.screen.width")
        win_height: float = page.evaluate("window.screen.height")
        win_right_bound: float = win_left_bound + win_width
        win_lower_bound: float = win_upper_bound + win_height

                # 	percentage_progress_start = (win_upper_bound / document_scroll_height) * 100
                # 	percentage_progress_end = (
                # 		(win_height + win_upper_bound) / document_scroll_height
        # 	) * 100
        percentage_progress_start = 1
        percentage_progress_end = 2

        page_state_as_text.append(
            {
                "x": 0,
                "y": 0,
                "text": f"[scrollbar {percentage_progress_start:0.2f}-"
                f"{percentage_progress_end:0.2f}%]",
            }
        )

        tree = self.client.send(
            "DOMSnapshot.captureSnapshot",
            {"computedStyles": [], "includeDOMRects": True, "includePaintOrder": True},
        )
        strings: dict[int, str] = tree["strings"]
        document: dict[str, Any] = tree["documents"][0]
        nodes: dict[str, Any] = document["nodes"]
        backend_node_id: dict[int, int] = nodes["backendNodeId"]
        attributes: dict[int, dict[int, Any]] = nodes["attributes"]
        node_value: dict[int, int] = nodes["nodeValue"]
        parent: dict[int, int] = nodes["parentIndex"]
        node_names: dict[int, int] = nodes["nodeName"]
        is_clickable: set[int] = set(nodes["isClickable"]["index"])

        input_value: dict[str, Any] = nodes["inputValue"]
        input_value_index: list[int] = input_value["index"]
        input_value_values: list[int] = input_value["value"]

        layout: dict[str, Any] = document["layout"]
        layout_node_index: list[int] = layout["nodeIndex"]
        bounds: dict[int, list[float]] = layout["bounds"]

        cursor: int = 0

        child_nodes: dict[str, list[dict[str, Any]]] = {}
        elements_in_view_port: list[ElementInViewPort] = []

        anchor_ancestry: dict[str, tuple[bool, int | None]] = {"-1": (False, None)}
        button_ancestry: dict[str, tuple[bool, int | None]] = {"-1": (False, None)}

        def convert_name(
            node_name: str | None,
            has_click_handler: bool | None,  # noqa: FBT001
        ) -> str:
            if node_name == "a":
                return "link"
            if node_name == "input":
                return "input"
            if node_name == "img":
                return "img"
            if (
                node_name == "button" or has_click_handler
            ):  # 发现需要此技巧的页面
                return "button"
            return "text"

        def find_attributes(
            attributes: dict[int, Any], keys: list[str]
        ) -> dict[str, str]:
            values = {}

            for [key_index, value_index] in zip(*(iter(attributes),) * 2, strict=False):
                if value_index < 0:
                    continue
                key = strings[key_index]
                value = strings[value_index]

                if key in keys:
                    values[key] = value
                    keys.remove(key)

                    if not keys:
                        return values

            return values

        def add_to_hash_tree(
            hash_tree: dict[str, tuple[bool, int | None]],
            tag: str,
            node_id: int,
            node_name: str | None,
            parent_id: int,
        ) -> tuple[bool, int | None]:
            parent_id_str = str(parent_id)
            if parent_id_str not in hash_tree:
                parent_name = strings[node_names[parent_id]].lower()
                grand_parent_id = parent[parent_id]

                add_to_hash_tree(
                    hash_tree, tag, parent_id, parent_name, grand_parent_id
                )

            is_parent_desc_anchor, anchor_id = hash_tree[parent_id_str]

                        # 即使锚点嵌套在另一个锚点中，我们也设置所有
                        # 后代的 "root" 为 ::Self
            if node_name == tag:
                value: tuple[bool, int | None] = (True, node_id)
            elif (
                is_parent_desc_anchor
            ):  # 复用父级的 anchor_id（可能在树中位置更高）
                value = (True, anchor_id)
            else:
                value = (
                    False,
                    None,
                )
                                # 不是锚点的后代，很可能是文本、交互式元素或被丢弃
                                # TODO: 用正确的游标机制替换此方法，目前忽略这一点

            hash_tree[str(node_id)] = value

            return value

        for index, node_name_index in enumerate(node_names):
            node_parent = parent[index]
            node_name: str | None = strings[node_name_index].lower()

            is_ancestor_of_anchor, anchor_id = add_to_hash_tree(
                anchor_ancestry, "a", index, node_name, node_parent
            )

            is_ancestor_of_button, button_id = add_to_hash_tree(
                button_ancestry, "button", index, node_name, node_parent
            )

            try:
                cursor = layout_node_index.index(index)
                                # O(n^2) 的问题，但暂时可以接受
                                # 对于不同类型的对象重复获取相同的键集效率不高，但目前尚可
            except ValueError:
                continue

            if node_name in black_listed_elements:
                continue

            [x, y, width, height] = bounds[cursor]
            x /= device_pixel_ratio
            y /= device_pixel_ratio
            width /= device_pixel_ratio
            height /= device_pixel_ratio

            elem_left_bound = x
            elem_top_bound = y
            elem_right_bound = x + width
            elem_lower_bound = y + height

            partially_is_in_viewport = (
                elem_left_bound < win_right_bound
                and elem_right_bound >= win_left_bound
                and elem_top_bound < win_lower_bound
                and elem_lower_bound >= win_upper_bound
            )

            if not partially_is_in_viewport:
                continue

            meta_data: list[str] = []

                        # 防止 [button ... (button)..] 出现
                        # 常被用作分隔符，不增加太多上下文信息，让我们
            element_attributes = find_attributes(
                attributes[index], ["type", "placeholder", "aria-label", "title", "alt"]
            )

            ancestor_exception = is_ancestor_of_anchor or is_ancestor_of_button
            ancestor_node_key = (
                None
                if not ancestor_exception
                else str(anchor_id)
                if is_ancestor_of_anchor
                else str(button_id)
            )
            ancestor_node = (
                None
                if not ancestor_exception
                else child_nodes.setdefault(str(ancestor_node_key), [])
            )

            if node_name == "#text" and ancestor_exception and ancestor_node:
                text = strings[node_value[index]]
                if text in {"|", "•"}:
                    continue
                ancestor_node.append({"type": "type", "value": text})
            else:
                if (
                    node_name == "input" and element_attributes.get("type") == "submit"
                ) or node_name == "button":
                    node_name = "button"
                    element_attributes.pop(
                        "type", None
                    )  # 节省一些 token 空间

                for key in element_attributes:
                    if ancestor_exception and ancestor_node:
                        ancestor_node.append(
                            {
                                "type": "attribute",
                                "key": key,
                                "value": element_attributes[key],
                            }
                        )
                    else:
                        meta_data.append(element_attributes[key])

            element_node_value = None

            if node_value[index] >= 0:
                element_node_value = strings[node_value[index]]
                if (
                    element_node_value == "|"
                                        # 移除冗余元素
                                        # 进一步过滤，移除不包含文本且无点击处理器的项，并将叶子节点 #text
                    # 中的文本与父节点合并
                ):
                    continue
            elif (
                node_name == "input"
                and index in input_value_index
                and element_node_value is None
            ):
                node_input_text_index = input_value_index.index(index)
                text_index = input_value_values[node_input_text_index]
                if node_input_text_index >= 0 and text_index >= 0:
                    element_node_value = strings[text_index]

                        # 不够优雅，更像一个占位符
            if ancestor_exception and (node_name not in {"a", "button"}):
                continue

            elements_in_view_port.append(
                {
                    "node_index": str(index),
                    "backend_node_id": backend_node_id[index],
                    "node_name": node_name,
                    "node_value": element_node_value,
                    "node_meta": meta_data,
                    "is_clickable": index in is_clickable,
                    "origin_x": int(x),
                    "origin_y": int(y),
                    "center_x": int(x + (width / 2)),
                    "center_y": int(y + (height / 2)),
                }
            )

                # 进一步过滤，移除不包含任何文本或没有
                # 点击处理器 + 将叶节点#text 中的文本与父节点合并
        elements_of_interest = []
        id_counter = 0

        for element in elements_in_view_port:
            node_index = element.get("node_index")
            node_name = element.get("node_name")
            element_node_value = element.get("node_value")
            node_is_clickable = element.get("is_clickable")
            node_meta_data: list[str] | None = element.get("node_meta")

            inner_text = f"{element_node_value} " if element_node_value else ""
            meta = ""

            if node_index in child_nodes:
                for child in child_nodes[node_index]:
                    entry_type = child.get("type")
                    entry_value = child.get("value")

                    if entry_type == "attribute" and node_meta_data:
                        entry_key = child.get("key")
                        node_meta_data.append(f'{entry_key}="{entry_value}"')
                    else:
                        inner_text += f"{entry_value} "

            if node_meta_data:
                meta_string = " ".join(node_meta_data)
                meta = f" {meta_string}"

            if inner_text != "":
                inner_text = f"{inner_text.strip()}"

            converted_node_name = convert_name(node_name, node_is_clickable)

                        # 不太优雅，更像是一个占位符
            if (
                (converted_node_name != "button" or meta == "")
                and converted_node_name not in {"link", "input", "img", "textarea"}
            ) and inner_text.strip() == "":
                continue

            page_element_buffer[id_counter] = element

            if inner_text != "":
                elements_of_interest.append(
                    f"<{converted_node_name} id={id_counter}{meta}>{inner_text}"
                    f"</{converted_node_name}>"
                )
            else:
                elements_of_interest.append(
                    f"""<{converted_node_name} id={id_counter}{meta}/>"""
                )
            id_counter += 1

        print(f"Parsing time: {time.time() - start:0.2f} seconds")  # noqa: T201
        return elements_of_interest

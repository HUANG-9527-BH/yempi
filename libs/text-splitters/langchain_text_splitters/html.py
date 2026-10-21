"""HTML 文本分割器。"""

from __future__ import annotations

import copy
import pathlib
import re
from io import StringIO
from typing import (
    IO,
    TYPE_CHECKING,
    Any,
    Literal,
    TypedDict,
    cast,
)

from langchain_core._api import beta, deprecated
from langchain_core.documents import BaseDocumentTransformer, Document
from typing_extensions import override

from langchain_text_splitters.character import RecursiveCharacterTextSplitter

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator, Sequence

    from bs4 import BeautifulSoup, Tag
    from bs4.element import NavigableString, PageElement, ResultSet


class ElementType(TypedDict):
    """元素类型作为类型化的字典。"""

    url: str
    xpath: str
    content: str
    metadata: dict[str, str]


def _import_bs4(
    *, import_error_message: str
) -> tuple[type[BeautifulSoup], type[Tag], type[NavigableString]]:
    try:
        from bs4 import BeautifulSoup, Tag  # noqa: PLC0415
        from bs4.element import NavigableString  # noqa: PLC0415
    except ImportError as err:
        raise ImportError(import_error_message) from err
    return BeautifulSoup, Tag, NavigableString


def _import_lxml_etree() -> object:
    try:
        from lxml import etree  # noqa: PLC0415
    except ImportError as err:
        msg = "Unable to import lxml, please install with `pip install lxml`."
        raise ImportError(msg) from err
    return etree


def _import_nltk() -> object:
    try:
        import nltk  # noqa: PLC0415
    except ImportError as err:
        msg = "Could not import nltk. Please install it with 'pip install nltk'."
        raise ImportError(msg) from err
    return nltk


# 不幸的是，BeautifulSoup 未为 Tag.find_all 定义重载。
# 因此我们自行进行类型解析。


def _find_all_strings(
    tag: Tag,
    *,
    recursive: bool = True,
) -> ResultSet[NavigableString]:
    return tag.find_all(string=True, recursive=recursive)


def _find_all_tags(
    tag: Tag,
    *,
    name: bool | str | list[str] | None = None,
    recursive: bool = True,
) -> ResultSet[Tag]:
    return tag.find_all(name, recursive=recursive)


class HTMLHeaderTextSplitter:
    """根据指定的标题将 HTML 内容分割为结构化文档。

    Splits HTML content by detecting specified header tags and creating hierarchical
    `Document` objects that reflect the semantic structure of the original content. For
    each identified section, the splitter associates the extracted text with metadata
    corresponding to the encountered headers.

    If no specified headers are found, the entire content is returned as a single
    `Document`. This allows for flexible handling of HTML input, ensuring that
    information is organized according to its semantic headers.

    The splitter provides the option to return each HTML element as a separate
    `Document` or aggregate them into semantically meaningful chunks. It also
    gracefully handles multiple levels of nested headers, creating a rich,
    hierarchical representation of the content.

    Example:
        ```python
        from langchain_text_splitters.html_header_text_splitter import (
            HTMLHeaderTextSplitter,
        )

        # Define headers for splitting on h1 and h2 tags.
        headers_to_split_on = [("h1", "Main Topic"), ("h2", "Sub Topic")]

        splitter = HTMLHeaderTextSplitter(
            headers_to_split_on=headers_to_split_on,
            return_each_element=False
        )

        html_content = \"\"\"
        <html>
            <body>
                <h1>Introduction</h1>
                <p>Welcome to the introduction section.</p>
                <h2>Background</h2>
                <p>Some background details here.</p>
                <h1>Conclusion</h1>
                <p>Final thoughts.</p>
            </body>
        </html>
        \"\"\"

        documents = splitter.split_text(html_content)

        # 'documents' now contains Document objects reflecting the hierarchy:
        # - Document with metadata={"Main Topic": "Introduction"} and
        #   content="Introduction"
        # - Document with metadata={"Main Topic": "Introduction"} and
        #   content="Welcome to the introduction section."
        # - Document with metadata={"Main Topic": "Introduction",
        #   "Sub Topic": "Background"} and content="Background"
        # - Document with metadata={"Main Topic": "Introduction",
        #   "Sub Topic": "Background"} and content="Some background details here."
        # - Document with metadata={"Main Topic": "Conclusion"} and
        #   content="Conclusion"
        # - Document with metadata={"Main Topic": "Conclusion"} and
        #   content="Final thoughts."
        ```
    """

    def __init__(
        self,
        headers_to_split_on: list[tuple[str, str]],
        return_each_element: bool = False,  # noqa: FBT001,FBT002
    ) -> None:
        """使用要分割的标题初始化。

        Args:
            headers_to_split_on: 一个 `(header_tag, header_name)`
            对列表，表示定义分割边界的标题。

                例如，`[("h1", "Header 1"), ("h2", "Header 2")]` 将按 `h1` 和 `h2`
                标签分割 内容，并将其文本内容分配给 `Document` 元数据。
                return_each_element: 若为 `True`，则每个遇到的 HTML
                元素（包括标题、段落等）都作为单独的 `Document` 返回。

                若为 `False`，则相同标题层级下的内容将被聚合为更少的 `Document` 对象。
        """
                # 按数字层级对标题排序，使得 h1 < h2 < h3...
        self.headers_to_split_on = sorted(
            headers_to_split_on, key=lambda x: int(x[0][1:])
        )
        self.header_mapping = dict(self.headers_to_split_on)
        self.header_tags = [tag for tag, _ in self.headers_to_split_on]
        self.return_each_element = return_each_element

    def split_text(self, text: str) -> list[Document]:
        """将给定文本分割为 `Document` 对象列表。

        Args:
            text: 要分割的 HTML 文本。

        Returns:
            一个分割后的 `Document` 对象列表。

                每个 `Document` 包含 `page_content`，其中保存提取的文本，
                以及 `metadata`，映射标题层级与其对应标题。
        """
        return self.split_text_from_file(StringIO(text))

    @deprecated(
        since="1.1.2",
        removal="2.0.0",
        addendum=(
            "Fetch the HTML content from the URL yourself and pass it to `split_text`."
        ),
    )
    def split_text_from_url(
        self,
        url: str,
        timeout: int = 10,
        **kwargs: Any,  # noqa: ARG002
    ) -> list[Document]:
        """从 URL 获取文本内容并将其拆分为文档。

        Args:
            url: 要获取内容的 URL。
            timeout: 请求的超时时间。
            **kwargs: 请求的额外关键字参数。

        Returns:
            拆分后的 `Document` 对象列表。

                每个 `Document` 包含 `page_content`，其中保存提取的文本，
                以及 `metadata`，用于映射标题层级与其对应标题。

        Raises:
            requests.RequestException: 如果 HTTP 请求失败。
        """
        from langchain_core._security._transport import (  # noqa: PLC0415
            ssrf_safe_client,
        )

        with ssrf_safe_client() as client:
            response = client.get(url, timeout=timeout)
            response.raise_for_status()
            return self.split_text(response.text)

    def split_text_from_file(self, file: str | IO[str]) -> list[Document]:
        """将文件中的 HTML 内容拆分为 `Document` 对象列表。

        Args:
            file: 包含 HTML 内容的文件路径或文件类对象。

        Returns:
            拆分后的 `Document` 对象列表。

                每个 `Document` 包含 `page_content`，其中保存提取的文本，
                以及 `metadata`，用于映射标题层级与其对应标题。
        """
        if isinstance(file, str):
            html_content = pathlib.Path(file).read_text(encoding="utf-8")
        else:
            html_content = file.read()
        return list(self._generate_documents(html_content))

    def _generate_documents(self, html_content: str) -> Iterator[Document]:
        """私有方法，对 DOM 执行深度优先遍历并逐个生成。

        Document objects on-the-fly. This approach maintains the same splitting logic
        (headers vs. non-headers, chunking, etc.) while walking the DOM explicitly in
        code.

        Args:
            html_content: 原始 HTML 内容。

        Yields:
            生成的 `Document` 对象。

        Raises:
            ImportError: 如果未安装 BeautifulSoup。
        """
        beautiful_soup, tag_cls, _ = _import_bs4(
            import_error_message=(
                "Unable to import BeautifulSoup. Please install via `pip install bs4`."
            )
        )

        soup = beautiful_soup(html_content, "html.parser")
        body = soup.body or soup

                # 活动标题的字典：
                #   键 = 用户定义的标题名称（例如 "Header 1"）
                #   值 = 标题文本、层级、DOM 深度元组
        active_headers: dict[str, tuple[str, int, int]] = {}
        current_chunk: list[str] = []

        def finalize_chunk() -> Document | None:
            """将累积的块最终合并为一个 `Document`。"""
            if not current_chunk:
                return None

            final_text = "  \n".join(line for line in current_chunk if line.strip())
            current_chunk.clear()
            if not final_text.strip():
                return None

            final_meta = {k: v[0] for k, v in active_headers.items()}
            return Document(page_content=final_text, metadata=final_meta)

                # 我们将使用栈进行深度优先遍历
        stack = [body]
        while stack:
            node = stack.pop()
            children = list(node.children)

            stack.extend(
                child for child in reversed(children) if isinstance(child, tag_cls)
            )

            tag = getattr(node, "name", None)
            if not tag:
                continue

            text_elements = [
                str(child).strip() for child in _find_all_strings(node, recursive=False)
            ]
            node_text = " ".join(elem for elem in text_elements if elem)
            if not node_text:
                continue

            dom_depth = len(list(node.parents))

                        # 如果此节点是我们的标题之一
            if tag in self.header_tags:
                                # 如果我们在聚合，就结束当前累积的块
                if not self.return_each_element:
                    doc = finalize_chunk()
                    if doc:
                        yield doc

                                # 确定数字层级（h1->1, h2->2 等）
                try:
                    level = int(tag[1:])
                except ValueError:
                    level = 9999

                                # 移除当前层级或更深的所有活动标题
                headers_to_remove = [
                    k for k, (_, lvl, d) in active_headers.items() if lvl >= level
                ]
                for key in headers_to_remove:
                    del active_headers[key]

                                # 添加/更新活动标题
                header_name = self.header_mapping[tag]
                active_headers[header_name] = (node_text, level, dom_depth)

                                # 始终为标题生成一个 `Document`
                header_meta = {k: v[0] for k, v in active_headers.items()}
                yield Document(page_content=node_text, metadata=header_meta)

            else:
                headers_out_of_scope = [
                    k for k, (_, _, d) in active_headers.items() if dom_depth < d
                ]
                for key in headers_out_of_scope:
                    del active_headers[key]

                if self.return_each_element:
                                        # 为每个元素的文本生成独立的 `Document`
                    meta = {k: v[0] for k, v in active_headers.items()}
                    yield Document(page_content=node_text, metadata=meta)
                else:
                                        # 在我们的块中累积文本
                    current_chunk.append(node_text)

                # 如果我们在聚合且仍有剩余块，则输出它
        if not self.return_each_element:
            doc = finalize_chunk()
            if doc:
                yield doc


class HTMLSectionSplitter:
    """基于指定标签和字体大小分割 HTML 文件。

    需要 lxml 包。
    """

    def __init__(
        self,
        headers_to_split_on: list[tuple[str, str]],
        **kwargs: Any,
    ) -> None:
        """创建一个新的 `HTMLSectionSplitter`。

        Args:
            headers_to_split_on: List of tuples of headers we want to track mapped to
                (arbitrary) keys for metadata.

                Allowed header values: `h1`, `h2`, `h3`, `h4`, `h5`, `h6`, e.g.:
                `[("h1", "Header 1"), ("h2", "Header 2"]`.
            **kwargs: Additional optional arguments for customizations.

        """
        self.headers_to_split_on = dict(headers_to_split_on)
        self.xslt_path = (
            pathlib.Path(__file__).parent / "xsl/converting_to_header.xslt"
        ).absolute()
        self.kwargs = kwargs

    def split_documents(self, documents: Iterable[Document]) -> list[Document]:
        """拆分文档。

        Args:
            documents: 需要拆分的 `Document` 对象可迭代对象。

        Returns:
            拆分后的 `Document` 对象列表。
        """
        texts, metadatas = [], []
        for doc in documents:
            texts.append(doc.page_content)
            metadatas.append(doc.metadata)
        results = self.create_documents(texts, metadatas=metadatas)

        text_splitter = RecursiveCharacterTextSplitter(**self.kwargs)

        return text_splitter.split_documents(results)

    def split_text(self, text: str) -> list[Document]:
        """拆分 HTML 文本字符串。

        Args:
            text: HTML 文本

        Returns:
            拆分后的 `Document` 对象列表。
        """
        return self.split_text_from_file(StringIO(text))

    def create_documents(
        self, texts: list[str], metadatas: list[dict[Any, Any]] | None = None
    ) -> list[Document]:
        """从文本列表创建 `Document` 对象列表。

        Args:
            texts: 需要拆分并转换为文档的文本列表。
            metadatas: 可选的元数据列表，用于关联每个文档。

        Returns:
            `Document` 对象列表。
        """
        metadatas_ = metadatas or [{}] * len(texts)
        documents = []
        for i, text in enumerate(texts):
            for chunk in self.split_text(text):
                metadata = copy.deepcopy(metadatas_[i])

                for key in chunk.metadata:
                    if chunk.metadata[key] == "#TITLE#":
                        chunk.metadata[key] = metadata["Title"]
                metadata = {**metadata, **chunk.metadata}
                new_doc = Document(page_content=chunk.page_content, metadata=metadata)
                documents.append(new_doc)
        return documents

    def split_html_by_headers(self, html_doc: str) -> list[dict[str, str | None]]:
        """根据指定的标题标签将 HTML 文档拆分为多个部分。

        此方法使用 BeautifulSoup 解析 HTML 内容，并根据 `headers_to_split_on`
        中定义的标题将其划分为
        多个部分。每个部分包含标题文本、标题下的内容以及标签名称。

        Args:
            html_doc: 需要拆分为部分的 HTML 文档。

        Returns:
            表示各部分的字典列表。

                每个字典包含：

                * `'header'`: 标题文本或第一个部分的默认标题。
                * `'content'`: 标题下的内容。
                * `'tag_name'`: 标题标签名称（如 `h1`、`h2`）。

        Raises:
            ImportError: 如果未安装 BeautifulSoup。
        """
        beautiful_soup, _, _ = _import_bs4(
            import_error_message=(
                "Unable to import BeautifulSoup/PageElement, "
                "please install with `pip install bs4`."
            )
        )

        soup = beautiful_soup(html_doc, "html.parser")
        header_names = list(self.headers_to_split_on.keys())
        sections: list[dict[str, str | None]] = []

        headers = _find_all_tags(soup, name=["body", *header_names])

        for i, header in enumerate(headers):
            if i == 0:
                current_header = "#TITLE#"
                current_header_tag = "h1"
                section_content: list[str] = []
            else:
                current_header = header.text.strip()
                current_header_tag = header.name
                section_content = []
            for element in header.next_elements:
                if i + 1 < len(headers) and element == headers[i + 1]:
                    break
                if isinstance(element, str):
                    section_content.append(element)
            content = " ".join(section_content).strip()

            if content:
                sections.append(
                    {
                        "header": current_header,
                        "content": content,
                        "tag_name": current_header_tag,
                    }
                )

        return sections

    def convert_possible_tags_to_header(self, html_content: str) -> str:
        """使用 XSLT 转换将特定的 HTML 标签转换为标题。

        This method uses an XSLT file to transform the HTML content, converting
        certain tags into headers for easier parsing. If no XSLT path is provided,
        the HTML content is returned unchanged.

        Args:
            html_content: 需要转换的 HTML 内容。

        Returns:
            转换后的 HTML 内容，以字符串形式返回。

        Raises:
            ImportError: 若未安装 `lxml` 库。
        """
        etree = cast("Any", _import_lxml_etree())
                # 使用 lxml 库解析 HTML 文档并返回 XML ElementTree
                # 创建安全的解析器以防止 XXE 攻击
        html_parser = etree.HTMLParser(no_network=True)
        xslt_parser = etree.XMLParser(
            resolve_entities=False, no_network=True, load_dtd=False
        )

                # 应用 XSLT 访问控制以防止文件/网络访问
                # DENY_ALL 是一个预定义的访问控制，用于阻止所有文件/网络访问
        ac = etree.XSLTAccessControl.DENY_ALL

        tree = etree.parse(StringIO(html_content), html_parser)
        xslt_tree = etree.parse(self.xslt_path, xslt_parser)
        transform = etree.XSLT(xslt_tree, access_control=ac)
        result = transform(tree)
        return str(result)

    def split_text_from_file(self, file: StringIO) -> list[Document]:
        """将文件中的 HTML 内容拆分为 `Document` 对象列表。

        Args:
            file: 包含 HTML 内容的文件路径或文件类对象。

        Returns:
            拆分后的 `Document` 对象列表。
        """
        file_content = file.getvalue()
        file_content = self.convert_possible_tags_to_header(file_content)
        sections = self.split_html_by_headers(file_content)

        return [
            Document(
                cast("str", section["content"]),
                metadata={
                    self.headers_to_split_on[str(section["tag_name"])]: section[
                        "header"
                    ]
                },
            )
            for section in sections
        ]


@beta()
class HTMLSemanticPreservingSplitter(BaseDocumentTransformer):
    """分割 HTML 内容并保留语义结构。

    Splits HTML content by headers into generalized chunks, preserving semantic
    structure. If chunks exceed the maximum chunk size, it uses
    `RecursiveCharacterTextSplitter` for further splitting.

    分割器保留完整的 HTML 元素，并将链接转换为类似 Markdown 的链接格式。
    它还可以通过转换为 Markdown
    格式来保留图片、视频和音频元素。请注意，某些块可能超过最大大小以保持语义完整性。

    !!! version-added "在 `langchain-text-splitters` 0.3.5 中新增"

    Example:
        ```python
        from langchain_text_splitters.html import HTMLSemanticPreservingSplitter

        def custom_iframe_extractor(iframe_tag):
            ```
            自定义处理函数，用于从 <iframe> 标签中提取 'src' 属性。
            将 iframe 转换为类似 Markdown 的链接：[iframe:<src>](src)。

            Args:
                iframe_tag (bs4.element.Tag): 待处理的 <iframe> 标签。

            Returns:
                str: 以 Markdown 格式表示的 iframe 字符串。
            ```
            iframe_src = iframe_tag.get('src', '')
            return f"[iframe:{iframe_src}]({iframe_src})"

        text_splitter = HTMLSemanticPreservingSplitter(
            headers_to_split_on=[("h1", "Header 1"), ("h2", "Header 2")],
            max_chunk_size=500,
            preserve_links=True,
            preserve_images=True,
            custom_handlers={"iframe": custom_iframe_extractor}
        )
        ```
    """  # noqa: D214

    def __init__(
        self,
        headers_to_split_on: list[tuple[str, str]],
        *,
        max_chunk_size: int = 1000,
        chunk_overlap: int = 0,
        separators: list[str] | None = None,
        elements_to_preserve: list[str] | None = None,
        preserve_links: bool = False,
        preserve_images: bool = False,
        preserve_videos: bool = False,
        preserve_audio: bool = False,
        custom_handlers: dict[str, Callable[[Tag], str]] | None = None,
        stopword_removal: bool = False,
        stopword_lang: str = "english",
        normalize_text: bool = False,
        external_metadata: dict[str, str] | None = None,
        allowlist_tags: list[str] | None = None,
        denylist_tags: list[str] | None = None,
        preserve_parent_metadata: bool = False,
        keep_separator: bool | Literal["start", "end"] = True,
    ) -> None:
        """初始化分割器。

        Args:
            headers_to_split_on: HTML headers (e.g., `h1`, `h2`) that define content
                sections.
            max_chunk_size: Maximum size for each chunk, with allowance for exceeding
                this limit to preserve semantics.
            chunk_overlap: Number of characters to overlap between chunks to ensure
                contextual continuity.
            separators: Delimiters used by `RecursiveCharacterTextSplitter` for
                further splitting.
            elements_to_preserve: HTML tags (e.g., `table`, `ul`) to remain
                intact during splitting.
            preserve_links: Converts `a` tags to Markdown links (`[text](url)`).
            preserve_images: Converts `img` tags to Markdown images (`![alt](src)`).
            preserve_videos: Converts `video` tags to Markdown video links
                (`![video](src)`).
            preserve_audio: Converts `audio` tags to Markdown audio links
                (`![audio](src)`).
            custom_handlers: Optional custom handlers for specific HTML tags, allowing
                tailored extraction or processing.
            stopword_removal: Optionally remove stopwords from the text.
            stopword_lang: The language of stopwords to remove.
            normalize_text: Optionally normalize text (e.g., lowercasing, removing
                punctuation).
            external_metadata: Additional metadata to attach to the Document objects.
            allowlist_tags: Only these tags will be retained in the HTML.
            denylist_tags: These tags will be removed from the HTML.
            preserve_parent_metadata: Whether to pass through parent document metadata
                to split documents when calling
                `transform_documents/atransform_documents()`.
            keep_separator: Whether separators should be at the beginning of a chunk, at
                the end, or not at all.

        Raises:
            ImportError: 若未安装 BeautifulSoup 或 NLTK（当启用停用词移除时）。
        """
        _import_bs4(
            import_error_message=(
                "Could not import BeautifulSoup. "
                "Please install it with 'pip install bs4'."
            )
        )

        self._headers_to_split_on = sorted(headers_to_split_on)
        self._max_chunk_size = max_chunk_size
        self._elements_to_preserve = elements_to_preserve or []
        self._preserve_links = preserve_links
        self._preserve_images = preserve_images
        self._preserve_videos = preserve_videos
        self._preserve_audio = preserve_audio
        self._custom_handlers = custom_handlers or {}
        self._stopword_removal = stopword_removal
        self._stopword_lang = stopword_lang
        self._normalize_text = normalize_text
        self._external_metadata = external_metadata or {}
        self._allowlist_tags = allowlist_tags
        self._preserve_parent_metadata = preserve_parent_metadata
        self._keep_separator = keep_separator
        if allowlist_tags:
            self._allowlist_tags = list(
                set(allowlist_tags + [header[0] for header in headers_to_split_on])
            )
        self._denylist_tags = denylist_tags
        if denylist_tags:
            self._denylist_tags = [
                tag
                for tag in denylist_tags
                if tag not in [header[0] for header in headers_to_split_on]
            ]
        if separators:
            self._recursive_splitter = RecursiveCharacterTextSplitter(
                separators=separators,
                keep_separator=keep_separator,
                chunk_size=max_chunk_size,
                chunk_overlap=chunk_overlap,
            )
        else:
            self._recursive_splitter = RecursiveCharacterTextSplitter(
                keep_separator=keep_separator,
                chunk_size=max_chunk_size,
                chunk_overlap=chunk_overlap,
            )

        if self._stopword_removal:
            nltk = cast("Any", _import_nltk())
            nltk.download("stopwords")
            self._stopwords = set(nltk.corpus.stopwords.words(self._stopword_lang))

    def split_text(self, text: str) -> list[Document]:
        """根据配置将提供的 HTML 文本拆分为较小的块。

        Args:
            text: 需要拆分的 HTML 内容。

        Returns:
            包含拆分后内容的 `Document` 对象列表。
        """
        beautiful_soup, _, _ = _import_bs4(
            import_error_message=(
                "Could not import BeautifulSoup. "
                "Please install it with 'pip install bs4'."
            )
        )
        soup = beautiful_soup(text, "html.parser")

        self._process_media(soup)

        if self._preserve_links:
            self._process_links(soup)

        if self._allowlist_tags or self._denylist_tags:
            self._filter_tags(soup)

        return self._process_html(soup)

    @override
    def transform_documents(
        self, documents: Sequence[Document], **kwargs: Any
    ) -> list[Document]:
        """通过拆分文档来转换文档序列。

        Args:
            documents: 需要拆分的 `Document` 对象序列。

        Returns:
            一组分割后的 `Document` 对象。
        """
        transformed = []
        for doc in documents:
            splits = self.split_text(doc.page_content)
            if self._preserve_parent_metadata:
                splits = [
                    Document(
                        page_content=split_doc.page_content,
                        metadata={**doc.metadata, **split_doc.metadata},
                    )
                    for split_doc in splits
                ]
            transformed.extend(splits)
        return transformed

    def _process_media(self, soup: BeautifulSoup) -> None:
        """处理媒体元素。

        通过将媒体元素包裹在 <media-wrapper> 标签中，并将其转换为 Markdown 格式，来处理
        HTML 内容中的元素。

        Args:
            soup: 使用 BeautifulSoup 解析的 HTML 内容。
        """
        if self._preserve_images:
            for img_tag in _find_all_tags(soup, name="img"):
                img_src = img_tag.get("src", "")
                markdown_img = f"![image:{img_src}]({img_src})"
                wrapper = soup.new_tag("media-wrapper")
                wrapper.string = markdown_img
                img_tag.replace_with(wrapper)

        if self._preserve_videos:
            for video_tag in _find_all_tags(soup, name="video"):
                video_src = video_tag.get("src", "")
                markdown_video = f"![video:{video_src}]({video_src})"
                wrapper = soup.new_tag("media-wrapper")
                wrapper.string = markdown_video
                video_tag.replace_with(wrapper)

        if self._preserve_audio:
            for audio_tag in _find_all_tags(soup, name="audio"):
                audio_src = audio_tag.get("src", "")
                markdown_audio = f"![audio:{audio_src}]({audio_src})"
                wrapper = soup.new_tag("media-wrapper")
                wrapper.string = markdown_audio
                audio_tag.replace_with(wrapper)

    @staticmethod
    def _process_links(soup: BeautifulSoup) -> None:
        """处理 HTML 内容中的链接。

        Args:
            soup: 使用 BeautifulSoup 解析的 HTML 内容。
        """
        _, _, navigable_string = _import_bs4(
            import_error_message=(
                "Could not import BeautifulSoup. "
                "Please install it with 'pip install bs4'."
            )
        )
        for a_tag in _find_all_tags(soup, name="a"):
            a_href = a_tag.get("href", "")
            a_text = a_tag.get_text(strip=True)
            markdown_link = f"[{a_text}]({a_href})"
            wrapper = soup.new_tag("link-wrapper")
            wrapper.string = markdown_link
            a_tag.replace_with(navigable_string(markdown_link))

    def _filter_tags(self, soup: BeautifulSoup) -> None:
        """根据允许列表和拒绝列表标签过滤 HTML 内容。

        Args:
            soup: 使用 BeautifulSoup 解析的 HTML 内容。
        """
        if self._allowlist_tags:
            for tag in _find_all_tags(soup, name=True):
                if tag.name not in self._allowlist_tags:
                    tag.decompose()

        if self._denylist_tags:
            for tag in _find_all_tags(soup, name=self._denylist_tags):
                tag.decompose()

    def _normalize_and_clean_text(self, text: str) -> str:
        """通过移除多余的空格和换行符来标准化文本。

        Args:
            text: 需要标准化的文本。

        Returns:
            标准化后的文本。
        """
        if self._normalize_text:
            text = text.lower()
            text = re.sub(r"[^\w\s]", "", text)
            text = re.sub(r"\s+", " ", text).strip()

        if self._stopword_removal:
            text = " ".join(
                [word for word in text.split() if word not in self._stopwords]
            )

        return text

    def _process_html(self, soup: BeautifulSoup) -> list[Document]:
        """使用 BeautifulSoup 处理 HTML 内容，并按标题进行分割。

        Args:
            soup: 使用 BeautifulSoup 解析的 HTML 内容。

        Returns:
            包含分割后内容的一组 `Document` 对象。
        """
        _, tag_cls, _ = _import_bs4(
            import_error_message=(
                "Could not import BeautifulSoup. "
                "Please install it with 'pip install bs4'."
            )
        )
        documents: list[Document] = []
        current_headers: dict[str, str] = {}
        current_content: list[str] = []
        preserved_elements: dict[str, str] = {}
        placeholder_count: int = 0

        def _get_element_text(element: PageElement) -> str:
            """递归提取并处理元素的文本。

            应用适用的自定义处理器，并确保正确的间距。

            Args:
                element: 要处理的 HTML 元素。

            Returns:
                The processed text of the element, or an empty string for
                    elements with no extractable text.
            """
            if isinstance(element, tag_cls):
                if element.name in self._custom_handlers:
                    return self._custom_handlers[element.name](element)

                text = ""
                for child in element.children:
                    child_text = _get_element_text(child).strip()
                    if text and child_text:
                        text += " "
                    text += child_text

                return self._normalize_and_clean_text(text)

            if hasattr(element, "string") and isinstance(element.string, str):
                return self._normalize_and_clean_text(element.string)

            return ""

        elements = _find_all_tags(soup, recursive=False)

        def _process_element(
            element: ResultSet[Tag],
            documents: list[Document],
            current_headers: dict[str, str],
            current_content: list[str],
            preserved_elements: dict[str, str],
            placeholder_count: int,
        ) -> tuple[list[Document], dict[str, str], list[str], dict[str, str], int]:
            for elem in element:
                if elem.name in [h[0] for h in self._headers_to_split_on]:
                    if current_content:
                        documents.extend(
                            self._create_documents(
                                current_headers,
                                " ".join(current_content),
                                preserved_elements,
                            )
                        )
                        current_content.clear()
                        preserved_elements.clear()
                    header_name = elem.get_text(strip=True)
                    current_headers = {
                        dict(self._headers_to_split_on)[elem.name]: header_name
                    }
                elif elem.name in self._elements_to_preserve:
                    placeholder = f"PRESERVED_{placeholder_count}"
                    preserved_elements[placeholder] = _get_element_text(elem)
                    current_content.append(placeholder)
                    placeholder_count += 1
                else:
                                        # 递归处理子元素以查找嵌套的标题或
                                        # 保留的元素。
                    children = _find_all_tags(elem, recursive=False)
                    if children:
                                                # 元素有子元素 - 递归处理它们。
                        (
                            documents,
                            current_headers,
                            current_content,
                            preserved_elements,
                            placeholder_count,
                        ) = _process_element(
                            children,
                            documents,
                            current_headers,
                            current_content,
                            preserved_elements,
                            placeholder_count,
                        )
                                                # 在处理子元素后，仅从此元素提取文本
                                                # 字符串（不包括其子元素）。使用
                                                # recursive=False 以避免重复计数。
                        content = " ".join(_find_all_strings(elem, recursive=False))
                        if content:
                            content = self._normalize_and_clean_text(content)
                            current_content.append(content)
                    else:
                                                # 无子元素的叶节点，因此我们提取其
                                                # 文本并添加到当前内容中。处理
                                                # 仅包含文本的元素，如 <p>、<span>、<div>
                        content = _get_element_text(elem)
                        if content:
                            current_content.append(content)

            return (
                documents,
                current_headers,
                current_content,
                preserved_elements,
                placeholder_count,
            )

                # 处理元素
        (
            documents,
            current_headers,
            current_content,
            preserved_elements,
            placeholder_count,
        ) = _process_element(
            elements,
            documents,
            current_headers,
            current_content,
            preserved_elements,
            placeholder_count,
        )

                # 处理任何剩余内容
        if current_content:
            documents.extend(
                self._create_documents(
                    current_headers,
                    " ".join(current_content),
                    preserved_elements,
                )
            )

        return documents

    def _create_documents(
        self, headers: dict[str, str], content: str, preserved_elements: dict[str, str]
    ) -> list[Document]:
        """从提供的标题、内容和元素创建 `Document` 对象。

        Args:
            headers: 要作为元数据附加到 `Document` 的标题。
            content: `Document` 的内容。
            preserved_elements: 保留的元素，需重新插入内容中。

        Returns:
            一组 `Document` 对象。
        """
        content = re.sub(r"\s+", " ", content).strip()

        metadata = {**headers, **self._external_metadata}

        if len(content) <= self._max_chunk_size:
            page_content = self._reinsert_preserved_elements(
                content, preserved_elements
            )
            return [Document(page_content=page_content, metadata=metadata)]
        return self._further_split_chunk(content, metadata, preserved_elements)

    def _further_split_chunk(
        self, content: str, metadata: dict[Any, Any], preserved_elements: dict[str, str]
    ) -> list[Document]:
        """进一步将内容分割为更小的块。

        Args:
            content: 需要分割的内容。
            metadata: 要附加到每个块的元数据。
            preserved_elements: 保留的元素，需重新插入每个块中。

        Returns:
            包含分割后内容的一组 `Document` 对象。
        """
        splits = self._recursive_splitter.split_text(content)
        result = []

        for split in splits:
            split_with_preserved = self._reinsert_preserved_elements(
                split, preserved_elements
            )
            if split_with_preserved.strip():
                result.append(
                    Document(
                        page_content=split_with_preserved.strip(),
                        metadata=metadata,
                    )
                )

        return result

    @staticmethod
    def _reinsert_preserved_elements(
        content: str, preserved_elements: dict[str, str]
    ) -> str:
        """将保留的元素重新插入内容中其原始位置。

        Args:
            content: 需要替换占位符的内容。
            preserved_elements: 需要重新插入的保留元素。

        Returns:
            将占位符替换为保留元素后的内容。
        """
        for placeholder, preserved_content in reversed(preserved_elements.items()):
            content = content.replace(placeholder, preserved_content.strip())
        return content


# %%

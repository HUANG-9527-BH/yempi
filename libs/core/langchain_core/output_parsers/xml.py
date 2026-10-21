"""XML 格式输出解析器。"""

import contextlib
import re
import xml
import xml.etree.ElementTree as ET
from collections.abc import AsyncIterator, Iterator
from typing import Any, Literal
from xml.etree.ElementTree import TreeBuilder

from typing_extensions import override

from langchain_core.exceptions import OutputParserException
from langchain_core.messages import BaseMessage
from langchain_core.output_parsers.transform import BaseTransformOutputParser
from langchain_core.runnables.utils import AddableDict

try:
    from defusedxml import ElementTree  # type: ignore[import-untyped]
    from defusedxml.ElementTree import XMLParser  # type: ignore[import-untyped]

    _HAS_DEFUSEDXML = True
except ImportError:
    _HAS_DEFUSEDXML = False

XML_FORMAT_INSTRUCTIONS = """The output should be formatted as a XML file.
1. Output should conform to the tags below.
2. If tags are not given, make them on your own.
3. Remember to always open and close all the tags.

As an example, for the tags ["foo", "bar", "baz"]:
1. String "<foo>\n   <bar>\n      <baz></baz>\n   </bar>\n</foo>" is a well-formatted instance of the schema.
2. String "<foo>\n   <bar>\n   </foo>" is a badly-formatted instance.
3. String "<foo>\n   <tag>\n   </tag>\n</foo>" is a badly-formatted instance.

Here are the output tags:
```
{tags}
```"""  # noqa: E501


class _StreamingParser:
    """XML 流式解析器。

    This implementation is pulled into a class to avoid implementation drift between
    `transform` and `atransform` of the `XMLOutputParser`.
    """

    def __init__(self, parser: Literal["defusedxml", "xml"]) -> None:
        """初始化流式解析器。

        Args:
            parser: 用于 XML 解析的解析器。

                Can be either `'defusedxml'` or `'xml'`. See documentation in
                `XMLOutputParser` for more information.

        Raises:
            ImportError: 如果未安装 `defusedxml` 但请求使用 `defusedxml` 解析器。
        """
        if parser == "defusedxml":
            if not _HAS_DEFUSEDXML:
                msg = (
                    "defusedxml is not installed. "
                    "Please install it to use the defusedxml parser. "
                    "You can install it with `pip install defusedxml`. "
                    "See https://github.com/tiran/defusedxml for more details"
                )
                raise ImportError(msg)
            parser_ = XMLParser(target=TreeBuilder())
        else:
            parser_ = None
        self.pull_parser = ET.XMLPullParser(["start", "end"], _parser=parser_)
        self.xml_start_re = re.compile(r"<[a-zA-Z:_]")
        self.current_path: list[str] = []
        self.current_path_has_children = False
        self.buffer = ""
        self.xml_started = False

    def parse(self, chunk: str | BaseMessage) -> Iterator[AddableDict]:
        """解析一段文本。

        Args:
            chunk: 要解析的文本块。可以是 `str` 或 `BaseMessage`。

        Yields:
            表示解析后的 XML 元素的 `dict`。

        Raises:
            xml.etree.ElementTree.ParseError: 如果 XML 格式不正确。
        """
        if isinstance(chunk, BaseMessage):
                        # extract text
            chunk_content = chunk.content
            if not isinstance(chunk_content, str):
                                # ignore non-string messages (e.g., function calls)
                return
            chunk = chunk_content
                # add chunk to buffer of unprocessed text
        self.buffer += chunk
                # if xml string hasn't started yet, continue to next chunk
        if not self.xml_started:
            if match := self.xml_start_re.search(self.buffer):
                                # if xml string has started, remove all text before it
                self.buffer = self.buffer[match.start() :]
                self.xml_started = True
            else:
                return
                # feed buffer to parser
        self.pull_parser.feed(self.buffer)
        self.buffer = ""
                # yield all events
        try:
            events = self.pull_parser.read_events()
            for event, elem in events:  # type: ignore[misc]
                if event == "start":
                                        # 更新当前路径
                    self.current_path.append(elem.tag)  # type: ignore[union-attr]
                    self.current_path_has_children = False
                elif event == "end":
                                        # 从当前路径中移除最后一个元素
                    #
                    self.current_path.pop()
                                        # 生成元素
                    if not self.current_path_has_children:
                        yield nested_element(self.current_path, elem)  # type: ignore[arg-type]
                                        # 防止生成父元素
                    if self.current_path:
                        self.current_path_has_children = True
                    else:
                        self.xml_started = False
        except xml.etree.ElementTree.ParseError:
                        # 这可能是 XML 输入末尾的垃圾数据。
                        # 让我们检查当前路径是否为空。
            if not self.current_path:
                                # 如果为空，我们可以忽略此错误。
                return
            else:
                raise

    def close(self) -> None:
        """关闭解析器。

        所有文本块解析完成后应调用此方法。
        """
                # 忽略 ParseError。这将忽略输入末尾任何不完整的 XML
        with contextlib.suppress(xml.etree.ElementTree.ParseError):
            self.pull_parser.close()


class XMLOutputParser(BaseTransformOutputParser[dict[str, Any]]):
    """使用 XML 格式解析输出。

    返回一个标签的字典。
    """

    tags: list[str] | None = None
    """用于告知 LLM 在 XML 输出中期望的标签。

    注意，这可能并不完美，具体取决于 LLM 的实现。

    例如，当 `tags=["foo", "bar", "baz"]` 时：

    1. 格式良好的 XML 实例：
        `'<foo>\n   <bar>\n      <baz></baz>\n   </bar>\n</foo>'`

    2. 格式不良的 XML 实例（缺少 'bar' 的闭合标签）：
        `'<foo>\n   <bar>\n   </foo>'`

    3. 格式不良的 XML 实例（包含意外的 'tag' 元素）：
        `'<foo>\n   <tag>\n   </tag>\n</foo>'`
    """
    encoding_matcher: re.Pattern[str] = re.compile(
        r"<([^>]*encoding[^>]*)>\n(.*)", re.MULTILINE | re.DOTALL
    )

    parser: Literal["defusedxml", "xml"] = "defusedxml"
    """用于 XML 解析的解析器。

    可以是 `'defusedxml'` 或 `'xml'`。

    - `'defusedxml'` is the default parser and is used to prevent XML vulnerabilities
        present in some distributions of Python's standard library xml. `defusedxml` is
        a wrapper around the standard library parser that sets up the parser with secure
        defaults.
    - `'xml'` is the standard library parser.

    !!! warning

        Use `xml` only if you are sure that your distribution of the standard library is
        not vulnerable to XML vulnerabilities.

    有关更多信息，请参阅以下资源：

    * https://docs.python.org/3/library/xml.html#xml-vulnerabilities
    * https://github.com/tiran/defusedxml

    The standard library relies on [`libexpat`](https://github.com/libexpat/libexpat)
    for parsing XML.
    """

    def get_format_instructions(self) -> str:
        """返回 XML 输出的格式指令。"""
        return XML_FORMAT_INSTRUCTIONS.format(tags=self.tags)

    def parse(self, text: str) -> dict[str, str | list[Any]]:
        """解析 LLM 调用的输出。

        Args:
            text: LLM 调用的输出。

        Returns:
            表示解析后 XML 的 `dict`。

        Raises:
            OutputParserException: 如果 XML 格式不正确。 ImportError: 如果未安装
            `defusedxml` 但请求使用 `defusedxml` 解析器。
        """
                # 尝试在三个反引号内找到 XML 字符串
                # 导入暂时放在这里以避免 CI 上缓存问题
                # 如果你正在阅读这段内容，可以将其移至文件顶部
        if self.parser == "defusedxml":
            if not _HAS_DEFUSEDXML:
                msg = (
                    "defusedxml is not installed. "
                    "Please install it to use the defusedxml parser. "
                    "You can install it with `pip install defusedxml`. "
                    "See https://github.com/tiran/defusedxml for more details"
                )
                raise ImportError(msg)
            et = ElementTree  # 使用 defusedxml 解析器
        else:
            et = ET  # 使用标准库解析器

        match = re.search(r"```(xml)?(.*)```", text, re.DOTALL)
        if match is not None:
                        # 如果找到匹配项，使用反引号内的内容
            text = match.group(2)
        encoding_match = self.encoding_matcher.search(text)
        if encoding_match:
            text = encoding_match.group(2)

        text = text.strip()
        try:
            root = et.fromstring(text)
            return self._root_to_dict(root)
        except et.ParseError as e:
            msg = f"Failed to parse XML format from completion {text}. Got: {e}"
            raise OutputParserException(msg, llm_output=text) from e

    @override
    def _transform(self, input: Iterator[str | BaseMessage]) -> Iterator[AddableDict]:
        streaming_parser = _StreamingParser(self.parser)
        for chunk in input:
            yield from streaming_parser.parse(chunk)
        streaming_parser.close()

    @override
    async def _atransform(
        self, input: AsyncIterator[str | BaseMessage]
    ) -> AsyncIterator[AddableDict]:
        streaming_parser = _StreamingParser(self.parser)
        async for chunk in input:
            for output in streaming_parser.parse(chunk):
                yield output
        streaming_parser.close()

    def _root_to_dict(self, root: ET.Element) -> dict[str, str | list[Any]]:
        """将 XML 树转换为 Python 字典。"""
        if root.text and bool(re.search(r"\S", root.text)):
                        # 如果根文本包含任何非空白字符，则
                        # 返回 {root.tag: root.text}
            return {root.tag: root.text}
        root_tag: list[Any] = []
        for child in root:
            if len(child) == 0:
                root_tag.append({child.tag: child.text})
            else:
                root_tag.append(self._root_to_dict(child))
        return {root.tag: root_tag}

    @property
    def _type(self) -> str:
        return "xml"


def nested_element(path: list[str], elem: ET.Element) -> Any:
    """从路径获取嵌套元素。

    Args:
        path: 元素的路径。
        elem: 要提取的元素。

    Returns:
        嵌套元素。
    """
    if len(path) == 0:
        return AddableDict({elem.tag: elem.text})
    return AddableDict({path[0]: [nested_element(path[1:], elem)]})

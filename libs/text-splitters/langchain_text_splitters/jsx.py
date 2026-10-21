"""JavaScript 框架文本分割器。"""

import re
from typing import Any

from typing_extensions import override

from langchain_text_splitters import RecursiveCharacterTextSplitter


class JSFrameworkTextSplitter(RecursiveCharacterTextSplitter):
    """可处理 React (JSX)、Vue 和 Svelte 代码的文本分割器。

    此分割器继承自 `RecursiveCharacterTextSplitter`，通过以下方式处理 React (JSX)、Vue,
    和 Svelte 代码：

    1. 检测并提取自定义组件标签
    2. 将这些标签作为额外分隔符，与标准 JS 语法一起使用

    该分割器结合了：

    * 自定义组件标签作为分隔符（例如 `<Component`, `<div`）
    * JavaScript 语法元素（function、const、if 等）
    * 标准换行符分割

    这使得分块可以在 React、Vue 和 Svelte 组件代码中自然边界处进行分割。
    """

    def __init__(
        self,
        separators: list[str] | None = None,
        chunk_size: int = 2000,
        chunk_overlap: int = 0,
        **kwargs: Any,
    ) -> None:
        """初始化 JavaScript 框架文本分割器。

        Args:
            separators: 可选的自定义分隔符字符串列表
            chunk_size: 返回分块的最大大小
            chunk_overlap: 分块之间的字符重叠量
            **kwargs: 传递给父类的其他参数
        """
        super().__init__(chunk_size=chunk_size, chunk_overlap=chunk_overlap, **kwargs)
        self._separators = separators or []

    @override
    def split_text(self, text: str) -> list[str]:
        """将文本分割为分块。

        此方法通过以下方式将文本分割为分块：

        * 使用正则表达式提取唯一的打开组件标签
        * 使用提取的标签和 JS 分隔符创建分隔符列表
        * 调用父类方法，使用分隔符进行文本分割

        Args:
            text: 包含待分割代码的字符串

        Returns:
            在组件和 JS 边界处分割后的文本分块列表
        """
                # 提取唯一打开组件标签的正则表达式
                # 匹配打开标签的正则表达式，不包括自闭合标签
        opening_tags = re.findall(r"<\s*([a-zA-Z0-9]+)[^>]*>", text)

        component_tags = []
        for tag in opening_tags:
            if tag not in component_tags:
                component_tags.append(tag)
        component_separators = [f"<{tag}" for tag in component_tags]

        js_separators = [
            "\nexport ",
            " export ",
            "\nfunction ",
            "\nasync function ",
            " async function ",
            "\nconst ",
            "\nlet ",
            "\nvar ",
            "\nclass ",
            " class ",
            "\nif ",
            " if ",
            "\nfor ",
            " for ",
            "\nwhile ",
            " while ",
            "\nswitch ",
            " switch ",
            "\ncase ",
            " case ",
            "\ndefault ",
            " default ",
        ]
                # 为本次调用构建有效的分隔符列表。
                # 不要将结果赋值回 self._separators：这样做会导致在每次调用时永久
                # 追加 js_separators + component_separators，
                # 当在同一个实例上多次调用 split_text() 时，导致列表无限制增长。
                # 此操作仅用于本次调用，不应修改实例状态。
        separators = (
            self._separators
            + js_separators
            + component_separators
            + ["<>", "\n\n", "&&\n", "||\n"]
        )
        return self._split_text(text, separators)

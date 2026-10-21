"""LangSmith 文档加载器。"""

import datetime
import json
import uuid
from collections.abc import Callable, Iterator, Mapping, Sequence
from typing import Any

from langsmith import Client as LangSmithClient
from typing_extensions import override

from langchain_core.document_loaders.base import BaseLoader
from langchain_core.documents import Document
from langchain_core.tracers._compat import pydantic_to_dict


class LangSmithLoader(BaseLoader):
    """将 LangSmith 数据集示例加载为 `Document` 对象。

    Loads the example inputs as the `Document` page content and places the entire
    example into the `Document` metadata. This allows you to easily create few-shot
    example retrievers from the loaded documents.

    ??? example "延迟加载"

        ```python
        from langchain_core.document_loaders import LangSmithLoader

        loader = LangSmithLoader(dataset_id="...", limit=100)
        docs = []
        for doc in loader.lazy_load():
            docs.append(doc)
        ```

        ```python
        # -> [Document("...", metadata={"inputs": {...}, "outputs": {...}, ...}), ...]
        ```
    """

    def __init__(
        self,
        *,
        dataset_id: uuid.UUID | str | None = None,
        dataset_name: str | None = None,
        example_ids: Sequence[uuid.UUID | str] | None = None,
        as_of: datetime.datetime | str | None = None,
        splits: Sequence[str] | None = None,
        inline_s3_urls: bool = True,
        offset: int = 0,
        limit: int | None = None,
        metadata: dict[str, Any] | None = None,
        filter: str | None = None,  # noqa: A002
        content_key: str = "",
        format_content: Callable[..., str] | None = None,
        client: LangSmithClient | None = None,
        **client_kwargs: Any,
    ) -> None:
        """创建一个 LangSmith 加载器。

        Args:
            dataset_id: 用于过滤的数据集 ID。
            dataset_name: 用于过滤的数据集名称。
            content_key: 用于设置为 `Document` 页面内容的输入键。

                `'.'` characters are interpreted as nested keys, e.g.
                `content_key="first.second"` will result in
                `Document(page_content=format_content(example.inputs["first"]["second"]))`
            format_content: Function for converting the content extracted from the example
                inputs into a string.

                默认为对内容进行 JSON 编码。
            example_ids: 用于过滤的示例 ID 列表。
            as_of: 要检索示例时所依据的数据集版本标签或时间戳。

                响应示例仅包含在该标签（或时间戳）版本时存在的示例。 splits:
                数据集划分列表，如 `train`、`test` 或 `validation`。

                仅返回指定划分中的示例。
            inline_s3_urls: 是否内联 S3 URL。
            offset: 开始位置的偏移量。
            limit: 最多返回的示例数量。
            metadata: 用于过滤的元数据。
            filter: 应用于示例的结构化过滤字符串。
            client: LangSmith 客户端。

                如果未提供，则将从以下参数初始化。
            client_kwargs: 传递给 LangSmith 客户端初始化的关键词参数。

                仅当未提供 `client` 时才应指定。

        Raises:
            ValueError: 如果同时提供了 `client` 和 `client_kwargs`。
        """  # noqa: E501
        if client and client_kwargs:
            msg = (
                "Received both `client` and `client_kwargs`. "
                "Pass `client_kwargs` only when `client` is not provided."
            )
            raise ValueError(msg)
        self._client = client or LangSmithClient(**client_kwargs)
        self.content_key = list(content_key.split(".")) if content_key else []
        self.format_content = format_content or _stringify
        self.dataset_id = dataset_id
        self.dataset_name = dataset_name
        self.example_ids = example_ids
        self.as_of = as_of
        self.splits = splits
        self.inline_s3_urls = inline_s3_urls
        self.offset = offset
        self.limit = limit
        self.metadata = metadata
        self.filter = filter

    @override
    def lazy_load(self) -> Iterator[Document]:
        for example in self._client.list_examples(
            dataset_id=self.dataset_id,
            dataset_name=self.dataset_name,
            example_ids=self.example_ids,
            as_of=self.as_of,
            splits=self.splits,
            inline_s3_urls=self.inline_s3_urls,
            offset=self.offset,
            limit=self.limit,
            metadata=self.metadata,
            filter=self.filter,
        ):
            content = _get_content_from_inputs(example.inputs, self.content_key)
            content_str = self.format_content(content)
            metadata = pydantic_to_dict(example)
                        # 将 datetime 和 UUID 类型转为字符串。
            for k in ("dataset_id", "created_at", "modified_at", "source_run_id", "id"):
                metadata[k] = str(metadata[k]) if metadata[k] else metadata[k]
            yield Document(content_str, metadata=metadata)


def _get_content_from_inputs(inputs: Any, content_key: Sequence[str]) -> Any:
    """为 `LangSmithLoader` 解析嵌套示例输入内容。

    Args:
        inputs: LangSmith 返回的示例输入负载。
        content_key: 用于提取文档内容的有序键路径。

    Returns:
        提取的内容值。

    Raises:
        ValueError: 如果 `content_key` 中的某个键缺失，或路径上的某个值（包括 `inputs`
        本身）不是映射类型。
    """
    content = inputs
    full_path = ".".join(content_key)

    for i, key in enumerate(content_key):
        current_path = ".".join(content_key[:i]) or "<root>"
        if not isinstance(content, Mapping):
            msg = (
                f"Could not resolve content_key {full_path!r}: expected a mapping at "
                f"{current_path!r}, but found {type(content).__name__}."
            )
                        # 过深的 `content_key`
            # 是无效参数错误，而非运行时类型错误，因此与缺失键情况统一为 `ValueError`。
                        # 已弃用
            raise ValueError(msg)  # noqa: TRY004
        if key not in content:
            msg = (
                f"Could not resolve content_key {full_path!r}: missing key {key!r} "
                f"under {current_path!r}."
            )
            raise ValueError(msg)
        content = content[key]

    return content


def _stringify(x: str | dict[str, Any]) -> str:
    if isinstance(x, str):
        return x
    try:
        return json.dumps(x, indent=2)
    except Exception:
        return str(x)

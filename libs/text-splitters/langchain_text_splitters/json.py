"""JSON 文本分割器。"""

from __future__ import annotations

import copy
import json
from typing import Any

from langchain_core.documents import Document


class RecursiveJsonSplitter:
    """将 JSON 数据分割为更小、结构化的块，同时保留层级关系。

    This class provides methods to split JSON data into smaller dictionaries or
    JSON-formatted strings based on configurable maximum and minimum chunk sizes.
    It supports nested JSON structures, optionally converts lists into dictionaries
    for better chunking, and allows the creation of document objects for further use.
    """

    max_chunk_size: int = 2000
    """每个块的最大大小。"""

    min_chunk_size: int = 1800
    """The minimum size for each chunk, derived from `max_chunk_size` if not
    explicitly provided.
    """

    def __init__(
        self, max_chunk_size: int = 2000, min_chunk_size: int | None = None
    ) -> None:
        """初始化文本处理的块大小配置。

        This constructor sets up the maximum and minimum chunk sizes, ensuring that
        the `min_chunk_size` defaults to a value slightly smaller than the
        `max_chunk_size` if not explicitly provided.

        Args:
            max_chunk_size: 块的最大大小。
            min_chunk_size: 块的最小大小。

                If `None`, defaults to the maximum chunk size minus 200, with a lower
                bound of 50.
        """
        super().__init__()
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = (
            min_chunk_size
            if min_chunk_size is not None
            else max(max_chunk_size - 200, 50)
        )

    @staticmethod
    def _json_size(data: dict[str, Any]) -> int:
        """计算序列化 JSON 对象的大小。"""
        return len(json.dumps(data))

    @staticmethod
    def _set_nested_dict(
        d: dict[str, Any],
        path: list[str],
        value: Any,  # noqa: ANN401
    ) -> None:
        """根据给定路径在嵌套字典中设置一个值。"""
        for key in path[:-1]:
            d = d.setdefault(key, {})
        d[path[-1]] = value

    def _list_to_dict_preprocessing(
        self,
        data: Any,  # noqa: ANN401
    ) -> Any:  # noqa: ANN401
        if isinstance(data, dict):
                        # 处理字典中的每个键值对
            return {k: self._list_to_dict_preprocessing(v) for k, v in data.items()}
        if isinstance(data, list):
                        # 将列表转换为基于索引的键的字典
            return {
                str(i): self._list_to_dict_preprocessing(item)
                for i, item in enumerate(data)
            }
                # 基础情况：该项既不是字典也不是列表，因此原样返回
        return data

    def _json_split(
        self,
        data: Any,  # noqa: ANN401
        current_path: list[str] | None = None,
        chunks: list[dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        """将 JSON 分割为最大大小的字典，同时保持结构。"""
        current_path = current_path or []
        chunks = chunks if chunks is not None else [{}]
        if isinstance(data, dict) and data:
            for key, value in data.items():
                new_path = [*current_path, key]
                chunk_size = self._json_size(chunks[-1])
                size = self._json_size({key: value})
                remaining = self.max_chunk_size - chunk_size

                if size < remaining:
                                        # 将项添加到当前块
                    self._set_nested_dict(chunks[-1], new_path, value)
                else:
                    if chunk_size >= self.min_chunk_size:
                                                # 块已足够大，开始新块
                        chunks.append({})

                                        # 迭代
                    self._json_split(value, new_path, chunks)
                # 处理叶值和空字典
        elif current_path:
            self._set_nested_dict(chunks[-1], current_path, data)
        return chunks

    def split_json(
        self,
        json_data: dict[str, Any],
        convert_lists: bool = False,  # noqa: FBT001,FBT002
    ) -> list[dict[str, Any]]:
        """将 JSON 分割为 JSON 块列表。

        Args:
            json_data: 需要分割的 JSON 数据。 convert_lists: 是否在分割前将 JSON
            中的列表转换为字典。

        Returns:
            JSON 块的列表。

        Raises:
            TypeError: 如果 `json_data` 不是字典且无法转换为字典。`None`
            返回空列表而非抛出异常。当 `convert_lists` 为 `True` 时，才接受顶层列表。
        """
        is_list_input = isinstance(json_data, list)

        if convert_lists:
            json_data = self._list_to_dict_preprocessing(json_data)

        if json_data is not None and not isinstance(json_data, dict):
            msg = f"json_data must be a dict, got {type(json_data).__name__}."
            if is_list_input and not convert_lists:
                msg += " Top-level lists can be split by passing convert_lists=True."
            raise TypeError(msg)

        chunks = self._json_split(json_data)

                # 如果最后一个块为空，则移除
        if not chunks[-1]:
            chunks.pop()
        return chunks

    def split_text(
        self,
        json_data: dict[str, Any],
        convert_lists: bool = False,  # noqa: FBT001,FBT002
        ensure_ascii: bool = True,  # noqa: FBT001,FBT002
    ) -> list[str]:
        """将 JSON 分割为 JSON 格式字符串的列表。

        Args:
            json_data: 需要分割的 JSON 数据。 convert_lists: 是否在分割前将 JSON
            中的列表转换为字典。 ensure_ascii: 是否确保 JSON 字符串使用 ASCII 编码。

        Returns:
            JSON 格式字符串的列表。
        """
        chunks = self.split_json(json_data=json_data, convert_lists=convert_lists)

                # 转换为字符串
        return [json.dumps(chunk, ensure_ascii=ensure_ascii) for chunk in chunks]

    def create_documents(
        self,
        texts: list[dict[str, Any]],
        convert_lists: bool = False,  # noqa: FBT001,FBT002
        ensure_ascii: bool = True,  # noqa: FBT001,FBT002
        metadatas: list[dict[Any, Any]] | None = None,
    ) -> list[Document]:
        """从 JSON 对象列表（`dict`）创建 `Document` 对象列表。

        Args:
            texts: 需要分割并转换为文档的 JSON 数据列表。
            convert_lists: 是否在分割前将列表转换为字典。
            ensure_ascii: 是否确保 JSON 字符串使用 ASCII 编码。
            metadatas: 可选的元数据列表，用于关联每个文档。

        Returns:
            `Document` 对象的列表。
        """
        metadatas_ = metadatas or [{}] * len(texts)
        documents = []
        for i, text in enumerate(texts):
            for chunk in self.split_text(
                json_data=text, convert_lists=convert_lists, ensure_ascii=ensure_ascii
            ):
                metadata = copy.deepcopy(metadatas_[i])
                new_doc = Document(page_content=chunk, metadata=metadata)
                documents.append(new_doc)
        return documents

"""Fireworks 文档重排序集成。"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import Any

from langchain_core._api import beta
from langchain_core.callbacks import Callbacks
from langchain_core.documents import BaseDocumentCompressor, Document
from langchain_core.utils import secret_from_env
from openai import AsyncOpenAI, OpenAI
from pydantic import ConfigDict, Field, SecretStr, model_validator
from typing_extensions import override

# OpenAI SDK 会解包 `get_args()` 上的 `dict` `cast_to`，因此一个裸 `dict`
# 在解析响应时会引发 `ValueError`。请保持该参数化。
_RESPONSE_TYPE = dict[str, Any]


@beta()
class FireworksRerank(BaseDocumentCompressor):
    """使用 Fireworks 重排序 API 的文档压缩器。"""

    client: Any = None
    """用于调用 Fireworks 的 OpenAI 兼容客户端。"""

    async_client: Any = None
    """用于调用 Fireworks 的异步 OpenAI 兼容客户端。"""

    top_n: int | None = 3
    """要返回的文档数量。"""

    model: str
    """要使用的 Fireworks 重排序模型。"""

    fireworks_api_key: SecretStr | None = Field(
        default_factory=secret_from_env("FIREWORKS_API_KEY", default=None)
    )
    """Fireworks API 密钥。"""

    base_url: str = "https://api.fireworks.ai/inference/v1"
    """Fireworks API 的基础 URL。"""

    user_agent: str = "langchain:partner"
    """请求发起应用的标识符。"""

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="forbid",
    )

    @model_validator(mode="after")
    def validate_environment(self) -> FireworksRerank:
        """创建未提供的 OpenAI 兼容客户端。"""
        if self.client is None or self.async_client is None:
            if self.fireworks_api_key is None:
                msg = (
                    "FIREWORKS_API_KEY is required unless both client and "
                    "async_client are supplied."
                )
                raise ValueError(msg)
            client_kwargs: dict[str, Any] = {
                "api_key": self.fireworks_api_key.get_secret_value(),
                "base_url": self.base_url,
                "default_headers": {"User-Agent": self.user_agent},
            }
            if self.client is None:
                self.client = OpenAI(**client_kwargs)
            if self.async_client is None:
                self.async_client = AsyncOpenAI(**client_kwargs)
        return self

    def _document_to_str(
        self,
        document: str | Document | Mapping[str, Any],
        rank_fields: Sequence[str] | None = None,
    ) -> str:
        """将受支持的文档值转换为字符串 API 格式。"""
        if isinstance(document, Document):
            return document.page_content
        if isinstance(document, Mapping):
            value: Mapping[str, Any] = document
            if rank_fields is not None:
                value = {key: document[key] for key in rank_fields if key in document}
            return json.dumps(value, ensure_ascii=False, default=str)
        return document

    def _build_payload(
        self,
        documents: Sequence[str | Document | Mapping[str, Any]],
        query: str,
        *,
        rank_fields: Sequence[str] | None,
        model: str | None,
        top_n: int | None,
        task: str | None,
    ) -> dict[str, Any]:
        """构建 `/rerank` 端点的请求负载。"""
        requested_top_n = top_n if top_n is None or top_n > 0 else self.top_n
        payload: dict[str, Any] = {
            "model": model or self.model,
            "query": query,
            "documents": [
                self._document_to_str(document, rank_fields) for document in documents
            ],
            "return_documents": False,
        }
        if requested_top_n is not None:
            payload["top_n"] = requested_top_n
        if task is not None:
            payload["task"] = task
        return payload

    @staticmethod
    def _parse_response(response: Mapping[str, Any]) -> list[dict[str, Any]]:
        """从重排序响应中提取索引和得分对。"""
        return [
            {
                "index": result["index"],
                "relevance_score": result["relevance_score"],
            }
            for result in response["data"]
        ]

    def rerank(
        self,
        documents: Sequence[str | Document | Mapping[str, Any]],
        query: str,
        *,
        rank_fields: Sequence[str] | None = None,
        model: str | None = None,
        top_n: int | None = -1,
        task: str | None = None,
    ) -> list[dict[str, Any]]:
        """按查询相关性排序的文档索引。

        Args:
            documents: 要重排序的文档。
            query: 用于重排序的查询。
            rank_fields: 序列化映射时包含的字段映射。
            model: 使用的模型，替代配置的模型。
            top_n: 要返回的结果数量。`None` 表示返回所有结果。
            task: 可选的任务指令，用于重排序模型。

        Returns:
            包含每个文档索引和得分的重排序结果。
        """
        if not documents:
            return []

        payload = self._build_payload(
            documents,
            query,
            rank_fields=rank_fields,
            model=model,
            top_n=top_n,
            task=task,
        )
        response = self.client.post("/rerank", cast_to=_RESPONSE_TYPE, body=payload)
        return self._parse_response(response)

    async def arerank(
        self,
        documents: Sequence[str | Document | Mapping[str, Any]],
        query: str,
        *,
        rank_fields: Sequence[str] | None = None,
        model: str | None = None,
        top_n: int | None = -1,
        task: str | None = None,
    ) -> list[dict[str, Any]]:
        """异步返回按查询相关性排序的文档索引。

        Args:
            documents: 要重排序的文档。
            query: 用于重排序的查询。
            rank_fields: 序列化映射时包含的字段映射。
            model: 使用的模型，替代配置的模型。
            top_n: 要返回的结果数量。`None` 表示返回所有结果。
            task: 可选的任务指令，用于重排序模型。

        Returns:
            包含每个文档索引和得分的重排序结果。
        """
        if not documents:
            return []

        payload = self._build_payload(
            documents,
            query,
            rank_fields=rank_fields,
            model=model,
            top_n=top_n,
            task=task,
        )
        response = await self.async_client.post(
            "/rerank", cast_to=_RESPONSE_TYPE, body=payload
        )
        return self._parse_response(response)

    @staticmethod
    def _apply_results(
        documents: Sequence[Document],
        results: Sequence[Mapping[str, Any]],
    ) -> list[Document]:
        """复制重排序后的文档，并记录每个相关性得分。"""
        compressed = []
        for result in results:
            document = documents[result["index"]]
            document_copy = Document(
                document.page_content,
                metadata=deepcopy(document.metadata),
            )
            document_copy.metadata["relevance_score"] = result["relevance_score"]
            compressed.append(document_copy)
        return compressed

    @override
    def compress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Callbacks | None = None,
    ) -> Sequence[Document]:
        """通过保留最相关的结果来压缩文档。"""
        return self._apply_results(documents, self.rerank(documents, query))

    @override
    async def acompress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Callbacks | None = None,
    ) -> Sequence[Document]:
        """通过保留最相关的结果，异步压缩文档。"""
        return self._apply_results(documents, await self.arerank(documents, query))

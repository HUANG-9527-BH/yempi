"""使用 Exa 搜索 API 的检索器。"""

from __future__ import annotations

from typing import Any, Literal

from exa_py import Exa  # type: ignore[untyped-import]
from exa_py.api import (
    HighlightsContentsOptions,  # type: ignore[untyped-import]
    TextContentsOptions,  # type: ignore[untyped-import]
)
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import Field, SecretStr, model_validator

from langchain_exa._utilities import initialize_client


def _get_metadata(result: Any) -> dict[str, Any]:
    """从结果对象中获取元数据。"""
    metadata = {
        "title": result.title,
        "url": result.url,
        "id": result.id,
        "score": result.score,
        "published_date": result.published_date,
        "author": result.author,
    }
    for attribute in ("highlights", "highlight_scores", "summary"):
        if value := getattr(result, attribute, None):
            metadata[attribute] = value
    return metadata


class ExaSearchRetriever(BaseRetriever):
    """Exa 搜索检索器。"""

    k: int = 10  # num_results
    """要返回的搜索结果数量（1 到 100）。"""
    include_domains: list[str] | None = None
    """要在搜索中包含的域名列表。"""
    exclude_domains: list[str] | None = None
    """要在搜索中排除的域名列表。"""
    start_crawl_date: str | None = None
    """爬取的起始日期（格式为 YYYY-MM-DD）。"""
    end_crawl_date: str | None = None
    """爬取的结束日期（格式为 YYYY-MM-DD）。"""
    start_published_date: str | None = None
    """文档发布日期的起始时间（格式为 YYYY-MM-DD）。"""
    end_published_date: str | None = None
    """文档发布日期的结束时间（格式为 YYYY-MM-DD）。"""
    use_autoprompt: bool | None = None
    """是否使用自动提示进行搜索。"""
    type: str = "auto"
    """搜索类型，可选 'auto'、'deep' 或 'fast'。默认：auto"""
    highlights: HighlightsContentsOptions | bool | None = None
    """是否将页面内容设置为结果的高亮部分。"""
    text_contents_options: TextContentsOptions | dict[str, Any] | Literal[True] = True
    """How to set the page content of the results. Can be True or a dict with options
    like max_characters."""
    livecrawl: Literal["always", "fallback", "never"] | None = None
    """Option to crawl live webpages if content is not in the index. Options: "always",
    "fallback", "never"."""
    summary: bool | dict[str, str] | None = None
    """Whether to include a summary of the content. Can be a boolean or a dict with a
    custom query."""

    client: Exa = Field(default=None)  # type: ignore[assignment]
    exa_api_key: SecretStr = Field(default=SecretStr(""))
    exa_base_url: str | None = None

    @model_validator(mode="before")
    @classmethod
    def validate_environment(cls, values: dict) -> Any:
        """验证环境。"""
        return initialize_client(values)

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        response = self.client.search_and_contents(  # type: ignore[call-overload]
            query,
            num_results=self.k,
            text=self.text_contents_options,
            highlights=self.highlights,
            include_domains=self.include_domains,
            exclude_domains=self.exclude_domains,
            start_crawl_date=self.start_crawl_date,
            end_crawl_date=self.end_crawl_date,
            start_published_date=self.start_published_date,
            end_published_date=self.end_published_date,
            use_autoprompt=self.use_autoprompt,
            livecrawl=self.livecrawl,
            summary=self.summary,
            type=self.type,
        )  # type: ignore[call-overload, misc]

        results = response.results

        return [
            Document(
                page_content=(result.text),
                metadata=_get_metadata(result),
            )
            for result in results
        ]

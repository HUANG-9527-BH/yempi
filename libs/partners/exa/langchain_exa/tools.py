"""Exa 搜索 API 的工具。"""

from __future__ import annotations

from typing import Any, Literal

from exa_py import Exa  # type: ignore[untyped-import]
from exa_py.api import (
    HighlightsContentsOptions,  # type: ignore[untyped-import]
    TextContentsOptions,  # type: ignore[untyped-import]
)
from langchain_core.callbacks import (
    CallbackManagerForToolRun,
)
from langchain_core.tools import BaseTool
from pydantic import Field, SecretStr, model_validator

from langchain_exa._utilities import initialize_client


class ExaSearchResults(BaseTool):  # type: ignore[override]
    r"""Exa 搜索工具。

    设置：
        安装 `langchain-exa` 并设置环境变量 `EXA_API_KEY`。

        ```bash
        pip install -U langchain-exa
        export EXA_API_KEY="your-api-key"
        ```

    实例化：
        ```python
        from langchain-exa import ExaSearchResults

        tool = ExaSearchResults()
        ```

    使用参数调用：
        ```python
        tool.invoke({"query": "what is the weather in SF", "num_results": 1})
        ```

        ```python
        SearchResponse(
            results=[
                Result(
                    url="https://www.wunderground.com/weather/37.8,-122.4",
                    id="https://www.wunderground.com/weather/37.8,-122.4",
                    title="San Francisco, CA Weather Conditionsstar_ratehome",
                    score=0.1843988299369812,
                    published_date="2023-02-23T01:17:06.594Z",
                    author=None,
                    text="The time period when the sun is no more than 6 degrees below the horizon at either sunrise or sunset. The horizon should be clearly defined and the brightest stars should be visible under good atmospheric conditions (i.e. no moonlight, or other lights). One still should be able to carry on ordinary outdoor activities. The time period when the sun is between 6 and 12 degrees below the horizon at either sunrise or sunset. The horizon is well defined and the outline of objects might be visible without artificial light. Ordinary outdoor activities are not possible at this time without extra illumination. The time period when the sun is between 12 and 18 degrees below the horizon at either sunrise or sunset. The sun does not contribute to the illumination of the sky before this time in the morning, or after this time in the evening. In the beginning of morning astronomical twilight and at the end of astronomical twilight in the evening, sky illumination is very faint, and might be undetectable. The time of Civil Sunset minus the time of Civil Sunrise. The time of Actual Sunset minus the time of Actual Sunrise. The change in length of daylight between today and tomorrow is also listed when available.",
                    highlights=None,
                    highlight_scores=None,
                    summary=None,
                )
            ],
            autoprompt_string=None,
        )
        ```

    使用 ToolCall 调用：

        ```python
        tool.invoke(
            {
                "args": {"query": "what is the weather in SF", "num_results": 1},
                "id": "1",
                "name": tool.name,
                "type": "tool_call",
            }
        )
        ```

        ```python
        ToolMessage(
            content="Title: San Francisco, CA Weather Conditionsstar_ratehome\nURL: https://www.wunderground.com/weather/37.8,-122.4\nID: https://www.wunderground.com/weather/37.8,-122.4\nScore: 0.1843988299369812\nPublished Date: 2023-02-23T01:17:06.594Z\nAuthor: None\nText: The time period when the sun is no more than 6 degrees below the horizon at either sunrise or sunset. The horizon should be clearly defined and the brightest stars should be visible under good atmospheric conditions (i.e. no moonlight, or other lights). One still should be able to carry on ordinary outdoor activities. The time period when the sun is between 6 and 12 degrees below the horizon at either sunrise or sunset. The horizon is well defined and the outline of objects might be visible without artificial light. Ordinary outdoor activities are not possible at this time without extra illumination. The time period when the sun is between 12 and 18 degrees below the horizon at either sunrise or sunset. The sun does not contribute to the illumination of the sky before this time in the morning, or after this time in the evening. In the beginning of morning astronomical twilight and at the end of astronomical twilight in the evening, sky illumination is very faint, and might be undetectable. The time of Civil Sunset minus the time of Civil Sunrise. The time of Actual Sunset minus the time of Actual Sunrise. The change in length of daylight between today and tomorrow is also listed when available.\nHighlights: None\nHighlight Scores: None\nSummary: None\n",
            name="exa_search_results_json",
            tool_call_id="1",
        )
        ```
    """  # noqa: E501

    name: str = "exa_search_results_json"
    description: str = (
        "Exa Search, one of the best web search APIs built for AI. "
        "Input should be an Exa-optimized query. "
        "Output is a JSON array of the query results"
    )
    client: Exa = Field(default=None)  # type: ignore[assignment]
    exa_api_key: SecretStr = Field(default=SecretStr(""))

    @model_validator(mode="before")
    @classmethod
    def validate_environment(cls, values: dict) -> Any:
        """验证环境。"""
        return initialize_client(values)

    def _run(
        self,
        query: str,
        num_results: int = 10,
        text_contents_options: TextContentsOptions  # noqa: FBT001
        | dict[str, Any]
        | bool
        | None = None,
        highlights: HighlightsContentsOptions | bool | None = None,  # noqa: FBT001
        include_domains: list[str] | None = None,
        exclude_domains: list[str] | None = None,
        start_crawl_date: str | None = None,
        end_crawl_date: str | None = None,
        start_published_date: str | None = None,
        end_published_date: str | None = None,
        use_autoprompt: bool | None = None,  # noqa: FBT001
        livecrawl: Literal["always", "fallback", "never"] | None = None,
        summary: bool | dict[str, str] | None = None,  # noqa: FBT001
        type: Literal["auto", "deep", "fast"] | None = None,  # noqa: A002
        run_manager: CallbackManagerForToolRun | None = None,
    ) -> list[dict] | str:
                # TODO: 将 `type` 重命名为其他名称，因为它是保留关键字
        """使用该工具。

        Args:
            query: 搜索查询。
            num_results: 返回的搜索结果数量（1 到 100）。默认值：10
            text_contents_options: 如何设置结果的页面内容。可以是 True 或包含 max_characters 等选项的字典。
            highlights: 是否在结果中包含高亮部分。
            include_domains: 要包含在搜索中的域名列表。
            exclude_domains: 要从搜索中排除的域名列表。
            start_crawl_date: 爬取开始日期（格式为 YYYY-MM-DD）。
            end_crawl_date: 爬取结束日期（格式为 YYYY-MM-DD）。
            start_published_date: 文档发布时间的开始日期（格式为 YYYY-MM-DD）。
            end_published_date: 文档发布时间的结束日期（格式为 YYYY-MM-DD）。
            use_autoprompt: 是否使用自动提示进行搜索。
            livecrawl: 如果内容不在索引中，是否爬取实时网页。选项："always", "fallback", "never"
            summary: 是否包含内容摘要。可以是布尔值或包含自定义查询的字典。
            type: 搜索类型，'auto'、'deep' 或 'fast'。
            run_manager: 用于回调的运行管理器。

        """  # noqa: E501
        try:
            return self.client.search_and_contents(
                query,
                num_results=num_results,
                text=text_contents_options,
                highlights=highlights,
                include_domains=include_domains,
                exclude_domains=exclude_domains,
                start_crawl_date=start_crawl_date,
                end_crawl_date=end_crawl_date,
                start_published_date=start_published_date,
                end_published_date=end_published_date,
                use_autoprompt=use_autoprompt,
                livecrawl=livecrawl,
                summary=summary,
                type=type,
            )  # type: ignore[call-overload, misc]
        except Exception as e:
            return repr(e)


class ExaFindSimilarResults(BaseTool):  # type: ignore[override]
    """查询 Metaphor 搜索 API 并返回 JSON 的工具。"""

    name: str = "exa_find_similar_results_json"
    description: str = (
        "A wrapper around Exa Find Similar. "
        "Input should be an Exa-optimized query. "
        "Output is a JSON array of the query results"
    )
    client: Exa = Field(default=None)  # type: ignore[assignment]
    exa_api_key: SecretStr = Field(default=SecretStr(""))
    exa_base_url: str | None = None

    @model_validator(mode="before")
    @classmethod
    def validate_environment(cls, values: dict) -> Any:
        """验证环境。"""
        return initialize_client(values)

    def _run(
        self,
        url: str,
        num_results: int = 10,
        text_contents_options: TextContentsOptions  # noqa: FBT001
        | dict[str, Any]
        | bool
        | None = None,
        highlights: HighlightsContentsOptions | bool | None = None,  # noqa: FBT001
        include_domains: list[str] | None = None,
        exclude_domains: list[str] | None = None,
        start_crawl_date: str | None = None,
        end_crawl_date: str | None = None,
        start_published_date: str | None = None,
        end_published_date: str | None = None,
        exclude_source_domain: bool | None = None,  # noqa: FBT001
        category: str | None = None,
        livecrawl: Literal["always", "fallback", "never"] | None = None,
        summary: bool | dict[str, str] | None = None,  # noqa: FBT001
        run_manager: CallbackManagerForToolRun | None = None,
    ) -> list[dict] | str:
        """使用该工具。

        Args:
            url: 用于查找相似页面的 URL。
            num_results: 返回的搜索结果数量（1 到 100）。默认值：10
            text_contents_options: 如何设置结果的页面内容。可以是 True 或包含 max_characters 等选项的字典。
            highlights: 是否在结果中包含高亮部分。
            include_domains: 要包含在搜索中的域名列表。
            exclude_domains: 要从搜索中排除的域名列表。
            start_crawl_date: 爬取开始日期（格式为 YYYY-MM-DD）。
            end_crawl_date: 爬取结束日期（格式为 YYYY-MM-DD）。
            start_published_date: 文档发布时间的开始日期（格式为 YYYY-MM-DD）。
            end_published_date: 文档发布时间的结束日期（格式为 YYYY-MM-DD）。
            exclude_source_domain: 如果为 `True`，则排除与源 URL 相同域名的页面。
            category: 按类别筛选相似页面。
            livecrawl: 如果内容不在索引中，是否爬取实时网页。选项："always", "fallback", "never"
            summary: 是否包含内容摘要。可以是布尔值或包含自定义查询的字典。
            run_manager: 用于回调的运行管理器。

        """  # noqa: E501
        try:
            return self.client.find_similar_and_contents(
                url,
                num_results=num_results,
                text=text_contents_options,
                highlights=highlights,
                include_domains=include_domains,
                exclude_domains=exclude_domains,
                start_crawl_date=start_crawl_date,
                end_crawl_date=end_crawl_date,
                start_published_date=start_published_date,
                end_published_date=end_published_date,
                exclude_source_domain=exclude_source_domain,
                category=category,
                livecrawl=livecrawl,
                summary=summary,
            )  # type: ignore[call-overload, misc]
        except Exception as e:
            return repr(e)

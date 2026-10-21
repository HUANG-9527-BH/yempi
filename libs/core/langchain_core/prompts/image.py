"""多模态模型的图像提示词模板。"""

from typing import Any, Literal, cast

from pydantic import Field

from langchain_core.prompt_values import ImagePromptValue, ImageURL, PromptValue
from langchain_core.prompts.base import BasePromptTemplate
from langchain_core.prompts.string import (
    DEFAULT_FORMATTER_MAPPING,
    PromptTemplateFormat,
    get_template_variables,
)
from langchain_core.runnables import run_in_executor


class ImagePromptTemplate(BasePromptTemplate[ImageURL]):
    """多模态模型的图像提示词模板。

    Example:
        ```python
        prompt = ImagePromptTemplate(
            input_variables=["image_id"],
            template={"url": "https://example.com/{image_id}.png", "detail": "high"},
            template_format="f-string",
        )
        prompt.format(image_id="cat")
        # {"url": "https://example.com/cat.png", "detail": "high"}
        ```
    """

    template: dict[str, Any] = Field(default_factory=dict)
    """提示词的模板。"""

    template_format: PromptTemplateFormat = "f-string"
    """提示词模板的格式。

    Options are: 可选项：`'f-string'`, `'mustache'`, `'jinja2'`。
    """

    def __init__(self, **kwargs: Any) -> None:
        """创建一个图像提示词模板。

        Raises:
            ValueError: 如果输入变量包含 `'url'`, `'path'`, 或 `'detail'`。
        """
        if "input_variables" not in kwargs:
            kwargs["input_variables"] = []

        overlap = set(kwargs["input_variables"]) & {"url", "path", "detail"}
        if overlap:
            msg = (
                "input_variables for the image template cannot contain"
                " any of 'url', 'path', or 'detail'."
                f" Found: {overlap}"
            )
            raise ValueError(msg)

        template = kwargs.get("template", {})
        template_format = kwargs.get("template_format", "f-string")
        for value in template.values():
            if isinstance(value, str):
                get_template_variables(value, template_format)

        super().__init__(**kwargs)

    @property
    def _prompt_type(self) -> str:
        """返回提示词类型键。"""
        return "image-prompt"

    @classmethod
    def get_lc_namespace(cls) -> list[str]:
        """获取 LangChain 对象的命名空间。

        Returns:
            `["langchain", "prompts", "image"]`
        """
        return ["langchain", "prompts", "image"]

    def format_prompt(self, **kwargs: Any) -> PromptValue:
        """使用输入格式化提示词。

        Args:
            **kwargs: 传递给提示词模板的任意参数。

        Returns:
            格式化后的字符串。
        """
        return ImagePromptValue(image_url=self.format(**kwargs))

    async def aformat_prompt(self, **kwargs: Any) -> PromptValue:
        """异步使用输入格式化提示词。

        Args:
            **kwargs: 传递给提示词模板的任意参数。

        Returns:
            格式化后的字符串。
        """
        return ImagePromptValue(image_url=await self.aformat(**kwargs))

    def format(
        self,
        **kwargs: Any,
    ) -> ImageURL:
        """使用输入格式化提示词。

        Args:
            **kwargs: 传递给提示词模板的任意参数。

        Returns:
            格式化后的字符串。

        Raises:
            ValueError: 如果未提供 url。
            ValueError: 如果 url 不是字符串。
            ValueError: 如果模板或 kwargs 中提供了 `'path'`。

        Example:
            ```python
            prompt.format(variable1="foo")
            ```
        """
        formatted = {}
        for k, v in self.template.items():
            if isinstance(v, str):
                formatted[k] = DEFAULT_FORMATTER_MAPPING[self.template_format](
                    v, **kwargs
                )
            else:
                formatted[k] = v
        url = kwargs.get("url") or formatted.get("url")
        if kwargs.get("path") or formatted.get("path"):
            msg = (
                "Loading images from 'path' has been removed as of 0.3.15 for security "
                "reasons. Please specify images by 'url'."
            )
            raise ValueError(msg)
        detail = kwargs.get("detail") or formatted.get("detail")
        if not url:
            msg = "Must provide url."
            raise ValueError(msg)
        if not isinstance(url, str):
            msg = "url must be a string."
            raise ValueError(msg)  # noqa: TRY004
        output: ImageURL = {"url": url}
        if detail:
                        # Don't check literal values here: let the API check them
            output["detail"] = cast("Literal['auto', 'low', 'high']", detail)
        return output

    async def aformat(self, **kwargs: Any) -> ImageURL:
        """异步使用输入格式化提示词。

        Args:
            **kwargs: 传递给提示词模板的任意参数。

        Returns:
            格式化后的字符串。
        """
        return await run_in_executor(None, self.format, **kwargs)

    def pretty_repr(
        self,
        html: bool = False,  # noqa: FBT001,FBT002
    ) -> str:
        """返回提示词的美化表示形式。

        Args:
            html: 是否返回 HTML 格式的字符串。

        Returns:
            提示词的美化表示形式。
        """
        raise NotImplementedError

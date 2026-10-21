"""实现一个由大语言模型驱动的浏览器。"""

from __future__ import annotations

import warnings
from typing import Any

from langchain_core._api import deprecated
from langchain_core.callbacks import CallbackManagerForChainRun
from langchain_core.language_models import BaseLanguageModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable
from pydantic import ConfigDict, model_validator

from langchain_classic.chains.base import Chain
from langchain_classic.chains.natbot.prompt import PROMPT


@deprecated(
    since="0.2.13",
    removal="2.0.0",
    alternative_import="langchain_community.chains.natbot.NatBotChain",
)
class NatBotChain(Chain):
    """实现一个由大语言模型驱动的浏览器。

    **安全提示**：此工具集提供控制网页浏览器的代码。

        网页浏览器可用于导航至：

        - 任意 URL（包括任何内部网络 URL）
        - 以及本地文件

        Exercise care if exposing this chain to end-users. Control who is able to
        access and use this chain, and isolate the network access of the server
        that hosts this chain.

        详情参见 https://docs.langchain.com/oss/python/security-policy。

    Example:
        ```python
        from langchain_classic.chains import NatBotChain

        natbot = NatBotChain.from_default("Buy me a new hat.")
        ```
    """

    llm_chain: Runnable
    objective: str
    """NatBot 被指派完成的目标。"""
    llm: BaseLanguageModel | None = None
    """[已弃用] 用于使用的 LLM 包装器。"""
    input_url_key: str = "url"
    input_browser_content_key: str = "browser_content"
    previous_command: str = ""
    output_key: str = "command"

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="forbid",
    )

    @model_validator(mode="before")
    @classmethod
    def _raise_deprecation(cls, values: dict) -> Any:
        if "llm" in values:
            warnings.warn(
                "Directly instantiating an NatBotChain with an llm is deprecated. "
                "Please instantiate with llm_chain argument or using the from_llm "
                "class method.",
                stacklevel=5,
            )
            if "llm_chain" not in values and values["llm"] is not None:
                values["llm_chain"] = PROMPT | values["llm"] | StrOutputParser()
        return values

    @classmethod
    def from_default(cls, objective: str, **kwargs: Any) -> NatBotChain:
        """使用默认的 LLMChain 加载。"""
        msg = (
            "This method is no longer implemented. Please use from_llm."
            "model = OpenAI(temperature=0.5, best_of=10, n=3, max_tokens=50)"
            "For example, NatBotChain.from_llm(model, objective)"
        )
        raise NotImplementedError(msg)

    @classmethod
    def from_llm(
        cls,
        llm: BaseLanguageModel,
        objective: str,
        **kwargs: Any,
    ) -> NatBotChain:
        """从 LLM 加载。"""
        llm_chain = PROMPT | llm | StrOutputParser()
        return cls(llm_chain=llm_chain, objective=objective, **kwargs)

    @property
    def input_keys(self) -> list[str]:
        """期望输入 url 和浏览器内容。"""
        return [self.input_url_key, self.input_browser_content_key]

    @property
    def output_keys(self) -> list[str]:
        """返回命令。"""
        return [self.output_key]

    def _call(
        self,
        inputs: dict[str, str],
        run_manager: CallbackManagerForChainRun | None = None,
    ) -> dict[str, str]:
        _run_manager = run_manager or CallbackManagerForChainRun.get_noop_manager()
        url = inputs[self.input_url_key]
        browser_content = inputs[self.input_browser_content_key]
        llm_cmd = self.llm_chain.invoke(
            {
                "objective": self.objective,
                "url": url[:100],
                "previous_command": self.previous_command,
                "browser_content": browser_content[:4500],
            },
            config={"callbacks": _run_manager.get_child()},
        )
        llm_cmd = llm_cmd.strip()
        self.previous_command = llm_cmd
        return {self.output_key: llm_cmd}

    def execute(self, url: str, browser_content: str) -> str:
        """确定下一个要运行的浏览器命令。

        Args:
            url: 当前所在网站的 URL。
            browser_content: 浏览器当前显示的页面内容。

        Returns:
            下一个要运行的浏览器命令。

        Example:
            ```python
            browser_content = "...."
            llm_command = natbot.run("www.google.com", browser_content)
            ```
        """
        _inputs = {
            self.input_url_key: url,
            self.input_browser_content_key: browser_content,
        }
        return self(_inputs)[self.output_key]

    @property
    def _chain_type(self) -> str:
        return "nat_bot_chain"

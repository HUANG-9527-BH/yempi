from __future__ import annotations

from typing import TYPE_CHECKING

from langchain_core.callbacks.base import BaseCallbackHandler

if TYPE_CHECKING:
    from langchain_community.callbacks import LLMThoughtLabeler
    from streamlit.delta_generator import DeltaGenerator


def StreamlitCallbackHandler(  # noqa: N802
    parent_container: DeltaGenerator,
    *,
    max_thought_containers: int = 4,
    expand_new_thoughts: bool = True,
    collapse_completed_thoughts: bool = True,
    thought_labeler: LLMThoughtLabeler | None = None,
) -> BaseCallbackHandler:
    """回调处理器，将输出写入 Streamlit 应用。

    This CallbackHandler is geared towards
    use with a LangChain Agent; it displays the Agent's LLM and tool-usage "thoughts"
    inside a series of Streamlit expanders.

    参数 ---------- parent_container 将包含处理器创建的所有 Streamlit 元素的
    `st.container`。 max_thought_containers 同时显示的最大已完成 LLM
    思考容器数量。当达到此阈值时，新的思考将导致最旧的思考被折叠进一个 "历史"
    展开区域。 expand_new_thoughts 每个 LLM "思考" 都会拥有自己的
    `st.expander`。此参数控制该展开区域是否默认展开。 collapse_completed_thoughts
    如果为 `True`，则在 LLM 思考完成后将其展开区域折叠。 thought_labeler 可选的自定义
    LLMThoughtLabeler 实例。若未指定，处理器将使用默认的思考标签逻辑。

    Returns:
    -------
    一个新的 StreamlitCallbackHandler 实例。

    Note that this is an "auto-updating" API: if the installed version of Streamlit
    has a more recent StreamlitCallbackHandler implementation, an instance of that class
    will be used.

    """
        # 如果正在使用的 Streamlit 版本实现了 StreamlitCallbackHandler，
        # 则委托给它，而不是使用我们内置的处理器。官方处理器可保证支持相同的参数集。
        # 这是官方处理器，因此可以直接返回它。
    try:
        from streamlit.external.langchain import StreamlitCallbackHandler

                # 这是官方处理器，因此可以直接返回它。
        return StreamlitCallbackHandler(
            parent_container,
            max_thought_containers=max_thought_containers,
            expand_new_thoughts=expand_new_thoughts,
            collapse_completed_thoughts=collapse_completed_thoughts,
            thought_labeler=thought_labeler,
        )
    except ImportError:
        try:
            from langchain_community.callbacks.streamlit.streamlit_callback_handler import (  # noqa: E501
                StreamlitCallbackHandler as _InternalStreamlitCallbackHandler,
            )
        except ImportError as e:
            msg = (
                "To use the StreamlitCallbackHandler, please install "
                "langchain-community with `pip install langchain-community`."
            )
            raise ImportError(msg) from e

        return _InternalStreamlitCallbackHandler(
            parent_container,
            max_thought_containers=max_thought_containers,
            expand_new_thoughts=expand_new_thoughts,
            collapse_completed_thoughts=collapse_completed_thoughts,
            thought_labeler=thought_labeler,
        )

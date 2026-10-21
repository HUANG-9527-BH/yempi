from __future__ import annotations

from langchain_core.language_models import LanguageModelLike
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import BasePromptTemplate
from langchain_core.retrievers import RetrieverLike, RetrieverOutputLike
from langchain_core.runnables import RunnableBranch


def create_history_aware_retriever(
    llm: LanguageModelLike,
    retriever: RetrieverLike,
    prompt: BasePromptTemplate,
) -> RetrieverOutputLike:
    """创建一个链，该链接收对话历史并返回文档。

    如果没有 `chat_history`，则 `input` 直接传递给检索器。
    如果存在 `chat_history`，则使用提示词和 LLM 生成搜索查询。
    该搜索查询随后传递给检索器。

    Args:
        llm: 用于根据对话历史生成搜索词的语言模型
        retriever: `RetrieverLike` 对象，接受字符串作为输入并输出
            一组 `Document` 对象。
        prompt: 用于为检索器生成搜索查询的提示词。

    Returns:
        一个 LCEL Runnable。Runnable 的输入必须包含 `input`，
        如果存在对话历史，则应以 `chat_history` 的形式提供。
        Runnable 的输出是一组 `Document` 对象

    Example:
        ```python
        # pip install -U langchain langchain-community

        from langchain_openai import ChatOpenAI
        from langchain_classic.chains import create_history_aware_retriever
        from langchain_classic import hub

        rephrase_prompt = hub.pull("langchain-ai/chat-langchain-rephrase")
        model = ChatOpenAI()
        retriever = ...
        chat_retriever_chain = create_history_aware_retriever(
            model, retriever, rephrase_prompt
        )

        chain.invoke({"input": "...", "chat_history": })

        ```
    """
    if "input" not in prompt.input_variables:
        msg = (
            "Expected `input` to be a prompt variable, "
            f"but got {prompt.input_variables}"
        )
        raise ValueError(msg)

    retrieve_documents: RetrieverOutputLike = RunnableBranch(
        (
                        # 两者（空字符串和空列表）都评估为 False
            lambda x: not x.get("chat_history", False),
                        # 如果没有对话历史，则直接将 input 传递给检索器
            (lambda x: x["input"]) | retriever,
        ),
                # 如果存在对话历史，则将输入传递给 LLM 链，再传递给检索器
        prompt | llm | StrOutputParser() | retriever,
    ).with_config(run_name="chat_retriever_chain")
    return retrieve_documents

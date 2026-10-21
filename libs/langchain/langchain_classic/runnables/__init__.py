"""LangChain **Runnable** 与 **LangChain 表达式语言 (LCEL)**。

LangChain 表达式语言 (LCEL) 提供了一种声明式方法，用于构建利用大语言模型 (LLM)
能力的生产级程序。

使用 LCEL 和 LangChain Runnables 创建的程序天然支持同步、异步、批处理和流式操作。

Support for **async** allows servers hosting the LCEL based programs
to scale better for higher concurrent loads.

**批处理** 操作允许并行处理多个输入。

**流式** 输出中间结果（在生成过程中）可创建更响应迅速的用户体验。

此模块包含非核心的 Runnable 类。
"""

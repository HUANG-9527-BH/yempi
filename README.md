<div align="center">
  <a href="https://docs.langchain.com/oss/python/langchain/overview">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset=".github/images/logo-dark.svg">
      <source media="(prefers-color-scheme: light)" srcset=".github/images/logo-light.svg">
      <img alt="LangChain Logo" src=".github/images/logo-dark.svg" width="50%">
    </picture>
  </a>
</div>

<div align="center">
  <h3>智能体工程平台。</h3>
</div>

<div align="center">
  <a href="https://opensource.org/licenses/MIT" target="_blank"><img src="https://img.shields.io/pypi/l/langchain" alt="PyPI - License"></a>
  <a href="https://pypistats.org/packages/langchain" target="_blank"><img src="https://img.shields.io/pepy/dt/langchain" alt="PyPI - Downloads"></a>
  <a href="https://pypi.org/project/langchain/#history" target="_blank"><img src="https://img.shields.io/pypi/v/langchain?label=%20" alt="Version"></a>
  <a href="https://x.com/langchain_oss" target="_blank"><img src="https://img.shields.io/twitter/url/https/twitter.com/langchain_oss.svg?style=social&label=Follow%20%40LangChain" alt="Twitter / X"></a>
</div>

<br>

LangChain 是一个用于构建智能体（Agent）和 LLM 应用的框架。它帮助你将可互操作的组件与第三方集成串联起来，简化 AI 应用开发——同时随着底层技术的演进，让你的决策更具前瞻性。

> [!TIP]
> 刚刚入门？不妨了解 **[Deep Agents](https://docs.langchain.com/oss/python/deepagents/)** —— 一个构建于 LangChain 之上的更高层级封装包，为智能体内置了规划、子智能体、文件系统使用等常见使用模式的能力。

## 快速开始

```bash
uv add langchain
```

```python
from langchain.chat_models import init_chat_model

model = init_chat_model("openai:gpt-5.5")
result = model.invoke("Hello, world!")
```

如果你需要更高级的定制或智能体编排，请查看 [LangGraph](https://github.com/langchain-ai/langgraph)，这是我们用于构建可控智能体工作流的框架。

如需等价的 JS/TS 库，请查看 [LangChain.js](https://github.com/langchain-ai/langchainjs)。

> [!TIP]
> 关于 AI 智能体与 LLM 应用的开发、调试和部署，请参阅 [LangSmith](https://docs.langchain.com/langsmith/home)。

## LangChain 生态系统

LangChain 框架既可以独立使用，也能与任何 LangChain 产品无缝集成，为开发者构建 LLM 应用提供一整套工具。

- **[Deep Agents](https://docs.langchain.com/oss/python/deepagents/)** —— 构建能够规划任务、调用子智能体并利用文件系统处理复杂任务的智能体
- **[LangGraph](https://docs.langchain.com/oss/python/langgraph/overview)** —— 使用我们的底层智能体编排框架，构建能够可靠处理复杂任务的智能体
- **[集成组件](https://docs.langchain.com/oss/python/integrations/providers/overview)** —— 对话与嵌入模型、工具与工具集等更多集成
- **[LangSmith](https://www.langchain.com/langsmith)** —— 面向 LLM 应用的智能体评估、可观测性与调试
- **[LangSmith 部署](https://docs.langchain.com/langsmith/deployments)** —— 借助专为长时间运行、有状态工作流打造的平台，部署并扩展你的智能体

## 为什么选择 LangChain？

LangChain 通过为模型、嵌入（Embeddings）、向量存储等组件提供统一的标准接口，帮助开发者构建由 LLM 驱动的应用。

- **实时数据增强** —— 借助 LangChain 海量的集成库（涵盖模型提供商、工具、向量存储、检索器等），轻松将 LLM 连接到多样化的数据源与内外部系统
- **模型可互换** —— 工程团队在试验过程中可自由切换模型，为应用需求找到最佳选择。随着行业前沿不断演进，你也能快速调整——LangChain 的抽象层让你持续前行、不失动力
- **快速原型开发** —— 利用 LangChain 模块化、组件化的架构，快速构建并迭代 LLM 应用。无需从零重建即可测试不同方案与工作流，加速你的开发周期
- **生产级特性** —— 通过 LangSmith 等集成获得内置的监控、评估与调试支持，部署可靠的应用。借助久经验证的模式与最佳实践，放心扩展规模
- **活跃的社区与生态** —— 利用由集成、模板和社区贡献组件构成的丰富生态。在活跃的开源社区中持续获益，紧跟最新的 AI 进展
- **灵活的抽象层次** —— 在适合你的抽象层级上工作——从快速上手的高层链到精细控制的底层组件。LangChain 与你的应用复杂度一同成长

---

## 资源

- [文档](https://docs.langchain.com/oss/python/langchain/overview) —— 概念概述与使用指南
- [LangChain 生态总览](https://docs.langchain.com/oss/python/concepts/products) —— LangChain、LangGraph 与 Deep Agents 如何协同配合
- [API 参考](https://reference.langchain.com/python) —— 所有公开类、函数与类型的完整参考
- [讨论区](https://forum.langchain.com/c/oss-product-help-lc-and-lg/langchain/14) —— 提出技术问题、分享想法与反馈的社区论坛
- [LangChain 学院](https://academy.langchain.com/) —— 由 LangChain 团队打造的关于 LangChain 库与产品的免费系统课程
- [贡献指南](https://docs.langchain.com/oss/python/contributing/overview) —— 如何参与贡献并找到适合新手的入门任务
- [行为准则](https://github.com/langchain-ai/langchain/?tab=coc-ov-file) —— 社区规范与标准

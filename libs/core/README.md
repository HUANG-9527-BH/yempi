# 🦜🍎️ LangChain Core

[![PyPI - Version](https://img.shields.io/pypi/v/langchain-core?label=%20)](https://pypi.org/project/langchain-core/#history)
[![PyPI - License](https://img.shields.io/pypi/l/langchain-core)](https://opensource.org/licenses/MIT)
[![PyPI - Downloads](https://img.shields.io/pepy/dt/langchain-core)](https://pypistats.org/packages/langchain-core)
[![Twitter](https://img.shields.io/twitter/url/https/twitter.com/langchain_oss.svg?style=social&label=Follow%20%40LangChain)](https://x.com/langchain_oss)

在寻找 JS/TS 版本？请查看 [LangChain.js](https://github.com/langchain-ai/langchainjs)。

为了帮助你更快地将 LangChain 应用交付到生产环境，请了解 [LangSmith](https://www.langchain.com/langsmith)。
[LangSmith](https://www.langchain.com/langsmith) 是一个统一的开发者平台，用于构建、测试和监控 LLM 应用。

## 快速安装

```bash
uv add langchain-core
```

## 🤔 这是什么？

LangChain Core 包含支撑整个 LangChain 生态系统的基础抽象。

这些抽象的设计力求模块化且尽量简洁。

拥有这些抽象的好处在于：任何提供商都可以实现所需的接口，从而轻松地被 LangChain 生态系统的其余部分所使用。

## ⛰️ 为什么基于 LangChain Core 构建？

整个 LangChain 生态系统都构建在 `langchain-core` 之上。其优势包括：

- **模块化**：我们围绕彼此独立、不绑定任何特定模型提供商的抽象来设计 Core。
- **稳定性**：我们致力于稳定的版本管理方案，任何破坏性变更都会提前告知并提升版本号。
- **久经验证**：Core 组件在 LLM 生态中拥有最大的安装基数，并被众多公司用于生产环境。

## 📖 文档

完整文档请参阅 [API 参考](https://reference.langchain.com/python/langchain_core/)。关于使用 LangChain 的概念指南、教程和示例，请查看 [LangChain 文档](https://docs.langchain.com/oss/python/langchain/overview)。你也可以通过 [Chat LangChain](https://chat.langchain.com) 与文档对话。

## 📕 发布与版本管理

请参阅我们的[发布策略](https://docs.langchain.com/oss/python/release-policy)和[版本管理策略](https://docs.langchain.com/oss/python/versioning)。

## 💁 参与贡献

作为一个处于快速发展领域的开源项目，我们对各种形式的贡献都极为欢迎——无论是新功能、基础设施改进，还是更完善的文档。

关于如何贡献的详细信息，请参阅[贡献指南](https://docs.langchain.com/oss/python/contributing/overview)。

## 资源

- [LangChain 学院](https://academy.langchain.com/) —— 由 LangChain 团队打造的关于 LangChain 库与产品的免费系统课程
- [行为准则](https://github.com/langchain-ai/langchain/?tab=coc-ov-file) —— 社区规范与标准

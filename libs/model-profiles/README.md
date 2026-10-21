# 🦜🪪 langchain-model-profiles

[![PyPI - Version](https://img.shields.io/pypi/v/langchain-model-profiles?label=%20)](https://pypi.org/project/langchain-model-profiles/#history)
[![PyPI - License](https://img.shields.io/pypi/l/langchain-model-profiles)](https://opensource.org/licenses/MIT)
[![PyPI - Downloads](https://img.shields.io/pepy/dt/langchain-model-profiles)](https://pypistats.org/packages/langchain-model-profiles)
[![Twitter](https://img.shields.io/twitter/url/https/twitter.com/langchain_oss.svg?style=social&label=Follow%20%40LangChain)](https://x.com/langchain_oss)

> [!WARNING]
> 该软件包目前正在开发中，API 可能会发生变化。

用于更新 LangChain 集成包中模型配置文件（profile）数据的 CLI 工具。

## 快速安装

```bash
uv add langchain-model-profiles
```

## 🤔 这是什么？

`langchain-model-profiles` 是一个 CLI 工具，用于从 [models.dev](https://github.com/sst/models.dev) 获取并更新模型能力数据，供 LangChain 集成包使用。

LangChain 聊天模型暴露了 `.profile` 字段，可通过编程方式访问模型能力，例如上下文窗口大小、支持的模态、工具调用、结构化输出等。该 CLI 工具可帮助维护者保持这些数据处于最新状态。

## 数据来源

该软件包构建在 [models.dev](https://github.com/sst/models.dev) 项目的出色工作之上，这是一个提供模型能力数据的开源项目。

LangChain 模型配置文件在 models.dev 数据的基础上补充了一些额外字段。我们打算随着上游项目的演进保持与其对齐。

## 📖 文档

完整文档请参阅 [API 参考](https://reference.langchain.com/python/langchain_model_profiles/)。有关使用 LangChain 的概念指南、教程和示例，请参阅 [LangChain 文档](https://docs.langchain.com/oss/python/langchain/overview)。你还可以通过 [Chat LangChain](https://chat.langchain.com) 与文档对话。

## 用法

更新特定提供商的模型配置文件数据：

```bash
langchain-profiles refresh --provider anthropic --data-dir ./langchain_anthropic/data
```

该命令会从 models.dev 下载最新的模型数据，将其与 `profile_augmentations.toml` 中定义的增补内容合并，并生成一个 `profiles.py` 文件。

## 资源

- [LangChain Academy](https://academy.langchain.com/) — 由 LangChain 团队打造的关于 LangChain 库与产品的全面免费课程
- [行为准则](https://github.com/langchain-ai/langchain/?tab=coc-ov-file) — 社区行为准则与规范

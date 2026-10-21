# langchain-anthropic

[![PyPI - Version](https://img.shields.io/pypi/v/langchain-anthropic?label=%20)](https://pypi.org/project/langchain-anthropic/#history)
[![PyPI - License](https://img.shields.io/pypi/l/langchain-anthropic)](https://opensource.org/licenses/MIT)
[![PyPI - Downloads](https://img.shields.io/pepy/dt/langchain-anthropic)](https://pypistats.org/packages/langchain-anthropic)
[![Twitter](https://img.shields.io/twitter/url/https/twitter.com/langchain_oss.svg?style=social&label=Follow%20%40LangChain)](https://x.com/langchain_oss)

在寻找 JS/TS 版本？请查看 [LangChain.js](https://github.com/langchain-ai/langchainjs)。

## 快速安装

```bash
uv add langchain-anthropic
```

## 🤔 这是什么？

本包包含 LangChain 与 Anthropic 生成式模型的集成。

## 📖 文档

完整文档请参阅 [API 参考](https://reference.langchain.com/python/integrations/langchain_anthropic/)。有关这些类的概念指南、教程和示例，请参阅 [LangChain 文档](https://docs.langchain.com/oss/python/integrations/providers/anthropic)。

## 📕 发布与版本控制

请参阅我们的[发布](https://docs.langchain.com/oss/python/release-policy)与[版本控制](https://docs.langchain.com/oss/python/versioning)政策。

## 💁 参与贡献

作为一个处于快速发展领域的开源项目，我们非常欢迎各种形式的贡献，无论是新功能、基础设施改进，还是更完善的文档。

有关如何贡献的详细信息，请参阅[贡献指南](https://docs.langchain.com/oss/python/contributing/overview)。

## 迁移到 Claude Sonnet 5.5

```python
from langchain_anthropic import ChatAnthropic

model = ChatAnthropic(
    model="claude-sonnet-5-5",
    max_tokens=16000,
    output_config={"effort": "medium"},
)
```

- 使用 `with_structured_output(schema, method="json_schema")` 获得原生结构化输出。Sonnet 5.5 会拒绝强制工具选择（`"any"` 或具体工具名）。函数调用式的结构化输出不会强制调用工具，如果模型未调用工具而直接作答，则会抛出解析错误。
- 思考（thinking）默认为自适应模式。若要禁用前置思考，请在 `high` 及以下档位使用 `thinking={"type": "between_tools"}`，且不附加其他思考相关字段。不支持 `disabled` 以及设置预算的 `enabled` 思考模式。
- 省略采样参数设置；非默认的 `temperature`、`top_p` 和 `top_k` 会被拒绝。请为思考与文本输出统一预算输出 token。
- 保留已签名的思考块（包括空块），并保持历史记录只追加。使用对话中途的系统消息来更改指令或工具，而不是编辑较早的轮次。
- 进度更新可能以思考块的形式到达。使用自适应思考并设置 `display="summarized"` 或 `display="updates"` 来展示它们；后者的 beta 请求头会自动添加。
- 在直接 Claude API 上使用计算机操作（computer use）需要 `computer_toolset_20260801`。请保留返回的工具使用内容：其 `toolset_name` 会在重放时被保留，并复制到匹配的工具结果中。使用工具集时，请移除旧的细粒度流式传输 beta。

平台特定的限制、advisor 配对以及拒绝/回退行为，请参阅[迁移指南](https://platform.claude.com/docs/en/models/sonnet-5-5/migration-guide)。

## 资源

- [LangChain Academy](https://academy.langchain.com/) — 由 LangChain 团队打造的关于 LangChain 库与产品的全面免费课程
- [行为准则](https://github.com/langchain-ai/langchain/?tab=coc-ov-file) — 社区规范与标准

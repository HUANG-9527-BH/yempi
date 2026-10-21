---
okf_version: "0.2"
---
# 文件

- [智能体执行流程与循环控制](agent-execution.md) - 跟踪从用户输入开始，经过模型调用、工具分发及循环终止条件的智能体运行时生命周期，包含详细的状态管理与中间件集成点。
- [智能体工厂与 create_agent](agent-factory.md) - 智能体工厂构建状态机，协调语言模型、工具执行与中间件层之间的对话流程。create_agent 函数负责工具绑定、结构化输出、状态模式解析以及图编译。
- [LangChain 系统架构](architecture.md) - 将 LangChain 框架高层分解为三层：langchain-core（抽象）、langchain（编排与智能体）、partners（提供商集成），展示依赖关系、组件职责与扩展边界。
- [回调系统与处理器集成](callbacks.md) - 文档化回调处理器架构，与 Runnable 及聊天模型的集成方式，以及追踪执行事件、流式处理和仪器化的设计模式。
- [聊天模型接口与生命周期](chat-models.md) - 文档化 BaseChatModel 协议、输入/输出处理、流式支持，以及与回调和模型性能分析的集成点。涵盖 init_chat_model() 工厂函数、提供商注册表与模型实例化。
- [CI/CD 工作流：GitHub Actions 与发布流程](ci-workflows.md) - LangChain 基于 GitHub Actions 的 CI/CD 系统，自动化测试、代码检查与发布管理，适用于具有智能变更检测、并行矩阵测试与严格发布门禁的单体仓库。
- [可组合性与 LCEL 链](composability.md) - 说明 Runnable 组件如何通过 LCEL 操作符组合，创建具备自动异步、批量与流式支持的可重用工作流。
- [开发命令与本地环境设置](dev-commands.md) - LangChain 单体仓库中 uv、make、lint、test 与类型检查命令的快速参考，包括环境配置、pre-commit 钩子与测试工作流。
- [集成测试：真实 API 测试与 VCR 录制匣](integration-tests.md) - 如何编写调用真实模型 API 的集成测试，使用 VCR 录制匣实现 CI 兼容性，包括环境设置、录制匣管理与参数化模式。
- [MCP（模型上下文协议）集成](mcp-integration.md) - MCPAdapter 将 MCP 服务器桥接到 LangChain 智能体，实现工具发现、通过 FastMCP 进行协议协商、多传输管理，并支持通过 LangGraph 中断在调用过程中接收用户输入。
- [消息类型与内容表示](messages.md) - 文档化消息抽象、用于多模态 LLM I/O 的标准化内容块、消息层级结构，以及提供商特定的内容块转换器。
- [智能体中间件：可组合的请求/响应处理](middleware.md) - 中间件组合系统的全面指南：钩子类型、类型定义、负载转换，以及在模型调用、工具调用与智能体生命周期中插入钩子的实际示例。
- [通过 init_chat_model 初始化聊天模型](model-initialization.md) - 从提供商字符串实例化聊天模型的工厂函数，提供统一配置与运行时模型切换能力。
- [OpenAI 集成：ChatOpenAI 与 Azure 支持](openai-provider.md) - ChatOpenAI 对 OpenAI 聊天补全与响应 API 的集成，支持工具调用、结构化输出、视觉理解、流式传输与 Azure 部署。
- [添加新的聊天模型提供商](partner-pattern.md) - 将新 LLM 服务商集成到 LangChain 单体仓库的逐步指南，包括包结构、ChatModel 实现、流式处理、函数调用、结构化输出与标准测试。涵盖消息转换、错误处理、模型配置文件，以及可选的高级 API 模式（如响应 API）。
- [提示词模板与少样本学习](prompts.md) - 提示词模板定义了消息序列与变量替换模式，用于聊天模型。少样本学习通过动态选择示例来实现以例教学。
- [LangChain 仓库快速入门](quickstart.md) - 工程师入口：了解单体仓库结构，运行首个测试，掌握常见任务的修改位置，并引导至主要开发区域。
- [Runnable：核心组合层](runnables.md) - 解释 Runnable 协议及其如何通过 LangChain 表达式语言（LCEL）实现 LLM 组件的可组合链式调用。
- [源码映射：仓库文件组织](source-map.md) - 快速参考：按主题定位代码，将 LangChain 概念映射到单体仓库中的实现路径，涵盖核心抽象、智能体、中间件、合作伙伴与配置文件。
- [流式：逐令牌输出](streaming.md) - 描述 LLM 组件与链中流式处理的工作机制，通过 AIMessageChunk 实现逐令牌交付，集成回调，并权衡内存与延迟。
- [结构化输出与响应格式化](structured-output.md) - 文档化结构化输出系统：响应格式规范、策略选择（自动、提供商、工具驱动）、解析、验证与中间件集成。
- [工具与工具绑定](tools.md) - LangChain 的工具系统使智能体与语言模型能够通过基于模式感知的组件执行结构化操作，具备自动验证、错误处理与回调集成能力。
- [单元测试与模拟](unit-tests.md) - 编写单元测试的指南：测试结构、模拟聊天模型、测试夹具、断言与对智能体与组件的测试覆盖率。

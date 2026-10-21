# 录制 `_ChatOpenAICodex` 的 VCR 录像带

`_ChatOpenAICodex` 通过 ChatGPT 订阅的 OAuth 凭据进行身份验证，因此其集成测试在 PR CI 中无法在无登录状态下游走。以下工作流允许你本地一次性使用真实令牌录制录像带，将 OAuth 秘密清除后再写入磁盘，并提交录像带供 CI 通过 `_test_vcr.yml`（`check_diffs.yml` 中的 `vcr-tests` 任务）回放。

> [!WARNING]
> `_ChatOpenAICodex` 处于实验阶段且非官方支持。仅在你的 OpenAI 账户、工作区、计划以及适用的 OpenAI 条款允许通过 ChatGPT 认证访问 Codex 时方可使用。你需自行确保实现符合 OpenAI 的条款、使用政策、账户限制、速率限制和防护机制。

## 前提条件

1. 一个 ChatGPT 订阅（Plus / Pro / Team / Enterprise）——用于访问 `chatgpt.com/backend-api/codex`。
2. 磁盘上的一个令牌捆绑包。可通过以下命令生成：

   ```bash
   uv run --group test python -c \
     "from langchain_openai.chatgpt_oauth import login_chatgpt; login_chatgpt()"
   ```

   默认存储位置为 `~/.langchain/chatgpt-auth.json`。此路径有意与 `~/.codex/auth.json` 区分，以避免此处的刷新令牌轮换导致 Codex CLI / VS Code 会话失效。
3. 一个实例化 `_ChatOpenAICodex` 的集成测试，并标记为 `@pytest.mark.vcr`。针对 API 密钥 `ChatOpenAI` 编写的测试不会触发 Codex 后端 —— 只有专用于 Codex 的测试才应传递给该脚本。

## 录制

从 `libs/partners/openai/` 目录下执行：

```bash
# Record every VCR-marked integration test (default).
scripts/record_codex_cassettes.sh

# Record one file or one test.
scripts/record_codex_cassettes.sh tests/integration_tests/chat_models/test_codex.py
scripts/record_codex_cassettes.sh \
    tests/integration_tests/chat_models/test_codex.py::test_invoke

# Forward extra pytest args.
PYTEST_EXTRA="-k streaming -x" scripts/record_codex_cassettes.sh
```

该脚本：

1. 验证令牌存储是否存在。
2. 在 pytest 外部强制刷新访问令牌，确保 VCR 永远不会看到 `auth.openai.com/oauth/token` 的通信往返。若刷新令牌已撤销，会在此处暴露，而非在长时间录制后出现。
3. 运行 `pytest --record-mode=once -m vcr <target>`，创建缺失的录像带并重播已有录像带。
4. 使用 `zgrep` 检查每个录像带中的 bearer token、JWT、刷新授权请求体、泄露的 API 密钥以及 ChatGPT 账户 ID 声明。任何匹配都会导致非零退出并输出每文件报告 —— **切勿** 提交这些录像带。
5. 打印被修改的录像带文件的差异。

## 自动清除的内容

`tests/conftest.py` 会清除：

- 所有请求和响应头（捕获 `Authorization: Bearer …`、`ChatGPT-Account-Id`、cookies、组织 ID）。
- 请求 URI（确保不会在录像带中留下按账户区分的 URL 参数）。
- JSON 请求/响应体中的 OAuth 秘密字段：`access_token`、`refresh_token`、`id_token`、`code`、`code_verifier`、`device_code`、`client_secret`。
- urlencoded 表单体中的相同字段（刷新授权 POST 请求）。
- 任何在 body 中呈现为 JWT 格式字符串的内容（`eyJ…`）。

录制脚本的后期扫描旨在捕捉上述规则遗漏的内容。

## 覆盖令牌存储位置

```bash
CHATGPT_AUTH_FILE=/tmp/codex-test-token.json \
    scripts/record_codex_cassettes.sh
```

当需要针对专用测试账户进行录制时非常有用，可避免刷新令牌轮换影响你个人的 `~/.langchain/chatgpt-auth.json`。

## 提交

```bash
git -C libs/partners/openai status tests/cassettes/
git -C libs/partners/openai diff --stat tests/cassettes/
```

至少抽查一个新录制的录像带：`gunzip -c tests/cassettes/<name>.yaml.gz | less`。  
确认每个 `Authorization` 头部显示为 `**REDACTED**`，且不存在任何 `eyJ…` 字符串。

## CI 回放

`.github/workflows/check_diffs.yml` 将 OpenAI 的变更通过 `_test_vcr.yml` 路由，该文件运行如下：

```bash
make test_vcr   # uv run pytest --record-mode=none -m vcr tests/integration_tests/
```

`--record-mode=none` 会使得 pytest 失败而不是进行实时网络调用，因此缺失或过期的录像带（cassette）会直接导致硬性失败——这正是你期望的信号。

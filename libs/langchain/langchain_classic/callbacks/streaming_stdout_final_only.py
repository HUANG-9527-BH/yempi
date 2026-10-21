"""回调处理器在新 LLM token 时将其流式输出到标准输出。"""

import sys
from typing import Any

from langchain_core.callbacks import StreamingStdOutCallbackHandler
from typing_extensions import override

DEFAULT_ANSWER_PREFIX_TOKENS = ["Final", "Answer", ":"]


class FinalStreamingStdOutCallbackHandler(StreamingStdOutCallbackHandler):
    """用于智能体流式处理的回调处理器。

    仅适用于支持流式的 LLM 的智能体。

    只有智能体的最终输出会被流式传输。
    """

    def append_to_last_tokens(self, token: str) -> None:
        """将 token 追加到最后一个 token 列表中。"""
        self.last_tokens.append(token)
        self.last_tokens_stripped.append(token.strip())
        if len(self.last_tokens) > len(self.answer_prefix_tokens):
            self.last_tokens.pop(0)
            self.last_tokens_stripped.pop(0)

    def check_if_answer_reached(self) -> bool:
        """检查是否已达到答案。"""
        if self.strip_tokens:
            return self.last_tokens_stripped == self.answer_prefix_tokens_stripped
        return self.last_tokens == self.answer_prefix_tokens

    def __init__(
        self,
        *,
        answer_prefix_tokens: list[str] | None = None,
        strip_tokens: bool = True,
        stream_prefix: bool = False,
    ) -> None:
        """实例化 FinalStreamingStdOutCallbackHandler。

        Args:
            answer_prefix_tokens: Token sequence that prefixes the answer.
                Default is ["Final", "Answer", ":"]
            strip_tokens: Ignore white spaces and new lines when comparing
                answer_prefix_tokens to last tokens? (to determine if answer has been
                reached)
            stream_prefix: Should answer prefix itself also be streamed?
        """
        super().__init__()
        if answer_prefix_tokens is None:
            self.answer_prefix_tokens = DEFAULT_ANSWER_PREFIX_TOKENS
        else:
            self.answer_prefix_tokens = answer_prefix_tokens
        if strip_tokens:
            self.answer_prefix_tokens_stripped = [
                token.strip() for token in self.answer_prefix_tokens
            ]
        else:
            self.answer_prefix_tokens_stripped = self.answer_prefix_tokens
        self.last_tokens = [""] * len(self.answer_prefix_tokens)
        self.last_tokens_stripped = [""] * len(self.answer_prefix_tokens)
        self.strip_tokens = strip_tokens
        self.stream_prefix = stream_prefix
        self.answer_reached = False

    @override
    def on_llm_start(
        self,
        serialized: dict[str, Any],
        prompts: list[str],
        **kwargs: Any,
    ) -> None:
        """LLM 开始运行时调用。"""
        self.answer_reached = False

    @override
    def on_llm_new_token(
        self, token: str | list[str | dict[str, Any]], **kwargs: Any
    ) -> None:
        """在新 LLM token 时调用。仅在启用流式处理时可用。"""
        token_str = token if isinstance(token, str) else str(token)

                # 记住最后 n 个 token，其中 n = len(answer_prefix_tokens)
        self.append_to_last_tokens(token_str)

                # 检查最后 n 个 token 是否与 answer_prefix_tokens 列表匹配...
        if self.check_if_answer_reached():
            self.answer_reached = True
            if self.stream_prefix:
                for t in self.last_tokens:
                    sys.stdout.write(t)
                sys.stdout.flush()
            return

                # ... 如果是，则从现在开始打印 token
        if self.answer_reached:
            sys.stdout.write(token_str)
            sys.stdout.flush()

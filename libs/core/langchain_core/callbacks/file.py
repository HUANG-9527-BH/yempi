"""将输出写入文件的回调处理器。"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, TextIO, cast

from typing_extensions import Self, override

from langchain_core._api import warn_deprecated
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.utils.input import print_text

if TYPE_CHECKING:
    from langchain_core.agents import AgentAction, AgentFinish


_GLOBAL_DEPRECATION_WARNED = False


class FileCallbackHandler(BaseCallbackHandler):
    """将输出写入文件的回调处理器。

    This handler supports both context manager usage (recommended) and direct
    instantiation (deprecated) for backwards compatibility.

    Examples:
        作为上下文管理器使用（推荐）：

        ```python
        with FileCallbackHandler("output.txt") as handler:
            # Use handler with your chain/agent
            chain.invoke(inputs, config={"callbacks": [handler]})
        ```

        直接实例化（已弃用）：

        ```python
        handler = FileCallbackHandler("output.txt")
        # File remains open until handler is garbage collected
        try:
            chain.invoke(inputs, config={"callbacks": [handler]})
        finally:
            handler.close()  # Explicit cleanup recommended
        ```

    Args:
        filename: 要写入的文件路径。
        mode: 文件打开模式。默认为 `'a'`（追加）。
        color: 文本输出的默认颜色。

    !!! note

        若未作为上下文管理器使用，首次使用时将发出弃用警告。文件将在 `__init__`
        中立即打开，并在 `__del__` 或显式调用 `close()` 时关闭。

    """

    def __init__(
        self, filename: str, mode: str = "a", color: str | None = None
    ) -> None:
        """初始化文件回调处理器。

        Args:
            filename: 输出文件路径。
            mode: 文件打开模式（例如 `'w'`, `'a'`, `'x'`）。默认为 `'a'`。
            color: 输出的默认文本颜色。

        """
        self.filename = filename
        self.mode = mode
        self.color = color
        self._file_opened_in_context = False
        self.file: TextIO = cast(
            "TextIO",
                        # 以指定模式打开文件，并使用 UTF-8 编码。
            Path(self.filename).open(self.mode, encoding="utf-8"),  # noqa: SIM115
        )

    def __enter__(self) -> Self:
        """进入上下文管理器。

        Returns:
            返回 `FileCallbackHandler` 实例。

        !!! note

            文件已在 `__init__` 中打开，因此此方法仅标记处理器正在作为上下文管理器使用。

        """
        self._file_opened_in_context = True
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        """退出上下文管理器并关闭文件。

        Args:
            exc_type: 若发生异常，则为异常类型。
            exc_val: 若发生异常，则为异常值。
            exc_tb: 若发生异常，则为异常回溯。

        """
        self.close()

    def __del__(self) -> None:
        """析构函数，用于完成后的清理工作。"""
        self.close()

    def close(self) -> None:
        """若文件处于打开状态则关闭它。

        This method is safe to call multiple times and will only close
        the file if it's currently open.

        """
        if hasattr(self, "file") and self.file and not self.file.closed:
            self.file.close()

    def _write(
        self,
        text: str,
        color: str | None = None,
        end: str = "",
    ) -> None:
        """向文件写入文本，必要时附带弃用警告。

        Args:
            text: 要写入文件的文本。
            color: 可选的颜色。默认为 `self.color`。
            end: 在文本后附加的字符串。

        Raises:
            RuntimeError: 若文件已关闭或不可用。

        """
        global _GLOBAL_DEPRECATION_WARNED  # noqa: PLW0603
        if not self._file_opened_in_context and not _GLOBAL_DEPRECATION_WARNED:
            warn_deprecated(
                since="0.3.67",
                pending=True,
                message=(
                    "Using FileCallbackHandler without a context manager is "
                    "deprecated. Use 'with FileCallbackHandler(...) as "
                    "handler:' instead."
                ),
            )
            _GLOBAL_DEPRECATION_WARNED = True

        if not hasattr(self, "file") or self.file is None or self.file.closed:
            msg = "File is not open. Use FileCallbackHandler as a context manager."
            raise RuntimeError(msg)

        print_text(text, file=self.file, color=color, end=end)

    @override
    def on_chain_start(
        self, serialized: dict[str, Any], inputs: dict[str, Any], **kwargs: Any
    ) -> None:
        """打印即将进入链的消息。

        Args:
            serialized: 序列化的链信息。
            inputs: 链的输入。
            **kwargs: 可能包含 `'name'` 的额外关键字参数。

        """
        name = (
            kwargs.get("name")
            or serialized.get("name", serialized.get("id", ["<unknown>"])[-1])
            or "<unknown>"
        )
        self._write(f"\n\n> Entering new {name} chain...", end="\n")

    @override
    def on_chain_end(self, outputs: dict[str, Any], **kwargs: Any) -> None:
        """打印链已完成的消息。

        Args:
            outputs: 链的输出。
            **kwargs: 额外关键字参数。

        """
        self._write("\n> Finished chain.", end="\n")

    @override
    def on_agent_action(
        self, action: AgentAction, color: str | None = None, **kwargs: Any
    ) -> Any:
        """通过写入操作日志来处理智能体动作。

        Args:
            action: The agent action containing the log to write.
            color: Color override for this specific output.

                若为 `None`，则使用 `self.color`。

        """
        self._write(action.log, color=color or self.color)

    @override
    def on_tool_end(
        self,
        output: str,
        color: str | None = None,
        observation_prefix: str | None = None,
        llm_prefix: str | None = None,
        **kwargs: Any,
    ) -> None:
        """处理工具结束时，写入输出并可选添加前缀。

        Args:
            output: The tool output to write.
            color: Color override for this specific output.

                若为 `None`，则使用 `self.color`。 observation_prefix:
                可选的前缀，写在输出之前。 llm_prefix: 可选的前缀，写在输出之后。

        """
        if observation_prefix is not None:
            self._write(f"\n{observation_prefix}")
        self._write(output)
        if llm_prefix is not None:
            self._write(f"\n{llm_prefix}")

    @override
    def on_text(
        self, text: str, color: str | None = None, end: str = "", **kwargs: Any
    ) -> None:
        """处理文本输出。

        Args:
            text: 要写入的文本。
            color: 此特定输出的颜色覆盖。

                如果为 `None`，则使用 `self.color`。
            end: 在文本后附加的字符串。
            **kwargs: 额外的关键字参数。

        """
        self._write(text, color=color or self.color, end=end)

    @override
    def on_agent_finish(
        self, finish: AgentFinish, color: str | None = None, **kwargs: Any
    ) -> None:
        """通过写入完成日志来处理智能体完成事件。

        Args:
            finish: 包含要写入日志的智能体完成对象。
            color: 此特定输出的颜色覆盖。

                如果为 `None`，则使用 `self.color`。
            **kwargs: 额外的关键字参数。

        """
        self._write(finish.log, color=color or self.color, end="\n")

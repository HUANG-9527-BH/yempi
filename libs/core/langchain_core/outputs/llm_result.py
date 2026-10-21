"""``LLMResult`` 类。"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Literal

from pydantic import BaseModel

from langchain_core.outputs.chat_generation import ChatGeneration, ChatGenerationChunk
from langchain_core.outputs.generation import Generation, GenerationChunk
from langchain_core.outputs.run_info import RunInfo


class LLMResult(BaseModel):
    """用于存储 LLM 调用结果的容器。

    无论是聊天模型还是 LLM，都会生成一个 ``LLMResult``
    对象。该对象包含生成的输出以及模型提供商希望返回的任何附加信息。
    """

    generations: list[
        list[Generation | ChatGeneration | GenerationChunk | ChatGenerationChunk]
    ]
    """生成的输出。

    列表的第一个维度代表不同输入提示的补全结果。

    The second dimension of the list represents different candidate generations for a
    given prompt.

    - 当从 **LLM** 返回时，类型为 ``list[list[Generation]]``。
    - 当从 **聊天模型** 返回时，类型为 ``list[list[ChatGeneration]]``。

    `ChatGeneration` is a subclass of `Generation` that has a field for a structured
    chat message.
    """

    llm_output: dict[str, Any] | None = None
    """用于任意模型提供商特定的输出。

    此字典是一个自由格式的字典，可包含提供商希望返回的任何信息。它未标准化，键可能随提供商和时间变化。

    Users should generally avoid relying on this field and instead rely on accessing
    relevant information from standardized fields present in AIMessage.
    """

    run: list[RunInfo] | None = None
    """每个输入对应的模型调用元数据信息列表。

    详情请参见 ``langchain_core.outputs.run_info.RunInfo``。
    """

    type: Literal["LLMResult"] = "LLMResult"
    """仅用于序列化目的的类型。"""

    def flatten(self) -> list[LLMResult]:
        """将生成结果展平为单个列表。

        Unpack `list[list[Generation]] -> list[LLMResult]` where each returned
        `LLMResult` contains only a single `Generation`. If token usage information is
        available, it is kept only for the `LLMResult` corresponding to the top-choice
        `Generation`, to avoid over-counting of token usage downstream.

        Returns:
            包含多个 ``LLMResult`` 对象的列表，每个返回的 ``LLMResult`` 均包含一个
                ``Generation``。
        """
        llm_results = []
        for i, gen_list in enumerate(self.generations):
                        # 避免在 OpenAICallback 中重复计算 token
            if i == 0:
                llm_results.append(
                    LLMResult(
                        generations=[gen_list],
                        llm_output=self.llm_output,
                    )
                )
            else:
                if self.llm_output is not None:
                    llm_output = deepcopy(self.llm_output)
                    llm_output["token_usage"] = {}
                else:
                    llm_output = None
                llm_results.append(
                    LLMResult(
                        generations=[gen_list],
                        llm_output=llm_output,
                    )
                )
        return llm_results

    def __eq__(self, other: object) -> bool:
        """通过忽略与运行相关的任何元数据来检查 ``LLMResult`` 的相等性。

        Args:
            other: 另一个要比较的 ``LLMResult`` 对象。

        Returns:
            如果生成结果和 ``llm_output`` 相等则返回 ``True``，否则返回 ``False``。
        """
        if not isinstance(other, LLMResult):
            return NotImplemented
        return (
            self.generations == other.generations
            and self.llm_output == other.llm_output
        )

    __hash__ = None  # type: ignore[assignment]

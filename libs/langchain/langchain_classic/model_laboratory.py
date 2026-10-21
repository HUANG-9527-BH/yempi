"""尝试不同的模型。"""

from __future__ import annotations

from collections.abc import Sequence

from langchain_core.language_models.llms import BaseLLM
from langchain_core.prompts.prompt import PromptTemplate
from langchain_core.utils.input import get_color_mapping, print_text

from langchain_classic.chains.base import Chain
from langchain_classic.chains.llm import LLMChain


class ModelLaboratory:
    """一个用于实验和比较不同模型性能的工具。"""

    def __init__(self, chains: Sequence[Chain], names: list[str] | None = None):
        """使用要实验的链初始化 ModelLaboratory。

        Args:
            chains: 要实验的链的序列。
                每个链必须恰好有一个输入变量和一个输出变量。
            names: 对应每个链的可选名称列表。
                如果提供，则其长度必须与链的数量匹配。


        Raises:
            ValueError: 如果任何链不是 `Chain` 的实例。
            ValueError: 如果链没有恰好一个输入变量。
            ValueError: 如果链没有恰好一个输出变量。
            ValueError: 如果 `names` 的长度不等于链的数量。
        """
        for chain in chains:
            if not isinstance(chain, Chain):
                msg = (  # type: ignore[unreachable]
                    "ModelLaboratory should now be initialized with Chains. "
                    "If you want to initialize with LLMs, use the `from_llms` method "
                    "instead (`ModelLaboratory.from_llms(...)`)"
                )
                raise ValueError(msg)  # noqa: TRY004
            if len(chain.input_keys) != 1:
                msg = (
                    "Currently only support chains with one input variable, "
                    f"got {chain.input_keys}"
                )
                raise ValueError(msg)
            if len(chain.output_keys) != 1:
                msg = (
                    "Currently only support chains with one output variable, "
                    f"got {chain.output_keys}"
                )
        if names is not None and len(names) != len(chains):
            msg = "Length of chains does not match length of names."
            raise ValueError(msg)
        self.chains = chains
        chain_range = [str(i) for i in range(len(self.chains))]
        self.chain_colors = get_color_mapping(chain_range)
        self.names = names

    @classmethod
    def from_llms(
        cls,
        llms: list[BaseLLM],
        prompt: PromptTemplate | None = None,
    ) -> ModelLaboratory:
        """使用 LLMs 和可选提示词初始化 ModelLaboratory。

        Args:
            llms: 要实验的 LLMs 列表。
            prompt: 可选的提示词，用于与 LLMs 一起使用。
                如果提供，则提示词必须恰好包含一个输入变量。

        Returns:
            使用 LLMs 初始化的 `ModelLaboratory` 实例。
        """
        if prompt is None:
            prompt = PromptTemplate(input_variables=["_input"], template="{_input}")
        chains = [LLMChain(llm=llm, prompt=prompt) for llm in llms]
        names = [str(llm) for llm in llms]
        return cls(chains, names=names)

    def compare(self, text: str) -> None:
        """在输入文本上比较模型输出。

        If a prompt was provided with starting the laboratory, then this text will be
        fed into the prompt. If no prompt was provided, then the input text is the
        entire prompt.

        Args:
            text: 要在所有模型上运行的输入文本。
        """
        print(f"\033[1mInput:\033[0m\n{text}\n")  # noqa: T201
        for i, chain in enumerate(self.chains):
            name = self.names[i] if self.names is not None else str(chain)
            print_text(name, end="\n")
            output = chain.run(text)
            print_text(output, color=self.chain_colors[str(i)], end="\n\n")

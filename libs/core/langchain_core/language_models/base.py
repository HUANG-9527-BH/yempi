"""基础语言模型类。"""

from __future__ import annotations

import builtins  # noqa: TC003  # runtime-evaluated; subclass `dict()` shadows the builtin
import warnings
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
from functools import cache
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    TypeAlias,
    TypeVar,
    cast,
)

from pydantic import BaseModel, ConfigDict, Field, field_validator
from typing_extensions import TypedDict, override

from langchain_core.caches import BaseCache  # noqa: TC001
from langchain_core.callbacks import Callbacks  # noqa: TC001
from langchain_core.globals import get_verbose
from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    BaseMessage,
    MessageLikeRepresentation,
    get_buffer_string,
)
from langchain_core.prompt_values import (
    ChatPromptValueConcrete,
    PromptValue,
    StringPromptValue,
)
from langchain_core.runnables import Runnable, RunnableSerializable

if TYPE_CHECKING:
    from langchain_core.outputs import LLMResult


class LangSmithParams(TypedDict, total=False):
    """LangSmith 追踪参数。"""

    ls_provider: str
    """模型提供方。"""

    ls_model_name: str
    """模型名称。"""

    ls_model_type: Literal["chat", "llm"]
    """模型类型。

    应为 `'chat'` 或 `'llm'`。
    """

    ls_temperature: float | None
    """生成时的温度。"""

    ls_max_tokens: int | None
    """生成的最大 token 数。"""

    ls_stop: list[str] | None
    """生成时的停止词。"""
    ls_integration: str
    """创建追踪的集成。"""


@cache
def _get_tokenizer_module() -> Callable[[str], Any] | None:
    """获取 `GPT2TokenizerFast` 中的 `from_pretrained` 函数。

    懒加载，因此仅导入此模块不会引入
    `transformers`（及其间接依赖的 `pytorch`）。缓存后，成本仅支付一次。

    Returns:
        `GPT2TokenizerFast.from_pretrained` 函数，若未安装 `transformers` 则为 `None`。

    """
    try:
        from transformers import (  # type: ignore[import-not-found] # noqa: PLC0415
            GPT2TokenizerFast,
        )
    except ImportError:
        return None
    return cast("Callable[[str], Any]", GPT2TokenizerFast.from_pretrained)


@cache  # 缓存分词器
def get_tokenizer() -> Any:
    """获取一个 GPT-2 分词器实例。

    此函数已缓存，以避免每次调用时重新加载分词器。

    Raises:
        ImportError: 若未安装 transformers 包。

    Returns:
        GPT-2 分词器实例。

    """
    loader_impl = _get_tokenizer_module()
    if loader_impl is None:
        msg = (
            "Could not import transformers python package. "
            "This is needed in order to calculate get_token_ids. "
            "Please install it with `pip install transformers`."
        )
        raise ImportError(msg)
        # 创建一个 GPT-2 分词器实例
    return loader_impl("gpt2")


_GPT2_TOKENIZER_WARNED = False


def _get_token_ids_default_method(text: str) -> list[int]:
    """使用备用 GPT-2 分词器将文本编码为 token ID。"""
    global _GPT2_TOKENIZER_WARNED  # noqa: PLW0603
    if not _GPT2_TOKENIZER_WARNED:
        warnings.warn(
            "Using fallback GPT-2 tokenizer for token counting. "
            "Token counts may be inaccurate for non-GPT-2 models. "
            "For accurate counts, use a model-specific method if available.",
            stacklevel=3,
        )
        _GPT2_TOKENIZER_WARNED = True

    tokenizer = get_tokenizer()

        # 传入 verbose=False 以抑制 HuggingFace 的 "Token indices sequence length is longer
    # than" 警告。该警告关于 GPT-2 的 1024 token
    # 上下文限制，但我们仅使用分词器进行计数，而非模型输入。
        # 该警告关于 GPT-2 的 1024 token
    # 上下文限制，但我们仅使用分词器进行计数，而非模型输入。
        # 该警告关于 GPT-2 的 1024 token
    # 上下文限制，但我们仅使用分词器进行计数，而非模型输入。
        # 在导入时预热缓存，此时我们保证处于同步导入路径且不在任何事件循环内。否则，在异步代码中首次构建模型时，会在线程事件循环中运行阻塞的
    # `os.stat`（通过 `importlib.metadata`），触发如 blockbuster 这类阻塞 I/O 检测器。
    return cast("list[int]", tokenizer.encode(text, verbose=False))


LanguageModelInput = PromptValue | str | Sequence[MessageLikeRepresentation]
"""Input to a language model."""

LanguageModelOutput = BaseMessage | str
"""Output from a language model."""

LanguageModelLike = Runnable[LanguageModelInput, LanguageModelOutput]
"""Input/output interface for a language model."""

LanguageModelOutputVar = TypeVar("LanguageModelOutputVar", AIMessage, str)
"""Type variable for the output of a language model."""


def _get_verbosity() -> bool:
    return get_verbose()


@cache
def _get_langchain_version() -> str | None:
    """返回已安装的 `langchain` 版本，若未安装则返回 `None`。

    因为 `importlib.metadata.version` 执行文件系统查找， 且 `model_post_init` 在每个
    `BaseLanguageModel` 实例化时都会运行。`langchain`
    是可选的同级包，因此其缺失是预期行为，不视为错误。
    """
    from importlib.metadata import PackageNotFoundError  # noqa: PLC0415
    from importlib.metadata import version as pkg_version  # noqa: PLC0415

    try:
        return pkg_version("langchain")
    except PackageNotFoundError:
        return None


# 这是 LanguageModelInput 的一个版本，它将抽象基类 BaseMessage
# 替换为子类的联合类型，使模式更加完善。
# 如果存在一种方式引导模型生成符合给定模式的响应，则应在子类中实现此方法。
# 如果存在一种方式引导模型生成符合给定模式的响应，则应在子类中实现此方法。
# 如果存在一种方式引导模型生成符合给定模式的响应，则应在子类中实现此方法。
_get_langchain_version()


class BaseLanguageModel(
    RunnableSerializable[LanguageModelInput, LanguageModelOutputVar], ABC
):
    """用于与语言模型交互的抽象基类。

    所有继承自 `BaseLanguageModel` 的语言模型包装器。

    """

    cache: BaseCache | bool | None = Field(default=None, exclude=True)
    """是否缓存响应。

    * 若为 `True`，将使用全局缓存。
    * 若为 `False`，将不使用缓存。
    * 若为 `None`，若设置了全局缓存则使用，否则不使用缓存。
    * 若为 `BaseCache` 实例，则使用提供的缓存。

    当前不支持模型流式方法的缓存。
    """

    verbose: bool = Field(default_factory=_get_verbosity, exclude=True, repr=False)
    """是否打印响应文本。"""

    callbacks: Callbacks = Field(default=None, exclude=True)
    """要添加到运行追踪中的回调。"""

    tags: list[str] | None = Field(default=None, exclude=True)
    """要添加到运行追踪中的标签。"""

    metadata: builtins.dict[str, Any] | None = Field(default=None, exclude=True)
    """要添加到运行追踪中的元数据。"""

    custom_get_token_ids: Callable[[str], list[int]] | None = Field(
        default=None, exclude=True
    )
    """可选的用于统计 token 的编码器。"""

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )

    def model_post_init(self, _context: Any, /) -> None:
        """Pydantic V2 生命周期钩子，在 `__init__` 后自动调用。

        将已安装的 `langchain-core`
        （以及已安装的 `langchain`）版本填充至 `metadata["lc_versions"]`，
        以便每个 LLM 追踪都携带生成它的包版本信息。

        Partner packages should **not** override this method. Instead, they
        should define a `@model_validator(mode="after")` that calls
        `_add_version` to append their own version to the same dict.

        !!! warning "验证器命名"

            每个子类的验证器 **必须** 命名唯一。Pydantic 会替换而非链式调用子类中同名的
            `model_validator` 方法。 例如，`BaseChatOpenAI` 子类应使用
            `_set_<partner>_version`，而非 `_set_version`，以避免静默丢弃父类条目。

        Args:
            _context: Pydantic 验证上下文（通常为 `None`）。
        """
        super().model_post_init(_context)
        from langchain_core.version import VERSION  # noqa: PLC0415

        self._add_version("langchain-core", VERSION)

        langchain_version = _get_langchain_version()
        if langchain_version is not None:
            self._add_version("langchain", langchain_version)

    def _add_version(self, pkg: str, version: str) -> None:
        """在 `metadata.lc_versions` 中记录一个包版本以用于追踪。

        Each layer in the class hierarchy (core -> langchain -> partner)
        calls this so that the resulting metadata dict accumulates *all*
        package versions involved in an invocation.

        示例生成的元数据：

        ```python
        {
            "lc_versions": {
                "langchain-core": "1.x.x",
                "langchain": "1.x.x",
                "langchain-openai": "1.x.x",
            }
        }
        ```

        Args:
            pkg: 包名称（例如 `'langchain-openai'`）。\nversion: 安装的版本字符串。
        """
        if self.metadata is None:
            self.metadata = {}
        existing = self.metadata.get("lc_versions")
        if existing is not None and not isinstance(existing, Mapping):
            warnings.warn(
                f"metadata['lc_versions'] expected a dict, got "
                f"{type(existing).__name__}; overwriting with package version dict",
                stacklevel=2,
            )
            existing = None
        self.metadata["lc_versions"] = {
            **(existing if isinstance(existing, Mapping) else {}),
            pkg: version,
        }

    @field_validator("verbose", mode="before")
    def set_verbose(cls, verbose: bool | None) -> bool:  # noqa: FBT001
        """如果 verbose 为 `None`，则设置它。

        这允许用户传入 `None` 以访问全局设置。

        Args:
            verbose: 要使用的详细程度设置。

        Returns:
            要使用的详细程度设置。

        """
        if verbose is None:
            return _get_verbosity()
        return verbose

    @property
    @override
    def InputType(self) -> TypeAlias:
        """获取此 `Runnable` 的输入类型。"""
                # 这是 LanguageModelInput 的一个版本，它将抽象基类 BaseMessage
        # 替换为子类的联合类型，使模式更加完善。
                # 这是 LanguageModelInput 的一个版本，它将抽象基类 BaseMessage
        # 替换为子类的联合类型，使模式更加完善。
                # 这是 LanguageModelInput 的一个版本，它将抽象基类 BaseMessage
        # 替换为子类的联合类型，使模式更加完善。
        return str | StringPromptValue | ChatPromptValueConcrete | list[AnyMessage]

    @abstractmethod
    def generate_prompt(
        self,
        prompts: list[PromptValue],
        stop: list[str] | None = None,
        callbacks: Callbacks = None,
        **kwargs: Any,
    ) -> LLMResult:
        """向模型传递一系列提示词并返回模型生成结果。

        This method should make use of batched calls for models that expose a batched
        API.

        在以下情况下使用此方法：

        1. Take advantage of batched calls,
        2. Need more output from the model than just the top generated value,
        3. Are building chains that are agnostic to the underlying language model
            type (e.g., pure text completion models vs chat models).

        Args:
            prompts: `PromptValue` 对象列表。

                A `PromptValue` is an object that can be converted to match the format
                of any language model (string for pure text generation models and
                `BaseMessage` objects for chat models).
            stop: Stop words to use when generating.

                模型输出在首次出现这些子串之一时被截断。\ncallbacks: 要传递的
                `Callbacks`。

                Used for executing additional functionality, such as logging or
                streaming, throughout generation.
            **kwargs: Arbitrary additional keyword arguments.

                这些参数通常传递给模型提供方的 API 调用。

        Returns:
            一个 `LLMResult`，包含每个输入提示词的候选 `Generation`
            对象列表以及额外的模型提供方特定输出。

        """

    @abstractmethod
    async def agenerate_prompt(
        self,
        prompts: list[PromptValue],
        stop: list[str] | None = None,
        callbacks: Callbacks = None,
        **kwargs: Any,
    ) -> LLMResult:
        """异步传递一系列提示词并返回模型生成结果。

        This method should make use of batched calls for models that expose a batched
        API.

        在以下情况下使用此方法：

        1. Take advantage of batched calls,
        2. Need more output from the model than just the top generated value,
        3. Are building chains that are agnostic to the underlying language model
            type (e.g., pure text completion models vs chat models).

        Args:
            prompts: `PromptValue` 对象列表。

                A `PromptValue` is an object that can be converted to match the format
                of any language model (string for pure text generation models and
                `BaseMessage` objects for chat models).
            stop: Stop words to use when generating.

                模型输出在首次出现这些子串之一时被截断。\ncallbacks: 要传递的
                `Callbacks`。

                Used for executing additional functionality, such as logging or
                streaming, throughout generation.
            **kwargs: Arbitrary additional keyword arguments.

                这些参数通常传递给模型提供商的 API 调用。

        Returns:
            一个 `LLMResult`，包含每个输入提示词的候选 `Generation`
            对象列表以及额外的模型提供商特定输出。

        """

    def with_structured_output(
        self, schema: dict[str, Any] | type, **kwargs: Any
    ) -> Runnable[LanguageModelInput, dict[str, Any] | BaseModel]:
        """此方法在该类中未实现。"""
                # 如果存在一种方式引导模型生成符合给定模式的响应，则应在子类中实现此方法。
                # 如果存在一种方式引导模型生成符合给定模式的响应，则应在子类中实现此方法。
        raise NotImplementedError

    def _get_ls_params(
        self,
        stop: list[str] | None = None,  # noqa: ARG002
        **kwargs: Any,  # noqa: ARG002
    ) -> LangSmithParams:
        """获取追踪的标准参数。"""
        return LangSmithParams()

    def _get_ls_params_with_defaults(
        self,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> LangSmithParams:
        """包装 _get_ls_params 以包含任何额外的默认参数。"""
        return self._get_ls_params(stop=stop, **kwargs)

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """获取标识参数。"""
        return self.lc_attributes

    def get_token_ids(self, text: str) -> list[int]:
        """返回文本中 token 的有序 ID 列表。

        Args:
            text: 需要分词的字符串输入。

        Returns:
            与文本中 token 顺序对应的 ID 列表。
        """
        if self.custom_get_token_ids is not None:
            return self.custom_get_token_ids(text)
        return _get_token_ids_default_method(text)

    def get_num_tokens(self, text: str) -> int:
        """获取文本中 token 的数量。

        用于检查输入是否适合模型的上下文窗口。

        应由模型特定的实现覆盖，以通过模型特定的分词器提供准确的 token 计数。

        Args:
            text: 需要分词的字符串输入。

        Returns:
            文本中的 token 数量（整数）。

        """
        return len(self.get_token_ids(text))

    def get_num_tokens_from_messages(
        self,
        messages: list[BaseMessage],
        tools: Sequence[Any] | None = None,
    ) -> int:
        """获取消息中 token 的数量。

        用于检查输入是否适合模型的上下文窗口。

        应由模型特定的实现覆盖，以通过模型特定的分词器提供准确的 token 计数。

        !!! note

            * 基类的 `get_num_tokens_from_messages` 实现忽略 tool 模式。 * 基类的
            `get_num_tokens_from_messages`
            实现为表示用户角色的消息添加额外前缀，这会增加总体 token
            数量。模型特定的实现可选择不同处理方式。

        Args:
            messages: 需要分词的消息输入。 tools: 如果提供，为
            dict、`BaseModel`、函数或 `BaseTool` 对象的序列，将被转换为 tool 模式。

        Returns:
            所有消息中 token 数量的总和。

        """
        if tools is not None:
            warnings.warn(
                "Counting tokens in tool schemas is not yet supported. Ignoring tools.",
                stacklevel=2,
            )
        return sum(self.get_num_tokens(get_buffer_string([m])) for m in messages)

"""失败时重试 `Runnable` 的 `Runnable`。"""

from collections.abc import Callable
from typing import (
    TYPE_CHECKING,
    Any,
    TypeVar,
    cast,
)

from tenacity import (
    AsyncRetrying,
    RetryCallState,
    RetryError,
    Retrying,
    retry_if_exception,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)
from typing_extensions import TypedDict, override

from langchain_core.runnables.base import RunnableBindingBase
from langchain_core.runnables.config import RunnableConfig, patch_config
from langchain_core.runnables.utils import Input, Output

if TYPE_CHECKING:
    from langchain_core.callbacks.manager import (
        AsyncCallbackManagerForChainRun,
        CallbackManagerForChainRun,
    )

    T = TypeVar("T", CallbackManagerForChainRun, AsyncCallbackManagerForChainRun)
U = TypeVar("U")


class ExponentialJitterParams(TypedDict, total=False):
    """`tenacity.wait_exponential_jitter` 的参数。"""

    initial: float
    """初始等待时间。"""
    max: float
    """最大等待时间。"""
    exp_base: float
    """指数退避的基数。"""
    jitter: float
    """从 `random.uniform(0, jitter)` 中采样的随机额外等待时间。"""


class RunnableRetry(RunnableBindingBase[Input, Output]):  # type: ignore[no-redef]
    """如果 `Runnable` 失败，则重试它。

    `RunnableRetry` 可用于为任何继承自基础 `Runnable` 的对象添加重试逻辑。

    Such retries are especially useful for network calls that may fail
    due to transient errors.

    `RunnableRetry` 作为 `RunnableBinding` 实现。最简单的使用方式是通过所有 `Runnable`
    上的 `.with_retry()` 方法。

    Example:
    以下示例使用 `RunnableLambda` 抛出异常

        ```python
        import time


        def foo(input) -> None:
            '''Fake function that raises an exception.'''
            raise ValueError(f"Invoking foo failed. At time {time.time()}")


        runnable = RunnableLambda(foo)

        runnable_with_retries = runnable.with_retry(
            retry_if_exception_type=(ValueError,),  # Retry only on ValueError
            wait_exponential_jitter=True,  # Add jitter to the exponential backoff
            stop_after_attempt=2,  # Try twice
            exponential_jitter_params={"initial": 2},  # if desired, customize backoff
        )

        # The method invocation above is equivalent to the longer form below:

        runnable_with_retries = RunnableRetry(
            bound=runnable,
            retry_exception_types=(ValueError,),
            max_attempt_number=2,
            wait_exponential_jitter=True,
            exponential_jitter_params={"initial": 2},
        )
        ```

    This logic can be used to retry any Runnable, including a chain of Runnables,
    but in general it's best practice to keep the scope of the retry as small as
    possible. For example, if you have a chain of Runnables, you should only retry
    the Runnable that is likely to fail, not the entire chain.

    Example:
        ```python
        from langchain_openai import ChatOpenAI
        from langchain_core.prompts import PromptTemplate

        template = PromptTemplate.from_template("tell me a joke about {topic}.")
        model = ChatOpenAI(temperature=0.5)

        # Good
        chain = template | model.with_retry()

        # Bad
        chain = template | model
        retryable_chain = chain.with_retry()
        ```
    """

    retry_exception_types: (
        tuple[type[BaseException], ...] | Callable[[BaseException], bool]
    ) = (Exception,)
    """The exception types to retry on, or a callable that receives the raised
    exception and returns whether to retry it. By default all exceptions are retried.

    In general you should only retry on exceptions that are likely to be
    transient, such as network errors.

    Good exceptions to retry are all server errors (5xx) and selected client
    errors (4xx) such as 429 Too Many Requests.
    """

    wait_exponential_jitter: bool = True
    """是否在指数退避中加入抖动。"""

    exponential_jitter_params: ExponentialJitterParams | None = None
    """Parameters for `tenacity.wait_exponential_jitter`. Namely: `initial`,
    `max`, `exp_base`, and `jitter` (all `float` values).
    """

    max_attempt_number: int = 3
    """重试 `Runnable` 的最大尝试次数。"""

    @property
    def _kwargs_retrying(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {}

        if self.max_attempt_number:
            kwargs["stop"] = stop_after_attempt(self.max_attempt_number)

        if self.wait_exponential_jitter:
            kwargs["wait"] = wait_exponential_jitter(
                **(self.exponential_jitter_params or {})
            )

        if callable(self.retry_exception_types):
            kwargs["retry"] = retry_if_exception(self.retry_exception_types)
        elif self.retry_exception_types:
            kwargs["retry"] = retry_if_exception_type(self.retry_exception_types)

        return kwargs

    def _sync_retrying(self, **kwargs: Any) -> Retrying:
        return Retrying(**self._kwargs_retrying, **kwargs)

    def _async_retrying(self, **kwargs: Any) -> AsyncRetrying:
        return AsyncRetrying(**self._kwargs_retrying, **kwargs)

    @staticmethod
    def _patch_config(
        config: RunnableConfig,
        run_manager: "T",
        retry_state: RetryCallState,
    ) -> RunnableConfig:
        attempt = retry_state.attempt_number
        tag = f"retry:attempt:{attempt}" if attempt > 1 else None
        return patch_config(config, callbacks=run_manager.get_child(tag))

    def _patch_config_list(
        self,
        config: list[RunnableConfig],
        run_manager: list["T"],
        retry_state: RetryCallState,
    ) -> list[RunnableConfig]:
        return [
            self._patch_config(c, rm, retry_state)
            for c, rm in zip(config, run_manager, strict=False)
        ]

    def _invoke(
        self,
        input_: Input,
        run_manager: "CallbackManagerForChainRun",
        config: RunnableConfig,
        **kwargs: Any,
    ) -> Output:
        for attempt in self._sync_retrying(reraise=True):
            with attempt:
                result = super().invoke(
                    input_,
                    self._patch_config(config, run_manager, attempt.retry_state),
                    **kwargs,
                )
            if attempt.retry_state.outcome and not attempt.retry_state.outcome.failed:
                attempt.retry_state.set_result(result)
        return result

    @override
    def invoke(
        self, input: Input, config: RunnableConfig | None = None, **kwargs: Any
    ) -> Output:
        return self._call_with_config(self._invoke, input, config, **kwargs)

    async def _ainvoke(
        self,
        input_: Input,
        run_manager: "AsyncCallbackManagerForChainRun",
        config: RunnableConfig,
        **kwargs: Any,
    ) -> Output:
        async for attempt in self._async_retrying(reraise=True):
            with attempt:
                result = await super().ainvoke(
                    input_,
                    self._patch_config(config, run_manager, attempt.retry_state),
                    **kwargs,
                )
            if attempt.retry_state.outcome and not attempt.retry_state.outcome.failed:
                attempt.retry_state.set_result(result)
        return result

    @override
    async def ainvoke(
        self, input: Input, config: RunnableConfig | None = None, **kwargs: Any
    ) -> Output:
        return await self._acall_with_config(self._ainvoke, input, config, **kwargs)

    def _batch(
        self,
        inputs: list[Input],
        run_manager: list["CallbackManagerForChainRun"],
        config: list[RunnableConfig],
        **kwargs: Any,
    ) -> list[Output | Exception]:
        results_map: dict[int, Output] = {}

        not_set: list[Output] = []
        result = not_set
        try:
            for attempt in self._sync_retrying():
                with attempt:
                                        # 重试尚未成功的输入
                                        # 确定仍保留的原始索引
                    remaining_indices = [
                        i for i in range(len(inputs)) if i not in results_map
                    ]
                    if not remaining_indices:
                        break
                    pending_inputs = [inputs[i] for i in remaining_indices]
                    pending_configs = [config[i] for i in remaining_indices]
                    pending_run_managers = [run_manager[i] for i in remaining_indices]
                                        # 仅对剩余元素调用底层批处理
                    result = super().batch(
                        pending_inputs,
                        self._patch_config_list(
                            pending_configs, pending_run_managers, attempt.retry_state
                        ),
                        return_exceptions=True,
                        **kwargs,
                    )
                                        # 注册已成功输入的结果，并映射回其原始索引
                                        # 若发生任何异常，则抛出，以重试失败项
                    first_exception = None
                    for offset, r in enumerate(result):
                        if isinstance(r, Exception):
                            if not first_exception:
                                first_exception = r
                            continue
                        orig_idx = remaining_indices[offset]
                        results_map[orig_idx] = r
                                        # 重试尚未成功的输入
                    if first_exception:
                        raise first_exception
                if (
                    attempt.retry_state.outcome
                    and not attempt.retry_state.outcome.failed
                ):
                    attempt.retry_state.set_result(result)
        except RetryError as e:
            if result is not_set:
                result = cast("list[Output]", [e] * len(inputs))

        outputs: list[Output | Exception] = []
        for idx in range(len(inputs)):
            if idx in results_map:
                outputs.append(results_map[idx])
            else:
                outputs.append(result.pop(0))
        return outputs

    @override
    def batch(
        self,
        inputs: list[Input],
        config: RunnableConfig | list[RunnableConfig] | None = None,
        *,
        return_exceptions: bool = False,
        **kwargs: Any,
    ) -> list[Output]:
        return self._batch_with_config(
            self._batch, inputs, config, return_exceptions=return_exceptions, **kwargs
        )

    async def _abatch(
        self,
        inputs: list[Input],
        run_manager: list["AsyncCallbackManagerForChainRun"],
        config: list[RunnableConfig],
        **kwargs: Any,
    ) -> list[Output | Exception]:
        results_map: dict[int, Output] = {}

        not_set: list[Output] = []
        result = not_set
        try:
            async for attempt in self._async_retrying():
                with attempt:
                                        # 确定仍保留的原始索引
                                        # 注册已成功输入的结果，并映射回其原始索引
                    remaining_indices = [
                        i for i in range(len(inputs)) if i not in results_map
                    ]
                    if not remaining_indices:
                        break
                    pending_inputs = [inputs[i] for i in remaining_indices]
                    pending_configs = [config[i] for i in remaining_indices]
                    pending_run_managers = [run_manager[i] for i in remaining_indices]
                    result = await super().abatch(
                        pending_inputs,
                        self._patch_config_list(
                            pending_configs, pending_run_managers, attempt.retry_state
                        ),
                        return_exceptions=True,
                        **kwargs,
                    )
                                        # 若发生任何异常，则抛出，以重试失败项
                                        # stream() 和 transform() 不会被重试，因为重试流式操作不直观。
                    first_exception = None
                    for offset, r in enumerate(result):
                        if isinstance(r, Exception):
                            if not first_exception:
                                first_exception = r
                            continue
                        orig_idx = remaining_indices[offset]
                        results_map[orig_idx] = r
                                        # 如果发生任何异常，请抛出该异常，以便重试失败的项
                    if first_exception:
                        raise first_exception
                if (
                    attempt.retry_state.outcome
                    and not attempt.retry_state.outcome.failed
                ):
                    attempt.retry_state.set_result(result)
        except RetryError as e:
            if result is not_set:
                result = cast("list[Output]", [e] * len(inputs))

        outputs: list[Output | Exception] = []
        for idx in range(len(inputs)):
            if idx in results_map:
                outputs.append(results_map[idx])
            else:
                outputs.append(result.pop(0))
        return outputs

    @override
    async def abatch(
        self,
        inputs: list[Input],
        config: RunnableConfig | list[RunnableConfig] | None = None,
        *,
        return_exceptions: bool = False,
        **kwargs: Any,
    ) -> list[Output]:
        return await self._abatch_with_config(
            self._abatch, inputs, config, return_exceptions=return_exceptions, **kwargs
        )

        # stream() 和 transform() 不会重试，因为流式操作重试存在困难
        # 不够直观。

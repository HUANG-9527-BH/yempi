"""Retriever 类根据文本查询返回 `Document` 对象。

It is more general than a vector store. A retriever does not need to be able to
store documents, only to return (or retrieve) it. Vector stores can be used as
the backbone of a retriever, but there are other types of retrievers as well.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from pydantic import ConfigDict
from typing_extensions import Self, TypedDict, override

from langchain_core.callbacks.manager import AsyncCallbackManager, CallbackManager
from langchain_core.documents import Document
from langchain_core.runnables import (
    Runnable,
    RunnableConfig,
    RunnableSerializable,
    ensure_config,
)
from langchain_core.runnables.config import run_in_executor
from langchain_core.utils._signature import _signature_parameters

if TYPE_CHECKING:
    from langchain_core.callbacks.manager import (
        AsyncCallbackManagerForRetrieverRun,
        CallbackManagerForRetrieverRun,
    )

RetrieverInput = str
RetrieverOutput = list[Document]
RetrieverLike = Runnable[RetrieverInput, RetrieverOutput]
RetrieverOutputLike = Runnable[Any, RetrieverOutput]


class LangSmithRetrieverParams(TypedDict, total=False):
    """LangSmith 用于追踪的参数。"""

    ls_retriever_name: str
    """检索器名称。"""

    ls_vector_store_provider: str | None
    """向量存储提供方。"""

    ls_embedding_provider: str | None
    """嵌入提供方。"""

    ls_embedding_model: str | None
    """嵌入模型。"""


class BaseRetriever(RunnableSerializable[RetrieverInput, RetrieverOutput], ABC):
    """文档检索系统的抽象基类。

    A retrieval system is defined as something that can take string queries and return
    the most 'relevant' documents from some source.

    Usage:

    检索器遵循标准的 `Runnable` 接口，应通过 `Runnable` 的标准方法
    `invoke`、`ainvoke`、`batch`、`abatch` 使用。

    实现：

    When implementing a custom retriever, the class should implement the
    `_get_relevant_documents` method to define the logic for retrieving documents.

    Optionally, an async native implementations can be provided by overriding the
    `_aget_relevant_documents` method.

    !!! example "返回文档列表中前 5 个文档的检索器"

        ```python
        from langchain_core.documents import Document
        from langchain_core.retrievers import BaseRetriever

        class SimpleRetriever(BaseRetriever):
            docs: list[Document]
            k: int = 5

            def _get_relevant_documents(self, query: str) -> list[Document]:
                \"\"\"Return the first k documents from the list of documents\"\"\"
                return self.docs[:self.k]

            async def _aget_relevant_documents(self, query: str) -> list[Document]:
                \"\"\"(Optional) async native implementation.\"\"\"
                return self.docs[:self.k]
        ```

    !!! example "基于 scikit-learn 向量化器的简单检索器"

        ```python
        from sklearn.metrics.pairwise import cosine_similarity


        class TFIDFRetriever(BaseRetriever, BaseModel):
            vectorizer: Any
            docs: list[Document]
            tfidf_array: Any
            k: int = 4

            class Config:
                arbitrary_types_allowed = True

            def _get_relevant_documents(self, query: str) -> list[Document]:
                # Ip -- (n_docs,x), Op -- (n_docs,n_Feats)
                query_vec = self.vectorizer.transform([query])
                # Op -- (n_docs,1) -- Cosine Sim with each doc
                results = cosine_similarity(self.tfidf_array, query_vec).reshape((-1,))
                return [self.docs[i] for i in results.argsort()[-self.k :][::-1]]
        ```
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )

    _new_arg_supported: bool = False

    _expects_other_args: bool = False

    tags: list[str] | None = None
    """与检索器关联的可选标签列表。

    These tags will be associated with each call to this retriever,
    and passed as arguments to the handlers defined in `callbacks`.

    You can use these to eg identify a specific instance of a retriever with its
    use case.
    """

    metadata: dict[str, Any] | None = None
    """与检索器关联的可选元数据。

    This metadata will be associated with each call to this retriever,
    and passed as arguments to the handlers defined in `callbacks`.

    You can use these to eg identify a specific instance of a retriever with its
    use case.
    """

    @override
    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        parameters = _signature_parameters(cls._get_relevant_documents)
        cls._new_arg_supported = parameters.get("run_manager") is not None
        if (
            not cls._new_arg_supported
            and cls._aget_relevant_documents == BaseRetriever._aget_relevant_documents
        ):
                        # 我们需要在 `_aget_relevant_documents` 签名中容忍没有 run_manager 的情况
            async def _aget_relevant_documents(
                self: Self, query: str
            ) -> list[Document]:
                return await run_in_executor(None, self._get_relevant_documents, query)  # type: ignore[call-arg]

            cls._aget_relevant_documents = _aget_relevant_documents  # type: ignore[assignment]

                # 如果 V1 检索器破坏了接口并期望额外参数
        cls._expects_other_args = (
            len(set(parameters.keys()) - {"self", "query", "run_manager"}) > 0
        )

    def _get_ls_params(self, **_kwargs: Any) -> LangSmithRetrieverParams:
        """获取追踪的标准参数。"""
        default_retriever_name = self.get_name()
        if default_retriever_name.startswith("Retriever"):
            default_retriever_name = default_retriever_name[9:]
        elif default_retriever_name.endswith("Retriever"):
            default_retriever_name = default_retriever_name[:-9]
        default_retriever_name = default_retriever_name.lower()

        return LangSmithRetrieverParams(ls_retriever_name=default_retriever_name)

    @override
    def invoke(
        self, input: str, config: RunnableConfig | None = None, **kwargs: Any
    ) -> list[Document]:
        """调用检索器以获取相关文档。

        同步检索器调用的主要入口点。

        Args:
            input: 查询字符串。
            config: 检索器的配置。
            **kwargs: 传递给检索器的额外参数。

        Returns:
            相关文档列表。

        Examples:
        ```python
        retriever.invoke("query")
        ```
        """
        config = ensure_config(config)
        inheritable_metadata = {
            **(config.get("metadata") or {}),
            **self._get_ls_params(**kwargs),
        }
        callback_manager = CallbackManager.configure(
            config.get("callbacks"),
            None,
            verbose=kwargs.get("verbose", False),
            inheritable_tags=config.get("tags"),
            local_tags=self.tags,
            inheritable_metadata=inheritable_metadata,
            local_metadata=self.metadata,
        )
        run_manager = callback_manager.on_retriever_start(
            None,
            input,
            name=config.get("run_name") or self.get_name(),
            run_id=kwargs.pop("run_id", None),
        )
        try:
            kwargs_ = kwargs if self._expects_other_args else {}
            if self._new_arg_supported:
                result = self._get_relevant_documents(
                    input, run_manager=run_manager, **kwargs_
                )
            else:
                result = self._get_relevant_documents(input, **kwargs_)
        except Exception as e:
            run_manager.on_retriever_error(e)
            raise
        else:
            run_manager.on_retriever_end(
                result,
            )
            return result

    @override
    async def ainvoke(
        self,
        input: str,
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> list[Document]:
        """异步调用检索器以获取相关文档。

        异步检索器调用的主要入口点。

        Args:
            input: 查询字符串。
            config: 检索器的配置。
            **kwargs: 传递给检索器的额外参数。

        Returns:
            相关文档列表。

        Examples:
        ```python
        await retriever.ainvoke("query")
        ```
        """
        config = ensure_config(config)
        inheritable_metadata = {
            **(config.get("metadata") or {}),
            **self._get_ls_params(**kwargs),
        }
        callback_manager = AsyncCallbackManager.configure(
            config.get("callbacks"),
            None,
            verbose=kwargs.get("verbose", False),
            inheritable_tags=config.get("tags"),
            local_tags=self.tags,
            inheritable_metadata=inheritable_metadata,
            local_metadata=self.metadata,
        )
        run_manager = await callback_manager.on_retriever_start(
            None,
            input,
            name=config.get("run_name") or self.get_name(),
            run_id=kwargs.pop("run_id", None),
        )
        try:
            kwargs_ = kwargs if self._expects_other_args else {}
            if self._new_arg_supported:
                result = await self._aget_relevant_documents(
                    input, run_manager=run_manager, **kwargs_
                )
            else:
                result = await self._aget_relevant_documents(input, **kwargs_)
        except Exception as e:
            await run_manager.on_retriever_error(e)
            raise
        else:
            await run_manager.on_retriever_end(
                result,
            )
            return result

    @abstractmethod
    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        """获取与查询相关的文档。

        Args:
            query: 用于查找相关文档的字符串。
            run_manager: 使用的回调处理器。

        Returns:
            相关文档列表。
        """

    async def _aget_relevant_documents(
        self, query: str, *, run_manager: AsyncCallbackManagerForRetrieverRun
    ) -> list[Document]:
        """异步获取与查询相关的文档。

        Args:
            query: 用于查找相关文档的字符串
            run_manager: 使用的回调处理器

        Returns:
            相关文档列表
        """
        return await run_in_executor(
            None,
            self._get_relevant_documents,
            query,
            run_manager=run_manager.get_sync(),
        )

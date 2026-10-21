"""Ollama 嵌入模型。"""

from __future__ import annotations

from typing import Any

from langchain_core.embeddings import Embeddings
from ollama import AsyncClient, Client
from pydantic import (
    BaseModel,
    ConfigDict,
    PrivateAttr,
    field_validator,
    model_validator,
)
from typing_extensions import Self

from langchain_ollama._utils import (
    merge_auth_headers,
    parse_url_with_auth,
    validate_model,
)


class OllamaEmbeddings(BaseModel, Embeddings):
    """Ollama 嵌入模型集成。

    设置本地 Ollama 实例：
        [安装 Ollama 包](https://github.com/ollama/ollama) 并设置一个
        本地 Ollama 实例。

        您需要选择一个要提供的模型。

        您可以通过 [模型库](https://ollama.com/library) 查看可用模型列表。

        要从 Ollama 模型库获取模型，请使用 `ollama pull <模型名称>`。

        例如，拉取 llama3 模型：

        ```bash
        ollama pull llama3
        ```

        这将下载该模型的默认标签版本。
        通常，默认标签指向最新、参数量最小的模型。

        * 在 Mac 上，模型将被下载到 `~/.ollama/models`
        * 在 Linux（或 WSL）上，模型将存储在 `/usr/share/ollama/.ollama/models`

        您可以指定所需模型的确切版本
        如此：`ollama pull vicuna:13b-v1.5-16k-q4_0`。

        查看已拉取的模型：

        ```bash
        ollama list
        ```

        启动服务：

        ```bash
        ollama serve
        ```

        查看 Ollama 文档以获取更多命令。

        ```bash
        ollama help
        ```

    安装 `langchain-ollama` 集成包：
        ```bash
        pip install -U langchain_ollama
        ```

    关键初始化参数 —— 补全参数：
        model: str
            要使用的 Ollama 模型名称。
        base_url: str | None
            模型托管的基地址。

    有关支持的初始化参数及其描述的完整列表，请参见参数部分。

    实例化：
        ```python
        from langchain_ollama import OllamaEmbeddings

        embed = OllamaEmbeddings(model="llama3")
        ```

    嵌入单个文本：
        ```python
        input_text = "The meaning of life is 42"
        vector = embed.embed_query(input_text)
        print(vector[:3])
        ```

        ```python
        [-0.024603435769677162, -0.007543657906353474, 0.0039630369283258915]
        ```

    嵌入多个文本：
        ```python
        input_texts = ["Document 1...", "Document 2..."]
        vectors = embed.embed_documents(input_texts)
        print(len(vectors))
        # The first 3 coordinates for the first vector
        print(vectors[0][:3])
        ```

        ```python
        2
        [-0.024603435769677162, -0.007543657906353474, 0.0039630369283258915]
        ```

    异步：
        ```python
        vector = await embed.aembed_query(input_text)
        print(vector[:3])

        # multiple:
        # await embed.aembed_documents(input_texts)
        ```

        ```python
        [-0.009100092574954033, 0.005071679595857859, -0.0029193938244134188]
        ```
    """  # noqa: E501

    model: str
    """要使用的模型名称。"""

    dimensions: int | None = None
    """输出嵌入向量的维度数。

    如果未提供，则使用模型的默认嵌入维度。
    """

    @field_validator("dimensions")
    @classmethod
    def _validate_dimensions(cls, v: int | None) -> int | None:
        if v is not None and v < 1:
            msg = "`dimensions` must be a positive integer."
            raise ValueError(msg)
        return v

    validate_model_on_init: bool = False
    """是否在初始化时验证模型是否存在于本地 Ollama 中。

    !!! version-added "在 `langchain-ollama` 0.3.4 中新增"

    """

    base_url: str | None = None
    """模型托管的基地址。

    如果为 None，则默认使用 Ollama 客户端的默认值。

    支持 `userinfo` 认证格式 `http://username:password@localhost:11434`。
    如果您的 Ollama 服务器位于代理后方，这很有用。

    !!! warning
        `userinfo` is not secure and should only be used for local testing or
        in secure environments. Avoid using it in production or over unsecured
        networks.

    !!! note
        使用 `userinfo` 时，请确保 Ollama 服务器已配置为
        接受并验证这些凭据。

    !!! note
        `userinfo` 头部会传递给同步和异步客户端。

    """

    client_kwargs: dict | None = {}
    """传递给 httpx 客户端的额外关键字参数。在此处传递头信息。

    这些参数会同时传递给同步和异步客户端。

    使用 `sync_client_kwargs` 和 `async_client_kwargs`
    分别向同步和异步客户端传递不同参数。
    """

    async_client_kwargs: dict | None = {}
    """与 `client_kwargs` 合并后传递给 httpx 客户端的额外关键字参数。

    这些是异步客户端独有的参数；共享参数请使用 `client_kwargs`。

    有关完整参数列表，请参见 [httpx 文档](https://www.python-httpx.org/api/#asyncclient)。
    """

    sync_client_kwargs: dict | None = {}
    """与 `client_kwargs` 合并后传递给 httpx 客户端的额外关键字参数。

    这些是同步客户端独有的客户端；共享参数请使用 `client_kwargs`。

    完整参数列表，请参见 [httpx 文档](https://www.python-httpx.org/api/#client)。
    """

    _client: Client | None = PrivateAttr(default=None)
    """用于发起请求的客户端。"""

    _async_client: AsyncClient | None = PrivateAttr(default=None)
    """用于发起请求的异步客户端。"""

    mirostat: int | None = None
    """启用 Mirostat 采样以控制困惑度。
    （默认：`0`，`0` = 禁用，`1` = Mirostat，`2` = Mirostat 2.0）"""

    mirostat_eta: float | None = None
    """Influences how quickly the algorithm responds to feedback
    from the generated text. A lower learning rate will result in
    slower adjustments, while a higher learning rate will make
    the algorithm more responsive. (Default: `0.1`)"""

    mirostat_tau: float | None = None
    """Controls the balance between coherence and diversity
    of the output. A lower value will result in more focused and
    coherent text. (Default: `5.0`)"""

    num_ctx: int | None = None
    """Sets the size of the context window used to generate the
    next token. (Default: `2048`)	"""

    num_gpu: int | None = None
    """要使用的 GPU 数量。在 macOS 上默认为 `1` 以启用 metal 支持，设为 `0` 则禁用。"""

    keep_alive: int | None = None
    """Controls how long the model will stay loaded into memory
    following the request (default: `5m`)
    """

    num_thread: int | None = None
    """Sets the number of threads to use during computation.
    By default, Ollama will detect this for optimal performance.
    It is recommended to set this value to the number of physical
    CPU cores your system has (as opposed to the logical number of cores)."""

    repeat_last_n: int | None = None
    """Sets how far back for the model to look back to prevent
    repetition. (Default: `64`, `0` = disabled, `-1` = `num_ctx`)"""

    repeat_penalty: float | None = None
    """设置惩罚重复的程度。值越高（例如 `1.5`）
    对重复的惩罚越强，而值越低（例如 `0.9`）
    则更宽容。（默认：`1.1`）"""

    temperature: float | None = None
    """The temperature of the model. Increasing the temperature will
    make the model answer more creatively. (Default: `0.8`)"""

    stop: list[str] | None = None
    """设置要使用的停止 token。"""

    tfs_z: float | None = None
    """Tail free sampling is used to reduce the impact of less probable
    tokens from the output. A higher value (e.g., `2.0`) will reduce the
    impact more, while a value of `1.0` disables this setting. (default: `1`)"""

    top_k: int | None = None
    """降低生成无意义内容的概率。值越高（例如 `100`）
    回答越多样化，而值越低（例如 `10`）
    则越保守。（默认：`40`）"""

    top_p: float | None = None
    """与 top-k 配合使用。值越高（例如 `0.95`）
    生成的文本越多样化，而值越低（例如 `0.5`）
    则生成更聚焦和保守的文本。（默认：`0.9`）"""

    model_config = ConfigDict(
        extra="forbid",
    )

    @property
    def _default_params(self) -> dict[str, Any]:
        """获取调用 Ollama 的默认参数。"""
        return {
            "mirostat": self.mirostat,
            "mirostat_eta": self.mirostat_eta,
            "mirostat_tau": self.mirostat_tau,
            "num_ctx": self.num_ctx,
            "num_gpu": self.num_gpu,
            "num_thread": self.num_thread,
            "repeat_last_n": self.repeat_last_n,
            "repeat_penalty": self.repeat_penalty,
            "temperature": self.temperature,
            "stop": self.stop,
            "tfs_z": self.tfs_z,
            "top_k": self.top_k,
            "top_p": self.top_p,
        }

    @model_validator(mode="after")
    def _set_clients(self) -> Self:
        """设置用于 Ollama 的客户端。"""
        client_kwargs = self.client_kwargs or {}

        cleaned_url, auth_headers = parse_url_with_auth(self.base_url)
        merge_auth_headers(client_kwargs, auth_headers)

        sync_client_kwargs = client_kwargs
        if self.sync_client_kwargs:
            sync_client_kwargs = {**sync_client_kwargs, **self.sync_client_kwargs}

        async_client_kwargs = client_kwargs
        if self.async_client_kwargs:
            async_client_kwargs = {**async_client_kwargs, **self.async_client_kwargs}

        self._client = Client(host=cleaned_url, **sync_client_kwargs)
        self._async_client = AsyncClient(host=cleaned_url, **async_client_kwargs)
        if self.validate_model_on_init:
            validate_model(self._client, self.model)
        return self

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """嵌入搜索文档。"""
        if not self._client:
            msg = (
                "Ollama sync client is not initialized. "
                "Make sure the model was properly constructed."
            )
            raise RuntimeError(msg)
        return self._client.embed(
            self.model,
            texts,
            dimensions=self.dimensions,
            options=self._default_params,
            keep_alive=self.keep_alive,
        )["embeddings"]

    def embed_query(self, text: str) -> list[float]:
        """嵌入查询文本。"""
        return self.embed_documents([text])[0]

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        """嵌入搜索文档。"""
        if not self._async_client:
            msg = (
                "Ollama async client is not initialized. "
                "Make sure the model was properly constructed."
            )
            raise RuntimeError(msg)
        return (
            await self._async_client.embed(
                self.model,
                texts,
                dimensions=self.dimensions,
                options=self._default_params,
                keep_alive=self.keep_alive,
            )
        )["embeddings"]

    async def aembed_query(self, text: str) -> list[float]:
        """嵌入查询文本。"""
        return (await self.aembed_documents([text]))[0]

from langchain_core.embeddings import Embeddings
from langchain_core.utils import secret_from_env
from openai import OpenAI
from pydantic import BaseModel, ConfigDict, Field, SecretStr, model_validator
from typing_extensions import Self


class FireworksEmbeddings(BaseModel, Embeddings):
    """Fireworks 嵌入模型集成。

    设置：

        安装 `langchain_fireworks` 并设置环境变量
        `FIREWORKS_API_KEY`。

        ```bash
        pip install -U langchain_fireworks
        export FIREWORKS_API_KEY="your-api-key"
        ```

    关键初始化参数 —— 补全参数：
        model:
            要使用的 Fireworks 模型名称。

    关键初始化参数 —— 客户端参数：
        fireworks_api_key:
            Fireworks API 密钥。

    有关支持的初始化参数及其描述的完整列表，请参见参数部分。

    实例化：

        ```python
        from langchain_fireworks import FireworksEmbeddings

        model = FireworksEmbeddings(
            model="nomic-ai/nomic-embed-text-v1.5"
            # Use FIREWORKS_API_KEY env var or pass it in directly
            # fireworks_api_key="..."
        )
        ```

    嵌入多个文本：

        ```python
        vectors = embeddings.embed_documents(["hello", "goodbye"])
        # Showing only the first 3 coordinates
        print(len(vectors))
        print(vectors[0][:3])
        ```
        ```python
        2
        [-0.024603435769677162, -0.007543657906353474, 0.0039630369283258915]
        ```

    嵌入单个文本：

        ```python
        input_text = "The meaning of life is 42"
        vector = embeddings.embed_query("hello")
        print(vector[:3])
        ```
        ```python
        [-0.024603435769677162, -0.007543657906353474, 0.0039630369283258915]
        ```
    """

    client: OpenAI = Field(default=None, exclude=True)  # type: ignore[assignment]

    fireworks_api_key: SecretStr = Field(
        alias="api_key",
        default_factory=secret_from_env(
            "FIREWORKS_API_KEY",
            error_message=(
                "You must specify an api key. "
                "You can pass it an argument as `api_key=...` or "
                "set the environment variable `FIREWORKS_API_KEY`."
            ),
        ),
    )
    """Fireworks API 密钥。

    如果未提供，则自动从环境变量 `FIREWORKS_API_KEY` 读取。
    """

    model: str = "nomic-ai/nomic-embed-text-v1.5"

    model_config = ConfigDict(
        populate_by_name=True,
        arbitrary_types_allowed=True,
    )

    @model_validator(mode="after")
    def validate_environment(self) -> Self:
        """验证环境变量。"""
        self.client = OpenAI(
            api_key=self.fireworks_api_key.get_secret_value(),
            base_url="https://api.fireworks.ai/inference/v1",
        )
        return self

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """嵌入搜索文档。"""
        return [
            i.embedding
            for i in self.client.embeddings.create(input=texts, model=self.model).data
        ]

    def embed_query(self, text: str) -> list[float]:
        """嵌入查询文本。"""
        return self.embed_documents([text])[0]

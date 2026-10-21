"""模块包含一些用于测试的假嵌入模型。"""

# 请勿在此处添加额外的假嵌入模型实现。
import contextlib
import hashlib

from pydantic import BaseModel
from typing_extensions import override

from langchain_core.embeddings import Embeddings

with contextlib.suppress(ImportError):
    import numpy as np


class FakeEmbeddings(Embeddings, BaseModel):
    """用于单元测试的假嵌入模型。

    此嵌入模型通过从正态分布中采样来创建嵌入。

    !!! danger "玩具模型"
        请勿在测试以外的场景使用，因为它不是真正的嵌入模型。

    实例化：
        ```python
        from langchain_core.embeddings import FakeEmbeddings

        embed = FakeEmbeddings(size=100)
        ```

    嵌入单个文本：
        ```python
        input_text = "The meaning of life is 42"
        vector = embed.embed_query(input_text)
        print(vector[:3])
        ```
        ```python
        [-0.700234640213188, -0.581266257710429, -1.1328482266445354]
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
        [-0.5670477847544458, -0.31403828652395727, -0.5840547508955257]
        ```
    """

    size: int
    """嵌入向量的大小。"""

    def _get_embedding(self) -> list[float]:
        return list(np.random.default_rng().normal(size=self.size))

    @override
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._get_embedding() for _ in texts]

    @override
    def embed_query(self, text: str) -> list[float]:
        return self._get_embedding()


class DeterministicFakeEmbedding(Embeddings, BaseModel):
    """用于单元测试的确定性假嵌入模型。

    This embedding model creates embeddings by sampling from a normal distribution
    with a seed based on the hash of the text.

    !!! danger "玩具模型"
        请勿在测试以外的场景使用，因为它不是真正的嵌入模型。

    实例化：
        ```python
        from langchain_core.embeddings import DeterministicFakeEmbedding

        embed = DeterministicFakeEmbedding(size=100)
        ```

    嵌入单个文本：
        ```python
        input_text = "The meaning of life is 42"
        vector = embed.embed_query(input_text)
        print(vector[:3])
        ```
        ```python
        [-0.700234640213188, -0.581266257710429, -1.1328482266445354]
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
        [-0.5670477847544458, -0.31403828652395727, -0.5840547508955257]
        ```
    """

    size: int
    """嵌入向量的大小。"""

    def _get_embedding(self, seed: int) -> list[float]:
                # 设置随机生成器的种子
        rng = np.random.default_rng(seed)
        return list(rng.normal(size=self.size))

    @staticmethod
    def _get_seed(text: str) -> int:
        """根据文本哈希值获取随机生成器的种子。"""
        return int(hashlib.sha256(text.encode("utf-8")).hexdigest(), 16) % 10**8

    @override
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._get_embedding(seed=self._get_seed(_)) for _ in texts]

    @override
    def embed_query(self, text: str) -> list[float]:
        return self._get_embedding(seed=self._get_seed(text))

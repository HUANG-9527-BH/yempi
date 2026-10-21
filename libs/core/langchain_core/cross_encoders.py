"""交叉编码器接口。"""

from abc import ABC, abstractmethod


class BaseCrossEncoder(ABC):
    """交叉编码器模型的接口。"""

    @abstractmethod
    def score(self, text_pairs: list[tuple[str, str]]) -> list[float]:
        """计算文本对之间的相似度。

        Args:
            text_pairs: 文本对列表。

        Returns:
            相似度得分列表。
        """

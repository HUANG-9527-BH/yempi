"""字符串输出解析器。"""

from typing_extensions import override

from langchain_core.output_parsers.transform import BaseTransformOutputParser


class StrOutputParser(BaseTransformOutputParser[str]):
    """从模型输出中提取文本内容作为字符串。

    Converts model outputs (such as `AIMessage` or `AIMessageChunk` objects) into plain
    text strings. It's the simplest output parser and is useful when you need string
    responses for downstream processing, display, or storage.

    支持流式输出，可逐块生成模型输出时的文本片段。

    Example:
        ```python
        from langchain_core.output_parsers import StrOutputParser
        from langchain_openai import ChatOpenAI

        model = ChatOpenAI(model="gpt-5.5")
        parser = StrOutputParser()

        # Get string output from a model
        message = model.invoke("Tell me a joke")
        result = parser.invoke(message)
        print(result)  # plain string

        # With streaming - use transform() to process a stream
        stream = model.stream("Tell me a story")
        for chunk in parser.transform(stream):
            print(chunk, end="", flush=True)
        ```
    """

    @classmethod
    def is_lc_serializable(cls) -> bool:
        """`StrOutputParser` 可序列化。

        Returns:
            `True`
        """
        return True

    @classmethod
    def get_lc_namespace(cls) -> list[str]:
        """获取 LangChain 对象的命名空间。

        Returns:
            `["langchain", "schema", "output_parser"]`
        """
        return ["langchain", "schema", "output_parser"]

    @property
    def _type(self) -> str:
        """返回用于序列化的输出解析器类型。"""
        return "default"

    @override
    def parse(self, text: str) -> str:
        """返回输入文本，不作任何更改。"""
        return text

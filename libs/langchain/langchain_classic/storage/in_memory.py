"""内存存储，线程不安全，且无淘汰策略。

This is a simple implementation of the BaseStore using a dictionary that is useful
primarily for unit testing purposes.
"""

from langchain_core.stores import InMemoryBaseStore, InMemoryByteStore, InMemoryStore

__all__ = [
    "InMemoryBaseStore",
    "InMemoryByteStore",
    "InMemoryStore",
]

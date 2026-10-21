"""键值存储和存储辅助工具的实现。

该模块提供了符合简单键值接口的各种键值存储的实现。

这些存储的主要目标是支持缓存的实现。
"""

from typing import TYPE_CHECKING, Any

from langchain_core.stores import (
    InMemoryByteStore,
    InMemoryStore,
    InvalidKeyException,
)

from langchain_classic._api import create_importer
from langchain_classic.storage._lc_store import create_kv_docstore, create_lc_store
from langchain_classic.storage.encoder_backed import EncoderBackedStore
from langchain_classic.storage.file_system import LocalFileStore

if TYPE_CHECKING:
    from langchain_community.storage import (
        RedisStore,
        UpstashRedisByteStore,
        UpstashRedisStore,
    )

# 创建一种动态查找已弃用导入的方式。
# 用于统一处理弃用警告的逻辑并
# 处理可选导入。
DEPRECATED_LOOKUP = {
    "RedisStore": "langchain_community.storage",
    "UpstashRedisByteStore": "langchain_community.storage",
    "UpstashRedisStore": "langchain_community.storage",
}

_import_attribute = create_importer(__package__, deprecated_lookups=DEPRECATED_LOOKUP)


def __getattr__(name: str) -> Any:
    """动态查找属性。"""
    return _import_attribute(name)


__all__ = [
    "EncoderBackedStore",
    "InMemoryByteStore",
    "InMemoryStore",
    "InvalidKeyException",
    "LocalFileStore",
    "RedisStore",
    "UpstashRedisByteStore",
    "UpstashRedisStore",
    "create_kv_docstore",
    "create_lc_store",
]

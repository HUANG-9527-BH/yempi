from typing import TYPE_CHECKING, Any

from langchain_classic._api import create_importer

if TYPE_CHECKING:
    from langchain_community.vectorstores.redis.schema import (
        FlatVectorField,
        HNSWVectorField,
        NumericFieldSchema,
        RedisDistanceMetric,
        RedisField,
        RedisModel,
        RedisVectorField,
        TagFieldSchema,
        TextFieldSchema,
        read_schema,
    )

# 创建一种动态查找已弃用导入的方式。
# 用于统一处理引发弃用警告的逻辑，并
# 处理可选导入。
DEPRECATED_LOOKUP = {
    "RedisDistanceMetric": "langchain_community.vectorstores.redis.schema",
    "RedisField": "langchain_community.vectorstores.redis.schema",
    "TextFieldSchema": "langchain_community.vectorstores.redis.schema",
    "TagFieldSchema": "langchain_community.vectorstores.redis.schema",
    "NumericFieldSchema": "langchain_community.vectorstores.redis.schema",
    "RedisVectorField": "langchain_community.vectorstores.redis.schema",
    "FlatVectorField": "langchain_community.vectorstores.redis.schema",
    "HNSWVectorField": "langchain_community.vectorstores.redis.schema",
    "RedisModel": "langchain_community.vectorstores.redis.schema",
    "read_schema": "langchain_community.vectorstores.redis.schema",
}

_import_attribute = create_importer(__package__, deprecated_lookups=DEPRECATED_LOOKUP)


def __getattr__(name: str) -> Any:
    """动态查找属性。"""
    return _import_attribute(name)


__all__ = [
    "FlatVectorField",
    "HNSWVectorField",
    "NumericFieldSchema",
    "RedisDistanceMetric",
    "RedisField",
    "RedisModel",
    "RedisVectorField",
    "TagFieldSchema",
    "TextFieldSchema",
    "read_schema",
]

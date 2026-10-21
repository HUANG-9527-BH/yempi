from typing import TYPE_CHECKING, Any

from langchain_classic._api import create_importer

if TYPE_CHECKING:
    from langchain_community.callbacks.utils import (
        BaseMetadataCallbackHandler,
        _flatten_dict,
        flatten_dict,
        hash_string,
        import_pandas,
        import_spacy,
        import_textstat,
        load_json,
    )

# 创建一种动态查找已弃用导入的方式。
# 用于合并引发弃用警告和处理可选导入的逻辑。
# 处理可选导入。
DEPRECATED_LOOKUP = {
    "import_spacy": "langchain_community.callbacks.utils",
    "import_pandas": "langchain_community.callbacks.utils",
    "import_textstat": "langchain_community.callbacks.utils",
    "_flatten_dict": "langchain_community.callbacks.utils",
    "flatten_dict": "langchain_community.callbacks.utils",
    "hash_string": "langchain_community.callbacks.utils",
    "load_json": "langchain_community.callbacks.utils",
    "BaseMetadataCallbackHandler": "langchain_community.callbacks.utils",
}

_import_attribute = create_importer(__file__, deprecated_lookups=DEPRECATED_LOOKUP)


def __getattr__(name: str) -> Any:
    """动态查找属性。"""
    return _import_attribute(name)


__all__ = [
    "BaseMetadataCallbackHandler",
    "_flatten_dict",
    "flatten_dict",
    "hash_string",
    "import_pandas",
    "import_spacy",
    "import_textstat",
    "load_json",
]

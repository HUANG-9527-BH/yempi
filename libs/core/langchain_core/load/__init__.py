"""``Load`` 模块有助于序列化和反序列化。"""

from typing import TYPE_CHECKING

from langchain_core._import_utils import import_attr

if TYPE_CHECKING:
    from langchain_core.load.dump import dumpd, dumps
    from langchain_core.load.load import InitValidator, loads
    from langchain_core.load.serializable import Serializable

# 不幸的是，我们必须提前从 langchain_core/load/load.py 导入 load
# 以避免命名空间冲突。我们希望用户仍然可以使用
# `from langchain_core.load import load` 来获取 load 函数，但
# 绝对导入 `from langchain_core.load.load import load` 也应正常工作。
from langchain_core.load.load import load

__all__ = (
    "InitValidator",
    "Serializable",
    "dumpd",
    "dumps",
    "load",
    "loads",
)

_dynamic_imports = {
    "dumpd": "dump",
    "dumps": "dump",
    "InitValidator": "load",
    "loads": "load",
    "Serializable": "serializable",
}


def __getattr__(attr_name: str) -> object:
    module_name = _dynamic_imports.get(attr_name)
    result = import_attr(attr_name, module_name, __spec__.parent)
    globals()[attr_name] = result
    return result


def __dir__() -> list[str]:
    return list(__all__)

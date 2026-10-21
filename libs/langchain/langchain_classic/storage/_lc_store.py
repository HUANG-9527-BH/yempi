"""为任何 LangChain 可序列化对象创建一个键值存储。"""

from collections.abc import Callable
from typing import Any

from langchain_core.documents import Document
from langchain_core.load import Serializable, dumps, loads
from langchain_core.stores import BaseStore, ByteStore

from langchain_classic.storage.encoder_backed import EncoderBackedStore


def _dump_as_bytes(obj: Serializable) -> bytes:
    """返回 `Document` 的字节表示形式。"""
    return dumps(obj).encode("utf-8")


def _dump_document_as_bytes(obj: Any) -> bytes:
    """返回 `Document` 的字节表示形式。"""
    if not isinstance(obj, Document):
        msg = "Expected a Document instance"
        raise TypeError(msg)
    return dumps(obj).encode("utf-8")


def _load_document_from_bytes(serialized: bytes) -> Document:
    """从字节表示形式返回一个文档。"""
    obj = loads(serialized.decode("utf-8"), allowed_objects=[Document])
    if not isinstance(obj, Document):
        msg = f"Expected a Document instance. Got {type(obj)}"
        raise TypeError(msg)
    return obj


def _load_from_bytes(serialized: bytes) -> Serializable:
    """从字节表示形式返回一个 `Serializable` 对象。"""
        # 默认允许列表（`'core'`）在不受信任输入下不安全——一个被篡改的字节负载可以重建任何核心类并使用攻击者控制的参数（自定义
    # `base_url`、headers、模型名称等）。
        # 此加载器背后的字节存储必须被视为信任边界——参见 `create_lc_store`
    # 上的危险提示。如果该存储可被任何你不信任的人写入，请改用 `create_kv_docstore`。
        # tampered byte payload can reconstruct any core class with
        # attacker-controlled kwargs (custom `base_url`, headers, model name,
        # etc.). The byte store backing this loader must be treated as a trust
        # boundary - see the danger note on `create_lc_store`. If the store can
        # be written to by anyone you do not already trust, use
    return loads(serialized.decode("utf-8"))


def _identity(x: str) -> str:
    """返回相同对象。"""
    return x


# PUBLIC API


def create_lc_store(
    store: ByteStore,
    *,
    key_encoder: Callable[[str], str] | None = None,
) -> BaseStore[str, Serializable]:
    """从字节存储创建 LangChain 可序列化对象的存储。

    !!! danger "将底层字节存储视为信任边界"

        Reads from this store are deserialized with
        `langchain_core.load.loads`, which instantiates Python objects from
        the stored payload. The same threat model applies: a payload can
        carry constructor kwargs (custom `base_url`, headers, model name,
        etc.) that get applied during `__init__`, so the bytes are
        effectively executable configuration rather than plain data.

        **Never back this store with anything an attacker can write to** —
        for example a shared cache that other tenants can populate, an
        S3 bucket without strict write controls, or a Redis instance reused
        across trust boundaries. A single tampered value will instantiate
        attacker-controlled classes the next time the store is read.

        If you cannot guarantee the store is write-restricted to your own
        process, use `create_kv_docstore` instead — it pins
        `allowed_objects=[Document]` so a tampered value can at worst
        produce a `Document`, never a chat model or LLM with a redirected
        endpoint.

    Args:
        store: 用作底层存储的字节存储。
        key_encoder: 用于编码键的函数；如果为 `None`，则使用恒等函数。

    Returns:
        用于 `Document` 对象的键值存储。
    """
    return EncoderBackedStore(
        store,
        key_encoder or _identity,
        _dump_as_bytes,
        _load_from_bytes,
    )


def create_kv_docstore(
    store: ByteStore,
    *,
    key_encoder: Callable[[str], str] | None = None,
) -> BaseStore[str, Document]:
    """从字节存储创建 LangChain `Document` 对象的存储。

    此存储在运行时进行类型检查，以确保值为 `Document` 对象。

    Args:
        store: 用作底层存储的字节存储。
        key_encoder: 用于编码键的函数；如果为 `None`，则使用恒等函数。

    Returns:
        用于 `Document` 对象的键值存储。
    """
    return EncoderBackedStore(
        store,
        key_encoder or _identity,
        _dump_document_as_bytes,
        _load_document_from_bytes,
    )

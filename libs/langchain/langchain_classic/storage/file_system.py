import os
import re
import time
from collections.abc import Iterator, Sequence
from pathlib import Path

from langchain_core.stores import ByteStore

from langchain_classic.storage.exceptions import InvalidKeyException


class LocalFileStore(ByteStore):
    """在本地文件系统上工作的 `BaseStore` 接口。

    Examples:
        创建一个 `LocalFileStore` 实例并对其执行操作：

        ```python
        from langchain_classic.storage import LocalFileStore

        # Instantiate the LocalFileStore with the root path
        file_store = LocalFileStore("/path/to/root")

        # Set values for keys
        file_store.mset([("key1", b"value1"), ("key2", b"value2")])

        # Get values for keys
        values = file_store.mget(["key1", "key2"])  # Returns [b"value1", b"value2"]

        # Delete keys
        file_store.mdelete(["key1"])

        # Iterate over keys
        for key in file_store.yield_keys():
            print(key)  # noqa: T201
        ```
    """

    def __init__(
        self,
        root_path: str | Path,
        *,
        chmod_file: int | None = None,
        chmod_dir: int | None = None,
        update_atime: bool = False,
    ) -> None:
        """为本地文件系统实现 `BaseStore` 接口。

        Args:
            root_path: The root path of the file store. All keys are interpreted as
                paths relative to this root.
            chmod_file: Sets permissions for newly created files, overriding the
                current `umask` if needed.
            chmod_dir: Sets permissions for newly created dirs, overriding the
                current `umask` if needed.
            update_atime: Updates the filesystem access time (but not the modified
                time) when a file is read. This allows MRU/LRU cache policies to be
                implemented for filesystems where access time updates are disabled.
        """
        self.root_path = Path(root_path).absolute()
        self.chmod_file = chmod_file
        self.chmod_dir = chmod_dir
        self.update_atime = update_atime

    def _get_full_path(self, key: str) -> Path:
        """获取相对于根路径的给定键的完整路径。

        Args:
            key: 相对于根路径的键。

        Returns:
            给定键的完整路径。
        """
        if not re.match(r"^[a-zA-Z0-9_.\-/]+$", key):
            msg = f"Invalid characters in key: {key}"
            raise InvalidKeyException(msg)
        full_path = (self.root_path / key).resolve()
        root_path = self.root_path.resolve()
        common_path = os.path.commonpath([root_path, full_path])
        if common_path != str(root_path):
            msg = (
                f"Invalid key: {key}. Key should be relative to the full path. "
                f"{root_path} vs. {common_path} and full path of {full_path}"
            )
            raise InvalidKeyException(msg)

        return full_path

    def _mkdir_for_store(self, dir_path: Path) -> None:
        """使用指定权限创建存储目录路径（包括父目录）。

        这是必需的，因为 `Path.mkdir()` 受当前 `umask` 限制，而此处使用的显式
        `os.chmod()` 不受限制。

        Args:
            dir_path: 要创建的存储目录。
        """
        if not dir_path.exists():
            self._mkdir_for_store(dir_path.parent)
            dir_path.mkdir(exist_ok=True)
            if self.chmod_dir is not None:
                dir_path.chmod(self.chmod_dir)

    def mget(self, keys: Sequence[str]) -> list[bytes | None]:
        """获取与给定键关联的值。

        Args:
            keys: 键的序列。

        Returns:
            与键关联的可选值序列。
            如果某个键未找到，对应的值将为 `None`。
        """
        values: list[bytes | None] = []
        for key in keys:
            full_path = self._get_full_path(key)
            if full_path.exists():
                value = full_path.read_bytes()
                values.append(value)
                if self.update_atime:
                                        # update access time only; preserve modified time
                    os.utime(full_path, (time.time(), full_path.stat().st_mtime))
            else:
                values.append(None)
        return values

    def mset(self, key_value_pairs: Sequence[tuple[str, bytes]]) -> None:
        """设置给定键的值。

        Args:
            key_value_pairs: 键值对的序列。
        """
        for key, value in key_value_pairs:
            full_path = self._get_full_path(key)
            self._mkdir_for_store(full_path.parent)
            full_path.write_bytes(value)
            if self.chmod_file is not None:
                full_path.chmod(self.chmod_file)

    def mdelete(self, keys: Sequence[str]) -> None:
        """删除给定键及其关联的值。

        Args:
            keys: 要删除的键的序列。
        """
        for key in keys:
            full_path = self._get_full_path(key)
            if full_path.exists():
                full_path.unlink()

    def yield_keys(self, *, prefix: str | None = None) -> Iterator[str]:
        """获取匹配给定前缀的键的迭代器。

        Args:
            prefix: 要匹配的前缀。

        Yields:
            匹配给定前缀的键。
        """
        prefix_path = self._get_full_path(prefix) if prefix else self.root_path
        for file in prefix_path.rglob("*"):
            if file.is_file():
                relative_path = file.relative_to(self.root_path)
                yield str(relative_path)

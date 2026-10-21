"""检查 `pyproject.toml` 和 `_version.py` 之间的版本一致性。

This script validates that the version defined in pyproject.toml matches the
`__version__` variable in `langchain_xai/_version.py`. Intended for use as a
CI check to prevent version mismatches.
"""

import re
import sys
from pathlib import Path


def get_pyproject_version(pyproject_path: Path) -> str | None:
    """从 `pyproject.toml` 提取版本。"""
    content = pyproject_path.read_text(encoding="utf-8")
    match = re.search(r'^version\s*=\s*"([^"]+)"', content, re.MULTILINE)
    return match.group(1) if match else None


def get_version_py_version(version_path: Path) -> str | None:
    """从 `_version.py` 提取 `__version__`。"""
    content = version_path.read_text(encoding="utf-8")
    match = re.search(r'^__version__\s*=\s*"([^"]+)"', content, re.MULTILINE)
    return match.group(1) if match else None


def main() -> int:
    """验证版本一致性。"""
    script_dir = Path(__file__).parent
    package_dir = script_dir.parent

    pyproject_path = package_dir / "pyproject.toml"
    version_path = package_dir / "langchain_xai" / "_version.py"

    if not pyproject_path.exists():
        print(f"Error: {pyproject_path} not found")  # noqa: T201
        return 1

    if not version_path.exists():
        print(f"Error: {version_path} not found")  # noqa: T201
        return 1

    pyproject_version = get_pyproject_version(pyproject_path)
    version_py_version = get_version_py_version(version_path)

    if pyproject_version is None:
        print("Error: Could not find version in pyproject.toml")  # noqa: T201
        return 1

    if version_py_version is None:
        print("Error: Could not find __version__ in langchain_xai/_version.py")  # noqa: T201
        return 1

    if pyproject_version != version_py_version:
        print("Error: Version mismatch detected!")  # noqa: T201
        print(f"  pyproject.toml: {pyproject_version}")  # noqa: T201
        print(f"  langchain_xai/_version.py: {version_py_version}")  # noqa: T201
        return 1

    print(f"Version check passed: {pyproject_version}")  # noqa: T201
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""检查 pyproject.toml 与 __init__.py 之间的版本一致性。

This script validates that the version defined in pyproject.toml matches
the __version__ variable in langchain/__init__.py. Intended for use as
a pre-commit hook to prevent version mismatches.
"""

import re
import sys
from pathlib import Path


def get_pyproject_version(pyproject_path: Path) -> str | None:
    """从 pyproject.toml 提取版本号。"""
    content = pyproject_path.read_text(encoding="utf-8")
    match = re.search(r'^version\s*=\s*"([^"]+)"', content, re.MULTILINE)
    return match.group(1) if match else None


def get_init_version(init_path: Path) -> str | None:
    """从 __init__.py 提取 __version__。"""
    content = init_path.read_text(encoding="utf-8")
    match = re.search(r'^__version__\s*=\s*"([^"]+)"', content, re.MULTILINE)
    return match.group(1) if match else None


def main() -> int:
    """验证版本一致性。"""
    script_dir = Path(__file__).parent
    package_dir = script_dir.parent

    pyproject_path = package_dir / "pyproject.toml"
    init_path = package_dir / "langchain" / "__init__.py"

    if not pyproject_path.exists():
        print(f"Error: {pyproject_path} not found")
        return 1

    if not init_path.exists():
        print(f"Error: {init_path} not found")
        return 1

    pyproject_version = get_pyproject_version(pyproject_path)
    init_version = get_init_version(init_path)

    if pyproject_version is None:
        print("Error: Could not find version in pyproject.toml")
        return 1

    if init_version is None:
        print("Error: Could not find __version__ in langchain/__init__.py")
        return 1

    if pyproject_version != init_version:
        print("Error: Version mismatch detected!")
        print(f"  pyproject.toml: {pyproject_version}")
        print(f"  langchain/__init__.py: {init_version}")
        return 1

    print(f"Version check passed: {pyproject_version}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

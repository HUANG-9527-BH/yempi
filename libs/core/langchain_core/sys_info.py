"""用于调试目的，打印有关系统和 langchain 包的信息。"""

import pkgutil
import platform
import re
import sys
from collections.abc import Sequence
from importlib import metadata, util


def _get_sub_deps(packages: Sequence[str]) -> list[str]:
    """获取任何指定的子依赖项。"""
    sub_deps = set()
    underscored_packages = {pkg.replace("-", "_") for pkg in packages}

    for pkg in packages:
        try:
            required = metadata.requires(pkg)
        except metadata.PackageNotFoundError:
            continue

        if not required:
            continue

        for req in required:
                        # 提取包名称（例如 "httpx<1,>=0.23.0" -> "httpx"）
            match = re.match(r"^([a-zA-Z0-9_.-]+)", req)
            if match:
                pkg_name = match.group(1)
                if pkg_name.replace("-", "_") not in underscored_packages:
                    sub_deps.add(pkg_name)

    return sorted(sub_deps, key=lambda x: x.lower())


def print_sys_info(*, additional_pkgs: Sequence[str] = ()) -> None:
    """用于调试目的，打印有关环境的信息。

    Args:
        additional_pkgs: 要包含在输出中的额外包。
    """
        # 不以 "langchain" 前缀开头的包。
    other_langchain_packages = [
        "langsmith",
        "deepagents",
        "deepagents-cli",
    ]

    langchain_pkgs = [
        name for _, name, _ in pkgutil.iter_modules() if name.startswith("langchain")
    ]

    langgraph_pkgs = [
        name for _, name, _ in pkgutil.iter_modules() if name.startswith("langgraph")
    ]

    all_packages = sorted(
        set(
            langchain_pkgs
            + langgraph_pkgs
            + other_langchain_packages
            + list(additional_pkgs)
        )
    )

        # 始终将这些包置于顶部
    order_by = ["langchain_core", "langchain", "langchain_community", "langsmith"]

    for pkg in reversed(order_by):
        if pkg in all_packages:
            all_packages.remove(pkg)
            all_packages = [pkg, *list(all_packages)]

    system_info = {
        "OS": platform.system(),
        "OS Version": platform.version(),
        "Python Version": sys.version,
    }
    print()
    print("System Information")
    print("------------------")
    print("> OS: ", system_info["OS"])
    print("> OS Version: ", system_info["OS Version"])
    print("> Python Version: ", system_info["Python Version"])

        # 仅打印 langchain 包
    print()
    print("Package Information")
    print("-------------------")

    not_installed = []

    for pkg in all_packages:
        try:
            found_package = util.find_spec(pkg)
        except Exception:
            found_package = None
        if found_package is None:
            not_installed.append(pkg)
            continue

                # 包版本
        try:
            package_version = metadata.version(pkg)
        except Exception:
            package_version = None

                # 打印带版本的包
        if package_version is not None:
            print(f"> {pkg}: {package_version}")

    if not_installed:
        print()
        print("Optional packages not installed")
        print("-------------------------------")
        for pkg in not_installed:
            print(f"> {pkg}")

    sub_dependencies = _get_sub_deps(all_packages)

    if sub_dependencies:
        print()
        print("Other Dependencies")
        print("------------------")

        for dep in sub_dependencies:
            try:
                dep_version = metadata.version(dep)
            except Exception:
                dep_version = None

            if dep_version is not None:
                print(f"> {dep}: {dep_version}")


if __name__ == "__main__":
    print_sys_info()

#!/usr/bin/env python3
"""中文化结果校验：语法可编译 + 残留英文注释/文档字符串统计。"""

from __future__ import annotations

import ast
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
STATE = ROOT / ".workbuddy" / "tools"

CJK = re.compile(r"[\u4e00-\u9fff]")
DOC_NODES = (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)


def docstrings(tree: ast.AST):
    for node in ast.walk(tree):
        if isinstance(node, DOC_NODES):
            body = getattr(node, "body", None)
            if not body:
                continue
            first = body[0]
            if (
                isinstance(first, ast.Expr)
                and isinstance(first.value, ast.Constant)
                and isinstance(first.value.value, str)
            ):
                yield first.value.value


def looks_english(text: str) -> bool:
    if CJK.search(text):
        return False
    words = re.findall(r"[A-Za-z]{3,}", text)
    return len(words) >= 4


def main() -> int:
    paths = [
        ROOT / ln.strip()
        for ln in (STATE / "changed.txt").read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]
    bad, leftover = [], []
    total_doc = 0
    for p in paths:
        if not p.exists():
            continue
        try:
            tree = ast.parse(p.read_text(encoding="utf-8"))
        except SyntaxError as exc:
            bad.append(f"{p.relative_to(ROOT)}: {exc}")
            continue
        for ds in docstrings(tree):
            total_doc += 1
            if looks_english(ds):
                leftover.append(str(p.relative_to(ROOT)))
                break
    print(f"检查文件 {len(paths)} 个，docstring {total_doc} 个")
    print(f"语法错误 {len(bad)} 个")
    for b in bad[:20]:
        print("  ", b)
    print(f"仍有整段英文 docstring 的文件 {len(leftover)} 个")
    for f in leftover[:25]:
        print("  ", f)
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())

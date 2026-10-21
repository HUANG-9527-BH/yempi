"""对比 HEAD 与工作区：docstring 归一化后的 AST 必须完全一致。"""
import ast, subprocess, sys
from pathlib import Path

NODES = (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)

def skeleton(src):
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, NODES):
            body = getattr(node, "body", None)
            if not body:
                continue
            first = body[0]
            if (isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant)
                    and isinstance(first.value.value, str)):
                first.value.value = "<DOC>"
    return ast.dump(tree, include_attributes=False)

files = [l.strip() for l in Path(".workbuddy/tools/changed_py.txt").read_text(encoding="utf-8").splitlines() if l.strip()]
bad = []
for f in files:
    old = subprocess.run(["git", "show", f"HEAD:{f}"], capture_output=True).stdout
    new = Path(f).read_bytes()
    try:
        s_old = skeleton(old.decode("utf-8"))
    except Exception as e:
        print("原文件解析失败", f, e)
        continue
    try:
        s_new = skeleton(new.decode("utf-8"))
    except Exception as e:
        bad.append((f, f"解析失败 {e}"))
        continue
    if s_old != s_new:
        bad.append((f, "AST 不一致"))
print(f"对比 {len(files)} 个文件，异常 {len(bad)}")
for b in bad[:30]:
    print("  ", b)

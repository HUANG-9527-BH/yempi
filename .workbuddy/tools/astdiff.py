"""归一化所有字符串常量后对比 AST，并列出非 docstring 的字符串改动。"""
import ast, subprocess
from pathlib import Path

NODES = (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)

def normalize(src, blank_all=False):
    tree = ast.parse(src)
    docs = set()
    for n in ast.walk(tree):
        if isinstance(n, NODES):
            b = getattr(n, "body", None)
            if b and isinstance(b[0], ast.Expr) and isinstance(b[0].value, ast.Constant) and isinstance(b[0].value.value, str):
                docs.add(id(b[0].value))
            if isinstance(n, ast.ClassDef):
                for stmt in b[1:]:
                    if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant) and isinstance(stmt.value.value, str):
                        docs.add(id(stmt.value))
    changed = []
    for n in ast.walk(tree):
        if isinstance(n, ast.Constant) and isinstance(n.value, str):
            if id(n) in docs or blank_all:
                changed.append(n.value)
                n.value = "<S>"
    return ast.dump(tree, include_attributes=False), changed

files = [l.strip() for l in Path(".workbuddy/tools/changed_py.txt").read_text(encoding="utf-8").splitlines() if l.strip()]
bad, extra = [], []
for f in files:
    old = subprocess.run(["git", "show", f"HEAD:{f}"], capture_output=True).stdout.decode("utf-8", "ignore")
    new = Path(f).read_text(encoding="utf-8")
    try:
        s_old, _ = normalize(old, True)
        s_new, _ = normalize(new, True)
    except Exception as e:
        bad.append((f, f"解析失败 {e}"))
        continue
    if s_old != s_new:
        bad.append((f, "去掉全部字符串后 AST 仍不一致"))
        continue
    # 统计非 docstring 字符串改动
    d_old, _ = normalize(old)
    d_new, _ = normalize(new)
    if d_old != d_new:
        extra.append(f)
print(f"对比 {len(files)} 个文件")
print(f"代码结构异常（除去所有字符串后仍不同）：{len(bad)}")
for b in bad[:20]:
    print("  ", b)
print(f"含非 docstring 字符串改动的文件：{len(extra)}")
for e in extra[:15]:
    print("  ", e)

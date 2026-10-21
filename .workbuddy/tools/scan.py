import ast, re
from pathlib import Path
CJK = re.compile(r"[\u4e00-\u9fff]")
NODES = (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
files = [l.strip() for l in Path(".workbuddy/tools/changed_py.txt").read_text(encoding="utf-8").splitlines() if l.strip()]
leftover, crlf, total = [], [], 0
for f in files:
    p = Path(f)
    b = p.read_bytes()
    if b"\r\n" in b:
        crlf.append(f)
    try:
        tree = ast.parse(b.decode("utf-8"))
    except Exception:
        continue
    hit = None
    for n in ast.walk(tree):
        if isinstance(n, NODES):
            body = getattr(n, "body", None)
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) and isinstance(body[0].value.value, str):
                ds = body[0].value.value
                total += 1
                if not CJK.search(ds) and len(re.findall(r"[A-Za-z]{3,}", ds)) >= 4:
                    hit = ds.strip().splitlines()[0][:60] if ds.strip() else ""
    if hit is not None:
        leftover.append((f, hit))
print("检查文件", len(files), "docstring", total)
print("仍整段英文的文件", len(leftover))
for x in leftover[:20]: print("  ", x)
print("CRLF 文件", len(crlf), crlf[:5])

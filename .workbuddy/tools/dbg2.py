import os, sys
sys.path.insert(0, ".workbuddy/tools")
import zh_translate as Z
from pathlib import Path

p = Path("libs/partners/anthropic/langchain_anthropic/chat_models.py")
src = p.read_text(encoding="utf-8")
segs = Z.collect_doc_runs(src) + Z.collect_comments(src)
Z.translate_segments(segs, "qwen-flash", 3000)
new = Z.apply_py(src, segs)
Path(".workbuddy/tools/broken.py").write_text(new, encoding="utf-8", newline="")
import ast
try:
    ast.parse(new)
except SyntaxError as e:
    ln = e.lineno
    lines = new.splitlines()
    print("错误行", ln, e.msg)
    for i in range(max(1, ln-6), min(len(lines), ln+3)+1):
        print(f"{i:5d}| {lines[i-1]}")

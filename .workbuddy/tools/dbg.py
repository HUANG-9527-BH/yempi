import sys, ast
sys.path.insert(0, ".workbuddy/tools")
import zh_translate as Z
from pathlib import Path

p = Path("libs/partners/anthropic/langchain_anthropic/chat_models.py")
src = p.read_text(encoding="utf-8")
segs = Z.collect_doc_runs(src) + Z.collect_comments(src)
for s in segs:
    if s["start"] <= src.find("\n", 0) :
        pass
# 找 line 3073 附近的段
lines = src.splitlines(keepends=True)
offs = Z.line_starts(src)
target = offs[3072]
cand = [s for s in segs if s["start"] <= target <= s["end"] + 200]
print("候选段", len(cand))
for s in cand[:3]:
    ln = src[:s["start"]].count("\n") + 1
    print("---", s["kind"], "line", ln, "len", len(s["text"]))
    print(repr(s["text"][:300]))

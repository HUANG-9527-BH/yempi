#!/usr/bin/env python3
"""翻译 GitHub Issue 表单（YAML）里的人读文本。

只改 name/description/label/placeholder 的值与 `value: |` 块内的行；
逐行对应、缩进不变；译文中禁止出现 ": "，避免破坏 YAML 结构。
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import zh_translate as Z  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
SYS = """你是 LangChain 开源项目的中文译者，正在翻译 GitHub Issue 表单（YAML）里的用户可见文案。

规则：
1. 逐行翻译，输出行数必须与输入行数完全一致，顺序一一对应。
2. 译文中禁止出现英文冒号加空格（": "），需要冒号时用全角"："。
3. 保留 markdown 结构（列表符号、链接、粗体）、行内代码 `xxx`、URL、版本号、字段 id。
4. 保留 `FIXME`、`TODO`、`PR`、`CI`、`API` 等专有名词；不要翻译占位符里的路径。
5. 已经是中文的行原样返回。
术语：issue=议题，pull request=拉取请求，contributor=贡献者，checklist=检查清单，reproduction=复现。

只输出 JSON：{"<id>": "<译文，行之间用 \\n 连接>"}。"""

FIELDS = re.compile(r"^(\s*(?:-\s+)?)(name|description|label|placeholder|title):\s*(\S.*)$")
BLOCK = re.compile(r"^(\s*)(value|description|label|placeholder):\s*[|>][-+]?\s*$")


def indent_of(line: str) -> int:
    return len(line) - len(line.lstrip(" "))


def collect(lines: list[str]) -> list[dict]:
    segs = []
    i = 0
    while i < len(lines):
        line = lines[i].rstrip("\n")
        mb = BLOCK.match(line)
        if mb:
            base = len(mb.group(1))
            j = i + 1
            block = []
            while j < len(lines) and (not lines[j].strip() or indent_of(lines[j]) > base):
                block.append(lines[j].rstrip("\n"))
                j += 1
            while block and not block[-1].strip():
                block.pop()
                j -= 1
            body = [b.strip() for b in block]
            if any(re.search(r"[A-Za-z]{3}", b) for b in body):
                segs.append({"start": i + 1, "stop": j, "lines": block, "kind": "block"})
            i = j
            continue
        mf = FIELDS.match(line)
        if mf and not mf.group(3).startswith(("|", ">")):
            value = mf.group(3)
            quote = ""
            if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
                quote = value[0]
                value = value[1:-1]
            if re.search(r"[A-Za-z]{3}", value):
                segs.append(
                    {"start": i, "stop": i + 1, "lines": [value], "kind": "field",
                     "prefix": f"{mf.group(1)}{mf.group(2)}: ", "quote": quote}
                )
        i += 1
    return segs


def main() -> int:
    files = [ROOT / l.strip() for l in Path(sys.argv[1]).read_text(encoding="utf-8").splitlines() if l.strip()]
    model = sys.argv[2] if len(sys.argv) > 2 else "qwen-flash"
    for f in files:
        text = f.read_text(encoding="utf-8")
        lines = text.split("\n")
        segs = collect(lines)
        if not segs:
            print("跳过", f.name)
            continue
        mapping: dict[str, str] = {}
        queue = [segs[i : i + 4] for i in range(0, len(segs), 4)]
        tries: dict[int, int] = {}
        while queue:
            chunk = queue.pop(0)
            payload = [
                {"id": str(i + 1), "text": "\n".join(x["lines"])}
                for i, x in enumerate(chunk)
            ]
            try:
                raw = Z.api_call(model, SYS, json.dumps(payload, ensure_ascii=False))
                part = Z.parse_json_map(raw)
            except Exception as exc:  # noqa: BLE001
                print("  ! 调用失败", exc)
                part = {}
            missing = []
            for i, item in enumerate(chunk, start=1):
                got = part.get(str(i), "").strip()
                if got:
                    mapping[str(segs.index(item) + 1)] = got
                    continue
                tries[id(item)] = tries.get(id(item), 0) + 1
                if tries[id(item)] <= 3:
                    missing.append(item)
            queue[0:0] = [[m] for m in missing]

        out = list(lines)
        for i, s in enumerate(segs, start=1):
            t = mapping.get(str(i), "").strip("\n")
            if not t:
                continue
            t_lines = [x.rstrip() for x in t.split("\n")]
            if len(t_lines) != len(s["lines"]):
                print(f"  ! {f.name} 第 {s['start']+1} 行行数不符，保留原文")
                continue
            if any(": " in x for x in t_lines):
                print(f"  ! {f.name} 第 {s['start']+1} 行含 ': '，保留原文")
                continue
            if s["kind"] == "field":
                val = t_lines[0].strip()
                q = s["quote"]
                out[s["start"]] = f"{s['prefix']}{q}{val}{q}"
            else:
                for k, ln in enumerate(s["lines"]):
                    ind = ln[: len(ln) - len(ln.lstrip(" "))]
                    body = t_lines[k].strip()
                    out[s["start"] + k] = (ind + body) if body else ""
        f.write_text("\n".join(out), encoding="utf-8", newline="")
        print("完成", f.name, len(segs), "段")
    return 0


if __name__ == "__main__":
    sys.exit(main())

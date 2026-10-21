#!/usr/bin/env python3
"""LangChain 仓库中文化流水线。

只改写 docstring 与 ``#`` 注释，代码本体逐字节保留；每个文件改写后做 AST 骨架
比对，校验不通过则不落盘。

docstring 处理要点：
* doctest（``>>>`` / ``...`` 及其输出）与 Google 风格分节标题本地锁定，绝不外发；
* 其余按「段落块」翻译，块内保持行数/缩进，过长行自动折到 88 列。
"""

from __future__ import annotations

import argparse
import ast
import io
import json
import os
import re
import subprocess
import sys
import threading
import time
import tokenize
import unicodedata
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
STATE_DIR = ROOT / ".workbuddy" / "tools"
LOG_PATH = Path(os.environ.get("ZH_LOG", STATE_DIR / "state.jsonl"))
FAIL_PATH = LOG_PATH.with_name(LOG_PATH.stem + "_failures.jsonl")
OUT_DIR = STATE_DIR / "dryrun"

API_BASE = os.environ.get("ZH_API_BASE", "https://maas.qianwenaiapi.com/compatible-mode/v1")
API_KEY = os.environ.get("ZH_API_KEY", "")
WIDTH = 88
FROM_HEAD = False
_print_lock = threading.Lock()


def log(msg: str) -> None:
    with _print_lock:
        print(msg, flush=True)


# ---------------------------------------------------------------------------
# 基础工具
# ---------------------------------------------------------------------------

def line_starts(src: str) -> list[int]:
    starts = [0]
    for m in re.finditer("\n", src):
        starts.append(m.end())
    return starts


def char_col(line: str, byte_col: int) -> int:
    if byte_col <= 0:
        return 0
    raw = line.encode("utf-8")[:byte_col]
    try:
        return len(raw.decode("utf-8"))
    except UnicodeDecodeError:
        return len(raw.decode("utf-8", "ignore"))


def display_width(text: str) -> int:
    return sum(2 if unicodedata.east_asian_width(ch) in "WF" else 1 for ch in text)


DOC_NODES = (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)

SECTION_RE = re.compile(
    r"^\s*("
    r"Args|Arguments|Parameters|Other Parameters|Keyword Args|Keyword Arguments"
    r"|Returns|Return type|Yields|Raises|Warns"
    r"|Example|Examples|Note|Notes|Warning|Warnings|Attributes|Methods"
    r"|See Also|References|Todo|Tip|Tips|Usage|Deprecated"
    r")\s*:"
)

_SKIP_COMMENT = re.compile(
    r"^#\s*("
    r"!|-?\*-|coding[:=]|type:|noqa|pragma:|fmt:|ruff:|mypy:|pylint:|isort:"
    r"|coverage|pyright:|nosec|vim:|kate:|markdownlint|cspell|SPDX-"
    r"|copyright|licen[cs]e[: ]"
    r")",
    re.IGNORECASE,
)


def is_doctest_line(text: str) -> bool:
    s = text.strip()
    return s.startswith(">>>") or s.startswith("...")


# ---------------------------------------------------------------------------
# 抽取：docstring → 可译段落块
# ---------------------------------------------------------------------------

def _split_quoted(literal: str) -> tuple[str, str] | None:
    m = re.match(r"^([rRuU]{0,2})('''|\"\"\"|'|\")", literal)
    if not m:
        return None
    quote = m.group(2)
    if not literal.endswith(quote) or len(literal) < len(m.group(0)) + len(quote):
        return None
    return m.group(0), quote


def collect_doc_runs(src: str) -> list[dict]:
    """把 docstring 拆成可译块；锁定行（doctest / 分节标题）原地保留。"""
    src_lines = src.splitlines(keepends=True)
    starts = line_starts(src)
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return []

    runs: list[dict] = []
    targets: list[ast.Constant] = []
    for node in ast.walk(tree):
        if not isinstance(node, DOC_NODES):
            continue
        body = getattr(node, "body", None)
        if not body:
            continue
        for idx, stmt in enumerate(body):
            # 首个语句是标准 docstring；类体里的独立字符串是字段说明（Pydantic 会用作描述）
            if (
                isinstance(stmt, ast.Expr)
                and isinstance(stmt.value, ast.Constant)
                and isinstance(stmt.value.value, str)
                and (idx == 0 or isinstance(node, ast.ClassDef))
            ):
                targets.append(stmt.value)

    for value in targets:
        start = starts[value.lineno - 1] + char_col(src_lines[value.lineno - 1], value.col_offset)
        end = starts[value.end_lineno - 1] + char_col(
            src_lines[value.end_lineno - 1], value.end_col_offset
        )
        parts = _split_quoted(src[start:end])
        if not parts:
            continue
        prefix, quote = parts
        c_start = start + len(prefix)
        c_end = end - len(quote)
        content = src[c_start:c_end]
        if not content.strip():
            continue

        lines = []
        pos = c_start
        for raw in content.split("\n"):
            lines.append({"off": pos, "raw": raw})
            pos += len(raw) + 1

        locked_next = 0
        block: list[dict] = []

        def flush() -> None:
            if not block:
                return
            r_start = block[0]["off"]
            r_end = block[-1]["off"] + len(block[-1]["raw"])
            text = src[r_start:r_end]
            if re.search(r"[A-Za-z]{3}", text):
                runs.append(
                    {
                        "kind": "doc",
                        "quote": quote,
                        "start": r_start,
                        "end": r_end,
                        "text": text,
                        "lines": [
                            {
                                "indent": ln["raw"][: len(ln["raw"]) - len(ln["raw"].lstrip())],
                                "body": ln["raw"].strip(),
                            }
                            for ln in block
                        ],
                    }
                )
            block.clear()

        fence = False
        for ln in lines:
            raw = ln["raw"]
            s = raw.strip()
            if s.startswith("```") or s.startswith("~~~"):
                flush()
                fence = not fence
                locked_next = 0
                continue
            if fence:
                continue
            if is_doctest_line(s):
                flush()
                locked_next = 1
                continue
            if not s:
                flush()
                locked_next = 0
                continue
            if locked_next:
                continue
            if SECTION_RE.match(raw) or s.endswith("::"):
                flush()
                continue
            block.append(ln)
        flush()
    return runs


# ---------------------------------------------------------------------------
# 抽取：注释
# ---------------------------------------------------------------------------

def collect_comments(src: str) -> list[dict]:
    starts = line_starts(src)
    out: list[dict] = []
    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(src).readline))
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return []
    for tok in tokens:
        if tok.type != tokenize.COMMENT:
            continue
        text = tok.string
        row, col = tok.start
        base = starts[row - 1] + col
        if row == 1 and text.startswith("#!"):
            continue
        body = text[1:]
        stripped = body.strip()
        if not stripped or not re.search(r"[A-Za-z]{2}", stripped):
            continue
        if _SKIP_COMMENT.match(text):
            continue
        if re.fullmatch(r"[\w\-.:/ ]{0,4}", stripped):
            continue
        lead = body[: len(body) - len(body.lstrip())]
        prefix = src[starts[row - 1] : base]
        out.append(
            {
                "kind": "comment",
                "start": base,
                "end": base + len(text),
                "text": stripped,
                "lead": lead,
                "inline": bool(prefix.strip()),
                "indent": prefix[: len(prefix) - len(prefix.lstrip())],
            }
        )
    return out


# ---------------------------------------------------------------------------
# 提示词
# ---------------------------------------------------------------------------

SYSTEM_PY = """你是 LangChain 开源项目的资深中文译者，把 Python 代码里的 docstring 段落和注释译成简体中文。

硬性规则：
1. 只译自然语言。类名、函数名、参数名、模块路径、URL、命令、环境变量、配置键一律原样。
2. 输出行数必须与原文行数完全一致，顺序一一对应；原文在哪里换行，译文就在哪里换行。
3. markdown 行内代码 `xxx`、列表符号 `-`/`*`、编号 `1.`、括号结构原样保留。
4. 形如 `name: 描述` 的参数说明，冒号前的 `name` 原样保留，只译冒号后的描述。
5. 保留 `TODO`、`FIXME`、`XXX`、`NOTE`、`DEPRECATED`、`BETA` 标记词本身，其后说明译成中文。
6. 已经是中文的段落原样返回，不要改写、不要润色、不要重复翻译。
7. 专业、简洁的技术文档口吻。不要加译注、不要输出解释。

术语表（必须统一）：agent=智能体，chain=链，runnable=Runnable（不译），callback=回调，streaming=流式，prompt=提示词，template=模板，tool=工具，toolkit=工具集，message=消息，middleware=中间件，checkpoint=检查点，embedding=嵌入，vector store=向量存储，retriever=检索器，document loader=文档加载器，text splitter=文本分割器，trace=追踪，run=运行，invoke=调用，batch=批处理，cache=缓存，token=token（不译），schema=模式，serialization=序列化，deprecated=已弃用，beta=测试版，namespace=命名空间，payload=负载，runtime=运行时，parser=解析器，evaluator=评估器，output=输出。

只输出 JSON 对象：{"<id>": "<译文，用 \\n 分隔各行>"}，键与输入 id 完全一致。"""

SYSTEM_MD = """你是 LangChain 开源项目的资深中文技术文档译者，把给定的 Markdown 片段译成简体中文。

硬性规则：
1. 保持 Markdown 结构：标题 `#` 层级与数量、列表符号、表格分隔行、引用块、HTML 标签、徽章排布都不变。
2. `[文字](url)`：只译文字，url 一字不改。`![alt](src)`：alt 可译，src 不动。
3. `[[CODE_BLOCK_n]]` 占位符必须原样保留在原位置、单独成行，不要改写、不要包进新的代码块。
4. 行内代码 `xxx`、YAML frontmatter 整体、文件名、命令、包名、类名、参数名保持原样。
5. 段落数量与原文一致，不要合并或拆分段落。
6. 术语表（必须统一）：agent=智能体，chain=链，runnable=Runnable，callback=回调，streaming=流式，prompt=提示词，template=模板，tool=工具，middleware=中间件，checkpoint=检查点，embedding=嵌入，vector store=向量存储，retriever=检索器，trace=追踪，deprecated=已弃用。

只输出译文正文，不要包在 ``` 中，不要任何解释。"""


# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------

_opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def api_call(model: str, system: str, user: str, retries: int = 5) -> str:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": 0.2,
        "max_tokens": 8192,
    }
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    last = ""
    for attempt in range(retries):
        req = urllib.request.Request(
            f"{API_BASE.rstrip('/')}/chat/completions",
            data=data,
            headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
            method="POST",
        )
        try:
            with _opener.open(req, timeout=180) as resp:
                body = json.loads(resp.read().decode("utf-8"))
            return body["choices"][0]["message"]["content"]
        except urllib.error.HTTPError as exc:
            detail = exc.read()[:200].decode("utf-8", "ignore")
            last = f"HTTP {exc.code}: {detail}"
            if exc.code in (429, 500, 502, 503, 504):
                time.sleep(min(2**attempt * 2, 30))
                continue
            if exc.code in (401, 403):
                raise RuntimeError(last) from exc
            time.sleep(2)
        except Exception as exc:  # noqa: BLE001
            last = f"{type(exc).__name__}: {exc}"
            time.sleep(min(2**attempt * 2, 30))
    raise RuntimeError(last)


def parse_json_map(raw: str) -> dict[str, str]:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```[a-zA-Z]*\n?", "", raw)
        raw = re.sub(r"\n?```$", "", raw)
    start, end = raw.find("{"), raw.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("响应里找不到 JSON 对象")
    obj = json.loads(raw[start : end + 1])
    return {str(k): (v if isinstance(v, str) else " ".join(map(str, v))) for k, v in obj.items()}


# ---------------------------------------------------------------------------
# 回填
# ---------------------------------------------------------------------------

def wrap_doc(text: str, indent: str) -> str:
    """把一段中文按 88 列折行。"""
    limit = max(WIDTH - display_width(indent), 40)
    out: list[str] = []
    for para in text.split("\n"):
        para = para.strip()
        if not para:
            out.append("")
            continue
        lines, cur, width = [], "", 0
        for token in re.findall(r"\S+\s*", para):
            tw = display_width(token)
            if width and width + tw > limit:
                lines.append(cur.rstrip())
                cur, width = token, tw
            else:
                cur += token
                width += tw
        if cur.strip():
            lines.append(cur.rstrip())
        out.extend(lines)
    return "\n".join(out)


CODE_HINT = re.compile(
    r"(^|\s)(>>>|\.\.\.|@|def |class |import |from |return |yield |async |await "
    r"|if |for |while |with |try:|except|raise )|\w+\s*=\s*\S|\)\s*$|\{.*:"
)


def looks_code(text: str) -> bool:
    return bool(CODE_HINT.search(text))


def _guard_identifier(orig_body: str, translated: str) -> str:
    """`name: 描述` 形式，强制保留冒号前的标识符。"""
    m = re.match(r"^([A-Za-z_][\w\.\*\[\], ]{0,60})\s*:\s+\S", orig_body)
    if not m:
        return translated
    ident = m.group(1).strip()
    tm = re.match(r"^([^\s:：]{1,60})\s*[:：]\s*", translated)
    if tm and tm.group(1).strip() == ident:
        return ident + ": " + translated[tm.end() :]
    return f"{ident}: {translated}"


def rebuild_doc(seg: dict, translated: str) -> str:
    o_lines = seg["lines"]
    t_lines = [ln.strip() for ln in translated.strip("\n").split("\n")]
    while t_lines and not t_lines[0]:
        t_lines.pop(0)
    while t_lines and not t_lines[-1]:
        t_lines.pop()
    if not t_lines:
        return seg["text"]

    if len(t_lines) == len(o_lines):
        out = []
        for o, t in zip(o_lines, t_lines):
            out.append("" if not t else o["indent"] + _guard_identifier(o["body"], t))
        return "\n".join(out)

    if looks_code(seg["text"]):
        return seg["text"]

    indent = o_lines[0]["indent"]
    joined = " ".join(t_lines)
    wrapped = wrap_doc(joined, indent)
    return "\n".join(indent + ln if ln else "" for ln in wrapped.split("\n"))


def rebuild_comment(seg: dict, translated: str) -> str:
    """单行注释必须留在一行内；独占一行的注释可以按宽度折行。"""
    body = " ".join(ln.strip() for ln in translated.split("\n") if ln.strip())
    if not body:
        return "#" + seg["lead"]
    if seg["inline"]:
        return "#" + seg["lead"] + body
    head = seg["indent"] + "#" + seg["lead"]
    limit = max(WIDTH - display_width(head), 40)
    lines, cur, width = [], "", 0
    for token in re.findall(r"\S+\s*", body):
        tw = display_width(token)
        if width and width + tw > limit:
            lines.append(cur.rstrip())
            cur, width = token, tw
        else:
            cur += token
            width += tw
    if cur.strip():
        lines.append(cur.rstrip())
    cont = seg["indent"] + "#" + seg["lead"]
    return "\n".join([head + lines[0]] + [cont + ln for ln in lines[1:]])


def apply_py(src: str, segments: list[dict]) -> str:
    for seg in sorted(segments, key=lambda s: s["start"], reverse=True):
        if "translated" not in seg:
            continue
        if seg["kind"] == "doc":
            repl = rebuild_doc(seg, seg["translated"])
            quote = seg.get("quote", "")
            if quote and quote in repl:
                repl = repl.replace(quote, "".join("\\" + ch for ch in quote))
        else:
            repl = rebuild_comment(seg, seg["translated"])
        src = src[: seg["start"]] + repl + src[seg["end"] :]
    return src


def skeleton(src: str) -> str:
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, DOC_NODES):
            body = getattr(node, "body", None)
            if not body:
                continue
            first = body[0]
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str):
                first.value.value = "<DOC>"
            if isinstance(node, ast.ClassDef):
                for stmt in body[1:]:
                    if (
                        isinstance(stmt, ast.Expr)
                        and isinstance(stmt.value, ast.Constant)
                        and isinstance(stmt.value.value, str)
                    ):
                        stmt.value.value = "<DOC>"
    return ast.dump(tree)


# ---------------------------------------------------------------------------
# 批量翻译
# ---------------------------------------------------------------------------

def chunk_segments(segments: list[dict], budget: int) -> list[list[dict]]:
    chunks, cur, size = [], [], 0
    for seg in segments:
        n = len(seg["text"])
        if cur and size + n > budget:
            chunks.append(cur)
            cur, size = [], 0
        cur.append(seg)
        size += n
    if cur:
        chunks.append(cur)
    return chunks


def translate_segments(segments: list[dict], model: str, budget: int) -> int:
    calls = 0
    attempts: dict[int, int] = {}
    pending = chunk_segments(segments, budget)
    while pending:
        chunk = pending.pop(0)
        payload = [
            {"id": str(i + 1), "kind": s["kind"], "text": s["text"]}
            for i, s in enumerate(chunk)
        ]
        try:
            raw = api_call(model, SYSTEM_PY, json.dumps(payload, ensure_ascii=False))
        except RuntimeError as exc:
            log(f"    ! 调用失败，跳过 {len(chunk)} 段：{exc}")
            continue
        calls += 1
        try:
            mapping = parse_json_map(raw)
        except (ValueError, json.JSONDecodeError):
            mapping = {}
        retry: list[dict] = []
        for i, seg in enumerate(chunk, start=1):
            got = mapping.get(str(i), "").strip()
            if got and re.search(r"[A-Za-z\u4e00-\u9fff]", got):
                seg["translated"] = got
                continue
            attempts[id(seg)] = attempts.get(id(seg), 0) + 1
            if attempts[id(seg)] > 3:
                continue
            retry.append(seg)
        if retry:
            if len(chunk) > 1 and len(retry) == len(chunk):
                mid = max(len(chunk) // 2, 1)
                pending.insert(0, chunk[mid:])
                pending.insert(0, chunk[:mid])
            else:
                pending.extend([s] for s in retry)
    return calls


# ---------------------------------------------------------------------------
# 单文件
# ---------------------------------------------------------------------------

def process_py(path: Path, model: str, budget: int, dry: bool) -> dict:
    if FROM_HEAD:
        rel = path.relative_to(ROOT).as_posix()
        raw = subprocess.run(
            ["git", "show", f"HEAD:{rel}"], capture_output=True, cwd=str(ROOT)
        ).stdout
        src = raw.decode("utf-8")
    else:
        src = path.read_text(encoding="utf-8")
    orig_skel = skeleton(src)
    segments = collect_doc_runs(src) + collect_comments(src)
    if not segments:
        return {"path": str(path), "ok": True, "seg": 0, "calls": 0}
    calls = translate_segments(segments, model, budget)
    new_src = apply_py(src, segments)
    try:
        new_skel = skeleton(new_src)
    except SyntaxError as exc:
        return {
            "path": str(path),
            "ok": False,
            "seg": len(segments),
            "calls": calls,
            "err": f"语法错误: {exc}",
        }
    if new_skel != orig_skel:
        return {
            "path": str(path),
            "ok": False,
            "seg": len(segments),
            "calls": calls,
            "err": "AST 骨架不一致",
        }
    if new_src == src:
        return {"path": str(path), "ok": True, "seg": len(segments), "calls": calls, "unchanged": True}
    if dry:
        out = OUT_DIR / path.relative_to(ROOT)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(new_src, encoding="utf-8", newline="")
    else:
        path.write_text(new_src, encoding="utf-8", newline="")
    return {"path": str(path), "ok": True, "seg": len(segments), "calls": calls}


FENCE = re.compile(r"^[ \t]*```.*?^[ \t]*```[ \t]*$", re.DOTALL | re.MULTILINE)


def process_md(path: Path, model: str, budget: int, dry: bool) -> dict:
    text = path.read_text(encoding="utf-8")
    front, body = "", text
    m = re.match(r"^---\n.*?\n---\n", text, re.DOTALL)
    if m:
        front, body = m.group(0), text[len(m.group(0)) :]

    blocks: list[str] = []

    def stash(match: re.Match[str]) -> str:
        blocks.append(match.group(0))
        return f"\n[[CODE_BLOCK_{len(blocks)}]]\n"

    stashed = FENCE.sub(stash, body)
    lines = stashed.split("\n")
    chunks: list[list[str]] = []
    cur: list[str] = []
    size = 0
    for line in lines:
        if cur and size + len(line) > budget and not line.strip():
            chunks.append(cur)
            cur, size = [], 0
        cur.append(line)
        size += len(line) + 1
    if cur:
        chunks.append(cur)

    calls = 0
    parts: list[str] = []
    for chunk in chunks:
        chunk_text = "\n".join(chunk)
        if not re.search(r"[A-Za-z]{3}", chunk_text):
            parts.append(chunk_text)
            continue
        try:
            raw = api_call(model, SYSTEM_MD, chunk_text)
            calls += 1
        except RuntimeError as exc:
            log(f"    ! {path.name} 片段翻译失败，保留原文：{exc}")
            parts.append(chunk_text)
            continue
        parts.append(raw.strip())

    new_body = "\n\n".join(p for p in parts if p.strip())
    for i, blk in enumerate(blocks, start=1):
        new_body = re.sub(rf"\[\[CODE_BLOCK_{i}\]\]", lambda _m, b=blk: b, new_body)
    new_text = front + new_body + ("\n" if text.endswith("\n") else "")
    if dry:
        out = OUT_DIR / path.relative_to(ROOT)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(new_text, encoding="utf-8", newline="")
    else:
        path.write_text(new_text, encoding="utf-8", newline="")
    return {"path": str(path), "ok": True, "seg": len(chunks), "calls": calls}


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def load_done() -> set[str]:
    done: set[str] = set()
    if LOG_PATH.exists():
        for line in LOG_PATH.read_text(encoding="utf-8").splitlines():
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("ok"):
                done.add(rec["path"])
    return done


def append_log(rec: dict) -> None:
    with _print_lock:
        with LOG_PATH.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        if not rec.get("ok"):
            with FAIL_PATH.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["py", "md"], required=True)
    ap.add_argument("--files-from", required=True)
    ap.add_argument("--model", default="qwen-flash")
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--budget", type=int, default=3000)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--from-head", action="store_true", help="以 git HEAD 版本为源重译")
    args = ap.parse_args()

    global FROM_HEAD
    FROM_HEAD = args.from_head
    if not API_KEY:
        log("缺少 ZH_API_KEY")
        return 2

    STATE_DIR.mkdir(parents=True, exist_ok=True)
    targets = [
        ROOT / ln.strip()
        for ln in Path(args.files_from).read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]
    targets = [p for p in targets if p.exists()]
    if not args.no_resume:
        done = load_done()
        targets = [p for p in targets if str(p) not in done]
    if args.limit:
        targets = targets[: args.limit]

    log(f"模式={args.mode} 模型={args.model} 文件={len(targets)} 并发={args.jobs}")
    handler = process_py if args.mode == "py" else process_md

    ok = fail = unchanged = segs = calls = 0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = {pool.submit(handler, p, args.model, args.budget, args.dry): p for p in targets}
        for idx, fut in enumerate(as_completed(futures), start=1):
            path = futures[fut]
            try:
                rec = fut.result()
            except Exception as exc:  # noqa: BLE001
                rec = {"path": str(path), "ok": False, "err": f"{type(exc).__name__}: {exc}"}
            append_log(rec)
            if rec.get("ok"):
                ok += 1
                unchanged += 1 if rec.get("unchanged") else 0
                segs += rec.get("seg", 0)
                calls += rec.get("calls", 0)
            else:
                fail += 1
                log(f"[{idx}] 失败 {path.relative_to(ROOT)} :: {rec.get('err')}")
            if idx % 50 == 0:
                log(
                    f"  {idx}/{len(targets)} ok={ok} fail={fail} 未变={unchanged} "
                    f"段={segs} 调用={calls} {time.time() - t0:.0f}s"
                )
    log(f"完成 ok={ok} fail={fail} 未变={unchanged} 段={segs} 调用={calls} 用时={time.time() - t0:.0f}s")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

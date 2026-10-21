"""生成模型配置变更的简洁英文摘要。

The `refresh_model_profiles` workflow opens an automated PR whenever the data
behind `_profiles.py` files changes. Those diffs are large blocks of generated
data, so a reviewer otherwise has to open *Files changed* and eyeball raw values
to learn what actually moved. This module turns the structured before/after data
into a skimmable Markdown summary (new models, removed models, and per-field
capability/metadata changes) for the PR body. The summary is generated
deterministically from the data, so there is no risk of an LLM misdescribing it.
"""

from __future__ import annotations

import ast
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple, TypedDict

if TYPE_CHECKING:
    from collections.abc import Mapping

    from langchain_core.language_models.model_profile import (
        ModelProfile,
        ModelProfileRegistry,
    )

# 单个部分最多渲染的项目行数，超过则截断。
_MAX_ROWS = 25

# 配置字段的人类可读标签。
_FIELD_LABELS: dict[str, str] = {
    "name": "display name",
    "status": "status",
    "release_date": "release date",
    "last_updated": "last updated",
    "open_weights": "open weights",
    "max_input_tokens": "max input tokens",
    "max_output_tokens": "max output tokens",
    "text_inputs": "text input",
    "image_inputs": "image input",
    "audio_inputs": "audio input",
    "pdf_inputs": "PDF input",
    "video_inputs": "video input",
    "text_outputs": "text output",
    "image_outputs": "image output",
    "audio_outputs": "audio output",
    "video_outputs": "video output",
    "reasoning_output": "reasoning",
    "tool_calling": "tool calling",
    "tool_choice": "tool choice",
    "tool_call_streaming": "tool call streaming",
    "structured_output": "structured output",
    "attachment": "attachments",
    "temperature": "temperature control",
    "image_url_inputs": "image URL input",
    "image_tool_message": "image tool messages",
    "pdf_tool_message": "PDF tool messages",
}

# 以千位分隔符显示 token 字段。
_TOKEN_FIELDS = frozenset({"max_input_tokens", "max_output_tokens"})


class ProfileParseError(ValueError):
    """_profiles.py 源文件存在，但其 `_PROFILES` 数据无法解析。

    Distinguished from a genuinely absent file (which yields an empty mapping)
    so that corrupt working-tree or committed data surfaces as an error rather
    than being silently diffed as a mass addition or removal of models.
    """


class FieldChange(NamedTuple):
    """单个已更改配置字段的旧值和新值。

    使用命名而非裸 `tuple`，以使渲染器依赖的旧 →
    新顺序成为类型的一部分，而非位置约定。值为异构的配置数据（bool、int、str
    或未设置），因此 `Any` 是此处最诚实的元素类型。
    """

    old: Any
    """刷新前的值，若不存在或未设置则为 `None`。"""

    new: Any
    """刷新后的值，若不存在或未设置则为 `None`。"""


class ProviderEntry(TypedDict):
    """一个提供方的身份及其持有 `_profiles.py` 的数据目录。"""

    provider: str
    """提供方标识符（例如 `'openai'`）。"""

    data_dir: str
    """相对于仓库根目录的提供方数据目录路径。"""


@dataclass
class ProfileDiff:
    """两组模型配置之间的结构化差异。"""

    added: list[str] = field(default_factory=list)
    """刷新后存在但刷新前不存在的模型 ID，已排序。"""

    removed: list[str] = field(default_factory=list)
    """刷新前存在但刷新后不存在的模型 ID，已排序。"""

    changed: dict[str, dict[str, FieldChange]] = field(default_factory=dict)
    """按模型 ID 和字段名索引的每模型字段变更。"""

    added_profiles: ModelProfileRegistry = field(default_factory=dict)
    """每个新增模型的完整配置，按模型 ID 索引。"""

    @property
    def is_empty(self) -> bool:
        """是否存在无模型新增、删除或字段变更的情况。"""
        return not (self.added or self.removed or self.changed)


def extract_profiles(source: str) -> ModelProfileRegistry:
    """从 `_profiles.py` 源中提取 `_PROFILES` 映射。

    使用 `ast.literal_eval`
    而非导入/执行模块，以确保生成的数据文件永远不会作为代码运行。

    Args:
        source: `_profiles.py` 模块的内容。空字符串（例如在 git
        提交点上文件不存在）会产生空映射。

    Returns:
        `_PROFILES` 映射，若源中无 `_PROFILES` 赋值则返回空字典。

    Raises:
        ProfileParseError: If the source is present but cannot be parsed, or its
            `_PROFILES` value is not a dict literal. Surfacing this rather than
            returning `{}` prevents a corrupt file from being misreported as
            every model added or removed.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        msg = f"Could not parse profile source as Python: {e}"
        raise ProfileParseError(msg) from e

    for node in tree.body:
        if isinstance(node, ast.AnnAssign):
            targets: list[ast.expr] = [node.target]
        elif isinstance(node, ast.Assign):
            targets = list(node.targets)
        else:
            continue
        is_profiles = any(
            isinstance(t, ast.Name) and t.id == "_PROFILES" for t in targets
        )
        if is_profiles and node.value is not None:
            try:
                value = ast.literal_eval(node.value)
            except (ValueError, SyntaxError) as e:
                msg = f"`_PROFILES` is not a literal expression: {e}"
                raise ProfileParseError(msg) from e
            if not isinstance(value, dict):
                msg = f"`_PROFILES` is not a dict (got {type(value).__name__})"
                raise ProfileParseError(msg)
            return value
    return {}


def diff_profiles(old: ModelProfileRegistry, new: ModelProfileRegistry) -> ProfileDiff:
    """计算两个 `_PROFILES` 映射之间的差异。

    Args:
        old: 刷新前的配置。
        new: 刷新后的配置。

    Returns:
        描述新增、移除及变更模型的 `ProfileDiff`。
    """
    added = sorted(set(new) - set(old))
    removed = sorted(set(old) - set(new))

    changed: dict[str, dict[str, FieldChange]] = {}
    for model_id in sorted(set(old) & set(new)):
                # 将配置视作普通映射以便遍历动态键（因为 `ModelProfile` TypedDict
        # 仅允许字面量键访问）。
                # 仅当多个提供商变更时，才将每个部分包裹在 <details>
        # 切换标签中，以确保单提供商摘要保持扁平。
        old_profile: Mapping[str, Any] = old[model_id]
        new_profile: Mapping[str, Any] = new[model_id]
        fields: dict[str, FieldChange] = {}
        for key in sorted(set(old_profile) | set(new_profile)):
            old_val = old_profile.get(key)
            new_val = new_profile.get(key)
            if old_val != new_val:
                fields[key] = FieldChange(old_val, new_val)
        if fields:
            changed[model_id] = fields

    added_profiles = {model_id: new[model_id] for model_id in added}
    return ProfileDiff(
        added=added,
        removed=removed,
        changed=changed,
        added_profiles=added_profiles,
    )


def _format_value(field_name: str, value: Any) -> str:  # noqa: ANN401
    """渲染单个字段值以供显示。"""
    if value is None:
        return "unset"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int) and field_name in _TOKEN_FIELDS:
        return f"{value:,}"
    return f"`{value}`" if isinstance(value, str) else str(value)


def _describe_field_change(
    field_name: str,
    old_val: Any,  # noqa: ANN401
    new_val: Any,  # noqa: ANN401
) -> str:
    """为单个字段变更生成一段简洁的英文描述。"""
    label = _FIELD_LABELS.get(field_name, field_name)
    if isinstance(old_val, bool) or isinstance(new_val, bool):
        if new_val and not old_val:
            return f"added {label}"
        if old_val and not new_val:
            return f"removed {label}"
    old_str = _format_value(field_name, old_val)
    new_str = _format_value(field_name, new_val)
    return f"{label} {old_str} → {new_str}"


def _describe_new_model(profile: ModelProfile) -> str:
    """为新添加的模型生成一个简短描述符。"""
    parts: list[str] = []
    context = profile.get("max_input_tokens")
    if context:
        parts.append(f"{context:,} ctx")
    output = profile.get("max_output_tokens")
    if output:
        parts.append(f"{output:,} out")
    modalities = [
        name
        for key, name in (
            ("image_inputs", "image"),
            ("audio_inputs", "audio"),
            ("video_inputs", "video"),
            ("pdf_inputs", "pdf"),
        )
        if profile.get(key)
    ]
    if modalities:
        parts.append("text+" + "+".join(modalities) + " in")
    if profile.get("reasoning_output"):
        parts.append("reasoning")
    if profile.get("tool_calling"):
        parts.append("tools")
    return ", ".join(parts)


def _truncate(rows: list[str]) -> list[str]:
    """限制列表的条目数量，截断时附加省略号行。"""
    if len(rows) <= _MAX_ROWS:
        return rows
    hidden = len(rows) - _MAX_ROWS
    return [*rows[:_MAX_ROWS], f"- …and {hidden} more"]


def render_provider_section(provider: str, diff: ProfileDiff) -> str | None:
    """渲染单个提供方的 Markdown 部分，若未变更则返回 None。

    Args:
        provider: 提供商标识符（例如 `'openai'`）。
        diff: 该提供商的计算 `ProfileDiff`。

    Returns:
        提供商变更的 Markdown 内容，若无变更则为 `None`。
    """
    if diff.is_empty:
        return None

    lines = [f"### {provider}"]

    if diff.added:
        lines.append(f"\n**➕ {len(diff.added)} added**")  # noqa: RUF001
        rows = []
        for model_id in diff.added:
            descriptor = _describe_new_model(diff.added_profiles[model_id])
            suffix = f" — {descriptor}" if descriptor else ""
            rows.append(f"- `{model_id}`{suffix}")
        lines.extend(_truncate(rows))

    if diff.removed:
        lines.append(f"\n**➖ {len(diff.removed)} removed**")  # noqa: RUF001
        lines.extend(_truncate([f"- `{m}`" for m in diff.removed]))

    if diff.changed:
        lines.append(f"\n**✏️ {len(diff.changed)} changed**")
        rows = []
        for model_id, fields in diff.changed.items():
            phrases = [
                _describe_field_change(name, change.old, change.new)
                for name, change in fields.items()
            ]
            rows.append(f"- `{model_id}`: " + "; ".join(phrases))
        lines.extend(_truncate(rows))

    return "\n".join(lines)


def build_summary(provider_diffs: dict[str, ProfileDiff]) -> str:
    """跨所有提供商汇总完整的 Markdown 摘要。

    当多个提供商有变更时，每个提供商的部分均包裹在 `<details>` 切换标签中，以保持 PR
    正文可快速浏览。

    Args:
        provider_diffs: 提供商名称到其 `ProfileDiff` 的映射。

    Returns:
        Markdown 摘要。若无任何变更，则返回简短说明。
    """
    provider_sections = [
        (provider, section)
        for provider in sorted(provider_diffs)
        if (section := render_provider_section(provider, provider_diffs[provider]))
    ]
    if not provider_sections:
        return "No model profile data changed."

    total_added = sum(len(d.added) for d in provider_diffs.values())
    total_removed = sum(len(d.removed) for d in provider_diffs.values())
    total_changed = sum(len(d.changed) for d in provider_diffs.values())
    headline = (
        f"**{total_added} added · {total_removed} removed · "
        f"{total_changed} changed** across {len(provider_sections)} provider(s)."
    )

        # 移除 "### {provider}" 标题，使 <summary> 标签成为切换的唯一标签。
        # 将条目视作不可信映射：在 CLI 边界处，它是任意解析的 JSON，而非保证的
    # `ProviderEntry`。
    wrap = len(provider_sections) > 1
    sections = []
    for provider, section in provider_sections:
        if not wrap:
            sections.append(section)
            continue
                # 损坏但可读的文件必须作为错误暴露：从中提取 `{}`
        # 会将其误判为所有模型新增（旧侧）或移除（新侧），导致生成看似可信但错误的摘要。
                # 切换按钮的标签。
        body = section.removeprefix(f"### {provider}").lstrip("\n")
        sections.append(
            f"<details>\n<summary>{provider}</summary>\n\n{body}\n\n</details>"
        )

    return "\n\n".join(["## Summary of changes", headline, *sections])


def _verify_ref(repo_root: Path, ref: str) -> None:
    """确认 `ref` 在 `repo_root` 中可解析为提交。

    Validating once up front lets `_git_show` treat a non-zero exit
    unambiguously as "path absent at this ref", rather than conflating a typo'd
    ref, an unfetched ref, or a non-repository root with a genuinely new file —
    which would otherwise render every existing model as newly added.

    Raises:
        RuntimeError: 若 git 不可用，`repo_root` 非仓库，或 `ref` 无法解析。
    """
    try:
        result = subprocess.run(  # noqa: S603
            [  # noqa: S607
                "git",
                "-C",
                str(repo_root),
                "rev-parse",
                "--verify",
                "--quiet",
                f"{ref}^{{commit}}",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError as e:
        msg = f"Could not run git (is it installed and on PATH?): {e}"
        raise RuntimeError(msg) from e
    if result.returncode != 0:
        msg = (
            f"Could not resolve base ref {ref!r} in {repo_root}; "
            "is it a valid git ref in this repository?"
        )
        raise RuntimeError(msg)


def _git_show(repo_root: Path, ref: str, rel_path: str) -> str | None:
    """返回 `ref` 处的文件内容，若文件不存在则返回 None。

    假设 `ref` 已由 `_verify_ref` 验证，因此此处非零退出码表示路径在 `ref`
    下不存在，而非引用错误。
    """
    try:
        result = subprocess.run(  # noqa: S603
            ["git", "-C", str(repo_root), "show", f"{ref}:{rel_path}"],  # noqa: S607
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return None
    return result.stdout if result.returncode == 0 else None


def summarize(
    providers: list[ProviderEntry],
    *,
    base_ref: str = "HEAD",
    repo_root: Path | None = None,
) -> str:
    """构建相对于 `base_ref` 的配置文件变更的 Markdown 摘要。

    Args:
        providers: List of `{'provider': ..., 'data_dir': ...}` entries,
            matching the workflow input. `data_dir` is relative to the repo
            root and contains `_profiles.py`.
        base_ref: Git ref to compare the working tree against.
        repo_root: Repository root. Defaults to the current directory.

    Returns:
        适用于 PR 正文的 Markdown 摘要。

    Raises:
        RuntimeError: 若 `base_ref` 无法解析，或存在配置文件但无法读取或解析。
        ValueError: 若 `providers` 条目缺少必需键。 TypeError: 若 `providers` 条目的
        `provider`/`data_dir` 非字符串。
    """
    root = (repo_root or Path.cwd()).resolve()
    _verify_ref(root, base_ref)
    provider_diffs: dict[str, ProfileDiff] = {}

    for entry in providers:
                # 将条目视为不受信任的映射：在 CLI 边界处
                # 任意解析的 JSON，不保证是 `ProviderEntry`。
        entry_map: Mapping[str, Any] = entry
        try:
            provider = entry_map["provider"]
            data_dir = entry_map["data_dir"]
        except (KeyError, TypeError) as e:
            msg = (
                f"Invalid provider entry {entry!r}: expected 'provider' and "
                f"'data_dir' keys ({e})"
            )
            raise ValueError(msg) from e
        if not isinstance(provider, str) or not isinstance(data_dir, str):
            msg = (
                f"Invalid provider entry {entry!r}: 'provider' and 'data_dir' "
                "must be strings"
            )
            raise TypeError(msg)
        rel_path = f"{data_dir.rstrip('/')}/_profiles.py"

        old_source = _git_show(root, base_ref, rel_path) or ""
        new_path = root / rel_path
        if new_path.exists():
            try:
                new_source = new_path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError) as e:
                msg = f"Could not read {new_path}: {e}"
                raise RuntimeError(msg) from e
        else:
            new_source = ""

                # 损坏但可读的文件必须作为错误抛出：提取 `{}`
                # 从它将被逐个模型添加（旧侧）或
                # 移除（新侧边），导致得出一个自信但错误的摘要。
        try:
            old_profiles = extract_profiles(old_source)
        except ProfileParseError as e:
            msg = f"Profile data for {provider!r} at {base_ref!r} is unparseable: {e}"
            raise RuntimeError(msg) from e
        try:
            new_profiles = extract_profiles(new_source)
        except ProfileParseError as e:
            msg = f"Profile data for {provider!r} at {new_path} is unparseable: {e}"
            raise RuntimeError(msg) from e

        provider_diffs[provider] = diff_profiles(old_profiles, new_profiles)

    return build_summary(provider_diffs)

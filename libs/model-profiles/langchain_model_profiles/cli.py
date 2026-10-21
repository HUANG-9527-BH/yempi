"""用于从 models.dev 刷新模型资料数据的命令行接口"""

import argparse
import json
import re
import sys
import tempfile
import warnings
from pathlib import Path
from typing import Any, get_type_hints

import httpx

# 使用 `sys.version_info` 保护而非 try/except：在严格的 mypy 环境下，
# `warn_unused_ignores` 无法在版本间通过 try/except 满足
# (3.10 需要在 `tomllib` 上添加 `import-not-found` 忽略；3.11+ 会将其标记为未使用)。
# mypy 会根据目标版本解析此保护，并将其排除在
# `warn_unreachable` 之外，因此无需忽略即可通过类型检查。
if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib


def _validate_data_dir(data_dir: Path) -> Path:
    """验证并规范化数据目录路径

    Args:
        data_dir: 用户提供的数据目录路径。

    Returns:
        解析后的规范路径。

    Raises:
        SystemExit: 如果用户拒绝在当前目录外写入，则抛出 SystemExit。
    """
        # 解析为绝对、规范路径（遵循符号链接）
    try:
        resolved = data_dir.resolve(strict=False)
    except (OSError, RuntimeError) as e:
        msg = f"Invalid data directory path: {e}"
        print(f"❌ {msg}", file=sys.stderr)
        sys.exit(1)

        # 如果在当前目录外写入则发出警告
    cwd = Path.cwd().resolve()
    try:
        resolved.relative_to(cwd)
    except ValueError:
                # 不相对于 cwd
        print("⚠️  WARNING: Writing outside current directory", file=sys.stderr)
        print(f"   Current directory: {cwd}", file=sys.stderr)
        print(f"   Target directory:  {resolved}", file=sys.stderr)
        print(file=sys.stderr)
        response = input("Continue? (y/N): ")
        if response.lower() != "y":
            print("Aborted.", file=sys.stderr)
            sys.exit(1)

    return resolved


def _load_augmentations(
    data_dir: Path,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """从 `profile_augmentations.toml` 加载扩充数据。

    Args:
        data_dir: 包含 `profile_augmentations.toml` 的目录。

    Returns:
        二元组 `(provider_augmentations, model_augmentations)`。
    """
    aug_file = data_dir / "profile_augmentations.toml"
    if not aug_file.exists():
        return {}, {}

    try:
        with aug_file.open("rb") as f:
            data = tomllib.load(f)
    except PermissionError:
        msg = f"Permission denied reading augmentations file: {aug_file}"
        print(f"❌ {msg}", file=sys.stderr)
        sys.exit(1)
    except tomllib.TOMLDecodeError as e:
        msg = f"Invalid TOML syntax in augmentations file: {e}"
        print(f"❌ {msg}", file=sys.stderr)
        sys.exit(1)
    except OSError as e:
        msg = f"Failed to read augmentations file: {e}"
        print(f"❌ {msg}", file=sys.stderr)
        sys.exit(1)

    overrides = data.get("overrides", {})
    provider_aug: dict[str, Any] = {}
    model_augs: dict[str, dict[str, Any]] = {}

    for key, value in overrides.items():
        if isinstance(value, dict):
            model_augs[key] = value
        else:
            provider_aug[key] = value

    return provider_aug, model_augs


def _model_data_to_profile(model_data: dict[str, Any]) -> dict[str, Any]:
    """将原始 models.dev 数据转换为规范的资料结构。"""
    limit = model_data.get("limit") or {}
    modalities = model_data.get("modalities") or {}
    input_modalities = modalities.get("input") or []
    output_modalities = modalities.get("output") or []

    profile = {
        "name": model_data.get("name"),
        "status": model_data.get("status"),
        "release_date": model_data.get("release_date"),
        "last_updated": model_data.get("last_updated"),
        "open_weights": model_data.get("open_weights"),
        "max_input_tokens": limit.get("context"),
        "max_output_tokens": limit.get("output"),
        "text_inputs": "text" in input_modalities,
        "image_inputs": "image" in input_modalities,
        "audio_inputs": "audio" in input_modalities,
        "pdf_inputs": "pdf" in input_modalities or model_data.get("pdf_inputs"),
        "video_inputs": "video" in input_modalities,
        "text_outputs": "text" in output_modalities,
        "image_outputs": "image" in output_modalities,
        "audio_outputs": "audio" in output_modalities,
        "video_outputs": "video" in output_modalities,
        "reasoning_output": model_data.get("reasoning"),
        "tool_calling": model_data.get("tool_call"),
        "tool_choice": model_data.get("tool_choice"),
        "tool_call_streaming": model_data.get("tool_call_streaming"),
        "structured_output": model_data.get("structured_output"),
        "attachment": model_data.get("attachment"),
        "temperature": model_data.get("temperature"),
        "image_url_inputs": model_data.get("image_url_inputs"),
        "image_tool_message": model_data.get("image_tool_message"),
        "pdf_tool_message": model_data.get("pdf_tool_message"),
    }

    return {k: v for k, v in profile.items() if v is not None}


def _apply_overrides(
    profile: dict[str, Any], *overrides: dict[str, Any] | None
) -> dict[str, Any]:
    """将提供者和模型的覆盖项合并到规范资料中。"""
    merged = dict(profile)
    for override in overrides:
        if not override:
            continue
        for key, value in override.items():
            if value is not None:
                merged[key] = value  # noqa: PERF403
    return merged


def _warn_undeclared_profile_keys(
    profiles: dict[str, dict[str, Any]],
) -> None:
    """如果任何资料键未在 `ModelProfile` 中声明，则发出警告。

    Args:
        profiles: 模型 ID 到其资料字典的映射。
    """
    try:
        from langchain_core.language_models.model_profile import ModelProfile
    except ImportError:
                # langchain-core 可能未安装或不可导入；跳过检查。
        return

    try:
        declared = set(get_type_hints(ModelProfile).keys())
    except (TypeError, NameError):
                # get_type_hints 在无法解析的正向引用上会引发 NameError，且
                # 当注解计算结果为非类型对象时会引发 TypeError。
        return
    extra = sorted({k for p in profiles.values() for k in p} - declared)
    if extra:
        warnings.warn(
            f"Profile keys not declared in langchain_core ModelProfile: {extra}. "
            f"Add these fields to "
            f"langchain_core.language_models.model_profile.ModelProfile and "
            f"release langchain-core before publishing partner packages that "
            f"use these profiles.",
            stacklevel=2,
        )


def _ensure_safe_output_path(base_dir: Path, output_file: Path) -> None:
    """确保解析后的输出路径仍位于预期目录内。"""
    if base_dir.exists() and base_dir.is_symlink():
        msg = f"Data directory {base_dir} is a symlink; refusing to write profiles."
        print(f"❌ {msg}", file=sys.stderr)
        sys.exit(1)

    if output_file.exists() and output_file.is_symlink():
        msg = (
            f"profiles.py at {output_file} is a symlink; refusing to overwrite it.\n"
            "Delete the symlink or point --data-dir to a safe location."
        )
        print(f"❌ {msg}", file=sys.stderr)
        sys.exit(1)

    try:
        output_file.resolve(strict=False).relative_to(base_dir.resolve())
    except (OSError, RuntimeError) as e:
        msg = f"Failed to resolve output path: {e}"
        print(f"❌ {msg}", file=sys.stderr)
        sys.exit(1)
    except ValueError:
        msg = f"Refusing to write outside of data directory: {output_file}"
        print(f"❌ {msg}", file=sys.stderr)
        sys.exit(1)


def _write_profiles_file(output_file: Path, contents: str) -> None:
    """原子化地写入生成的模块，且不跟随符号链接。"""
    _ensure_safe_output_path(output_file.parent, output_file)

    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=output_file.parent, delete=False
        ) as tmp_file:
            tmp_file.write(contents)
            temp_path = Path(tmp_file.name)
        temp_path.replace(output_file)
    except PermissionError:
        msg = f"Permission denied writing file: {output_file}"
        print(f"❌ {msg}", file=sys.stderr)
        if temp_path:
            temp_path.unlink(missing_ok=True)
        sys.exit(1)
    except OSError as e:
        msg = f"Failed to write file: {e}"
        print(f"❌ {msg}", file=sys.stderr)
        if temp_path:
            temp_path.unlink(missing_ok=True)
        sys.exit(1)


MODULE_ADMONITION = """Auto-generated model profiles.

DO NOT EDIT THIS FILE MANUALLY.
This file is generated by the langchain-profiles CLI tool.

It contains data derived from the models.dev project.

Source: https://github.com/sst/models.dev
License: MIT License

To update these data, refer to the instructions here:

https://docs.langchain.com/oss/python/langchain/models#updating-or-overwriting-profile-data
"""


def refresh(provider: str, data_dir: Path) -> None:  # noqa: C901, PLR0915
    """下载并合并特定提供者的模型资料数据。

    Args:
        provider: Provider ID from models.dev (e.g., `'anthropic'`, `'openai'`).
        data_dir: Directory containing `profile_augmentations.toml` and where
            `profiles.py` will be written.
    """
        # 验证并规范化数据目录路径
    data_dir = _validate_data_dir(data_dir)

    api_url = "https://models.dev/api.json"

    print(f"Provider: {provider}")
    print(f"Data directory: {data_dir}")
    print()

        # 从 models.dev 下载数据
    print(f"Downloading data from {api_url}...")
    try:
        response = httpx.get(api_url, timeout=30)
        response.raise_for_status()
    except httpx.TimeoutException:
        msg = f"Request timed out connecting to {api_url}"
        print(f"❌ {msg}", file=sys.stderr)
        sys.exit(1)
    except httpx.HTTPStatusError as e:
        msg = f"HTTP error {e.response.status_code} from {api_url}"
        print(f"❌ {msg}", file=sys.stderr)
        sys.exit(1)
    except httpx.RequestError as e:
        msg = f"Failed to connect to {api_url}: {e}"
        print(f"❌ {msg}", file=sys.stderr)
        sys.exit(1)

    try:
        all_data = response.json()
    except json.JSONDecodeError as e:
        msg = f"Invalid JSON response from API: {e}"
        print(f"❌ {msg}", file=sys.stderr)
        sys.exit(1)

        # 基本验证
    if not isinstance(all_data, dict):
        msg = "Expected API response to be a dictionary"
        print(f"❌ {msg}", file=sys.stderr)
        sys.exit(1)

    provider_count = len(all_data)
    model_count = sum(len(p.get("models", {})) for p in all_data.values())
    print(f"Downloaded {provider_count} providers with {model_count} models")

        # 提取此提供者的数据
    if provider not in all_data:
        msg = f"Provider '{provider}' not found in models.dev data"
        print(msg, file=sys.stderr)
        sys.exit(1)

    provider_data = all_data[provider]
    models = provider_data.get("models", {})
    print(f"Extracted {len(models)} models for {provider}")

        # 加载扩充数据
    print("Loading augmentations...")
    provider_aug, model_augs = _load_augmentations(data_dir)

        # 合并并转换为资料
    profiles: dict[str, dict[str, Any]] = {}
    for model_id, model_data in models.items():
        base_profile = _model_data_to_profile(model_data)
        profiles[model_id] = _apply_overrides(
            base_profile, provider_aug, model_augs.get(model_id)
        )

        # 包含仅通过扩充定义的新模型
    extra_models = set(model_augs) - set(models)
    if extra_models:
        print(f"Adding {len(extra_models)} models from augmentations only...")
    for model_id in sorted(extra_models):
        profiles[model_id] = _apply_overrides({}, provider_aug, model_augs[model_id])

    _warn_undeclared_profile_keys(profiles)

        # 确保目录存在
    try:
        data_dir.mkdir(parents=True, exist_ok=True, mode=0o755)
    except PermissionError:
        msg = f"Permission denied creating directory: {data_dir}"
        print(f"❌ {msg}", file=sys.stderr)
        sys.exit(1)
    except OSError as e:
        msg = f"Failed to create directory: {e}"
        print(f"❌ {msg}", file=sys.stderr)
        sys.exit(1)

        # 以 Python 模块形式写入
    output_file = data_dir / "_profiles.py"
    print(f"Writing to {output_file}...")
    module_content = [f'"""{MODULE_ADMONITION}"""\n\n', "from typing import Any\n\n"]
    module_content.append("_PROFILES: dict[str, dict[str, Any]] = ")
    json_str = json.dumps(dict(sorted(profiles.items())), indent=4)
    json_str = (
        json_str.replace("true", "True")
        .replace("false", "False")
        .replace("null", "None")
    )
        # 添加尾随逗号以符合 ruff 格式要求
    json_str = re.sub(r"([^\s,{\[])(?=\n\s*[\}\]])", r"\1,", json_str)
    module_content.append(f"{json_str}\n")
    _write_profiles_file(output_file, "".join(module_content))

    print(
        f"✓ Successfully refreshed {len(profiles)} model profiles "
        f"({output_file.stat().st_size:,} bytes)"
    )


def main() -> None:
    """CLI 入口点。"""
    parser = argparse.ArgumentParser(
        description="Refresh model profile data from models.dev",
        prog="langchain-profiles",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

        # refresh 命令
    refresh_parser = subparsers.add_parser(
        "refresh", help="Download and merge model profile data for a provider"
    )
    refresh_parser.add_argument(
        "--provider",
        required=True,
        help="Provider ID from models.dev (e.g., 'anthropic', 'openai', 'google')",
    )
    refresh_parser.add_argument(
        "--data-dir",
        required=True,
        type=Path,
        help="Data directory containing profile_augmentations.toml",
    )

        # summarize 命令
    summarize_parser = subparsers.add_parser(
        "summarize",
        help="Summarize profile changes vs a git ref as Markdown (for PR bodies)",
    )
    summarize_parser.add_argument(
        "--providers",
        required=True,
        help=(
            "JSON array of objects with 'provider' and 'data_dir' keys "
            "(data_dir relative to the repo root)."
        ),
    )
    summarize_parser.add_argument(
        "--base-ref",
        default="HEAD",
        help="Git ref to compare the working tree against (default: HEAD).",
    )
    summarize_parser.add_argument(
        "--repo-root",
        type=Path,
        default=None,
        help="Repository root (default: current directory).",
    )

    args = parser.parse_args()

    if args.command == "refresh":
        refresh(args.provider, args.data_dir)
    elif args.command == "summarize":
        from langchain_model_profiles._summary import summarize

        try:
            providers = json.loads(args.providers)
        except json.JSONDecodeError as e:
            parser.error(f"--providers is not valid JSON: {e}")

        if not isinstance(providers, list):
            parser.error("--providers must be a JSON array")

        try:
            output = summarize(
                providers, base_ref=args.base_ref, repo_root=args.repo_root
            )
        except (RuntimeError, ValueError, TypeError) as e:
            parser.error(str(e))
        print(output)


if __name__ == "__main__":
    main()

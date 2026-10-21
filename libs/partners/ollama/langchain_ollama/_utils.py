"""验证 Ollama 模型的工具函数。"""

from __future__ import annotations

import base64
from urllib.parse import ParseResult, unquote, urlparse, urlunparse

from httpx import ConnectError
from ollama import Client, ResponseError


def validate_model(client: Client, model_name: str) -> None:
    """验证本地 Ollama 实例中是否存在指定模型。

    Args:
        client: Ollama 客户端。
        model_name: 要验证的模型名称。

    Raises:
        ValueError: 如果未找到模型或存在连接问题。
    """
    try:
        response = client.list()

        model_names: list[str] = [model["model"] for model in response["models"]]

        if not any(
            model_name == m or m.startswith(f"{model_name}:") for m in model_names
        ):
            msg = (
                f"Model `{model_name}` not found in Ollama. Please pull the "
                f"model (using `ollama pull {model_name}`) or specify a valid "
                f"model name. Available local models: {', '.join(model_names)}"
            )
            raise ValueError(msg)
    except ConnectError as e:
        msg = (
            "Failed to connect to Ollama. Please check that Ollama is downloaded, "
            "running and accessible. https://ollama.com/download"
        )
        raise ValueError(msg) from e
    except ResponseError as e:
        msg = (
            "Received an error from the Ollama API. "
            "Please check your Ollama server logs."
        )
        raise ValueError(msg) from e


def _build_cleaned_url(parsed: ParseResult) -> str:
    """从解析后的组件重建 URL，不包含 userinfo 部分。

    Args:
        parsed: 解析后的 URL 组件。

    Returns:
        去除 userinfo 后的清理版 URL 字符串。
    """
    hostname = parsed.hostname or ""
    if ":" in hostname:  # IPv6 — 重新添加 urlparse 移除的方括号
        hostname = f"[{hostname}]"
    cleaned_netloc = hostname
    if parsed.port is not None:
        cleaned_netloc += f":{parsed.port}"
    return urlunparse(
        (
            parsed.scheme,
            cleaned_netloc,
            parsed.path,
            parsed.params,
            parsed.query,
            parsed.fragment,
        )
    )


def parse_url_with_auth(
    url: str | None,
) -> tuple[str | None, dict[str, str] | None]:
    """解析 URL 并提取 `userinfo` 凭证以用于 headers。

    Handles URLs of the form: 支持格式为：`https://user:password@host:port/path` 的 URL

    也接受无协议头的 URL（例如 `host:port`），并将默认赋予 `http://` 协议。

    Args:
        url: 要解析的 URL。

    Returns:
        A tuple of `(cleaned_url, headers_dict)` where:
        - `cleaned_url` is a normalized URL with credentials stripped (if any
            were present) and a scheme guaranteed (defaulting to `http://` for
            scheme-less inputs). Returns the original URL unchanged when it
            already has a valid scheme and no credentials.
        - `headers_dict` contains Authorization header if credentials were found.
    """
    if not url:
        return None, None

    parsed = urlparse(url)
    needs_reconstruction = False
    valid = False

    if parsed.scheme in {"http", "https"} and parsed.netloc and parsed.hostname:
        valid = True
    elif not (parsed.scheme and parsed.netloc) and ":" in url:
                # 无有效协议但包含冒号 — 尝试作为无协议头的 host:port 处理
        parsed_with_scheme = urlparse(f"http://{url}")
        if parsed_with_scheme.netloc and parsed_with_scheme.hostname:
            parsed = parsed_with_scheme
            needs_reconstruction = True
            valid = True

        # 验证端口为数值（urlparse 对非数值端口会抛出 ValueError）
    if valid:
        try:
            _ = parsed.port
        except ValueError:
            valid = False

    if not valid:
        return None, None

    if not parsed.username:
        cleaned = _build_cleaned_url(parsed) if needs_reconstruction else url
        return cleaned, None

        # 处理密码可能为空字符串或 None 的情况
    password = parsed.password or ""

        # 创建 basic auth 头（解码百分比编码）
    username = unquote(parsed.username)
    password = unquote(password)
    credentials = f"{username}:{password}"
    encoded_credentials = base64.b64encode(credentials.encode()).decode()
    headers = {"Authorization": f"Basic {encoded_credentials}"}

    return _build_cleaned_url(parsed), headers


def merge_auth_headers(
    client_kwargs: dict,
    auth_headers: dict[str, str] | None,
) -> None:
    """将认证头合并到客户端 kwargs 中（就地更新）。

    Args:
        client_kwargs: 要更新的客户端 kwargs 字典。
        auth_headers: 要合并的头信息（通常来自 `parse_url_with_auth`）。
    """
    if auth_headers:
        headers = client_kwargs.get("headers", {})
        headers.update(auth_headers)
        client_kwargs["headers"] = headers

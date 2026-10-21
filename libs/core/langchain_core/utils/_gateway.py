"""用于解析 LangSmith 网关配置的私有辅助函数。

[LangSmith LLM 网关](https://docs.langchain.com/langsmith/llm-gateway)
允许聊天模型通过环境变量配置的代理访问提供方。这些辅助函数集中处理（非平凡的）优先级规则，使每个提供方集成都能以相同方式解析其基础
URL 和 API 密钥。

This module is private: 此模块为私有模块：API 不稳定，可能随时更改而无需通知。
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, NamedTuple

from pydantic import SecretStr

if TYPE_CHECKING:
    from collections.abc import Sequence

    from pydantic import BaseModel

# 集成方在生成的 `generation_info` 中放置解析后的网关元数据的键。
# `generation_info`。这是 LangSmith 追踪器所消费的契约，它会将该元数据提升为运行元数据。
# 将其提升为运行元数据。此处定义以确保集成和追踪器达成一致
# 使用单一名称。
GATEWAY_METADATA_RESPONSE_KEY = "lc_gateway_metadata"

# LangSmith 网关用于返回每请求元数据的 HTTP 响应头
# (例如已解析的提供方和模型)。该值为 JSON 编码。
GATEWAY_METADATA_HEADER = "X-LangSmith-Gateway-Metadata"

_LANGSMITH_GATEWAY_ENV = "LANGSMITH_GATEWAY"
_LANGSMITH_GATEWAY_API_KEY_ENV = "LANGSMITH_GATEWAY_API_KEY"
_LANGSMITH_API_KEY_ENV = "LANGSMITH_API_KEY"
_LANGSMITH_GATEWAY_DEFAULT_BASE = "https://gateway.smith.langchain.com"

_TRUE_VALUES = ("true", "1", "yes")
_FALSE_VALUES = ("false", "0", "no")


class GatewayConfig(NamedTuple):
    """已解析的网关配置。

    Attributes:
        base_url: The base URL the client should use, or None to defer to the
            provider SDK's own default.
        api_key: The API key to use. A ``SecretStr`` when derived from the
            environment; the caller's value returned unchanged when one was
            passed explicitly; None when no key could be resolved.
        base_url_from_gateway: Whether ``base_url`` was populated from the
            LangSmith gateway (as opposed to an explicit value, a provider env
            var, or a default).
    """

    base_url: str | None
    api_key: Any
    base_url_from_gateway: bool


def _first_env(names: str | Sequence[str]) -> str | None:
    """返回 ``names`` 中第一个非空值，或 None。"""
    if isinstance(names, str):
        names = (names,)
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    return None


def _resolve_gateway_base_url(provider_path: str) -> str | None:
    """为提供方解析 LangSmith 网关基础 URL。

    ``LANGSMITH_GATEWAY`` 接受布尔值字符串或显式基础 URL：

    - ``true`` / ``1`` / ``yes`` -> 默认网关主机。 - ``false`` / ``0`` / ``no`` /
    未设置 / 空 -> 网关被禁用（None）。 - 其他任何内容 -> 视为自定义网关基础 URL。

    在所有启用情况下，都会附加提供方特定路径。

    Args:
        provider_path: Path segment for the provider, e.g. ``"openai/v1"`` or
            ``"anthropic"``.

    Returns:
        The provider base URL on the gateway, or None if the gateway is
        disabled.
    """
    raw = os.getenv(_LANGSMITH_GATEWAY_ENV)
    if not raw or raw.lower() in _FALSE_VALUES:
        return None
    base = (
        _LANGSMITH_GATEWAY_DEFAULT_BASE
        if raw.lower() in _TRUE_VALUES
        else raw.rstrip("/")
    )
    return f"{base}/{provider_path}"


def _resolve_gateway_config(
    *,
    base_url: str | None,
    api_key: Any,
    provider_path: str,
    base_url_env: str | Sequence[str] = (),
    api_key_env: str | Sequence[str] = (),
    default_base_url: str | None = None,
) -> GatewayConfig:
    """解析提供方的基础 URL 和 API 密钥，并应用网关设置。

    优先级：

    - **base_url:** explicit ``base_url`` > ``base_url_env`` > LangSmith gateway
      > ``default_base_url``.
    - **api_key:** explicit ``api_key`` (returned unchanged) > the gateway key if
      the base URL came from the gateway, otherwise the provider key > the other.
      The gateway key uses ``LANGSMITH_GATEWAY_API_KEY`` when the gateway is
      enabled, with ``LANGSMITH_API_KEY`` as a fallback only when the base URL came
      from the gateway.

    The provenance flip on the key means an ``OPENAI_API_KEY``-style provider key
    is preferred whenever the caller pointed the base URL at a non-gateway
    endpoint (so a stray gateway key is not sent to the provider), while the
    gateway key wins for the common "just enable the gateway" setup even if a
    provider key happens to be present in the environment.

    Args:
        base_url: Explicitly-provided base URL (e.g. a ``base_url`` kwarg), or
            None if not set by the caller.
        api_key: Explicitly-provided API key, or None if not set by the caller.
            Returned unchanged when not None, so a caller-supplied value (secret
            or callable) always wins.
        provider_path: Path segment appended to the gateway host.
        base_url_env: Env var name(s) for the provider base URL, in priority
            order.
        api_key_env: Env var name(s) for the provider API key, in priority order.
        default_base_url: Base URL used when nothing else is set.

    Returns:
        已解析的 `GatewayConfig`。
    """
    gateway_base_url = _resolve_gateway_base_url(provider_path)

    resolved_base_url = base_url
    base_url_from_gateway = False
    if resolved_base_url is None:
        resolved_base_url = _first_env(base_url_env)
        if resolved_base_url is None:
            if gateway_base_url is not None:
                resolved_base_url = gateway_base_url
                base_url_from_gateway = True
            else:
                resolved_base_url = default_base_url

    if api_key is not None:
        resolved_api_key: Any = api_key
    else:
        gateway_api_key = (
            os.getenv(_LANGSMITH_GATEWAY_API_KEY_ENV)
            if gateway_base_url is not None
            else None
        )
        if not gateway_api_key and base_url_from_gateway:
            gateway_api_key = os.getenv(_LANGSMITH_API_KEY_ENV)
        provider_api_key = _first_env(api_key_env)
        chosen = (
            (gateway_api_key or provider_api_key)
            if base_url_from_gateway
            else (provider_api_key or gateway_api_key)
        )
        resolved_api_key = SecretStr(chosen) if chosen else None

    return GatewayConfig(resolved_base_url, resolved_api_key, base_url_from_gateway)


def _pop_provided(values: dict[str, Any], cls: type[BaseModel], field: str) -> Any:
    """弹出调用方提供的字段值，按名称或别名；若不存在则返回 None。

    Handles models with ``populate_by_name=True``, where a field may be supplied
    under either its name or its alias. Both keys are removed so the resolved
    value can be written back canonically under the field name. The alias is read
    from the model rather than hard-coded, so callers pass only field names.
    """
    value = values.pop(field, None)
    alias = cls.model_fields[field].alias
    if alias is not None:
        alias_value = values.pop(alias, None)
        if value is None:
            value = alias_value
    return value


def _apply_gateway_config(
    values: dict[str, Any],
    cls: type[BaseModel],
    *,
    base_url_field: str,
    api_key_field: str,
    provider_path: str,
    base_url_env: str | Sequence[str] = (),
    api_key_env: str | Sequence[str] = (),
    default_base_url: str | None = None,
) -> GatewayConfig:
    """在 "before" 验证器中，从模型的原始输入解析网关设置。

    读取调用方提供的基础 URL 和 API 密钥（按字段名称或别名）， 通过
    `_resolve_gateway_config` 对其与网关和提供方环境变量进行解析， 然后将结果写回到
    ``values`` 中的规范字段名称下。 这使得提供方集成可以保留一个非可选的关键字段：
    解析后的值在字段验证前注入，因此字段始终接收到具体值。

    The resolved API key is written only when non-None, so the field's own
    default applies when no key is found. Returns the `GatewayConfig` so the
    caller can, for example, raise when a required key is missing.

    Args:
        values: 传递给模型的原始输入映射，会就地修改。
        cls: 模型类，用于查找字段别名。
        base_url_field: 基础 URL 字段的名称（例如 ``"anthropic_api_url"``）。
        api_key_field: API 密钥字段的名称（例如 ``"anthropic_api_key"``）。
        provider_path: 追加到网关主机的路径段。
        base_url_env: 提供方基础 URL 的环境变量名称。
        api_key_env: 提供方 API 密钥的环境变量名称。
        default_base_url: 当其他配置均未设置时使用的默认基础 URL。

    Returns:
        已解析的 `GatewayConfig`。
    """
    config = _resolve_gateway_config(
        base_url=_pop_provided(values, cls, base_url_field),
        api_key=_pop_provided(values, cls, api_key_field),
        provider_path=provider_path,
        base_url_env=base_url_env,
        api_key_env=api_key_env,
        default_base_url=default_base_url,
    )
    values[base_url_field] = config.base_url
    if config.api_key is not None:
        values[api_key_field] = config.api_key
    return config


def _parse_gateway_metadata(headers: Any) -> dict[str, Any] | None:
    """从 HTTP 响应头中解析 LangSmith 网关元数据。

    提供方集成会在原始响应头上调用此方法，以便解析结果能通过
    `GATEWAY_METADATA_RESPONSE_KEY` 表面化， 并由 LangSmith 追踪器后续提升为运行元数据。

    Args:
        headers: 响应头。

    Returns:
        反序列化的网关元数据，当头部缺失、非映射类型或非 JSON 对象时返回 None。
    """
    if not isinstance(headers, Mapping):
        return None

    raw = headers.get(GATEWAY_METADATA_HEADER)
    if not raw:
        return None
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return None
    return parsed if isinstance(parsed, dict) else None

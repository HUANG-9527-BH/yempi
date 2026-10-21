"""中间件组件共享的脱敏工具。"""

from __future__ import annotations

import hashlib
import ipaddress
import operator
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal
from urllib.parse import urlparse

from typing_extensions import TypedDict

RedactionStrategy = Literal["block", "redact", "mask", "hash"]
"""Supported strategies for handling detected sensitive values."""


class PIIMatch(TypedDict):
    """表示敏感数据的单个匹配项。"""

    type: str
    value: str
    start: int
    end: int


class PIIDetectionError(Exception):
    """当配置为在检测到敏感值时阻断时抛出。"""

    def __init__(self, pii_type: str, matches: Sequence[PIIMatch]) -> None:
        """使用匹配上下文初始化异常。

        Args:
            pii_type: 检测到的敏感类型名称。
            matches: 该类型的所有检测到的匹配项。
        """
        self.pii_type = pii_type
        self.matches = list(matches)
        count = len(matches)
        msg = f"Detected {count} instance(s) of {pii_type} in text content"
        super().__init__(msg)


Detector = Callable[[str], list[PIIMatch]]
"""Callable signature for detectors that locate sensitive values."""


def detect_email(content: str) -> list[PIIMatch]:
    """在内容中检测电子邮件地址。

    Args:
        content: 要扫描电子邮件地址的文本内容。

    Returns:
        检测到的电子邮件匹配项列表。
    """
    pattern = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
    return [
        PIIMatch(
            type="email",
            value=match.group(),
            start=match.start(),
            end=match.end(),
        )
        for match in re.finditer(pattern, content)
    ]


def detect_credit_card(content: str) -> list[PIIMatch]:
    """使用 Luhn 校验在内容中检测信用卡号码。

    Args:
        content: 要扫描信用卡号码的文本内容。

    Returns:
        检测到的信用卡号码匹配项列表。
    """
    pattern = r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b"
    matches = []

    for match in re.finditer(pattern, content):
        card_number = match.group()
        if _passes_luhn(card_number):
            matches.append(
                PIIMatch(
                    type="credit_card",
                    value=card_number,
                    start=match.start(),
                    end=match.end(),
                )
            )

    return matches


def detect_ip(content: str) -> list[PIIMatch]:
    """在内容中检测 IPv4 或 IPv6 地址。

    Args:
        content: 要扫描 IP 地址的文本内容。

    Returns:
        检测到的 IP 地址匹配项列表。
    """
    matches: list[PIIMatch] = []
    ipv4_pattern = r"\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b"

    for match in re.finditer(ipv4_pattern, content):
        ip_candidate = match.group()
        try:
            ipaddress.ip_address(ip_candidate)
        except ValueError:
            continue
        matches.append(
            PIIMatch(
                type="ip",
                value=ip_candidate,
                start=match.start(),
                end=match.end(),
            )
        )

    return matches


def detect_mac_address(content: str) -> list[PIIMatch]:
    """在内容中检测 MAC 地址。

    Args:
        content: 要扫描 MAC 地址的文本内容。

    Returns:
        检测到的 MAC 地址匹配项列表。
    """
    pattern = r"\b([0-9A-Fa-f]{2}[:-]){5}[0-9A-Fa-f]{2}\b"
    return [
        PIIMatch(
            type="mac_address",
            value=match.group(),
            start=match.start(),
            end=match.end(),
        )
        for match in re.finditer(pattern, content)
    ]


def detect_url(content: str) -> list[PIIMatch]:
    """使用正则表达式和标准库验证在内容中检测 URL。

    Args:
        content: 要扫描 URL 的文本内容。

    Returns:
        检测到的 URL 匹配项列表。
    """
    matches: list[PIIMatch] = []

        # Pattern 1: URLs with scheme (http:// or https://)
    scheme_pattern = r"https?://[^\s<>\"{}|\\^`\[\]]+"

    for match in re.finditer(scheme_pattern, content):
        url = match.group()
        result = urlparse(url)
        if result.scheme in {"http", "https"} and result.netloc:
            matches.append(
                PIIMatch(
                    type="url",
                    value=url,
                    start=match.start(),
                    end=match.end(),
                )
            )

        # Pattern 2: URLs without scheme (www.example.com or example.com/path)
        # 更保守以避免误报
    bare_pattern = (
        r"\b(?:www\.)?[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?"
        r"(?:\.[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?)+(?:/[^\s]*)?"
    )

    for match in re.finditer(bare_pattern, content):
        start, end = match.start(), match.end()
                # 如果已用方案匹配则跳过
        if any(m["start"] <= start < m["end"] or m["start"] < end <= m["end"] for m in matches):
            continue

        url = match.group()
                # 仅接受具有路径或以 www 开头的
                # 这减少了如 "example.com" 在正文中的误报
        if "/" in url or url.startswith("www."):
                        # 添加方案以进行验证（urlparse 正常工作所需）
            test_url = f"http://{url}"
            result = urlparse(test_url)
            if result.netloc and "." in result.netloc:
                matches.append(
                    PIIMatch(
                        type="url",
                        value=url,
                        start=start,
                        end=end,
                    )
                )

    return matches


BUILTIN_DETECTORS: dict[str, Detector] = {
    "email": detect_email,
    "credit_card": detect_credit_card,
    "ip": detect_ip,
    "mac_address": detect_mac_address,
    "url": detect_url,
}
"""Registry of built-in detectors keyed by type name."""

_CARD_NUMBER_MIN_DIGITS = 13
_CARD_NUMBER_MAX_DIGITS = 19


def _passes_luhn(card_number: str) -> bool:
    """使用 Luhn 校验码验证信用卡号码。"""
    digits = [int(d) for d in card_number if d.isdigit()]
    if not _CARD_NUMBER_MIN_DIGITS <= len(digits) <= _CARD_NUMBER_MAX_DIGITS:
        return False

    checksum = 0
    for index, digit in enumerate(reversed(digits)):
        value = digit
        if index % 2 == 1:
            value *= 2
            if value > 9:  # noqa: PLR2004
                value -= 9
        checksum += value
    return checksum % 10 == 0


def _apply_redact_strategy(content: str, matches: list[PIIMatch]) -> str:
    result = content
    for match in sorted(matches, key=operator.itemgetter("start"), reverse=True):
        replacement = f"[REDACTED_{match['type'].upper()}]"
        result = result[: match["start"]] + replacement + result[match["end"] :]
    return result


_UNMASKED_CHAR_NUMBER = 4
_IPV4_PARTS_NUMBER = 4


def _apply_mask_strategy(content: str, matches: list[PIIMatch]) -> str:
    result = content
    for match in sorted(matches, key=operator.itemgetter("start"), reverse=True):
        value = match["value"]
        pii_type = match["type"]
        if pii_type == "email":
            parts = value.split("@")
            if len(parts) == 2:  # noqa: PLR2004
                domain_parts = parts[1].split(".")
                masked = (
                    f"{parts[0]}@****.{domain_parts[-1]}"
                    if len(domain_parts) > 1
                    else f"{parts[0]}@****"
                )
            else:
                masked = "****"
        elif pii_type == "credit_card":
            digits_only = "".join(c for c in value if c.isdigit())
            separator = "-" if "-" in value else " " if " " in value else ""
            if separator:
                masked = (
                    f"****{separator}****{separator}****{separator}"
                    f"{digits_only[-_UNMASKED_CHAR_NUMBER:]}"
                )
            else:
                masked = f"************{digits_only[-_UNMASKED_CHAR_NUMBER:]}"
        elif pii_type == "ip":
            octets = value.split(".")
            masked = f"*.*.*.{octets[-1]}" if len(octets) == _IPV4_PARTS_NUMBER else "****"
        elif pii_type == "mac_address":
            separator = ":" if ":" in value else "-"
            masked = (
                f"**{separator}**{separator}**{separator}**{separator}**{separator}{value[-2:]}"
            )
        elif pii_type == "url":
            masked = "[MASKED_URL]"
        else:
            masked = (
                f"****{value[-_UNMASKED_CHAR_NUMBER:]}"
                if len(value) > _UNMASKED_CHAR_NUMBER
                else "****"
            )
        result = result[: match["start"]] + masked + result[match["end"] :]
    return result


def _apply_hash_strategy(content: str, matches: list[PIIMatch]) -> str:
    result = content
    for match in sorted(matches, key=operator.itemgetter("start"), reverse=True):
        digest = hashlib.sha256(match["value"].encode()).hexdigest()[:8]
        replacement = f"<{match['type']}_hash:{digest}>"
        result = result[: match["start"]] + replacement + result[match["end"] :]
    return result


def apply_strategy(
    content: str,
    matches: list[PIIMatch],
    strategy: RedactionStrategy,
) -> str:
    """对内容中的匹配项应用配置的策略。

    Args:
        content: 要应用策略的内容。
        matches: 检测到的 PII 匹配项列表。
        strategy: 要应用的脱敏策略。

    Returns:
        应用策略后的内容。

    Raises:
        PIIDetectionError: 如果策略为 `'block'` 且检测到匹配项。
        ValueError: 如果策略未知。
    """
    if not matches:
        return content
    if strategy == "redact":
        return _apply_redact_strategy(content, matches)
    if strategy == "mask":
        return _apply_mask_strategy(content, matches)
    if strategy == "hash":
        return _apply_hash_strategy(content, matches)
    if strategy == "block":
        raise PIIDetectionError(matches[0]["type"], matches)
    msg = f"Unknown redaction strategy: {strategy}"  # type: ignore[unreachable]
    raise ValueError(msg)


def resolve_detector(pii_type: str, detector: Detector | str | None) -> Detector:
    """返回给定配置的可调用检测器。

    Args:
        pii_type: The PII type name.
        detector: Optional custom detector or regex pattern. If `None`, a built-in detector
            for the given PII type will be used.

    Returns:
        解析后的检测器。

    Raises:
        ValueError: 如果指定了未知的 PII 类型且未提供自定义检测器或正则表达式。
    """
    if detector is None:
        if pii_type not in BUILTIN_DETECTORS:
            msg = (
                f"Unknown PII type: {pii_type}. "
                f"Must be one of {list(BUILTIN_DETECTORS.keys())} or provide a custom detector."
            )
            raise ValueError(msg)
        return BUILTIN_DETECTORS[pii_type]
    if isinstance(detector, str):
        pattern = re.compile(detector)

        def regex_detector(content: str) -> list[PIIMatch]:
            return [
                PIIMatch(
                    type=pii_type,
                    value=match.group(),
                    start=match.start(),
                    end=match.end(),
                )
                for match in pattern.finditer(content)
            ]

        return regex_detector

        # 包装自定义可调用对象以标准化其输出。
        # 自定义检测器可能返回包含 "text" 而非 "value" 的字典
        # 并且可能省略 "type"。将其映射为正确的 PIIMatch 对象，以便
        # 下游策略（哈希、掩码）可以访问 match["value"]。
    raw_detector = detector

    def _normalizing_detector(content: str) -> list[PIIMatch]:
        return [
            PIIMatch(
                type=m.get("type", pii_type),
                value=m.get("value", m.get("text", "")),
                start=m["start"],
                end=m["end"],
            )
            for m in raw_detector(content)
        ]

    return _normalizing_detector


@dataclass(frozen=True)
class RedactionRule:
    """处理单个 PII 类型的配置。"""

    pii_type: str
    strategy: RedactionStrategy = "redact"
    detector: Detector | str | None = None

    def resolve(self) -> ResolvedRedactionRule:
        """解析运行时检测器并返回不可变规则。

        Returns:
            解析后的脱敏规则。
        """
        resolved_detector = resolve_detector(self.pii_type, self.detector)
        return ResolvedRedactionRule(
            pii_type=self.pii_type,
            strategy=self.strategy,
            detector=resolved_detector,
        )


@dataclass(frozen=True)
class ResolvedRedactionRule:
    """已解析的脱敏规则，准备执行。"""

    pii_type: str
    strategy: RedactionStrategy
    detector: Detector

    def apply(self, content: str) -> tuple[str, list[PIIMatch]]:
        """将此规则应用于内容，返回更新后的内容和匹配项。

        Args:
            content: 要扫描和脱敏的文本内容。

        Returns:
            一个元组（更新后的内容，检测到的匹配项列表）。
        """
        matches = self.detector(content)
        if not matches:
            return content, []
        updated = apply_strategy(content, matches, self.strategy)
        return updated, matches


__all__ = [
    "PIIDetectionError",
    "PIIMatch",
    "RedactionRule",
    "ResolvedRedactionRule",
    "apply_strategy",
    "detect_credit_card",
    "detect_email",
    "detect_ip",
    "detect_mac_address",
    "detect_url",
]

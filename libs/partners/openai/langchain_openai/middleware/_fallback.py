"""当 API 拒绝包含图片的请求时，重试不带图片的请求。"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import openai

if TYPE_CHECKING:
    from collections.abc import Sequence

    from langchain_openai.decisions import (
        DecisionRequest,
        DecisionResponse,
        OpenAIDecisions,
    )

_RETRY_STATUS_CODES = frozenset({400, 413})

logger = logging.getLogger(__name__)


def decide_with_text_fallback(
    decisions: OpenAIDecisions, requests: Sequence[DecisionRequest]
) -> DecisionResponse:
    """按顺序尝试每个请求，当包含图片的请求被拒绝时回退。

    Args:
        decisions: 用于回答请求的决策模型。
        requests: 要尝试的请求列表，以不包含图片的请求结尾。

    Returns:
        API 接受的第一个请求的响应。
    """
    for request in requests[:-1]:
        try:
            return decisions.invoke(request)
        except openai.APIStatusError as e:
            _raise_unless_retryable(e)
    return decisions.invoke(requests[-1])


async def adecide_with_text_fallback(
    decisions: OpenAIDecisions, requests: Sequence[DecisionRequest]
) -> DecisionResponse:
    """异步尝试每个请求，当包含图片的请求被拒绝时回退。

    Args:
        decisions: 用于回答请求的决策模型。
        requests: 要尝试的请求列表，以不包含图片的请求结尾。

    Returns:
        API 接受的第一个请求的响应。
    """
    for request in requests[:-1]:
        try:
            return await decisions.ainvoke(request)
        except openai.APIStatusError as e:
            _raise_unless_retryable(e)
    return await decisions.ainvoke(requests[-1])


def _raise_unless_retryable(error: openai.APIStatusError) -> None:
    if error.status_code not in _RETRY_STATUS_CODES:
        raise error
    logger.warning(
        "Decision request with images was rejected (HTTP %s); retrying without images.",
        error.status_code,
    )

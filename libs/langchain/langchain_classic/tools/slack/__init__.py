"""Slack 工具。"""

from typing import TYPE_CHECKING, Any

from langchain_classic._api import create_importer

if TYPE_CHECKING:
    from langchain_community.tools import (
        SlackGetChannel,
        SlackGetMessage,
        SlackScheduleMessage,
        SlackSendMessage,
    )

# 创建一种动态查找已弃用导入的方式。
# 用于统一处理弃用警告的逻辑并
# 处理可选导入。
DEPRECATED_LOOKUP = {
    "SlackGetChannel": "langchain_community.tools",
    "SlackGetMessage": "langchain_community.tools",
    "SlackScheduleMessage": "langchain_community.tools",
    "SlackSendMessage": "langchain_community.tools",
}

_import_attribute = create_importer(__package__, deprecated_lookups=DEPRECATED_LOOKUP)


def __getattr__(name: str) -> Any:
    """动态查找属性。"""
    return _import_attribute(name)


__all__ = [
    "SlackGetChannel",
    "SlackGetMessage",
    "SlackScheduleMessage",
    "SlackSendMessage",
]

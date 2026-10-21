from typing import Any

from langchain_classic._api import create_importer

_importer = create_importer(
    __package__,
    fallback_module="langchain_community.agent_toolkits.load_tools",
)


def __getattr__(name: str) -> Any:
    """动态查找属性。"""
    return _importer(name)

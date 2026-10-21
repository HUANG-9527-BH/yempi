"""Amadeus 工具。"""

from typing import TYPE_CHECKING, Any

from langchain_classic._api import create_importer

if TYPE_CHECKING:
    from langchain_community.tools.amadeus.closest_airport import AmadeusClosestAirport
    from langchain_community.tools.amadeus.flight_search import AmadeusFlightSearch

# 创建一种动态查找已弃用导入的方式。
# 用于统一处理弃用警告的逻辑并
# 处理可选导入。
DEPRECATED_LOOKUP = {
    "AmadeusClosestAirport": "langchain_community.tools.amadeus.closest_airport",
    "AmadeusFlightSearch": "langchain_community.tools.amadeus.flight_search",
}

_import_attribute = create_importer(__package__, deprecated_lookups=DEPRECATED_LOOKUP)


def __getattr__(name: str) -> Any:
    """动态查找属性。"""
    return _import_attribute(name)


__all__ = [
    "AmadeusClosestAirport",
    "AmadeusFlightSearch",
]

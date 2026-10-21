from typing import TYPE_CHECKING, Any

from langchain_classic._api import create_importer

if TYPE_CHECKING:
    from langchain_community.tools.amadeus.flight_search import (
        AmadeusFlightSearch,
        FlightSearchSchema,
    )

# 创建一种动态查找已弃用导入的方式。
# 用于合并用于发出弃用警告和
# 处理可选导入的逻辑。
DEPRECATED_LOOKUP = {
    "FlightSearchSchema": "langchain_community.tools.amadeus.flight_search",
    "AmadeusFlightSearch": "langchain_community.tools.amadeus.flight_search",
}

_import_attribute = create_importer(__package__, deprecated_lookups=DEPRECATED_LOOKUP)


def __getattr__(name: str) -> Any:
    """动态查找属性。"""
    return _import_attribute(name)


__all__ = [
    "AmadeusFlightSearch",
    "FlightSearchSchema",
]

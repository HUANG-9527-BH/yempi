from typing import TYPE_CHECKING, Any

from langchain_classic._api import create_importer

if TYPE_CHECKING:
    from langchain_community.agent_toolkits.spark_sql.prompt import (
        SQL_PREFIX,
        SQL_SUFFIX,
    )

# 创建一种动态查找已弃用导入的方式。
# 用于整合引发弃用警告和处理可选导入的逻辑。
# 处理可选导入。
DEPRECATED_LOOKUP = {
    "SQL_PREFIX": "langchain_community.agent_toolkits.spark_sql.prompt",
    "SQL_SUFFIX": "langchain_community.agent_toolkits.spark_sql.prompt",
}

_import_attribute = create_importer(__package__, deprecated_lookups=DEPRECATED_LOOKUP)


def __getattr__(name: str) -> Any:
    """动态查找属性。"""
    return _import_attribute(name)


__all__ = ["SQL_PREFIX", "SQL_SUFFIX"]

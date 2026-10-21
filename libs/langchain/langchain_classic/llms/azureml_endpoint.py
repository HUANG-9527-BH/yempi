from typing import TYPE_CHECKING, Any

from langchain_classic._api import create_importer

if TYPE_CHECKING:
    from langchain_community.llms import AzureMLOnlineEndpoint
    from langchain_community.llms.azureml_endpoint import (
        AzureMLEndpointClient,
        ContentFormatterBase,
        CustomOpenAIContentFormatter,
        DollyContentFormatter,
        GPT2ContentFormatter,
        HFContentFormatter,
        OSSContentFormatter,
    )

# 创建一种动态查找已弃用导入的方式。
# 用于统一处理引发弃用警告和
# 处理可选导入的逻辑。
DEPRECATED_LOOKUP = {
    "AzureMLEndpointClient": "langchain_community.llms.azureml_endpoint",
    "ContentFormatterBase": "langchain_community.llms.azureml_endpoint",
    "GPT2ContentFormatter": "langchain_community.llms.azureml_endpoint",
    "OSSContentFormatter": "langchain_community.llms.azureml_endpoint",
    "HFContentFormatter": "langchain_community.llms.azureml_endpoint",
    "DollyContentFormatter": "langchain_community.llms.azureml_endpoint",
    "CustomOpenAIContentFormatter": "langchain_community.llms.azureml_endpoint",
    "AzureMLOnlineEndpoint": "langchain_community.llms",
}

_import_attribute = create_importer(__package__, deprecated_lookups=DEPRECATED_LOOKUP)


def __getattr__(name: str) -> Any:
    """动态查找属性。"""
    return _import_attribute(name)


__all__ = [
    "AzureMLEndpointClient",
    "AzureMLOnlineEndpoint",
    "ContentFormatterBase",
    "CustomOpenAIContentFormatter",
    "DollyContentFormatter",
    "GPT2ContentFormatter",
    "HFContentFormatter",
    "OSSContentFormatter",
]

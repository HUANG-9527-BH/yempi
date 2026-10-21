from typing import TYPE_CHECKING, Any

from langchain_classic._api import create_importer

if TYPE_CHECKING:
    from langchain_community.chat_message_histories import (
        AstraDBChatMessageHistory,
        CassandraChatMessageHistory,
        ChatMessageHistory,
        CosmosDBChatMessageHistory,
        DynamoDBChatMessageHistory,
        ElasticsearchChatMessageHistory,
        FileChatMessageHistory,
        FirestoreChatMessageHistory,
        MomentoChatMessageHistory,
        MongoDBChatMessageHistory,
        Neo4jChatMessageHistory,
        PostgresChatMessageHistory,
        RedisChatMessageHistory,
        RocksetChatMessageHistory,
        SingleStoreDBChatMessageHistory,
        SQLChatMessageHistory,
        StreamlitChatMessageHistory,
        UpstashRedisChatMessageHistory,
        XataChatMessageHistory,
        ZepChatMessageHistory,
    )

# 创建一种动态查找已弃用导入的方式。
# 用于统一处理引发弃用警告和
# 处理可选导入的逻辑。
DEPRECATED_LOOKUP = {
    "AstraDBChatMessageHistory": "langchain_community.chat_message_histories",
    "CassandraChatMessageHistory": "langchain_community.chat_message_histories",
    "ChatMessageHistory": "langchain_community.chat_message_histories",
    "CosmosDBChatMessageHistory": "langchain_community.chat_message_histories",
    "DynamoDBChatMessageHistory": "langchain_community.chat_message_histories",
    "ElasticsearchChatMessageHistory": "langchain_community.chat_message_histories",
    "FileChatMessageHistory": "langchain_community.chat_message_histories",
    "FirestoreChatMessageHistory": "langchain_community.chat_message_histories",
    "MomentoChatMessageHistory": "langchain_community.chat_message_histories",
    "MongoDBChatMessageHistory": "langchain_community.chat_message_histories",
    "Neo4jChatMessageHistory": "langchain_community.chat_message_histories",
    "PostgresChatMessageHistory": "langchain_community.chat_message_histories",
    "RedisChatMessageHistory": "langchain_community.chat_message_histories",
    "RocksetChatMessageHistory": "langchain_community.chat_message_histories",
    "SQLChatMessageHistory": "langchain_community.chat_message_histories",
    "SingleStoreDBChatMessageHistory": "langchain_community.chat_message_histories",
    "StreamlitChatMessageHistory": "langchain_community.chat_message_histories",
    "UpstashRedisChatMessageHistory": "langchain_community.chat_message_histories",
    "XataChatMessageHistory": "langchain_community.chat_message_histories",
    "ZepChatMessageHistory": "langchain_community.chat_message_histories",
}

_import_attribute = create_importer(__package__, deprecated_lookups=DEPRECATED_LOOKUP)


def __getattr__(name: str) -> Any:
    """动态查找属性。"""
    return _import_attribute(name)


__all__ = [
    "AstraDBChatMessageHistory",
    "CassandraChatMessageHistory",
    "ChatMessageHistory",
    "CosmosDBChatMessageHistory",
    "DynamoDBChatMessageHistory",
    "ElasticsearchChatMessageHistory",
    "FileChatMessageHistory",
    "FirestoreChatMessageHistory",
    "MomentoChatMessageHistory",
    "MongoDBChatMessageHistory",
    "Neo4jChatMessageHistory",
    "PostgresChatMessageHistory",
    "RedisChatMessageHistory",
    "RocksetChatMessageHistory",
    "SQLChatMessageHistory",
    "SingleStoreDBChatMessageHistory",
    "StreamlitChatMessageHistory",
    "UpstashRedisChatMessageHistory",
    "XataChatMessageHistory",
    "ZepChatMessageHistory",
]

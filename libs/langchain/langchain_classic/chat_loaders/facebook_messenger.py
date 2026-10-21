from typing import TYPE_CHECKING, Any

from langchain_classic._api.module_import import create_importer

if TYPE_CHECKING:
    from langchain_community.chat_loaders.facebook_messenger import (
        FolderFacebookMessengerChatLoader,
        SingleFileFacebookMessengerChatLoader,
    )

module_lookup = {
    "SingleFileFacebookMessengerChatLoader": (
        "langchain_community.chat_loaders.facebook_messenger"
    ),
    "FolderFacebookMessengerChatLoader": (
        "langchain_community.chat_loaders.facebook_messenger"
    ),
}

# 临时代码，用于兼容已弃用的导入。
# 最终将被移除。
import_lookup = create_importer(
    __package__,
    deprecated_lookups=module_lookup,
)


def __getattr__(name: str) -> Any:
    return import_lookup(name)


__all__ = ["FolderFacebookMessengerChatLoader", "SingleFileFacebookMessengerChatLoader"]

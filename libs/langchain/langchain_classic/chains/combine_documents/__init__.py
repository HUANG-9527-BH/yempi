"""组合文档的不同方式。"""

from langchain_classic.chains.combine_documents.reduce import (
    acollapse_docs,
    collapse_docs,
    split_list_of_docs,
)
from langchain_classic.chains.combine_documents.stuff import (
    create_stuff_documents_chain,
)

__all__ = [
    "acollapse_docs",
    "collapse_docs",
    "create_stuff_documents_chain",
    "split_list_of_docs",
]

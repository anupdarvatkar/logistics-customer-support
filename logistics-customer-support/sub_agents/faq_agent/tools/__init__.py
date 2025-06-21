"""
RAG Tools package for interacting with Vertex AI RAG corpora.
"""

from .rag_query import rag_query
from .utils import (
    check_corpus_exists,
    get_corpus_resource_name,
    set_current_corpus,
)

__all__ = [
    "rag_query",
]

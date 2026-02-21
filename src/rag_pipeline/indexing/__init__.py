# Stage 6: Indexing
# Store embeddings in Weaviate vector database

from .weaviate_client import (
    get_client,
    create_collection,
    create_raptor_collection,
    delete_collection,
    upload_embeddings,
    get_collection_count,
)
from .weaviate_query import (
    SearchResult,
    query_similar,
    query_hybrid,
    list_available_books,
)

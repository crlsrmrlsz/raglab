"""Entity extraction for GraphRAG queries.

## RAG Theory: Query Entity Extraction

GraphRAG uses entity extraction to identify concepts in user queries,
then traverses the knowledge graph from matched entities to find
related chunks.

**Extraction method**: Embedding similarity search against entity descriptions
stored in Weaviate. This matches the Microsoft GraphRAG reference implementation.
Entity descriptions are indexed from the same Neo4j source, so all embedding
matches are guaranteed to exist in the graph — no separate validation needed.

## Library Usage

- Weaviate for embedding similarity search (HNSW index)

## Data Flow

Query → Embed → Weaviate vector search → Entity names
"""

from src.config import (
    GRAPHRAG_ENTITY_EXTRACTION_TOP_K,
    GRAPHRAG_ENTITY_MIN_SIMILARITY,
    get_entity_collection_name,
)
from src.shared.files import setup_logging

logger = setup_logging(__name__)


def extract_query_entities(
    query: str,
    top_k: int = GRAPHRAG_ENTITY_EXTRACTION_TOP_K,
    min_similarity: float = GRAPHRAG_ENTITY_MIN_SIMILARITY,
) -> list[str]:
    """Extract entity mentions from query using embedding similarity.

    Searches entity descriptions in Weaviate for semantic matches to the query.
    Matches Microsoft GraphRAG reference implementation. All returned entities
    are guaranteed to exist in Neo4j since both stores are populated from
    the same indexing pipeline.

    Args:
        query: User query string.
        top_k: Maximum entities to return.
        min_similarity: Minimum cosine similarity threshold.

    Returns:
        List of entity names found via embedding search (may be empty).

    Example:
        >>> extract_query_entities("How does dopamine affect motivation?")
        ["dopamine", "motivation", "reward"]
    """
    from src.rag_pipeline.embedder import embed_texts
    from src.rag_pipeline.indexing.weaviate_client import (
        get_client as get_weaviate_client,
        query_entities_by_vector,
    )

    collection_name = get_entity_collection_name()

    try:
        client = get_weaviate_client()

        # Check if entity collection exists
        if not client.collections.exists(collection_name):
            logger.debug(f"Entity collection {collection_name} not found, skipping embedding extraction")
            client.close()
            return []

        # Embed the query
        query_embedding = embed_texts([query])[0]

        # Search for similar entities
        results = query_entities_by_vector(
            client=client,
            collection_name=collection_name,
            query_embedding=query_embedding,
            top_k=top_k,
            min_similarity=min_similarity,
        )

        client.close()

        # Extract entity names from results
        entity_names = [r["entity_name"] for r in results]
        if entity_names:
            logger.info(f"Embedding extraction found {len(entity_names)} entities: {entity_names[:5]}")

        return entity_names

    except Exception as e:
        logger.warning(f"Embedding extraction failed: {e}")
        return []

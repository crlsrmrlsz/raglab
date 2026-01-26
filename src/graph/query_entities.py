"""Entity extraction for GraphRAG queries.

## RAG Theory: Query Entity Extraction

GraphRAG uses entity extraction to identify concepts in user queries,
then traverses the knowledge graph from matched entities to find
related chunks.

**Extraction method**: Embedding similarity search against entity descriptions
stored in Weaviate. This matches the Microsoft GraphRAG reference implementation.

## Library Usage

- Weaviate for embedding similarity search (HNSW index)
- Neo4j for entity validation (optional)

## Data Flow

Query → Embed → Weaviate vector search → Entity names
                       ↓
               Neo4j validation (optional)
                       ↓
               Validated entity names
"""

from typing import Optional
import re

from neo4j import Driver

from src.config import (
    GRAPHRAG_ENTITY_EXTRACTION_TOP_K,
    GRAPHRAG_ENTITY_MIN_SIMILARITY,
    get_entity_collection_name,
)
from src.shared.files import setup_logging
from .neo4j_client import find_entities_by_names

logger = setup_logging(__name__)


# ============================================================================
# Embedding-Based Entity Extraction
# ============================================================================


def extract_query_entities_embedding(
    query: str,
    top_k: int = GRAPHRAG_ENTITY_EXTRACTION_TOP_K,
    min_similarity: float = GRAPHRAG_ENTITY_MIN_SIMILARITY,
) -> list[str]:
    """Extract entities from query using embedding similarity.

    Searches entity descriptions in Weaviate for semantic matches to the query.
    This is the primary extraction method per Microsoft GraphRAG reference.

    Args:
        query: User query string.
        top_k: Maximum entities to return.
        min_similarity: Minimum cosine similarity threshold.

    Returns:
        List of entity names found via embedding search.

    Example:
        >>> entities = extract_query_entities_embedding("What is dopamine?")
        >>> entities
        ["dopamine", "neurotransmitter", "reward"]
    """
    from src.rag_pipeline.embedding.embedder import embed_texts
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


# ============================================================================
# Main Entity Extraction Function
# ============================================================================


def extract_query_entities(
    query: str,
    driver: Optional[Driver] = None,
    use_embedding: bool = GRAPHRAG_USE_EMBEDDING_EXTRACTION,
    use_llm_fallback: bool = True,
) -> list[str]:
    """Extract entity mentions from query using embedding similarity + LLM fallback.

    Primary method: Embedding-based extraction (fast, semantic matching)
    Fallback 1: LLM-based extraction (handles conceptual terms)
    Fallback 2: Regex for capitalized words (if both above fail)
    Validation: Neo4j lookup to verify entities exist in graph

    Args:
        query: User query string.
        driver: Optional Neo4j driver for entity lookup.
        use_embedding: Whether to try embedding extraction first.
        use_llm_fallback: Whether to fall back to LLM if embedding returns empty.

    Returns:
        List of entity names found in query.

    Example:
        >>> extract_query_entities("What creates lasting happiness?")
        ["happiness", "pleasure", "hedonic adaptation"]
        >>> extract_query_entities("How does Sapolsky explain stress?")
        ["Sapolsky", "stress"]
    """
    entities = []

    # Primary: Embedding-based extraction (fast, semantic)
    if use_embedding:
        entities = extract_query_entities_embedding(query)
        if entities:
            logger.info(f"Using embedding extraction: {entities}")

    # Fallback 1: LLM-based extraction
    if not entities and use_llm_fallback:
        entities = extract_query_entities_llm(query)
        if entities:
            logger.info(f"Using LLM fallback: {entities}")

    # Fallback 2: Regex for capitalized words
    if not entities:
        cap_pattern = r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b'
        capitalized = re.findall(cap_pattern, query)
        entities.extend(capitalized)
        if entities:
            logger.info(f"Using regex fallback: {entities}")

    # Validate against Neo4j if driver provided
    if driver and entities:
        db_entities = find_entities_by_names(driver, entities)
        validated = [e["name"] for e in db_entities]
        # Add validated entities (may have different casing)
        entities.extend(validated)

    # Deduplicate while preserving order
    seen = set()
    unique = []
    for e in entities:
        e_lower = e.lower()
        if e_lower not in seen:
            seen.add(e_lower)
            unique.append(e)

    return unique

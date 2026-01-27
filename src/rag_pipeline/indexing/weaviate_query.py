"""Weaviate query functions for RAGLab.

Provides search capabilities for the RAG knowledge base:
- Vector similarity search using pre-embedded queries
- Filtering by book_id for focused retrieval
- Hybrid search combining vector and keyword matching

Uses Weaviate Python client v4 query API.
"""

from dataclasses import dataclass
from typing import Optional, Union

import weaviate
from weaviate.classes.query import MetadataQuery, Filter

from src.config import get_collection_name
from src.rag_pipeline.embedder import embed_texts
from src.shared.files import setup_logging

logger = setup_logging(__name__)


@dataclass
class SearchResult:
    """A single search result with chunk data and relevance score.

    Attributes:
        chunk_id: Unique identifier for the chunk.
        book_id: Source book identifier.
        section: Section/chapter title within the book.
        context: Hierarchical context (book > chapter > section).
        text: The actual chunk text content.
        token_count: Number of tokens in the chunk.
        score: Relevance score (similarity for vector, combined for hybrid).
        is_summary: For RAPTOR collections, True if this is a summary node.
        tree_level: For RAPTOR collections, depth in tree (0=leaf, 1+=summary).
    """

    chunk_id: str
    book_id: str
    section: str
    context: str
    text: str
    token_count: int
    score: float
    is_summary: bool = False
    tree_level: int = 0


def _build_book_filter(
    book_ids: Optional[Union[str, list[str]]]
) -> Optional[Filter]:
    """
    Build a Weaviate filter for book_id(s).

    Args:
        book_ids: Single book ID, list of book IDs, or None.

    Returns:
        Weaviate Filter object or None if no filtering needed.

    Note:
        Weaviate v4 uses the Filter class for type-safe filtering.
        For a single value, use Filter.by_property().equal().
        For multiple values, use Filter.by_property().contains_any().
    """
    if book_ids is None:
        return None

    # Normalize to list for consistent handling
    if isinstance(book_ids, str):
        book_ids = [book_ids]

    if len(book_ids) == 1:
        return Filter.by_property("book_id").equal(book_ids[0])
    else:
        return Filter.by_property("book_id").contains_any(book_ids)


def _parse_results(
    response_objects: list,
    use_distance: bool = True,
) -> list[SearchResult]:
    """
    Convert Weaviate response objects to SearchResult dataclasses.

    Args:
        response_objects: Raw objects from Weaviate query response.
        use_distance: If True, convert distance to similarity (1 - distance).
                     If False, use score directly (for hybrid search).

    Returns:
        List of SearchResult objects with extracted metadata.
    """
    results = []

    for obj in response_objects:
        props = obj.properties

        # Distance: lower is better (0 = identical), range [0, 2] for cosine
        # Score: higher is better (for hybrid search)
        if use_distance and hasattr(obj.metadata, "distance") and obj.metadata.distance is not None:
            score = 1.0 - obj.metadata.distance  # Convert to similarity
        elif hasattr(obj.metadata, "score") and obj.metadata.score is not None:
            score = obj.metadata.score
        else:
            score = 0.0

        results.append(
            SearchResult(
                chunk_id=props.get("chunk_id", ""),
                book_id=props.get("book_id", ""),
                section=props.get("section", ""),
                context=props.get("context", ""),
                text=props.get("text", ""),
                token_count=props.get("token_count", 0),
                score=score,
                is_summary=props.get("is_summary", False),
                tree_level=props.get("tree_level", 0),
            )
        )

    return results


def query_similar(
    client: weaviate.WeaviateClient,
    query_text: str,
    top_k: int = 5,
    book_ids: Optional[Union[str, list[str]]] = None,
    collection_name: Optional[str] = None,
) -> list[SearchResult]:
    """
    Search for chunks semantically similar to the query text.

    This function:
    1. Embeds the query text using the same model as the chunks
    2. Performs vector similarity search in Weaviate
    3. Optionally filters by one or more book IDs

    Args:
        client: Connected Weaviate client.
        query_text: The natural language query to search for.
        top_k: Number of results to return (default: 5).
        book_ids: Optional book ID(s) to filter results. Can be:
            - None: search all books
            - str: search single book
            - List[str]: search multiple books
        collection_name: Target collection (default: from config).

    Returns:
        List of SearchResult objects, ordered by relevance (most similar first).

    Raises:
        weaviate.exceptions.WeaviateQueryError: If the search fails.

    Example:
        >>> client = get_client()
        >>> results = query_similar(
        ...     client,
        ...     "What is System 1 thinking?",
        ...     top_k=3,
        ...     book_ids="Thinking Fast and Slow (Daniel Kahneman)"
        ... )
        >>> for r in results:
        ...     print(f"{r.book_id}: {r.text[:100]}...")
    """
    if collection_name is None:
        collection_name = get_collection_name()

    # Step 1: Embed the query using the SAME model as the stored chunks
    logger.info(f"[query_similar] Collection: {collection_name}")
    logger.info(f"Embedding query: {query_text[:50]}...")
    query_embedding = embed_texts([query_text])[0]

    # Step 2: Get the collection reference
    collection = client.collections.get(collection_name)

    # Step 3: Build optional filter
    book_filter = _build_book_filter(book_ids)

    # Step 4: Execute vector search
    response = collection.query.near_vector(
        near_vector=query_embedding,
        limit=top_k,
        filters=book_filter,
        return_metadata=MetadataQuery(distance=True),
    )

    # Step 5: Parse and return results
    return _parse_results(response.objects, use_distance=True)


def query_hybrid(
    client: weaviate.WeaviateClient,
    query_text: str,
    top_k: int = 5,
    alpha: float = 0.5,
    book_ids: Optional[Union[str, list[str]]] = None,
    collection_name: Optional[str] = None,
    precomputed_embedding: Optional[list[float]] = None,
) -> list[SearchResult]:
    """
    Perform hybrid search combining vector similarity and keyword matching.

    Hybrid search balances semantic understanding (what the query means)
    with lexical matching (exact words in the query). The alpha parameter
    controls this balance.

    Args:
        client: Connected Weaviate client.
        query_text: The natural language query to search for (also used for BM25).
        top_k: Number of results to return (default: 5).
        alpha: Balance between vector (1.0) and keyword (0.0) search.
            - 1.0: Pure vector search (semantic only)
            - 0.5: Equal weight (recommended default)
            - 0.0: Pure keyword search (BM25 only)
        book_ids: Optional book ID(s) to filter results.
        collection_name: Target collection (default: from config).
        precomputed_embedding: Optional pre-computed embedding vector. If provided,
            skips embedding computation (useful for HyDE K=5 averaged embeddings).

    Returns:
        List of SearchResult objects, ordered by combined relevance.

    Raises:
        weaviate.exceptions.WeaviateQueryError: If the search fails.

    Example:
        >>> results = query_hybrid(
        ...     client,
        ...     "amygdala fear response",
        ...     alpha=0.7,  # Favor semantic similarity
        ...     book_ids=["Behave", "Biopsychology"]
        ... )
    """
    if collection_name is None:
        collection_name = get_collection_name()

    logger.info(f"[query_hybrid] Collection: {collection_name}")

    # Use precomputed embedding or embed the query
    if precomputed_embedding is not None:
        logger.info(f"Hybrid search (alpha={alpha}, precomputed embedding): {query_text[:50]}...")
        query_embedding = precomputed_embedding
    else:
        logger.info(f"Hybrid search (alpha={alpha}): {query_text[:50]}...")
        query_embedding = embed_texts([query_text])[0]

    collection = client.collections.get(collection_name)
    book_filter = _build_book_filter(book_ids)

    # Hybrid search combines:
    # - query: the text for BM25 keyword matching
    # - vector: the embedding for vector similarity
    # - alpha: weight between vector (1.0) and keyword (0.0)
    response = collection.query.hybrid(
        query=query_text,
        vector=query_embedding,
        alpha=alpha,
        limit=top_k,
        filters=book_filter,
        return_metadata=MetadataQuery(score=True),
    )

    return _parse_results(response.objects, use_distance=False)


def query_bm25(
    client: weaviate.WeaviateClient,
    query_text: str,
    top_k: int = 5,
    book_ids: Optional[Union[str, list[str]]] = None,
    collection_name: Optional[str] = None,
) -> list[SearchResult]:
    """
    Perform pure BM25 keyword search without vector similarity.

    BM25 (Best Matching 25) is a probabilistic ranking function that scores
    documents based on term frequency, document length, and inverse document
    frequency. Unlike hybrid search, this function does not use embeddings
    at all, making it faster but potentially less semantically aware.

    Use this when:
    - Testing keyword-only baselines
    - Queries contain specific technical terms or proper nouns
    - Semantic similarity might be misleading (e.g., "not X" queries)

    Args:
        client: Connected Weaviate client.
        query_text: The natural language query to search for.
        top_k: Number of results to return (default: 5).
        book_ids: Optional book ID(s) to filter results.
        collection_name: Target collection (default: from config).

    Returns:
        List of SearchResult objects, ordered by BM25 relevance.

    Raises:
        weaviate.exceptions.WeaviateQueryError: If the search fails.

    Example:
        >>> results = query_bm25(
        ...     client,
        ...     "amygdala fear response",
        ...     book_ids=["Behave", "Biopsychology"]
        ... )
    """
    if collection_name is None:
        collection_name = get_collection_name()

    logger.info(f"[query_bm25] Collection: {collection_name}")
    logger.info(f"BM25 search: {query_text[:50]}...")

    collection = client.collections.get(collection_name)
    book_filter = _build_book_filter(book_ids)

    # Pure BM25 search - no vector similarity
    response = collection.query.bm25(
        query=query_text,
        limit=top_k,
        filters=book_filter,
        return_metadata=MetadataQuery(score=True),
    )

    return _parse_results(response.objects, use_distance=False)


def list_available_books(
    client: weaviate.WeaviateClient,
    collection_name: Optional[str] = None,
) -> list[str]:
    """
    Get list of unique book_ids in the collection.

    Args:
        client: Connected Weaviate client.
        collection_name: Target collection (default: from config).

    Returns:
        Sorted list of unique book identifiers.
    """
    if collection_name is None:
        collection_name = get_collection_name()

    collection = client.collections.get(collection_name)

    # Use aggregation to get unique values
    result = collection.aggregate.over_all(
        group_by="book_id",
    )

    book_ids = [group.grouped_by.value for group in result.groups]
    return sorted(book_ids)

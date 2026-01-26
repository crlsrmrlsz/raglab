"""RetrievalStrategy protocol for unified preprocessing + retrieval.

This module defines the contract that all retrieval strategies must implement.
Each strategy encapsulates its own:
- Query preprocessing logic
- Retrieval execution logic
- Result merging/fusion logic

This eliminates the confusing multi_queries dual semantics where the same
data structure was used for both HyDE (embedding averaging) and Decomposition
(RRF merging).

## Design Rationale

Before: Conditional logic checking `if strategy == "hyde" and multi_queries`
After: Polymorphic dispatch via `strategy.execute(query, context)`

## Usage

    from src.rag_pipeline.retrieval.strategy_factory import get_strategy

    strategy = get_strategy("hyde")
    result = strategy.execute(query, context)
"""

from dataclasses import dataclass, field
from typing import Any, Optional, Protocol, TYPE_CHECKING, runtime_checkable

if TYPE_CHECKING:
    import weaviate
    from neo4j import Driver
    from src.rag_pipeline.indexing.weaviate_query import SearchResult
    from src.rag_pipeline.retrieval.query_preprocessing import PreprocessedQuery


@dataclass
class RetrievalContext:
    """Context parameters needed for retrieval execution.

    This bundles all the parameters that retrieval strategies need,
    avoiding long parameter lists and making the interface cleaner.

    Attributes:
        client: Connected Weaviate client (caller manages lifecycle).
        collection_name: Weaviate collection to search.
        top_k: Final number of results to return.
        use_reranking: Whether to apply cross-encoder reranking.
        initial_k: Number of candidates before reranking (only if use_reranking=True).
        alpha: Hybrid search balance (0=keyword, 1=semantic).
        search_type: "vector" or "hybrid".
        neo4j_driver: Optional Neo4j driver for GraphRAG (None for other strategies).
    """

    client: "weaviate.WeaviateClient"
    collection_name: str
    top_k: int
    use_reranking: bool
    initial_k: int
    alpha: float
    search_type: str = "hybrid"
    neo4j_driver: Optional["Driver"] = None


@dataclass
class RetrievalResult:
    """Result of strategy execution.

    Provides a unified structure for all strategy outputs, regardless
    of whether they used embedding averaging, RRF merging, or graph traversal.

    Attributes:
        results: List of SearchResult objects (final ranked list).
        preprocessing: PreprocessedQuery with preprocessing metadata.
        metadata: Strategy-specific metadata for UI logging (RRF scores, graph context, etc.).
    """

    results: list["SearchResult"] = field(default_factory=list)
    preprocessing: Optional["PreprocessedQuery"] = None
    metadata: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class RetrievalStrategy(Protocol):
    """Protocol for retrieval strategies.

    Each strategy implements the full preprocessing + retrieval pipeline:
    1. Preprocess query (generate hypotheticals, decompose, extract entities, etc.)
    2. Execute retrieval (single search, multi-query RRF, graph traversal, etc.)
    3. Return unified results with metadata

    This eliminates the need for conditional logic based on strategy type.

    Implementations:
        - StandardRetrieval: No preprocessing, direct Weaviate search
        - HyDERetrieval: Generate hypotheticals → average embeddings → single search
        - DecompositionRetrieval: Break into sub-questions → parallel searches → RRF merge
        - GraphRAGRetrieval: Extract entities → graph traversal → combined_degree ranking

    Example:
        >>> strategy = HyDERetrieval()
        >>> result = strategy.execute("What is consciousness?", context)
        >>> print(result.results[0].text)
    """

    strategy_id: str  # Must be set by implementations ("none", "hyde", etc.)

    def execute(
        self,
        query: str,
        context: RetrievalContext,
        preprocessing_model: Optional[str] = None,
    ) -> RetrievalResult:
        """Execute full preprocessing + retrieval pipeline.

        Args:
            query: User's original query text.
            context: Retrieval context (client, top_k, reranking settings, etc.).
            preprocessing_model: Override model for LLM preprocessing calls.

        Returns:
            RetrievalResult with results, preprocessing metadata, and strategy metadata.

        Raises:
            WeaviateConnectionError: If Weaviate connection fails.
            Neo4jConnectionError: If Neo4j connection fails (GraphRAG only).
        """
        ...

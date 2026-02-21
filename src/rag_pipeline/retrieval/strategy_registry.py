"""Unified retrieval strategy registry: protocol, functional strategies, and factory.

## RAG Theory: Strategy Pattern for Query Preprocessing

Each strategy applies its transformation directly to any query, following
the original research papers' design. The strategies don't use classification
routing - they simply transform queries for better retrieval.

## Contents

1. **Protocol + dataclasses**: RetrievalContext, RetrievalResult, RetrievalStrategy
2. **Functional strategies**: none_strategy, hyde_strategy, decomposition_strategy, graphrag_strategy
3. **Factory**: get_strategy (object-based), list_strategies, register_strategy

## Available Strategies

Preprocessing Strategies (query transformation):
- none: No transformation (baseline for comparison)
- hyde: Generate hypothetical answer for semantic matching (arXiv:2212.10496)
- decomposition: Break into sub-questions + union merge + rerank (+36.7% MRR@10, arXiv:2507.00355)
- graphrag: Pure graph retrieval with combined_degree ranking (arXiv:2404.16130)

Note: Keyword/BM25 search is a search_type dimension, not a preprocessing strategy.
See config.py SEARCH_TYPES for search_type options (keyword, hybrid).
"""

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Protocol, Type, TYPE_CHECKING, runtime_checkable

from src.config import PREPROCESSING_MODEL, HYDE_K
from src.rag_pipeline.retrieval.query_preprocessing import (
    PreprocessedQuery,
    hyde_prompt as _hyde_prompt_fn,
    decompose_query,
)
from src.shared.files import setup_logging

logger = setup_logging(__name__)

if TYPE_CHECKING:
    import weaviate
    from neo4j import Driver
    from src.rag_pipeline.indexing.weaviate_query import SearchResult


# =============================================================================
# PROTOCOL + DATACLASSES
# =============================================================================


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
    of whether they used embedding averaging, union merging, or graph traversal.

    Attributes:
        results: List of SearchResult objects (final ranked list).
        preprocessing: PreprocessedQuery with preprocessing metadata.
        metadata: Strategy-specific metadata for UI logging (rerank scores, graph context, etc.).
    """

    results: list["SearchResult"] = field(default_factory=list)
    preprocessing: Optional["PreprocessedQuery"] = None
    metadata: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class RetrievalStrategy(Protocol):
    """Protocol for retrieval strategies.

    Each strategy implements the full preprocessing + retrieval pipeline:
    1. Preprocess query (generate hypotheticals, decompose, extract entities, etc.)
    2. Execute retrieval (single search, multi-query union, graph traversal, etc.)
    3. Return unified results with metadata

    Implementations:
        - StandardRetrieval: No preprocessing, direct Weaviate search
        - HyDERetrieval: Generate hypotheticals -> average embeddings -> single search
        - DecompositionRetrieval: Break into sub-questions -> parallel searches -> union merge + rerank
        - GraphRAGRetrieval: Extract entities -> graph traversal -> combined_degree ranking
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
        """
        ...


# =============================================================================
# FUNCTIONAL STRATEGIES
# =============================================================================

# Type alias for strategy functions
StrategyFunction = Callable[[str, Optional[str]], PreprocessedQuery]


def none_strategy(query: str, model: Optional[str] = None) -> PreprocessedQuery:
    """No preprocessing - return original query unchanged.

    Use case: Baseline comparison for evaluation. Returns the query exactly
    as entered, with no LLM calls. Fastest strategy.

    Args:
        query: The user's original query.
        model: Ignored (no LLM call made).

    Returns:
        PreprocessedQuery with original query as search_query.
    """
    return PreprocessedQuery(
        original_query=query,
        search_query=query,
        strategy_used="none",
        preprocessing_time_ms=0.0,
        model=model or PREPROCESSING_MODEL,
    )


def hyde_strategy(query: str, model: Optional[str] = None) -> PreprocessedQuery:
    """HyDE: Generate K hypothetical answers, average embeddings for retrieval.

    Hypothetical Document Embeddings (HyDE) generates plausible answers
    to the query, then searches for real passages similar to these answers.
    Multiple hypotheticals improve retrieval robustness by covering
    diverse phrasings and perspectives.

    Paper Alignment (arXiv:2212.10496):
    - Embeddings averaged: original_query + K hypotheticals
    - Search mode: Pure semantic (alpha=1.0, no BM25 component)

    Args:
        query: The user's original query.
        model: Model for HyDE LLM call.

    Returns:
        PreprocessedQuery with original + K hypotheticals in generated_queries.
        search_query contains first passage for backward compatibility.
    """
    start_time = time.time()
    model = model or PREPROCESSING_MODEL

    # Generate K hypothetical answers (configurable via HYDE_K)
    # Returns list of {"domain": "neuroscience|philosophy", "passage": "..."}
    hyde_results = _hyde_prompt_fn(query, model=model, k=HYDE_K)
    first_passage = hyde_results[0]["passage"]
    logger.info(f"[hyde] Generated {len(hyde_results)} hypotheticals, first: {first_passage[:80]}...")

    elapsed_ms = (time.time() - start_time) * 1000

    # Paper alignment: include original query + hypotheticals for embedding averaging
    # The retrieval layer will average embeddings of ALL entries in generated_queries
    generated_queries = [{"type": "original", "query": query}]
    for result in hyde_results:
        generated_queries.append({
            "type": f"hyde_{result['domain']}",  # e.g., "hyde_neuroscience"
            "query": result["passage"],
            "domain": result["domain"],
        })

    return PreprocessedQuery(
        original_query=query,
        search_query=first_passage,  # First hypothetical for backward compatibility
        hyde_passage=first_passage,  # Keep first for logging
        generated_queries=generated_queries,  # Original + hypotheticals for averaging
        strategy_used="hyde",
        preprocessing_time_ms=elapsed_ms,
        model=model,
    )


def decomposition_strategy(query: str, model: Optional[str] = None) -> PreprocessedQuery:
    """Always decompose query into sub-questions for union merge + rerank.

    This strategy handles comparison and multi-aspect questions by:
    1. Decomposing into 2-4 sub-questions
    2. Using sub-questions for union-merged retrieval + reranking

    Based on Query Decomposition research showing +36.7% MRR@10 improvement
    for complex multi-hop queries. Works safely on simpler queries too.

    Args:
        query: The user's original query.
        model: Model for LLM calls.

    Returns:
        PreprocessedQuery with sub_queries and generated_queries for retrieval.
    """
    start_time = time.time()
    model = model or PREPROCESSING_MODEL

    # Decompose query into sub-questions
    sub_queries, decomposition_response = decompose_query(query, model=model)
    logger.info(f"[decomposition] Decomposed into {len(sub_queries)} sub-queries")

    # Build generated_queries format for search compatibility
    # This allows reuse of existing multi-query retrieval infrastructure
    generated_queries = [{"type": "original", "query": query}]
    for i, sq in enumerate(sub_queries):
        generated_queries.append({"type": f"sub_{i+1}", "query": sq})

    elapsed_ms = (time.time() - start_time) * 1000

    return PreprocessedQuery(
        original_query=query,
        search_query=query,  # Keep original for display
        sub_queries=sub_queries,
        strategy_used="decomposition",
        preprocessing_time_ms=elapsed_ms,
        model=model,
        generated_queries=generated_queries,  # For multi-query search compatibility
        decomposition_response=decomposition_response,
    )


def graphrag_strategy(query: str, model: Optional[str] = None) -> PreprocessedQuery:
    """GraphRAG: Pure graph retrieval with combined_degree ranking.

    Extracts entities from the query using embedding similarity against
    entity descriptions in Weaviate (~50ms). Entities are stored in
    PreprocessedQuery.query_entities for graph traversal at retrieval time.

    Research: arXiv:2404.16130 - GraphRAG: +72-83% win rate vs baseline

    Args:
        query: The user's original query.
        model: Model for preprocessing (optional).

    Returns:
        PreprocessedQuery with query_entities for retrieval layer.
    """
    start_time = time.time()
    model = model or PREPROCESSING_MODEL

    # Import here to avoid circular dependency
    from src.graph.query_entities import extract_query_entities

    # Extract entities using embedding similarity against Weaviate entity descriptions
    query_entities = extract_query_entities(query)
    logger.info(f"[graphrag] Extracted entities: {query_entities}")

    elapsed_ms = (time.time() - start_time) * 1000

    return PreprocessedQuery(
        original_query=query,
        search_query=query,  # Keep original for vector search
        strategy_used="graphrag",
        preprocessing_time_ms=elapsed_ms,
        model=model,
        query_entities=query_entities,  # For retrieval layer to use
    )


# =============================================================================
# FUNCTIONAL STRATEGY REGISTRY
# =============================================================================

# Maps strategy ID to strategy function
# Note: keyword is NOT a preprocessing strategy - it's a search_type
STRATEGIES: dict[str, StrategyFunction] = {
    "none": none_strategy,
    "hyde": hyde_strategy,
    "decomposition": decomposition_strategy,
    "graphrag": graphrag_strategy,
}


def get_functional_strategy(strategy_id: str) -> StrategyFunction:
    """Get strategy function by ID.

    Args:
        strategy_id: One of "none", "hyde", "decomposition", "graphrag".

    Returns:
        Strategy function that takes (query, model) and returns PreprocessedQuery.

    Raises:
        ValueError: If strategy_id is not registered.
    """
    if strategy_id not in STRATEGIES:
        available = list(STRATEGIES.keys())
        raise ValueError(f"Unknown strategy '{strategy_id}'. Available: {available}")
    return STRATEGIES[strategy_id]


# =============================================================================
# OBJECT-BASED STRATEGY FACTORY
# =============================================================================

# Lazy-loaded to avoid circular imports (strategy impls import from this module)
_STRATEGY_CLASSES: Optional[Dict[str, Type]] = None


def _get_strategy_classes() -> Dict[str, Type]:
    """Lazily load strategy class registry to avoid circular imports."""
    global _STRATEGY_CLASSES
    if _STRATEGY_CLASSES is None:
        from src.rag_pipeline.retrieval.strategies.standard import StandardRetrieval
        from src.rag_pipeline.retrieval.strategies.hyde import HyDERetrieval
        from src.rag_pipeline.retrieval.strategies.decomposition import DecompositionRetrieval
        from src.rag_pipeline.retrieval.strategies.graphrag import GraphRAGRetrieval

        _STRATEGY_CLASSES = {
            "none": StandardRetrieval,
            "hyde": HyDERetrieval,
            "decomposition": DecompositionRetrieval,
            "graphrag": GraphRAGRetrieval,
        }
    return _STRATEGY_CLASSES


def get_strategy(strategy_id: str) -> RetrievalStrategy:
    """Get strategy instance by ID.

    Args:
        strategy_id: One of "none", "hyde", "decomposition", "graphrag".

    Returns:
        RetrievalStrategy instance ready for execute() calls.

    Raises:
        ValueError: If strategy_id is not registered.
    """
    classes = _get_strategy_classes()
    if strategy_id not in classes:
        available = list(classes.keys())
        raise ValueError(f"Unknown strategy '{strategy_id}'. Available: {available}")

    strategy_class = classes[strategy_id]
    return strategy_class()


def list_strategies() -> list[str]:
    """List all registered strategy IDs.

    Returns:
        List of strategy IDs (e.g., ["none", "hyde", "decomposition", "graphrag"]).
    """
    return list(_get_strategy_classes().keys())


def register_strategy(strategy_id: str, strategy_class: Type) -> None:
    """Register a new strategy class (for extensions/plugins).

    Args:
        strategy_id: Unique identifier for the strategy.
        strategy_class: Class implementing RetrievalStrategy protocol.
    """
    classes = _get_strategy_classes()
    classes[strategy_id] = strategy_class

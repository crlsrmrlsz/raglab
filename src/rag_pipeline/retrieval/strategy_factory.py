"""Factory for creating retrieval strategy instances.

This replaces the functional STRATEGIES registry with an object-based registry.
Backward compatible with existing code that uses strategy_id strings.

Usage:
    from src.rag_pipeline.retrieval.strategy_factory import get_strategy

    # Get strategy by ID
    strategy = get_strategy("hyde")
    result = strategy.execute(query, context)

    # List available strategies
    strategies = list_strategies()  # ["none", "hyde", "decomposition", "graphrag"]
"""

from typing import Dict, Type

from src.rag_pipeline.retrieval.strategy_protocol import RetrievalStrategy
from src.rag_pipeline.retrieval.strategies.standard import StandardRetrieval
from src.rag_pipeline.retrieval.strategies.hyde import HyDERetrieval
from src.rag_pipeline.retrieval.strategies.decomposition import DecompositionRetrieval
from src.rag_pipeline.retrieval.strategies.graphrag import GraphRAGRetrieval


# Strategy registry: Maps strategy_id → strategy class
STRATEGY_CLASSES: Dict[str, Type] = {
    "none": StandardRetrieval,
    "hyde": HyDERetrieval,
    "decomposition": DecompositionRetrieval,
    "graphrag": GraphRAGRetrieval,
}


def get_strategy(strategy_id: str) -> RetrievalStrategy:
    """Get strategy instance by ID.

    Args:
        strategy_id: One of "none", "hyde", "decomposition", "graphrag".

    Returns:
        RetrievalStrategy instance ready for execute() calls.

    Raises:
        ValueError: If strategy_id is not registered.

    Example:
        >>> strategy = get_strategy("hyde")
        >>> result = strategy.execute("What is consciousness?", context)
        >>> print(result.results[0].text)
    """
    if strategy_id not in STRATEGY_CLASSES:
        available = list(STRATEGY_CLASSES.keys())
        raise ValueError(f"Unknown strategy '{strategy_id}'. Available: {available}")

    strategy_class = STRATEGY_CLASSES[strategy_id]
    return strategy_class()


def list_strategies() -> list[str]:
    """List all registered strategy IDs.

    Returns:
        List of strategy IDs (e.g., ["none", "hyde", "decomposition", "graphrag"]).
    """
    return list(STRATEGY_CLASSES.keys())


def register_strategy(strategy_id: str, strategy_class: Type) -> None:
    """Register a new strategy class (for extensions/plugins).

    Args:
        strategy_id: Unique identifier for the strategy.
        strategy_class: Class implementing RetrievalStrategy protocol.

    Example:
        >>> class CustomRetrieval:
        ...     strategy_id = "custom"
        ...     def execute(self, query, context, model=None):
        ...         ...
        >>> register_strategy("custom", CustomRetrieval)
    """
    STRATEGY_CLASSES[strategy_id] = strategy_class

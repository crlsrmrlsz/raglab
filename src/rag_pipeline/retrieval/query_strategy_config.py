"""Strategy configuration system for preprocessing strategies.

This module provides a declarative configuration system where each preprocessing
strategy declares its requirements as data, not code. This is the SINGLE SOURCE
OF TRUTH for strategy constraints, used by:

- UI (app.py): Renders constraint-aware controls (disables invalid options)
- Evaluation (run_stage_7_evaluation.py): Filters invalid combinations
- Retrieval (ragas_evaluator.py): Validates at runtime

## Design Principles

1. **Invalid states unrepresentable**: UI/evaluation can't create invalid combinations
2. **Single source of truth**: All constraints in StrategyConfig
3. **Self-documenting**: Config dataclass IS the documentation
4. **Extensible**: Add new strategies by creating StrategyConfig entry

## Constraint Dimensions

Each strategy declares constraints for:
- **Alpha**: Search balance (0.0=BM25, 0.5=hybrid, 1.0=semantic)
- **Reranking**: Whether cross-encoder reranking is used
- **Collection**: Which chunking strategies are compatible

## Example

    >>> config = get_strategy_config("hyde")
    >>> config.alpha_constraint.mode
    'fixed'
    >>> config.reranking_constraint.mode
    'forbidden'
    >>> config.is_valid_alpha(0.5)
    False
"""

from dataclasses import dataclass, field
from typing import Optional, Literal

from src.config import get_graphrag_chunk_collection_name


# =============================================================================
# EVALUATION CONSTANTS
# =============================================================================
# Fixed parameters for comprehensive evaluation (not grid dimensions)

EVAL_TOP_K = 15  # Fixed top_k for all evaluation runs


# =============================================================================
# COLLECTION TYPES
# =============================================================================
# All available chunking strategies that produce collections

ALL_COLLECTIONS = frozenset({
    "section",
    "semantic_std2",
    "semantic_std3",
    "contextual",
    "raptor",
})


# =============================================================================
# CONSTRAINT CLASSES
# =============================================================================


@dataclass
class CollectionConstraint:
    """Defines collection requirements for a strategy.

    Attributes:
        mode: How collection is constrained:
            - "any": Strategy works with any collection in allowed_collections
            - "dedicated": Strategy uses its own dedicated index (ignores selection)
        allowed_collections: Set of compatible collection types (when mode="any")
        dedicated_collection: Collection name when mode="dedicated"
    """

    mode: Literal["any", "dedicated"]
    allowed_collections: frozenset[str] = field(default_factory=lambda: ALL_COLLECTIONS)
    dedicated_collection: Optional[str] = None

    def __post_init__(self):
        """Validate constraint configuration."""
        if self.mode == "dedicated" and self.dedicated_collection is None:
            raise ValueError("dedicated mode requires dedicated_collection")

    def uses_dedicated_index(self) -> bool:
        """Check if strategy uses dedicated index (ignores collection selection)."""
        return self.mode == "dedicated"

    def is_compatible(self, collection_type: str) -> bool:
        """Check if strategy works with a collection type.

        Args:
            collection_type: Base collection type (e.g., "section", "semantic_std2").

        Returns:
            True if compatible.
        """
        if self.mode == "dedicated":
            return False  # Dedicated strategies ignore collection selection
        return collection_type in self.allowed_collections

    def get_allowed_collections(self) -> list[str]:
        """Get list of allowed collections for evaluation grid.

        Returns:
            List of collection type strings.
        """
        if self.mode == "dedicated":
            return []  # Dedicated strategies don't participate in grid
        return sorted(self.allowed_collections)


@dataclass
class AlphaConstraint:
    """Defines alpha (search balance) requirements for a strategy.

    Alpha controls the balance between keyword (BM25) and semantic (vector) search:
    - alpha=0.0: Pure keyword search
    - alpha=0.5: Balanced hybrid
    - alpha=1.0: Pure semantic search

    Attributes:
        mode: How alpha is constrained:
            - "fixed": Strategy requires exact alpha value
            - "range": Strategy allows alpha within [min_value, max_value]
            - "any": Strategy works with any alpha (0.0-1.0)
        fixed_value: Required alpha when mode="fixed"
        min_value: Minimum alpha when mode="range"
        max_value: Maximum alpha when mode="range"
    """

    mode: Literal["fixed", "range", "any"]
    fixed_value: Optional[float] = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None

    def __post_init__(self):
        """Validate constraint configuration."""
        if self.mode == "fixed" and self.fixed_value is None:
            raise ValueError("fixed mode requires fixed_value")
        if self.mode == "range" and (self.min_value is None or self.max_value is None):
            raise ValueError("range mode requires min_value and max_value")

    def is_valid(self, alpha: float) -> bool:
        """Check if alpha value is valid for this constraint.

        Args:
            alpha: Alpha value to validate (0.0-1.0).

        Returns:
            True if alpha satisfies this constraint.
        """
        if self.mode == "fixed":
            return abs(alpha - self.fixed_value) < 0.001
        elif self.mode == "range":
            return self.min_value <= alpha <= self.max_value
        return True  # mode="any"

    def get_default(self) -> float:
        """Return default alpha for this constraint."""
        if self.mode == "fixed":
            return self.fixed_value
        elif self.mode == "range":
            return (self.min_value + self.max_value) / 2
        return 0.5  # balanced default for "any"

    def get_allowed_values(self) -> list[float]:
        """Return list of allowed alpha values for evaluation grid.

        For comprehensive evaluation, we test specific alpha values.
        This returns the appropriate values based on constraint mode.
        """
        if self.mode == "fixed":
            return [self.fixed_value]
        elif self.mode == "range":
            # Test boundary and midpoint values within range
            values = []
            for v in [0.0, 0.5, 1.0]:
                if self.min_value <= v <= self.max_value:
                    values.append(v)
            return values if values else [self.get_default()]
        else:  # any
            return [0.0, 0.5, 1.0]


@dataclass
class RerankingConstraint:
    """Defines reranking requirements for a strategy.

    Reranking uses a cross-encoder model to re-score retrieved chunks
    for higher precision. Some strategies require it (decomposition),
    others forbid it (hyde, graphrag), and some allow testing both.

    Attributes:
        mode: How reranking is constrained:
            - "required": Must use reranking (always ON)
            - "forbidden": Cannot use reranking (always OFF)
            - "optional": Can test both ON and OFF
    """

    mode: Literal["required", "forbidden", "optional"]

    def is_valid(self, use_reranking: bool) -> bool:
        """Check if reranking setting is valid for this constraint.

        Args:
            use_reranking: Whether reranking is enabled.

        Returns:
            True if setting satisfies this constraint.
        """
        if self.mode == "required":
            return use_reranking is True
        elif self.mode == "forbidden":
            return use_reranking is False
        return True  # mode="optional"

    def get_default(self) -> bool:
        """Return default reranking setting for this constraint."""
        if self.mode == "required":
            return True
        elif self.mode == "forbidden":
            return False
        return False  # Default to OFF for optional

    def get_allowed_values(self) -> list[bool]:
        """Return list of allowed reranking values for evaluation grid."""
        if self.mode == "required":
            return [True]
        elif self.mode == "forbidden":
            return [False]
        return [False, True]  # Test both for optional


# =============================================================================
# STRATEGY CONFIG
# =============================================================================


@dataclass
class StrategyConfig:
    """Declarative configuration for a preprocessing strategy.

    This is the SINGLE SOURCE OF TRUTH for strategy constraints.
    All UI rendering, evaluation validation, and retrieval logic
    consult this configuration.

    Attributes:
        strategy_id: Unique identifier (e.g., "hyde", "none").
        display_name: Human-readable name for UI.
        description: Short description for UI help text.
        alpha_constraint: How alpha (search balance) is constrained.
        reranking_constraint: How reranking is constrained.
        collection_constraint: How collection selection is constrained.
        has_internal_retrieval: Whether strategy performs its own retrieval
            (e.g., GraphRAG does graph traversal + vector search internally).
        includes_original_in_embedding: Whether original query should be
            included when averaging embeddings (HyDE paper requirement).

    Example:
        >>> hyde_config = get_strategy_config("hyde")
        >>> hyde_config.is_valid_alpha(1.0)
        True
        >>> hyde_config.reranking_constraint.mode
        'optional'
        >>> graphrag_config = get_strategy_config("graphrag")
        >>> graphrag_config.has_internal_search()
        True
    """

    strategy_id: str
    display_name: str
    description: str
    alpha_constraint: AlphaConstraint = field(
        default_factory=lambda: AlphaConstraint(mode="any")
    )
    reranking_constraint: RerankingConstraint = field(
        default_factory=lambda: RerankingConstraint(mode="optional")
    )
    collection_constraint: CollectionConstraint = field(
        default_factory=lambda: CollectionConstraint(mode="any")
    )
    has_internal_retrieval: bool = False
    includes_original_in_embedding: bool = False

    # ==========================================================================
    # Convenience methods
    # ==========================================================================

    def uses_dedicated_index(self) -> bool:
        """Check if strategy uses dedicated index (ignores collection selection)."""
        return self.collection_constraint.uses_dedicated_index()

    def has_internal_search(self) -> bool:
        """Check if strategy performs its own retrieval."""
        return self.has_internal_retrieval

    def is_valid_alpha(self, alpha: float) -> bool:
        """Check if alpha is valid for this strategy."""
        return self.alpha_constraint.is_valid(alpha)

    def is_valid_reranking(self, use_reranking: bool) -> bool:
        """Check if reranking setting is valid for this strategy."""
        return self.reranking_constraint.is_valid(use_reranking)

    def is_compatible_with_collection(self, collection_type: str) -> bool:
        """Check if strategy works with a collection type."""
        return self.collection_constraint.is_compatible(collection_type)

    def get_default_alpha(self) -> float:
        """Get default alpha for this strategy."""
        return self.alpha_constraint.get_default()

    def get_default_reranking(self) -> bool:
        """Get default reranking for this strategy."""
        return self.reranking_constraint.get_default()

    def get_allowed_alphas(self) -> list[float]:
        """Get list of allowed alpha values for evaluation."""
        return self.alpha_constraint.get_allowed_values()

    def get_allowed_rerankings(self) -> list[bool]:
        """Get list of allowed reranking values for evaluation."""
        return self.reranking_constraint.get_allowed_values()

    def get_allowed_collections(self) -> list[str]:
        """Get list of allowed collections for evaluation."""
        return self.collection_constraint.get_allowed_collections()

    # ==========================================================================
    # Validation
    # ==========================================================================

    def validate(
        self, collection_type: str, alpha: float, use_reranking: bool
    ) -> tuple[bool, Optional[str]]:
        """Validate if this strategy works with given parameters.

        Args:
            collection_type: Base collection type (e.g., "section").
            alpha: Alpha value (0.0-1.0).
            use_reranking: Whether reranking is enabled.

        Returns:
            (is_valid, error_message) tuple.
        """
        # Check collection compatibility
        if not self.is_compatible_with_collection(collection_type):
            return (
                False,
                f"{self.strategy_id} not compatible with {collection_type}",
            )

        # Check alpha constraint
        if not self.is_valid_alpha(alpha):
            if self.alpha_constraint.mode == "fixed":
                return (
                    False,
                    f"{self.strategy_id} requires alpha={self.alpha_constraint.fixed_value}",
                )
            elif self.alpha_constraint.mode == "range":
                return (
                    False,
                    f"{self.strategy_id} requires alpha in "
                    f"[{self.alpha_constraint.min_value}, {self.alpha_constraint.max_value}]",
                )

        # Check reranking constraint
        if not self.is_valid_reranking(use_reranking):
            if self.reranking_constraint.mode == "required":
                return (False, f"{self.strategy_id} requires reranking=ON")
            elif self.reranking_constraint.mode == "forbidden":
                return (False, f"{self.strategy_id} requires reranking=OFF")

        return True, None


# =============================================================================
# STRATEGY REGISTRY
# =============================================================================

STRATEGY_CONFIGS: dict[str, StrategyConfig] = {
    "none": StrategyConfig(
        strategy_id="none",
        display_name="None",
        description="No preprocessing, use original query",
        # Alpha: any (test all values)
        alpha_constraint=AlphaConstraint(mode="any"),
        # Reranking: optional (test both ON and OFF)
        reranking_constraint=RerankingConstraint(mode="optional"),
        # Collection: any standard collection
        collection_constraint=CollectionConstraint(mode="any"),
    ),
    "hyde": StrategyConfig(
        strategy_id="hyde",
        display_name="HyDE",
        description="Hypothetical Document Embeddings (arXiv:2212.10496)",
        # Paper requirement: pure semantic search (dense retrieval)
        alpha_constraint=AlphaConstraint(mode="fixed", fixed_value=1.0),
        # Reranking optional: test if cross-encoder improves HyDE results
        reranking_constraint=RerankingConstraint(mode="optional"),
        # Collection: any standard collection
        collection_constraint=CollectionConstraint(mode="any"),
        # Paper requirement: average original query + hypotheticals
        includes_original_in_embedding=True,
    ),
    "decomposition": StrategyConfig(
        strategy_id="decomposition",
        display_name="Decomposition",
        description="Break into sub-questions + rerank (arXiv:2507.00355)",
        # Paper uses dense retrieval (bge-large-en-v1.5), no BM25
        alpha_constraint=AlphaConstraint(mode="fixed", fixed_value=1.0),
        # Paper requires cross-encoder reranking for sub-question results
        reranking_constraint=RerankingConstraint(mode="required"),
        # Collection: any standard collection
        collection_constraint=CollectionConstraint(mode="any"),
    ),
    "graphrag": StrategyConfig(
        strategy_id="graphrag",
        display_name="GraphRAG",
        description="Knowledge graph + community retrieval (arXiv:2404.16130)",
        # Alpha is fixed for internal search consistency
        alpha_constraint=AlphaConstraint(mode="fixed", fixed_value=1.0),
        # GraphRAG performs its own ranking via combined_degree; reranking not applicable
        reranking_constraint=RerankingConstraint(mode="forbidden"),
        # GraphRAG uses dedicated semantic_std2 collection (entity-chunk ID matching)
        collection_constraint=CollectionConstraint(
            mode="dedicated",
            dedicated_collection=get_graphrag_chunk_collection_name(),
        ),
        # GraphRAG performs its own hybrid retrieval (graph + vector)
        has_internal_retrieval=True,
    ),
}


# =============================================================================
# PUBLIC API
# =============================================================================


def get_strategy_config(strategy_id: str) -> StrategyConfig:
    """Get strategy configuration by ID.

    Args:
        strategy_id: Strategy identifier (e.g., "hyde").

    Returns:
        StrategyConfig for the strategy.

    Raises:
        ValueError: If strategy_id is unknown.
    """
    if strategy_id not in STRATEGY_CONFIGS:
        available = list(STRATEGY_CONFIGS.keys())
        raise ValueError(f"Unknown strategy '{strategy_id}'. Available: {available}")
    return STRATEGY_CONFIGS[strategy_id]


def list_strategy_configs() -> list[StrategyConfig]:
    """List all strategy configurations.

    Returns:
        List of StrategyConfig objects.
    """
    return list(STRATEGY_CONFIGS.values())


def list_strategy_ids() -> list[str]:
    """List all strategy IDs.

    Returns:
        List of strategy ID strings.
    """
    return list(STRATEGY_CONFIGS.keys())


def is_valid_combination(
    strategy_id: str,
    collection_type: str,
    alpha: float,
    use_reranking: bool,
) -> tuple[bool, Optional[str]]:
    """Check if (strategy, collection, alpha, reranking) combination is valid.

    Args:
        strategy_id: Strategy identifier.
        collection_type: Base collection type.
        alpha: Alpha value (0.0-1.0).
        use_reranking: Whether reranking is enabled.

    Returns:
        (is_valid, error_message) tuple.
    """
    try:
        config = get_strategy_config(strategy_id)
    except ValueError as e:
        return False, str(e)
    return config.validate(collection_type, alpha, use_reranking)


def get_valid_strategies_for_collection(collection_type: str) -> list[str]:
    """Get list of valid strategy IDs for a collection type.

    Args:
        collection_type: Base collection type (e.g., "section", "semantic_std2").

    Returns:
        List of valid strategy IDs.
    """
    valid = []
    for strategy_id, config in STRATEGY_CONFIGS.items():
        if config.is_compatible_with_collection(collection_type):
            valid.append(strategy_id)
    return valid


def get_all_collections() -> list[str]:
    """Get list of all available collection types.

    Returns:
        Sorted list of collection type strings.
    """
    return sorted(ALL_COLLECTIONS)

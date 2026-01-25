"""Strategy configuration system for preprocessing strategies.

This module provides a declarative configuration system where each preprocessing
strategy declares its requirements as data, not code. This is the SINGLE SOURCE
OF TRUTH for strategy constraints, used by:

- UI (app.py): Renders constraint-aware controls
- Evaluation (run_stage_7_evaluation.py): Filters invalid combinations
- Retrieval (ragas_evaluator.py): Validates at runtime

## Design Principles

1. **Invalid states unrepresentable**: UI can't create invalid combinations
2. **Single source of truth**: All constraints in StrategyConfig
3. **Self-documenting**: Config dataclass IS the documentation
4. **Extensible**: Add new strategies by creating StrategyConfig entry

## Alpha Semantics

Alpha controls the search balance (replaces search_type dimension):
- alpha=0.0: Pure keyword (BM25 only)
- 0 < alpha < 1: Hybrid (vector + BM25)
- alpha=1.0: Pure semantic (vector only, dense retrieval)

## Example

    >>> config = get_strategy_config("hyde")
    >>> config.alpha_constraint.mode
    'fixed'
    >>> config.alpha_constraint.fixed_value
    1.0
    >>> config.is_valid_alpha(0.5)
    False
"""

from dataclasses import dataclass, field
from typing import Optional, Literal


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
        compatible_collections: If set, restrict to these collection types.
            None means compatible with all collections.
        includes_original_in_embedding: Whether original query should be
            included when averaging embeddings (HyDE paper requirement).

    Example:
        >>> hyde_config = get_strategy_config("hyde")
        >>> hyde_config.is_valid_alpha(1.0)
        True
        >>> hyde_config.is_valid_alpha(0.5)
        False
    """

    strategy_id: str
    display_name: str
    description: str
    alpha_constraint: AlphaConstraint = field(
        default_factory=lambda: AlphaConstraint(mode="any")
    )
    compatible_collections: Optional[set[str]] = None
    includes_original_in_embedding: bool = False
    requires_reranking: bool = False  # If True, cross-encoder reranking is mandatory

    def is_valid_alpha(self, alpha: float) -> bool:
        """Check if alpha is valid for this strategy.

        Args:
            alpha: Alpha value (0.0-1.0).

        Returns:
            True if alpha is valid.
        """
        return self.alpha_constraint.is_valid(alpha)

    def get_default_alpha(self) -> float:
        """Get default alpha for this strategy."""
        return self.alpha_constraint.get_default()

    def get_allowed_alphas(self) -> list[float]:
        """Get list of allowed alpha values for evaluation."""
        return self.alpha_constraint.get_allowed_values()

    def is_compatible_with_collection(self, collection_type: str) -> bool:
        """Check if strategy works with a collection type.

        Args:
            collection_type: Base collection type (e.g., "section", "semantic").

        Returns:
            True if compatible.
        """
        if self.compatible_collections is None:
            return True
        return collection_type in self.compatible_collections

    def validate(
        self, collection_type: str, alpha: float
    ) -> tuple[bool, Optional[str]]:
        """Validate if this strategy works with given parameters.

        Args:
            collection_type: Base collection type (e.g., "section").
            alpha: Alpha value (0.0-1.0).

        Returns:
            (is_valid, error_message) tuple.
        """
        if not self.is_compatible_with_collection(collection_type):
            return (
                False,
                f"{self.strategy_id} requires collection in {self.compatible_collections}",
            )
        if not self.is_valid_alpha(alpha):
            if self.alpha_constraint.mode == "fixed":
                return (
                    False,
                    f"{self.strategy_id} requires alpha={self.alpha_constraint.fixed_value}",
                )
            elif self.alpha_constraint.mode == "range":
                return (
                    False,
                    f"{self.strategy_id} requires alpha in [{self.alpha_constraint.min_value}, {self.alpha_constraint.max_value}]",
                )
            else:
                # Should not reach here for mode="any" since is_valid() returns True
                return (False, f"{self.strategy_id} has invalid alpha configuration")
        return True, None


# =============================================================================
# STRATEGY REGISTRY
# =============================================================================

STRATEGY_CONFIGS: dict[str, StrategyConfig] = {
    "none": StrategyConfig(
        strategy_id="none",
        display_name="None",
        description="No preprocessing, use original query",
        alpha_constraint=AlphaConstraint(mode="any"),
    ),
    "hyde": StrategyConfig(
        strategy_id="hyde",
        display_name="HyDE",
        description="Hypothetical Document Embeddings (arXiv:2212.10496)",
        # Paper requirement: pure semantic search (dense retrieval)
        alpha_constraint=AlphaConstraint(mode="fixed", fixed_value=1.0),
        # Paper requirement: average original query + hypotheticals
        includes_original_in_embedding=True,
    ),
    "decomposition": StrategyConfig(
        strategy_id="decomposition",
        display_name="Decomposition",
        description="Break into sub-questions + rerank (arXiv:2507.00355)",
        # Paper uses dense retrieval (bge-large-en-v1.5), no BM25
        alpha_constraint=AlphaConstraint(mode="fixed", fixed_value=1.0),
        requires_reranking=True,  # Paper requires cross-encoder reranking
    ),
    "graphrag": StrategyConfig(
        strategy_id="graphrag",
        display_name="GraphRAG",
        description="Pure graph retrieval with combined_degree ranking (arXiv:2404.16130)",
        # GraphRAG requires matching chunk IDs (only section collections)
        compatible_collections={"section"},
        # GraphRAG constraints TBD - for now allow any alpha
        alpha_constraint=AlphaConstraint(mode="any"),
    ),
}


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


def get_valid_strategies_for_collection(collection_type: str) -> list[str]:
    """Get list of valid strategy IDs for a collection type.

    Args:
        collection_type: Base collection type (e.g., "section", "semantic").

    Returns:
        List of valid strategy IDs.
    """
    valid = []
    for strategy_id, config in STRATEGY_CONFIGS.items():
        if config.is_compatible_with_collection(collection_type):
            valid.append(strategy_id)
    return valid


def is_valid_combination(
    strategy_id: str, collection_type: str, alpha: float
) -> tuple[bool, Optional[str]]:
    """Check if (strategy, collection, alpha) combination is valid.

    Args:
        strategy_id: Strategy identifier.
        collection_type: Base collection type.
        alpha: Alpha value (0.0-1.0).

    Returns:
        (is_valid, error_message) tuple.
    """
    try:
        config = get_strategy_config(strategy_id)
    except ValueError as e:
        return False, str(e)
    return config.validate(collection_type, alpha)


def list_strategy_configs() -> list[StrategyConfig]:
    """List all strategy configurations.

    Returns:
        List of StrategyConfig objects.
    """
    return list(STRATEGY_CONFIGS.values())

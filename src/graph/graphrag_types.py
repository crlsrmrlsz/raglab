"""GraphRAG entity type definitions.

This module provides access to the curated entity types optimized for
the dual-domain corpus (neuroscience + philosophy/wisdom).

Design Decisions:
    - 20 entity types vs Microsoft's default 4 (due to dual-domain nature)
    - Types are predefined (constrained extraction) vs auto-discovered
    - Bridge types (COGNITIVE_PROCESS, EMOTION, BEHAVIOR) enable cross-domain queries
    - Relationship types are NOT predefined (following GraphRAG paper)

Usage:
    >>> from src.graph.graphrag_types import get_entity_types, get_entity_type_names
    >>> types = get_entity_types()  # Full type definitions
    >>> names = get_entity_type_names()  # Just names for prompts
    >>> print(names)
    ['BRAIN_REGION', 'NEURAL_STRUCTURE', ...]

See Also:
    - graphrag_types.yaml: Full type definitions with examples
    - auto_tuning.py: Uses these types for constrained extraction
    - query_entities.py: Uses these types for query-time entity matching
"""

from pathlib import Path
from typing import Any

import yaml

from src.shared.setup_logging import get_logger

logger = get_logger(__name__)

# Path to the YAML configuration
_TYPES_YAML_PATH = Path(__file__).parent / "graphrag_types.yaml"

# Cached data
_cached_config: dict[str, Any] | None = None


def _load_config() -> dict[str, Any]:
    """Load and cache the GraphRAG types configuration.

    Returns:
        Parsed YAML configuration as dictionary.

    Raises:
        FileNotFoundError: If graphrag_types.yaml doesn't exist.
        yaml.YAMLError: If YAML parsing fails.
    """
    global _cached_config

    if _cached_config is not None:
        return _cached_config

    if not _TYPES_YAML_PATH.exists():
        raise FileNotFoundError(
            f"GraphRAG types configuration not found: {_TYPES_YAML_PATH}"
        )

    with open(_TYPES_YAML_PATH, "r") as f:
        _cached_config = yaml.safe_load(f)

    logger.debug(
        "Loaded GraphRAG types: %d entity types, %d relationship types",
        len(_cached_config.get("entity_types", [])),
        len(_cached_config.get("relationship_types", [])),
    )

    return _cached_config


def get_entity_types() -> list[dict[str, Any]]:
    """Get full entity type definitions including examples and descriptions.

    Returns:
        List of entity type dictionaries with keys:
            - name: Type name (e.g., 'BRAIN_REGION')
            - domain: 'neuroscience', 'philosophy', or 'shared'
            - description: What this type captures
            - examples: List of example entities
            - query_patterns: Common query patterns using this type

    Example:
        >>> types = get_entity_types()
        >>> brain_region = next(t for t in types if t['name'] == 'BRAIN_REGION')
        >>> print(brain_region['examples'][:3])
        ['amygdala', 'prefrontal cortex', 'hippocampus']
    """
    config = _load_config()
    return config.get("entity_types", [])


def get_entity_type_names() -> list[str]:
    """Get just the entity type names for use in prompts.

    Returns:
        List of entity type names in UPPERCASE_SNAKE_CASE.

    Example:
        >>> names = get_entity_type_names()
        >>> print(', '.join(names[:5]))
        'BRAIN_REGION, NEURAL_STRUCTURE, NEUROTRANSMITTER, HORMONE, COGNITIVE_PROCESS'
    """
    types = get_entity_types()
    return [t["name"] for t in types]


def get_entity_types_by_domain(domain: str) -> list[dict[str, Any]]:
    """Get entity types filtered by domain.

    Args:
        domain: One of 'neuroscience', 'philosophy', or 'shared'.

    Returns:
        List of entity type dictionaries for the specified domain.

    Example:
        >>> neuro_types = get_entity_types_by_domain('neuroscience')
        >>> print([t['name'] for t in neuro_types])
        ['BRAIN_REGION', 'NEURAL_STRUCTURE', ...]
    """
    types = get_entity_types()
    return [t for t in types if t.get("domain") == domain]


def get_entity_type_prompt_string() -> str:
    """Get entity types formatted for inclusion in LLM prompts.

    Returns:
        Comma-separated string of entity type names.

    Example:
        >>> prompt_str = get_entity_type_prompt_string()
        >>> print(prompt_str)
        'BRAIN_REGION, NEURAL_STRUCTURE, NEUROTRANSMITTER, ...'
    """
    return ", ".join(get_entity_type_names())


def get_entity_type_with_examples_prompt() -> str:
    """Get entity types with examples formatted for LLM prompts.

    This provides more context than just names, helping the LLM
    understand what each type should capture.

    Returns:
        Multi-line string with type names and example entities.

    Example:
        >>> prompt = get_entity_type_with_examples_prompt()
        >>> print(prompt[:200])
        BRAIN_REGION (e.g., amygdala, prefrontal cortex, hippocampus)
        NEURAL_STRUCTURE (e.g., neuron, synapse, axon)
        ...
    """
    types = get_entity_types()
    lines = []
    for t in types:
        examples = t.get("examples", [])[:3]  # First 3 examples
        if examples:
            examples_str = ", ".join(examples)
            lines.append(f"{t['name']} (e.g., {examples_str})")
        else:
            lines.append(t["name"])
    return "\n".join(lines)


def get_metadata() -> dict[str, Any]:
    """Get configuration metadata (version, creation date, etc.).

    Returns:
        Metadata dictionary from the YAML configuration.
    """
    config = _load_config()
    return config.get("metadata", {})


# Convenience: Pre-load on import to validate configuration
def _validate_on_import():
    """Validate configuration exists and is parseable on module import."""
    try:
        config = _load_config()
        entity_count = len(config.get("entity_types", []))
        if entity_count == 0:
            logger.warning("GraphRAG types config has no entity types defined")
    except FileNotFoundError:
        logger.warning(
            "GraphRAG types config not found at %s - using fallback types",
            _TYPES_YAML_PATH,
        )
    except yaml.YAMLError as e:
        logger.error("Failed to parse GraphRAG types config: %s", e)


_validate_on_import()

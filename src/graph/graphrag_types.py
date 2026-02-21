"""GraphRAG entity type definitions.

Provides access to curated entity types for the dual-domain corpus
(neuroscience + philosophy/wisdom). Follows Microsoft GraphRAG structure.

Usage:
    >>> from src.graph.graphrag_types import get_entity_types
    >>> types = get_entity_types()
    ['BRAIN_REGION', 'NEURAL_STRUCTURE', ...]
"""

from pathlib import Path

import yaml

from src.shared.files import setup_logging

logger = setup_logging(__name__)

_TYPES_YAML_PATH = Path(__file__).parent / "graphrag_types.yaml"
_cached_types: list[str] | None = None


def get_entity_types() -> list[str]:
    """Get entity type names for use in extraction prompts.

    Returns:
        List of entity type names (e.g., ['BRAIN_REGION', 'NEUROTRANSMITTER', ...])
    """
    global _cached_types

    if _cached_types is not None:
        return _cached_types

    if not _TYPES_YAML_PATH.exists():
        logger.warning("GraphRAG types not found at %s", _TYPES_YAML_PATH)
        return []

    with open(_TYPES_YAML_PATH, "r") as f:
        config = yaml.safe_load(f)

    _cached_types = config.get("entity_types", [])
    logger.debug("Loaded %d entity types", len(_cached_types))
    return _cached_types


def get_entity_types_string() -> str:
    """Get entity types as comma-separated string for prompts.

    Returns:
        String like 'BRAIN_REGION, NEURAL_STRUCTURE, NEUROTRANSMITTER, ...'
    """
    return ", ".join(get_entity_types())

"""Pydantic schemas for GraphRAG entities, relationships, and communities.

## RAG Theory: Knowledge Graph Data Model

GraphRAG represents extracted knowledge as a property graph:
- Nodes (entities): Named concepts with types (PERSON, CONCEPT, etc.)
- Edges (relationships): Typed connections between entities
- Communities: Clusters of related entities detected by Leiden

Using Pydantic models provides:
- Structured LLM output parsing (JSON Schema enforcement)
- Type validation at runtime
- Easy serialization for Neo4j and logging

## Library Usage

Pydantic v2 features used:
- model_validate_json(): Direct JSON string parsing
- model_json_schema(): Generate JSON Schema for LLM prompts
- Field(description=...): Self-documenting schemas for LLM guidance
"""

from typing import Optional, Any
import unicodedata
import re

from pydantic import BaseModel, Field

# Stopwords to remove from leading/trailing positions during normalization
EDGE_STOPWORDS = frozenset({'the', 'a', 'an', 'of', 'in', 'on', 'for', 'to', 'and'})


class GraphEntity(BaseModel):
    """Single entity extracted from text.

    Represents a node in the knowledge graph. Entity names are normalized
    (lowercased, trimmed) for deduplication during Neo4j MERGE operations.

    Attributes:
        name: Entity name (e.g., "prefrontal cortex", "Marcus Aurelius").
        entity_type: One of GRAPHRAG_ENTITY_TYPES from config.
        description: Brief description of the entity in context (1-2 sentences).
        source_chunk_id: ID of the chunk this entity was extracted from.

    Example:
        >>> entity = GraphEntity(
        ...     name="dopamine",
        ...     entity_type="NEUROTRANSMITTER",
        ...     description="Neurotransmitter involved in reward and motivation",
        ...     source_chunk_id="behave::chunk_42"
        ... )
    """

    name: str = Field(
        ...,
        description="The entity name as it appears in the text",
        min_length=1,
    )
    entity_type: str = Field(
        ...,
        description="Entity type from the allowed types list",
    )
    description: str = Field(
        default="",
        description="Brief description of this entity in context (1-2 sentences)",
    )
    source_chunk_id: str = Field(
        default="",
        description="Chunk ID where this entity was extracted from",
    )

    def normalized_name(self) -> str:
        """Normalize entity name for deduplication.

        Applies: Unicode NFKC, lowercase, edge stopword removal,
        punctuation stripping, whitespace normalization.

        Returns:
            Normalized name string for use as merge key.

        Example:
            >>> GraphEntity(name="The Dopamine", entity_type="X").normalized_name()
            'dopamine'
        """
        name = self.name.strip()
        # Unicode normalization (café → cafe)
        name = unicodedata.normalize('NFKC', name)
        name = name.lower()

        # Remove leading/trailing stopwords
        words = name.split()
        while words and words[0] in EDGE_STOPWORDS:
            words.pop(0)
        while words and words[-1] in EDGE_STOPWORDS:
            words.pop()

        name = ' '.join(words)
        # Strip punctuation (keep only alphanumeric and spaces)
        name = re.sub(r'[^\w\s]', '', name)
        # Normalize whitespace
        return ' '.join(name.split())



class GraphRelationship(BaseModel):
    """Relationship between two entities.

    Represents an edge in the knowledge graph. Relationships are directional
    (source → target) and typed.

    Attributes:
        source_entity: Name of the source entity.
        target_entity: Name of the target entity.
        relationship_type: One of GRAPHRAG_RELATIONSHIP_TYPES from config.
        description: Brief description of the relationship (1 sentence).
        weight: Confidence or strength of the relationship (0.0-1.0).
        source_chunk_id: ID of the chunk this relationship was extracted from.

    Example:
        >>> rel = GraphRelationship(
        ...     source_entity="dopamine",
        ...     target_entity="reward",
        ...     relationship_type="MODULATES",
        ...     description="Dopamine modulates reward processing",
        ... )
    """

    source_entity: str = Field(
        ...,
        description="Name of the source entity (from)",
    )
    target_entity: str = Field(
        ...,
        description="Name of the target entity (to)",
    )
    relationship_type: str = Field(
        ...,
        description="Relationship type from the allowed types list",
    )
    description: str = Field(
        default="",
        description="Brief description of this relationship (1 sentence)",
    )
    weight: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Confidence or strength of relationship (0.0-1.0)",
    )
    source_chunk_id: str = Field(
        default="",
        description="Chunk ID where this relationship was extracted from",
    )



class ExtractionResult(BaseModel):
    """Result of entity/relationship extraction from a single chunk.

    This is the Pydantic model passed to call_structured_completion()
    for JSON Schema enforcement during LLM extraction.

    Attributes:
        entities: List of extracted entities.
        relationships: List of extracted relationships.

    Example:
        >>> result = ExtractionResult(
        ...     entities=[GraphEntity(name="X", entity_type="CONCEPT")],
        ...     relationships=[GraphRelationship(
        ...         source_entity="X",
        ...         target_entity="Y",
        ...         relationship_type="CAUSES"
        ...     )],
        ... )
    """

    entities: list[GraphEntity] = Field(
        default_factory=list,
        description="List of entities extracted from the text",
    )
    relationships: list[GraphRelationship] = Field(
        default_factory=list,
        description="List of relationships between entities",
    )


class CommunityRelationship(BaseModel):
    """Relationship within a community for structured storage.

    Stores relationship data directly in JSON for offline access
    without requiring Neo4j queries at runtime.

    Attributes:
        source: Source entity name.
        target: Target entity name.
        relationship_type: Type of relationship (e.g., "CAUSES", "MODULATES").
        description: Brief description of the relationship.
        weight: Relationship strength (0.0-1.0).
    """

    source: str = Field(..., description="Source entity name")
    target: str = Field(..., description="Target entity name")
    relationship_type: str = Field(..., description="Relationship type")
    description: str = Field(default="", description="Relationship description")
    weight: float = Field(default=1.0, ge=0.0, le=1.0, description="Relationship strength")


class CommunityMember(BaseModel):
    """Entity within a Leiden community.

    Stores entity information plus its community assignment and
    centrality score within the community.

    Attributes:
        entity_name: Normalized entity name.
        entity_type: Entity type label.
        description: Entity description from extraction.
        degree: Number of relationships this entity has.
        pagerank: PageRank centrality score (importance in community).
    """

    entity_name: str = Field(..., description="Entity name")
    entity_type: str = Field(..., description="Entity type label")
    description: str = Field(default="", description="Entity description")
    degree: int = Field(default=0, description="Number of relationships")
    pagerank: float = Field(default=0.0, description="PageRank centrality score")


class Community(BaseModel):
    """Leiden community with members and summary.

    Represents a cluster of related entities detected by the Leiden
    algorithm. Each community has an LLM-generated summary that
    captures the main theme and relationships.

    Hierarchy levels follow Microsoft GraphRAG convention:
    - Level 0 (C0): Coarsest granularity, corpus-wide themes
    - Level 1 (C1): Medium granularity, domain themes
    - Level 2 (C2): Finest granularity, specific topics

    Attributes:
        community_id: Unique identifier (e.g., "community_L0_42").
        level: Hierarchy level (0 = coarsest, higher = finer).
        parent_id: Parent community ID at coarser level (None if top).
        members: List of entities in this community.
        member_count: Number of entities in this community.
        relationships: Structured relationships within community.
        relationship_count: Number of relationships within community.
        summary: LLM-generated summary of the community theme.
        embedding: Vector embedding of the summary (for retrieval).

    Example:
        >>> community = Community(
        ...     community_id="community_L0_42",
        ...     level=0,
        ...     parent_id="community_L1_10",
        ...     members=[CommunityMember(entity_name="dopamine", ...)],
        ...     relationships=[CommunityRelationship(...)],
        ...     summary="This community focuses on neurotransmitters...",
        ... )
    """

    community_id: str = Field(..., description="Unique community identifier")
    level: int = Field(default=0, description="Hierarchy level (0 = coarsest, higher = finer)")
    parent_id: Optional[str] = Field(
        default=None,
        description="Parent community ID at coarser level",
    )
    members: list[CommunityMember] = Field(
        default_factory=list,
        description="Entities in this community",
    )
    member_count: int = Field(default=0, description="Number of members")
    relationships: list[CommunityRelationship] = Field(
        default_factory=list,
        description="Structured relationships within community",
    )
    relationship_count: int = Field(
        default=0,
        description="Relationships within community",
    )
    summary: str = Field(default="", description="LLM-generated community summary")
    embedding: Optional[list[float]] = Field(
        default=None,
        description="Vector embedding of summary",
    )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dict for JSON serialization."""
        return {
            "community_id": self.community_id,
            "level": self.level,
            "parent_id": self.parent_id,
            "member_count": self.member_count,
            "relationship_count": self.relationship_count,
            "summary": self.summary,
            "embedding": self.embedding,
            "members": [m.model_dump() for m in self.members],
            "relationships": [r.model_dump() for r in self.relationships],
        }

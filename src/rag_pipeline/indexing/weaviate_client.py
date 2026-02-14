"""Weaviate client wrapper for RAGLab.

Provides functions for:
- Connecting to local Weaviate instance
- Creating/deleting collections with appropriate schema
- Batch uploading embeddings with metadata

Uses Weaviate Python client v4 (requires gRPC).
"""

from typing import Any
import uuid

import weaviate
from weaviate.classes.config import Configure, Property, DataType, VectorDistances

from src.config import (
    WEAVIATE_HOST,
    WEAVIATE_HTTP_PORT,
    WEAVIATE_GRPC_PORT,
    WEAVIATE_BATCH_SIZE,
)
from src.shared.files import setup_logging

logger = setup_logging(__name__)


def get_client() -> weaviate.WeaviateClient:
    """
    Create and return a Weaviate client connected to local instance.

    Returns:
        Connected WeaviateClient instance.

    Raises:
        weaviate.exceptions.WeaviateConnectionError: If connection fails.
    """
    client = weaviate.connect_to_local(
        host=WEAVIATE_HOST,
        port=WEAVIATE_HTTP_PORT,
        grpc_port=WEAVIATE_GRPC_PORT,
    )
    return client


def create_collection(
    client: weaviate.WeaviateClient,
    collection_name: str,
) -> None:
    """
    Create a new collection with the RAG chunk schema.

    Args:
        client: Connected Weaviate client.
        collection_name: Name for the new collection.

    Raises:
        weaviate.exceptions.WeaviateBaseError: If collection creation fails.
    """
    client.collections.create(
        name=collection_name,
        vector_config=Configure.Vectors.self_provided(
            vector_index_config=Configure.VectorIndex.hnsw(
                distance_metric=VectorDistances.COSINE,
            ),
        ),
        properties=[
            Property(name="chunk_id", data_type=DataType.TEXT),
            Property(name="book_id", data_type=DataType.TEXT),
            Property(name="section", data_type=DataType.TEXT),
            Property(name="context", data_type=DataType.TEXT),
            Property(name="text", data_type=DataType.TEXT),
            Property(name="token_count", data_type=DataType.INT),
            Property(name="chunking_strategy", data_type=DataType.TEXT),
            Property(name="embedding_model", data_type=DataType.TEXT),
        ],
    )


def create_raptor_collection(
    client: weaviate.WeaviateClient,
    collection_name: str,
) -> None:
    """
    Create a collection with extended schema for RAPTOR trees.

    Extends the base RAG chunk schema with tree-specific properties:
    - tree_level: Depth in tree (0=leaf, 1+=summary)
    - is_summary: Quick filter for summary nodes
    - parent_ids: Parent chunk IDs (for tree traversal)
    - child_ids: Child chunk IDs (for tree traversal)
    - cluster_id: Cluster identifier at this level
    - source_chunk_ids: (Summaries) Original leaf chunks in subtree

    Args:
        client: Connected Weaviate client.
        collection_name: Name for the new collection.

    Raises:
        weaviate.exceptions.WeaviateBaseError: If collection creation fails.
    """
    client.collections.create(
        name=collection_name,
        vector_config=Configure.Vectors.self_provided(
            vector_index_config=Configure.VectorIndex.hnsw(
                distance_metric=VectorDistances.COSINE,
            ),
        ),
        properties=[
            # Base chunk properties (same as create_collection)
            Property(name="chunk_id", data_type=DataType.TEXT),
            Property(name="book_id", data_type=DataType.TEXT),
            Property(name="section", data_type=DataType.TEXT),
            Property(name="context", data_type=DataType.TEXT),
            Property(name="text", data_type=DataType.TEXT),
            Property(name="token_count", data_type=DataType.INT),
            Property(name="chunking_strategy", data_type=DataType.TEXT),
            Property(name="embedding_model", data_type=DataType.TEXT),
            # RAPTOR-specific tree properties
            Property(name="tree_level", data_type=DataType.INT),
            Property(name="is_summary", data_type=DataType.BOOL),
            Property(name="parent_ids", data_type=DataType.TEXT_ARRAY),
            Property(name="child_ids", data_type=DataType.TEXT_ARRAY),
            Property(name="cluster_id", data_type=DataType.TEXT),
            Property(name="source_chunk_ids", data_type=DataType.TEXT_ARRAY),
        ],
    )


def delete_collection(
    client: weaviate.WeaviateClient,
    collection_name: str,
) -> bool:
    """
    Delete a collection if it exists.

    Args:
        client: Connected Weaviate client.
        collection_name: Name of collection to delete.

    Returns:
        True if collection was deleted, False if it did not exist.
    """
    if client.collections.exists(collection_name):
        client.collections.delete(collection_name)
        return True
    return False


def _generate_uuid_from_chunk_id(chunk_id: str) -> str:
    """
    Generate deterministic UUID from chunk_id for idempotent uploads.

    Args:
        chunk_id: Unique chunk identifier.

    Returns:
        UUID string derived from chunk_id.
    """
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk_id))


def upload_embeddings(
    client: weaviate.WeaviateClient,
    collection_name: str,
    chunks: list[dict[str, Any]],
    batch_size: int = WEAVIATE_BATCH_SIZE,
    is_raptor: bool = False,
) -> int:
    """
    Upload embedded chunks to a Weaviate collection.

    Handles both regular chunks and RAPTOR tree nodes. For RAPTOR nodes,
    includes additional tree properties (tree_level, parent_ids, etc.).

    Args:
        client: Connected Weaviate client.
        collection_name: Target collection name.
        chunks: List of chunk dicts with 'embedding' and metadata fields.
        batch_size: Number of objects per batch (default from config).
        is_raptor: If True, include RAPTOR tree properties.

    Returns:
        Number of successfully uploaded objects.

    Raises:
        weaviate.exceptions.WeaviateBaseError: If batch upload fails.
    """
    collection = client.collections.get(collection_name)
    uploaded_count = 0

    with collection.batch.fixed_size(batch_size=batch_size) as batch:
        for chunk in chunks:
            # Base properties (all strategies)
            properties = {
                "chunk_id": chunk["chunk_id"],
                "book_id": chunk["book_id"],
                "section": chunk.get("section", ""),
                "context": chunk.get("context", ""),
                "text": chunk["text"],
                "token_count": chunk.get("token_count", 0),
                "chunking_strategy": chunk.get("chunking_strategy", ""),
                "embedding_model": chunk.get("embedding_model", ""),
            }

            # Add RAPTOR tree properties if applicable
            if is_raptor:
                properties.update({
                    "tree_level": chunk.get("tree_level", 0),
                    "is_summary": chunk.get("is_summary", False),
                    "parent_ids": chunk.get("parent_ids", []),
                    "child_ids": chunk.get("child_ids", []),
                    "cluster_id": chunk.get("cluster_id", ""),
                    "source_chunk_ids": chunk.get("source_chunk_ids", []),
                })

            batch.add_object(
                properties=properties,
                vector=chunk["embedding"],
                uuid=_generate_uuid_from_chunk_id(chunk["chunk_id"]),
            )
            uploaded_count += 1

    return uploaded_count


def get_collection_count(
    client: weaviate.WeaviateClient,
    collection_name: str,
) -> int:
    """
    Get the number of objects in a collection.

    Args:
        client: Connected Weaviate client.
        collection_name: Name of collection to count.

    Returns:
        Number of objects in the collection.
    """
    collection = client.collections.get(collection_name)
    result = collection.aggregate.over_all(total_count=True)
    return result.total_count


# ============================================================================
# Community Collection (for GraphRAG crash-proof design)
# ============================================================================


def create_community_collection(
    client: weaviate.WeaviateClient,
    collection_name: str,
) -> None:
    """Create a collection for GraphRAG community summaries.

    Stores community metadata and embeddings for vector search.
    Replaces the 383MB JSON file with efficient Weaviate storage.

    Args:
        client: Connected Weaviate client.
        collection_name: Name for the community collection.

    Raises:
        weaviate.exceptions.WeaviateBaseError: If collection creation fails.
    """
    client.collections.create(
        name=collection_name,
        vector_config=Configure.Vectors.self_provided(
            vector_index_config=Configure.VectorIndex.hnsw(
                distance_metric=VectorDistances.COSINE,
            ),
        ),
        properties=[
            Property(name="community_id", data_type=DataType.TEXT),
            Property(name="summary", data_type=DataType.TEXT),
            Property(name="member_count", data_type=DataType.INT),
            Property(name="relationship_count", data_type=DataType.INT),
            Property(name="level", data_type=DataType.INT),
            Property(name="members_json", data_type=DataType.TEXT),
            Property(name="relationships_json", data_type=DataType.TEXT),
        ],
    )


def _generate_uuid_from_community_id(community_id: str) -> str:
    """Generate deterministic UUID from community_id for idempotent uploads.

    Args:
        community_id: Unique community identifier (e.g., "community_42").

    Returns:
        UUID string derived from community_id.
    """
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, f"community:{community_id}"))


def upload_community(
    client: weaviate.WeaviateClient,
    collection_name: str,
    community_id: str,
    summary: str,
    embedding: list[float],
    member_count: int,
    relationship_count: int = 0,
    level: int = 0,
    members_json: str = "[]",
    relationships_json: str = "[]",
) -> None:
    """Upload a single community to Weaviate.

    Used for incremental saves during summarization (crash-proof).
    Uses deterministic UUID for idempotent uploads.

    Args:
        client: Connected Weaviate client.
        collection_name: Target community collection name.
        community_id: Unique community identifier.
        summary: LLM-generated community summary.
        embedding: Summary embedding vector.
        member_count: Number of entities in community.
        relationship_count: Number of relationships in community.
        level: Hierarchy level (0 = base level).
        members_json: JSON-serialized list of CommunityMember dicts.
        relationships_json: JSON-serialized list of CommunityRelationship dicts.
    """
    collection = client.collections.get(collection_name)

    collection.data.insert(
        properties={
            "community_id": community_id,
            "summary": summary,
            "member_count": member_count,
            "relationship_count": relationship_count,
            "level": level,
            "members_json": members_json,
            "relationships_json": relationships_json,
        },
        vector=embedding,
        uuid=_generate_uuid_from_community_id(community_id),
    )


def get_existing_community_ids(
    client: weaviate.WeaviateClient,
    collection_name: str,
) -> set:
    """Get set of community IDs already in Weaviate.

    Used for resume functionality - skip communities already summarized.

    Args:
        client: Connected Weaviate client.
        collection_name: Community collection name.

    Returns:
        Set of community_id strings already in collection.
    """
    if not client.collections.exists(collection_name):
        return set()

    collection = client.collections.get(collection_name)
    existing_ids = set()

    # Fetch all community_ids (no limit)
    for obj in collection.iterator(return_properties=["community_id"]):
        existing_ids.add(obj.properties["community_id"])

    return existing_ids


def query_communities_by_vector(
    client: weaviate.WeaviateClient,
    collection_name: str,
    query_embedding: list[float],
    top_k: int = 3,
) -> list[dict[str, Any]]:
    """Query communities by vector similarity.

    Replaces the in-memory cosine similarity loop with Weaviate HNSW search.
    O(log n) instead of O(n) for large community sets.

    Args:
        client: Connected Weaviate client.
        collection_name: Community collection name.
        query_embedding: Query embedding vector.
        top_k: Number of communities to return.

    Returns:
        List of dicts with community_id, summary, member_count, and score.
    """
    from weaviate.classes.query import MetadataQuery

    if not client.collections.exists(collection_name):
        return []

    collection = client.collections.get(collection_name)

    response = collection.query.near_vector(
        near_vector=query_embedding,
        limit=top_k,
        return_metadata=MetadataQuery(distance=True),
    )

    results = []
    for obj in response.objects:
        # Convert distance to similarity (cosine distance = 1 - similarity)
        similarity = 1.0 - obj.metadata.distance if obj.metadata.distance else 0.0

        results.append({
            "community_id": obj.properties["community_id"],
            "summary": obj.properties["summary"],
            "member_count": obj.properties["member_count"],
            "score": similarity,
        })

    return results


def fetch_all_communities_by_level(
    client: weaviate.WeaviateClient,
    collection_name: str,
    level: int,
) -> list[dict[str, Any]]:
    """Fetch all communities at a specific hierarchy level.

    Originally used for map-reduce global queries where ALL L0 communities
    were needed. Active global queries now use DRIFT search
    (src.graph.drift.drift_search) with HNSW top-K selection instead.
    This function is retained for the deprecated map-reduce path.
    No vector ranking — retrieves every community matching the level filter.

    Args:
        client: Connected Weaviate client.
        collection_name: Community collection name.
        level: Hierarchy level to filter (0 = coarsest for global queries).

    Returns:
        List of dicts with community_id, summary, member_count,
        members_json, and relationships_json.
    """
    from weaviate.classes.query import Filter

    if not client.collections.exists(collection_name):
        return []

    collection = client.collections.get(collection_name)

    response = collection.query.fetch_objects(
        filters=Filter.by_property("level").equal(level),
        limit=1000,
    )

    results = []
    for obj in response.objects:
        results.append({
            "community_id": obj.properties["community_id"],
            "summary": obj.properties["summary"],
            "member_count": obj.properties["member_count"],
            "relationship_count": obj.properties.get("relationship_count", 0),
            "level": obj.properties.get("level", 0),
            "members_json": obj.properties.get("members_json", "[]"),
            "relationships_json": obj.properties.get("relationships_json", "[]"),
        })

    return results


def fetch_communities_by_ids(
    client: weaviate.WeaviateClient,
    collection_name: str,
    community_ids: list[str],
) -> list[dict[str, Any]]:
    """Fetch communities by their community_id values.

    Used for local queries where community context is retrieved
    by entity membership (specific community IDs from Neo4j).

    Args:
        client: Connected Weaviate client.
        collection_name: Community collection name.
        community_ids: List of community_id strings to fetch.

    Returns:
        List of dicts with community_id, summary, member_count,
        members_json, and relationships_json.
    """
    from weaviate.classes.query import Filter

    if not community_ids or not client.collections.exists(collection_name):
        return []

    collection = client.collections.get(collection_name)

    response = collection.query.fetch_objects(
        filters=Filter.by_property("community_id").contains_any(community_ids),
        limit=len(community_ids),
    )

    results = []
    for obj in response.objects:
        results.append({
            "community_id": obj.properties["community_id"],
            "summary": obj.properties["summary"],
            "member_count": obj.properties["member_count"],
            "relationship_count": obj.properties.get("relationship_count", 0),
            "level": obj.properties.get("level", 0),
            "members_json": obj.properties.get("members_json", "[]"),
            "relationships_json": obj.properties.get("relationships_json", "[]"),
        })

    return results


# ============================================================================
# Entity Collection (for GraphRAG embedding-based query extraction)
# ============================================================================


def create_entity_collection(
    client: weaviate.WeaviateClient,
    collection_name: str,
) -> None:
    """Create a collection for GraphRAG entity description embeddings.

    Stores entity descriptions with embeddings for semantic similarity search
    at query time. This enables embedding-based entity extraction per the
    Microsoft GraphRAG reference implementation.

    Args:
        client: Connected Weaviate client.
        collection_name: Name for the entity collection.

    Raises:
        weaviate.exceptions.WeaviateBaseError: If collection creation fails.
    """
    client.collections.create(
        name=collection_name,
        vector_config=Configure.Vectors.self_provided(
            vector_index_config=Configure.VectorIndex.hnsw(
                distance_metric=VectorDistances.COSINE,
            ),
        ),
        properties=[
            Property(name="entity_name", data_type=DataType.TEXT),
            Property(name="normalized_name", data_type=DataType.TEXT),
            Property(name="entity_type", data_type=DataType.TEXT),
            Property(name="description", data_type=DataType.TEXT),
        ],
    )


def _generate_uuid_from_entity_name(entity_name: str) -> str:
    """Generate deterministic UUID from entity name for idempotent uploads.

    Args:
        entity_name: Entity name (normalized).

    Returns:
        UUID string derived from entity name.
    """
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, f"entity:{entity_name}"))


def upload_entity_descriptions(
    client: weaviate.WeaviateClient,
    collection_name: str,
    entities: list[dict[str, Any]],
    batch_size: int = 100,
) -> int:
    """Upload entity descriptions with embeddings to Weaviate.

    Args:
        client: Connected Weaviate client.
        collection_name: Target entity collection name.
        entities: List of entity dicts with fields:
            - entity_name: Original entity name
            - normalized_name: Normalized name for matching
            - entity_type: Entity type label
            - description: Entity description (embedded)
            - embedding: Pre-computed embedding vector
        batch_size: Number of objects per batch.

    Returns:
        Number of successfully uploaded entities.
    """
    collection = client.collections.get(collection_name)
    uploaded_count = 0

    with collection.batch.fixed_size(batch_size=batch_size) as batch:
        for entity in entities:
            if not entity.get("embedding") or not entity.get("description"):
                continue  # Skip entities without embeddings or descriptions

            batch.add_object(
                properties={
                    "entity_name": entity["entity_name"],
                    "normalized_name": entity["normalized_name"],
                    "entity_type": entity.get("entity_type", "UNKNOWN"),
                    "description": entity["description"],
                },
                vector=entity["embedding"],
                uuid=_generate_uuid_from_entity_name(entity["normalized_name"]),
            )
            uploaded_count += 1

    return uploaded_count


def query_entities_by_vector(
    client: weaviate.WeaviateClient,
    collection_name: str,
    query_embedding: list[float],
    top_k: int = 10,
    min_similarity: float = 0.0,
) -> list[dict[str, Any]]:
    """Query entities by embedding similarity.

    Searches entity descriptions for semantic matches to the query embedding.
    Used for embedding-based entity extraction per Microsoft GraphRAG reference.

    Args:
        client: Connected Weaviate client.
        collection_name: Entity collection name.
        query_embedding: Query embedding vector.
        top_k: Maximum number of entities to return.
        min_similarity: Minimum similarity threshold (0.0-1.0).

    Returns:
        List of dicts with entity_name, normalized_name, entity_type, description, similarity.
    """
    from weaviate.classes.query import MetadataQuery

    if not client.collections.exists(collection_name):
        return []

    collection = client.collections.get(collection_name)

    response = collection.query.near_vector(
        near_vector=query_embedding,
        limit=top_k,
        return_metadata=MetadataQuery(distance=True),
    )

    results = []
    for obj in response.objects:
        # Convert distance to similarity (cosine distance = 1 - similarity)
        similarity = 1.0 - obj.metadata.distance if obj.metadata.distance else 0.0

        if similarity >= min_similarity:
            results.append({
                "entity_name": obj.properties["entity_name"],
                "normalized_name": obj.properties["normalized_name"],
                "entity_type": obj.properties["entity_type"],
                "description": obj.properties["description"],
                "similarity": similarity,
            })

    return results


def get_entity_collection_count(
    client: weaviate.WeaviateClient,
    collection_name: str,
) -> int:
    """Get the number of entities in the collection.

    Args:
        client: Connected Weaviate client.
        collection_name: Entity collection name.

    Returns:
        Number of entities, or 0 if collection doesn't exist.
    """
    if not client.collections.exists(collection_name):
        return 0

    collection = client.collections.get(collection_name)
    result = collection.aggregate.over_all(total_count=True)
    return result.total_count

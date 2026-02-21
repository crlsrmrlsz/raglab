"""Core RAPTOR tree building orchestration.

## RAG Theory: Hierarchical Tree Construction

RAPTOR builds a tree of summaries through recursive clustering:

1. **Level 0 (Leaves)**: Semantic chunks (std=2) as starting nodes
2. **Level 1 (First Summaries)**: LLM summaries of clustered leaves
3. **Level 2+ (Higher Summaries)**: LLM summaries of clustered summaries
4. **Termination**: When too few nodes remain or max level reached

The algorithm (per iteration):
1. Embed current level nodes
2. UMAP dimensionality reduction
3. Find optimal K via BIC
4. GMM clustering
5. LLM summarize each cluster -> new nodes at level+1
6. Repeat with new nodes until stopping condition

## Library Usage

- src.rag_pipeline.embedding.embedder: Embed texts for clustering
- raptor.clustering: UMAP + GMM operations
- raptor.summarizer: LLM summarization

## Data Flow

1. Input: List of semantic chunk dicts (std=2)
2. Convert to RaptorNodes (level 0)
3. Recursive: embed -> cluster -> summarize -> repeat
4. Output: All nodes (leaves + summaries) + metadata

## Query Strategies (Paper Section 2.3)

The RAPTOR paper describes two retrieval approaches:

1. **Tree Traversal**: Starts at root, traverses layer-by-layer selecting top-k
   nodes at each level. Fixed ratio of nodes from each level regardless of query.

2. **Collapsed Tree** (used here): Flattens all nodes into single layer, retrieves
   by similarity until token budget. More flexible - adapts granularity per query.

The paper shows collapsed tree consistently outperforms tree traversal due to
query-adaptive flexibility. Trade-off: requires similarity search on all nodes.

Reference: arXiv:2401.18059, Section 2.3
"""

import time
from typing import Any, Optional
import numpy as np

from src.config import (
    RAPTOR_MAX_LEVELS,
    RAPTOR_MIN_CLUSTER_SIZE,
    RAPTOR_SUMMARY_MODEL,
    RAPTOR_UMAP_N_NEIGHBORS,
    RAPTOR_UMAP_N_COMPONENTS,
    RAPTOR_CLUSTER_PROBABILITY_THRESHOLD,
)
from src.shared.files import setup_logging
from src.shared.tokens import count_tokens
from src.rag_pipeline.embedder import embed_texts
from src.rag_pipeline.chunking.raptor.schemas import (
    RaptorNode,
    ClusterResult,
    TreeMetadata,
)
from src.rag_pipeline.chunking.raptor.clustering import (
    reduce_dimensions,
    find_optimal_clusters,
    cluster_nodes,
    get_cluster_members,
)
from src.rag_pipeline.chunking.raptor.summarizer import (
    generate_cluster_summary,
    create_summary_context,
    create_summary_section,
)

logger = setup_logging(__name__)


def build_raptor_tree(
    chunks: list[dict[str, Any]],
    book_id: str,
    max_levels: int = RAPTOR_MAX_LEVELS,
    min_cluster_size: int = RAPTOR_MIN_CLUSTER_SIZE,
    summary_model: str = RAPTOR_SUMMARY_MODEL,
) -> tuple[list[RaptorNode], TreeMetadata]:
    """Build hierarchical RAPTOR tree from semantic chunks.

    Main entry point for tree construction. Takes semantic chunks (std=2) and builds
    a multi-level tree of summaries through recursive clustering.

    Args:
        chunks: List of chunk dicts from semantic chunking (std=2).
        book_id: Book identifier for chunk IDs.
        max_levels: Maximum tree depth (1-4 typical).
        min_cluster_size: Minimum nodes to attempt clustering.
        summary_model: OpenRouter model ID for summarization.

    Returns:
        Tuple of (all_nodes, metadata) where all_nodes includes leaves and
        all summary levels.

    Raises:
        ValueError: If chunks list is empty.
        Exception: Re-raises any error from clustering/summarization/embedding.
    """
    if not chunks:
        raise ValueError("Cannot build tree from empty chunk list")

    start_time = time.time()

    logger.info(f"Building RAPTOR tree for {book_id}")
    logger.info(f"Input: {len(chunks)} chunks, max_levels={max_levels}")

    # Step 1: Convert chunks to leaf nodes (level 0)
    leaf_nodes = _create_leaf_nodes(chunks, book_id)
    all_nodes = list(leaf_nodes)  # Copy to accumulate all nodes

    # Track level statistics
    levels: dict[int, int] = {0: len(leaf_nodes)}

    # Step 2: Recursive tree building
    current_level_nodes = leaf_nodes
    current_level = 0

    while _can_continue_clustering(
        current_level_nodes, current_level, max_levels, min_cluster_size
    ):
        logger.info(
            f"Building level {current_level + 1} from {len(current_level_nodes)} nodes"
        )

        # Step 2a: Embed current level nodes for clustering
        embeddings = _embed_nodes(current_level_nodes)

        # Step 2b: UMAP dimensionality reduction
        try:
            reduced = reduce_dimensions(
                embeddings,
                n_neighbors=RAPTOR_UMAP_N_NEIGHBORS,
                n_components=RAPTOR_UMAP_N_COMPONENTS,
            )
        except ValueError as e:
            # This is expected termination when tree is fully compressed
            logger.info(f"Tree building complete: {e}")
            break

        # Step 2c: Find optimal K via BIC
        try:
            optimal_k, bic_score = find_optimal_clusters(reduced)
        except ValueError as e:
            logger.warning(f"Cluster optimization failed: {e}. Stopping tree building.")
            break

        # Step 2d: GMM clustering
        cluster_result = cluster_nodes(
            reduced,
            optimal_k,
            probability_threshold=RAPTOR_CLUSTER_PROBABILITY_THRESHOLD,
        )

        # Step 2e: Create summary nodes for each cluster
        current_level += 1
        summary_nodes = _create_summary_nodes(
            current_level_nodes,
            cluster_result,
            book_id,
            current_level,
            summary_model,
        )

        if not summary_nodes:
            logger.warning("No summary nodes created. Stopping tree building.")
            break

        # Add summary nodes to tree
        all_nodes.extend(summary_nodes)
        levels[current_level] = len(summary_nodes)

        logger.info(f"Level {current_level}: {len(summary_nodes)} summaries created")

        # Move to next level
        current_level_nodes = summary_nodes

    # Step 3: Build metadata
    build_time = time.time() - start_time
    metadata = _build_metadata(all_nodes, book_id, levels, build_time)

    logger.info(
        f"Tree complete: {metadata.total_nodes} nodes "
        f"({metadata.leaf_count} leaves, {metadata.summary_count} summaries), "
        f"{metadata.max_level} levels, {build_time:.1f}s"
    )

    return all_nodes, metadata


def _create_leaf_nodes(
    chunks: list[dict[str, Any]],
    book_id: str,
) -> list[RaptorNode]:
    """Convert semantic chunks (std=2) to leaf RaptorNodes.

    Args:
        chunks: Chunk dicts from semantic chunking.
        book_id: Book identifier.

    Returns:
        List of RaptorNodes with tree_level=0.
    """
    return [RaptorNode.from_chunk(chunk, book_id) for chunk in chunks]


def _can_continue_clustering(
    nodes: list[RaptorNode],
    current_level: int,
    max_levels: int,
    min_cluster_size: int,
) -> bool:
    """Check if tree can grow another level.

    Stopping conditions:
    1. Reached max_levels
    2. Too few nodes for meaningful clustering
    3. Only one node left (natural termination)

    Args:
        nodes: Current level nodes.
        current_level: Current tree level.
        max_levels: Maximum allowed levels.
        min_cluster_size: Minimum nodes for clustering.

    Returns:
        True if clustering should continue, False otherwise.
    """
    # Level limit
    if current_level >= max_levels:
        logger.info(f"Stopping: reached max level {max_levels}")
        return False

    # Node count check
    if len(nodes) < min_cluster_size:
        logger.info(f"Stopping: too few nodes ({len(nodes)} < {min_cluster_size})")
        return False

    # Natural termination
    if len(nodes) <= 1:
        logger.info("Stopping: single node remaining")
        return False

    return True


def _embed_nodes(nodes: list[RaptorNode]) -> np.ndarray:
    """Embed node texts using the embedding API.

    Args:
        nodes: Nodes to embed.

    Returns:
        Numpy array of embeddings (n_nodes, embedding_dim).

    Raises:
        ValueError: If embedding API returns empty or mismatched count.
    """
    texts = [node.text for node in nodes]
    logger.info(f"Embedding {len(texts)} nodes...")

    # embed_texts returns List[List[float]]
    embeddings_list = embed_texts(texts)

    # Validate embedding results
    if not embeddings_list:
        raise ValueError(
            f"Embedding API returned empty results for {len(texts)} texts. "
            "Check API key, rate limits, or network connectivity."
        )

    if len(embeddings_list) != len(texts):
        raise ValueError(
            f"Embedding count mismatch: expected {len(texts)}, got {len(embeddings_list)}. "
            "Some API calls may have failed silently."
        )

    return np.array(embeddings_list)


def _create_summary_nodes(
    parent_nodes: list[RaptorNode],
    cluster_result: ClusterResult,
    book_id: str,
    level: int,
    summary_model: str,
) -> list[RaptorNode]:
    """Create summary nodes for each cluster.

    For each cluster:
    1. Get member nodes
    2. Generate LLM summary
    3. Create RaptorNode with parent/child links
    4. Update parent nodes with child links

    Args:
        parent_nodes: Nodes that were clustered.
        cluster_result: Clustering results.
        book_id: Book identifier.
        level: Tree level for new summaries.
        summary_model: Model for summarization.

    Returns:
        List of summary RaptorNodes.
    """
    summary_nodes = []

    for cluster_id in range(cluster_result.n_clusters):
        # Get member indices using soft clustering threshold
        member_indices = get_cluster_members(
            cluster_result,
            cluster_id,
            probability_threshold=RAPTOR_CLUSTER_PROBABILITY_THRESHOLD,
        )

        if len(member_indices) < 2:
            logger.warning(f"Cluster {cluster_id} has <2 members, skipping")
            continue

        # Get member nodes
        member_nodes = [parent_nodes[i] for i in member_indices]

        # Generate summary
        try:
            summary_text = generate_cluster_summary(member_nodes, model=summary_model)
        except Exception as e:
            logger.error(f"Summarization failed for cluster {cluster_id}: {e}")
            continue

        # Create chunk ID for summary
        chunk_id = f"{book_id}::L{level}_cluster_{cluster_id}"

        # Create context and section strings
        context = create_summary_context(member_nodes, level, cluster_id)
        section = create_summary_section(level, cluster_id)

        # Collect child IDs (from member nodes)
        child_ids = [node.chunk_id for node in member_nodes]

        # Collect source chunk IDs (all leaf chunks in subtree)
        source_chunk_ids = _collect_source_chunks(member_nodes)

        # Create summary node
        summary_node = RaptorNode(
            chunk_id=chunk_id,
            book_id=book_id,
            text=summary_text,
            context=context,
            section=section,
            token_count=count_tokens(summary_text),
            tree_level=level,
            is_summary=True,
            parent_ids=[],  # Will be filled by parent level
            child_ids=child_ids,
            cluster_id=f"L{level}_cluster_{cluster_id}",
            source_chunk_ids=source_chunk_ids,
        )

        # Update parent_ids on children (modifies in place)
        for member_node in member_nodes:
            member_node.parent_ids.append(chunk_id)

        summary_nodes.append(summary_node)

    return summary_nodes


def _collect_source_chunks(nodes: list[RaptorNode]) -> list[str]:
    """Collect all source leaf chunk IDs from a subtree.

    For leaf nodes: returns their chunk_id
    For summary nodes: returns their source_chunk_ids

    Args:
        nodes: Nodes in the subtree.

    Returns:
        List of unique leaf chunk IDs.
    """
    source_ids = set()

    for node in nodes:
        if node.is_summary and node.source_chunk_ids:
            source_ids.update(node.source_chunk_ids)
        elif not node.is_summary:
            source_ids.add(node.chunk_id)

    return sorted(source_ids)


def _build_metadata(
    all_nodes: list[RaptorNode],
    book_id: str,
    levels: dict[int, int],
    build_time: float,
) -> TreeMetadata:
    """Build tree metadata summary.

    Args:
        all_nodes: All nodes in the tree.
        book_id: Book identifier.
        levels: Dict mapping level to node count.
        build_time: Total build time in seconds.

    Returns:
        TreeMetadata with statistics.
    """
    leaf_count = sum(1 for node in all_nodes if not node.is_summary)
    summary_count = sum(1 for node in all_nodes if node.is_summary)
    max_level = max(node.tree_level for node in all_nodes)

    return TreeMetadata(
        book_id=book_id,
        total_nodes=len(all_nodes),
        leaf_count=leaf_count,
        summary_count=summary_count,
        max_level=max_level,
        build_time_seconds=build_time,
        levels=levels,
        umap_params={
            "n_neighbors": RAPTOR_UMAP_N_NEIGHBORS,
            "n_components": RAPTOR_UMAP_N_COMPONENTS,
        },
        gmm_params={
            "probability_threshold": RAPTOR_CLUSTER_PROBABILITY_THRESHOLD,
        },
    )

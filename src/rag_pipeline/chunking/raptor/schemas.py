"""Dataclass schemas for RAPTOR tree structure.

## RAG Theory: RAPTOR Data Model

RAPTOR builds a hierarchical tree where:
- Level 0: Leaf nodes (original document chunks)
- Level 1+: Summary nodes (LLM-generated summaries of clusters)

Each node contains:
- Content: text, context, section
- Tree structure: parent_ids, child_ids, tree_level
- Clustering metadata: cluster_id, is_summary

The collapsed tree approach stores all nodes in a flat list but preserves
tree relationships via parent_ids/child_ids for optional traversal.

## Library Usage

Uses Python dataclasses for:
- Immutable, typed data structures
- Self-documenting field names
- Easy serialization to dict/JSON
"""

from dataclasses import dataclass, field
from typing import Optional, Any


@dataclass
class RaptorNode:
    """Single node in RAPTOR tree (leaf or summary).

    A RaptorNode represents either:
    - A leaf node (original chunk from section chunker, tree_level=0)
    - A summary node (LLM-generated cluster summary, tree_level>0)

    Attributes:
        chunk_id: Unique identifier.
            - Leaves: "{book_id}::chunk_{N}"
            - Summaries: "{book_id}::L{level}_cluster_{N}"
        book_id: Source book identifier.
        text: Node content (original chunk or LLM summary).
        context: Hierarchical context string.
            - Leaves: "Book > Chapter > Section"
            - Summaries: "Book > Level N Summary"
        section: Section name for display.
        token_count: Token count of text field.
        tree_level: Depth in tree (0=leaf, 1+=summary levels).
        is_summary: True if this is a summary node, False for leaves.
        parent_ids: List of parent chunk IDs (empty for root).
        child_ids: List of child chunk IDs (empty for leaves).
        cluster_id: Cluster identifier at this level.
        source_chunk_ids: (Summaries only) Original leaf chunks in subtree.
    """

    chunk_id: str
    book_id: str
    text: str
    context: str
    section: str
    token_count: int
    tree_level: int
    is_summary: bool
    parent_ids: list[str] = field(default_factory=list)
    child_ids: list[str] = field(default_factory=list)
    cluster_id: str = ""
    source_chunk_ids: Optional[list[str]] = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dict for JSON serialization.

        Returns:
            Dict with all fields, including chunking_strategy="raptor"
            and empty embedding fields (to be filled by Stage 5).
        """
        return {
            "chunk_id": self.chunk_id,
            "book_id": self.book_id,
            "text": self.text,
            "context": self.context,
            "section": self.section,
            "token_count": self.token_count,
            "chunking_strategy": "raptor",
            "tree_level": self.tree_level,
            "is_summary": self.is_summary,
            "parent_ids": self.parent_ids,
            "child_ids": self.child_ids,
            "cluster_id": self.cluster_id,
            "source_chunk_ids": self.source_chunk_ids or [],
        }

    @classmethod
    def from_chunk(cls, chunk: dict[str, Any], book_id: str) -> "RaptorNode":
        """Create a leaf node from a section chunk dict.

        Args:
            chunk: Chunk dict from section_chunker.py.
            book_id: Book identifier.

        Returns:
            RaptorNode with tree_level=0, is_summary=False.
        """
        return cls(
            chunk_id=chunk["chunk_id"],
            book_id=book_id,
            text=chunk["text"],
            context=chunk.get("context", ""),
            section=chunk.get("section", ""),
            token_count=chunk.get("token_count", 0),
            tree_level=0,
            is_summary=False,
            parent_ids=[],
            child_ids=[],
            cluster_id="",
            source_chunk_ids=None,
        )


@dataclass
class ClusterResult:
    """Result of GMM clustering operation.

    Contains both hard assignments (for tree building) and soft probabilities
    (for potential future multi-cluster membership).

    Attributes:
        n_clusters: Number of clusters found (optimal K from BIC).
        cluster_assignments: Hard cluster assignment per node (list of ints).
        cluster_probabilities: Soft probabilities (n_nodes x n_clusters matrix).
        bic_score: Bayesian Information Criterion score (lower is better).
        nodes_per_cluster: Count of nodes in each cluster (for logging).
    """

    n_clusters: int
    cluster_assignments: list[int]
    cluster_probabilities: list[list[float]]
    bic_score: float
    nodes_per_cluster: list[int] = field(default_factory=list)


@dataclass
class TreeMetadata:
    """Summary statistics for a RAPTOR tree.

    Stored alongside chunks in the output JSON for debugging and analysis.

    Attributes:
        book_id: Source book identifier.
        total_nodes: Total node count (leaves + summaries).
        leaf_count: Number of leaf nodes (tree_level=0).
        summary_count: Number of summary nodes (tree_level>0).
        max_level: Maximum tree depth reached.
        build_time_seconds: Time to build tree (including LLM calls).
        levels: Dict mapping level to node count at that level.
        umap_params: UMAP parameters used (n_neighbors, n_components).
        gmm_params: GMM parameters used (probability_threshold).
    """

    book_id: str
    total_nodes: int
    leaf_count: int
    summary_count: int
    max_level: int
    build_time_seconds: float
    levels: dict[int, int] = field(default_factory=dict)
    umap_params: dict[str, Any] = field(default_factory=dict)
    gmm_params: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dict for JSON serialization."""
        return {
            "book_id": self.book_id,
            "total_nodes": self.total_nodes,
            "leaf_count": self.leaf_count,
            "summary_count": self.summary_count,
            "max_level": self.max_level,
            "build_time_seconds": round(self.build_time_seconds, 2),
            "levels": self.levels,
            "umap_params": self.umap_params,
            "gmm_params": self.gmm_params,
        }
